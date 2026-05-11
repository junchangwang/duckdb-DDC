// =============================================================================
// TPC-H Q15 — Top Supplier Query (spec v3.0.1 §2.4.15)
//
//   create view revenue0 (supplier_no, total_revenue) as
//     select l_suppkey,
//            sum(l_extendedprice * (1 - l_discount))
//       from lineitem
//      where l_shipdate >= date '[DATE]'
//        and l_shipdate <  date '[DATE]' + interval '3' month
//      group by l_suppkey;
//
//   select s_suppkey, s_name, s_address, s_phone, total_revenue
//     from supplier, revenue0
//    where s_suppkey = supplier_no
//      and total_revenue = (select max(total_revenue) from revenue0)
//    order by s_suppkey;
//
//   Qualification DATE = '1996-01-01' (spec §2.4.15.4).
//
// Bitmap pipeline (all bitmaps aligned to lineitem.rowid):
//
//   T0: ship_filter = OR_many(shipdate days [1461..1552))   (91 day bitmaps)
//   T1: walk ship_filter rows → revenue_by_suppkey[col_suppkey[r]]
//                                  += col_price[r] * (100 - col_disc[r])
//       (price stored ×100, disc stored ×100 → product ×10000;
//        the same shared factor is divided out at print time.)
//   T2: scan revenue_by_suppkey → max_revenue
//                                + collect tied suppkeys (sorted asc)
//
// The shipdate bitmaps are reused (via symlink) from Q6's exports:
//   tpch_q15_<fmt>/shipdate/{day}.bm  -> tpch_q6_<fmt>/shipdate/{day}.bm
// l_suppkey, l_extendedprice and l_discount are read directly from
// DuckDB storage at query time (mirrors Q1's pre-load), so no extra
// bitmap or column .bin file needs to be exported for Q15.
//
// Output (one row per supplier whose total_revenue equals the max):
//   s_suppkey, s_name, s_address, s_phone, total_revenue
// =============================================================================

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "execution/tpch/bitmap_table_scan.hpp"
#include "duckdb/common/enums/operator_result_type.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"

#include "combit_adapter.h"
#include "combit/include/combit.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"

#include "Concise/concise.h"
#include "execution/tpch/bm_bench_common.hpp"
#include "execution/tpch/indexed_bitmap.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstring>

namespace duckdb {

// --- Backend selection (DEBIT_BM=all|wah|cb|cb_bpe|cr|cr_bpe|crr|ew|con) ---
// Q15's predicate is a single contiguous shipdate range OR — BPE adds no
// range-skip benefit here (the range is exactly 91 days, smaller than any
// reasonable BPE bucket).  CB_BPE / CR_BPE therefore route to the same
// per-day OR path as CB / CR; the bench's PATTERN_BACKEND_CSV_PREFIX maps
// those columns to the BPE column in the merged Excel.
using Q15BmType = bm_bench::Backend;
static const Q15BmType Q15_BM = bm_bench::parse_backend("Q15_BM");

static bool run_all() { return Q15_BM == Q15BmType::ALL; }
static bool run_wah() { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::WAH; }
static bool run_cb()  { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CB
                            || Q15_BM == Q15BmType::CB_BPE; }
static bool run_cr()  { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CR
                            || Q15_BM == Q15BmType::CR_BPE; }
static bool run_crr() { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CRR; }
static bool run_ew()  { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::EW;  }
static bool run_con() { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CON; }

static const char* q15_bm_label()     { return bm_bench::backend_label(Q15_BM); }
static std::string q15_get_sf_label() { return bm_bench::sf_label(); }

// --- Q15 predicate (TPC-H spec §2.4.15, qualification DATE = 1996-01-01) ---
// Day range [1996-01-01, 1996-04-01) = epoch-day [9496, 9587).
// IndexedX is built via PRAGMA load_bitmap('shipdate'), which keys by
// DuckDB's int32 epoch-day (days since 1970-01-01), so we use the same
// convention as the load layer (debit_extension.cpp: 1996 boundary = 9496).
//   91 day bitmaps, inclusive day range [9496, 9586].
static const int Q15_DATE_START = 9496;  // 1996-01-01 inclusive
static const int Q15_DATE_END   = 9586;  // 1996-03-31 inclusive (1996-04-01 = 9587)

// --- Iteration counts (DEBIT_ITER / DEBIT_WARMUP) ---
static const int Q15_ITERATIONS = bm_bench::iter_count(10);
static const int Q15_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q15_once_flag;

// Byte-LUT for MSB-first bit extraction (same as Q1/Q14).
// Q15 fused per-row contribution (units: hundredths-of-a-cent — both
// `pp` and `dp` are stored ×100; the printed dollars value divides by
// 1e4 at the end).  Cross-backend consistency is checked on the int64
// accumulator vector so floating-point reformulation never matters.
#define Q15_REV_CONTRIB(pp, dp, r) \
    ((pp)[(r)] * (100 - (dp)[(r)]))

// Q15 result row (returned by the SELECT).  The bitmap pipeline only
// produces (suppkey, fp_revenue) pairs; supplier metadata is filled in
// once at the end via a small SQL lookup over the tied keys.
struct Q15Row {
    int32_t s_suppkey;
    int64_t total_rev_fp;       // accumulator: price ×100 × (100 - disc_pct)
    std::string s_name, s_address, s_phone;
};

// Convert int64 fixed-point accumulator to dollars (2 decimals).
static inline double q15_fp_to_dollars(int64_t fp) {
    return static_cast<double>(fp) / 10000.0;
}

// Find the max value in revenue_by_suppkey and the sorted tied suppkeys.
// Returns {max_fp, ties_sorted_asc}.
static std::pair<int64_t, std::vector<int32_t>>
q15_find_max(const std::vector<int64_t>& rev_by_sk) {
    int64_t mx = 0;
    for (auto v : rev_by_sk) if (v > mx) mx = v;
    std::vector<int32_t> ties;
    if (mx == 0) return {0, ties};
    for (size_t sk = 0; sk < rev_by_sk.size(); sk++)
        if (rev_by_sk[sk] == mx) ties.push_back(static_cast<int32_t>(sk));
    // already sorted ascending (we walked in suppkey order)
    return {mx, std::move(ties)};
}

// =============================================================================
// BMTPCH_Q15 — main benchmark entry point
// =============================================================================
void BMTableScan::BMTPCH_Q15(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q15_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    bm_bench::warn_if_sf1();

    // ============================================================
    // 0. Banner
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q15 Benchmark — 8 backends ("
                  << q15_get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q15 Benchmark — " << q15_bm_label() << " only ("
                  << q15_get_sf_label() << ")" << std::endl;
    }
    std::cout << "  OR shipdate days " << Q15_DATE_START << ".." << Q15_DATE_END
              << " (" << (Q15_DATE_END - Q15_DATE_START + 1) << " bitmaps), "
              << "then per-row revenue aggregation grouped by l_suppkey"
              << std::endl;
    std::cout << "  TPC-H params: l_shipdate [1996-01-01, 1996-04-01)" << std::endl;
    std::cout << "  Iterations: " << Q15_ITERATIONS
              << " (first " << Q15_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 1. Pre-load lineitem.l_suppkey / l_extendedprice / l_discount
    //    (one-shot scan, outside the timed loop, mirrors Q1's design)
    // ============================================================
    auto &lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");
    auto &lineitem_transaction = DuckTransaction::Get(context.client, lineitem_table.catalog);

    TableScanState scan_state;
    vector<StorageIndex> col_ids;
    col_ids.push_back(StorageIndex(2));  // l_suppkey
    col_ids.push_back(StorageIndex(5));  // l_extendedprice
    col_ids.push_back(StorageIndex(6));  // l_discount
    lineitem_table.GetStorage().InitializeScan(context.client, lineitem_transaction, scan_state, col_ids);

    vector<LogicalType> types;
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[2]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[5]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[6]);

    std::cout << "\n[Pre-load] Streaming lineitem (suppkey, price, discount) ..." << std::endl;
    auto t_pre0 = std::chrono::high_resolution_clock::now();

    // DuckDB stores l_suppkey as BIGINT (int64) — verified via
    // information_schema.columns at SF10.  We read it as int64 and
    // narrow to int32 element-wise (suppkey ∈ [1, 100k×SF] fits
    // comfortably).  l_extendedprice / l_discount are DECIMAL(15,2)
    // → int64 on disk; FlatVector<int64_t> reads them raw.
    std::vector<int32_t> col_suppkey;
    std::vector<int64_t> col_price;
    std::vector<int64_t> col_disc;
    col_suppkey.reserve(64'000'000);
    col_price.reserve(64'000'000);
    col_disc.reserve(64'000'000);

    while (true) {
        DataChunk chunk;
        chunk.Initialize(context.client, types);
        lineitem_table.GetStorage().Scan(lineitem_transaction, chunk, scan_state);
        if (chunk.size() == 0) break;
        auto sk = FlatVector::GetData<int64_t>(chunk.data[0]);   // BIGINT
        auto pp = FlatVector::GetData<int64_t>(chunk.data[1]);   // DECIMAL→int64
        auto dp = FlatVector::GetData<int64_t>(chunk.data[2]);   // DECIMAL→int64
        size_t old = col_suppkey.size();
        col_suppkey.resize(old + chunk.size());
        col_price.resize(old + chunk.size());
        col_disc.resize(old + chunk.size());
        for (size_t i = 0; i < chunk.size(); i++)
            col_suppkey[old + i] = static_cast<int32_t>(sk[i]);
        std::memcpy(col_price.data() + old, pp, chunk.size() * sizeof(int64_t));
        std::memcpy(col_disc.data()  + old, dp, chunk.size() * sizeof(int64_t));
    }
    const size_t num_rows = col_suppkey.size();
    int32_t max_suppkey = 0;
    for (auto sk : col_suppkey) if (sk > max_suppkey) max_suppkey = sk;
    auto t_pre1 = std::chrono::high_resolution_clock::now();
    std::cout << "[Pre-load] " << num_rows << " rows in " << ms(t_pre0, t_pre1)
              << " ms  (max_suppkey=" << max_suppkey << ")" << std::endl;

    if (num_rows == 0) {
        std::cerr << "Q15: lineitem is empty?" << std::endl;
        return;
    }

    const int32_t* sp = col_suppkey.data();
    const int64_t* pp = col_price.data();
    const int64_t* dp = col_disc.data();
    const size_t   sk_dim = static_cast<size_t>(max_suppkey) + 1;

    // ============================================================
    // 2. Look up the shipdate bitmap index (built by PRAGMA load_bitmap).
    //    Same pattern as Q14: backend-specific dynamic_cast on the
    //    shared context.client.bitmap_shipdate slot.
    // ============================================================
    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    if (!idx_ship) {
        std::cerr << "Q15: bitmap_shipdate not loaded.  "
                     "Run PRAGMA load_bitmap('shipdate'); first." << std::endl;
        return;
    }
    auto* cb_ship  = dynamic_cast<bm_index::IndexedComBit*>(idx_ship);
    auto* cr_ship  = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ship);
    auto* wah_ship = dynamic_cast<bm_index::IndexedWAH*>(idx_ship);
    auto* ew_ship  = dynamic_cast<bm_index::IndexedEWAH*>(idx_ship);
    auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_ship);

    // ============================================================
    // 3. Per-backend aggregation buffer (one int64 per suppkey).
    //    Reset to zero at the top of every iteration; result of the
    //    last iteration is what gets validated against ground truth.
    // ============================================================
    std::vector<int64_t> rev_buf(sk_dim, 0);

    // Per-iteration timing vectors (median ± stddev computed over
    // measured iterations, i.e. iterations beyond the warm-up).
    std::vector<double> cb_or_t,  cb_agg_t,  cb_tot_t;
    std::vector<double> cr_or_t,  cr_agg_t,  cr_tot_t;
    std::vector<double> crr_or_t, crr_agg_t, crr_tot_t;
    std::vector<double> wah_or_t, wah_agg_t, wah_tot_t;
    std::vector<double> ew_or_t,  ew_agg_t,  ew_tot_t;
    std::vector<double> con_or_t, con_agg_t, con_tot_t;

    // Final validated state per backend (populated on the last iter).
    int64_t cb_max = 0,  cr_max = 0,  crr_max = 0, wah_max = 0;
    int64_t ew_max = 0,  con_max = 0;
    std::vector<int32_t> cb_ties,  cr_ties,  crr_ties, wah_ties;
    std::vector<int32_t> ew_ties,  con_ties;
    // ============================================================
    // 4. Benchmark loop — per backend:
    //    OR  (T0→T1): build the shipdate filter
    //    Agg (T1→T2): walk the filter, accumulate revenue_by_suppkey,
    //                 then scan the buffer for max + tied keys
    // ============================================================
    for (int iter = 0; iter < Q15_ITERATIONS; iter++) {
        bool warmup = (iter < Q15_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q15_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ================= ComBit =================
        if (run_cb() && cb_ship) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ComBit cb_filt = ComBit::from_sparse_positions(
                {}, cb_ship->num_rows(), cb_ship->segment_bits());
            cb_ship->apply_or_range_to(cb_filt, Q15_DATE_START, Q15_DATE_END);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            size_t row_base = 0;
            for (size_t s = 0; s < cb_filt.num_segments(); s++) {
                const auto& seg = cb_filt.segment(s);
                const uint8_t* data = seg.l1_literal_data();
                size_t n = seg.num_literals();
                for (size_t bi = 0; bi < n; bi++) {
                    uint8_t b = data[bi];
                    if (b == 0) { row_base += 8; continue; }
                    const auto& entry = bm_bench::byte_lut[b];
                    for (int k = 0; k < entry.count; k++) {
                        size_t r = row_base + entry.pos[k];
                        rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
                    }
                    row_base += 8;
                }
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  CB:   OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                cb_or_t.push_back(d_or);
                cb_agg_t.push_back(d_agg);
                cb_tot_t.push_back(d_total);
            }
            cb_max = pr.first; cb_ties = std::move(pr.second);
        }

        // ================= CRoaring (vanilla addMany) =================
        if (run_cr() && cr_ship && !cr_ship->run_optimized()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring cr_filt = cr_ship->or_range(Q15_DATE_START, Q15_DATE_END);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            for (auto it = cr_filt.begin(); it != cr_filt.end(); ++it) {
                size_t r = *it;
                rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  CR:   OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                cr_or_t.push_back(d_or);
                cr_agg_t.push_back(d_agg);
                cr_tot_t.push_back(d_total);
            }
            cr_max = pr.first; cr_ties = std::move(pr.second);
        }

        // ================= CRoaring + Run (fastunion) =================
        if (run_crr() && cr_ship && cr_ship->run_optimized()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring crr_filt = cr_ship->or_range(Q15_DATE_START, Q15_DATE_END);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            for (auto it = crr_filt.begin(); it != crr_filt.end(); ++it) {
                size_t r = *it;
                rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  CRR:  OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                crr_or_t.push_back(d_or);
                crr_agg_t.push_back(d_agg);
                crr_tot_t.push_back(d_total);
            }
            crr_max = pr.first; crr_ties = std::move(pr.second);
        }

        // ================= WAH =================
        if (run_wah() && wah_ship) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ibis::bitvector wah_filt = wah_ship->or_range(Q15_DATE_START, Q15_DATE_END);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            ibis::bitvector::pit pit(wah_filt);
            while (*pit != 0xFFFFFFFFU) {
                size_t r = *pit;
                rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
                pit.next();
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  WAH:  OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                wah_or_t.push_back(d_or);
                wah_agg_t.push_back(d_agg);
                wah_tot_t.push_back(d_total);
            }
            wah_max = pr.first; wah_ties = std::move(pr.second);
        }

        // ================= EWAH (fast_logicalor) =================
        if (run_ew() && ew_ship) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ewah::EWAHBoolArray<uint64_t> ew_filt = ew_ship->or_range(Q15_DATE_START, Q15_DATE_END);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            for (auto it = ew_filt.begin(); it != ew_filt.end(); ++it) {
                size_t r = *it;
                rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  EW:   OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                ew_or_t.push_back(d_or);
                ew_agg_t.push_back(d_agg);
                ew_tot_t.push_back(d_total);
            }
            ew_max = pr.first; ew_ties = std::move(pr.second);
        }

        // ================= Concise (fast_logicalor) =================
        if (run_con() && con_ship) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ConciseSet<false> con_filt = con_ship->or_range(Q15_DATE_START, Q15_DATE_END);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            for (auto it = con_filt.begin(); it != con_filt.end(); ++it) {
                size_t r = *it;
                rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  CON:  OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                con_or_t.push_back(d_or);
                con_agg_t.push_back(d_agg);
                con_tot_t.push_back(d_total);
            }
            con_max = pr.first; con_ties = std::move(pr.second);
        }
    } // end iteration loop

    // ============================================================
    // 5. Cross-backend consistency check
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q15 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Pick a canonical print source (prefer ComBit when ALL).
    int64_t print_max = 0;
    const std::vector<int32_t>* print_ties = nullptr;
    const char* print_label = "";
    if      (Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CB
          || Q15_BM == Q15BmType::CB_BPE)                          { print_max = cb_max;  print_ties = &cb_ties;  print_label = "ComBit"; }
    else if (Q15_BM == Q15BmType::WAH)                             { print_max = wah_max; print_ties = &wah_ties; print_label = "WAH"; }
    else if (Q15_BM == Q15BmType::CR || Q15_BM == Q15BmType::CR_BPE) { print_max = cr_max;  print_ties = &cr_ties;  print_label = "CRoaring"; }
    else if (Q15_BM == Q15BmType::CRR)                             { print_max = crr_max; print_ties = &crr_ties; print_label = "CRoaring+Run"; }
    else if (Q15_BM == Q15BmType::EW)                              { print_max = ew_max;  print_ties = &ew_ties;  print_label = "EWAH"; }
    else if (Q15_BM == Q15BmType::CON)                             { print_max = con_max; print_ties = &con_ties; print_label = "Concise"; }

    if (print_ties) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Q15 Result (" << print_label << "):" << std::endl;
        std::cout << "    max total_revenue = " << q15_fp_to_dollars(print_max)
                  << "   (fp=" << print_max << ")"
                  << "   ties=" << print_ties->size() << std::endl;
        std::cout << "    suppkey(s):";
        for (auto k : *print_ties) std::cout << " " << k;
        std::cout << std::endl;
    }

    if (run_all()) {
        bool consistent = true;
        auto cmp = [&](const char* lbl, int64_t mx, const std::vector<int32_t>& ties) {
            if (mx != cb_max || ties != cb_ties) {
                std::cout << "  *** MISMATCH " << lbl
                          << " (max=" << mx << " ties=" << ties.size()
                          << " vs CB max=" << cb_max
                          <<  " ties=" << cb_ties.size() << ") ***\n";
                consistent = false;
            }
        };
        cmp("CR",  cr_max,  cr_ties);
        cmp("CRR", crr_max, crr_ties);
        cmp("WAH", wah_max, wah_ties);
        cmp("EW",  ew_max,  ew_ties);
        cmp("CON", con_max, con_ties);
        std::cout << "  Consistency: " << (consistent ? "ALL MATCH" : "MISMATCH DETECTED") << std::endl;
    }

    // ============================================================
    // 6. Lookup supplier metadata for the tied keys + DuckDB native
    //    SQL ground truth (single CTE-form Q15 query).
    // ============================================================
    if (!print_ties || print_ties->empty()) {
        std::cerr << "Q15: no result rows produced." << std::endl;
        return;
    }

    // Build SQL IN-list of tied suppkeys.
    std::ostringstream in_list;
    in_list << "(";
    for (size_t i = 0; i < print_ties->size(); i++) {
        if (i) in_list << ",";
        in_list << (*print_ties)[i];
    }
    in_list << ")";

    std::vector<Q15Row> bm_rows;
    bm_rows.reserve(print_ties->size());
    {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT s_suppkey, s_name, s_address, s_phone "
            "FROM supplier "
            "WHERE s_suppkey IN " + in_list.str() +
            " ORDER BY s_suppkey";
        auto r = con.Query(sql);
        if (r && !r->HasError()) {
            for (idx_t i = 0; i < r->RowCount(); i++) {
                Q15Row row;
                row.s_suppkey   = r->GetValue(0, i).GetValue<int32_t>();
                row.s_name      = r->GetValue(1, i).ToString();
                row.s_address   = r->GetValue(2, i).ToString();
                row.s_phone     = r->GetValue(3, i).ToString();
                row.total_rev_fp = print_max;  // shared by all ties by construction
                bm_rows.push_back(std::move(row));
            }
        } else if (r && r->HasError()) {
            std::cerr << "Q15: supplier lookup failed: " << r->GetError() << std::endl;
        }
    }

    std::cout << "\n  Tied supplier rows (bitmap pipeline):" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    for (auto& row : bm_rows) {
        std::cout << "    " << row.s_suppkey
                  << "  " << row.s_name
                  << "  total_revenue=" << q15_fp_to_dollars(row.total_rev_fp)
                  << std::endl;
    }

    // --- DuckDB native SQL ground truth (CTE form, semantically
    //     identical to the spec view-based query). ---
    struct GtRow {
        int32_t s_suppkey;
        std::string s_name, s_address, s_phone;
        double total_revenue;
    };
    std::vector<GtRow> gt_rows;
    double gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "WITH revenue0 AS ("
            "  SELECT l_suppkey AS supplier_no,"
            "         sum(l_extendedprice * (1 - l_discount)) AS total_revenue"
            "    FROM lineitem"
            "   WHERE l_shipdate >= DATE '1996-01-01'"
            "     AND l_shipdate <  DATE '1996-04-01'"
            "   GROUP BY l_suppkey"
            ")"
            "SELECT s.s_suppkey, s.s_name, s.s_address, s.s_phone, r.total_revenue "
            "  FROM supplier s, revenue0 r "
            " WHERE s.s_suppkey = r.supplier_no "
            "   AND r.total_revenue = (SELECT max(total_revenue) FROM revenue0) "
            " ORDER BY s.s_suppkey";
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = con.Query(sql);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (r && !r->HasError()) {
            gt_sql_ms = ms(t0, t1);
            for (idx_t i = 0; i < r->RowCount(); i++) {
                GtRow gr;
                gr.s_suppkey     = r->GetValue(0, i).GetValue<int32_t>();
                gr.s_name        = r->GetValue(1, i).ToString();
                gr.s_address     = r->GetValue(2, i).ToString();
                gr.s_phone       = r->GetValue(3, i).ToString();
                gr.total_revenue = r->GetValue(4, i).GetValue<double>();
                gt_rows.push_back(std::move(gr));
            }
        } else if (r && r->HasError()) {
            std::cerr << "[Baseline] SQL error: " << r->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    std::cout << "\n[Baseline] DuckDB native SQL  (single run: "
              << std::fixed << std::setprecision(2) << gt_sql_ms << " ms)" << std::endl;
    std::cout << "  SQL ground truth rows: " << gt_rows.size() << std::endl;
    for (auto& gr : gt_rows) {
        std::cout << "    " << gr.s_suppkey << "  " << gr.s_name
                  << "  total_revenue=" << gr.total_revenue << std::endl;
    }

    // --- Compare bitmap result vs SQL ground truth ---
    bool ok = (bm_rows.size() == gt_rows.size());
    if (ok) {
        for (size_t i = 0; i < bm_rows.size(); i++) {
            if (bm_rows[i].s_suppkey != gt_rows[i].s_suppkey) { ok = false; break; }
            // Cross-check fp accumulator vs SQL revenue (within 0.01 cents).
            double diff = std::abs(q15_fp_to_dollars(bm_rows[i].total_rev_fp)
                                   - gt_rows[i].total_revenue);
            if (diff > 0.01) { ok = false; break; }
        }
    }
    if (ok) {
        std::cout << "[OK] all active backends match DuckDB SQL ground truth ("
                  << bm_rows.size() << " row(s), revenue within 0.01)." << std::endl;
    } else {
        std::cerr << "[FAIL] bitmap result and SQL ground truth differ." << std::endl;
    }

    // ============================================================
    // 7. Final timing summary (median ± stddev)
    // ============================================================
    auto print_stats = [&](const char* lbl, std::vector<double>& or_t,
                           std::vector<double>& agg_t, std::vector<double>& tot_t) {
        if (or_t.empty()) return;
        auto so = bm_bench::compute_stats(or_t);
        auto sa = bm_bench::compute_stats(agg_t);
        auto st = bm_bench::compute_stats(tot_t);
        std::cout << "  " << std::left << std::setw(5) << lbl
                  << " OR="     << std::fixed << std::setprecision(2) << std::setw(7) << so.median
                  << " +/- "    << std::setw(5) << so.stddev
                  << "  Agg="   << std::setw(7) << sa.median
                  << " +/- "    << std::setw(5) << sa.stddev
                  << "  Total=" << std::setw(7) << st.median
                  << " +/- "    << std::setw(5) << st.stddev
                  << " (ms)" << std::endl;
    };

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q15 RESULTS (" << (Q15_ITERATIONS - Q15_WARMUP)
              << " measured iterations, median +/- stddev)" << std::endl;
    std::cout << "================================================================" << std::endl;
    print_stats("CB",  cb_or_t,  cb_agg_t,  cb_tot_t);
    print_stats("CR",  cr_or_t,  cr_agg_t,  cr_tot_t);
    print_stats("CRR", crr_or_t, crr_agg_t, crr_tot_t);
    print_stats("WAH", wah_or_t, wah_agg_t, wah_tot_t);
    print_stats("EW",  ew_or_t,  ew_agg_t,  ew_tot_t);
    print_stats("CON", con_or_t, con_agg_t, con_tot_t);
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 8. CSV export (wide schema matching Q1/Q3/Q4/Q14 — bench_suite.py
    //    parses one CSV per backend and merges into Excel rows.  Each
    //    `operation` row carries one cell per backend; the active backend
    //    fills its column, the rest are zero.)
    // ============================================================
    {
        std::string sf       = q15_get_sf_label();
        std::string csv_path = "q15_results_" + sf + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << std::fixed << std::setprecision(4);
            csv << "sf,operation,"
                << "wah_median_ms,wah_stddev_ms,wah_min_ms,wah_max_ms,"
                << "combit_median_ms,combit_stddev_ms,combit_min_ms,combit_max_ms,"
                << "croaring_median_ms,croaring_stddev_ms,croaring_min_ms,croaring_max_ms,"
                << "croaring_run_median_ms,croaring_run_stddev_ms,croaring_run_min_ms,croaring_run_max_ms,"
                << "ewah_median_ms,ewah_stddev_ms,ewah_min_ms,ewah_max_ms,"
                << "bs_median_ms,bs_stddev_ms,bs_min_ms,bs_max_ms,"
                << "bsa_median_ms,bsa_stddev_ms,bsa_min_ms,bsa_max_ms,"
                << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms,"
                << "cb_vs_wah,cr_vs_wah,crr_vs_wah,ew_vs_wah,bs_vs_wah,bsa_vs_wah,con_vs_wah\n";

            bm_bench::Stats z{0, 0, 0, 0};
            auto stats_or_z = [&](std::vector<double>& v) {
                return v.empty() ? z : bm_bench::compute_stats(v);
            };

            // Pick the *one* active backend's stats for this run.  When
            // DEBIT_BM=all every vector is populated; we still emit one
            // row per operation with the active backend cell set, which
            // bench_suite's _merge_pattern_csvs combines across runs.
            struct PerOp { bm_bench::Stats wah, cb, cr, crr, ew, con; };
            auto fill = [&](std::vector<double>& wah_v,
                            std::vector<double>& cb_v,
                            std::vector<double>& cr_v,
                            std::vector<double>& crr_v,
                            std::vector<double>& ew_v,
                            std::vector<double>& con_v) {
                return PerOp{
                    stats_or_z(wah_v),
                    stats_or_z(cb_v),
                    stats_or_z(cr_v),
                    stats_or_z(crr_v),
                    stats_or_z(ew_v),
                    stats_or_z(con_v),
                };
            };

            PerOp op_or  = fill(wah_or_t,  cb_or_t,  cr_or_t,  crr_or_t,  ew_or_t,  con_or_t);
            PerOp op_agg = fill(wah_agg_t, cb_agg_t, cr_agg_t, crr_agg_t, ew_agg_t, con_agg_t);
            PerOp op_tot = fill(wah_tot_t, cb_tot_t, cr_tot_t, crr_tot_t, ew_tot_t, con_tot_t);

            auto put = [&](const std::string& opname, const PerOp& p) {
                csv << sf << "," << opname << ",";
                auto cell = [&](bm_bench::Stats v) {
                    csv << v.median << "," << v.stddev << "," << v.min_val << "," << v.max_val << ",";
                };
                cell(p.wah);
                cell(p.cb);
                cell(p.cr);
                cell(p.crr);
                cell(p.ew);
                cell(z); cell(z);   // BS / BSA: not run for Q15 in PRAGMA-load pattern
                cell(p.con);
                csv << "0,0,0,0,0,0,0\n";
            };

            put("PhaseA_OR",  op_or);
            put("PhaseB_Agg", op_agg);
            put("TOTAL",      op_tot);

            std::cout << "\n  [CSV] q15_results_" << sf << ".csv\n";
        }
    }

    });  // end std::call_once
}

}  // namespace duckdb
