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

#include "bitset_simple.h"
#include "Concise/concise.h"
#include "execution/tpch/bm_baseline_loaders.hpp"
#include "execution/tpch/bm_bench_common.hpp"

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

// --- Q15 bitmap directories ---
static const std::string Q15_SF      = bm_bench::sf_suffix();
static const std::string Q15_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q15" + Q15_SF + "_combit");
static const std::string Q15_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q15" + Q15_SF + "_wah");
static const std::string Q15_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q15" + Q15_SF + "_croaring");
static const std::string Q15_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q15" + Q15_SF + "_ewah");

// --- Backend selection (DEBIT_BM=all|wah|cb|cr|crr|ew|bs|bsa|con) ---
using Q15BmType = bm_bench::Backend;
static const Q15BmType Q15_BM = bm_bench::parse_backend("Q15_BM");

static bool run_all() { return Q15_BM == Q15BmType::ALL; }
static bool run_wah() { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::WAH; }
static bool run_cb()  { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CB;  }
static bool run_cr()  { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CR;  }
static bool run_crr() { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CRR; }
static bool run_ew()  { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::EW;  }
static bool run_bs()  { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::BS;  }
static bool run_bsa() { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::BSA; }
static bool run_con() { return Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CON; }

static const char* q15_bm_label()     { return bm_bench::backend_label(Q15_BM); }
static std::string q15_get_sf_label() { return bm_bench::sf_label(); }

// --- Q15 predicate (TPC-H spec §2.4.15, qualification DATE = 1996-01-01) ---
// Day range [1996-01-01, 1996-04-01) = days [1461, 1552) since 1992-01-01
//   (1992 leap, 1996 leap):
//     1996-01-01 = 366 + 365 + 365 + 365         = 1461
//     1996-04-01 = 1461 + 31 + 29 + 31           = 1552  (exclusive)
//   91 day bitmaps, inclusive day range [1461, 1551].
static const int Q15_DATE_START = 1461;  // 1996-01-01 inclusive
static const int Q15_DATE_END   = 1551;  // 1996-03-31 inclusive (= 1552-1)

// --- Iteration counts (DEBIT_ITER / DEBIT_WARMUP) ---
static const int Q15_ITERATIONS = bm_bench::iter_count(10);
static const int Q15_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q15_once_flag;

// Statistics helper.
struct Q15Stats {
    double median = 0, stddev = 0, min_val = 0, max_val = 0;
};
static Q15Stats q15_compute_stats(std::vector<double>& v) {
    Q15Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    s.median  = (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
    s.min_val = v.front();
    s.max_val = v.back();
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double sq = 0;
    for (auto x : v) sq += (x - mean) * (x - mean);
    s.stddev = std::sqrt(sq / n);
    return s;
}

// Bitmap loaders (same set as Q1/Q12/Q14).
static ComBit q15_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q15_load_cr(const std::string& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) return roaring::Roaring();
    auto sz = in.tellg(); in.seekg(0);
    uint32_t ls; in.read(reinterpret_cast<char*>(&ls), 4);
    auto bsz = sz - 4;
    if (bsz > 0) {
        std::vector<char> buf(bsz);
        in.read(buf.data(), bsz);
        return roaring::Roaring::readSafe(buf.data(), bsz);
    }
    return roaring::Roaring();
}
static ibis::bitvector q15_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q15_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// Byte-LUT for MSB-first bit extraction (same as Q1/Q14).
struct Q15ByteEntry { uint8_t count; uint8_t pos[8]; };
static Q15ByteEntry q15_byte_lut[256];
static bool q15_byte_lut_init = []() {
    for (int v = 0; v < 256; v++) {
        uint8_t c = 0;
        for (int b = 7; b >= 0; b--)
            if (v & (1 << b))
                q15_byte_lut[v].pos[c++] = 7 - b;
        q15_byte_lut[v].count = c;
    }
    return true;
}();

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
    std::cout << "  Bitmap dirs:";
    if (run_cb())              std::cout << " " << Q15_CB_DIR;
    if (run_cr() || run_crr()) std::cout << " " << Q15_CR_DIR;
    if (run_wah())             std::cout << " " << Q15_WAH_DIR;
    if (run_ew())              std::cout << " " << Q15_EW_DIR;
    std::cout << std::endl;
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
    // 2. Load shipdate bitmaps
    // ============================================================
    std::cout << "\n[Load] Loading shipdate bitmaps (mode=" << q15_bm_label() << ")..." << std::endl;
    const int n_dates = Q15_DATE_END - Q15_DATE_START + 1;

    // --- ComBit ---
    std::vector<ComBit> cb_date;
    std::vector<const ComBit*> cb_date_ptrs;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_date.resize(Q15_DATE_END + 1);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            cb_date[d] = q15_load_cb(Q15_CB_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cb_date_ptrs.reserve(n_dates);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            cb_date_ptrs.push_back(&cb_date[d]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring (vanilla) ---
    std::vector<roaring::Roaring> cr_date;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_date.resize(Q15_DATE_END + 1);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            cr_date[d] = q15_load_cr(Q15_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring + Run (loads fresh + runOptimize) ---
    std::vector<roaring::Roaring> crr_date;
    std::vector<const roaring::Roaring*> crr_date_ptrs;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_date.resize(Q15_DATE_END + 1);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++) {
            crr_date[d] = q15_load_cr(Q15_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
            crr_date[d].runOptimize();
        }
        crr_date_ptrs.reserve(n_dates);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            crr_date_ptrs.push_back(&crr_date[d]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- WAH ---
    std::vector<ibis::bitvector> wah_date;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wah_date.resize(Q15_DATE_END + 1);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            wah_date[d] = q15_load_wah(Q15_WAH_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- EWAH ---
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_date;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_date_ptrs;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ew_date.resize(Q15_DATE_END + 1);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            ew_date[d] = q15_load_ew(Q15_EW_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        ew_date_ptrs.reserve(n_dates);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            ew_date_ptrs.push_back(&ew_date[d]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Bitset (BS / BSA share) ---
    std::vector<bs::Bitmap> bs_date;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_date.resize(Q15_DATE_END + 1);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            bs_date[d] = bm_bench::load_bitmap_from_croaring(
                Q15_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Concise ---
    std::vector<ConciseSet<false>> con_date;
    std::vector<const ConciseSet<false>*> con_date_ptrs;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_date.resize(Q15_DATE_END + 1);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            con_date[d] = bm_bench::load_concise_from_croaring(
                Q15_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        con_date_ptrs.reserve(n_dates);
        for (int d = Q15_DATE_START; d <= Q15_DATE_END; d++)
            con_date_ptrs.push_back(&con_date[d]);
        con_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:      " << wah_load_ms << " ms" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:   " << cb_load_ms  << " ms" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load: " << cr_load_ms  << " ms" << std::endl;
    if (run_crr()) std::cout << "  CRR load:      " << crr_load_ms << " ms" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:     " << ew_load_ms  << " ms" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:   " << bs_load_ms  << " ms (shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:  " << con_load_ms << " ms" << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q15_WAH_DIR)) << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q15_CB_DIR))  << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q15_CR_DIR))  << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q15_EW_DIR))  << " MiB" << std::endl;

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
    std::vector<double> bs_or_t,  bs_agg_t,  bs_tot_t;
    std::vector<double> bsa_or_t, bsa_agg_t, bsa_tot_t;
    std::vector<double> con_or_t, con_agg_t, con_tot_t;

    // Final validated state per backend (populated on the last iter).
    int64_t cb_max = 0,  cr_max = 0,  crr_max = 0, wah_max = 0;
    int64_t ew_max = 0,  bs_max = 0,  bsa_max = 0, con_max = 0;
    std::vector<int32_t> cb_ties,  cr_ties,  crr_ties, wah_ties;
    std::vector<int32_t> ew_ties,  bs_ties,  bsa_ties, con_ties;
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
        // OR_many returns a Decompressed ComBit (the canonical
        // post-merge state).  We then walk it directly via
        // `seg.l1_literal_data()` + the byte-LUT — the same fast path
        // Q1 / Q14 use for per-row aggregation.  No `&=` here because
        // the filter is already final after OR_many (Q15's only
        // bitmap-side predicate is the 91-day shipdate window).
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ComBit cb_filt = ComBit::OR_many(cb_date_ptrs.size(), cb_date_ptrs.data());
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
                    const auto& entry = q15_byte_lut[b];
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

        // ================= CRoaring (vanilla pairwise) =================
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring cr_filt = cr_date[Q15_DATE_START];
            for (int d = Q15_DATE_START + 1; d <= Q15_DATE_END; d++)
                cr_filt |= cr_date[d];
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
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring crr_filt = roaring::Roaring::fastunion(
                crr_date_ptrs.size(), crr_date_ptrs.data());
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
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ibis::bitvector wah_filt = wah_date[Q15_DATE_START];
            wah_filt.decompress();
            for (int d = Q15_DATE_START + 1; d <= Q15_DATE_END; d++)
                wah_filt |= wah_date[d];
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
        if (run_ew()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ewah::EWAHBoolArray<uint64_t> ew_filt = ewah::fast_logicalor(
                ew_date_ptrs.size(), ew_date_ptrs.data());
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

        // ================= Bitset (scalar) =================
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bs_filt = bs_date[Q15_DATE_START].clone();
            for (int d = Q15_DATE_START + 1; d <= Q15_DATE_END; d++)
                bs::or_inplace(bs_filt, bs_date[d], false);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            for (size_t i = 0; i < bs_filt.nwords; ++i) {
                uint64_t w = bs_filt.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= bs_filt.nbits) break;
                    rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
                    w &= w - 1;
                }
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  BS:   OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                bs_or_t.push_back(d_or);
                bs_agg_t.push_back(d_agg);
                bs_tot_t.push_back(d_total);
            }
            bs_max = pr.first; bs_ties = std::move(pr.second);
        }

        // ================= Bitset + AVX-512 =================
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bsa_filt = bs_date[Q15_DATE_START].clone();
            for (int d = Q15_DATE_START + 1; d <= Q15_DATE_END; d++)
                bs::or_inplace(bsa_filt, bs_date[d], true);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::fill(rev_buf.begin(), rev_buf.end(), 0);
            for (size_t i = 0; i < bsa_filt.nwords; ++i) {
                uint64_t w = bsa_filt.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= bsa_filt.nbits) break;
                    rev_buf[sp[r]] += Q15_REV_CONTRIB(pp, dp, r);
                    w &= w - 1;
                }
            }
            auto pr = q15_find_max(rev_buf);
            auto t2 = std::chrono::high_resolution_clock::now();

            double d_or = ms(t0, t1), d_agg = ms(t1, t2);
            double d_total = ms(t0, t2);
            std::cout << "  BSA:  OR=" << d_or << "  Agg=" << d_agg
                      << "  Total=" << d_total
                      << "  max=" << q15_fp_to_dollars(pr.first)
                      << "  ties=" << pr.second.size() << std::endl;
            if (!warmup) {
                bsa_or_t.push_back(d_or);
                bsa_agg_t.push_back(d_agg);
                bsa_tot_t.push_back(d_total);
            }
            bsa_max = pr.first; bsa_ties = std::move(pr.second);
        }

        // ================= Concise (fast_logicalor) =================
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ConciseSet<false> con_filt = ConciseSet<false>::fast_logicalor(
                con_date_ptrs.size(), con_date_ptrs.data());
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
    if      (Q15_BM == Q15BmType::ALL || Q15_BM == Q15BmType::CB)  { print_max = cb_max;  print_ties = &cb_ties;  print_label = "ComBit"; }
    else if (Q15_BM == Q15BmType::WAH)                             { print_max = wah_max; print_ties = &wah_ties; print_label = "WAH"; }
    else if (Q15_BM == Q15BmType::CR)                              { print_max = cr_max;  print_ties = &cr_ties;  print_label = "CRoaring"; }
    else if (Q15_BM == Q15BmType::CRR)                             { print_max = crr_max; print_ties = &crr_ties; print_label = "CRoaring+Run"; }
    else if (Q15_BM == Q15BmType::EW)                              { print_max = ew_max;  print_ties = &ew_ties;  print_label = "EWAH"; }
    else if (Q15_BM == Q15BmType::BS)                              { print_max = bs_max;  print_ties = &bs_ties;  print_label = "Bitset"; }
    else if (Q15_BM == Q15BmType::BSA)                             { print_max = bsa_max; print_ties = &bsa_ties; print_label = "Bitset+AVX512"; }
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
        cmp("BS",  bs_max,  bs_ties);
        cmp("BSA", bsa_max, bsa_ties);
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
        auto so = q15_compute_stats(or_t);
        auto sa = q15_compute_stats(agg_t);
        auto st = q15_compute_stats(tot_t);
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
    print_stats("BS",  bs_or_t,  bs_agg_t,  bs_tot_t);
    print_stats("BSA", bsa_or_t, bsa_agg_t, bsa_tot_t);
    print_stats("CON", con_or_t, con_agg_t, con_tot_t);
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 8. CSV export (optional reproducibility artifact)
    // ============================================================
    {
        std::string csv_path = "q15_results_" + q15_get_sf_label() + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << "backend,or_ms,agg_ms,total_ms\n";
            auto wrow = [&](const char* lbl, std::vector<double>& or_t,
                            std::vector<double>& agg_t, std::vector<double>& tot_t) {
                if (or_t.empty()) return;
                auto so = q15_compute_stats(or_t);
                auto sa = q15_compute_stats(agg_t);
                auto st = q15_compute_stats(tot_t);
                csv << lbl << "," << so.median << "," << sa.median << "," << st.median << "\n";
            };
            wrow("CB",  cb_or_t,  cb_agg_t,  cb_tot_t);
            wrow("CR",  cr_or_t,  cr_agg_t,  cr_tot_t);
            wrow("CRR", crr_or_t, crr_agg_t, crr_tot_t);
            wrow("WAH", wah_or_t, wah_agg_t, wah_tot_t);
            wrow("EW",  ew_or_t,  ew_agg_t,  ew_tot_t);
            wrow("BS",  bs_or_t,  bs_agg_t,  bs_tot_t);
            wrow("BSA", bsa_or_t, bsa_agg_t, bsa_tot_t);
            wrow("CON", con_or_t, con_agg_t, con_tot_t);
            std::cout << "\n  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    });  // end std::call_once
}

}  // namespace duckdb
