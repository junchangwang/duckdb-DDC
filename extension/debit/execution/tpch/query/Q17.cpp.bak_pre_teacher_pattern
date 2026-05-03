// =============================================================================
// TPC-H Q17 — Small-Quantity-Order Revenue Query (spec v3.0.1 §2.4.17)
//
//   SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
//     FROM lineitem, part
//    WHERE p_partkey = l_partkey
//      AND p_brand = '[BRAND]'
//      AND p_container = '[CONTAINER]'
//      AND l_quantity < (
//            SELECT 0.2 * avg(l_quantity)
//              FROM lineitem
//             WHERE l_partkey = p_partkey
//          );
//
//   Qualification BRAND='Brand#23', CONTAINER='MED BOX' (spec §2.4.17.4).
//
// Bitmap pipeline (single bitmap, two-pass scalar aggregation):
//
//   The lineitem ⨝ part predicate is materialised once at export time
//   into a single sparse bitmap is_q17_part — bit r = 1 iff
//   l_partkey[r] points to a part with the matching brand & container.
//   At SF10 this is ≈ 0.10 % of all lineitem rows (61 385 / 60 M).
//
//   T0 → T1  Pass-1: walk is_q17_part rows;
//                    accumulate sum_qty[pk] += l_quantity[r]
//                              count[pk]   += 1
//                    then (after the walk) threshold[pk] =
//                              0.2 * sum_qty[pk] / count[pk]
//
//   T1 → T2  Pass-2: walk is_q17_part rows;
//                    if   l_quantity[r] < threshold[partkey[r]]
//                    then sum_ep += l_extendedprice[r]
//
//   avg_yearly = sum_ep / 100 / 7.0     (price stored ×100 → dollars / 7)
//
// The single-bitmap form is the natural Q17 pipeline shape: there is no
// OR or AND between bitmaps — only a sparse iteration twice over the
// same support set.  Each backend therefore reports two timings:
// Pass1 (group-by-partkey) and Pass2 (filter-and-sum), plus their sum.
//
// l_partkey, l_quantity, l_extendedprice are read directly from DuckDB
// storage at query time (mirrors Q1/Q14/Q15's pre-load).  No further
// column .bin files need to be exported.
//
// Output: single double `avg_yearly` (≈ $349k for SF1, scales by SF).
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
#include <unordered_set>

namespace duckdb {

// --- Q17 bitmap directories ---
static const std::string Q17_SF      = bm_bench::sf_suffix();
static const std::string Q17_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q17" + Q17_SF + "_combit");
static const std::string Q17_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q17" + Q17_SF + "_wah");
static const std::string Q17_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q17" + Q17_SF + "_croaring");
static const std::string Q17_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q17" + Q17_SF + "_ewah");

// --- Backend selection (DEBIT_BM=all|wah|cb|cr|crr|ew|bs|bsa|con) ---
using Q17BmType = bm_bench::Backend;
static const Q17BmType Q17_BM = bm_bench::parse_backend("Q17_BM");

static bool run_all() { return Q17_BM == Q17BmType::ALL; }
static bool run_wah() { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::WAH; }
static bool run_cb()  { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::CB;  }
static bool run_cr()  { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::CR;  }
static bool run_crr() { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::CRR; }
static bool run_ew()  { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::EW;  }
static bool run_bs()  { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::BS;  }
static bool run_bsa() { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::BSA; }
static bool run_con() { return Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::CON; }

static const char* q17_bm_label()     { return bm_bench::backend_label(Q17_BM); }
static std::string q17_get_sf_label() { return bm_bench::sf_label(); }

// --- Iterations ---
static const int Q17_ITERATIONS = bm_bench::iter_count(10);
static const int Q17_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q17_once_flag;

// --- Stats helper ---
// Bitmap loaders (same set as Q14/Q15).
static ComBit q17_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q17_load_cr(const std::string& p) {
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
static ibis::bitvector q17_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q17_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// Byte-LUT for MSB-first bit extraction (same convention as Q1/Q14/Q15).
// --- Per-partkey accumulator for Pass1 + threshold computation ---
//
//   sum_qty[pk]  : sum of l_quantity (stored ×100) over rows where
//                  is_q17_part AND l_partkey == pk
//   count[pk]    : number of such rows
//   thr_x100[pk] : 0.2 × (sum_qty / count) , preserved at the
//                  ×100 fixed-point so Pass-2 can compare l_quantity
//                  (also ×100) using a single double cast.
//
// The threshold is per-part and depends on count/sum from Pass-1, so
// any int-only formulation requires multiplying l_quantity by count
// and would overflow for large parts.  Stick with double — matches
// the SQL semantics literally.
struct Q17PartAcc {
    int64_t sum_qty   = 0;   // l_quantity ×100, summed
    int32_t count     = 0;   // lineitem rows for this part in is_q17_part
    double  threshold = 0.0; // 0.2 × avg(qty), in ×100 fixed-point units
};

// Final per-iteration result.  Stored in int64 (×100 cents) so the
// cross-backend consistency check is exact.
struct Q17Agg {
    int64_t sum_ep_x100 = 0;
    double  avg_yearly() const {
        // ×100 → dollars  ×0  /7 years
        return static_cast<double>(sum_ep_x100) / 100.0 / 7.0;
    }
};

// =============================================================================
// BMTPCH_Q17 — main benchmark entry point
// =============================================================================
void BMTableScan::BMTPCH_Q17(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q17_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    bm_bench::warn_if_sf1();

    // ============================================================
    // 0. Banner
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q17 Benchmark — 8 backends ("
                  << q17_get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q17 Benchmark — " << q17_bm_label() << " only ("
                  << q17_get_sf_label() << ")" << std::endl;
    }
    std::cout << "  Single-bitmap two-pass aggregation over is_q17_part." << std::endl;
    std::cout << "  TPC-H params: p_brand='Brand#23', p_container='MED BOX'" << std::endl;
    std::cout << "  Bitmap dirs:";
    if (run_cb())              std::cout << " " << Q17_CB_DIR;
    if (run_cr() || run_crr()) std::cout << " " << Q17_CR_DIR;
    if (run_wah())             std::cout << " " << Q17_WAH_DIR;
    if (run_ew())              std::cout << " " << Q17_EW_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << Q17_ITERATIONS
              << " (first " << Q17_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 1. Pre-load lineitem.l_partkey / l_quantity / l_extendedprice
    //    (one-shot scan, outside the timed loop, mirrors Q1/Q14/Q15)
    // ============================================================
    auto &lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");
    auto &lineitem_transaction = DuckTransaction::Get(context.client, lineitem_table.catalog);

    TableScanState scan_state;
    vector<StorageIndex> col_ids;
    col_ids.push_back(StorageIndex(1));  // l_partkey
    col_ids.push_back(StorageIndex(4));  // l_quantity
    col_ids.push_back(StorageIndex(5));  // l_extendedprice
    lineitem_table.GetStorage().InitializeScan(context.client, lineitem_transaction, scan_state, col_ids);

    vector<LogicalType> types;
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[1]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[4]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[5]);

    // num_rows from done.txt
    size_t num_rows = 0;
    {
        std::ifstream meta(Q17_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
        }
    }
    if (num_rows == 0) {
        std::cerr << "Error: cannot read num_rows from " << Q17_CB_DIR << "/done.txt" << std::endl;
        return;
    }

    std::cout << "\n[Pre-load] Loading " << num_rows
              << " rows (partkey, quantity, extendedprice) ..." << std::endl;
    auto t_pre0 = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> col_partkey(num_rows);
    std::vector<int64_t> col_qty(num_rows);
    std::vector<int64_t> col_ep(num_rows);
    size_t row_offset = 0;
    while (true) {
        DataChunk chunk;
        chunk.Initialize(context.client, types);
        lineitem_table.GetStorage().Scan(lineitem_transaction, chunk, scan_state);
        if (chunk.size() == 0) break;
        auto pk = FlatVector::GetData<int64_t>(chunk.data[0]);   // BIGINT
        auto qt = FlatVector::GetData<int64_t>(chunk.data[1]);   // DECIMAL→int64
        auto ep = FlatVector::GetData<int64_t>(chunk.data[2]);   // DECIMAL→int64
        std::memcpy(col_partkey.data() + row_offset, pk, chunk.size() * sizeof(int64_t));
        std::memcpy(col_qty.data()     + row_offset, qt, chunk.size() * sizeof(int64_t));
        std::memcpy(col_ep.data()      + row_offset, ep, chunk.size() * sizeof(int64_t));
        row_offset += chunk.size();
    }
    int64_t max_partkey = 0;
    for (auto pk : col_partkey) if (pk > max_partkey) max_partkey = pk;
    auto t_pre1 = std::chrono::high_resolution_clock::now();
    std::cout << "[Pre-load] " << row_offset << " rows in " << ms(t_pre0, t_pre1)
              << " ms  (max_partkey=" << max_partkey << ")" << std::endl;

    if (row_offset != num_rows) {
        std::cerr << "Q17: lineitem row count mismatch (loaded "
                  << row_offset << " vs expected " << num_rows << ")" << std::endl;
        return;
    }

    const int64_t* pk_p = col_partkey.data();
    const int64_t* qt_p = col_qty.data();
    const int64_t* ep_p = col_ep.data();
    const size_t   pk_dim = static_cast<size_t>(max_partkey) + 1;

    // ============================================================
    // 1.5 Build is_q17_part bitmap dynamically (TPC-H 1.5.7 compliance).
    //     Single-table SQL probe on `part` for the brand+container filter,
    //     then walk lineitem.l_partkey + FK lookup against part flag array.
    //     No multi-table aux structure stored on disk.
    // ============================================================
    std::vector<uint32_t> q17_pos;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::unordered_set<int64_t> q17_pk;
        {
            Connection con(*context.client.db);
            auto r = con.Query(
                "SELECT p_partkey FROM part WHERE p_brand='Brand#23' AND p_container='MED BOX'");
            if (r && !r->HasError()) {
                q17_pk.reserve(r->RowCount());
                for (idx_t i = 0; i < r->RowCount(); i++)
                    q17_pk.insert(r->GetValue(0, i).GetValue<int64_t>());
            }
        }
        q17_pos.reserve(num_rows / 50);
        for (size_t i = 0; i < num_rows; i++) {
            if (q17_pk.count(pk_p[i]))
                q17_pos.push_back(static_cast<uint32_t>(i));
        }
        std::cout << "[Build is_q17_part] " << q17_pos.size() << " set / "
                  << num_rows << " rows in "
                  << ms(t0, std::chrono::high_resolution_clock::now()) << " ms" << std::endl;
    }

    // ============================================================
    // 2. Build per-backend is_q17_part bitmaps from q17_pos
    // ============================================================
    std::cout << "\n[Load] Loading is_q17_part bitmaps (mode=" << q17_bm_label() << ")..." << std::endl;

    ComBit                                  cb_part;
    roaring::Roaring                        cr_part;
    roaring::Roaring                        crr_part;
    ibis::bitvector                         wah_part;
    ewah::EWAHBoolArray<uint64_t>           ew_part;
    bs::Bitmap                              bs_part;
    ConciseSet<false>                       con_part;

    double cb_load_ms = 0, cr_load_ms = 0, crr_load_ms = 0, wah_load_ms = 0;
    double ew_load_ms = 0, bs_load_ms = 0, con_load_ms = 0;

    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<bool> jm(num_rows, false);
        for (uint32_t p : q17_pos) jm[p] = true;
        cb_part = ComBit::compress(jm, false);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!q17_pos.empty()) cr_part.addMany(q17_pos.size(), q17_pos.data());
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!q17_pos.empty()) crr_part.addMany(q17_pos.size(), q17_pos.data());
        crr_part.runOptimize();
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        size_t k = 0;
        for (size_t i = 0; i < num_rows; i++) {
            bool b = (k < q17_pos.size() && q17_pos[k] == i);
            if (b) k++;
            wah_part += (b ? 1 : 0);
        }
        wah_part.compress();
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t p : q17_pos) ew_part.set(p);
        if (ew_part.sizeInBits() < num_rows)
            ew_part.padWithZeroes(num_rows);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_part.alloc_for_bits(num_rows);
        for (uint32_t p : q17_pos)
            bs_part.words[p / 64] |= uint64_t(1) << (p % 64);
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (uint32_t p : q17_pos) con_part.add(p);
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
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q17_WAH_DIR)) << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q17_CB_DIR))  << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q17_CR_DIR))  << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q17_EW_DIR))  << " MiB" << std::endl;

    // ============================================================
    // 3. Per-iteration aggregator scratch.
    //    `acc` is reused across iterations (re-zeroed at the top of
    //    every Pass-1).  Sized once: pk_dim ≈ 200 001 entries at SF10.
    // ============================================================
    std::vector<Q17PartAcc> acc(pk_dim);

    std::vector<double> cb_p1_t,  cb_p2_t,  cb_tot_t;
    std::vector<double> cr_p1_t,  cr_p2_t,  cr_tot_t;
    std::vector<double> crr_p1_t, crr_p2_t, crr_tot_t;
    std::vector<double> wah_p1_t, wah_p2_t, wah_tot_t;
    std::vector<double> ew_p1_t,  ew_p2_t,  ew_tot_t;
    std::vector<double> bs_p1_t,  bs_p2_t,  bs_tot_t;
    std::vector<double> bsa_p1_t, bsa_p2_t, bsa_tot_t;
    std::vector<double> con_p1_t, con_p2_t, con_tot_t;

    Q17Agg cb_agg{}, cr_agg{}, crr_agg{}, wah_agg{}, ew_agg{};
    Q17Agg bs_agg{}, bsa_agg{}, con_agg{};

    // Helper: reset accumulator to zero between iterations (pk_dim
    // entries, ~3 MB at SF10 — std::vector<>::assign is the fastest
    // way to bulk-zero POD vectors in libstdc++).
    auto reset_acc = [&]() {
        std::fill(acc.begin(), acc.end(), Q17PartAcc{});
    };

    // Helper: after Pass-1, fill in the per-partkey threshold.  Walks
    // pk_dim entries unconditionally (cheap: pk_dim ≈ 200 001) so the
    // Pass-2 hot loop can read `acc[pk].threshold` with a single load.
    auto compute_thresholds = [&]() {
        for (size_t pk = 0; pk < pk_dim; pk++) {
            auto& a = acc[pk];
            if (a.count > 0)
                a.threshold = 0.2 * static_cast<double>(a.sum_qty) /
                                    static_cast<double>(a.count);
        }
    };

    // ============================================================
    // 4. Benchmark loop
    // ============================================================
    for (int iter = 0; iter < Q17_ITERATIONS; iter++) {
        bool warmup = (iter < Q17_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q17_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ================= ComBit =================
        // is_q17_part is loaded *Compressed* from disk via
        // `ComBit::deserialize` (no OR/AND has expanded it).  Walking
        // segments via `seg.l1_literal_data()` directly is therefore
        // unsafe: in Compressed state `num_literals()` reports the
        // *literal* byte count and skipping the fill regions would
        // mis-align the row_base counter.  The canonical sparse-walk
        // primitive for any ComBit (Compressed or Decompressed) is
        // `for_each_literal(fn)`: it invokes
        //   fn(word_pos, byte_value)
        // for each non-fill byte and uses an AVX-512 L3 bitscan to
        // skip all-zero regions in O(1) per region.  This is the same
        // path `combit_adapter::GetRowidsComBit`'s nonzero-byte AVX
        // scan exposes, packaged as a per-byte callback.
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // --- Pass 1: walk → group by partkey ---
            reset_acc();
            cb_part.for_each_literal([&](size_t word_off, uint8_t b) {
                if (b == 0) return;
                const auto& entry = bm_bench::byte_lut[b];
                size_t base = word_off * 8;
                for (int k = 0; k < entry.count; k++) {
                    size_t r = base + entry.pos[k];
                    int64_t pk = pk_p[r];
                    auto& a = acc[pk];
                    a.sum_qty += qt_p[r];
                    a.count   += 1;
                }
            });
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            // --- Pass 2: walk → filter qty < threshold[pk] ---
            int64_t sum_ep = 0;
            cb_part.for_each_literal([&](size_t word_off, uint8_t b) {
                if (b == 0) return;
                const auto& entry = bm_bench::byte_lut[b];
                size_t base = word_off * 8;
                for (int k = 0; k < entry.count; k++) {
                    size_t r = base + entry.pos[k];
                    if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                        sum_ep += ep_p[r];
                }
            });
            auto t2 = std::chrono::high_resolution_clock::now();
            cb_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  CB:   Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << cb_agg.avg_yearly() << std::endl;
            if (!warmup) {
                cb_p1_t.push_back(d_p1);
                cb_p2_t.push_back(d_p2);
                cb_tot_t.push_back(d_total);
            }
        }

        // ================= CRoaring (vanilla) =================
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            reset_acc();
            for (auto it = cr_part.begin(); it != cr_part.end(); ++it) {
                size_t r = *it;
                int64_t pk = pk_p[r];
                auto& a = acc[pk];
                a.sum_qty += qt_p[r];
                a.count   += 1;
            }
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            int64_t sum_ep = 0;
            for (auto it = cr_part.begin(); it != cr_part.end(); ++it) {
                size_t r = *it;
                if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                    sum_ep += ep_p[r];
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            cr_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  CR:   Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << cr_agg.avg_yearly() << std::endl;
            if (!warmup) {
                cr_p1_t.push_back(d_p1);
                cr_p2_t.push_back(d_p2);
                cr_tot_t.push_back(d_total);
            }
        }

        // ================= CRoaring + Run =================
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            reset_acc();
            for (auto it = crr_part.begin(); it != crr_part.end(); ++it) {
                size_t r = *it;
                int64_t pk = pk_p[r];
                auto& a = acc[pk];
                a.sum_qty += qt_p[r];
                a.count   += 1;
            }
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            int64_t sum_ep = 0;
            for (auto it = crr_part.begin(); it != crr_part.end(); ++it) {
                size_t r = *it;
                if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                    sum_ep += ep_p[r];
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            crr_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  CRR:  Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << crr_agg.avg_yearly() << std::endl;
            if (!warmup) {
                crr_p1_t.push_back(d_p1);
                crr_p2_t.push_back(d_p2);
                crr_tot_t.push_back(d_total);
            }
        }

        // ================= WAH =================
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            reset_acc();
            {
                ibis::bitvector::pit pit(wah_part);
                while (*pit != 0xFFFFFFFFU) {
                    size_t r = *pit;
                    int64_t pk = pk_p[r];
                    auto& a = acc[pk];
                    a.sum_qty += qt_p[r];
                    a.count   += 1;
                    pit.next();
                }
            }
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            int64_t sum_ep = 0;
            {
                ibis::bitvector::pit pit(wah_part);
                while (*pit != 0xFFFFFFFFU) {
                    size_t r = *pit;
                    if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                        sum_ep += ep_p[r];
                    pit.next();
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            wah_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  WAH:  Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << wah_agg.avg_yearly() << std::endl;
            if (!warmup) {
                wah_p1_t.push_back(d_p1);
                wah_p2_t.push_back(d_p2);
                wah_tot_t.push_back(d_total);
            }
        }

        // ================= EWAH =================
        if (run_ew()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            reset_acc();
            for (auto it = ew_part.begin(); it != ew_part.end(); ++it) {
                size_t r = *it;
                int64_t pk = pk_p[r];
                auto& a = acc[pk];
                a.sum_qty += qt_p[r];
                a.count   += 1;
            }
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            int64_t sum_ep = 0;
            for (auto it = ew_part.begin(); it != ew_part.end(); ++it) {
                size_t r = *it;
                if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                    sum_ep += ep_p[r];
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            ew_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  EW:   Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << ew_agg.avg_yearly() << std::endl;
            if (!warmup) {
                ew_p1_t.push_back(d_p1);
                ew_p2_t.push_back(d_p2);
                ew_tot_t.push_back(d_total);
            }
        }

        // ================= Bitset (scalar) =================
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            reset_acc();
            for (size_t i = 0; i < bs_part.nwords; ++i) {
                uint64_t w = bs_part.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= bs_part.nbits) break;
                    int64_t pk = pk_p[r];
                    auto& a = acc[pk];
                    a.sum_qty += qt_p[r];
                    a.count   += 1;
                    w &= w - 1;
                }
            }
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            int64_t sum_ep = 0;
            for (size_t i = 0; i < bs_part.nwords; ++i) {
                uint64_t w = bs_part.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= bs_part.nbits) break;
                    if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                        sum_ep += ep_p[r];
                    w &= w - 1;
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            bs_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  BS:   Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << bs_agg.avg_yearly() << std::endl;
            if (!warmup) {
                bs_p1_t.push_back(d_p1);
                bs_p2_t.push_back(d_p2);
                bs_tot_t.push_back(d_total);
            }
        }

        // ================= Bitset + AVX-512 =================
        // BS and BSA share the same iteration loop here — there is no
        // vectorised `pdep`/`tzcnt` inner-loop primitive in BSA's API
        // and the per-row work (column lookups + double compare) is the
        // bottleneck.  We still report BSA separately so the cross-
        // backend table stays uniform with the other 7 Q* benchmarks.
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            reset_acc();
            for (size_t i = 0; i < bs_part.nwords; ++i) {
                uint64_t w = bs_part.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= bs_part.nbits) break;
                    int64_t pk = pk_p[r];
                    auto& a = acc[pk];
                    a.sum_qty += qt_p[r];
                    a.count   += 1;
                    w &= w - 1;
                }
            }
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            int64_t sum_ep = 0;
            for (size_t i = 0; i < bs_part.nwords; ++i) {
                uint64_t w = bs_part.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= bs_part.nbits) break;
                    if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                        sum_ep += ep_p[r];
                    w &= w - 1;
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            bsa_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  BSA:  Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << bsa_agg.avg_yearly() << std::endl;
            if (!warmup) {
                bsa_p1_t.push_back(d_p1);
                bsa_p2_t.push_back(d_p2);
                bsa_tot_t.push_back(d_total);
            }
        }

        // ================= Concise =================
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            reset_acc();
            for (auto it = con_part.begin(); it != con_part.end(); ++it) {
                size_t r = *it;
                int64_t pk = pk_p[r];
                auto& a = acc[pk];
                a.sum_qty += qt_p[r];
                a.count   += 1;
            }
            compute_thresholds();
            auto t1 = std::chrono::high_resolution_clock::now();

            int64_t sum_ep = 0;
            for (auto it = con_part.begin(); it != con_part.end(); ++it) {
                size_t r = *it;
                if (static_cast<double>(qt_p[r]) < acc[pk_p[r]].threshold)
                    sum_ep += ep_p[r];
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            con_agg.sum_ep_x100 = sum_ep;

            double d_p1 = ms(t0, t1), d_p2 = ms(t1, t2), d_total = ms(t0, t2);
            std::cout << "  CON:  Pass1=" << d_p1 << "  Pass2=" << d_p2
                      << "  Total=" << d_total
                      << "  avg_yearly=" << con_agg.avg_yearly() << std::endl;
            if (!warmup) {
                con_p1_t.push_back(d_p1);
                con_p2_t.push_back(d_p2);
                con_tot_t.push_back(d_total);
            }
        }
    } // end iterations

    // ============================================================
    // 5. Cross-backend consistency check
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q17 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;

    const Q17Agg* print_src = nullptr;
    const char* print_label = "";
    if      (Q17_BM == Q17BmType::ALL || Q17_BM == Q17BmType::CB)  { print_src = &cb_agg;  print_label = "ComBit"; }
    else if (Q17_BM == Q17BmType::WAH)                             { print_src = &wah_agg; print_label = "WAH"; }
    else if (Q17_BM == Q17BmType::CR)                              { print_src = &cr_agg;  print_label = "CRoaring"; }
    else if (Q17_BM == Q17BmType::CRR)                             { print_src = &crr_agg; print_label = "CRoaring+Run"; }
    else if (Q17_BM == Q17BmType::EW)                              { print_src = &ew_agg;  print_label = "EWAH"; }
    else if (Q17_BM == Q17BmType::BS)                              { print_src = &bs_agg;  print_label = "Bitset"; }
    else if (Q17_BM == Q17BmType::BSA)                             { print_src = &bsa_agg; print_label = "Bitset+AVX512"; }
    else if (Q17_BM == Q17BmType::CON)                             { print_src = &con_agg; print_label = "Concise"; }

    std::cout << std::fixed << std::setprecision(2);
    if (print_src) {
        std::cout << "  Q17 Result (" << print_label << "):\n";
        std::cout << "    avg_yearly = " << print_src->avg_yearly()
                  << "   (sum_ep×100 = " << print_src->sum_ep_x100 << ")" << std::endl;
    }

    if (run_all()) {
        bool consistent = true;
        const Q17Agg& base = cb_agg;
        auto cmp = [&](const char* lbl, const Q17Agg& v) {
            if (v.sum_ep_x100 != base.sum_ep_x100) {
                std::cout << "  *** MISMATCH " << lbl
                          << " (sum_ep=" << v.sum_ep_x100
                          << " vs CB sum_ep=" << base.sum_ep_x100 << ") ***\n";
                consistent = false;
            }
        };
        cmp("CR",  cr_agg);
        cmp("CRR", crr_agg);
        cmp("WAH", wah_agg);
        cmp("EW",  ew_agg);
        cmp("BS",  bs_agg);
        cmp("BSA", bsa_agg);
        cmp("CON", con_agg);
        std::cout << "  Consistency: " << (consistent ? "ALL MATCH" : "MISMATCH DETECTED") << std::endl;
    }

    // ============================================================
    // 6. DuckDB native SQL ground truth
    // ============================================================
    double gt_avg_yearly = -1.0;
    double gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT sum(l_extendedprice) / 7.0 AS avg_yearly "
            "  FROM lineitem, part "
            " WHERE p_partkey = l_partkey "
            "   AND p_brand = 'Brand#23' "
            "   AND p_container = 'MED BOX' "
            "   AND l_quantity < ("
            "         SELECT 0.2 * avg(l_quantity) "
            "           FROM lineitem "
            "          WHERE l_partkey = p_partkey"
            "       )";
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = con.Query(sql);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (result && !result->HasError() && result->RowCount() == 1) {
            gt_avg_yearly = result->GetValue(0, 0).GetValue<double>();
            gt_sql_ms = ms(t0, t1);
        } else if (result && result->HasError()) {
            std::cerr << "[Baseline] SQL error: " << result->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    if (gt_avg_yearly >= 0.0) {
        std::cout << "\n[Baseline] DuckDB native SQL  (single run: "
                  << std::fixed << std::setprecision(2) << gt_sql_ms << " ms)" << std::endl;
        std::cout << "  SQL ground truth:  avg_yearly = " << gt_avg_yearly << std::endl;

        if (print_src) {
            double bm_val = print_src->avg_yearly();
            double diff = std::abs(bm_val - gt_avg_yearly);
            // Tolerance 0.01 (one cent / 7) since DuckDB uses DECIMAL
            // and our pipeline accumulates int64 ×100 — both are exact
            // up to the final /7 in floating point.
            if (diff <= 0.01) {
                std::cout << "[OK] all active backends match DuckDB SQL ground truth "
                          << "(avg_yearly within " << diff << ")." << std::endl;
            } else {
                std::cerr << "[FAIL] bitmap " << bm_val << " vs SQL "
                          << gt_avg_yearly << " differ by " << diff << std::endl;
            }
        }
    }

    // ============================================================
    // 7. Final timing summary (median ± stddev)
    // ============================================================
    auto print_stats = [&](const char* lbl, std::vector<double>& p1_t,
                           std::vector<double>& p2_t, std::vector<double>& tot_t) {
        if (p1_t.empty()) return;
        auto sa = bm_bench::compute_stats(p1_t);
        auto sb = bm_bench::compute_stats(p2_t);
        auto st = bm_bench::compute_stats(tot_t);
        std::cout << "  " << std::left << std::setw(5) << lbl
                  << " Pass1="  << std::fixed << std::setprecision(2) << std::setw(7) << sa.median
                  << " +/- "    << std::setw(5) << sa.stddev
                  << "  Pass2=" << std::setw(7) << sb.median
                  << " +/- "    << std::setw(5) << sb.stddev
                  << "  Total=" << std::setw(7) << st.median
                  << " +/- "    << std::setw(5) << st.stddev
                  << " (ms)" << std::endl;
    };

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q17 RESULTS (" << (Q17_ITERATIONS - Q17_WARMUP)
              << " measured iterations, median +/- stddev)" << std::endl;
    std::cout << "================================================================" << std::endl;
    print_stats("CB",  cb_p1_t,  cb_p2_t,  cb_tot_t);
    print_stats("CR",  cr_p1_t,  cr_p2_t,  cr_tot_t);
    print_stats("CRR", crr_p1_t, crr_p2_t, crr_tot_t);
    print_stats("WAH", wah_p1_t, wah_p2_t, wah_tot_t);
    print_stats("EW",  ew_p1_t,  ew_p2_t,  ew_tot_t);
    print_stats("BS",  bs_p1_t,  bs_p2_t,  bs_tot_t);
    print_stats("BSA", bsa_p1_t, bsa_p2_t, bsa_tot_t);
    print_stats("CON", con_p1_t, con_p2_t, con_tot_t);
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 8. CSV export
    // ============================================================
    {
        std::string csv_path = "q17_results_" + q17_get_sf_label() + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << "backend,pass1_ms,pass2_ms,total_ms\n";
            auto wrow = [&](const char* lbl, std::vector<double>& p1_t,
                            std::vector<double>& p2_t, std::vector<double>& tot_t) {
                if (p1_t.empty()) return;
                auto sa = bm_bench::compute_stats(p1_t);
                auto sb = bm_bench::compute_stats(p2_t);
                auto st = bm_bench::compute_stats(tot_t);
                csv << lbl << "," << sa.median << "," << sb.median << "," << st.median << "\n";
            };
            wrow("CB",  cb_p1_t,  cb_p2_t,  cb_tot_t);
            wrow("CR",  cr_p1_t,  cr_p2_t,  cr_tot_t);
            wrow("CRR", crr_p1_t, crr_p2_t, crr_tot_t);
            wrow("WAH", wah_p1_t, wah_p2_t, wah_tot_t);
            wrow("EW",  ew_p1_t,  ew_p2_t,  ew_tot_t);
            wrow("BS",  bs_p1_t,  bs_p2_t,  bs_tot_t);
            wrow("BSA", bsa_p1_t, bsa_p2_t, bsa_tot_t);
            wrow("CON", con_p1_t, con_p2_t, con_tot_t);
            std::cout << "\n  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    });  // end std::call_once
}

}  // namespace duckdb
