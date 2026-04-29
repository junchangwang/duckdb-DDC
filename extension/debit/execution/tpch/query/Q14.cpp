// =============================================================================
// TPC-H Q14 — Promotion Effect Query (spec v3.0.1 §2.4.14)
//
//   SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%'
//                              THEN l_extendedprice * (1 - l_discount)
//                              ELSE 0 END)
//                / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
//   FROM   lineitem, part
//   WHERE  l_partkey  = p_partkey
//     AND  l_shipdate >= DATE '[DATE]'
//     AND  l_shipdate <  DATE '[DATE]' + INTERVAL '1' MONTH;
//
// Qualification DATE = '1995-09-01' (spec §2.4.14.4).
//
// Bitmap pipeline (all bitmaps aligned to lineitem.rowid):
//
//   T0: ship_filter = OR_many(shipdate days [1339..1369))    (30 day bitmaps)
//   T1: promo_set   = ship_filter AND is_promo
//   T2: iterate ship_filter rows → total_rev += price * (100 - disc)
//       iterate promo_set    rows → promo_rev += price * (100 - disc)
//
// The is_promo bitmap is pre-joined at export time:
//   tpch_q14_<fmt>/is_promo/0.bm   :  l_partkey -> p.p_type LIKE 'PROMO%'
// The shipdate bitmaps are reused (via symlink) from Q6's exports:
//   tpch_q14_<fmt>/shipdate/{day}.bm  -> tpch_q6_<fmt>/shipdate/{day}.bm
//
// Output: single double `promo_revenue` (≈ 16.65 for SF10).
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

// --- Q14 bitmap directories ---
static const std::string Q14_SF      = bm_bench::sf_suffix();
static const std::string Q14_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q14" + Q14_SF + "_combit");
static const std::string Q14_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q14" + Q14_SF + "_wah");
static const std::string Q14_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q14" + Q14_SF + "_croaring");
static const std::string Q14_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q14" + Q14_SF + "_ewah");

// --- Backend selection (DEBIT_BM=all|wah|cb|cr|crr|ew|bs|bsa|con) ---
using Q14BmType = bm_bench::Backend;
static const Q14BmType Q14_BM = bm_bench::parse_backend("Q14_BM");

static bool run_all() { return Q14_BM == Q14BmType::ALL; }
static bool run_wah() { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::WAH; }
static bool run_cb()  { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::CB;  }
static bool run_cr()  { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::CR;  }
static bool run_crr() { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::CRR; }
static bool run_ew()  { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::EW;  }
static bool run_bs()  { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::BS;  }
static bool run_bsa() { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::BSA; }
static bool run_con() { return Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::CON; }

static const char* q14_bm_label()     { return bm_bench::backend_label(Q14_BM); }
static std::string q14_get_sf_label() { return bm_bench::sf_label(); }

// --- Q14 predicate (TPC-H spec §2.4.14, qualification DATE = 1995-09-01) ---
// Day range [1995-09-01, 1995-10-01) = days [1339, 1369) since 1992-01-01
// (1992 leap year:  1992-01-01 = day 0,  1995-01-01 = day 1096,
//   1995-09-01 = 1096 + 31+28+31+30+31+30+31+31 = 1339,
//   1995-10-01 = 1339 + 30 = 1369).
static const int Q14_DATE_START = 1339;  // inclusive
static const int Q14_DATE_END   = 1368;  // inclusive (= 1369-1, exclusive-end form)

// --- Iteration counts (DEBIT_ITER / DEBIT_WARMUP) ---
static const int Q14_ITERATIONS = bm_bench::iter_count(10);
static const int Q14_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q14_once_flag;

// Statistics helper.
struct Q14Stats {
    double median = 0, stddev = 0, min_val = 0, max_val = 0;
};
static Q14Stats q14_compute_stats(std::vector<double>& v) {
    Q14Stats s{};
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

// Bitmap loaders (same set as Q1/Q12).
static ComBit q14_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q14_load_cr(const std::string& p) {
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
static ibis::bitvector q14_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q14_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// Byte-LUT for MSB-first bit extraction (same convention as Q1.cpp /
// combit_adapter.cpp).  256 × 9 = 2304 bytes, permanently in L1d.
struct Q14ByteEntry { uint8_t count; uint8_t pos[8]; };
static Q14ByteEntry q14_byte_lut[256];
static bool q14_byte_lut_init = []() {
    for (int v = 0; v < 256; v++) {
        uint8_t c = 0;
        for (int b = 7; b >= 0; b--)
            if (v & (1 << b))
                q14_byte_lut[v].pos[c++] = 7 - b;
        q14_byte_lut[v].count = c;
    }
    return true;
}();

// Per-row aggregator: revenue contribution = price * (100 - disc),
// kept as int64 (price stored ×100, discount stored ×100 → product ×10000).
// promo_revenue is computed as 100.0 * promo_rev / total_rev, so the
// shared ×10000 factor cancels and never needs to be divided out.
struct Q14Agg {
    int64_t total_rev = 0;
    int64_t promo_rev = 0;
    double  promo_pct() const {
        return (total_rev > 0)
                 ? 100.0 * static_cast<double>(promo_rev) /
                            static_cast<double>(total_rev)
                 : 0.0;
    }
};

#define Q14_REV_CONTRIB(pp, dp, r) \
    ((pp)[(r)] * (100 - (dp)[(r)]))

// =============================================================================
// BMTPCH_Q14 — main benchmark entry point
// =============================================================================
void BMTableScan::BMTPCH_Q14(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q14_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    bm_bench::warn_if_sf1();

    // ============================================================
    // 0. Banner
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q14 Benchmark — 8 backends ("
                  << q14_get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q14 Benchmark — " << q14_bm_label() << " only ("
                  << q14_get_sf_label() << ")" << std::endl;
    }
    std::cout << "  OR shipdate days " << Q14_DATE_START << ".." << Q14_DATE_END
              << " (" << (Q14_DATE_END - Q14_DATE_START + 1) << " bitmaps), "
              << "AND is_promo, then per-row revenue aggregation"
              << std::endl;
    std::cout << "  TPC-H params: l_shipdate [1995-09-01, 1995-10-01), "
                 "p_type LIKE 'PROMO%'"
              << std::endl;
    std::cout << "  Bitmap dirs:";
    if (run_cb())              std::cout << " " << Q14_CB_DIR;
    if (run_cr() || run_crr()) std::cout << " " << Q14_CR_DIR;
    if (run_wah())             std::cout << " " << Q14_WAH_DIR;
    if (run_ew())              std::cout << " " << Q14_EW_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << Q14_ITERATIONS
              << " (first " << Q14_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 1. Pre-load lineitem.l_extendedprice / l_discount columns
    //    (one-shot scan, outside the timed loop, mirrors Q1's design)
    // ============================================================
    auto &lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");
    auto &lineitem_transaction = DuckTransaction::Get(context.client, lineitem_table.catalog);

    TableScanState scan_state;
    vector<StorageIndex> col_ids;
    col_ids.push_back(StorageIndex(5));  // l_extendedprice
    col_ids.push_back(StorageIndex(6));  // l_discount
    lineitem_table.GetStorage().InitializeScan(context.client, lineitem_transaction, scan_state, col_ids);

    vector<LogicalType> types;
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[5]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[6]);

    // num_rows from done.txt
    size_t num_rows = 0;
    {
        std::ifstream meta(Q14_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
        }
    }
    if (num_rows == 0) {
        std::cerr << "Error: cannot read num_rows from " << Q14_CB_DIR << "/done.txt" << std::endl;
        return;
    }

    std::cout << "\n[Pre-load] Loading " << num_rows << " rows (price, discount) ..." << std::endl;
    auto t_pre0 = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> col_price(num_rows);
    std::vector<int64_t> col_disc(num_rows);
    size_t row_offset = 0;
    while (true) {
        DataChunk chunk;
        chunk.Initialize(context.client, types);
        lineitem_table.GetStorage().Scan(lineitem_transaction, chunk, scan_state);
        if (chunk.size() == 0) break;
        auto p = FlatVector::GetData<int64_t>(chunk.data[0]);
        auto d = FlatVector::GetData<int64_t>(chunk.data[1]);
        std::memcpy(col_price.data() + row_offset, p, chunk.size() * sizeof(int64_t));
        std::memcpy(col_disc.data()  + row_offset, d, chunk.size() * sizeof(int64_t));
        row_offset += chunk.size();
    }
    auto t_pre1 = std::chrono::high_resolution_clock::now();
    std::cout << "[Pre-load] Done in " << ms(t_pre0, t_pre1) << " ms" << std::endl;

    const int64_t* pp = col_price.data();
    const int64_t* dp = col_disc.data();

    // ============================================================
    // 2. Load bitmaps
    // ============================================================
    std::cout << "\n[Load] Loading bitmaps (mode=" << q14_bm_label() << ")..." << std::endl;
    const int n_dates = Q14_DATE_END - Q14_DATE_START + 1;

    // --- ComBit ---
    std::vector<ComBit> cb_date;
    std::vector<const ComBit*> cb_date_ptrs;
    ComBit cb_promo;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_date.resize(Q14_DATE_END + 1);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            cb_date[d] = q14_load_cb(Q14_CB_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cb_promo = q14_load_cb(Q14_CB_DIR + "/is_promo/0.bm");
        cb_date_ptrs.reserve(n_dates);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            cb_date_ptrs.push_back(&cb_date[d]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring (vanilla) ---
    std::vector<roaring::Roaring> cr_date;
    roaring::Roaring cr_promo;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_date.resize(Q14_DATE_END + 1);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            cr_date[d] = q14_load_cr(Q14_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cr_promo = q14_load_cr(Q14_CR_DIR + "/is_promo/0.bm");
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring + Run (loads fresh + runOptimize) ---
    std::vector<roaring::Roaring> crr_date;
    std::vector<const roaring::Roaring*> crr_date_ptrs;
    roaring::Roaring crr_promo;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_date.resize(Q14_DATE_END + 1);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++) {
            crr_date[d] = q14_load_cr(Q14_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
            crr_date[d].runOptimize();
        }
        crr_promo = q14_load_cr(Q14_CR_DIR + "/is_promo/0.bm");
        crr_promo.runOptimize();
        crr_date_ptrs.reserve(n_dates);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            crr_date_ptrs.push_back(&crr_date[d]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- WAH ---
    std::vector<ibis::bitvector> wah_date;
    ibis::bitvector wah_promo;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wah_date.resize(Q14_DATE_END + 1);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            wah_date[d] = q14_load_wah(Q14_WAH_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        wah_promo = q14_load_wah(Q14_WAH_DIR + "/is_promo/0.bm");
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- EWAH ---
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_date;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_date_ptrs;
    ewah::EWAHBoolArray<uint64_t> ew_promo;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ew_date.resize(Q14_DATE_END + 1);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            ew_date[d] = q14_load_ew(Q14_EW_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        ew_promo = q14_load_ew(Q14_EW_DIR + "/is_promo/0.bm");
        ew_date_ptrs.reserve(n_dates);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            ew_date_ptrs.push_back(&ew_date[d]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Bitset (BS / BSA share) ---
    std::vector<bs::Bitmap> bs_date;
    bs::Bitmap bs_promo;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_date.resize(Q14_DATE_END + 1);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            bs_date[d] = bm_bench::load_bitmap_from_croaring(
                Q14_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        bs_promo = bm_bench::load_bitmap_from_croaring(Q14_CR_DIR + "/is_promo/0.bm");
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Concise ---
    std::vector<ConciseSet<false>> con_date;
    std::vector<const ConciseSet<false>*> con_date_ptrs;
    ConciseSet<false> con_promo;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_date.resize(Q14_DATE_END + 1);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
            con_date[d] = bm_bench::load_concise_from_croaring(
                Q14_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        con_promo = bm_bench::load_concise_from_croaring(Q14_CR_DIR + "/is_promo/0.bm");
        con_date_ptrs.reserve(n_dates);
        for (int d = Q14_DATE_START; d <= Q14_DATE_END; d++)
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
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q14_WAH_DIR)) << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q14_CB_DIR))  << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q14_CR_DIR))  << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q14_EW_DIR))  << " MiB" << std::endl;

    // ============================================================
    // 3. Benchmark loop — phases:
    //    OR_date  : OR_many over 30 shipdate days
    //    AND_pro  : AND ship_filter with is_promo (= promo_set)
    //    Agg      : per-row aggregation over ship_filter and promo_set
    // ============================================================
    std::vector<double> cb_or_t,  cb_and_t,  cb_agg_t,  cb_tot_t;
    std::vector<double> cr_or_t,  cr_and_t,  cr_agg_t,  cr_tot_t;
    std::vector<double> crr_or_t, crr_and_t, crr_agg_t, crr_tot_t;
    std::vector<double> wah_or_t, wah_and_t, wah_agg_t, wah_tot_t;
    std::vector<double> ew_or_t,  ew_and_t,  ew_agg_t,  ew_tot_t;
    std::vector<double> bs_or_t,  bs_and_t,  bs_agg_t,  bs_tot_t;
    std::vector<double> bsa_or_t, bsa_and_t, bsa_agg_t, bsa_tot_t;
    std::vector<double> con_or_t, con_and_t, con_agg_t, con_tot_t;

    Q14Agg cb_agg{}, cr_agg{}, crr_agg{}, wah_agg{}, ew_agg{};
    Q14Agg bs_agg{}, bsa_agg{}, con_agg{};

    for (int iter = 0; iter < Q14_ITERATIONS; iter++) {
        bool warmup = (iter < Q14_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q14_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ================= ComBit =================
        // OR_many → copy → in-place &= → for_each segment + byte_lut
        //   decode + Q14_REV_CONTRIB.  This is the SAME pipeline shape
        //   used by Q1/Q12: `OR_many` returns a Decompressed ComBit;
        //   `ComBitBtv::operator&=` asserts state==Decompressed and runs
        //   the canonical 64-byte AVX-512 fast path; the subsequent
        //   per-row scan walks `seg.l1_literal_data()` directly without
        //   any decode.  We make a *copy* of `cb_ship` so the unmodified
        //   `cb_ship` survives for the total_rev iteration below — this
        //   avoids re-running OR_many or recomputing the union mask.
        //   (The free `operator&` returning a fresh ComBit is NOT used:
        //   `ComBitBtv::operator&` is documented "FIXME: only support
        //   compressed versions so far" and asserts state==Compressed,
        //   which is incompatible with the Decompressed result of
        //   `OR_many` and would silently misroute in release builds.)
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ComBit cb_ship = ComBit::OR_many(cb_date_ptrs.size(), cb_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ComBit cb_pro = cb_ship;
            cb_pro &= cb_promo;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            // total_rev: iterate ship_filter
            {
                size_t row_base = 0;
                for (size_t s = 0; s < cb_ship.num_segments(); s++) {
                    const auto& seg = cb_ship.segment(s);
                    const uint8_t* data = seg.l1_literal_data();
                    size_t n = seg.num_literals();
                    for (size_t bi = 0; bi < n; bi++) {
                        uint8_t b = data[bi];
                        if (b == 0) { row_base += 8; continue; }
                        const auto& entry = q14_byte_lut[b];
                        for (int k = 0; k < entry.count; k++)
                            agg.total_rev += Q14_REV_CONTRIB(pp, dp, row_base + entry.pos[k]);
                        row_base += 8;
                    }
                }
            }
            // promo_rev: iterate promo_set
            {
                size_t row_base = 0;
                for (size_t s = 0; s < cb_pro.num_segments(); s++) {
                    const auto& seg = cb_pro.segment(s);
                    const uint8_t* data = seg.l1_literal_data();
                    size_t n = seg.num_literals();
                    for (size_t bi = 0; bi < n; bi++) {
                        uint8_t b = data[bi];
                        if (b == 0) { row_base += 8; continue; }
                        const auto& entry = q14_byte_lut[b];
                        for (int k = 0; k < entry.count; k++)
                            agg.promo_rev += Q14_REV_CONTRIB(pp, dp, row_base + entry.pos[k]);
                        row_base += 8;
                    }
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            cb_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CB:   OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                cb_or_t.push_back(d_or); cb_and_t.push_back(d_and);
                cb_agg_t.push_back(d_agg); cb_tot_t.push_back(d_total);
            }
        }

        // ================= CRoaring (vanilla pairwise) =================
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring cr_ship = cr_date[Q14_DATE_START];
            for (int d = Q14_DATE_START + 1; d <= Q14_DATE_END; d++)
                cr_ship |= cr_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();

            roaring::Roaring cr_pro = cr_ship & cr_promo;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            for (auto it = cr_ship.begin(); it != cr_ship.end(); ++it)
                agg.total_rev += Q14_REV_CONTRIB(pp, dp, *it);
            for (auto it = cr_pro.begin();  it != cr_pro.end();  ++it)
                agg.promo_rev += Q14_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            cr_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CR:   OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                cr_or_t.push_back(d_or); cr_and_t.push_back(d_and);
                cr_agg_t.push_back(d_agg); cr_tot_t.push_back(d_total);
            }
        }

        // ================= CRoaring + Run (fastunion) =================
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring crr_ship = roaring::Roaring::fastunion(
                crr_date_ptrs.size(), crr_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            roaring::Roaring crr_pro = crr_ship & crr_promo;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            for (auto it = crr_ship.begin(); it != crr_ship.end(); ++it)
                agg.total_rev += Q14_REV_CONTRIB(pp, dp, *it);
            for (auto it = crr_pro.begin();  it != crr_pro.end();  ++it)
                agg.promo_rev += Q14_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            crr_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CRR:  OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                crr_or_t.push_back(d_or); crr_and_t.push_back(d_and);
                crr_agg_t.push_back(d_agg); crr_tot_t.push_back(d_total);
            }
        }

        // ================= WAH =================
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ibis::bitvector wah_ship = wah_date[Q14_DATE_START];
            wah_ship.decompress();
            for (int d = Q14_DATE_START + 1; d <= Q14_DATE_END; d++)
                wah_ship |= wah_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();

            ibis::bitvector wah_pro;
            wah_pro.copy(wah_ship);
            wah_pro &= wah_promo;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            {
                ibis::bitvector::pit pit(wah_ship);
                while (*pit != 0xFFFFFFFFU) {
                    agg.total_rev += Q14_REV_CONTRIB(pp, dp, *pit);
                    pit.next();
                }
            }
            {
                ibis::bitvector::pit pit(wah_pro);
                while (*pit != 0xFFFFFFFFU) {
                    agg.promo_rev += Q14_REV_CONTRIB(pp, dp, *pit);
                    pit.next();
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            wah_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  WAH:  OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                wah_or_t.push_back(d_or); wah_and_t.push_back(d_and);
                wah_agg_t.push_back(d_agg); wah_tot_t.push_back(d_total);
            }
        }

        // ================= EWAH (fast_logicalor) =================
        if (run_ew()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ewah::EWAHBoolArray<uint64_t> ew_ship = ewah::fast_logicalor(
                ew_date_ptrs.size(), ew_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ewah::EWAHBoolArray<uint64_t> ew_pro;
            ew_ship.logicaland(ew_promo, ew_pro);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            for (auto it = ew_ship.begin(); it != ew_ship.end(); ++it)
                agg.total_rev += Q14_REV_CONTRIB(pp, dp, *it);
            for (auto it = ew_pro.begin();  it != ew_pro.end();  ++it)
                agg.promo_rev += Q14_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            ew_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  EW:   OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                ew_or_t.push_back(d_or); ew_and_t.push_back(d_and);
                ew_agg_t.push_back(d_agg); ew_tot_t.push_back(d_total);
            }
        }

        // ================= Bitset (scalar) =================
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bs_ship = bs_date[Q14_DATE_START].clone();
            for (int d = Q14_DATE_START + 1; d <= Q14_DATE_END; d++)
                bs::or_inplace(bs_ship, bs_date[d], false);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bs_pro = bs_ship.clone();
            bs::and_inplace(bs_pro, bs_promo, false);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            // total_rev: scan ship_filter words + ctzll
            for (size_t i = 0; i < bs_ship.nwords; ++i) {
                uint64_t w = bs_ship.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t row = base + __builtin_ctzll(w);
                    if (row >= bs_ship.nbits) break;
                    agg.total_rev += Q14_REV_CONTRIB(pp, dp, row);
                    w &= w - 1;
                }
            }
            // promo_rev: scan promo_set words + ctzll
            for (size_t i = 0; i < bs_pro.nwords; ++i) {
                uint64_t w = bs_pro.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t row = base + __builtin_ctzll(w);
                    if (row >= bs_pro.nbits) break;
                    agg.promo_rev += Q14_REV_CONTRIB(pp, dp, row);
                    w &= w - 1;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            bs_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  BS:   OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                bs_or_t.push_back(d_or); bs_and_t.push_back(d_and);
                bs_agg_t.push_back(d_agg); bs_tot_t.push_back(d_total);
            }
        }

        // ================= Bitset + AVX-512 =================
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bsa_ship = bs_date[Q14_DATE_START].clone();
            for (int d = Q14_DATE_START + 1; d <= Q14_DATE_END; d++)
                bs::or_inplace(bsa_ship, bs_date[d], true);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bsa_pro = bsa_ship.clone();
            bs::and_inplace(bsa_pro, bs_promo, true);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            for (size_t i = 0; i < bsa_ship.nwords; ++i) {
                uint64_t w = bsa_ship.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t row = base + __builtin_ctzll(w);
                    if (row >= bsa_ship.nbits) break;
                    agg.total_rev += Q14_REV_CONTRIB(pp, dp, row);
                    w &= w - 1;
                }
            }
            for (size_t i = 0; i < bsa_pro.nwords; ++i) {
                uint64_t w = bsa_pro.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t row = base + __builtin_ctzll(w);
                    if (row >= bsa_pro.nbits) break;
                    agg.promo_rev += Q14_REV_CONTRIB(pp, dp, row);
                    w &= w - 1;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            bsa_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  BSA:  OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                bsa_or_t.push_back(d_or); bsa_and_t.push_back(d_and);
                bsa_agg_t.push_back(d_agg); bsa_tot_t.push_back(d_total);
            }
        }

        // ================= Concise (fast_logicalor) =================
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ConciseSet<false> con_ship = ConciseSet<false>::fast_logicalor(
                con_date_ptrs.size(), con_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_pro = con_ship.logicaland(con_promo);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q14Agg agg{};
            for (auto it = con_ship.begin(); it != con_ship.end(); ++it)
                agg.total_rev += Q14_REV_CONTRIB(pp, dp, *it);
            for (auto it = con_pro.begin();  it != con_pro.end();  ++it)
                agg.promo_rev += Q14_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            con_agg = agg;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CON:  OR=" << d_or << "  AND_pro=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  promo=" << agg.promo_pct() << "%" << std::endl;
            if (!warmup) {
                con_or_t.push_back(d_or); con_and_t.push_back(d_and);
                con_agg_t.push_back(d_agg); con_tot_t.push_back(d_total);
            }
        }
    } // end iterations

    // ============================================================
    // 4. Correctness validation
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q14 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Pick a canonical print source (prefer ComBit when ALL).
    const Q14Agg* print_src = nullptr;
    const char* print_src_label = "";
    if      (Q14_BM == Q14BmType::ALL || Q14_BM == Q14BmType::CB)  { print_src = &cb_agg;  print_src_label = "ComBit"; }
    else if (Q14_BM == Q14BmType::WAH)                             { print_src = &wah_agg; print_src_label = "WAH"; }
    else if (Q14_BM == Q14BmType::CR)                              { print_src = &cr_agg;  print_src_label = "CRoaring"; }
    else if (Q14_BM == Q14BmType::CRR)                             { print_src = &crr_agg; print_src_label = "CRoaring+Run"; }
    else if (Q14_BM == Q14BmType::EW)                              { print_src = &ew_agg;  print_src_label = "EWAH"; }
    else if (Q14_BM == Q14BmType::BS)                              { print_src = &bs_agg;  print_src_label = "Bitset"; }
    else if (Q14_BM == Q14BmType::BSA)                             { print_src = &bsa_agg; print_src_label = "Bitset+AVX512"; }
    else if (Q14_BM == Q14BmType::CON)                             { print_src = &con_agg; print_src_label = "Concise"; }

    std::cout << std::fixed << std::setprecision(4);
    if (print_src) {
        std::cout << "  Q14 Result (" << print_src_label << "):\n";
        std::cout << "    promo_revenue = " << print_src->promo_pct() << "  %"
                  << "  (promo_rev=" << print_src->promo_rev
                  <<   " total_rev=" << print_src->total_rev << ")" << std::endl;
    }

    if (run_all()) {
        bool consistent = true;
        const Q14Agg& base = cb_agg;
        auto cmp = [&](const char* lbl, const Q14Agg& v) {
            if (v.total_rev != base.total_rev || v.promo_rev != base.promo_rev) {
                std::cout << "  *** MISMATCH " << lbl
                          << " (total=" << v.total_rev << " promo=" << v.promo_rev
                          << " vs CB total=" << base.total_rev
                          <<           " promo=" << base.promo_rev << ") ***\n";
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
    // 5. DuckDB native SQL ground-truth
    // ============================================================
    double gt_promo_revenue = -1.0;
    double gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' "
            "                          THEN l_extendedprice * (1 - l_discount) "
            "                          ELSE 0 END) "
            "             / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue "
            "FROM lineitem, part "
            "WHERE l_partkey = p_partkey "
            "  AND l_shipdate >= DATE '1995-09-01' "
            "  AND l_shipdate <  DATE '1995-10-01'";
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = con.Query(sql);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (result && !result->HasError() && result->RowCount() == 1) {
            gt_promo_revenue = result->GetValue(0, 0).GetValue<double>();
            gt_sql_ms = ms(t0, t1);
        } else if (result && result->HasError()) {
            std::cerr << "[Baseline] SQL error: " << result->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    if (gt_promo_revenue >= 0.0) {
        std::cout << "\n[Baseline] DuckDB native SQL  (single run: "
                  << std::fixed << std::setprecision(2) << gt_sql_ms << " ms)" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  SQL ground truth:  promo_revenue = " << gt_promo_revenue << " %" << std::endl;

        // Tolerance: SQL pipeline does DECIMAL(15,2) × DECIMAL(15,2) →
        // DECIMAL(30,4) and final division in DECIMAL/double; our integer
        // accumulator computes the same ratio exactly because the ×10000
        // factor cancels in 100.0 × promo_rev / total_rev.  But because
        // the SQL pipeline may go through a different rounding step
        // (CAST to DOUBLE), allow 1e-6 absolute tolerance on the final
        // percentage value.
        const double tol = 1e-6;
        auto check = [&](const char* label, bool active, const Q14Agg& got) {
            if (!active) return;
            double our = got.promo_pct();
            if (std::fabs(our - gt_promo_revenue) > tol) {
                std::ostringstream oss;
                oss << "[FAIL] Q14 " << label
                    << " promo_revenue " << std::setprecision(8) << our
                    << " differs from SQL " << gt_promo_revenue
                    << " (tolerance " << tol << ")";
                throw std::runtime_error(oss.str());
            }
        };
        check("WAH",           run_wah(), wah_agg);
        check("ComBit",        run_cb(),  cb_agg);
        check("CRoaring",      run_cr(),  cr_agg);
        check("CRoaring+Run",  run_crr(), crr_agg);
        check("EWAH",          run_ew(),  ew_agg);
        check("Bitset",        run_bs(),  bs_agg);
        check("Bitset+AVX512", run_bsa(), bsa_agg);
        check("Concise",       run_con(), con_agg);

        std::cout << "[OK] all active backends match DuckDB SQL ground truth "
                  << "(promo_revenue within " << tol << ")." << std::endl;
    } else {
        std::cout << "\n[Baseline] DuckDB SQL ground truth unavailable — skipping assert." << std::endl;
    }

    // ============================================================
    // 6. Statistics summary
    // ============================================================
    auto cb_or_s   = q14_compute_stats(cb_or_t);
    auto cb_and_s  = q14_compute_stats(cb_and_t);
    auto cb_agg_s  = q14_compute_stats(cb_agg_t);
    auto cb_tot_s  = q14_compute_stats(cb_tot_t);

    auto cr_or_s   = q14_compute_stats(cr_or_t);
    auto cr_and_s  = q14_compute_stats(cr_and_t);
    auto cr_agg_s  = q14_compute_stats(cr_agg_t);
    auto cr_tot_s  = q14_compute_stats(cr_tot_t);

    auto crr_or_s   = q14_compute_stats(crr_or_t);
    auto crr_and_s  = q14_compute_stats(crr_and_t);
    auto crr_agg_s  = q14_compute_stats(crr_agg_t);
    auto crr_tot_s  = q14_compute_stats(crr_tot_t);

    auto wah_or_s   = q14_compute_stats(wah_or_t);
    auto wah_and_s  = q14_compute_stats(wah_and_t);
    auto wah_agg_s  = q14_compute_stats(wah_agg_t);
    auto wah_tot_s  = q14_compute_stats(wah_tot_t);

    auto ew_or_s   = q14_compute_stats(ew_or_t);
    auto ew_and_s  = q14_compute_stats(ew_and_t);
    auto ew_agg_s  = q14_compute_stats(ew_agg_t);
    auto ew_tot_s  = q14_compute_stats(ew_tot_t);

    auto bs_or_s   = q14_compute_stats(bs_or_t);
    auto bs_and_s  = q14_compute_stats(bs_and_t);
    auto bs_agg_s  = q14_compute_stats(bs_agg_t);
    auto bs_tot_s  = q14_compute_stats(bs_tot_t);

    auto bsa_or_s   = q14_compute_stats(bsa_or_t);
    auto bsa_and_s  = q14_compute_stats(bsa_and_t);
    auto bsa_agg_s  = q14_compute_stats(bsa_agg_t);
    auto bsa_tot_s  = q14_compute_stats(bsa_tot_t);

    auto con_or_s   = q14_compute_stats(con_or_t);
    auto con_and_s  = q14_compute_stats(con_and_t);
    auto con_agg_s  = q14_compute_stats(con_agg_t);
    auto con_tot_s  = q14_compute_stats(con_tot_t);

    int measured = Q14_ITERATIONS - Q14_WARMUP;
    std::cout << std::fixed << std::setprecision(2);

    if (run_all()) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q14 RESULTS (" << measured << " measured iterations, median +/- stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  CB (ms)         CR (ms)        CRR (ms)        WAH (ms)        EW (ms)" << std::endl;
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;

        auto print_row = [](const char* label, Q14Stats& cb, Q14Stats& cr,
                            Q14Stats& crr, Q14Stats& wah, Q14Stats& ew) {
            std::cout << "  " << std::left << std::setw(14) << label
                      << std::right
                      << std::setw(8) << cb.median  << " +/- " << std::setw(5) << cb.stddev
                      << "  " << std::setw(8) << cr.median  << " +/- " << std::setw(5) << cr.stddev
                      << "  " << std::setw(8) << crr.median << " +/- " << std::setw(5) << crr.stddev
                      << "  " << std::setw(8) << wah.median << " +/- " << std::setw(5) << wah.stddev
                      << "  " << std::setw(8) << ew.median  << " +/- " << std::setw(5) << ew.stddev
                      << std::endl;
        };

        print_row("OR_date",  cb_or_s,   cr_or_s,   crr_or_s,   wah_or_s,   ew_or_s);
        print_row("AND_pro",  cb_and_s,  cr_and_s,  crr_and_s,  wah_and_s,  ew_and_s);
        print_row("Agg",      cb_agg_s,  cr_agg_s,  crr_agg_s,  wah_agg_s,  ew_agg_s);
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL",    cb_tot_s,  cr_tot_s,  crr_tot_s,  wah_tot_s,  ew_tot_s);
        std::cout << "================================================================\n" << std::endl;

        std::cout << "================================================================" << std::endl;
        std::cout << "  Q14 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  BS (ms)         BSA (ms)        Concise (ms)     BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        auto print_baseline_row = [](const char* label, Q14Stats& w,
                                     Q14Stats& b, Q14Stats& ba, Q14Stats& c) {
            double bs_sp  = (b.median  > 0) ? w.median / b.median  : 0;
            double bsa_sp = (ba.median > 0) ? w.median / ba.median : 0;
            double con_sp = (c.median  > 0) ? w.median / c.median  : 0;
            std::cout << "  " << std::left << std::setw(14) << label
                      << std::right
                      << std::setw(8) << b.median  << " +/- " << std::setw(5) << b.stddev
                      << "  " << std::setw(8) << ba.median << " +/- " << std::setw(5) << ba.stddev
                      << "  " << std::setw(8) << c.median  << " +/- " << std::setw(5) << c.stddev
                      << "     " << std::setw(5) << bs_sp  << "x"
                      << "     " << std::setw(5) << bsa_sp << "x"
                      << "     " << std::setw(5) << con_sp << "x"
                      << std::endl;
        };
        print_baseline_row("OR_date", wah_or_s,   bs_or_s,   bsa_or_s,   con_or_s);
        print_baseline_row("AND_pro", wah_and_s,  bs_and_s,  bsa_and_s,  con_and_s);
        print_baseline_row("Agg",     wah_agg_s,  bs_agg_s,  bsa_agg_s,  con_agg_s);
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        print_baseline_row("TOTAL",   wah_tot_s,  bs_tot_s,  bsa_tot_s,  con_tot_s);
        std::cout << "================================================================\n" << std::endl;
    } else {
        Q14Stats *sel_or = nullptr, *sel_and = nullptr, *sel_agg = nullptr, *sel_tot = nullptr;
        switch (Q14_BM) {
            case Q14BmType::WAH: sel_or = &wah_or_s; sel_and = &wah_and_s; sel_agg = &wah_agg_s; sel_tot = &wah_tot_s; break;
            case Q14BmType::CB:  sel_or = &cb_or_s;  sel_and = &cb_and_s;  sel_agg = &cb_agg_s;  sel_tot = &cb_tot_s;  break;
            case Q14BmType::CR:  sel_or = &cr_or_s;  sel_and = &cr_and_s;  sel_agg = &cr_agg_s;  sel_tot = &cr_tot_s;  break;
            case Q14BmType::CRR: sel_or = &crr_or_s; sel_and = &crr_and_s; sel_agg = &crr_agg_s; sel_tot = &crr_tot_s; break;
            case Q14BmType::EW:  sel_or = &ew_or_s;  sel_and = &ew_and_s;  sel_agg = &ew_agg_s;  sel_tot = &ew_tot_s;  break;
            case Q14BmType::BS:  sel_or = &bs_or_s;  sel_and = &bs_and_s;  sel_agg = &bs_agg_s;  sel_tot = &bs_tot_s;  break;
            case Q14BmType::BSA: sel_or = &bsa_or_s; sel_and = &bsa_and_s; sel_agg = &bsa_agg_s; sel_tot = &bsa_tot_s; break;
            case Q14BmType::CON: sel_or = &con_or_s; sel_and = &con_and_s; sel_agg = &con_agg_s; sel_tot = &con_tot_s; break;
            case Q14BmType::ALL: break;
        }

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q14 RESULTS — " << q14_bm_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;

        auto print_single = [](const char* label, Q14Stats& s) {
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(9) << s.median
                      << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val
                      << std::setw(10) << s.max_val << std::endl;
        };

        if (sel_or) {
            print_single("OR_date",  *sel_or);
            print_single("AND_pro",  *sel_and);
            print_single("Agg",      *sel_agg);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            print_single("TOTAL",    *sel_tot);
        }
        std::cout << "================================================================\n" << std::endl;
    }

    // ============================================================
    // 7. CSV export — ALL mode only
    // ============================================================
    if (run_all()) {
        std::string sf_label = q14_get_sf_label();
        std::string csv_path = "q14_results_" + sf_label + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << std::fixed << std::setprecision(4);
            csv << "sf,operation,"
                << "cb_median_ms,cb_stddev_ms,cb_min_ms,cb_max_ms,"
                << "cr_median_ms,cr_stddev_ms,cr_min_ms,cr_max_ms,"
                << "crr_median_ms,crr_stddev_ms,crr_min_ms,crr_max_ms,"
                << "wah_median_ms,wah_stddev_ms,wah_min_ms,wah_max_ms,"
                << "ew_median_ms,ew_stddev_ms,ew_min_ms,ew_max_ms,"
                << "bs_median_ms,bs_stddev_ms,bs_min_ms,bs_max_ms,"
                << "bsa_median_ms,bsa_stddev_ms,bsa_min_ms,bsa_max_ms,"
                << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms\n";

            auto csv_row = [&](const std::string& op,
                               Q14Stats& cb, Q14Stats& cr, Q14Stats& crr,
                               Q14Stats& wah, Q14Stats& ew,
                               Q14Stats& b, Q14Stats& ba, Q14Stats& co) {
                csv << sf_label << "," << op << ","
                    << cb.median  << "," << cb.stddev  << "," << cb.min_val  << "," << cb.max_val  << ","
                    << cr.median  << "," << cr.stddev  << "," << cr.min_val  << "," << cr.max_val  << ","
                    << crr.median << "," << crr.stddev << "," << crr.min_val << "," << crr.max_val << ","
                    << wah.median << "," << wah.stddev << "," << wah.min_val << "," << wah.max_val << ","
                    << ew.median  << "," << ew.stddev  << "," << ew.min_val  << "," << ew.max_val  << ","
                    << b.median   << "," << b.stddev   << "," << b.min_val   << "," << b.max_val   << ","
                    << ba.median  << "," << ba.stddev  << "," << ba.min_val  << "," << ba.max_val  << ","
                    << co.median  << "," << co.stddev  << "," << co.min_val  << "," << co.max_val  << "\n";
            };

            csv_row("OR_date", cb_or_s,   cr_or_s,   crr_or_s,   wah_or_s,   ew_or_s,   bs_or_s,   bsa_or_s,   con_or_s);
            csv_row("AND_pro", cb_and_s,  cr_and_s,  crr_and_s,  wah_and_s,  ew_and_s,  bs_and_s,  bsa_and_s,  con_and_s);
            csv_row("Agg",     cb_agg_s,  cr_agg_s,  crr_agg_s,  wah_agg_s,  ew_agg_s,  bs_agg_s,  bsa_agg_s,  con_agg_s);
            csv_row("TOTAL",   cb_tot_s,  cr_tot_s,  crr_tot_s,  wah_tot_s,  ew_tot_s,  bs_tot_s,  bsa_tot_s,  con_tot_s);

            csv.close();
            std::cout << "  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
