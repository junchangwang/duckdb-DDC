// TPC-H Q4 — Order Priority Checking Query (spec v3.0.1 §2.4.4)
//
//   SELECT o_orderpriority, count(*) AS order_count
//   FROM orders
//   WHERE o_orderdate >= DATE '[DATE]'
//     AND o_orderdate <  DATE '[DATE]' + INTERVAL '3' MONTH
//     AND EXISTS (SELECT * FROM lineitem
//                 WHERE l_orderkey = o_orderkey
//                   AND l_commitdate < l_receiptdate)
//   GROUP BY o_orderpriority
//   ORDER BY o_orderpriority;
//
// Spec validation params: DATE = '1993-07-01' → range [1993-07-01, 1993-10-01).
//
// Bitmaps are aligned to the orders table (rowid order, 15M rows for SF10):
//
//   orderdate/D.bm       D in [547..638]  (1993-07-01..1993-09-30,
//                                          days since 1992-01-01)
//   orderpriority/P.bm   P in [1..5]      (1-URGENT .. 5-LOW)
//   late_lineitem/0.bm   pre-encoded EXISTS predicate (orders with
//                        ≥ 1 lineitem where l_commitdate < l_receiptdate)
//
// Pipeline:
//   1. OR orderdate days [547..638]            → date_filter
//   2. AND late_lineitem                       → qualifying
//   3. For each priority p ∈ [1..5]:
//        AND priority_p → popcount → count_p
//
// The pre-encoded EXISTS bitmap is consistent with Q3's `join_result/0.bm`
// pattern: the relational join is computed once at export time, and the
// bitmap operations measured at query time are the standard
// OR (date) → AND (EXISTS) → AND (priority) → popcount pipeline that all
// backends share.  Spec compliance is asserted at the end against
// DuckDB's native SQL execution of the spec-form Q4 query (per-priority
// counts must match exactly).
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

#include "combit_adapter.h"
#include "combit/include/combit.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"

#include "bitset_simple.h"
#include "Concise/concise.h"
#include "execution/tpch/bm_baseline_loaders.hpp"
#include "execution/tpch/bm_bench_common.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace duckdb {

// --- Bitmap directories -----------------------------------------------------
static const std::string Q4_SF      = bm_bench::sf_suffix();
static const std::string Q4_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q4" + Q4_SF + "_combit");
static const std::string Q4_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q4" + Q4_SF + "_wah");
static const std::string Q4_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q4" + Q4_SF + "_croaring");
static const std::string Q4_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q4" + Q4_SF + "_ewah");

// --- Backend selection (DEBIT_BM, legacy Q4_BM also honoured) ---------------
using Q4BmType = bm_bench::Backend;
static const Q4BmType Q4_BM = bm_bench::parse_backend("Q4_BM");

static bool run_all() { return Q4_BM == Q4BmType::ALL; }
static bool run_wah() { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::WAH; }
static bool run_cb()  { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::CB;  }
static bool run_cr()  { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::CR;  }
static bool run_crr() { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::CRR; }
static bool run_ew()  { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::EW;  }
static bool run_bs()  { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::BS;  }
static bool run_bsa() { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::BSA; }
static bool run_con() { return Q4_BM == Q4BmType::ALL || Q4_BM == Q4BmType::CON; }

static const char* q4_label()    { return bm_bench::backend_label(Q4_BM); }
static std::string q4_sf_label() { return bm_bench::sf_label(); }

// --- Q4 predicate constants (spec §2.4.4) -----------------------------------
//   o_orderdate >= 1993-07-01  ↔  day >= 547  (days since 1992-01-01)
//   o_orderdate <  1993-10-01  ↔  day <  639  (i.e., day <= 638)
static const int  Q4_DATE_START = 547;
static const int  Q4_DATE_END   = 638;
static const int  Q4_PRIORITY_COUNT = 5;
static const std::array<const char*, Q4_PRIORITY_COUNT> Q4_PRIORITY_NAMES = {
    "1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"
};

static const int Q4_ITERATIONS = bm_bench::iter_count(10);
static const int Q4_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q4_once_flag;

// --- Per-format bitmap loaders (BS/CON share bm_baseline_loaders.hpp) -------
static ComBit q4_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q4_load_cr(const std::string& p) {
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
static ibis::bitvector q4_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q4_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// --- Stats helper ---
struct Q4Stats { double median = 0, stddev = 0, min_val = 0, max_val = 0; };
static Q4Stats q4_stats(std::vector<double> v) {
    Q4Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    s.median  = (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
    s.min_val = v.front();
    s.max_val = v.back();
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / n, sq = 0;
    for (auto x : v) sq += (x - mean) * (x - mean);
    s.stddev = std::sqrt(sq / n);
    return s;
}

static inline double q4_ms(std::chrono::high_resolution_clock::time_point a,
                           std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

// Per-priority count vector (size 5).
using Q4Counts = std::array<size_t, Q4_PRIORITY_COUNT>;

// ===========================================================================
// BMTPCH_Q4 — main benchmark entry point
// ===========================================================================
void BMTableScan::BMTPCH_Q4(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q4_once_flag, [&]() {

    using clock = std::chrono::high_resolution_clock;

    bm_bench::warn_if_sf1();

    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q4 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << q4_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q4 Benchmark — " << q4_label() << " only ("
                  << q4_sf_label() << ")" << std::endl;
    }
    std::cout << "  OR orderdate days " << Q4_DATE_START << ".." << Q4_DATE_END
              << " (" << (Q4_DATE_END - Q4_DATE_START + 1) << " bitmaps)"
              << " -> AND late_lineitem -> per-priority popcount" << std::endl;
    std::cout << "  TPC-H params: orderdate range [1993-07-01, 1993-10-01),"
              << " EXISTS l_commitdate < l_receiptdate" << std::endl;
    std::cout << "  Iterations: " << Q4_ITERATIONS
              << " (first " << Q4_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // -----------------------------------------------------------------------
    // 1. Read num_rows from the ComBit dir (compress_any writes done.txt).
    // -----------------------------------------------------------------------
    size_t num_rows = 0;
    {
        std::ifstream meta(Q4_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line))
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
    }
    if (num_rows == 0) {
        std::cerr << "Error: cannot read num_rows from " << Q4_CB_DIR << "/done.txt" << std::endl;
        return;
    }
    std::cout << "\n[Probe] orders num_rows = " << num_rows << std::endl;

    // -----------------------------------------------------------------------
    // 2. Load bitmaps per active backend.
    //    Each backend loads the same three sets (orderdate days, priorities,
    //    late_lineitem); load is timed per backend.
    // -----------------------------------------------------------------------
    std::cout << "\n[Load] Loading bitmaps (mode=" << q4_label() << ")..." << std::endl;
    const int date_n = Q4_DATE_END - Q4_DATE_START + 1;

    // ComBit
    std::vector<ComBit> cb_date; cb_date.reserve(date_n);
    std::vector<const ComBit*> cb_date_ptrs;
    std::array<ComBit, Q4_PRIORITY_COUNT> cb_prio;
    ComBit cb_late;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = clock::now();
        for (int d = Q4_DATE_START; d <= Q4_DATE_END; d++)
            cb_date.push_back(q4_load_cb(Q4_CB_DIR + "/orderdate/" + std::to_string(d) + ".bm"));
        for (auto& b : cb_date) cb_date_ptrs.push_back(&b);
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
            cb_prio[p] = q4_load_cb(Q4_CB_DIR + "/orderpriority/" + std::to_string(p+1) + ".bm");
        cb_late = q4_load_cb(Q4_CB_DIR + "/late_lineitem/0.bm");
        cb_load_ms = q4_ms(t0, clock::now());
    }

    // CRoaring (vanilla / no run-length).  Uses the plain pairwise
    // operator|= for OR — the naive CRoaring-API baseline.  CRR (below)
    // adds both runOptimize and fastunion to capture the "optimized
    // CRoaring" variant.  The two are intentionally distinct baselines.
    std::vector<roaring::Roaring> cr_date;
    std::array<roaring::Roaring, Q4_PRIORITY_COUNT> cr_prio;
    roaring::Roaring cr_late;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = clock::now();
        cr_date.reserve(date_n);
        for (int d = Q4_DATE_START; d <= Q4_DATE_END; d++)
            cr_date.push_back(q4_load_cr(Q4_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm"));
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
            cr_prio[p] = q4_load_cr(Q4_CR_DIR + "/orderpriority/" + std::to_string(p+1) + ".bm");
        cr_late = q4_load_cr(Q4_CR_DIR + "/late_lineitem/0.bm");
        cr_load_ms = q4_ms(t0, clock::now());
    }

    // CRoaring + Run (fastunion + runOptimize)
    std::vector<roaring::Roaring> crr_date;
    std::vector<const roaring::Roaring*> crr_date_ptrs;
    std::array<roaring::Roaring, Q4_PRIORITY_COUNT> crr_prio;
    roaring::Roaring crr_late;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = clock::now();
        crr_date.reserve(date_n);
        for (int d = Q4_DATE_START; d <= Q4_DATE_END; d++) {
            crr_date.push_back(q4_load_cr(Q4_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm"));
            crr_date.back().runOptimize();
        }
        for (auto& b : crr_date) crr_date_ptrs.push_back(&b);
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++) {
            crr_prio[p] = q4_load_cr(Q4_CR_DIR + "/orderpriority/" + std::to_string(p+1) + ".bm");
            crr_prio[p].runOptimize();
        }
        crr_late = q4_load_cr(Q4_CR_DIR + "/late_lineitem/0.bm");
        crr_late.runOptimize();
        crr_load_ms = q4_ms(t0, clock::now());
    }

    // WAH
    std::vector<ibis::bitvector> wah_date;
    std::array<ibis::bitvector, Q4_PRIORITY_COUNT> wah_prio;
    ibis::bitvector wah_late;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = clock::now();
        wah_date.reserve(date_n);
        for (int d = Q4_DATE_START; d <= Q4_DATE_END; d++)
            wah_date.push_back(q4_load_wah(Q4_WAH_DIR + "/orderdate/" + std::to_string(d) + ".bm"));
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
            wah_prio[p] = q4_load_wah(Q4_WAH_DIR + "/orderpriority/" + std::to_string(p+1) + ".bm");
        wah_late = q4_load_wah(Q4_WAH_DIR + "/late_lineitem/0.bm");
        wah_load_ms = q4_ms(t0, clock::now());
    }

    // EWAH.  Uses ewah::fast_logicalor (priority-queue k-way merge,
    // declared in ewah/ewah-inl.h) for k-way OR — the direct
    // counterpart of CRoaring's fastunion and Concise's
    // fast_logicalor, ensuring EWAH is not handicapped by a forced
    // sequential-pairwise OR loop.
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_date;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_date_ptrs;
    std::array<ewah::EWAHBoolArray<uint64_t>, Q4_PRIORITY_COUNT> ew_prio;
    ewah::EWAHBoolArray<uint64_t> ew_late;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = clock::now();
        ew_date.reserve(date_n);
        for (int d = Q4_DATE_START; d <= Q4_DATE_END; d++)
            ew_date.push_back(q4_load_ew(Q4_EW_DIR + "/orderdate/" + std::to_string(d) + ".bm"));
        for (auto& b : ew_date) ew_date_ptrs.push_back(&b);
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
            ew_prio[p] = q4_load_ew(Q4_EW_DIR + "/orderpriority/" + std::to_string(p+1) + ".bm");
        ew_late = q4_load_ew(Q4_EW_DIR + "/late_lineitem/0.bm");
        ew_load_ms = q4_ms(t0, clock::now());
    }

    // Bitset (BS / BSA share the same materialised bitmaps).
    std::vector<bs::Bitmap> bs_date;
    std::array<bs::Bitmap, Q4_PRIORITY_COUNT> bs_prio;
    bs::Bitmap bs_late;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = clock::now();
        bs_date.reserve(date_n);
        for (int d = Q4_DATE_START; d <= Q4_DATE_END; d++)
            bs_date.push_back(bm_bench::load_bitmap_from_croaring(
                Q4_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm"));
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
            bs_prio[p] = bm_bench::load_bitmap_from_croaring(
                Q4_CR_DIR + "/orderpriority/" + std::to_string(p+1) + ".bm");
        bs_late = bm_bench::load_bitmap_from_croaring(
            Q4_CR_DIR + "/late_lineitem/0.bm");
        bs_load_ms = q4_ms(t0, clock::now());
    }

    // Concise
    std::vector<ConciseSet<false>> con_date;
    std::vector<const ConciseSet<false>*> con_date_ptrs;
    std::array<ConciseSet<false>, Q4_PRIORITY_COUNT> con_prio;
    ConciseSet<false> con_late;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = clock::now();
        con_date.reserve(date_n);
        for (int d = Q4_DATE_START; d <= Q4_DATE_END; d++)
            con_date.push_back(bm_bench::load_concise_from_croaring(
                Q4_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm"));
        for (auto& b : con_date) con_date_ptrs.push_back(&b);
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
            con_prio[p] = bm_bench::load_concise_from_croaring(
                Q4_CR_DIR + "/orderpriority/" + std::to_string(p+1) + ".bm");
        con_late = bm_bench::load_concise_from_croaring(
            Q4_CR_DIR + "/late_lineitem/0.bm");
        con_load_ms = q4_ms(t0, clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:      " << wah_load_ms << " ms" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:   " << cb_load_ms  << " ms" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load: " << cr_load_ms  << " ms" << std::endl;
    if (run_crr()) std::cout << "  CRR load:      " << crr_load_ms << " ms" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:     " << ew_load_ms  << " ms" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:   " << bs_load_ms  << " ms (shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:  " << con_load_ms << " ms" << std::endl;

    // -----------------------------------------------------------------------
    // 3. Benchmark loop — per-iteration OR / AND_late / 5x AND_priority +
    //    popcount.  Each phase is timed separately so we can compare like
    //    for like across backends.
    // -----------------------------------------------------------------------
    std::vector<double> cb_or, cb_and_late, cb_and_prio, cb_pop, cb_tot;
    std::vector<double> cr_or, cr_and_late, cr_and_prio, cr_pop, cr_tot;
    std::vector<double> crr_or, crr_and_late, crr_and_prio, crr_pop, crr_tot;
    std::vector<double> wah_or, wah_and_late, wah_and_prio, wah_pop, wah_tot;
    std::vector<double> ew_or, ew_and_late, ew_and_prio, ew_pop, ew_tot;
    std::vector<double> bs_or, bs_and_late, bs_and_prio, bs_pop, bs_tot;
    std::vector<double> bsa_or, bsa_and_late, bsa_and_prio, bsa_pop, bsa_tot;
    std::vector<double> con_or, con_and_late, con_and_prio, con_pop, con_tot;

    Q4Counts cb_cnt{}, cr_cnt{}, crr_cnt{}, wah_cnt{}, ew_cnt{};
    Q4Counts bs_cnt{}, bsa_cnt{}, con_cnt{};

    for (int iter = 0; iter < Q4_ITERATIONS; iter++) {
        bool warm = iter < Q4_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q4_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        // ===== ComBit =====
        // popcount_and: fused AND + VPOPCNTDQ-popcount, no result
        // materialisation.  Library-side counterpart of the fused APIs
        // exposed by CRoaring (and_cardinality), EWAH (logicalandcount),
        // ibis::bitvector (count(mask)), and Concise (logicalandCount).
        if (run_cb()) {
            auto t0 = clock::now();
            ComBit cb_or_b = ComBit::OR_many(cb_date_ptrs.size(), cb_date_ptrs.data());
            auto t1 = clock::now();
            cb_or_b &= cb_late;
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = cb_or_b.popcount_and(cb_prio[p]);
            auto t3 = clock::now();
            cb_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  CB:   OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { cb_or.push_back(d0); cb_and_late.push_back(d1);
                         cb_and_prio.push_back(d2); cb_tot.push_back(dt); }
        }

        // ===== CRoaring (vanilla pairwise) =====
        // Plain operator|= over the date list — the naive CRoaring API
        // baseline.  CRR below adds fastunion + runOptimize.
        // and_cardinality: fused AND + popcount, no result materialisation.
        if (run_cr()) {
            auto t0 = clock::now();
            roaring::Roaring cr_or_b = cr_date[0];
            for (int i = 1; i < date_n; i++) cr_or_b |= cr_date[i];
            auto t1 = clock::now();
            cr_or_b &= cr_late;
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = cr_or_b.and_cardinality(cr_prio[p]);
            auto t3 = clock::now();
            cr_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  CR:   OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { cr_or.push_back(d0); cr_and_late.push_back(d1);
                         cr_and_prio.push_back(d2); cr_tot.push_back(dt); }
        }

        // ===== CRoaring+Run (fastunion) =====
        // and_cardinality: fused AND + popcount, no result materialisation.
        if (run_crr()) {
            auto t0 = clock::now();
            roaring::Roaring crr_or_b = roaring::Roaring::fastunion(
                crr_date_ptrs.size(), crr_date_ptrs.data());
            auto t1 = clock::now();
            crr_or_b &= crr_late;
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = crr_or_b.and_cardinality(crr_prio[p]);
            auto t3 = clock::now();
            crr_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  CRR:  OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { crr_or.push_back(d0); crr_and_late.push_back(d1);
                         crr_and_prio.push_back(d2); crr_tot.push_back(dt); }
        }

        // ===== WAH =====
        // ibis::bitvector::count(mask) is the fused AND-count primitive.
        if (run_wah()) {
            auto t0 = clock::now();
            ibis::bitvector wah_or_b = wah_date[0]; wah_or_b.decompress();
            for (int i = 1; i < date_n; i++) wah_or_b |= wah_date[i];
            auto t1 = clock::now();
            wah_or_b &= wah_late;
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = wah_or_b.count(wah_prio[p]);
            auto t3 = clock::now();
            wah_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  WAH:  OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { wah_or.push_back(d0); wah_and_late.push_back(d1);
                         wah_and_prio.push_back(d2); wah_tot.push_back(dt); }
        }

        // ===== EWAH =====
        // fast_logicalor: priority-queue k-way merge (ewah-inl.h:1129),
        // matches CRR fastunion / Concise fast_logicalor.  Without
        // this, EWAH was being forced through 91 sequential pairwise
        // ORs with 91 temp allocations — a fairness bug, not an
        // EWAH-vs-WAH algorithmic gap.
        // logicalandcount: fused AND + popcount, no materialisation.
        if (run_ew()) {
            auto t0 = clock::now();
            ewah::EWAHBoolArray<uint64_t> ew_or_b = ewah::fast_logicalor(
                ew_date_ptrs.size(), ew_date_ptrs.data());
            auto t1 = clock::now();
            ewah::EWAHBoolArray<uint64_t> ew_q;
            ew_or_b.logicaland(ew_late, ew_q);
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = ew_q.logicalandcount(ew_prio[p]);
            auto t3 = clock::now();
            ew_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  EW:   OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { ew_or.push_back(d0); ew_and_late.push_back(d1);
                         ew_and_prio.push_back(d2); ew_tot.push_back(dt); }
        }

        // ===== Bitset (scalar) =====
        // bs::and_popcount: fused AND + popcount in a single pass over
        // the word arrays — no scratch bitmap allocation per priority.
        if (run_bs()) {
            auto t0 = clock::now();
            bs::Bitmap bs_or_b = bs_date[0].clone();
            for (int i = 1; i < date_n; i++)
                bs::or_inplace(bs_or_b, bs_date[i], false);
            auto t1 = clock::now();
            bs::and_inplace(bs_or_b, bs_late, false);
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = bs::and_popcount(bs_or_b, bs_prio[p], false);
            auto t3 = clock::now();
            bs_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  BS:   OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { bs_or.push_back(d0); bs_and_late.push_back(d1);
                         bs_and_prio.push_back(d2); bs_tot.push_back(dt); }
        }

        // ===== Bitset + AVX-512 =====
        // bs::and_popcount: fused AND + popcount via AVX-512 VPOPCNTDQ.
        if (run_bsa()) {
            auto t0 = clock::now();
            bs::Bitmap bsa_or_b = bs_date[0].clone();
            for (int i = 1; i < date_n; i++)
                bs::or_inplace(bsa_or_b, bs_date[i], true);
            auto t1 = clock::now();
            bs::and_inplace(bsa_or_b, bs_late, true);
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = bs::and_popcount(bsa_or_b, bs_prio[p], true);
            auto t3 = clock::now();
            bsa_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  BSA:  OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { bsa_or.push_back(d0); bsa_and_late.push_back(d1);
                         bsa_and_prio.push_back(d2); bsa_tot.push_back(dt); }
        }

        // ===== Concise =====
        // logicalandCount: fused AND + popcount, no result materialisation.
        if (run_con()) {
            auto t0 = clock::now();
            ConciseSet<false> con_or_b = ConciseSet<false>::fast_logicalor(
                con_date_ptrs.size(), con_date_ptrs.data());
            auto t1 = clock::now();
            ConciseSet<false> con_q = con_or_b.logicaland(con_late);
            auto t2 = clock::now();
            Q4Counts cnt{};
            for (int p = 0; p < Q4_PRIORITY_COUNT; p++)
                cnt[p] = con_q.logicalandCount(con_prio[p]);
            auto t3 = clock::now();
            con_cnt = cnt;

            double d0=q4_ms(t0,t1), d1=q4_ms(t1,t2), d2=q4_ms(t2,t3), dt=q4_ms(t0,t3);
            std::cout << "  CON:  OR=" << d0 << "  AND_late=" << d1
                      << "  AND_prio+Pop=" << d2 << "  Total=" << dt << std::endl;
            if (!warm) { con_or.push_back(d0); con_and_late.push_back(d1);
                         con_and_prio.push_back(d2); con_tot.push_back(dt); }
        }
    } // end iterations

    // -----------------------------------------------------------------------
    // 4. DuckDB native SQL ground truth — per-priority counts.
    // -----------------------------------------------------------------------
    Q4Counts gt{};
    bool gt_ok = false;
    double gt_sql_ms = 0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT o_orderpriority, count(*) AS order_count "
            "FROM orders "
            "WHERE o_orderdate >= DATE '1993-07-01' "
            "  AND o_orderdate <  DATE '1993-10-01' "
            "  AND EXISTS (SELECT * FROM lineitem "
            "              WHERE l_orderkey = o_orderkey "
            "                AND l_commitdate < l_receiptdate) "
            "GROUP BY o_orderpriority "
            "ORDER BY o_orderpriority";
        auto t0 = clock::now();
        auto r = con.Query(sql);
        auto t1 = clock::now();
        if (r && !r->HasError()) {
            gt_sql_ms = q4_ms(t0, t1);
            for (idx_t i = 0; i < r->RowCount() && i < Q4_PRIORITY_COUNT; i++) {
                std::string name = r->GetValue(0, i).ToString();
                int64_t count = r->GetValue(1, i).GetValue<int64_t>();
                for (int p = 0; p < Q4_PRIORITY_COUNT; p++) {
                    if (name == Q4_PRIORITY_NAMES[p]) {
                        gt[p] = static_cast<size_t>(count);
                        break;
                    }
                }
            }
            gt_ok = true;
        } else if (r) {
            std::cerr << "[Baseline] SQL error: " << r->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    std::cout << std::fixed << std::setprecision(2);
    if (gt_ok) {
        std::cout << "\n[Baseline] DuckDB native SQL Q4  (single run: "
                  << gt_sql_ms << " ms)" << std::endl;
    } else {
        std::cout << "\n[Baseline] DuckDB SQL ground truth unavailable — skipping assert." << std::endl;
    }

    // Per-backend per-priority count vs SQL ground truth (exact match).
    auto check_counts = [&](const char* label, bool active, const Q4Counts& our) {
        if (!active || !gt_ok) return;
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++) {
            if (our[p] != gt[p]) {
                std::ostringstream oss;
                oss << "[FAIL] Q4 " << label << " priority " << Q4_PRIORITY_NAMES[p]
                    << " count mismatch: got " << our[p] << " vs SQL " << gt[p];
                throw std::runtime_error(oss.str());
            }
        }
    };
    if (gt_ok) {
        check_counts("WAH",           run_wah(), wah_cnt);
        check_counts("ComBit",        run_cb(),  cb_cnt);
        check_counts("CRoaring",      run_cr(),  cr_cnt);
        check_counts("CRoaring+Run",  run_crr(), crr_cnt);
        check_counts("EWAH",          run_ew(),  ew_cnt);
        check_counts("Bitset",        run_bs(),  bs_cnt);
        check_counts("Bitset+AVX512", run_bsa(), bsa_cnt);
        check_counts("Concise",       run_con(), con_cnt);
        std::cout << "[OK] all active backends match DuckDB SQL ground truth (per-priority counts exact)." << std::endl;
    }

    // -----------------------------------------------------------------------
    // 5. Print canonical per-priority counts (active backend with priority
    //    CB > CR > CRR > WAH > EW > BS > BSA > CON).
    // -----------------------------------------------------------------------
    const Q4Counts* canon = nullptr;
    const char* canon_label = "";
    if      (run_cb())  { canon = &cb_cnt;  canon_label = "ComBit"; }
    else if (run_cr())  { canon = &cr_cnt;  canon_label = "CRoaring"; }
    else if (run_crr()) { canon = &crr_cnt; canon_label = "CRoaring+Run"; }
    else if (run_wah()) { canon = &wah_cnt; canon_label = "WAH"; }
    else if (run_ew())  { canon = &ew_cnt;  canon_label = "EWAH"; }
    else if (run_bs())  { canon = &bs_cnt;  canon_label = "Bitset"; }
    else if (run_bsa()) { canon = &bsa_cnt; canon_label = "Bitset+AVX512"; }
    else if (run_con()) { canon = &con_cnt; canon_label = "Concise"; }

    if (canon) {
        std::cout << "\n  Q4 per-priority counts (source=" << canon_label << "):" << std::endl;
        std::cout << "  " << std::left << std::setw(20) << "o_orderpriority"
                  << std::right << std::setw(14) << "order_count" << std::endl;
        for (int p = 0; p < Q4_PRIORITY_COUNT; p++) {
            std::cout << "  " << std::left << std::setw(20) << Q4_PRIORITY_NAMES[p]
                      << std::right << std::setw(14) << (*canon)[p] << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // 6. Statistics tables + CSV (mirrors Q3 layout).
    // -----------------------------------------------------------------------
    auto cb_or_s   = q4_stats(cb_or),   cb_al_s  = q4_stats(cb_and_late),
         cb_ap_s   = q4_stats(cb_and_prio), cb_tot_s = q4_stats(cb_tot);
    auto cr_or_s   = q4_stats(cr_or),   cr_al_s  = q4_stats(cr_and_late),
         cr_ap_s   = q4_stats(cr_and_prio), cr_tot_s = q4_stats(cr_tot);
    auto crr_or_s  = q4_stats(crr_or),  crr_al_s = q4_stats(crr_and_late),
         crr_ap_s  = q4_stats(crr_and_prio), crr_tot_s = q4_stats(crr_tot);
    auto wah_or_s  = q4_stats(wah_or),  wah_al_s = q4_stats(wah_and_late),
         wah_ap_s  = q4_stats(wah_and_prio), wah_tot_s = q4_stats(wah_tot);
    auto ew_or_s   = q4_stats(ew_or),   ew_al_s  = q4_stats(ew_and_late),
         ew_ap_s   = q4_stats(ew_and_prio), ew_tot_s = q4_stats(ew_tot);
    auto bs_or_s   = q4_stats(bs_or),   bs_al_s  = q4_stats(bs_and_late),
         bs_ap_s   = q4_stats(bs_and_prio), bs_tot_s = q4_stats(bs_tot);
    auto bsa_or_s  = q4_stats(bsa_or),  bsa_al_s = q4_stats(bsa_and_late),
         bsa_ap_s  = q4_stats(bsa_and_prio), bsa_tot_s = q4_stats(bsa_tot);
    auto con_or_s  = q4_stats(con_or),  con_al_s = q4_stats(con_and_late),
         con_ap_s  = q4_stats(con_and_prio), con_tot_s = q4_stats(con_tot);

    int measured = Q4_ITERATIONS - Q4_WARMUP;

    if (run_all()) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q4 RESULTS (" << measured << " measured iterations, median +/- stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  CB (ms)         CR (ms)        CRR (ms)        WAH (ms)        EW (ms)" << std::endl;
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        auto pr = [](const char* l, Q4Stats& a, Q4Stats& b, Q4Stats& c, Q4Stats& d, Q4Stats& e) {
            std::cout << "  " << std::left << std::setw(14) << l << std::right
                << std::setw(8) << a.median << " +/- " << std::setw(5) << a.stddev
                << "  " << std::setw(8) << b.median << " +/- " << std::setw(5) << b.stddev
                << "  " << std::setw(8) << c.median << " +/- " << std::setw(5) << c.stddev
                << "  " << std::setw(8) << d.median << " +/- " << std::setw(5) << d.stddev
                << "  " << std::setw(8) << e.median << " +/- " << std::setw(5) << e.stddev << std::endl;
        };
        pr("OR_date",      cb_or_s,  cr_or_s,  crr_or_s,  wah_or_s,  ew_or_s);
        pr("AND_late",     cb_al_s,  cr_al_s,  crr_al_s,  wah_al_s,  ew_al_s);
        pr("AND_prio+Pop", cb_ap_s,  cr_ap_s,  crr_ap_s,  wah_ap_s,  ew_ap_s);
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        pr("TOTAL",        cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s);
        std::cout << "================================================================\n" << std::endl;

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q4 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  BS (ms)         BSA (ms)        Concise (ms)     BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        auto bp = [](const char* l, Q4Stats& w, Q4Stats& b, Q4Stats& ba, Q4Stats& c) {
            double s_b  = (b.median  > 0) ? w.median/b.median  : 0;
            double s_ba = (ba.median > 0) ? w.median/ba.median : 0;
            double s_c  = (c.median  > 0) ? w.median/c.median  : 0;
            std::cout << "  " << std::left << std::setw(14) << l << std::right
                << std::setw(8) << b.median  << " +/- " << std::setw(5) << b.stddev
                << "  " << std::setw(8) << ba.median << " +/- " << std::setw(5) << ba.stddev
                << "  " << std::setw(8) << c.median  << " +/- " << std::setw(5) << c.stddev
                << "     " << std::setw(5) << s_b  << "x"
                << "     " << std::setw(5) << s_ba << "x"
                << "     " << std::setw(5) << s_c  << "x" << std::endl;
        };
        bp("OR_date",      wah_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
        bp("AND_late",     wah_al_s,  bs_al_s,  bsa_al_s,  con_al_s);
        bp("AND_prio+Pop", wah_ap_s,  bs_ap_s,  bsa_ap_s,  con_ap_s);
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        bp("TOTAL",        wah_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);
        std::cout << "================================================================\n" << std::endl;
    } else {
        Q4Stats *sel_or=nullptr, *sel_al=nullptr, *sel_ap=nullptr, *sel_tot=nullptr;
        switch (Q4_BM) {
            case Q4BmType::WAH: sel_or=&wah_or_s; sel_al=&wah_al_s; sel_ap=&wah_ap_s; sel_tot=&wah_tot_s; break;
            case Q4BmType::CB:  sel_or=&cb_or_s;  sel_al=&cb_al_s;  sel_ap=&cb_ap_s;  sel_tot=&cb_tot_s;  break;
            case Q4BmType::CR:  sel_or=&cr_or_s;  sel_al=&cr_al_s;  sel_ap=&cr_ap_s;  sel_tot=&cr_tot_s;  break;
            case Q4BmType::CRR: sel_or=&crr_or_s; sel_al=&crr_al_s; sel_ap=&crr_ap_s; sel_tot=&crr_tot_s; break;
            case Q4BmType::EW:  sel_or=&ew_or_s;  sel_al=&ew_al_s;  sel_ap=&ew_ap_s;  sel_tot=&ew_tot_s;  break;
            case Q4BmType::BS:  sel_or=&bs_or_s;  sel_al=&bs_al_s;  sel_ap=&bs_ap_s;  sel_tot=&bs_tot_s;  break;
            case Q4BmType::BSA: sel_or=&bsa_or_s; sel_al=&bsa_al_s; sel_ap=&bsa_ap_s; sel_tot=&bsa_tot_s; break;
            case Q4BmType::CON: sel_or=&con_or_s; sel_al=&con_al_s; sel_ap=&con_ap_s; sel_tot=&con_tot_s; break;
            case Q4BmType::ALL: break;
        }
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q4 RESULTS — " << q4_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;
        auto pr_one = [](const char* l, Q4Stats& s) {
            std::cout << "  " << std::left << std::setw(16) << l << std::right
                      << std::setw(9) << s.median << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val << std::setw(10) << s.max_val << std::endl;
        };
        if (sel_or) {
            pr_one("OR_date",      *sel_or);
            pr_one("AND_late",     *sel_al);
            pr_one("AND_prio+Pop", *sel_ap);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            pr_one("TOTAL",        *sel_tot);
        }
        std::cout << "================================================================\n" << std::endl;
    }

    if (run_all()) {
        std::string sf = q4_sf_label();
        std::ofstream csv("q4_results_" + sf + ".csv");
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
            auto row = [&](const std::string& op,
                           Q4Stats& w, Q4Stats& c, Q4Stats& r, Q4Stats& rr, Q4Stats& e,
                           Q4Stats& b, Q4Stats& ba, Q4Stats& co) {
                auto sp = [](double w, double v) { return v > 0 ? w/v : 0.0; };
                csv << sf << "," << op << ","
                    << w.median  << "," << w.stddev  << "," << w.min_val  << "," << w.max_val  << ","
                    << c.median  << "," << c.stddev  << "," << c.min_val  << "," << c.max_val  << ","
                    << r.median  << "," << r.stddev  << "," << r.min_val  << "," << r.max_val  << ","
                    << rr.median << "," << rr.stddev << "," << rr.min_val << "," << rr.max_val << ","
                    << e.median  << "," << e.stddev  << "," << e.min_val  << "," << e.max_val  << ","
                    << b.median  << "," << b.stddev  << "," << b.min_val  << "," << b.max_val  << ","
                    << ba.median << "," << ba.stddev << "," << ba.min_val << "," << ba.max_val << ","
                    << co.median << "," << co.stddev << "," << co.min_val << "," << co.max_val << ","
                    << sp(w.median, c.median)  << "," << sp(w.median, r.median)  << ","
                    << sp(w.median, rr.median) << "," << sp(w.median, e.median)  << ","
                    << sp(w.median, b.median)  << "," << sp(w.median, ba.median) << ","
                    << sp(w.median, co.median) << "\n";
            };
            row("OR_date",      wah_or_s,  cb_or_s,  cr_or_s,  crr_or_s,  ew_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
            row("AND_late",     wah_al_s,  cb_al_s,  cr_al_s,  crr_al_s,  ew_al_s,  bs_al_s,  bsa_al_s,  con_al_s);
            row("AND_prio+Pop", wah_ap_s,  cb_ap_s,  cr_ap_s,  crr_ap_s,  ew_ap_s,  bs_ap_s,  bsa_ap_s,  con_ap_s);
            row("TOTAL",        wah_tot_s, cb_tot_s, cr_tot_s, crr_tot_s, ew_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);
            std::cout << "  [CSV] q4_results_" << sf << ".csv" << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
