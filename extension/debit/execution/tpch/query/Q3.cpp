// TPC-H Q3 — Shipping Priority Query (spec v3.0.1 §2.4.3)
//
//   SELECT l_orderkey,
//          sum(l_extendedprice * (1 - l_discount)) AS revenue,
//          o_orderdate, o_shippriority
//     FROM customer, orders, lineitem
//    WHERE c_mktsegment = '[SEGMENT]'
//      AND c_custkey  = o_custkey
//      AND l_orderkey = o_orderkey
//      AND o_orderdate < DATE '[DATE]'
//      AND l_shipdate  > DATE '[DATE]'
//    GROUP BY l_orderkey, o_orderdate, o_shippriority
//    ORDER BY revenue DESC, o_orderdate
//    LIMIT 10;
//
// Spec validation params: SEGMENT='BUILDING', DATE=1995-03-15.
//
// Bitmap pipeline:
//   1. join_result/0.bm pre-encodes the customer/orders/lineitem join
//      with c_mktsegment='BUILDING' AND o_orderdate<'1995-03-15'
//      (one bit per lineitem row; built once at export time).
//   2. OR shipdate days [1..1169]  → date_lo (= shipdate <= 1169)
//   3. NOT date_lo                 → date_hi (= shipdate >  1169)
//   4. AND with join_result        → filter
//   5. Walk filter, aggregate revenue per l_orderkey
//   6. Top-10 via heap, sorted DESC by (revenue, o_orderdate)
//
// Day 1169 = 1995-03-15 in days-since-1992-01-01.  The bitmap exporter
// emits days 1..2526 (full TPC-H shipdate domain), so the complement
// "days >1169" requires OR over 1169 bitmaps + NOT over [0, num_rows).
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

// --- Bitmap directories -----------------------------------------------------
static const std::string Q3_SF      = bm_bench::sf_suffix();
static const std::string Q3_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q3" + Q3_SF + "_combit");
static const std::string Q3_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q3" + Q3_SF + "_wah");
static const std::string Q3_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q3" + Q3_SF + "_croaring");
static const std::string Q3_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q3" + Q3_SF + "_ewah");

// --- Backend selection (DEBIT_BM, legacy Q3_BM also honoured) ---------------
using Q3BmType = bm_bench::Backend;
static const Q3BmType Q3_BM = bm_bench::parse_backend("Q3_BM");

static bool run_all() { return Q3_BM == Q3BmType::ALL; }
static bool run_wah() { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::WAH; }
static bool run_cb()  { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::CB;  }
static bool run_cr()  { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::CR;  }
static bool run_crr() { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::CRR; }
static bool run_ew()  { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::EW;  }
static bool run_bs()  { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::BS;  }
static bool run_bsa() { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::BSA; }
static bool run_con() { return Q3_BM == Q3BmType::ALL || Q3_BM == Q3BmType::CON; }

static const char* q3_label()    { return bm_bench::backend_label(Q3_BM); }
static std::string q3_sf_label() { return bm_bench::sf_label(); }

// --- Q3 predicate (spec §2.4.3) ---
//   l_shipdate > 1995-03-15  ↔  day > 1169 (days since 1992-01-01)
//   Complement: OR days [1..1169], NOT.
static const int Q3_SHIP_CUTOFF = 1169;

static const int Q3_ITERATIONS = bm_bench::iter_count(10);
static const int Q3_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q3_once_flag;

// --- ComBit byte-LUT (shared pattern with Q5/Q6/Q10) ---
struct Q3ByteEntry { uint8_t count; uint8_t pos[8]; };
static Q3ByteEntry q3_byte_lut[256];
static bool q3_byte_lut_init = []() {
    for (int v = 0; v < 256; v++) {
        uint8_t c = 0;
        for (int b = 7; b >= 0; b--)
            if (v & (1 << b)) q3_byte_lut[v].pos[c++] = 7 - b;
        q3_byte_lut[v].count = c;
    }
    return true;
}();

// --- Per-format bitmap loaders (BS/CON share bm_baseline_loaders.hpp) ---
static ComBit q3_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q3_load_cr(const std::string& p) {
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
static ibis::bitvector q3_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q3_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// --- WAH NOT (in-place AVX-512 flip on decompressed bitvector) ---
static void q3_wah_flip(ibis::bitvector* btv) {
#if defined(__AVX512F__)
    auto* it = btv->m_vec.begin();
    while (it + 15 < btv->m_vec.end()) {
        _mm512_storeu_epi32(it, _mm512_andnot_epi32(
            _mm512_loadu_epi32(it), _mm512_set1_epi32(0x7fffffff)));
        it += 16;
    }
    for (; it < btv->m_vec.end(); it++) *it ^= ibis::bitvector::ALLONES;
    if (btv->active.nbits > 0) btv->active.val ^= ((1u << btv->active.nbits) - 1);
#else
    for (auto* it = btv->m_vec.begin(); it < btv->m_vec.end(); ++it)
        *it ^= ibis::bitvector::ALLONES;
    if (btv->active.nbits > 0) btv->active.val ^= ((1u << btv->active.nbits) - 1);
#endif
}

// --- Stats helper ---
struct Q3Stats { double median = 0, stddev = 0, min_val = 0, max_val = 0; };
static Q3Stats q3_stats(std::vector<double> v) {
    Q3Stats s{};
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

// --- Top-10 element + min-heap comparator ---
//   Min-heap is keyed so that the *worst* row sits on top:
//   smallest revenue (or, on tie, latest orderdate).  When a better
//   candidate arrives we pop top and push it.  Final sort uses the
//   spec ORDER BY: revenue DESC, o_orderdate ASC.
struct Q3Row {
    int64_t orderkey;
    int64_t revenue_fp;   // fixed-point: l_extendedprice × (100 - l_discount)
    int32_t orderdate_epoch;
    int32_t shippriority;
};
struct Q3MinHeapCmp {
    bool operator()(const Q3Row& a, const Q3Row& b) const {
        if (a.revenue_fp != b.revenue_fp) return a.revenue_fp > b.revenue_fp;
        return a.orderdate_epoch < b.orderdate_epoch;  // earlier date wins
    }
};

static inline double q3_ms(std::chrono::high_resolution_clock::time_point a,
                           std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

// ===========================================================================
// BMTPCH_Q3 — main benchmark entry point
// ===========================================================================
void BMTableScan::BMTPCH_Q3(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q3_once_flag, [&]() {

    using clock = std::chrono::high_resolution_clock;

    bm_bench::warn_if_sf1();

    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q3 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << q3_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q3 Benchmark — " << q3_label() << " only ("
                  << q3_sf_label() << ")" << std::endl;
    }
    std::cout << "  Complement OR shipdate days 1.." << Q3_SHIP_CUTOFF
              << " -> NOT -> AND join_result -> Top-10" << std::endl;
    std::cout << "  TPC-H params: c_mktsegment='BUILDING', date=1995-03-15" << std::endl;
    std::cout << "  Iterations: " << Q3_ITERATIONS << " (first " << Q3_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // -----------------------------------------------------------------------
    // 1. Side-table load: orders_meta — qualifying (orderkey -> orderdate, shippriority).
    //    Loaded directly via DuckDB SQL (no external CSV); only orders that
    //    pass the join+segment+date predicate are returned.
    // -----------------------------------------------------------------------
    std::unordered_map<int64_t, std::pair<int32_t, int32_t>> orders_meta;
    {
        auto t0 = clock::now();
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT o.o_orderkey, "
            "       (o.o_orderdate - DATE '1970-01-01')::INT AS orderdate_epoch, "
            "       o.o_shippriority "
            "FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey "
            "WHERE c.c_mktsegment = 'BUILDING' AND o.o_orderdate < DATE '1995-03-15'";
        auto r = con.Query(sql);
        if (r && !r->HasError()) {
            orders_meta.reserve(r->RowCount());
            for (idx_t i = 0; i < r->RowCount(); i++) {
                orders_meta[r->GetValue(0, i).GetValue<int64_t>()] = {
                    r->GetValue(1, i).GetValue<int32_t>(),
                    r->GetValue(2, i).GetValue<int32_t>()
                };
            }
        }
        std::cout << "\n[Side-load] orders_meta: " << orders_meta.size()
                  << " qualifying orders in " << q3_ms(t0, clock::now()) << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. Pre-load lineitem columns: l_orderkey(0), l_extendedprice(5), l_discount(6).
    // -----------------------------------------------------------------------
    size_t num_rows = 0;
    {
        std::ifstream meta(Q3_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line))
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
    }
    if (num_rows == 0) {
        std::cerr << "Error: cannot read num_rows from " << Q3_CB_DIR << "/done.txt" << std::endl;
        return;
    }

    auto& tbl = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");
    auto& tx  = DuckTransaction::Get(context.client, tbl.catalog);
    TableScanState st;
    vector<StorageIndex> cols{ StorageIndex(0), StorageIndex(5), StorageIndex(6) };
    tbl.GetStorage().InitializeScan(context.client, tx, st, cols);
    vector<LogicalType> types{
        tbl.GetColumns().GetColumnTypes()[0],
        tbl.GetColumns().GetColumnTypes()[5],
        tbl.GetColumns().GetColumnTypes()[6] };

    std::vector<int64_t> col_okey(num_rows), col_price(num_rows), col_disc(num_rows);
    {
        auto t0 = clock::now();
        size_t off = 0;
        while (true) {
            DataChunk ch; ch.Initialize(context.client, types);
            tbl.GetStorage().Scan(tx, ch, st);
            if (ch.size() == 0) break;
            std::memcpy(col_okey.data()  + off, FlatVector::GetData<int64_t>(ch.data[0]), ch.size() * 8);
            std::memcpy(col_price.data() + off, FlatVector::GetData<int64_t>(ch.data[1]), ch.size() * 8);
            std::memcpy(col_disc.data()  + off, FlatVector::GetData<int64_t>(ch.data[2]), ch.size() * 8);
            off += ch.size();
        }
        std::cout << "[Pre-load] lineitem: " << off << " rows (l_orderkey, l_extendedprice, l_discount) in "
                  << q3_ms(t0, clock::now()) << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 3. Load bitmaps (per-backend gated).
    //    join_result is a single bitmap; shipdate is 1169 bitmaps for the
    //    complement OR.  All loaders are timed individually.
    // -----------------------------------------------------------------------
    std::cout << "\n[Load] Loading bitmaps (mode=" << q3_label() << ")..." << std::endl;
    const int ship_or_end = Q3_SHIP_CUTOFF;

    std::vector<ComBit> cb_ship; ComBit cb_join;
    std::vector<const ComBit*> cb_ship_ptrs;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = clock::now();
        cb_join = q3_load_cb(Q3_CB_DIR + "/join_result/0.bm");
        cb_ship.resize(ship_or_end + 1);
        for (int d = 1; d <= ship_or_end; d++)
            cb_ship[d] = q3_load_cb(Q3_CB_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cb_ship_ptrs.reserve(ship_or_end);
        for (int d = 1; d <= ship_or_end; d++) cb_ship_ptrs.push_back(&cb_ship[d]);
        cb_load_ms = q3_ms(t0, clock::now());
    }

    // CRoaring (vanilla / no run-length).  Uses the plain pairwise
    // operator|= for OR — the naive CRoaring-API baseline.  CRR (below)
    // adds both runOptimize and fastunion to capture the "optimized
    // CRoaring" variant.  The two are intentionally distinct baselines.
    std::vector<roaring::Roaring> cr_ship; roaring::Roaring cr_join;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = clock::now();
        cr_join = q3_load_cr(Q3_CR_DIR + "/join_result/0.bm");
        cr_ship.resize(ship_or_end + 1);
        for (int d = 1; d <= ship_or_end; d++)
            cr_ship[d] = q3_load_cr(Q3_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cr_load_ms = q3_ms(t0, clock::now());
    }

    std::vector<roaring::Roaring> crr_ship; roaring::Roaring crr_join;
    std::vector<const roaring::Roaring*> crr_ship_ptrs;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = clock::now();
        crr_join = q3_load_cr(Q3_CR_DIR + "/join_result/0.bm");
        crr_join.runOptimize();
        crr_ship.resize(ship_or_end + 1);
        for (int d = 1; d <= ship_or_end; d++) {
            crr_ship[d] = q3_load_cr(Q3_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
            crr_ship[d].runOptimize();
        }
        crr_ship_ptrs.reserve(ship_or_end);
        for (int d = 1; d <= ship_or_end; d++) crr_ship_ptrs.push_back(&crr_ship[d]);
        crr_load_ms = q3_ms(t0, clock::now());
    }

    std::vector<ibis::bitvector> wah_ship; ibis::bitvector wah_join;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = clock::now();
        wah_join = q3_load_wah(Q3_WAH_DIR + "/join_result/0.bm");
        wah_ship.resize(ship_or_end + 1);
        for (int d = 1; d <= ship_or_end; d++)
            wah_ship[d] = q3_load_wah(Q3_WAH_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        wah_load_ms = q3_ms(t0, clock::now());
    }

    // EWAH.  Pre-build the ptr array so the OR phase can use
    // ewah::fast_logicalor (priority-queue k-way merge), the direct
    // counterpart of CRR fastunion / Concise fast_logicalor.  Without
    // this, EWAH was being forced through a sequential pairwise OR loop
    // with one temp allocation per step — a fairness bug, not an
    // EWAH-vs-WAH algorithmic gap.
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_ship; ewah::EWAHBoolArray<uint64_t> ew_join;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_ship_ptrs;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = clock::now();
        ew_join = q3_load_ew(Q3_EW_DIR + "/join_result/0.bm");
        ew_ship.resize(ship_or_end + 1);
        for (int d = 1; d <= ship_or_end; d++)
            ew_ship[d] = q3_load_ew(Q3_EW_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        ew_ship_ptrs.reserve(ship_or_end);
        for (int d = 1; d <= ship_or_end; d++) ew_ship_ptrs.push_back(&ew_ship[d]);
        ew_load_ms = q3_ms(t0, clock::now());
    }

    // Bitset (uncompressed; shared by BS / BSA).  Loaded from CRoaring
    // file via the shared bm_bench helper.
    std::vector<bs::Bitmap> bs_ship; bs::Bitmap bs_join;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = clock::now();
        bs_join = bm_bench::load_bitmap_from_croaring(Q3_CR_DIR + "/join_result/0.bm");
        bs_ship.resize(ship_or_end + 1);
        for (int d = 1; d <= ship_or_end; d++)
            bs_ship[d] = bm_bench::load_bitmap_from_croaring(Q3_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        bs_load_ms = q3_ms(t0, clock::now());
    }

    // Concise.  con_full universe is built once at load to support NOT
    // via `con_full.logicalandnot(future)` (no native flip in Concise).
    std::vector<ConciseSet<false>> con_ship; ConciseSet<false> con_join;
    std::vector<const ConciseSet<false>*> con_ship_ptrs;
    ConciseSet<false> con_full;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = clock::now();
        con_join = bm_bench::load_concise_from_croaring(Q3_CR_DIR + "/join_result/0.bm");
        con_ship.resize(ship_or_end + 1);
        for (int d = 1; d <= ship_or_end; d++)
            con_ship[d] = bm_bench::load_concise_from_croaring(Q3_CR_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        con_ship_ptrs.reserve(ship_or_end);
        for (int d = 1; d <= ship_or_end; d++) con_ship_ptrs.push_back(&con_ship[d]);
        for (uint32_t i = 0; i < num_rows; i++) con_full.add(i);
        con_load_ms = q3_ms(t0, clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:      " << wah_load_ms << " ms" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:   " << cb_load_ms  << " ms" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load: " << cr_load_ms  << " ms" << std::endl;
    if (run_crr()) std::cout << "  CRR load:      " << crr_load_ms << " ms" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:     " << ew_load_ms  << " ms" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:   " << bs_load_ms  << " ms (shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:  " << con_load_ms << " ms (incl. universe build)" << std::endl;

    // -----------------------------------------------------------------------
    // 4. Top-K extractor (lives outside the timed phases so per-backend
    //    Agg time only measures the per-row aggregation, not the Top-10
    //    sort — same convention as Q10).
    // -----------------------------------------------------------------------
    auto extract_topk = [&](const std::unordered_map<int64_t, int64_t>& rev,
                            std::vector<Q3Row>& out) {
        std::priority_queue<Q3Row, std::vector<Q3Row>, Q3MinHeapCmp> heap;
        for (auto& [ok, r] : rev) {
            if (r == 0) continue;
            auto mit = orders_meta.find(ok);
            if (mit == orders_meta.end()) continue;
            Q3Row row{ok, r, mit->second.first, mit->second.second};
            if ((int)heap.size() < 10) {
                heap.push(row);
            } else {
                const auto& top = heap.top();
                if (row.revenue_fp > top.revenue_fp ||
                    (row.revenue_fp == top.revenue_fp && row.orderdate_epoch < top.orderdate_epoch)) {
                    heap.pop();
                    heap.push(row);
                }
            }
        }
        out.clear();
        while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
        std::sort(out.begin(), out.end(), [](const Q3Row& a, const Q3Row& b) {
            if (a.revenue_fp != b.revenue_fp) return a.revenue_fp > b.revenue_fp;
            return a.orderdate_epoch < b.orderdate_epoch;
        });
    };

    // -----------------------------------------------------------------------
    // 5. Benchmark loop.
    // -----------------------------------------------------------------------
    std::vector<double> cb_or, cb_not, cb_and, cb_agg, cb_tot;
    std::vector<double> cr_or, cr_not, cr_and, cr_agg, cr_tot;
    std::vector<double> crr_or, crr_not, crr_and, crr_agg, crr_tot;
    std::vector<double> wah_or, wah_not, wah_and, wah_agg, wah_tot;
    std::vector<double> ew_or, ew_not, ew_and, ew_agg, ew_tot;
    std::vector<double> bs_or, bs_not, bs_and, bs_agg, bs_tot;
    std::vector<double> bsa_or, bsa_not, bsa_and, bsa_agg, bsa_tot;
    std::vector<double> con_or, con_not, con_and, con_agg, con_tot;

    std::vector<Q3Row> cb_top, cr_top, crr_top, wah_top, ew_top, bs_top, bsa_top, con_top;
    size_t cb_rows=0, cr_rows=0, crr_rows=0, wah_rows=0, ew_rows=0, bs_rows=0, bsa_rows=0, con_rows=0;

    for (int iter = 0; iter < Q3_ITERATIONS; iter++) {
        bool warm = iter < Q3_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q3_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        // ===== ComBit =====
        // Decode via for_each_literal — the AVX-512 L3-bitscan / per-segment
        // byte-mask path inside ComBit::for_each_literal skips all-zero
        // regions in bulk, which is the optimal walk for the sparse
        // result produced here (~0.5% density on SF10).  Same pattern as Q10.
        if (run_cb()) {
            auto t0 = clock::now();
            ComBit cb_comp = ComBit::OR_many(cb_ship_ptrs.size(), cb_ship_ptrs.data());
            auto t1 = clock::now();
            ComBit cb_filt = ~cb_comp;
            auto t2 = clock::now();
            cb_filt &= cb_join;
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            cb_filt.for_each_literal(
                [&](uint32_t word_pos, uint8_t val) {
                    size_t rbase = static_cast<size_t>(word_pos) * 8;
                    const auto& e = q3_byte_lut[val];
                    for (int k = 0; k < e.count; k++) {
                        size_t r = rbase + e.pos[k];
                        rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                        cnt++;
                    }
                });
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            cb_top = top; cb_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  CB:   OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { cb_or.push_back(d0); cb_not.push_back(d1); cb_and.push_back(d2);
                         cb_agg.push_back(d3); cb_tot.push_back(dt); }
        }

        // ===== CRoaring (vanilla pairwise) =====
        // Plain operator|= over the date list — the naive CRoaring API
        // baseline.  CRR below adds fastunion + runOptimize.
        if (run_cr()) {
            auto t0 = clock::now();
            roaring::Roaring comp = cr_ship[1];
            for (int d = 2; d <= ship_or_end; d++) comp |= cr_ship[d];
            auto t1 = clock::now();
            comp.flip(0, static_cast<uint64_t>(num_rows));
            auto t2 = clock::now();
            roaring::Roaring filt = comp & cr_join;
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            for (auto it = filt.begin(); it != filt.end(); ++it) {
                size_t r = *it;
                rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            cr_top = top; cr_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  CR:   OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { cr_or.push_back(d0); cr_not.push_back(d1); cr_and.push_back(d2);
                         cr_agg.push_back(d3); cr_tot.push_back(dt); }
        }

        // ===== CRoaring+Run (fastunion) =====
        if (run_crr()) {
            auto t0 = clock::now();
            roaring::Roaring comp = roaring::Roaring::fastunion(crr_ship_ptrs.size(), crr_ship_ptrs.data());
            auto t1 = clock::now();
            comp.flip(0, static_cast<uint64_t>(num_rows));
            auto t2 = clock::now();
            roaring::Roaring filt = comp & crr_join;
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            for (auto it = filt.begin(); it != filt.end(); ++it) {
                size_t r = *it;
                rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            crr_top = top; crr_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  CRR:  OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { crr_or.push_back(d0); crr_not.push_back(d1); crr_and.push_back(d2);
                         crr_agg.push_back(d3); crr_tot.push_back(dt); }
        }

        // ===== WAH =====
        if (run_wah()) {
            auto t0 = clock::now();
            ibis::bitvector comp = wah_ship[1]; comp.decompress();
            for (int d = 2; d <= ship_or_end; d++) comp |= wah_ship[d];
            auto t1 = clock::now();
            q3_wah_flip(&comp);
            auto t2 = clock::now();
            ibis::bitvector filt; filt.copy(comp); filt &= wah_join;
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            ibis::bitvector::pit pit(filt);
            while (*pit != 0xFFFFFFFFU) {
                size_t r = *pit;
                rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                cnt++; pit.next();
            }
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            wah_top = top; wah_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  WAH:  OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { wah_or.push_back(d0); wah_not.push_back(d1); wah_and.push_back(d2);
                         wah_agg.push_back(d3); wah_tot.push_back(dt); }
        }

        // ===== EWAH =====
        // fast_logicalor: priority-queue k-way merge (ewah-inl.h:1129).
        // Measured ~1.5x faster than the iterative pairwise loop on
        // 1169 dense shipdate bitmaps (Q3 SF10).
        if (run_ew()) {
            auto t0 = clock::now();
            ewah::EWAHBoolArray<uint64_t> comp = ewah::fast_logicalor(
                ew_ship_ptrs.size(), ew_ship_ptrs.data());
            auto t1 = clock::now();
            comp.padWithZeroes(num_rows);
            comp.inplace_logicalnot();
            auto t2 = clock::now();
            ewah::EWAHBoolArray<uint64_t> filt;
            comp.logicaland(ew_join, filt);
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            for (auto it = filt.begin(); it != filt.end(); ++it) {
                size_t r = *it;
                rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            ew_top = top; ew_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  EW:   OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { ew_or.push_back(d0); ew_not.push_back(d1); ew_and.push_back(d2);
                         ew_agg.push_back(d3); ew_tot.push_back(dt); }
        }

        // ===== Bitset (scalar) =====
        if (run_bs()) {
            auto t0 = clock::now();
            bs::Bitmap comp = bs_ship[1].clone();
            for (int d = 2; d <= ship_or_end; d++)
                bs::or_inplace(comp, bs_ship[d], false);
            auto t1 = clock::now();
            bs::not_inplace(comp, false);
            auto t2 = clock::now();
            bs::and_inplace(comp, bs_join, false);
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            for (size_t i = 0; i < comp.nwords; ++i) {
                uint64_t w = comp.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= comp.nbits) break;
                    rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                    cnt++;
                    w &= w - 1;
                }
            }
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            bs_top = top; bs_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  BS:   OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { bs_or.push_back(d0); bs_not.push_back(d1); bs_and.push_back(d2);
                         bs_agg.push_back(d3); bs_tot.push_back(dt); }
        }

        // ===== Bitset + AVX-512 =====
        if (run_bsa()) {
            auto t0 = clock::now();
            bs::Bitmap comp = bs_ship[1].clone();
            for (int d = 2; d <= ship_or_end; d++)
                bs::or_inplace(comp, bs_ship[d], true);
            auto t1 = clock::now();
            bs::not_inplace(comp, true);
            auto t2 = clock::now();
            bs::and_inplace(comp, bs_join, true);
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            for (size_t i = 0; i < comp.nwords; ++i) {
                uint64_t w = comp.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= comp.nbits) break;
                    rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                    cnt++;
                    w &= w - 1;
                }
            }
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            bsa_top = top; bsa_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  BSA:  OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { bsa_or.push_back(d0); bsa_not.push_back(d1); bsa_and.push_back(d2);
                         bsa_agg.push_back(d3); bsa_tot.push_back(dt); }
        }

        // ===== Concise =====
        if (run_con()) {
            auto t0 = clock::now();
            ConciseSet<false> comp = ConciseSet<false>::fast_logicalor(
                con_ship_ptrs.size(), con_ship_ptrs.data());
            auto t1 = clock::now();
            ConciseSet<false> filter = con_full.logicalandnot(comp);
            auto t2 = clock::now();
            ConciseSet<false> filt = filter.logicaland(con_join);
            auto t3 = clock::now();
            std::unordered_map<int64_t, int64_t> rev;
            size_t cnt = 0;
            for (auto it = filt.begin(); it != filt.end(); ++it) {
                size_t r = *it;
                rev[col_okey[r]] += col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t4 = clock::now();
            std::vector<Q3Row> top; extract_topk(rev, top);
            con_top = top; con_rows = cnt;

            double d0=q3_ms(t0,t1), d1=q3_ms(t1,t2), d2=q3_ms(t2,t3), d3=q3_ms(t3,t4), dt=q3_ms(t0,t4);
            std::cout << "  CON:  OR=" << d0 << "  NOT=" << d1 << "  AND=" << d2
                      << "  Agg=" << d3 << "  Total=" << dt << "  rows=" << cnt << std::endl;
            if (!warm) { con_or.push_back(d0); con_not.push_back(d1); con_and.push_back(d2);
                         con_agg.push_back(d3); con_tot.push_back(dt); }
        }
    } // end iterations

    // -----------------------------------------------------------------------
    // 6. DuckDB native SQL ground truth — Top-10 by spec.
    // -----------------------------------------------------------------------
    struct Q3Gt { int64_t orderkey; double revenue; int32_t orderdate_epoch; int32_t shippriority; };
    std::vector<Q3Gt> gt;
    double gt_sql_ms = 0;
    bool gt_ok = false;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT l.l_orderkey, "
            "       CAST(sum(l.l_extendedprice * (1 - l.l_discount)) AS DOUBLE) AS revenue, "
            "       (o.o_orderdate - DATE '1970-01-01')::INT AS orderdate_epoch, "
            "       o.o_shippriority "
            "FROM customer c, orders o, lineitem l "
            "WHERE c.c_mktsegment = 'BUILDING' "
            "  AND c.c_custkey = o.o_custkey "
            "  AND l.l_orderkey = o.o_orderkey "
            "  AND o.o_orderdate < DATE '1995-03-15' "
            "  AND l.l_shipdate  > DATE '1995-03-15' "
            "GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority "
            "ORDER BY revenue DESC, o.o_orderdate "
            "LIMIT 10";
        auto t0 = clock::now();
        auto r = con.Query(sql);
        auto t1 = clock::now();
        if (r && !r->HasError()) {
            gt_sql_ms = q3_ms(t0, t1);
            for (idx_t i = 0; i < r->RowCount(); i++)
                gt.push_back({r->GetValue(0, i).GetValue<int64_t>(),
                              r->GetValue(1, i).GetValue<double>(),
                              r->GetValue(2, i).GetValue<int32_t>(),
                              r->GetValue(3, i).GetValue<int32_t>()});
            gt_ok = true;
        } else if (r) {
            std::cerr << "[Baseline] SQL error: " << r->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    std::cout << std::fixed << std::setprecision(2);
    if (gt_ok) {
        std::cout << "\n[Baseline] DuckDB native SQL Top-" << gt.size()
                  << "  (single run: " << gt_sql_ms << " ms)" << std::endl;
    } else {
        std::cout << "\n[Baseline] DuckDB SQL ground truth unavailable — skipping assert." << std::endl;
    }

    // Per-backend Top-10 vs SQL ground truth (orderkey identity + revenue
    // tolerance 0.01; same convention as Q10).
    auto check_top = [&](const char* label, bool active, const std::vector<Q3Row>& our) {
        if (!active || !gt_ok) return;
        if (our.size() != gt.size()) {
            std::ostringstream oss;
            oss << "[FAIL] Q3 " << label << " produced " << our.size()
                << " rows, SQL ground truth has " << gt.size();
            throw std::runtime_error(oss.str());
        }
        for (size_t i = 0; i < our.size(); i++) {
            double our_rev = static_cast<double>(our[i].revenue_fp) / 10000.0;
            if (our[i].orderkey != gt[i].orderkey ||
                std::fabs(our_rev - gt[i].revenue) > 0.01) {
                std::ostringstream oss;
                oss << "[FAIL] Q3 " << label << " row " << i
                    << ": got (orderkey=" << our[i].orderkey << ", revenue=" << our_rev << ")"
                    << " vs SQL (orderkey=" << gt[i].orderkey << ", revenue=" << gt[i].revenue << ")";
                throw std::runtime_error(oss.str());
            }
        }
    };
    if (gt_ok) {
        check_top("WAH",           run_wah(), wah_top);
        check_top("ComBit",        run_cb(),  cb_top);
        check_top("CRoaring",      run_cr(),  cr_top);
        check_top("CRoaring+Run",  run_crr(), crr_top);
        check_top("EWAH",          run_ew(),  ew_top);
        check_top("Bitset",        run_bs(),  bs_top);
        check_top("Bitset+AVX512", run_bsa(), bsa_top);
        check_top("Concise",       run_con(), con_top);
        std::cout << "[OK] all active backends match DuckDB SQL ground truth (Top-"
                  << gt.size() << ", revenue within 0.01)." << std::endl;
    }

    // -----------------------------------------------------------------------
    // 7. Print canonical Top-10.
    // -----------------------------------------------------------------------
    const std::vector<Q3Row>* canonical = nullptr;
    const char* canonical_label = "";
    if      (run_cb())  { canonical = &cb_top;  canonical_label = "ComBit"; }
    else if (run_cr())  { canonical = &cr_top;  canonical_label = "CRoaring"; }
    else if (run_crr()) { canonical = &crr_top; canonical_label = "CRoaring+Run"; }
    else if (run_wah()) { canonical = &wah_top; canonical_label = "WAH"; }
    else if (run_ew())  { canonical = &ew_top;  canonical_label = "EWAH"; }
    else if (run_bs())  { canonical = &bs_top;  canonical_label = "Bitset"; }
    else if (run_bsa()) { canonical = &bsa_top; canonical_label = "Bitset+AVX512"; }
    else if (run_con()) { canonical = &con_top; canonical_label = "Concise"; }

    if (canonical && !canonical->empty()) {
        std::cout << "\n  Q3 Top-10 (revenue DESC, o_orderdate ASC, source=" << canonical_label << "):" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  " << std::left << std::setw(12) << "l_orderkey"
                  << std::right << std::setw(16) << "revenue"
                  << "  " << std::left << std::setw(12) << "o_orderdate"
                  << std::setw(15) << "o_shippriority" << std::endl;
        for (auto& r : *canonical) {
            // orderdate stored as (date - epoch) days; render as YYYY-MM-DD via DuckDB Date.
            date_t dt = Date::EpochDaysToDate(r.orderdate_epoch);
            std::cout << "  " << std::left << std::setw(12) << r.orderkey
                      << std::right << std::setw(16) << (static_cast<double>(r.revenue_fp) / 10000.0)
                      << "  " << std::left << std::setw(12) << Date::ToString(dt)
                      << std::setw(15) << r.shippriority << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // 8. Statistics tables + CSV (mirrors Q10 layout).
    // -----------------------------------------------------------------------
    auto cb_or_s=q3_stats(cb_or),  cb_not_s=q3_stats(cb_not),  cb_and_s=q3_stats(cb_and),  cb_agg_s=q3_stats(cb_agg),  cb_tot_s=q3_stats(cb_tot);
    auto cr_or_s=q3_stats(cr_or),  cr_not_s=q3_stats(cr_not),  cr_and_s=q3_stats(cr_and),  cr_agg_s=q3_stats(cr_agg),  cr_tot_s=q3_stats(cr_tot);
    auto crr_or_s=q3_stats(crr_or),crr_not_s=q3_stats(crr_not),crr_and_s=q3_stats(crr_and),crr_agg_s=q3_stats(crr_agg),crr_tot_s=q3_stats(crr_tot);
    auto wah_or_s=q3_stats(wah_or),wah_not_s=q3_stats(wah_not),wah_and_s=q3_stats(wah_and),wah_agg_s=q3_stats(wah_agg),wah_tot_s=q3_stats(wah_tot);
    auto ew_or_s=q3_stats(ew_or),  ew_not_s=q3_stats(ew_not),  ew_and_s=q3_stats(ew_and),  ew_agg_s=q3_stats(ew_agg),  ew_tot_s=q3_stats(ew_tot);
    auto bs_or_s=q3_stats(bs_or),  bs_not_s=q3_stats(bs_not),  bs_and_s=q3_stats(bs_and),  bs_agg_s=q3_stats(bs_agg),  bs_tot_s=q3_stats(bs_tot);
    auto bsa_or_s=q3_stats(bsa_or),bsa_not_s=q3_stats(bsa_not),bsa_and_s=q3_stats(bsa_and),bsa_agg_s=q3_stats(bsa_agg),bsa_tot_s=q3_stats(bsa_tot);
    auto con_or_s=q3_stats(con_or),con_not_s=q3_stats(con_not),con_and_s=q3_stats(con_and),con_agg_s=q3_stats(con_agg),con_tot_s=q3_stats(con_tot);

    int measured = Q3_ITERATIONS - Q3_WARMUP;

    if (run_all()) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q3 RESULTS (" << measured << " measured iterations, median +/- stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  CB (ms)         CR (ms)        CRR (ms)        WAH (ms)        EW (ms)" << std::endl;
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        auto pr = [](const char* l, Q3Stats& a, Q3Stats& b, Q3Stats& c, Q3Stats& d, Q3Stats& e) {
            std::cout << "  " << std::left << std::setw(14) << l << std::right
                << std::setw(8) << a.median << " +/- " << std::setw(5) << a.stddev
                << "  " << std::setw(8) << b.median << " +/- " << std::setw(5) << b.stddev
                << "  " << std::setw(8) << c.median << " +/- " << std::setw(5) << c.stddev
                << "  " << std::setw(8) << d.median << " +/- " << std::setw(5) << d.stddev
                << "  " << std::setw(8) << e.median << " +/- " << std::setw(5) << e.stddev << std::endl;
        };
        pr("OR_ship", cb_or_s,  cr_or_s,  crr_or_s,  wah_or_s,  ew_or_s);
        pr("NOT",     cb_not_s, cr_not_s, crr_not_s, wah_not_s, ew_not_s);
        pr("AND",     cb_and_s, cr_and_s, crr_and_s, wah_and_s, ew_and_s);
        pr("Agg",     cb_agg_s, cr_agg_s, crr_agg_s, wah_agg_s, ew_agg_s);
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        pr("TOTAL",   cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s);
        std::cout << "\n  CB rows: " << cb_rows << "  CR rows: " << cr_rows
                  << "  CRR rows: " << crr_rows << "  WAH rows: " << wah_rows
                  << "  EW rows: " << ew_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;

        // Baseline backends
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q3 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  BS (ms)         BSA (ms)        Concise (ms)     BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        auto bp = [](const char* l, Q3Stats& w, Q3Stats& b, Q3Stats& ba, Q3Stats& c) {
            double s_b = (b.median > 0) ? w.median/b.median : 0;
            double s_ba = (ba.median > 0) ? w.median/ba.median : 0;
            double s_c = (c.median > 0) ? w.median/c.median : 0;
            std::cout << "  " << std::left << std::setw(14) << l << std::right
                << std::setw(8) << b.median << " +/- " << std::setw(5) << b.stddev
                << "  " << std::setw(8) << ba.median << " +/- " << std::setw(5) << ba.stddev
                << "  " << std::setw(8) << c.median << " +/- " << std::setw(5) << c.stddev
                << "     " << std::setw(5) << s_b << "x"
                << "     " << std::setw(5) << s_ba << "x"
                << "     " << std::setw(5) << s_c << "x" << std::endl;
        };
        bp("OR_ship", wah_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
        bp("NOT",     wah_not_s, bs_not_s, bsa_not_s, con_not_s);
        bp("AND",     wah_and_s, bs_and_s, bsa_and_s, con_and_s);
        bp("Agg",     wah_agg_s, bs_agg_s, bsa_agg_s, con_agg_s);
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        bp("TOTAL",   wah_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);
        std::cout << "\n  BS rows: " << bs_rows << "  BSA rows: " << bsa_rows
                  << "  CON rows: " << con_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;
    } else {
        Q3Stats *sel_or=nullptr, *sel_not=nullptr, *sel_and=nullptr, *sel_agg=nullptr, *sel_tot=nullptr;
        size_t sel_rows = 0;
        switch (Q3_BM) {
            case Q3BmType::WAH: sel_or=&wah_or_s; sel_not=&wah_not_s; sel_and=&wah_and_s; sel_agg=&wah_agg_s; sel_tot=&wah_tot_s; sel_rows=wah_rows; break;
            case Q3BmType::CB:  sel_or=&cb_or_s;  sel_not=&cb_not_s;  sel_and=&cb_and_s;  sel_agg=&cb_agg_s;  sel_tot=&cb_tot_s;  sel_rows=cb_rows;  break;
            case Q3BmType::CR:  sel_or=&cr_or_s;  sel_not=&cr_not_s;  sel_and=&cr_and_s;  sel_agg=&cr_agg_s;  sel_tot=&cr_tot_s;  sel_rows=cr_rows;  break;
            case Q3BmType::CRR: sel_or=&crr_or_s; sel_not=&crr_not_s; sel_and=&crr_and_s; sel_agg=&crr_agg_s; sel_tot=&crr_tot_s; sel_rows=crr_rows; break;
            case Q3BmType::EW:  sel_or=&ew_or_s;  sel_not=&ew_not_s;  sel_and=&ew_and_s;  sel_agg=&ew_agg_s;  sel_tot=&ew_tot_s;  sel_rows=ew_rows;  break;
            case Q3BmType::BS:  sel_or=&bs_or_s;  sel_not=&bs_not_s;  sel_and=&bs_and_s;  sel_agg=&bs_agg_s;  sel_tot=&bs_tot_s;  sel_rows=bs_rows;  break;
            case Q3BmType::BSA: sel_or=&bsa_or_s; sel_not=&bsa_not_s; sel_and=&bsa_and_s; sel_agg=&bsa_agg_s; sel_tot=&bsa_tot_s; sel_rows=bsa_rows; break;
            case Q3BmType::CON: sel_or=&con_or_s; sel_not=&con_not_s; sel_and=&con_and_s; sel_agg=&con_agg_s; sel_tot=&con_tot_s; sel_rows=con_rows; break;
            case Q3BmType::ALL: break;
        }
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q3 RESULTS — " << q3_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;
        auto pr_one = [](const char* l, Q3Stats& s) {
            std::cout << "  " << std::left << std::setw(16) << l << std::right
                      << std::setw(9) << s.median << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val << std::setw(10) << s.max_val << std::endl;
        };
        if (sel_or) {
            pr_one("OR_ship", *sel_or);
            pr_one("NOT",     *sel_not);
            pr_one("AND",     *sel_and);
            pr_one("Agg",     *sel_agg);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            pr_one("TOTAL",   *sel_tot);
            std::cout << "\n  " << q3_label() << " rows: " << sel_rows << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }

    if (run_all()) {
        std::string sf = q3_sf_label();
        std::ofstream csv("q3_results_" + sf + ".csv");
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
                           Q3Stats& w, Q3Stats& c, Q3Stats& r, Q3Stats& rr, Q3Stats& e,
                           Q3Stats& b, Q3Stats& ba, Q3Stats& co) {
                auto sp = [](double w, double v) { return v > 0 ? w/v : 0.0; };
                csv << sf << "," << op << ","
                    << w.median << "," << w.stddev << "," << w.min_val << "," << w.max_val << ","
                    << c.median << "," << c.stddev << "," << c.min_val << "," << c.max_val << ","
                    << r.median << "," << r.stddev << "," << r.min_val << "," << r.max_val << ","
                    << rr.median << "," << rr.stddev << "," << rr.min_val << "," << rr.max_val << ","
                    << e.median << "," << e.stddev << "," << e.min_val << "," << e.max_val << ","
                    << b.median << "," << b.stddev << "," << b.min_val << "," << b.max_val << ","
                    << ba.median << "," << ba.stddev << "," << ba.min_val << "," << ba.max_val << ","
                    << co.median << "," << co.stddev << "," << co.min_val << "," << co.max_val << ","
                    << sp(w.median, c.median) << "," << sp(w.median, r.median) << ","
                    << sp(w.median, rr.median) << "," << sp(w.median, e.median) << ","
                    << sp(w.median, b.median) << "," << sp(w.median, ba.median) << ","
                    << sp(w.median, co.median) << "\n";
            };
            row("OR_ship", wah_or_s,  cb_or_s,  cr_or_s,  crr_or_s,  ew_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
            row("NOT",     wah_not_s, cb_not_s, cr_not_s, crr_not_s, ew_not_s, bs_not_s, bsa_not_s, con_not_s);
            row("AND",     wah_and_s, cb_and_s, cr_and_s, crr_and_s, ew_and_s, bs_and_s, bsa_and_s, con_and_s);
            row("Agg",     wah_agg_s, cb_agg_s, cr_agg_s, crr_agg_s, ew_agg_s, bs_agg_s, bsa_agg_s, con_agg_s);
            row("TOTAL",   wah_tot_s, cb_tot_s, cr_tot_s, crr_tot_s, ew_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);
            std::cout << "  [CSV] q3_results_" << sf << ".csv" << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
