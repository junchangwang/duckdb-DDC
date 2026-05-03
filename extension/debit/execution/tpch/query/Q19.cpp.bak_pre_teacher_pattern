// =============================================================================
// TPC-H Q19 — Discounted Revenue Query (spec v3.0.1 §2.4.19)
//
//   SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue
//     FROM lineitem, part
//    WHERE
//      ( p_partkey = l_partkey
//        AND p_brand = '[BRAND1]'
//        AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
//        AND l_quantity >= [QUANTITY1] AND l_quantity <= [QUANTITY1]+10
//        AND p_size BETWEEN 1 AND 5
//        AND l_shipmode IN ('AIR','AIR REG')
//        AND l_shipinstruct = 'DELIVER IN PERSON' )
//      OR ( ... [BRAND2], MED *, [Q2..Q2+10], size 1..10  ... )
//      OR ( ... [BRAND3], LG *,  [Q3..Q3+10], size 1..15  ... );
//
//   Validation: BRAND1=Brand#12, BRAND2=Brand#23, BRAND3=Brand#34,
//               QUANTITY1=1, QUANTITY2=10, QUANTITY3=20.
//
// Bitmap pipeline (all bitmaps aligned to lineitem.rowid):
//
//   T0 → T1  OR_qty:    q1_or = OR_many(quantity[1..11])     (11 bitmaps)
//                       q2_or = OR_many(quantity[10..20])    (11 bitmaps)
//                       q3_or = OR_many(quantity[20..30])    (11 bitmaps)
//
//   T1 → T2  Branch:    b1 = q1_or AND branch1_part
//                       b2 = q2_or AND branch2_part
//                       b3 = q3_or AND branch3_part
//                       branches = b1 OR b2 OR b3
//
//   T2 → T3  Final AND: filter = branches AND shipmode_air
//                                          AND shipinstruct_dip
//
//   T3 → T4  Agg:       walk filter rows → revenue += pp[r] * (100 - dp[r])
//
// Pre-joined per-lineitem bitmaps generated at export time:
//
//   tpch_q19_<fmt>/branch1_part/0.bm   l_partkey -> p with
//                                       brand=Brand#12 +
//                                       container IN (SM CASE/BOX/PACK/PKG) +
//                                       p_size 1..5
//   tpch_q19_<fmt>/branch2_part/0.bm   ... brand=Brand#23 + MED * + size 1..10
//   tpch_q19_<fmt>/branch3_part/0.bm   ... brand=Brand#34 + LG  * + size 1..15
//   tpch_q19_<fmt>/shipmode_air/0.bm        l_shipmode IN ('AIR', 'AIR REG')
//   tpch_q19_<fmt>/shipinstruct_dip/0.bm    l_shipinstruct = 'DELIVER IN PERSON'
//
// Quantity bitmaps are reused (via symlink) from Q6's exports:
//   tpch_q19_<fmt>/quantity/{1..50}.bm  -> tpch_q6_<fmt>/quantity/{1..50}.bm
//
// l_extendedprice / l_discount are read from DuckDB storage at query
// time (mirrors Q1/Q14/Q15/Q17 pre-load).
//
// Output: single double `revenue` (≈ $30 M at SF10; spec sample SF1 = $3,083,843).
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

// --- Q19 bitmap directories ---
static const std::string Q19_SF      = bm_bench::sf_suffix();
static const std::string Q19_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q19" + Q19_SF + "_combit");
static const std::string Q19_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q19" + Q19_SF + "_wah");
static const std::string Q19_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q19" + Q19_SF + "_croaring");
static const std::string Q19_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q19" + Q19_SF + "_ewah");

// --- Backend selection (DEBIT_BM=all|wah|cb|cr|crr|ew|bs|bsa|con) ---
using Q19BmType = bm_bench::Backend;
static const Q19BmType Q19_BM = bm_bench::parse_backend("Q19_BM");

static bool run_all() { return Q19_BM == Q19BmType::ALL; }
static bool run_wah() { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::WAH; }
static bool run_cb()  { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::CB;  }
static bool run_cr()  { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::CR;  }
static bool run_crr() { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::CRR; }
static bool run_ew()  { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::EW;  }
static bool run_bs()  { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::BS;  }
static bool run_bsa() { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::BSA; }
static bool run_con() { return Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::CON; }

static const char* q19_bm_label()     { return bm_bench::backend_label(Q19_BM); }
static std::string q19_get_sf_label() { return bm_bench::sf_label(); }

// --- Q19 predicate constants (spec §2.4.19, validation parameters) ---
// l_quantity ranges [Q, Q+10] for Q in {1, 10, 20} → covers values 1..30.
// We use equality bitmaps for each l_quantity value; the three OR_many
// merges build q1_or, q2_or, q3_or at query time (matches Q6's design,
// avoids materialising redundant range-encoded bitmaps).
static const int Q19_QTY_RANGES[3][2] = {
    { 1, 11},   // BRAND1: l_quantity in [1, 11]   (=  1..1+10)
    {10, 20},   // BRAND2: l_quantity in [10, 20]  (= 10..10+10)
    {20, 30},   // BRAND3: l_quantity in [20, 30]  (= 20..20+10)
};
static const int Q19_QTY_MIN = 1;   // smallest value referenced by any range
static const int Q19_QTY_MAX = 30;  // largest  value referenced by any range

// --- Iterations ---
static const int Q19_ITERATIONS = bm_bench::iter_count(10);
static const int Q19_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q19_once_flag;

// Bitmap loaders (same set as Q14/Q15/Q17).
static ComBit q19_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q19_load_cr(const std::string& p) {
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
static ibis::bitvector q19_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q19_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// Byte-LUT for MSB-first bit extraction (same convention as Q1/Q14/Q15/Q17).
// Per-row revenue contribution (×10000 fixed-point):
//   pp[r] is l_extendedprice ×100 (cents);
//   dp[r] is l_discount      ×100 (percentage points);
//   contribution = pp * (100 - dp)  -- units: cents × percentage_remaining
//   revenue dollars = sum / 10000.
#define Q19_REV_CONTRIB(pp, dp, r) \
    ((pp)[(r)] * (100 - (dp)[(r)]))

struct Q19Agg {
    int64_t sum_rev_x10000 = 0;
    double  revenue() const {
        return static_cast<double>(sum_rev_x10000) / 10000.0;
    }
};

// =============================================================================
// BMTPCH_Q19 — main benchmark entry point
// =============================================================================
void BMTableScan::BMTPCH_Q19(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q19_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    bm_bench::warn_if_sf1();

    // ============================================================
    // 0. Banner
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q19 Benchmark — 8 backends ("
                  << q19_get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q19 Benchmark — " << q19_bm_label() << " only ("
                  << q19_get_sf_label() << ")" << std::endl;
    }
    std::cout << "  3-branch OR over (brand × container × p_size × l_quantity range)" << std::endl;
    std::cout << "        AND with l_shipmode IN ('AIR','AIR REG')" << std::endl;
    std::cout << "        AND with l_shipinstruct = 'DELIVER IN PERSON'" << std::endl;
    std::cout << "  TPC-H params: BRAND1=Brand#12 (Q1=1..11),"
                 " BRAND2=Brand#23 (Q2=10..20),"
                 " BRAND3=Brand#34 (Q3=20..30)" << std::endl;
    std::cout << "  Bitmap dirs:";
    if (run_cb())              std::cout << " " << Q19_CB_DIR;
    if (run_cr() || run_crr()) std::cout << " " << Q19_CR_DIR;
    if (run_wah())             std::cout << " " << Q19_WAH_DIR;
    if (run_ew())              std::cout << " " << Q19_EW_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << Q19_ITERATIONS
              << " (first " << Q19_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 1. Pre-load lineitem.l_extendedprice / l_discount
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

    size_t num_rows = 0;
    {
        std::ifstream meta(Q19_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
        }
    }
    if (num_rows == 0) {
        std::cerr << "Error: cannot read num_rows from " << Q19_CB_DIR << "/done.txt" << std::endl;
        return;
    }

    std::cout << "\n[Pre-load] Loading " << num_rows
              << " rows (price, discount) ..." << std::endl;
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
    // 1.5 Build branch[123]_part bitmaps dynamically (TPC-H 1.5.7 compliance).
    //     Each branch query is single-table on `part`.  Result is a set of
    //     qualifying part keys; we walk lineitem.l_partkey + FK lookup to
    //     materialize per-lineitem positions.  No multi-table aux on disk.
    // ============================================================
    std::vector<uint32_t> b1_pos, b2_pos, b3_pos;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::unordered_set<int64_t> b1_pk, b2_pk, b3_pk;
        Connection con(*context.client.db);
        auto run_branch = [&](const char* sql, std::unordered_set<int64_t>& out) {
            auto r = con.Query(sql);
            if (r && !r->HasError()) {
                out.reserve(r->RowCount());
                for (idx_t i = 0; i < r->RowCount(); i++)
                    out.insert(r->GetValue(0, i).GetValue<int64_t>());
            }
        };
        run_branch(
            "SELECT p_partkey FROM part WHERE p_brand='Brand#12' "
            "AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG') "
            "AND p_size BETWEEN 1 AND 5", b1_pk);
        run_branch(
            "SELECT p_partkey FROM part WHERE p_brand='Brand#23' "
            "AND p_container IN ('MED BAG','MED BOX','MED PKG','MED PACK') "
            "AND p_size BETWEEN 1 AND 10", b2_pk);
        run_branch(
            "SELECT p_partkey FROM part WHERE p_brand='Brand#34' "
            "AND p_container IN ('LG CASE','LG BOX','LG PACK','LG PKG') "
            "AND p_size BETWEEN 1 AND 15", b3_pk);

        // Load lineitem.l_partkey, walk once, FK-lookup all 3 branches.
        std::vector<int64_t> li_pkey(num_rows);
        {
            TableScanState st2;
            vector<StorageIndex> cols2{ StorageIndex(1) };
            lineitem_table.GetStorage().InitializeScan(
                context.client, lineitem_transaction, st2, cols2);
            vector<LogicalType> types2{ lineitem_table.GetColumns().GetColumnTypes()[1] };
            size_t off = 0;
            while (true) {
                DataChunk ch; ch.Initialize(context.client, types2);
                lineitem_table.GetStorage().Scan(lineitem_transaction, ch, st2);
                if (ch.size() == 0) break;
                std::memcpy(li_pkey.data() + off,
                            FlatVector::GetData<int64_t>(ch.data[0]),
                            ch.size() * 8);
                off += ch.size();
            }
        }
        b1_pos.reserve(num_rows / 100);
        b2_pos.reserve(num_rows / 100);
        b3_pos.reserve(num_rows / 100);
        for (size_t i = 0; i < num_rows; i++) {
            int64_t pk = li_pkey[i];
            if (b1_pk.count(pk)) b1_pos.push_back(static_cast<uint32_t>(i));
            else if (b2_pk.count(pk)) b2_pos.push_back(static_cast<uint32_t>(i));
            else if (b3_pk.count(pk)) b3_pos.push_back(static_cast<uint32_t>(i));
        }
        std::cout << "[Build branch_part] b1=" << b1_pos.size()
                  << " b2=" << b2_pos.size() << " b3=" << b3_pos.size()
                  << " in " << ms(t0, std::chrono::high_resolution_clock::now()) << " ms" << std::endl;
    }

    // ============================================================
    // 2. Load bitmaps:
    //      5 own bitmaps + Q19_QTY_MAX equality bitmaps for quantity.
    // ============================================================
    std::cout << "\n[Load] Loading bitmaps (mode=" << q19_bm_label() << ")..." << std::endl;

    const int n_qty = Q19_QTY_MAX - Q19_QTY_MIN + 1;  // 30 bitmaps

    // ---- ComBit ----
    std::vector<ComBit> cb_qty;
    ComBit cb_b1, cb_b2, cb_b3, cb_smair, cb_sidip;
    std::vector<const ComBit*> cb_q1_ptrs, cb_q2_ptrs, cb_q3_ptrs;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_qty.resize(Q19_QTY_MAX + 1);
        for (int q = Q19_QTY_MIN; q <= Q19_QTY_MAX; q++)
            cb_qty[q] = q19_load_cb(Q19_CB_DIR + "/quantity/" + std::to_string(q) + ".bm");
        // Branch bitmaps built in-memory from FK-lookup positions (L2 compliance).
        {
            std::vector<bool> jm(num_rows, false);
            for (uint32_t p : b1_pos) jm[p] = true; cb_b1 = ComBit::compress(jm, false);
            std::fill(jm.begin(), jm.end(), false);
            for (uint32_t p : b2_pos) jm[p] = true; cb_b2 = ComBit::compress(jm, false);
            std::fill(jm.begin(), jm.end(), false);
            for (uint32_t p : b3_pos) jm[p] = true; cb_b3 = ComBit::compress(jm, false);
        }
        cb_smair  = q19_load_cb(Q19_CB_DIR + "/shipmode_air/0.bm");
        cb_sidip  = q19_load_cb(Q19_CB_DIR + "/shipinstruct_dip/0.bm");
        for (int q = Q19_QTY_RANGES[0][0]; q <= Q19_QTY_RANGES[0][1]; q++) cb_q1_ptrs.push_back(&cb_qty[q]);
        for (int q = Q19_QTY_RANGES[1][0]; q <= Q19_QTY_RANGES[1][1]; q++) cb_q2_ptrs.push_back(&cb_qty[q]);
        for (int q = Q19_QTY_RANGES[2][0]; q <= Q19_QTY_RANGES[2][1]; q++) cb_q3_ptrs.push_back(&cb_qty[q]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // ---- CRoaring (vanilla) ----
    std::vector<roaring::Roaring> cr_qty;
    roaring::Roaring cr_b1, cr_b2, cr_b3, cr_smair, cr_sidip;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_qty.resize(Q19_QTY_MAX + 1);
        for (int q = Q19_QTY_MIN; q <= Q19_QTY_MAX; q++)
            cr_qty[q] = q19_load_cr(Q19_CR_DIR + "/quantity/" + std::to_string(q) + ".bm");
        if (!b1_pos.empty()) cr_b1.addMany(b1_pos.size(), b1_pos.data());
        if (!b2_pos.empty()) cr_b2.addMany(b2_pos.size(), b2_pos.data());
        if (!b3_pos.empty()) cr_b3.addMany(b3_pos.size(), b3_pos.data());
        cr_smair  = q19_load_cr(Q19_CR_DIR + "/shipmode_air/0.bm");
        cr_sidip  = q19_load_cr(Q19_CR_DIR + "/shipinstruct_dip/0.bm");
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // ---- CRoaring + Run (loads fresh + runOptimize) ----
    std::vector<roaring::Roaring> crr_qty;
    roaring::Roaring crr_b1, crr_b2, crr_b3, crr_smair, crr_sidip;
    std::vector<const roaring::Roaring*> crr_q1_ptrs, crr_q2_ptrs, crr_q3_ptrs;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_qty.resize(Q19_QTY_MAX + 1);
        for (int q = Q19_QTY_MIN; q <= Q19_QTY_MAX; q++) {
            crr_qty[q] = q19_load_cr(Q19_CR_DIR + "/quantity/" + std::to_string(q) + ".bm");
            crr_qty[q].runOptimize();
        }
        if (!b1_pos.empty()) crr_b1.addMany(b1_pos.size(), b1_pos.data()); crr_b1.runOptimize();
        if (!b2_pos.empty()) crr_b2.addMany(b2_pos.size(), b2_pos.data()); crr_b2.runOptimize();
        if (!b3_pos.empty()) crr_b3.addMany(b3_pos.size(), b3_pos.data()); crr_b3.runOptimize();
        crr_smair  = q19_load_cr(Q19_CR_DIR + "/shipmode_air/0.bm");     crr_smair.runOptimize();
        crr_sidip  = q19_load_cr(Q19_CR_DIR + "/shipinstruct_dip/0.bm"); crr_sidip.runOptimize();
        for (int q = Q19_QTY_RANGES[0][0]; q <= Q19_QTY_RANGES[0][1]; q++) crr_q1_ptrs.push_back(&crr_qty[q]);
        for (int q = Q19_QTY_RANGES[1][0]; q <= Q19_QTY_RANGES[1][1]; q++) crr_q2_ptrs.push_back(&crr_qty[q]);
        for (int q = Q19_QTY_RANGES[2][0]; q <= Q19_QTY_RANGES[2][1]; q++) crr_q3_ptrs.push_back(&crr_qty[q]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // ---- WAH ----
    std::vector<ibis::bitvector> wah_qty;
    ibis::bitvector wah_b1, wah_b2, wah_b3, wah_smair, wah_sidip;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wah_qty.resize(Q19_QTY_MAX + 1);
        for (int q = Q19_QTY_MIN; q <= Q19_QTY_MAX; q++)
            wah_qty[q] = q19_load_wah(Q19_WAH_DIR + "/quantity/" + std::to_string(q) + ".bm");
        // WAH branches via sequential bit append.
        auto wah_build = [&](ibis::bitvector& w, const std::vector<uint32_t>& pos) {
            size_t k = 0;
            for (size_t i = 0; i < num_rows; i++) {
                bool b = (k < pos.size() && pos[k] == i);
                if (b) k++;
                w += (b ? 1 : 0);
            }
            w.compress();
        };
        wah_build(wah_b1, b1_pos);
        wah_build(wah_b2, b2_pos);
        wah_build(wah_b3, b3_pos);
        wah_smair  = q19_load_wah(Q19_WAH_DIR + "/shipmode_air/0.bm");
        wah_sidip  = q19_load_wah(Q19_WAH_DIR + "/shipinstruct_dip/0.bm");
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // ---- EWAH ----
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_qty;
    ewah::EWAHBoolArray<uint64_t> ew_b1, ew_b2, ew_b3, ew_smair, ew_sidip;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_q1_ptrs, ew_q2_ptrs, ew_q3_ptrs;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ew_qty.resize(Q19_QTY_MAX + 1);
        for (int q = Q19_QTY_MIN; q <= Q19_QTY_MAX; q++)
            ew_qty[q] = q19_load_ew(Q19_EW_DIR + "/quantity/" + std::to_string(q) + ".bm");
        auto ew_build = [&](ewah::EWAHBoolArray<uint64_t>& e, const std::vector<uint32_t>& pos) {
            for (uint32_t p : pos) e.set(p);
            if (e.sizeInBits() < num_rows) e.padWithZeroes(num_rows);
        };
        ew_build(ew_b1, b1_pos);
        ew_build(ew_b2, b2_pos);
        ew_build(ew_b3, b3_pos);
        ew_smair  = q19_load_ew(Q19_EW_DIR + "/shipmode_air/0.bm");
        ew_sidip  = q19_load_ew(Q19_EW_DIR + "/shipinstruct_dip/0.bm");
        for (int q = Q19_QTY_RANGES[0][0]; q <= Q19_QTY_RANGES[0][1]; q++) ew_q1_ptrs.push_back(&ew_qty[q]);
        for (int q = Q19_QTY_RANGES[1][0]; q <= Q19_QTY_RANGES[1][1]; q++) ew_q2_ptrs.push_back(&ew_qty[q]);
        for (int q = Q19_QTY_RANGES[2][0]; q <= Q19_QTY_RANGES[2][1]; q++) ew_q3_ptrs.push_back(&ew_qty[q]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // ---- Bitset (BS / BSA share) ----
    std::vector<bs::Bitmap> bs_qty;
    bs::Bitmap bs_b1, bs_b2, bs_b3, bs_smair, bs_sidip;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_qty.resize(Q19_QTY_MAX + 1);
        for (int q = Q19_QTY_MIN; q <= Q19_QTY_MAX; q++)
            bs_qty[q] = bm_bench::load_bitmap_from_croaring(
                Q19_CR_DIR + "/quantity/" + std::to_string(q) + ".bm");
        auto bs_build = [&](bs::Bitmap& b, const std::vector<uint32_t>& pos) {
            b.alloc_for_bits(num_rows);
            for (uint32_t p : pos) b.words[p / 64] |= uint64_t(1) << (p % 64);
        };
        bs_build(bs_b1, b1_pos);
        bs_build(bs_b2, b2_pos);
        bs_build(bs_b3, b3_pos);
        bs_smair  = bm_bench::load_bitmap_from_croaring(Q19_CR_DIR + "/shipmode_air/0.bm");
        bs_sidip  = bm_bench::load_bitmap_from_croaring(Q19_CR_DIR + "/shipinstruct_dip/0.bm");
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // ---- Concise ----
    std::vector<ConciseSet<false>> con_qty;
    ConciseSet<false> con_b1, con_b2, con_b3, con_smair, con_sidip;
    std::vector<const ConciseSet<false>*> con_q1_ptrs, con_q2_ptrs, con_q3_ptrs;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_qty.resize(Q19_QTY_MAX + 1);
        for (int q = Q19_QTY_MIN; q <= Q19_QTY_MAX; q++)
            con_qty[q] = bm_bench::load_concise_from_croaring(
                Q19_CR_DIR + "/quantity/" + std::to_string(q) + ".bm");
        for (uint32_t p : b1_pos) con_b1.add(p);
        for (uint32_t p : b2_pos) con_b2.add(p);
        for (uint32_t p : b3_pos) con_b3.add(p);
        con_smair  = bm_bench::load_concise_from_croaring(Q19_CR_DIR + "/shipmode_air/0.bm");
        con_sidip  = bm_bench::load_concise_from_croaring(Q19_CR_DIR + "/shipinstruct_dip/0.bm");
        for (int q = Q19_QTY_RANGES[0][0]; q <= Q19_QTY_RANGES[0][1]; q++) con_q1_ptrs.push_back(&con_qty[q]);
        for (int q = Q19_QTY_RANGES[1][0]; q <= Q19_QTY_RANGES[1][1]; q++) con_q2_ptrs.push_back(&con_qty[q]);
        for (int q = Q19_QTY_RANGES[2][0]; q <= Q19_QTY_RANGES[2][1]; q++) con_q3_ptrs.push_back(&con_qty[q]);
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
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q19_WAH_DIR)) << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q19_CB_DIR))  << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q19_CR_DIR))  << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q19_EW_DIR))  << " MiB" << std::endl;

    // ============================================================
    // 3. Per-iteration timing buffers and final aggregate per backend
    // ============================================================
    std::vector<double> cb_or_t,  cb_and_t,  cb_agg_t,  cb_tot_t;
    std::vector<double> cr_or_t,  cr_and_t,  cr_agg_t,  cr_tot_t;
    std::vector<double> crr_or_t, crr_and_t, crr_agg_t, crr_tot_t;
    std::vector<double> wah_or_t, wah_and_t, wah_agg_t, wah_tot_t;
    std::vector<double> ew_or_t,  ew_and_t,  ew_agg_t,  ew_tot_t;
    std::vector<double> bs_or_t,  bs_and_t,  bs_agg_t,  bs_tot_t;
    std::vector<double> bsa_or_t, bsa_and_t, bsa_agg_t, bsa_tot_t;
    std::vector<double> con_or_t, con_and_t, con_agg_t, con_tot_t;

    Q19Agg cb_agg{}, cr_agg{}, crr_agg{}, wah_agg{}, ew_agg{};
    Q19Agg bs_agg{}, bsa_agg{}, con_agg{};

    // Phases reported per iteration:
    //   OR  : T0 → T1   build q1_or, q2_or, q3_or via OR_many (33 bitmaps)
    //   AND : T1 → T2   per-branch AND + branch union + final shipmode/instruct AND
    //   Agg : T2 → T3   walk filter rows, accumulate revenue
    //   Total = T0 → T3

    for (int iter = 0; iter < Q19_ITERATIONS; iter++) {
        bool warmup = (iter < Q19_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q19_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ================= ComBit =================
        // Pipeline: 3 OR_many (q1_or, q2_or, q3_or — Decompressed) →
        //           3 in-place &= against the (Compressed-from-disk)
        //           branchN_part bitmaps (canonical Q12/Q14 pattern:
        //           &= asserts LHS Decompressed and runs the AVX-512
        //           VBMI2 fast path) → pairwise |= to union the three
        //           branches → 2 more &= for shipmode_air and
        //           shipinstruct_dip.  Walk via seg.l1_literal_data()
        //           since the final filter is Decompressed (post AND).
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ComBit q1_or = ComBit::OR_many(cb_q1_ptrs.size(), cb_q1_ptrs.data());
            ComBit q2_or = ComBit::OR_many(cb_q2_ptrs.size(), cb_q2_ptrs.data());
            ComBit q3_or = ComBit::OR_many(cb_q3_ptrs.size(), cb_q3_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            // Per-branch AND: q*_or is Decompressed, branch*_part is
            // Compressed (loaded straight from disk).  &= accepts a
            // Compressed RHS and keeps the LHS Decompressed.
            ComBit b1 = q1_or; b1 &= cb_b1;
            ComBit b2 = q2_or; b2 &= cb_b2;
            ComBit b3 = q3_or; b3 &= cb_b3;
            // Branch union: |= keeps LHS Decompressed.
            ComBit filter = b1;
            filter |= b2;
            filter |= b3;
            // Final shared lineitem-side AND.
            filter &= cb_smair;
            filter &= cb_sidip;
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            {
                size_t row_base = 0;
                for (size_t s = 0; s < filter.num_segments(); s++) {
                    const auto& seg = filter.segment(s);
                    const uint8_t* data = seg.l1_literal_data();
                    size_t n = seg.num_literals();
                    for (size_t bi = 0; bi < n; bi++) {
                        uint8_t b = data[bi];
                        if (b == 0) { row_base += 8; continue; }
                        const auto& entry = bm_bench::byte_lut[b];
                        for (int k = 0; k < entry.count; k++)
                            sum_rev += Q19_REV_CONTRIB(pp, dp, row_base + entry.pos[k]);
                        row_base += 8;
                    }
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            cb_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CB:   OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << cb_agg.revenue() << std::endl;
            if (!warmup) {
                cb_or_t.push_back(d_or); cb_and_t.push_back(d_and);
                cb_agg_t.push_back(d_agg); cb_tot_t.push_back(d_total);
            }
        }

        // ================= CRoaring (vanilla pairwise OR) =================
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            roaring::Roaring q1_or = cr_qty[Q19_QTY_RANGES[0][0]];
            for (int q = Q19_QTY_RANGES[0][0] + 1; q <= Q19_QTY_RANGES[0][1]; q++) q1_or |= cr_qty[q];
            roaring::Roaring q2_or = cr_qty[Q19_QTY_RANGES[1][0]];
            for (int q = Q19_QTY_RANGES[1][0] + 1; q <= Q19_QTY_RANGES[1][1]; q++) q2_or |= cr_qty[q];
            roaring::Roaring q3_or = cr_qty[Q19_QTY_RANGES[2][0]];
            for (int q = Q19_QTY_RANGES[2][0] + 1; q <= Q19_QTY_RANGES[2][1]; q++) q3_or |= cr_qty[q];
            auto t1 = std::chrono::high_resolution_clock::now();

            roaring::Roaring b1 = q1_or & cr_b1;
            roaring::Roaring b2 = q2_or & cr_b2;
            roaring::Roaring b3 = q3_or & cr_b3;
            roaring::Roaring filter = b1 | b2 | b3;
            filter &= cr_smair;
            filter &= cr_sidip;
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            for (auto it = filter.begin(); it != filter.end(); ++it)
                sum_rev += Q19_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            cr_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CR:   OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << cr_agg.revenue() << std::endl;
            if (!warmup) {
                cr_or_t.push_back(d_or); cr_and_t.push_back(d_and);
                cr_agg_t.push_back(d_agg); cr_tot_t.push_back(d_total);
            }
        }

        // ================= CRoaring + Run (fastunion) =================
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            roaring::Roaring q1_or = roaring::Roaring::fastunion(crr_q1_ptrs.size(), crr_q1_ptrs.data());
            roaring::Roaring q2_or = roaring::Roaring::fastunion(crr_q2_ptrs.size(), crr_q2_ptrs.data());
            roaring::Roaring q3_or = roaring::Roaring::fastunion(crr_q3_ptrs.size(), crr_q3_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            roaring::Roaring b1 = q1_or & crr_b1;
            roaring::Roaring b2 = q2_or & crr_b2;
            roaring::Roaring b3 = q3_or & crr_b3;
            roaring::Roaring filter = b1 | b2 | b3;
            filter &= crr_smair;
            filter &= crr_sidip;
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            for (auto it = filter.begin(); it != filter.end(); ++it)
                sum_rev += Q19_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            crr_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CRR:  OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << crr_agg.revenue() << std::endl;
            if (!warmup) {
                crr_or_t.push_back(d_or); crr_and_t.push_back(d_and);
                crr_agg_t.push_back(d_agg); crr_tot_t.push_back(d_total);
            }
        }

        // ================= WAH =================
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ibis::bitvector q1_or = wah_qty[Q19_QTY_RANGES[0][0]];
            q1_or.decompress();
            for (int q = Q19_QTY_RANGES[0][0] + 1; q <= Q19_QTY_RANGES[0][1]; q++) q1_or |= wah_qty[q];
            ibis::bitvector q2_or = wah_qty[Q19_QTY_RANGES[1][0]];
            q2_or.decompress();
            for (int q = Q19_QTY_RANGES[1][0] + 1; q <= Q19_QTY_RANGES[1][1]; q++) q2_or |= wah_qty[q];
            ibis::bitvector q3_or = wah_qty[Q19_QTY_RANGES[2][0]];
            q3_or.decompress();
            for (int q = Q19_QTY_RANGES[2][0] + 1; q <= Q19_QTY_RANGES[2][1]; q++) q3_or |= wah_qty[q];
            auto t1 = std::chrono::high_resolution_clock::now();

            // WAH: clone the Decompressed q*_or, AND with branch_part,
            // OR-union the branches, AND with shipmode/instruct.
            ibis::bitvector b1; b1.copy(q1_or); b1 &= wah_b1;
            ibis::bitvector b2; b2.copy(q2_or); b2 &= wah_b2;
            ibis::bitvector b3; b3.copy(q3_or); b3 &= wah_b3;
            ibis::bitvector filter; filter.copy(b1);
            filter |= b2;
            filter |= b3;
            filter &= wah_smair;
            filter &= wah_sidip;
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            {
                ibis::bitvector::pit pit(filter);
                while (*pit != 0xFFFFFFFFU) {
                    sum_rev += Q19_REV_CONTRIB(pp, dp, *pit);
                    pit.next();
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            wah_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  WAH:  OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << wah_agg.revenue() << std::endl;
            if (!warmup) {
                wah_or_t.push_back(d_or); wah_and_t.push_back(d_and);
                wah_agg_t.push_back(d_agg); wah_tot_t.push_back(d_total);
            }
        }

        // ================= EWAH (fast_logicalor) =================
        if (run_ew()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ewah::EWAHBoolArray<uint64_t> q1_or = ewah::fast_logicalor(ew_q1_ptrs.size(), ew_q1_ptrs.data());
            ewah::EWAHBoolArray<uint64_t> q2_or = ewah::fast_logicalor(ew_q2_ptrs.size(), ew_q2_ptrs.data());
            ewah::EWAHBoolArray<uint64_t> q3_or = ewah::fast_logicalor(ew_q3_ptrs.size(), ew_q3_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ewah::EWAHBoolArray<uint64_t> b1, b2, b3, u12, u123, f12, filter;
            q1_or.logicaland(ew_b1, b1);
            q2_or.logicaland(ew_b2, b2);
            q3_or.logicaland(ew_b3, b3);
            b1.logicalor(b2, u12);
            u12.logicalor(b3, u123);
            u123.logicaland(ew_smair, f12);
            f12.logicaland(ew_sidip, filter);
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            for (auto it = filter.begin(); it != filter.end(); ++it)
                sum_rev += Q19_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            ew_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  EW:   OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << ew_agg.revenue() << std::endl;
            if (!warmup) {
                ew_or_t.push_back(d_or); ew_and_t.push_back(d_and);
                ew_agg_t.push_back(d_agg); ew_tot_t.push_back(d_total);
            }
        }

        // ================= Bitset (scalar) =================
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap q1_or = bs_qty[Q19_QTY_RANGES[0][0]].clone();
            for (int q = Q19_QTY_RANGES[0][0] + 1; q <= Q19_QTY_RANGES[0][1]; q++)
                bs::or_inplace(q1_or, bs_qty[q], false);
            bs::Bitmap q2_or = bs_qty[Q19_QTY_RANGES[1][0]].clone();
            for (int q = Q19_QTY_RANGES[1][0] + 1; q <= Q19_QTY_RANGES[1][1]; q++)
                bs::or_inplace(q2_or, bs_qty[q], false);
            bs::Bitmap q3_or = bs_qty[Q19_QTY_RANGES[2][0]].clone();
            for (int q = Q19_QTY_RANGES[2][0] + 1; q <= Q19_QTY_RANGES[2][1]; q++)
                bs::or_inplace(q3_or, bs_qty[q], false);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::Bitmap b1 = q1_or.clone(); bs::and_inplace(b1, bs_b1, false);
            bs::Bitmap b2 = q2_or.clone(); bs::and_inplace(b2, bs_b2, false);
            bs::Bitmap b3 = q3_or.clone(); bs::and_inplace(b3, bs_b3, false);
            bs::Bitmap filter = b1.clone();
            bs::or_inplace(filter, b2, false);
            bs::or_inplace(filter, b3, false);
            bs::and_inplace(filter, bs_smair, false);
            bs::and_inplace(filter, bs_sidip, false);
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            for (size_t i = 0; i < filter.nwords; ++i) {
                uint64_t w = filter.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= filter.nbits) break;
                    sum_rev += Q19_REV_CONTRIB(pp, dp, r);
                    w &= w - 1;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            bs_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  BS:   OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << bs_agg.revenue() << std::endl;
            if (!warmup) {
                bs_or_t.push_back(d_or); bs_and_t.push_back(d_and);
                bs_agg_t.push_back(d_agg); bs_tot_t.push_back(d_total);
            }
        }

        // ================= Bitset + AVX-512 =================
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap q1_or = bs_qty[Q19_QTY_RANGES[0][0]].clone();
            for (int q = Q19_QTY_RANGES[0][0] + 1; q <= Q19_QTY_RANGES[0][1]; q++)
                bs::or_inplace(q1_or, bs_qty[q], true);
            bs::Bitmap q2_or = bs_qty[Q19_QTY_RANGES[1][0]].clone();
            for (int q = Q19_QTY_RANGES[1][0] + 1; q <= Q19_QTY_RANGES[1][1]; q++)
                bs::or_inplace(q2_or, bs_qty[q], true);
            bs::Bitmap q3_or = bs_qty[Q19_QTY_RANGES[2][0]].clone();
            for (int q = Q19_QTY_RANGES[2][0] + 1; q <= Q19_QTY_RANGES[2][1]; q++)
                bs::or_inplace(q3_or, bs_qty[q], true);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::Bitmap b1 = q1_or.clone(); bs::and_inplace(b1, bs_b1, true);
            bs::Bitmap b2 = q2_or.clone(); bs::and_inplace(b2, bs_b2, true);
            bs::Bitmap b3 = q3_or.clone(); bs::and_inplace(b3, bs_b3, true);
            bs::Bitmap filter = b1.clone();
            bs::or_inplace(filter, b2, true);
            bs::or_inplace(filter, b3, true);
            bs::and_inplace(filter, bs_smair, true);
            bs::and_inplace(filter, bs_sidip, true);
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            for (size_t i = 0; i < filter.nwords; ++i) {
                uint64_t w = filter.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t r = base + __builtin_ctzll(w);
                    if (r >= filter.nbits) break;
                    sum_rev += Q19_REV_CONTRIB(pp, dp, r);
                    w &= w - 1;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            bsa_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  BSA:  OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << bsa_agg.revenue() << std::endl;
            if (!warmup) {
                bsa_or_t.push_back(d_or); bsa_and_t.push_back(d_and);
                bsa_agg_t.push_back(d_agg); bsa_tot_t.push_back(d_total);
            }
        }

        // ================= Concise (fast_logicalor) =================
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> q1_or = ConciseSet<false>::fast_logicalor(con_q1_ptrs.size(), con_q1_ptrs.data());
            ConciseSet<false> q2_or = ConciseSet<false>::fast_logicalor(con_q2_ptrs.size(), con_q2_ptrs.data());
            ConciseSet<false> q3_or = ConciseSet<false>::fast_logicalor(con_q3_ptrs.size(), con_q3_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> b1 = q1_or.logicaland(con_b1);
            ConciseSet<false> b2 = q2_or.logicaland(con_b2);
            ConciseSet<false> b3 = q3_or.logicaland(con_b3);
            ConciseSet<false> filter = b1.logicalor(b2).logicalor(b3)
                                          .logicaland(con_smair).logicaland(con_sidip);
            auto t2 = std::chrono::high_resolution_clock::now();

            int64_t sum_rev = 0;
            for (auto it = filter.begin(); it != filter.end(); ++it)
                sum_rev += Q19_REV_CONTRIB(pp, dp, *it);
            auto t3 = std::chrono::high_resolution_clock::now();
            con_agg.sum_rev_x10000 = sum_rev;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_agg = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CON:  OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_total
                      << "  revenue=" << con_agg.revenue() << std::endl;
            if (!warmup) {
                con_or_t.push_back(d_or); con_and_t.push_back(d_and);
                con_agg_t.push_back(d_agg); con_tot_t.push_back(d_total);
            }
        }
    } // end iteration loop

    // ============================================================
    // 5. Cross-backend consistency check
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q19 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;

    const Q19Agg* print_src = nullptr;
    const char* print_label = "";
    if      (Q19_BM == Q19BmType::ALL || Q19_BM == Q19BmType::CB)  { print_src = &cb_agg;  print_label = "ComBit"; }
    else if (Q19_BM == Q19BmType::WAH)                             { print_src = &wah_agg; print_label = "WAH"; }
    else if (Q19_BM == Q19BmType::CR)                              { print_src = &cr_agg;  print_label = "CRoaring"; }
    else if (Q19_BM == Q19BmType::CRR)                             { print_src = &crr_agg; print_label = "CRoaring+Run"; }
    else if (Q19_BM == Q19BmType::EW)                              { print_src = &ew_agg;  print_label = "EWAH"; }
    else if (Q19_BM == Q19BmType::BS)                              { print_src = &bs_agg;  print_label = "Bitset"; }
    else if (Q19_BM == Q19BmType::BSA)                             { print_src = &bsa_agg; print_label = "Bitset+AVX512"; }
    else if (Q19_BM == Q19BmType::CON)                             { print_src = &con_agg; print_label = "Concise"; }

    std::cout << std::fixed << std::setprecision(2);
    if (print_src) {
        std::cout << "  Q19 Result (" << print_label << "):" << std::endl;
        std::cout << "    revenue = " << print_src->revenue()
                  << "   (sum×10000 = " << print_src->sum_rev_x10000 << ")" << std::endl;
    }

    if (run_all()) {
        bool consistent = true;
        const Q19Agg& base = cb_agg;
        auto cmp = [&](const char* lbl, const Q19Agg& v) {
            if (v.sum_rev_x10000 != base.sum_rev_x10000) {
                std::cout << "  *** MISMATCH " << lbl
                          << " (sum=" << v.sum_rev_x10000
                          << " vs CB sum=" << base.sum_rev_x10000 << ") ***\n";
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
    // 6. DuckDB native SQL ground truth (verbatim spec §2.4.19)
    // ============================================================
    double gt_revenue = -1.0;
    double gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue "
            "  FROM lineitem, part "
            " WHERE "
            "   ( p_partkey = l_partkey "
            "     AND p_brand = 'Brand#12' "
            "     AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG') "
            "     AND l_quantity >=  1 AND l_quantity <=  1 + 10 "
            "     AND p_size BETWEEN 1 AND 5 "
            "     AND l_shipmode IN ('AIR','AIR REG') "
            "     AND l_shipinstruct = 'DELIVER IN PERSON' ) "
            " OR ( p_partkey = l_partkey "
            "      AND p_brand = 'Brand#23' "
            "      AND p_container IN ('MED BAG','MED BOX','MED PKG','MED PACK') "
            "      AND l_quantity >= 10 AND l_quantity <= 10 + 10 "
            "      AND p_size BETWEEN 1 AND 10 "
            "      AND l_shipmode IN ('AIR','AIR REG') "
            "      AND l_shipinstruct = 'DELIVER IN PERSON' ) "
            " OR ( p_partkey = l_partkey "
            "      AND p_brand = 'Brand#34' "
            "      AND p_container IN ('LG CASE','LG BOX','LG PACK','LG PKG') "
            "      AND l_quantity >= 20 AND l_quantity <= 20 + 10 "
            "      AND p_size BETWEEN 1 AND 15 "
            "      AND l_shipmode IN ('AIR','AIR REG') "
            "      AND l_shipinstruct = 'DELIVER IN PERSON' )";
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = con.Query(sql);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (result && !result->HasError() && result->RowCount() == 1) {
            gt_revenue = result->GetValue(0, 0).GetValue<double>();
            gt_sql_ms = ms(t0, t1);
        } else if (result && result->HasError()) {
            std::cerr << "[Baseline] SQL error: " << result->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    if (gt_revenue >= 0.0) {
        std::cout << "\n[Baseline] DuckDB native SQL  (single run: "
                  << std::fixed << std::setprecision(2) << gt_sql_ms << " ms)" << std::endl;
        std::cout << "  SQL ground truth:  revenue = " << gt_revenue << std::endl;

        if (print_src) {
            double bm_val = print_src->revenue();
            double diff = std::abs(bm_val - gt_revenue);
            // Tolerance 0.01 (a cent) — DuckDB uses DECIMAL while our
            // pipeline accumulates int64 ×10000; the only floating-point
            // step is the final /10000 at print time.
            if (diff <= 0.01) {
                std::cout << "[OK] all active backends match DuckDB SQL ground truth "
                          << "(revenue within " << diff << ")." << std::endl;
            } else {
                std::cerr << "[FAIL] bitmap " << bm_val << " vs SQL "
                          << gt_revenue << " differ by " << diff << std::endl;
            }
        }
    }

    // ============================================================
    // 7. Final timing summary (median ± stddev)
    // ============================================================
    auto print_stats = [&](const char* lbl, std::vector<double>& or_t,
                           std::vector<double>& and_t, std::vector<double>& agg_t,
                           std::vector<double>& tot_t) {
        if (or_t.empty()) return;
        auto so = bm_bench::compute_stats(or_t);
        auto sa = bm_bench::compute_stats(and_t);
        auto sg = bm_bench::compute_stats(agg_t);
        auto st = bm_bench::compute_stats(tot_t);
        std::cout << "  " << std::left << std::setw(5) << lbl
                  << " OR="     << std::fixed << std::setprecision(2) << std::setw(7) << so.median
                  << " +/- "    << std::setw(5) << so.stddev
                  << "  AND="   << std::setw(7) << sa.median
                  << " +/- "    << std::setw(5) << sa.stddev
                  << "  Agg="   << std::setw(7) << sg.median
                  << " +/- "    << std::setw(5) << sg.stddev
                  << "  Total=" << std::setw(7) << st.median
                  << " +/- "    << std::setw(5) << st.stddev
                  << " (ms)" << std::endl;
    };

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q19 RESULTS (" << (Q19_ITERATIONS - Q19_WARMUP)
              << " measured iterations, median +/- stddev)" << std::endl;
    std::cout << "================================================================" << std::endl;
    print_stats("CB",  cb_or_t,  cb_and_t,  cb_agg_t,  cb_tot_t);
    print_stats("CR",  cr_or_t,  cr_and_t,  cr_agg_t,  cr_tot_t);
    print_stats("CRR", crr_or_t, crr_and_t, crr_agg_t, crr_tot_t);
    print_stats("WAH", wah_or_t, wah_and_t, wah_agg_t, wah_tot_t);
    print_stats("EW",  ew_or_t,  ew_and_t,  ew_agg_t,  ew_tot_t);
    print_stats("BS",  bs_or_t,  bs_and_t,  bs_agg_t,  bs_tot_t);
    print_stats("BSA", bsa_or_t, bsa_and_t, bsa_agg_t, bsa_tot_t);
    print_stats("CON", con_or_t, con_and_t, con_agg_t, con_tot_t);
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // 8. CSV export
    // ============================================================
    {
        std::string csv_path = "q19_results_" + q19_get_sf_label() + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << "backend,or_ms,and_ms,agg_ms,total_ms\n";
            auto wrow = [&](const char* lbl, std::vector<double>& or_t,
                            std::vector<double>& and_t, std::vector<double>& agg_t,
                            std::vector<double>& tot_t) {
                if (or_t.empty()) return;
                auto so = bm_bench::compute_stats(or_t);
                auto sa = bm_bench::compute_stats(and_t);
                auto sg = bm_bench::compute_stats(agg_t);
                auto st = bm_bench::compute_stats(tot_t);
                csv << lbl << "," << so.median << "," << sa.median << ","
                    << sg.median << "," << st.median << "\n";
            };
            wrow("CB",  cb_or_t,  cb_and_t,  cb_agg_t,  cb_tot_t);
            wrow("CR",  cr_or_t,  cr_and_t,  cr_agg_t,  cr_tot_t);
            wrow("CRR", crr_or_t, crr_and_t, crr_agg_t, crr_tot_t);
            wrow("WAH", wah_or_t, wah_and_t, wah_agg_t, wah_tot_t);
            wrow("EW",  ew_or_t,  ew_and_t,  ew_agg_t,  ew_tot_t);
            wrow("BS",  bs_or_t,  bs_and_t,  bs_agg_t,  bs_tot_t);
            wrow("BSA", bsa_or_t, bsa_and_t, bsa_agg_t, bsa_tot_t);
            wrow("CON", con_or_t, con_and_t, con_agg_t, con_tot_t);
            std::cout << "\n  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    });  // end std::call_once
}

}  // namespace duckdb
