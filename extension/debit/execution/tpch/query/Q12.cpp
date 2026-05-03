// =============================================================================
// TPC-H Q12 — Shipping Modes and Order Priority (spec v3.0.1 §2.4.12)
//
//   SELECT l_shipmode,
//          SUM(CASE WHEN o_orderpriority = '1-URGENT'
//                     OR o_orderpriority = '2-HIGH'
//                   THEN 1 ELSE 0 END) AS high_line_count,
//          SUM(CASE WHEN o_orderpriority <> '1-URGENT'
//                    AND o_orderpriority <> '2-HIGH'
//                   THEN 1 ELSE 0 END) AS low_line_count
//   FROM   orders, lineitem
//   WHERE  o_orderkey = l_orderkey
//     AND  l_shipmode IN ('MAIL', 'SHIP')
//     AND  l_commitdate < l_receiptdate
//     AND  l_shipdate   < l_commitdate
//     AND  l_receiptdate >= DATE '1994-01-01'
//     AND  l_receiptdate <  DATE '1995-01-01'
//   GROUP BY l_shipmode
//   ORDER BY l_shipmode;
//
// Bitmap pipeline (all bitmaps aligned to lineitem.rowid):
//
//   T0: OR_many(receiptdate days [731..1095])  → date_mask
//   T1: date_mask AND commit_lt_receipt        → c_mask
//        c_mask     AND ship_lt_commit          → cs_mask
//   T2: For each shipmode m in {MAIL, SHIP}:
//        sm_filt = cs_mask AND shipmode[m]
//        high[m] = popcount(sm_filt AND priority_high)
//        low[m]  = popcount(sm_filt AND priority_low)
//
// Output: 2 rows (MAIL, SHIP) × (high_line_count, low_line_count).
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
#include <map>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <unordered_set>

namespace duckdb {

// --- Q12 bitmap directories ---
static const std::string Q12_SF      = bm_bench::sf_suffix();
static const std::string Q12_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q12" + Q12_SF + "_combit");
static const std::string Q12_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q12" + Q12_SF + "_wah");
static const std::string Q12_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q12" + Q12_SF + "_croaring");
static const std::string Q12_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q12" + Q12_SF + "_ewah");

// --- Backend selection (DEBIT_BM=all|wah|cb|cr|crr|ew|bs|bsa|con) ---
using Q12BmType = bm_bench::Backend;
static const Q12BmType Q12_BM = bm_bench::parse_backend("Q12_BM");

static bool run_all() { return Q12_BM == Q12BmType::ALL; }
static bool run_wah() { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::WAH; }
static bool run_cb()  { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::CB;  }
static bool run_cr()  { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::CR;  }
static bool run_crr() { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::CRR; }
static bool run_ew()  { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::EW;  }
static bool run_bs()  { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::BS;  }
static bool run_bsa() { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::BSA; }
static bool run_con() { return Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::CON; }

static const char* q12_bm_label()     { return bm_bench::backend_label(Q12_BM); }
static std::string q12_get_sf_label() { return bm_bench::sf_label(); }

// --- Q12 predicate (TPC-H spec §2.4.12) ---
// Receipt range: 1994-01-01..1995-01-01 (exclusive end) = days [731..1095].
// Shipmodes: MAIL, SHIP.  Priority partition: HIGH = {'1-URGENT','2-HIGH'}.
static const int Q12_DATE_START = 731;
static const int Q12_DATE_END   = 1095;
static const std::array<const char*, 2> Q12_SHIPMODES = {"MAIL", "SHIP"};

// --- Iteration counts (DEBIT_ITER / DEBIT_WARMUP) ---
static const int Q12_ITERATIONS = bm_bench::iter_count(10);
static const int Q12_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q12_once_flag;

// Statistics helper.
// Bitmap loaders (same set as Q4/Q5/Q8/Q10).
static ComBit q12_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q12_load_cr(const std::string& p) {
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
static ibis::bitvector q12_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q12_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// Per-shipmode counts: index 0 = MAIL, 1 = SHIP.
struct Q12ShipmodeCounts {
    int64_t high = 0;
    int64_t low  = 0;
};
using Q12Counts = std::array<Q12ShipmodeCounts, 2>;

// =============================================================================
// BMTPCH_Q12 — main benchmark entry point
// =============================================================================
void BMTableScan::BMTPCH_Q12(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q12_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    bm_bench::warn_if_sf1();

    // ============================================================
    // 0. Banner
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q12 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << q12_get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q12 Benchmark — " << q12_bm_label() << " only ("
                  << q12_get_sf_label() << ")" << std::endl;
    }
    std::cout << "  OR receiptdate days " << Q12_DATE_START << ".." << Q12_DATE_END
              << " (" << (Q12_DATE_END - Q12_DATE_START + 1) << " bitmaps)" << std::endl;
    std::cout << "  AND commit_lt_receipt, ship_lt_commit, then per-shipmode AND + 2x popcount_and"
              << std::endl;
    std::cout << "  TPC-H params: shipmodes={MAIL, SHIP}, receiptdate [1994-01-01, 1995-01-01)"
              << std::endl;
    std::cout << "  Bitmap dirs:";
    if (run_cb())              std::cout << " " << Q12_CB_DIR;
    if (run_cr() || run_crr()) std::cout << " " << Q12_CR_DIR;
    if (run_wah())             std::cout << " " << Q12_WAH_DIR;
    if (run_ew())              std::cout << " " << Q12_EW_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << Q12_ITERATIONS
              << " (first " << Q12_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    size_t num_rows = 0;
    {
        std::ifstream meta(Q12_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
        }
    }
    if (num_rows == 0) {
        std::cerr << "Error: cannot read num_rows from " << Q12_CB_DIR << "/done.txt" << std::endl;
        return;
    }

    // ============================================================
    // 0.5 Build priority_high/low bitmaps dynamically (TPC-H 1.5.7).
    //
    //    spec: "high" = orders with o_orderpriority IN ('1-URGENT','2-HIGH'),
    //          "low"  = otherwise.  Project per-orders flag down to per-
    //          lineitem via FK lookup on l_orderkey (sort-merge: both
    //          tables ordered by orderkey).  Replaces the previously
    //          stored multi-table BJI on disk.
    // ============================================================
    std::vector<uint32_t> prio_high_pos, prio_low_pos;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        // Step 1: per-orders priority flag from single-table query.
        std::unordered_set<int64_t> high_okeys;
        {
            Connection con(*context.client.db);
            auto r = con.Query(
                "SELECT o_orderkey FROM orders "
                "WHERE o_orderpriority IN ('1-URGENT','2-HIGH')");
            if (r && !r->HasError()) {
                high_okeys.reserve(r->RowCount());
                for (idx_t i = 0; i < r->RowCount(); i++)
                    high_okeys.insert(r->GetValue(0, i).GetValue<int64_t>());
            }
        }
        // Step 2: load lineitem.l_orderkey single-table data.
        std::vector<int64_t> li_okey(num_rows);
        {
            auto& tbl = Catalog::GetEntry<TableCatalogEntry>(
                context.client, "", "", "lineitem");
            auto& tx = DuckTransaction::Get(context.client, tbl.catalog);
            TableScanState st;
            vector<StorageIndex> cols{ StorageIndex(0) };
            tbl.GetStorage().InitializeScan(context.client, tx, st, cols);
            vector<LogicalType> types{ tbl.GetColumns().GetColumnTypes()[0] };
            size_t off = 0;
            while (true) {
                DataChunk ch; ch.Initialize(context.client, types);
                tbl.GetStorage().Scan(tx, ch, st);
                if (ch.size() == 0) break;
                std::memcpy(li_okey.data() + off,
                            FlatVector::GetData<int64_t>(ch.data[0]),
                            ch.size() * 8);
                off += ch.size();
            }
        }
        // Step 3: FK lookup, distribute lineitems by priority.
        prio_high_pos.reserve(num_rows / 2);
        prio_low_pos.reserve(num_rows / 2);
        for (size_t i = 0; i < num_rows; i++) {
            if (high_okeys.count(li_okey[i])) prio_high_pos.push_back(static_cast<uint32_t>(i));
            else                              prio_low_pos.push_back(static_cast<uint32_t>(i));
        }
        std::cout << "[Build priority] high=" << prio_high_pos.size()
                  << " low=" << prio_low_pos.size() << " in "
                  << ms(t0, std::chrono::high_resolution_clock::now()) << " ms" << std::endl;
    }

    // ============================================================
    // 1. Load bitmaps (per-backend gated + timed)
    // ============================================================
    std::cout << "\n[Load] Loading bitmaps (mode=" << q12_bm_label() << ")..." << std::endl;

    // --- ComBit ---
    std::vector<ComBit> cb_date;
    std::vector<const ComBit*> cb_date_ptrs;
    ComBit cb_clt_r, cb_slt_c;
    std::array<ComBit, 2> cb_shipmode;
    ComBit cb_prio_high, cb_prio_low;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_date.resize(Q12_DATE_END + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            cb_date[d] = q12_load_cb(Q12_CB_DIR + "/receiptdate/" + std::to_string(d) + ".bm");
        cb_clt_r = q12_load_cb(Q12_CB_DIR + "/commit_lt_receipt/0.bm");
        cb_slt_c = q12_load_cb(Q12_CB_DIR + "/ship_lt_commit/0.bm");
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++)
            cb_shipmode[mi] = q12_load_cb(Q12_CB_DIR + "/shipmode/" + Q12_SHIPMODES[mi] + ".bm");
        // priority_high/low built in-memory (TPC-H 1.5.7 compliance).
        {
            std::vector<bool> jm(num_rows, false);
            for (uint32_t p : prio_high_pos) jm[p] = true;
            cb_prio_high = ComBit::compress(jm, false);
            std::fill(jm.begin(), jm.end(), false);
            for (uint32_t p : prio_low_pos) jm[p] = true;
            cb_prio_low = ComBit::compress(jm, false);
        }

        cb_date_ptrs.reserve(Q12_DATE_END - Q12_DATE_START + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            cb_date_ptrs.push_back(&cb_date[d]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring (vanilla) ---
    std::vector<roaring::Roaring> cr_date;
    roaring::Roaring cr_clt_r, cr_slt_c, cr_prio_high, cr_prio_low;
    std::array<roaring::Roaring, 2> cr_shipmode;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_date.resize(Q12_DATE_END + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            cr_date[d] = q12_load_cr(Q12_CR_DIR + "/receiptdate/" + std::to_string(d) + ".bm");
        cr_clt_r     = q12_load_cr(Q12_CR_DIR + "/commit_lt_receipt/0.bm");
        cr_slt_c     = q12_load_cr(Q12_CR_DIR + "/ship_lt_commit/0.bm");
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++)
            cr_shipmode[mi] = q12_load_cr(Q12_CR_DIR + "/shipmode/" + Q12_SHIPMODES[mi] + ".bm");
        if (!prio_high_pos.empty()) cr_prio_high.addMany(prio_high_pos.size(), prio_high_pos.data());
        if (!prio_low_pos.empty())  cr_prio_low.addMany(prio_low_pos.size(),  prio_low_pos.data());
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring + Run (loads fresh + runOptimize) ---
    std::vector<roaring::Roaring> crr_date;
    std::vector<const roaring::Roaring*> crr_date_ptrs;
    roaring::Roaring crr_clt_r, crr_slt_c, crr_prio_high, crr_prio_low;
    std::array<roaring::Roaring, 2> crr_shipmode;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_date.resize(Q12_DATE_END + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++) {
            crr_date[d] = q12_load_cr(Q12_CR_DIR + "/receiptdate/" + std::to_string(d) + ".bm");
            crr_date[d].runOptimize();
        }
        crr_clt_r     = q12_load_cr(Q12_CR_DIR + "/commit_lt_receipt/0.bm"); crr_clt_r.runOptimize();
        crr_slt_c     = q12_load_cr(Q12_CR_DIR + "/ship_lt_commit/0.bm");    crr_slt_c.runOptimize();
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
            crr_shipmode[mi] = q12_load_cr(Q12_CR_DIR + "/shipmode/" + Q12_SHIPMODES[mi] + ".bm");
            crr_shipmode[mi].runOptimize();
        }
        if (!prio_high_pos.empty()) crr_prio_high.addMany(prio_high_pos.size(), prio_high_pos.data());
        crr_prio_high.runOptimize();
        if (!prio_low_pos.empty()) crr_prio_low.addMany(prio_low_pos.size(), prio_low_pos.data());
        crr_prio_low.runOptimize();

        crr_date_ptrs.reserve(Q12_DATE_END - Q12_DATE_START + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            crr_date_ptrs.push_back(&crr_date[d]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- WAH ---
    std::vector<ibis::bitvector> wah_date;
    ibis::bitvector wah_clt_r, wah_slt_c, wah_prio_high, wah_prio_low;
    std::array<ibis::bitvector, 2> wah_shipmode;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wah_date.resize(Q12_DATE_END + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            wah_date[d] = q12_load_wah(Q12_WAH_DIR + "/receiptdate/" + std::to_string(d) + ".bm");
        wah_clt_r     = q12_load_wah(Q12_WAH_DIR + "/commit_lt_receipt/0.bm");
        wah_slt_c     = q12_load_wah(Q12_WAH_DIR + "/ship_lt_commit/0.bm");
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++)
            wah_shipmode[mi] = q12_load_wah(Q12_WAH_DIR + "/shipmode/" + Q12_SHIPMODES[mi] + ".bm");
        auto wah_build = [&](ibis::bitvector& w, const std::vector<uint32_t>& pos) {
            size_t k = 0;
            for (size_t i = 0; i < num_rows; i++) {
                bool b = (k < pos.size() && pos[k] == i);
                if (b) k++;
                w += (b ? 1 : 0);
            }
            w.compress();
        };
        wah_build(wah_prio_high, prio_high_pos);
        wah_build(wah_prio_low,  prio_low_pos);
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- EWAH ---
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_date;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_date_ptrs;
    ewah::EWAHBoolArray<uint64_t> ew_clt_r, ew_slt_c, ew_prio_high, ew_prio_low;
    std::array<ewah::EWAHBoolArray<uint64_t>, 2> ew_shipmode;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ew_date.resize(Q12_DATE_END + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            ew_date[d] = q12_load_ew(Q12_EW_DIR + "/receiptdate/" + std::to_string(d) + ".bm");
        ew_clt_r     = q12_load_ew(Q12_EW_DIR + "/commit_lt_receipt/0.bm");
        ew_slt_c     = q12_load_ew(Q12_EW_DIR + "/ship_lt_commit/0.bm");
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++)
            ew_shipmode[mi] = q12_load_ew(Q12_EW_DIR + "/shipmode/" + Q12_SHIPMODES[mi] + ".bm");
        for (uint32_t p : prio_high_pos) ew_prio_high.set(p);
        if (ew_prio_high.sizeInBits() < num_rows) ew_prio_high.padWithZeroes(num_rows);
        for (uint32_t p : prio_low_pos) ew_prio_low.set(p);
        if (ew_prio_low.sizeInBits() < num_rows)  ew_prio_low.padWithZeroes(num_rows);
        ew_date_ptrs.reserve(Q12_DATE_END - Q12_DATE_START + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            ew_date_ptrs.push_back(&ew_date[d]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Bitset (BS / BSA share) ---
    std::vector<bs::Bitmap> bs_date;
    bs::Bitmap bs_clt_r, bs_slt_c, bs_prio_high, bs_prio_low;
    std::array<bs::Bitmap, 2> bs_shipmode;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_date.resize(Q12_DATE_END + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            bs_date[d] = bm_bench::load_bitmap_from_croaring(
                Q12_CR_DIR + "/receiptdate/" + std::to_string(d) + ".bm");
        bs_clt_r     = bm_bench::load_bitmap_from_croaring(Q12_CR_DIR + "/commit_lt_receipt/0.bm");
        bs_slt_c     = bm_bench::load_bitmap_from_croaring(Q12_CR_DIR + "/ship_lt_commit/0.bm");
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++)
            bs_shipmode[mi] = bm_bench::load_bitmap_from_croaring(
                Q12_CR_DIR + "/shipmode/" + Q12_SHIPMODES[mi] + ".bm");
        bs_prio_high.alloc_for_bits(num_rows);
        for (uint32_t p : prio_high_pos) bs_prio_high.words[p / 64] |= uint64_t(1) << (p % 64);
        bs_prio_low.alloc_for_bits(num_rows);
        for (uint32_t p : prio_low_pos)  bs_prio_low.words[p / 64]  |= uint64_t(1) << (p % 64);
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Concise ---
    std::vector<ConciseSet<false>> con_date;
    std::vector<const ConciseSet<false>*> con_date_ptrs;
    ConciseSet<false> con_clt_r, con_slt_c, con_prio_high, con_prio_low;
    std::array<ConciseSet<false>, 2> con_shipmode;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_date.resize(Q12_DATE_END + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
            con_date[d] = bm_bench::load_concise_from_croaring(
                Q12_CR_DIR + "/receiptdate/" + std::to_string(d) + ".bm");
        con_clt_r     = bm_bench::load_concise_from_croaring(Q12_CR_DIR + "/commit_lt_receipt/0.bm");
        con_slt_c     = bm_bench::load_concise_from_croaring(Q12_CR_DIR + "/ship_lt_commit/0.bm");
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++)
            con_shipmode[mi] = bm_bench::load_concise_from_croaring(
                Q12_CR_DIR + "/shipmode/" + Q12_SHIPMODES[mi] + ".bm");
        for (uint32_t p : prio_high_pos) con_prio_high.add(p);
        for (uint32_t p : prio_low_pos)  con_prio_low.add(p);
        con_date_ptrs.reserve(Q12_DATE_END - Q12_DATE_START + 1);
        for (int d = Q12_DATE_START; d <= Q12_DATE_END; d++)
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
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q12_WAH_DIR)) << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q12_CB_DIR))  << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q12_CR_DIR))  << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q12_EW_DIR))  << " MiB" << std::endl;

    // ============================================================
    // 2. Benchmark loop — phases:
    //    OR_date  : OR_many over 365 receiptdate days
    //    AND_pred : AND with commit_lt_receipt + ship_lt_commit
    //    AND+Pop  : per-shipmode AND + 2x popcount_and (high, low)
    // ============================================================
    std::vector<double> cb_or_t,  cb_pred_t,  cb_pop_t,  cb_tot_t;
    std::vector<double> cr_or_t,  cr_pred_t,  cr_pop_t,  cr_tot_t;
    std::vector<double> crr_or_t, crr_pred_t, crr_pop_t, crr_tot_t;
    std::vector<double> wah_or_t, wah_pred_t, wah_pop_t, wah_tot_t;
    std::vector<double> ew_or_t,  ew_pred_t,  ew_pop_t,  ew_tot_t;
    std::vector<double> bs_or_t,  bs_pred_t,  bs_pop_t,  bs_tot_t;
    std::vector<double> bsa_or_t, bsa_pred_t, bsa_pop_t, bsa_tot_t;
    std::vector<double> con_or_t, con_pred_t, con_pop_t, con_tot_t;

    Q12Counts cb_cnt{}, cr_cnt{}, crr_cnt{}, wah_cnt{}, ew_cnt{};
    Q12Counts bs_cnt{}, bsa_cnt{}, con_cnt{};

    for (int iter = 0; iter < Q12_ITERATIONS; iter++) {
        bool warmup = (iter < Q12_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q12_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ========== ComBit ==========
        // OR_many → in-place &= chain → per-shipmode copy + &= + 2x
        // popcount_and (fused AND+VPOPCNTDQ).  Result of OR_many is
        // Decompressed; subsequent &= preserves the layout, so the
        // popcount_and walks a Decompressed bitmap with L3 bypass.
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ComBit cb_or = ComBit::OR_many(cb_date_ptrs.size(), cb_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            cb_or &= cb_clt_r;
            cb_or &= cb_slt_c;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                ComBit sm_filt = cb_or;
                sm_filt &= cb_shipmode[mi];
                cnt[mi].high = static_cast<int64_t>(sm_filt.popcount_and(cb_prio_high));
                cnt[mi].low  = static_cast<int64_t>(sm_filt.popcount_and(cb_prio_low));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            cb_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CB:   OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total
                      << "  MAIL=(" << cnt[0].high << "," << cnt[0].low << ")"
                      << "  SHIP=(" << cnt[1].high << "," << cnt[1].low << ")" << std::endl;
            if (!warmup) {
                cb_or_t.push_back(d_or);
                cb_pred_t.push_back(d_pred);
                cb_pop_t.push_back(d_pop);
                cb_tot_t.push_back(d_total);
            }
        }

        // ========== CRoaring (vanilla pairwise OR) ==========
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring cr_filt = cr_date[Q12_DATE_START];
            for (int d = Q12_DATE_START + 1; d <= Q12_DATE_END; d++)
                cr_filt |= cr_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();
            cr_filt &= cr_clt_r;
            cr_filt &= cr_slt_c;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                roaring::Roaring sm_filt = cr_filt & cr_shipmode[mi];
                cnt[mi].high = static_cast<int64_t>(sm_filt.and_cardinality(cr_prio_high));
                cnt[mi].low  = static_cast<int64_t>(sm_filt.and_cardinality(cr_prio_low));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            cr_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CR:   OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total << std::endl;
            if (!warmup) {
                cr_or_t.push_back(d_or);
                cr_pred_t.push_back(d_pred);
                cr_pop_t.push_back(d_pop);
                cr_tot_t.push_back(d_total);
            }
        }

        // ========== CRoaring + Run (fastunion) ==========
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring crr_filt = roaring::Roaring::fastunion(
                crr_date_ptrs.size(), crr_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            crr_filt &= crr_clt_r;
            crr_filt &= crr_slt_c;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                roaring::Roaring sm_filt = crr_filt & crr_shipmode[mi];
                cnt[mi].high = static_cast<int64_t>(sm_filt.and_cardinality(crr_prio_high));
                cnt[mi].low  = static_cast<int64_t>(sm_filt.and_cardinality(crr_prio_low));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            crr_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CRR:  OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total << std::endl;
            if (!warmup) {
                crr_or_t.push_back(d_or);
                crr_pred_t.push_back(d_pred);
                crr_pop_t.push_back(d_pop);
                crr_tot_t.push_back(d_total);
            }
        }

        // ========== WAH ==========
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ibis::bitvector wah_filt = wah_date[Q12_DATE_START];
            wah_filt.decompress();
            for (int d = Q12_DATE_START + 1; d <= Q12_DATE_END; d++)
                wah_filt |= wah_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();
            wah_filt &= wah_clt_r;
            wah_filt &= wah_slt_c;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                ibis::bitvector sm_filt;
                sm_filt.copy(wah_filt);
                sm_filt &= wah_shipmode[mi];
                // ibis::bitvector::count(mask) is fused AND+popcount.
                cnt[mi].high = static_cast<int64_t>(sm_filt.count(wah_prio_high));
                cnt[mi].low  = static_cast<int64_t>(sm_filt.count(wah_prio_low));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            wah_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  WAH:  OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total << std::endl;
            if (!warmup) {
                wah_or_t.push_back(d_or);
                wah_pred_t.push_back(d_pred);
                wah_pop_t.push_back(d_pop);
                wah_tot_t.push_back(d_total);
            }
        }

        // ========== EWAH (fast_logicalor) ==========
        if (run_ew()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ewah::EWAHBoolArray<uint64_t> ew_filt = ewah::fast_logicalor(
                ew_date_ptrs.size(), ew_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ewah::EWAHBoolArray<uint64_t> stage1, stage2;
            ew_filt.logicaland(ew_clt_r, stage1);
            stage1.logicaland(ew_slt_c, stage2);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                ewah::EWAHBoolArray<uint64_t> sm_filt;
                stage2.logicaland(ew_shipmode[mi], sm_filt);
                // EWAH logicalandcount fused AND+popcount.
                cnt[mi].high = static_cast<int64_t>(sm_filt.logicalandcount(ew_prio_high));
                cnt[mi].low  = static_cast<int64_t>(sm_filt.logicalandcount(ew_prio_low));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            ew_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  EW:   OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total << std::endl;
            if (!warmup) {
                ew_or_t.push_back(d_or);
                ew_pred_t.push_back(d_pred);
                ew_pop_t.push_back(d_pop);
                ew_tot_t.push_back(d_total);
            }
        }

        // ========== Bitset (scalar) ==========
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bs_filt = bs_date[Q12_DATE_START].clone();
            for (int d = Q12_DATE_START + 1; d <= Q12_DATE_END; d++)
                bs::or_inplace(bs_filt, bs_date[d], false);
            auto t1 = std::chrono::high_resolution_clock::now();
            bs::and_inplace(bs_filt, bs_clt_r, false);
            bs::and_inplace(bs_filt, bs_slt_c, false);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                bs::Bitmap sm_filt = bs_filt.clone();
                bs::and_inplace(sm_filt, bs_shipmode[mi], false);
                cnt[mi].high = static_cast<int64_t>(bs::and_popcount(sm_filt, bs_prio_high, false));
                cnt[mi].low  = static_cast<int64_t>(bs::and_popcount(sm_filt, bs_prio_low,  false));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            bs_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  BS:   OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total << std::endl;
            if (!warmup) {
                bs_or_t.push_back(d_or);
                bs_pred_t.push_back(d_pred);
                bs_pop_t.push_back(d_pop);
                bs_tot_t.push_back(d_total);
            }
        }

        // ========== Bitset + AVX-512 ==========
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bsa_filt = bs_date[Q12_DATE_START].clone();
            for (int d = Q12_DATE_START + 1; d <= Q12_DATE_END; d++)
                bs::or_inplace(bsa_filt, bs_date[d], true);
            auto t1 = std::chrono::high_resolution_clock::now();
            bs::and_inplace(bsa_filt, bs_clt_r, true);
            bs::and_inplace(bsa_filt, bs_slt_c, true);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                bs::Bitmap sm_filt = bsa_filt.clone();
                bs::and_inplace(sm_filt, bs_shipmode[mi], true);
                cnt[mi].high = static_cast<int64_t>(bs::and_popcount(sm_filt, bs_prio_high, true));
                cnt[mi].low  = static_cast<int64_t>(bs::and_popcount(sm_filt, bs_prio_low,  true));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            bsa_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  BSA:  OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total << std::endl;
            if (!warmup) {
                bsa_or_t.push_back(d_or);
                bsa_pred_t.push_back(d_pred);
                bsa_pop_t.push_back(d_pop);
                bsa_tot_t.push_back(d_total);
            }
        }

        // ========== Concise (fast_logicalor) ==========
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ConciseSet<false> con_filt = ConciseSet<false>::fast_logicalor(
                con_date_ptrs.size(), con_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            con_filt = con_filt.logicaland(con_clt_r);
            con_filt = con_filt.logicaland(con_slt_c);
            auto t2 = std::chrono::high_resolution_clock::now();

            Q12Counts cnt{};
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                ConciseSet<false> sm_filt = con_filt.logicaland(con_shipmode[mi]);
                cnt[mi].high = static_cast<int64_t>(sm_filt.logicalandCount(con_prio_high));
                cnt[mi].low  = static_cast<int64_t>(sm_filt.logicalandCount(con_prio_low));
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            con_cnt = cnt;

            double d_or = ms(t0, t1), d_pred = ms(t1, t2), d_pop = ms(t2, t3);
            double d_total = ms(t0, t3);
            std::cout << "  CON:  OR=" << d_or << "  AND_pred=" << d_pred
                      << "  AND+Pop=" << d_pop << "  Total=" << d_total << std::endl;
            if (!warmup) {
                con_or_t.push_back(d_or);
                con_pred_t.push_back(d_pred);
                con_pop_t.push_back(d_pop);
                con_tot_t.push_back(d_total);
            }
        }
    } // end iterations

    // ============================================================
    // 3. Correctness validation
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q12 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;

    const Q12Counts* print_src = nullptr;
    const char* print_src_label = "";
    if      (Q12_BM == Q12BmType::ALL || Q12_BM == Q12BmType::CB)  { print_src = &cb_cnt;  print_src_label = "ComBit"; }
    else if (Q12_BM == Q12BmType::WAH)                             { print_src = &wah_cnt; print_src_label = "WAH"; }
    else if (Q12_BM == Q12BmType::CR)                              { print_src = &cr_cnt;  print_src_label = "CRoaring"; }
    else if (Q12_BM == Q12BmType::CRR)                             { print_src = &crr_cnt; print_src_label = "CRoaring+Run"; }
    else if (Q12_BM == Q12BmType::EW)                              { print_src = &ew_cnt;  print_src_label = "EWAH"; }
    else if (Q12_BM == Q12BmType::BS)                              { print_src = &bs_cnt;  print_src_label = "Bitset"; }
    else if (Q12_BM == Q12BmType::BSA)                             { print_src = &bsa_cnt; print_src_label = "Bitset+AVX512"; }
    else if (Q12_BM == Q12BmType::CON)                             { print_src = &con_cnt; print_src_label = "Concise"; }

    std::cout << "\n  Q12 Results (" << print_src_label << " values):" << std::endl;
    std::cout << "  l_shipmode  high_line_count  low_line_count" << std::endl;
    if (print_src) {
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
            std::cout << "  " << std::left << std::setw(11) << Q12_SHIPMODES[mi]
                      << std::right << std::setw(15) << (*print_src)[mi].high
                      << std::setw(16) << (*print_src)[mi].low << std::endl;
        }
    }

    if (run_all()) {
        bool consistent = true;
        for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
            const auto& base = cb_cnt[mi];
            auto cmp = [&](const char* lbl, const Q12ShipmodeCounts& v) {
                if (v.high != base.high || v.low != base.low) {
                    std::cout << "  *** MISMATCH for " << Q12_SHIPMODES[mi]
                              << " (" << lbl << ": high=" << v.high << " low=" << v.low
                              << " vs CB: high=" << base.high << " low=" << base.low << ") ***" << std::endl;
                    consistent = false;
                }
            };
            cmp("CR",  cr_cnt[mi]);
            cmp("CRR", crr_cnt[mi]);
            cmp("WAH", wah_cnt[mi]);
            cmp("EW",  ew_cnt[mi]);
            cmp("BS",  bs_cnt[mi]);
            cmp("BSA", bsa_cnt[mi]);
            cmp("CON", con_cnt[mi]);
        }
        std::cout << "  Consistency: " << (consistent ? "ALL MATCH" : "MISMATCH DETECTED") << std::endl;
    }

    // ============================================================
    // 4. DuckDB native SQL ground-truth
    // ============================================================
    std::array<int64_t, 2> gt_high = {-1, -1};
    std::array<int64_t, 2> gt_low  = {-1, -1};
    double gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT l_shipmode, "
            "  SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' "
            "           THEN 1 ELSE 0 END) AS high_line_count, "
            "  SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' "
            "           THEN 1 ELSE 0 END) AS low_line_count "
            "FROM orders, lineitem "
            "WHERE o_orderkey = l_orderkey "
            "  AND l_shipmode IN ('MAIL', 'SHIP') "
            "  AND l_commitdate < l_receiptdate "
            "  AND l_shipdate   < l_commitdate "
            "  AND l_receiptdate >= DATE '1994-01-01' "
            "  AND l_receiptdate <  DATE '1995-01-01' "
            "GROUP BY l_shipmode "
            "ORDER BY l_shipmode";
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = con.Query(sql);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (result && !result->HasError() && result->RowCount() == 2) {
            // ORDER BY l_shipmode → row 0 = MAIL, row 1 = SHIP (alphabetical).
            for (size_t r = 0; r < 2; r++) {
                std::string sm = result->GetValue(0, r).GetValue<std::string>();
                size_t idx = (sm == "MAIL") ? 0 : 1;
                gt_high[idx] = result->GetValue(1, r).GetValue<int64_t>();
                gt_low[idx]  = result->GetValue(2, r).GetValue<int64_t>();
            }
            gt_sql_ms = ms(t0, t1);
        } else if (result && result->HasError()) {
            std::cerr << "[Baseline] SQL error: " << result->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    if (gt_high[0] >= 0) {
        std::cout << "\n[Baseline] DuckDB native SQL  (single run: "
                  << std::fixed << std::setprecision(2) << gt_sql_ms << " ms)" << std::endl;
        std::cout << "  SQL ground truth: MAIL=(" << gt_high[0] << "," << gt_low[0]
                  << ")  SHIP=(" << gt_high[1] << "," << gt_low[1] << ")" << std::endl;

        auto check = [&](const char* label, bool active, const Q12Counts& got) {
            if (!active) return;
            for (size_t mi = 0; mi < Q12_SHIPMODES.size(); mi++) {
                if (got[mi].high != gt_high[mi] || got[mi].low != gt_low[mi]) {
                    std::ostringstream oss;
                    oss << "[FAIL] Q12 " << label << " " << Q12_SHIPMODES[mi]
                        << " counts (high=" << got[mi].high << ", low=" << got[mi].low
                        << ") differ from SQL (high=" << gt_high[mi] << ", low=" << gt_low[mi] << ")";
                    throw std::runtime_error(oss.str());
                }
            }
        };
        check("WAH",           run_wah(), wah_cnt);
        check("ComBit",        run_cb(),  cb_cnt);
        check("CRoaring",      run_cr(),  cr_cnt);
        check("CRoaring+Run",  run_crr(), crr_cnt);
        check("EWAH",          run_ew(),  ew_cnt);
        check("Bitset",        run_bs(),  bs_cnt);
        check("Bitset+AVX512", run_bsa(), bsa_cnt);
        check("Concise",       run_con(), con_cnt);

        std::cout << "[OK] all active backends match DuckDB SQL ground truth "
                  << "(per-shipmode high/low counts exact)." << std::endl;
    } else {
        std::cout << "\n[Baseline] DuckDB SQL ground truth unavailable — skipping assert." << std::endl;
    }

    // ============================================================
    // 5. Statistics summary
    // ============================================================
    auto cb_or_s   = bm_bench::compute_stats(cb_or_t);
    auto cb_pred_s = bm_bench::compute_stats(cb_pred_t);
    auto cb_pop_s  = bm_bench::compute_stats(cb_pop_t);
    auto cb_tot_s  = bm_bench::compute_stats(cb_tot_t);

    auto cr_or_s   = bm_bench::compute_stats(cr_or_t);
    auto cr_pred_s = bm_bench::compute_stats(cr_pred_t);
    auto cr_pop_s  = bm_bench::compute_stats(cr_pop_t);
    auto cr_tot_s  = bm_bench::compute_stats(cr_tot_t);

    auto crr_or_s   = bm_bench::compute_stats(crr_or_t);
    auto crr_pred_s = bm_bench::compute_stats(crr_pred_t);
    auto crr_pop_s  = bm_bench::compute_stats(crr_pop_t);
    auto crr_tot_s  = bm_bench::compute_stats(crr_tot_t);

    auto wah_or_s   = bm_bench::compute_stats(wah_or_t);
    auto wah_pred_s = bm_bench::compute_stats(wah_pred_t);
    auto wah_pop_s  = bm_bench::compute_stats(wah_pop_t);
    auto wah_tot_s  = bm_bench::compute_stats(wah_tot_t);

    auto ew_or_s   = bm_bench::compute_stats(ew_or_t);
    auto ew_pred_s = bm_bench::compute_stats(ew_pred_t);
    auto ew_pop_s  = bm_bench::compute_stats(ew_pop_t);
    auto ew_tot_s  = bm_bench::compute_stats(ew_tot_t);

    auto bs_or_s   = bm_bench::compute_stats(bs_or_t);
    auto bs_pred_s = bm_bench::compute_stats(bs_pred_t);
    auto bs_pop_s  = bm_bench::compute_stats(bs_pop_t);
    auto bs_tot_s  = bm_bench::compute_stats(bs_tot_t);

    auto bsa_or_s   = bm_bench::compute_stats(bsa_or_t);
    auto bsa_pred_s = bm_bench::compute_stats(bsa_pred_t);
    auto bsa_pop_s  = bm_bench::compute_stats(bsa_pop_t);
    auto bsa_tot_s  = bm_bench::compute_stats(bsa_tot_t);

    auto con_or_s   = bm_bench::compute_stats(con_or_t);
    auto con_pred_s = bm_bench::compute_stats(con_pred_t);
    auto con_pop_s  = bm_bench::compute_stats(con_pop_t);
    auto con_tot_s  = bm_bench::compute_stats(con_tot_t);

    int measured = Q12_ITERATIONS - Q12_WARMUP;
    std::cout << std::fixed << std::setprecision(2);

    if (run_all()) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q12 RESULTS (" << measured << " measured iterations, median +/- stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  CB (ms)         CR (ms)        CRR (ms)        WAH (ms)        EW (ms)" << std::endl;
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;

        auto print_row = [](const char* label, bm_bench::Stats& cb, bm_bench::Stats& cr,
                            bm_bench::Stats& crr, bm_bench::Stats& wah, bm_bench::Stats& ew) {
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
        print_row("AND_pred", cb_pred_s, cr_pred_s, crr_pred_s, wah_pred_s, ew_pred_s);
        print_row("AND+Pop",  cb_pop_s,  cr_pop_s,  crr_pop_s,  wah_pop_s,  ew_pop_s);
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL",    cb_tot_s,  cr_tot_s,  crr_tot_s,  wah_tot_s,  ew_tot_s);
        std::cout << "================================================================\n" << std::endl;

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q12 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  BS (ms)         BSA (ms)        Concise (ms)     BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        auto print_baseline_row = [](const char* label, bm_bench::Stats& w,
                                     bm_bench::Stats& b, bm_bench::Stats& ba, bm_bench::Stats& c) {
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
        print_baseline_row("OR_date",  wah_or_s,   bs_or_s,   bsa_or_s,   con_or_s);
        print_baseline_row("AND_pred", wah_pred_s, bs_pred_s, bsa_pred_s, con_pred_s);
        print_baseline_row("AND+Pop",  wah_pop_s,  bs_pop_s,  bsa_pop_s,  con_pop_s);
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        print_baseline_row("TOTAL",    wah_tot_s,  bs_tot_s,  bsa_tot_s,  con_tot_s);
        std::cout << "================================================================\n" << std::endl;
    } else {
        bm_bench::Stats *sel_or = nullptr, *sel_pred = nullptr, *sel_pop = nullptr, *sel_tot = nullptr;
        switch (Q12_BM) {
            case Q12BmType::WAH: sel_or = &wah_or_s; sel_pred = &wah_pred_s; sel_pop = &wah_pop_s; sel_tot = &wah_tot_s; break;
            case Q12BmType::CB:  sel_or = &cb_or_s;  sel_pred = &cb_pred_s;  sel_pop = &cb_pop_s;  sel_tot = &cb_tot_s;  break;
            case Q12BmType::CR:  sel_or = &cr_or_s;  sel_pred = &cr_pred_s;  sel_pop = &cr_pop_s;  sel_tot = &cr_tot_s;  break;
            case Q12BmType::CRR: sel_or = &crr_or_s; sel_pred = &crr_pred_s; sel_pop = &crr_pop_s; sel_tot = &crr_tot_s; break;
            case Q12BmType::EW:  sel_or = &ew_or_s;  sel_pred = &ew_pred_s;  sel_pop = &ew_pop_s;  sel_tot = &ew_tot_s;  break;
            case Q12BmType::BS:  sel_or = &bs_or_s;  sel_pred = &bs_pred_s;  sel_pop = &bs_pop_s;  sel_tot = &bs_tot_s;  break;
            case Q12BmType::BSA: sel_or = &bsa_or_s; sel_pred = &bsa_pred_s; sel_pop = &bsa_pop_s; sel_tot = &bsa_tot_s; break;
            case Q12BmType::CON: sel_or = &con_or_s; sel_pred = &con_pred_s; sel_pop = &con_pop_s; sel_tot = &con_tot_s; break;
            case Q12BmType::ALL: break;
        }

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q12 RESULTS — " << q12_bm_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;

        auto print_single = [](const char* label, bm_bench::Stats& s) {
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(9) << s.median
                      << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val
                      << std::setw(10) << s.max_val << std::endl;
        };

        if (sel_or) {
            print_single("OR_date",  *sel_or);
            print_single("AND_pred", *sel_pred);
            print_single("AND+Pop",  *sel_pop);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            print_single("TOTAL",    *sel_tot);
        }
        std::cout << "================================================================\n" << std::endl;
    }

    // ============================================================
    // 6. CSV export — ALL mode only
    // ============================================================
    if (run_all()) {
        std::string sf_label = q12_get_sf_label();
        std::string csv_path = "q12_results_" + sf_label + ".csv";
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
                               bm_bench::Stats& cb, bm_bench::Stats& cr, bm_bench::Stats& crr,
                               bm_bench::Stats& wah, bm_bench::Stats& ew,
                               bm_bench::Stats& b, bm_bench::Stats& ba, bm_bench::Stats& co) {
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

            csv_row("OR_date",  cb_or_s,   cr_or_s,   crr_or_s,   wah_or_s,   ew_or_s,   bs_or_s,   bsa_or_s,   con_or_s);
            csv_row("AND_pred", cb_pred_s, cr_pred_s, crr_pred_s, wah_pred_s, ew_pred_s, bs_pred_s, bsa_pred_s, con_pred_s);
            csv_row("AND+Pop",  cb_pop_s,  cr_pop_s,  crr_pop_s,  wah_pop_s,  ew_pop_s,  bs_pop_s,  bsa_pop_s,  con_pop_s);
            csv_row("TOTAL",    cb_tot_s,  cr_tot_s,  crr_tot_s,  wah_tot_s,  ew_tot_s,  bs_tot_s,  bsa_tot_s,  con_tot_s);

            csv.close();
            std::cout << "  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
