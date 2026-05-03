// =============================================================================
// TPC-H Q8 — National Market Share (spec v3.0.1 §2.4.8)
//
//   SELECT
//       o_year,
//       SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume)
//         AS mkt_share
//   FROM (
//       SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year,
//              l_extendedprice * (1 - l_discount) AS volume,
//              n2.n_name AS nation
//       FROM part, supplier, lineitem, orders, customer,
//            nation n1, region, nation n2
//       WHERE p_partkey = l_partkey
//         AND s_suppkey = l_suppkey
//         AND l_orderkey = o_orderkey
//         AND o_custkey  = c_custkey
//         AND c_nationkey = n1.n_nationkey
//         AND n1.n_regionkey = r_regionkey
//         AND r_name = 'AMERICA'
//         AND s_nationkey = n2.n_nationkey
//         AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
//         AND p_type = 'ECONOMY ANODIZED STEEL'
//   ) AS all_nations
//   GROUP BY o_year
//   ORDER BY o_year;
//
// Bitmap pipeline (all bitmaps aligned to lineitem rowid):
//
//   T0: OR_many(orderdate days [1995-01-01..1996-12-31])  → date_mask
//   T1: date_mask AND join_result                          → joined_filt
//        (join_result = in_america AND part_match,
//         pre-encoded at export time)
//   T2: For each year y in {1995, 1996}:
//        year_filt   = joined_filt AND year_y
//        denom_y     = SUM(volume[r] for r in year_filt)
//        year_brazil = year_filt AND brazil
//        numer_y     = SUM(volume[r] for r in year_brazil)
//   T3: mkt_share_y = numer_y / denom_y
//
// volume column (l_extendedprice * (1 - l_discount)) is loaded at
// runtime from DuckDB.  Pattern matches Q5 / Q10.
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

// Uncompressed bitset (BS / BSA) + Concise (CON) baselines and shared
// from-CRoaring loaders.
#include "bitset_simple.h"
#include "Concise/concise.h"
#include "execution/tpch/bm_baseline_loaders.hpp"

// Shared benchmark helpers (DEBIT_BM, DEBIT_ITER, DEBIT_WARMUP, etc.).
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

namespace duckdb {

// --- Q8 bitmap directories ---
static const std::string Q8_SF      = bm_bench::sf_suffix();
static const std::string Q8_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q8" + Q8_SF + "_combit");
static const std::string Q8_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q8" + Q8_SF + "_wah");
static const std::string Q8_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q8" + Q8_SF + "_croaring");
static const std::string Q8_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q8" + Q8_SF + "_ewah");

// --- Backend selection (DEBIT_BM=all|wah|cb|cr|crr|ew|bs|bsa|con) ---
using Q8BmType = bm_bench::Backend;
static const Q8BmType Q8_BM = bm_bench::parse_backend("Q8_BM");

static bool run_all() { return Q8_BM == Q8BmType::ALL; }
static bool run_wah() { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::WAH; }
static bool run_cb()  { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::CB;  }
static bool run_cr()  { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::CR;  }
static bool run_crr() { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::CRR; }
static bool run_ew()  { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::EW;  }
static bool run_bs()  { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::BS;  }
static bool run_bsa() { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::BSA; }
static bool run_con() { return Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::CON; }

static const char* q8_bm_label()     { return bm_bench::backend_label(Q8_BM); }
static std::string q8_get_sf_label() { return bm_bench::sf_label(); }

// --- Q8 predicate (TPC-H spec §2.4.8) ---
// Orderdate range: 1995-01-01..1996-12-31 = days 1096..1826 (since 1992-01-01,
// counting 1992 as a leap year).
// Years: 1995, 1996.
// Region: AMERICA, Type: 'ECONOMY ANODIZED STEEL', Numerator nation: BRAZIL.
static const int Q8_DATE_START = 1096;
static const int Q8_DATE_END   = 1826;
static const std::array<int, 2> Q8_YEARS = {1995, 1996};

// --- Iteration counts (override via DEBIT_ITER / DEBIT_WARMUP) ---
static const int Q8_ITERATIONS = bm_bench::iter_count(10);
static const int Q8_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q8_once_flag;

// Byte-LUT for ComBit aggregation (same pattern as Q5).
// Bitmap loaders.
static ComBit q8_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q8_load_cr(const std::string& p) {
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
static ibis::bitvector q8_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q8_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// Statistics helper.
// Per-year aggregation result (denom = total revenue across all suppliers
// in AMERICA + ECONOMY ANODIZED STEEL + that year; numer = subset where
// supplier is BRAZIL).
struct Q8Year {
    int64_t denom = 0;   // raw cents-pennies (price * (100 - disc))
    int64_t numer = 0;   // same units as denom
};
using Q8Result = std::array<Q8Year, 2>;   // index 0 = 1995, 1 = 1996

static double q8_mkt_share(const Q8Year& y) {
    if (y.denom == 0) return 0.0;
    return static_cast<double>(y.numer) / static_cast<double>(y.denom);
}

// =============================================================================
// BMTPCH_Q8 — main benchmark entry point
// =============================================================================
void BMTableScan::BMTPCH_Q8(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q8_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    bm_bench::warn_if_sf1();

    // ============================================================
    // 0. Banner + pre-load lineitem volume columns
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q8 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << q8_get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q8 Benchmark — " << q8_bm_label() << " only ("
                  << q8_get_sf_label() << ")" << std::endl;
    }
    std::cout << "  OR orderdate days " << Q8_DATE_START << ".." << Q8_DATE_END
              << " (" << (Q8_DATE_END - Q8_DATE_START + 1) << " bitmaps)" << std::endl;
    std::cout << "  AND join_result, then per-year split (1995, 1996), aggregate revenue" << std::endl;
    std::cout << "  TPC-H params: r_name='AMERICA', p_type='ECONOMY ANODIZED STEEL', "
                 "BRAZIL nation numerator" << std::endl;
    std::cout << "  Bitmap dirs:";
    if (run_cb())              std::cout << " " << Q8_CB_DIR;
    if (run_cr() || run_crr()) std::cout << " " << Q8_CR_DIR;
    if (run_wah())             std::cout << " " << Q8_WAH_DIR;
    if (run_ew())              std::cout << " " << Q8_EW_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << Q8_ITERATIONS
              << " (first " << Q8_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // num_rows from done.txt of whichever backend dir is present.
    size_t num_rows = 0;
    {
        std::ifstream meta(Q8_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
        }
    }
    if (num_rows == 0) {
        std::cerr << "Error: cannot read num_rows from " << Q8_CB_DIR << "/done.txt" << std::endl;
        return;
    }

    // Pre-load lineitem (l_extendedprice, l_discount) — same pattern as Q5.
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

    std::cout << "\n[Pre-load] Loading " << num_rows
              << " rows (l_extendedprice, l_discount) ..." << std::endl;
    auto t_preload = std::chrono::high_resolution_clock::now();

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

    auto t_preload_done = std::chrono::high_resolution_clock::now();
    std::cout << "[Pre-load] Done in " << ms(t_preload, t_preload_done) << " ms" << std::endl;

    // ============================================================
    // 0.5 Build all multi-table BJI bitmaps dynamically (TPC-H 1.5.7).
    //     Single 7-table SQL JOIN streams per-lineitem (day, year,
    //     join_match, is_brazil) tuples through DuckDB's hash join engine.
    //     We bucket these into per-day, per-year, join, and brazil
    //     position lists, then encode each into the requested backend
    //     formats below.  No multi-table BJI files on disk.
    // ============================================================
    std::vector<std::vector<uint32_t>> q8_date_pos(Q8_DATE_END + 1);
    std::vector<std::vector<uint32_t>> q8_year_pos(2);    // 1995, 1996
    std::vector<uint32_t> q8_join_pos, q8_brazil_pos;
    double q8_join_setup_ms = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT (o.o_orderdate - DATE '1992-01-01')::INT, "
            "       EXTRACT(year FROM o.o_orderdate)::INT, "
            "       CASE WHEN r1.r_name = 'AMERICA' "
            "             AND p.p_type = 'ECONOMY ANODIZED STEEL' THEN 1 ELSE 0 END, "
            "       CASE WHEN n2.n_name = 'BRAZIL' THEN 1 ELSE 0 END "
            "FROM lineitem l "
            "JOIN orders o ON l.l_orderkey = o.o_orderkey "
            "JOIN customer c ON o.o_custkey = c.c_custkey "
            "JOIN nation n1 ON c.c_nationkey = n1.n_nationkey "
            "JOIN region r1 ON n1.n_regionkey = r1.r_regionkey "
            "JOIN supplier s ON l.l_suppkey = s.s_suppkey "
            "JOIN nation n2 ON s.s_nationkey = n2.n_nationkey "
            "JOIN part p ON l.l_partkey = p.p_partkey "
            "ORDER BY l.rowid");
        if (!r || r->HasError()) {
            std::cerr << "Q8 build SQL failed: "
                      << (r ? r->GetError() : "null result") << std::endl;
            return;
        }
        size_t pos = 0;
        while (auto chunk = r->Fetch()) {
            auto day = FlatVector::GetData<int32_t>(chunk->data[0]);
            auto yr  = FlatVector::GetData<int32_t>(chunk->data[1]);
            auto jm  = FlatVector::GetData<int32_t>(chunk->data[2]);
            auto br  = FlatVector::GetData<int32_t>(chunk->data[3]);
            for (idx_t j = 0; j < chunk->size(); j++, pos++) {
                int d = day[j];
                if (d >= Q8_DATE_START && d <= Q8_DATE_END)
                    q8_date_pos[d].push_back(static_cast<uint32_t>(pos));
                if (yr[j] == 1995) q8_year_pos[0].push_back(static_cast<uint32_t>(pos));
                else if (yr[j] == 1996) q8_year_pos[1].push_back(static_cast<uint32_t>(pos));
                if (jm[j]) q8_join_pos.push_back(static_cast<uint32_t>(pos));
                if (br[j]) q8_brazil_pos.push_back(static_cast<uint32_t>(pos));
            }
        }
        q8_join_setup_ms = ms(t0, std::chrono::high_resolution_clock::now());
        std::cout << "[Build Q8 join] " << pos << " lineitems scanned in "
                  << q8_join_setup_ms << " ms (counted in Total)"
                  << "  (join_pos=" << q8_join_pos.size()
                  << " brazil_pos=" << q8_brazil_pos.size() << ")" << std::endl;
    }

    // ============================================================
    // 1. Build per-backend bitmaps from position lists (per-backend gated)
    // ============================================================
    std::cout << "\n[Load] Building bitmaps (mode=" << q8_bm_label() << ")..." << std::endl;

    // --- ComBit ---
    std::vector<ComBit> cb_date;
    std::vector<const ComBit*> cb_date_ptrs;
    ComBit cb_join;                           // sparse: 80K / 60M = 0.13%
    std::array<FlatByteBuf, 2> cb_year_flat = { FlatByteBuf(0), FlatByteBuf(0) };
    FlatByteBuf cb_brazil_flat(0);
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_date.resize(Q8_DATE_END + 1);
        std::vector<bool> mask(num_rows, false);
        std::vector<uint32_t>* prev = nullptr;
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++) {
            if (prev) for (uint32_t p : *prev) mask[p] = false;
            for (uint32_t p : q8_date_pos[d]) mask[p] = true;
            cb_date[d] = ComBit::compress(mask, false);
            prev = &q8_date_pos[d];
        }
        if (prev) for (uint32_t p : *prev) mask[p] = false;
        for (uint32_t p : q8_join_pos) mask[p] = true;
        cb_join = ComBit::compress(mask, false);
        for (uint32_t p : q8_join_pos) mask[p] = false;

        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
            for (uint32_t p : q8_year_pos[yi]) mask[p] = true;
            ComBit y = ComBit::compress(mask, false);
            cb_year_flat[yi] = combit_decompress_to_flat(y);
            for (uint32_t p : q8_year_pos[yi]) mask[p] = false;
        }
        for (uint32_t p : q8_brazil_pos) mask[p] = true;
        ComBit b = ComBit::compress(mask, false);
        cb_brazil_flat = combit_decompress_to_flat(b);
        for (uint32_t p : q8_brazil_pos) mask[p] = false;

        cb_date_ptrs.reserve(Q8_DATE_END - Q8_DATE_START + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            cb_date_ptrs.push_back(&cb_date[d]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring ---
    std::vector<roaring::Roaring> cr_date;
    roaring::Roaring cr_join, cr_brazil;
    std::array<roaring::Roaring, 2> cr_year;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_date.resize(Q8_DATE_END + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++) {
            auto& pos = q8_date_pos[d];
            if (!pos.empty()) cr_date[d].addMany(pos.size(), pos.data());
        }
        if (!q8_join_pos.empty())   cr_join.addMany(q8_join_pos.size(), q8_join_pos.data());
        if (!q8_brazil_pos.empty()) cr_brazil.addMany(q8_brazil_pos.size(), q8_brazil_pos.data());
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++)
            if (!q8_year_pos[yi].empty()) cr_year[yi].addMany(q8_year_pos[yi].size(), q8_year_pos[yi].data());
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring + Run (loads fresh + runOptimize) ---
    std::vector<roaring::Roaring> crr_date;
    std::vector<const roaring::Roaring*> crr_date_ptrs;
    roaring::Roaring crr_join, crr_brazil;
    std::array<roaring::Roaring, 2> crr_year;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_date.resize(Q8_DATE_END + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++) {
            auto& pos = q8_date_pos[d];
            if (!pos.empty()) crr_date[d].addMany(pos.size(), pos.data());
            crr_date[d].runOptimize();
        }
        if (!q8_join_pos.empty()) crr_join.addMany(q8_join_pos.size(), q8_join_pos.data());
        crr_join.runOptimize();
        if (!q8_brazil_pos.empty()) crr_brazil.addMany(q8_brazil_pos.size(), q8_brazil_pos.data());
        crr_brazil.runOptimize();
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
            if (!q8_year_pos[yi].empty()) crr_year[yi].addMany(q8_year_pos[yi].size(), q8_year_pos[yi].data());
            crr_year[yi].runOptimize();
        }
        crr_date_ptrs.reserve(Q8_DATE_END - Q8_DATE_START + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            crr_date_ptrs.push_back(&crr_date[d]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- WAH ---
    std::vector<ibis::bitvector> wah_date;
    ibis::bitvector wah_join, wah_brazil;
    std::array<ibis::bitvector, 2> wah_year;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto wah_build = [&](ibis::bitvector& w, const std::vector<uint32_t>& pos) {
            size_t k = 0;
            for (size_t i = 0; i < num_rows; i++) {
                bool b = (k < pos.size() && pos[k] == i);
                if (b) k++;
                w += (b ? 1 : 0);
            }
            w.compress();
        };
        wah_date.resize(Q8_DATE_END + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            wah_build(wah_date[d], q8_date_pos[d]);
        wah_build(wah_join, q8_join_pos);
        wah_build(wah_brazil, q8_brazil_pos);
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++)
            wah_build(wah_year[yi], q8_year_pos[yi]);
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- EWAH ---
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_date;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_date_ptrs;
    ewah::EWAHBoolArray<uint64_t> ew_join, ew_brazil;
    std::array<ewah::EWAHBoolArray<uint64_t>, 2> ew_year;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto ew_build = [&](ewah::EWAHBoolArray<uint64_t>& e, const std::vector<uint32_t>& pos) {
            for (uint32_t p : pos) e.set(p);
            if (e.sizeInBits() < num_rows) e.padWithZeroes(num_rows);
        };
        ew_date.resize(Q8_DATE_END + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            ew_build(ew_date[d], q8_date_pos[d]);
        ew_build(ew_join, q8_join_pos);
        ew_build(ew_brazil, q8_brazil_pos);
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++)
            ew_build(ew_year[yi], q8_year_pos[yi]);
        ew_date_ptrs.reserve(Q8_DATE_END - Q8_DATE_START + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            ew_date_ptrs.push_back(&ew_date[d]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Bitset (BS / BSA share data; rebuilt from CRoaring) ---
    std::vector<bs::Bitmap> bs_date;
    bs::Bitmap bs_join, bs_brazil;
    std::array<bs::Bitmap, 2> bs_year;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto bs_build = [&](bs::Bitmap& b, const std::vector<uint32_t>& pos) {
            b.alloc_for_bits(num_rows);
            for (uint32_t p : pos) b.words[p / 64] |= uint64_t(1) << (p % 64);
        };
        bs_date.resize(Q8_DATE_END + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            bs_build(bs_date[d], q8_date_pos[d]);
        bs_build(bs_join, q8_join_pos);
        bs_build(bs_brazil, q8_brazil_pos);
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++)
            bs_build(bs_year[yi], q8_year_pos[yi]);
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Concise (rebuilt from CRoaring) ---
    std::vector<ConciseSet<false>> con_date;
    std::vector<const ConciseSet<false>*> con_date_ptrs;
    ConciseSet<false> con_join, con_brazil;
    std::array<ConciseSet<false>, 2> con_year;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_date.resize(Q8_DATE_END + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            for (uint32_t p : q8_date_pos[d]) con_date[d].add(p);
        for (uint32_t p : q8_join_pos)   con_join.add(p);
        for (uint32_t p : q8_brazil_pos) con_brazil.add(p);
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++)
            for (uint32_t p : q8_year_pos[yi]) con_year[yi].add(p);
        con_date_ptrs.reserve(Q8_DATE_END - Q8_DATE_START + 1);
        for (int d = Q8_DATE_START; d <= Q8_DATE_END; d++)
            con_date_ptrs.push_back(&con_date[d]);
        con_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:      " << wah_load_ms << " ms" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:   " << cb_load_ms  << " ms (incl. year/brazil decompress+flatten)" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load: " << cr_load_ms  << " ms" << std::endl;
    if (run_crr()) std::cout << "  CRR load:      " << crr_load_ms << " ms" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:     " << ew_load_ms  << " ms" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:   " << bs_load_ms  << " ms (shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:  " << con_load_ms << " ms" << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q8_WAH_DIR)) << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q8_CB_DIR))  << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q8_CR_DIR))  << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q8_EW_DIR))  << " MiB" << std::endl;

    // ============================================================
    // 2. Benchmark loop
    // ============================================================
    std::vector<double> cb_or_t,  cb_and_t,  cb_tot_t;
    std::vector<double> cr_or_t,  cr_and_t,  cr_tot_t;
    std::vector<double> crr_or_t, crr_and_t, crr_tot_t;
    std::vector<double> wah_or_t, wah_and_t, wah_tot_t;
    std::vector<double> ew_or_t,  ew_and_t,  ew_tot_t;
    std::vector<double> bs_or_t,  bs_and_t,  bs_tot_t;
    std::vector<double> bsa_or_t, bsa_and_t, bsa_tot_t;
    std::vector<double> con_or_t, con_and_t, con_tot_t;

    Q8Result cb_res{}, cr_res{}, crr_res{}, wah_res{}, ew_res{};
    Q8Result bs_res{}, bsa_res{}, con_res{};
    size_t cb_rows = 0, cr_rows = 0, crr_rows = 0, wah_rows = 0, ew_rows = 0;
    size_t bs_rows = 0, bsa_rows = 0, con_rows = 0;

    for (int iter = 0; iter < Q8_ITERATIONS; iter++) {
        bool warmup = (iter < Q8_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q8_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ========== ComBit ==========
        // OR_many → &= cb_join (in place).  Walk the result via
        // for_each_literal; cb_join's sparsity (0.13%) makes the
        // post-AND result equally sparse, so the walk visits very few
        // L1 bytes.  Per-byte: AND with year_flat[y] gives the year's
        // contribution, AND with brazil_flat gives the BRAZIL subset.
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ComBit cb_or = ComBit::OR_many(cb_date_ptrs.size(), cb_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            cb_or &= cb_join;
            auto t2 = std::chrono::high_resolution_clock::now();

            Q8Result res{};
            size_t total_rows = 0;
            const int64_t* pp = col_price.data();
            const int64_t* dp = col_disc.data();
            const uint8_t* yfa = cb_year_flat[0].data();
            const uint8_t* yfb = cb_year_flat[1].data();
            const uint8_t* brf = cb_brazil_flat.data();

            cb_or.for_each_literal([&](uint32_t word_pos, uint8_t join_byte) {
                if (join_byte == 0) return;
                size_t rbase = static_cast<size_t>(word_pos) * 8;

                // Year 1995 contribution.
                uint8_t b95 = join_byte & yfa[word_pos];
                if (b95) {
                    const auto& e = bm_bench::byte_lut[b95];
                    for (int k = 0; k < e.count; k++) {
                        size_t row = rbase + e.pos[k];
                        int64_t rev = pp[row] * (100 - dp[row]);
                        res[0].denom += rev;
                    }
                    uint8_t br95 = b95 & brf[word_pos];
                    if (br95) {
                        const auto& eb = bm_bench::byte_lut[br95];
                        for (int k = 0; k < eb.count; k++) {
                            size_t row = rbase + eb.pos[k];
                            int64_t rev = pp[row] * (100 - dp[row]);
                            res[0].numer += rev;
                        }
                    }
                    total_rows += e.count;
                }

                // Year 1996 contribution.
                uint8_t b96 = join_byte & yfb[word_pos];
                if (b96) {
                    const auto& e = bm_bench::byte_lut[b96];
                    for (int k = 0; k < e.count; k++) {
                        size_t row = rbase + e.pos[k];
                        int64_t rev = pp[row] * (100 - dp[row]);
                        res[1].denom += rev;
                    }
                    uint8_t br96 = b96 & brf[word_pos];
                    if (br96) {
                        const auto& eb = bm_bench::byte_lut[br96];
                        for (int k = 0; k < eb.count; k++) {
                            size_t row = rbase + eb.pos[k];
                            int64_t rev = pp[row] * (100 - dp[row]);
                            res[1].numer += rev;
                        }
                    }
                    total_rows += e.count;
                }
            });
            auto t3 = std::chrono::high_resolution_clock::now();

            cb_res = res;
            cb_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t3), d_total = ms(t0, t3) + q8_join_setup_ms;
            std::cout << "  CB:   OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                cb_or_t.push_back(d_or);
                cb_and_t.push_back(d_and);
                cb_tot_t.push_back(d_total);
            }
        }

        // ========== CRoaring (vanilla pairwise OR) ==========
        // Naive |= chain — separate baseline from CRR.  After the OR
        // and join AND, we materialize per-year filters as new Roaring
        // bitmaps and walk them via iterator.
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring cr_filt = cr_date[Q8_DATE_START];
            for (int d = Q8_DATE_START + 1; d <= Q8_DATE_END; d++)
                cr_filt |= cr_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();
            cr_filt &= cr_join;

            Q8Result res{};
            size_t total_rows = 0;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                roaring::Roaring year_filt = cr_filt & cr_year[yi];
                int64_t denom = 0;
                size_t cnt = 0;
                for (auto it = year_filt.begin(); it != year_filt.end(); ++it) {
                    size_t row = *it;
                    denom += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }
                roaring::Roaring brazil_filt = year_filt & cr_brazil;
                int64_t numer = 0;
                for (auto it = brazil_filt.begin(); it != brazil_filt.end(); ++it) {
                    size_t row = *it;
                    numer += col_price[row] * (100 - col_disc[row]);
                }
                res[yi].denom = denom;
                res[yi].numer = numer;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            cr_res = res;
            cr_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_total = ms(t0, t2) + q8_join_setup_ms;
            std::cout << "  CR:   OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                cr_or_t.push_back(d_or);
                cr_and_t.push_back(d_and);
                cr_tot_t.push_back(d_total);
            }
        }

        // ========== CRoaring + Run (fastunion) ==========
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            roaring::Roaring crr_filt = roaring::Roaring::fastunion(
                crr_date_ptrs.size(), crr_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            crr_filt &= crr_join;

            Q8Result res{};
            size_t total_rows = 0;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                roaring::Roaring year_filt = crr_filt & crr_year[yi];
                int64_t denom = 0;
                size_t cnt = 0;
                for (auto it = year_filt.begin(); it != year_filt.end(); ++it) {
                    size_t row = *it;
                    denom += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }
                roaring::Roaring brazil_filt = year_filt & crr_brazil;
                int64_t numer = 0;
                for (auto it = brazil_filt.begin(); it != brazil_filt.end(); ++it) {
                    size_t row = *it;
                    numer += col_price[row] * (100 - col_disc[row]);
                }
                res[yi].denom = denom;
                res[yi].numer = numer;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            crr_res = res;
            crr_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_total = ms(t0, t2) + q8_join_setup_ms;
            std::cout << "  CRR:  OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                crr_or_t.push_back(d_or);
                crr_and_t.push_back(d_and);
                crr_tot_t.push_back(d_total);
            }
        }

        // ========== WAH ==========
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ibis::bitvector wah_filt = wah_date[Q8_DATE_START];
            wah_filt.decompress();
            for (int d = Q8_DATE_START + 1; d <= Q8_DATE_END; d++)
                wah_filt |= wah_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();
            wah_filt &= wah_join;

            Q8Result res{};
            size_t total_rows = 0;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                ibis::bitvector year_filt;
                year_filt.copy(wah_filt);
                year_filt &= wah_year[yi];

                int64_t denom = 0;
                size_t cnt = 0;
                ibis::bitvector::pit pit(year_filt);
                while (*pit != 0xFFFFFFFFU) {
                    size_t row = *pit;
                    denom += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                    pit.next();
                }

                ibis::bitvector brazil_filt;
                brazil_filt.copy(year_filt);
                brazil_filt &= wah_brazil;
                int64_t numer = 0;
                ibis::bitvector::pit bpit(brazil_filt);
                while (*bpit != 0xFFFFFFFFU) {
                    size_t row = *bpit;
                    numer += col_price[row] * (100 - col_disc[row]);
                    bpit.next();
                }

                res[yi].denom = denom;
                res[yi].numer = numer;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            wah_res = res;
            wah_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_total = ms(t0, t2) + q8_join_setup_ms;
            std::cout << "  WAH:  OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                wah_or_t.push_back(d_or);
                wah_and_t.push_back(d_and);
                wah_tot_t.push_back(d_total);
            }
        }

        // ========== EWAH (fast_logicalor) ==========
        if (run_ew()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ewah::EWAHBoolArray<uint64_t> ew_filt = ewah::fast_logicalor(
                ew_date_ptrs.size(), ew_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ewah::EWAHBoolArray<uint64_t> joined;
            ew_filt.logicaland(ew_join, joined);

            Q8Result res{};
            size_t total_rows = 0;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                ewah::EWAHBoolArray<uint64_t> year_filt;
                joined.logicaland(ew_year[yi], year_filt);

                int64_t denom = 0;
                size_t cnt = 0;
                for (auto it = year_filt.begin(); it != year_filt.end(); ++it) {
                    size_t row = *it;
                    denom += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }

                ewah::EWAHBoolArray<uint64_t> brazil_filt;
                year_filt.logicaland(ew_brazil, brazil_filt);
                int64_t numer = 0;
                for (auto it = brazil_filt.begin(); it != brazil_filt.end(); ++it) {
                    size_t row = *it;
                    numer += col_price[row] * (100 - col_disc[row]);
                }

                res[yi].denom = denom;
                res[yi].numer = numer;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            ew_res = res;
            ew_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_total = ms(t0, t2) + q8_join_setup_ms;
            std::cout << "  EW:   OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                ew_or_t.push_back(d_or);
                ew_and_t.push_back(d_and);
                ew_tot_t.push_back(d_total);
            }
        }

        // ========== Bitset (scalar) ==========
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bs_filt = bs_date[Q8_DATE_START].clone();
            for (int d = Q8_DATE_START + 1; d <= Q8_DATE_END; d++)
                bs::or_inplace(bs_filt, bs_date[d], false);
            auto t1 = std::chrono::high_resolution_clock::now();
            bs::and_inplace(bs_filt, bs_join, false);

            Q8Result res{};
            size_t total_rows = 0;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                bs::Bitmap year_filt = bs_filt.clone();
                bs::and_inplace(year_filt, bs_year[yi], false);

                int64_t denom = 0;
                size_t cnt = 0;
                for (size_t i = 0; i < year_filt.nwords; ++i) {
                    uint64_t w = year_filt.words[i];
                    const size_t base = i * 64;
                    while (w) {
                        size_t row = base + __builtin_ctzll(w);
                        if (row >= year_filt.nbits) break;
                        denom += col_price[row] * (100 - col_disc[row]);
                        cnt++;
                        w &= w - 1;
                    }
                }

                bs::Bitmap brazil_filt = year_filt.clone();
                bs::and_inplace(brazil_filt, bs_brazil, false);
                int64_t numer = 0;
                for (size_t i = 0; i < brazil_filt.nwords; ++i) {
                    uint64_t w = brazil_filt.words[i];
                    const size_t base = i * 64;
                    while (w) {
                        size_t row = base + __builtin_ctzll(w);
                        if (row >= brazil_filt.nbits) break;
                        numer += col_price[row] * (100 - col_disc[row]);
                        w &= w - 1;
                    }
                }

                res[yi].denom = denom;
                res[yi].numer = numer;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            bs_res = res;
            bs_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_total = ms(t0, t2) + q8_join_setup_ms;
            std::cout << "  BS:   OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                bs_or_t.push_back(d_or);
                bs_and_t.push_back(d_and);
                bs_tot_t.push_back(d_total);
            }
        }

        // ========== Bitset + AVX-512 ==========
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            bs::Bitmap bsa_filt = bs_date[Q8_DATE_START].clone();
            for (int d = Q8_DATE_START + 1; d <= Q8_DATE_END; d++)
                bs::or_inplace(bsa_filt, bs_date[d], true);
            auto t1 = std::chrono::high_resolution_clock::now();
            bs::and_inplace(bsa_filt, bs_join, true);

            Q8Result res{};
            size_t total_rows = 0;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                bs::Bitmap year_filt = bsa_filt.clone();
                bs::and_inplace(year_filt, bs_year[yi], true);

                int64_t denom = 0;
                size_t cnt = 0;
                for (size_t i = 0; i < year_filt.nwords; ++i) {
                    uint64_t w = year_filt.words[i];
                    const size_t base = i * 64;
                    while (w) {
                        size_t row = base + __builtin_ctzll(w);
                        if (row >= year_filt.nbits) break;
                        denom += col_price[row] * (100 - col_disc[row]);
                        cnt++;
                        w &= w - 1;
                    }
                }

                bs::Bitmap brazil_filt = year_filt.clone();
                bs::and_inplace(brazil_filt, bs_brazil, true);
                int64_t numer = 0;
                for (size_t i = 0; i < brazil_filt.nwords; ++i) {
                    uint64_t w = brazil_filt.words[i];
                    const size_t base = i * 64;
                    while (w) {
                        size_t row = base + __builtin_ctzll(w);
                        if (row >= brazil_filt.nbits) break;
                        numer += col_price[row] * (100 - col_disc[row]);
                        w &= w - 1;
                    }
                }

                res[yi].denom = denom;
                res[yi].numer = numer;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            bsa_res = res;
            bsa_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_total = ms(t0, t2) + q8_join_setup_ms;
            std::cout << "  BSA:  OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                bsa_or_t.push_back(d_or);
                bsa_and_t.push_back(d_and);
                bsa_tot_t.push_back(d_total);
            }
        }

        // ========== Concise (fast_logicalor) ==========
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ConciseSet<false> con_filt = ConciseSet<false>::fast_logicalor(
                con_date_ptrs.size(), con_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            con_filt = con_filt.logicaland(con_join);

            Q8Result res{};
            size_t total_rows = 0;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                ConciseSet<false> year_filt = con_filt.logicaland(con_year[yi]);
                int64_t denom = 0;
                size_t cnt = 0;
                for (auto it = year_filt.begin(); it != year_filt.end(); ++it) {
                    size_t row = *it;
                    denom += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }
                ConciseSet<false> brazil_filt = year_filt.logicaland(con_brazil);
                int64_t numer = 0;
                for (auto it = brazil_filt.begin(); it != brazil_filt.end(); ++it) {
                    size_t row = *it;
                    numer += col_price[row] * (100 - col_disc[row]);
                }
                res[yi].denom = denom;
                res[yi].numer = numer;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            con_res = res;
            con_rows = total_rows;

            double d_or = ms(t0, t1), d_and = ms(t1, t2), d_total = ms(t0, t2) + q8_join_setup_ms;
            std::cout << "  CON:  OR_date=" << d_or
                      << "  AND+Agg=" << d_and
                      << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                con_or_t.push_back(d_or);
                con_and_t.push_back(d_and);
                con_tot_t.push_back(d_total);
            }
        }

    } // end iterations

    // ============================================================
    // 3. Correctness validation
    //
    // DuckDB native SQL for Q8 yields 2 rows: (year, mkt_share).
    // Each backend's per-year (denom, numer) integer tuple must match
    // every other active backend exactly, and mkt_share = numer/denom
    // must agree with DuckDB's value to within 1e-6 (compensates the
    // pure floating-point division at the SQL side).
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q8 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "  Row counts:";
    if (run_wah()) std::cout << " WAH=" << wah_rows;
    if (run_cb())  std::cout << " CB="  << cb_rows;
    if (run_cr())  std::cout << " CR="  << cr_rows;
    if (run_crr()) std::cout << " CRR=" << crr_rows;
    if (run_ew())  std::cout << " EW="  << ew_rows;
    if (run_bs())  std::cout << " BS="  << bs_rows;
    if (run_bsa()) std::cout << " BSA=" << bsa_rows;
    if (run_con()) std::cout << " CON=" << con_rows;
    std::cout << std::endl;

    const Q8Result* print_src = nullptr;
    const char* print_src_label = "";
    if      (Q8_BM == Q8BmType::ALL || Q8_BM == Q8BmType::CB)  { print_src = &cb_res;  print_src_label = "ComBit"; }
    else if (Q8_BM == Q8BmType::WAH)                           { print_src = &wah_res; print_src_label = "WAH";    }
    else if (Q8_BM == Q8BmType::CR)                            { print_src = &cr_res;  print_src_label = "CRoaring"; }
    else if (Q8_BM == Q8BmType::CRR)                           { print_src = &crr_res; print_src_label = "CRoaring+Run"; }
    else if (Q8_BM == Q8BmType::EW)                            { print_src = &ew_res;  print_src_label = "EWAH"; }
    else if (Q8_BM == Q8BmType::BS)                            { print_src = &bs_res;  print_src_label = "Bitset"; }
    else if (Q8_BM == Q8BmType::BSA)                           { print_src = &bsa_res; print_src_label = "Bitset+AVX512"; }
    else if (Q8_BM == Q8BmType::CON)                           { print_src = &con_res; print_src_label = "Concise"; }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n  Q8 Results (mkt_share = sum(BRAZIL volume) / sum(volume), per year, "
              << print_src_label << " values):" << std::endl;
    std::cout << "  year  mkt_share" << std::endl;
    if (print_src) {
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
            std::cout << "  " << Q8_YEARS[yi] << " "
                      << std::setw(10) << q8_mkt_share((*print_src)[yi]) << std::endl;
        }
    }

    if (run_all()) {
        bool consistent = true;
        for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
            const auto& base = cb_res[yi];
            auto cmp = [&](const char* lbl, const Q8Year& y) {
                if (y.denom != base.denom || y.numer != base.numer) {
                    std::cout << "  *** MISMATCH for year " << Q8_YEARS[yi]
                              << " (" << lbl << ": denom=" << y.denom << " numer=" << y.numer
                              << " vs CB: denom=" << base.denom << " numer=" << base.numer << ") ***" << std::endl;
                    consistent = false;
                }
            };
            cmp("CR",  cr_res[yi]);
            cmp("CRR", crr_res[yi]);
            cmp("WAH", wah_res[yi]);
            cmp("EW",  ew_res[yi]);
            cmp("BS",  bs_res[yi]);
            cmp("BSA", bsa_res[yi]);
            cmp("CON", con_res[yi]);
        }
        std::cout << "  Consistency: " << (consistent ? "ALL MATCH" : "MISMATCH DETECTED") << std::endl;
    }

    // ============================================================
    // 4. DuckDB native SQL ground-truth
    //
    // Runs after the bitmap iteration loop so it does not pollute the
    // CPU cache.  Uses cast(... as DOUBLE) to avoid integer division,
    // exact same algebra as the spec's mkt_share formula.
    // ============================================================
    std::array<double, 2> gt_share = {-1, -1};
    int64_t gt_rows = -1;
    double gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT EXTRACT(YEAR FROM o_orderdate)::INTEGER AS yr, "
            "       SUM(CASE WHEN n2.n_name = 'BRAZIL' "
            "                THEN l_extendedprice * (1 - l_discount) "
            "                ELSE 0 END)::DOUBLE / "
            "       SUM(l_extendedprice * (1 - l_discount))::DOUBLE AS mkt_share, "
            "       count(*) AS rows "
            "FROM lineitem l, orders o, customer c, supplier s, part p, "
            "     nation n1, nation n2, region r "
            "WHERE l.l_orderkey = o.o_orderkey "
            "  AND l.l_partkey  = p.p_partkey "
            "  AND l.l_suppkey  = s.s_suppkey "
            "  AND o.o_custkey  = c.c_custkey "
            "  AND c.c_nationkey = n1.n_nationkey "
            "  AND n1.n_regionkey = r.r_regionkey "
            "  AND s.s_nationkey = n2.n_nationkey "
            "  AND r.r_name = 'AMERICA' "
            "  AND p.p_type = 'ECONOMY ANODIZED STEEL' "
            "  AND o.o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31' "
            "GROUP BY EXTRACT(YEAR FROM o_orderdate) "
            "ORDER BY yr";
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = con.Query(sql);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (result && !result->HasError() && result->RowCount() == 2) {
            gt_share[0] = result->GetValue(1, 0).GetValue<double>();
            gt_share[1] = result->GetValue(1, 1).GetValue<double>();
            gt_rows     = result->GetValue(2, 0).GetValue<int64_t>()
                        + result->GetValue(2, 1).GetValue<int64_t>();
            gt_sql_ms = ms(t0, t1);
        } else if (result && result->HasError()) {
            std::cerr << "[Baseline] SQL error: " << result->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    std::cout << std::fixed << std::setprecision(6);
    if (gt_rows >= 0) {
        std::cout << "\n[Baseline] DuckDB native SQL: " << gt_rows << " rows total"
                  << "  (single run: " << std::setprecision(2) << gt_sql_ms << " ms)" << std::endl;
        std::cout << std::setprecision(6);
        std::cout << "  SQL ground truth mkt_share: 1995=" << gt_share[0]
                  << "  1996=" << gt_share[1] << std::endl;
    } else {
        std::cout << "\n[Baseline] DuckDB SQL ground truth unavailable — skipping assert." << std::endl;
    }

    if (gt_rows >= 0) {
        // Row count assert.
        auto check_rows = [&](const char* label, bool active, size_t got) {
            if (!active) return;
            if (static_cast<int64_t>(got) == gt_rows) return;
            std::ostringstream oss;
            oss << "[FAIL] Q8 " << label << " total rows " << got
                << " != DuckDB SQL ground truth " << gt_rows
                << " — bitmap pipeline is incorrect";
            throw std::runtime_error(oss.str());
        };
        check_rows("WAH",           run_wah(), wah_rows);
        check_rows("ComBit",        run_cb(),  cb_rows);
        check_rows("CRoaring",      run_cr(),  cr_rows);
        check_rows("CRoaring+Run",  run_crr(), crr_rows);
        check_rows("EWAH",          run_ew(),  ew_rows);
        check_rows("Bitset",        run_bs(),  bs_rows);
        check_rows("Bitset+AVX512", run_bsa(), bsa_rows);
        check_rows("Concise",       run_con(), con_rows);

        // mkt_share assert (1e-6 tolerance — denom/numer integers, only
        // rounding-error in the final division).
        auto check_share = [&](const char* label, bool active, const Q8Result& res) {
            if (!active) return;
            for (size_t yi = 0; yi < Q8_YEARS.size(); yi++) {
                double got = q8_mkt_share(res[yi]);
                double diff = std::fabs(got - gt_share[yi]);
                if (diff > 1e-6) {
                    std::ostringstream oss;
                    oss << "[FAIL] Q8 " << label << " mkt_share[" << Q8_YEARS[yi]
                        << "] = " << got << " differs from SQL " << gt_share[yi]
                        << " by " << diff;
                    throw std::runtime_error(oss.str());
                }
            }
        };
        check_share("WAH",           run_wah(), wah_res);
        check_share("ComBit",        run_cb(),  cb_res);
        check_share("CRoaring",      run_cr(),  cr_res);
        check_share("CRoaring+Run",  run_crr(), crr_res);
        check_share("EWAH",          run_ew(),  ew_res);
        check_share("Bitset",        run_bs(),  bs_res);
        check_share("Bitset+AVX512", run_bsa(), bsa_res);
        check_share("Concise",       run_con(), con_res);

        std::cout << "[OK] all active backends match DuckDB SQL ground truth ("
                  << gt_rows << " rows total, mkt_share within 1e-6)." << std::endl;
    }

    // ============================================================
    // 5. Statistics summary
    // ============================================================
    auto cb_or_s  = bm_bench::compute_stats(cb_or_t);
    auto cb_and_s = bm_bench::compute_stats(cb_and_t);
    auto cb_tot_s = bm_bench::compute_stats(cb_tot_t);

    auto cr_or_s  = bm_bench::compute_stats(cr_or_t);
    auto cr_and_s = bm_bench::compute_stats(cr_and_t);
    auto cr_tot_s = bm_bench::compute_stats(cr_tot_t);

    auto crr_or_s  = bm_bench::compute_stats(crr_or_t);
    auto crr_and_s = bm_bench::compute_stats(crr_and_t);
    auto crr_tot_s = bm_bench::compute_stats(crr_tot_t);

    auto wah_or_s  = bm_bench::compute_stats(wah_or_t);
    auto wah_and_s = bm_bench::compute_stats(wah_and_t);
    auto wah_tot_s = bm_bench::compute_stats(wah_tot_t);

    auto ew_or_s  = bm_bench::compute_stats(ew_or_t);
    auto ew_and_s = bm_bench::compute_stats(ew_and_t);
    auto ew_tot_s = bm_bench::compute_stats(ew_tot_t);

    auto bs_or_s   = bm_bench::compute_stats(bs_or_t);
    auto bs_and_s  = bm_bench::compute_stats(bs_and_t);
    auto bs_tot_s  = bm_bench::compute_stats(bs_tot_t);

    auto bsa_or_s  = bm_bench::compute_stats(bsa_or_t);
    auto bsa_and_s = bm_bench::compute_stats(bsa_and_t);
    auto bsa_tot_s = bm_bench::compute_stats(bsa_tot_t);

    auto con_or_s  = bm_bench::compute_stats(con_or_t);
    auto con_and_s = bm_bench::compute_stats(con_and_t);
    auto con_tot_s = bm_bench::compute_stats(con_tot_t);

    int measured = Q8_ITERATIONS - Q8_WARMUP;
    std::cout << std::fixed << std::setprecision(2);

    if (run_all()) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q8 RESULTS (" << measured << " measured iterations, median +/- stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  CB (ms)         CR (ms)        CRR (ms)        WAH (ms)        EW (ms)" << std::endl;
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;

        auto print_row = [](const char* label, bm_bench::Stats& cb, bm_bench::Stats& cr,
                            bm_bench::Stats& crr, bm_bench::Stats& wah, bm_bench::Stats& ew) {
            std::cout << "  " << std::left << std::setw(14) << label
                      << std::right
                      << std::setw(8) << cb.median << " +/- " << std::setw(5) << cb.stddev
                      << "  " << std::setw(8) << cr.median << " +/- " << std::setw(5) << cr.stddev
                      << "  " << std::setw(8) << crr.median << " +/- " << std::setw(5) << crr.stddev
                      << "  " << std::setw(8) << wah.median << " +/- " << std::setw(5) << wah.stddev
                      << "  " << std::setw(8) << ew.median << " +/- " << std::setw(5) << ew.stddev
                      << std::endl;
        };

        print_row("OR_date", cb_or_s,  cr_or_s,  crr_or_s,  wah_or_s,  ew_or_s);
        print_row("AND+Agg", cb_and_s, cr_and_s, crr_and_s, wah_and_s, ew_and_s);
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL",   cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s);

        std::cout << "\n  CB rows:  " << cb_rows
                  << "  CR rows:  " << cr_rows
                  << "  CRR rows: " << crr_rows
                  << "  WAH rows: " << wah_rows
                  << "  EW rows:  " << ew_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q8 BASELINE BACKENDS (no compression / Concise)" << std::endl;
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
        print_baseline_row("OR_date", wah_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
        print_baseline_row("AND+Agg", wah_and_s, bs_and_s, bsa_and_s, con_and_s);
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        print_baseline_row("TOTAL",   wah_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);

        std::cout << "\n  BS rows:  " << bs_rows
                  << "  BSA rows: " << bsa_rows
                  << "  CON rows: " << con_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;
    } else {
        bm_bench::Stats *sel_or = nullptr, *sel_and = nullptr, *sel_tot = nullptr;
        size_t sel_rows = 0;
        switch (Q8_BM) {
            case Q8BmType::WAH: sel_or = &wah_or_s; sel_and = &wah_and_s; sel_tot = &wah_tot_s; sel_rows = wah_rows; break;
            case Q8BmType::CB:  sel_or = &cb_or_s;  sel_and = &cb_and_s;  sel_tot = &cb_tot_s;  sel_rows = cb_rows;  break;
            case Q8BmType::CR:  sel_or = &cr_or_s;  sel_and = &cr_and_s;  sel_tot = &cr_tot_s;  sel_rows = cr_rows;  break;
            case Q8BmType::CRR: sel_or = &crr_or_s; sel_and = &crr_and_s; sel_tot = &crr_tot_s; sel_rows = crr_rows; break;
            case Q8BmType::EW:  sel_or = &ew_or_s;  sel_and = &ew_and_s;  sel_tot = &ew_tot_s;  sel_rows = ew_rows;  break;
            case Q8BmType::BS:  sel_or = &bs_or_s;  sel_and = &bs_and_s;  sel_tot = &bs_tot_s;  sel_rows = bs_rows;  break;
            case Q8BmType::BSA: sel_or = &bsa_or_s; sel_and = &bsa_and_s; sel_tot = &bsa_tot_s; sel_rows = bsa_rows; break;
            case Q8BmType::CON: sel_or = &con_or_s; sel_and = &con_and_s; sel_tot = &con_tot_s; sel_rows = con_rows; break;
            case Q8BmType::ALL: break;
        }

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q8 RESULTS — " << q8_bm_label() << " only ("
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
            print_single("OR_date", *sel_or);
            print_single("AND+Agg", *sel_and);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            print_single("TOTAL",   *sel_tot);
            std::cout << "\n  " << q8_bm_label() << " rows: " << sel_rows << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }

    // ============================================================
    // 6. CSV export — ALL mode only
    // ============================================================
    if (run_all()) {
        std::string sf_label = q8_get_sf_label();
        std::string csv_path = "q8_results_" + sf_label + ".csv";
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

            csv_row("OR_date", cb_or_s,  cr_or_s,  crr_or_s,  wah_or_s,  ew_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
            csv_row("AND+Agg", cb_and_s, cr_and_s, crr_and_s, wah_and_s, ew_and_s, bs_and_s, bsa_and_s, con_and_s);
            csv_row("TOTAL",   cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);

            csv.close();
            std::cout << "  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
