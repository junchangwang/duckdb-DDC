// TPC-H Q5 — Local Supplier Volume Query (BitEngine pattern, all backends)
//
//   SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
//   FROM   customer, orders, lineitem, supplier, nation, region
//   WHERE  c_custkey = o_custkey AND l_orderkey = o_orderkey AND
//          l_suppkey = s_suppkey AND c_nationkey = s_nationkey AND
//          s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND
//          r_name = 'ASIA' AND o_orderdate ∈ [1994-01-01, 1995-01-01)
//   GROUP BY n_name ORDER BY revenue DESC.
//
// Mirrors teacher's BMTPCH_Q5 in BitEngine branch:
//   Phase A — small-table semi-joins (region → nation → customer →
//             orders → supplier).  Produces order_nation_map and
//             supp_nation_map.
//   Phase B — per-value indexed bitmap on lineitem (built once via
//             IndexedBitmap, mirroring rabit::Rabit per-value Btvs[]).
//             Backend-specific: ComBit uses SparseComBit (sparse-segment
//             storage); CRoaring uses native sparse container array;
//             WAH/EWAH/Concise use RLE.
//   Phase C — bitmap multi-OR + AND join with lineitem (per-backend timed):
//                  btv_or  = OR over orderkeys in order_nation_map
//                  btv_supp= OR over suppkeys in supp_nation_map
//                  filter  = btv_or AND btv_supp
//                  walk filter, aggregate revenue per nation.
//
// All time (Phase A + Phase B build + Phase C) is reported in Total — no
// SQL JOIN engine work hidden.
//
// BS / BSA skipped: per-orderkey uncompressed bitset would need 15M ×
// 7.5 MB = 112 TB; not implementable.

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
#include "execution/tpch/indexed_bitmap.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {

using clock_t_ns = std::chrono::high_resolution_clock;
static inline double q5_ms(clock_t_ns::time_point a, clock_t_ns::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

// --- Backend selection (DEBIT_BM env) ---
using Q5BmType = bm_bench::Backend;
static const Q5BmType Q5_BM = bm_bench::parse_backend("Q5_BM");
static bool run_all() { return Q5_BM == Q5BmType::ALL; }
static bool run_wah() { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::WAH; }
static bool run_cb()  { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CB;  }
static bool run_cr()  { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CR;  }
static bool run_crr() { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CRR; }
static bool run_ew()  { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::EW;  }
static bool run_con() { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CON; }
static const char* q5_bm_label() { return bm_bench::backend_label(Q5_BM); }
static std::string q5_get_sf_label() { return bm_bench::sf_label(); }

static const int Q5_ITERATIONS = bm_bench::iter_count(5);
static const int Q5_WARMUP     = bm_bench::warmup_count(1);

static std::once_flag q5_once_flag;

void BMTableScan::BMTPCH_Q5(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q5_once_flag, [&]() {

    using clock = clock_t_ns;
    bm_bench::warn_if_sf1();

    auto& region_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "region");
    auto& nation_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "nation");
    auto& customer_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "customer");
    auto& orders_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
    auto& supplier_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "supplier");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q5 (BitEngine pattern) — " << q5_bm_label()
              << " (" << q5_get_sf_label() << ")" << std::endl;
    std::cout << "  All Phase A + B + C costs counted in Total." << std::endl;
    std::cout << "================================================================" << std::endl;

    auto a0 = clock::now();

    std::unordered_set<int32_t> r_regionkey_set;
    {
        auto& tx = DuckTransaction::Get(context.client, region_table.catalog);
        TableScanState ss;
        vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(1)};
        region_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
        vector<LogicalType> types = {
            region_table.GetColumns().GetColumnTypes()[0],
            region_table.GetColumns().GetColumnTypes()[1]
        };
        while (true) {
            DataChunk chunk; chunk.Initialize(context.client, types);
            region_table.GetStorage().Scan(tx, chunk, ss);
            if (chunk.size() == 0) break;
            auto rk = FlatVector::GetData<int32_t>(chunk.data[0]);
            auto rn = reinterpret_cast<string_t*>(chunk.data[1].GetData());
            for (idx_t i = 0; i < chunk.size(); i++)
                if (rn[i].GetString() == "ASIA")
                    r_regionkey_set.insert(rk[i]);
        }
    }

    std::unordered_set<int32_t> n_nationkey_set;
    std::unordered_map<int32_t, std::string> n_name_map;
    {
        auto& tx = DuckTransaction::Get(context.client, nation_table.catalog);
        TableScanState ss;
        vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(1), StorageIndex(2)};
        nation_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
        vector<LogicalType> types = {
            nation_table.GetColumns().GetColumnTypes()[0],
            nation_table.GetColumns().GetColumnTypes()[1],
            nation_table.GetColumns().GetColumnTypes()[2]
        };
        while (true) {
            DataChunk chunk; chunk.Initialize(context.client, types);
            nation_table.GetStorage().Scan(tx, chunk, ss);
            if (chunk.size() == 0) break;
            auto nk = FlatVector::GetData<int32_t>(chunk.data[0]);
            auto nn = reinterpret_cast<string_t*>(chunk.data[1].GetData());
            auto rk = FlatVector::GetData<int32_t>(chunk.data[2]);
            for (idx_t i = 0; i < chunk.size(); i++)
                if (r_regionkey_set.count(rk[i])) {
                    n_nationkey_set.insert(nk[i]);
                    n_name_map[nk[i]] = nn[i].GetString();
                }
        }
    }

    std::unordered_map<int64_t, int32_t> customer_nation_map;
    {
        auto& tx = DuckTransaction::Get(context.client, customer_table.catalog);
        TableScanState ss;
        vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(3)};
        customer_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
        vector<LogicalType> types = {
            customer_table.GetColumns().GetColumnTypes()[0],
            customer_table.GetColumns().GetColumnTypes()[3]
        };
        while (true) {
            DataChunk chunk; chunk.Initialize(context.client, types);
            customer_table.GetStorage().Scan(tx, chunk, ss);
            if (chunk.size() == 0) break;
            auto ck = FlatVector::GetData<int64_t>(chunk.data[0]);
            auto nk = FlatVector::GetData<int32_t>(chunk.data[1]);
            for (idx_t i = 0; i < chunk.size(); i++)
                if (n_nationkey_set.count(nk[i]))
                    customer_nation_map[ck[i]] = nk[i];
        }
    }

    constexpr int32_t Q5_DATE_LO = 8766;
    constexpr int32_t Q5_DATE_HI = 9131;

    std::unordered_map<int64_t, int32_t> order_nation_map;
    {
        auto& tx = DuckTransaction::Get(context.client, orders_table.catalog);
        TableScanState ss;
        vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(1), StorageIndex(4)};
        orders_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
        vector<LogicalType> types = {
            orders_table.GetColumns().GetColumnTypes()[0],
            orders_table.GetColumns().GetColumnTypes()[1],
            orders_table.GetColumns().GetColumnTypes()[4]
        };
        while (true) {
            DataChunk chunk; chunk.Initialize(context.client, types);
            orders_table.GetStorage().Scan(tx, chunk, ss);
            if (chunk.size() == 0) break;
            auto okey  = FlatVector::GetData<int64_t>(chunk.data[0]);
            auto ckey  = FlatVector::GetData<int64_t>(chunk.data[1]);
            auto odate = FlatVector::GetData<int32_t>(chunk.data[2]);
            for (idx_t i = 0; i < chunk.size(); i++) {
                if (odate[i] < Q5_DATE_LO || odate[i] >= Q5_DATE_HI) continue;
                auto it = customer_nation_map.find(ckey[i]);
                if (it != customer_nation_map.end())
                    order_nation_map[okey[i]] = it->second;
            }
        }
    }

    std::unordered_map<int64_t, int32_t> supp_nation_map;
    {
        auto& tx = DuckTransaction::Get(context.client, supplier_table.catalog);
        TableScanState ss;
        vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(3)};
        supplier_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
        vector<LogicalType> types = {
            supplier_table.GetColumns().GetColumnTypes()[0],
            supplier_table.GetColumns().GetColumnTypes()[3]
        };
        while (true) {
            DataChunk chunk; chunk.Initialize(context.client, types);
            supplier_table.GetStorage().Scan(tx, chunk, ss);
            if (chunk.size() == 0) break;
            auto sk = FlatVector::GetData<int64_t>(chunk.data[0]);
            auto nk = FlatVector::GetData<int32_t>(chunk.data[1]);
            for (idx_t i = 0; i < chunk.size(); i++)
                if (n_nationkey_set.count(nk[i]))
                    supp_nation_map[sk[i]] = nk[i];
        }
    }

    auto a1 = clock::now();
    double phase_a_ms = q5_ms(a0, a1);
    std::cout << "  [Phase A] small-table joins: " << phase_a_ms << " ms ("
              << order_nation_map.size() << " qualifying orderkeys, "
              << supp_nation_map.size()  << " qualifying suppkeys)" << std::endl;

    // -----------------------------------------------------------------------
    // Phase B: lineitem column scan + per-backend index build
    // -----------------------------------------------------------------------
    auto b0 = clock::now();
    std::vector<int64_t> col_okey, col_skey, col_price, col_disc;
    {
        auto& tx = DuckTransaction::Get(context.client, lineitem_table.catalog);
        TableScanState ss;
        vector<StorageIndex> col_ids = {
            StorageIndex(0), StorageIndex(2), StorageIndex(5), StorageIndex(6)
        };
        lineitem_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
        vector<LogicalType> types = {
            lineitem_table.GetColumns().GetColumnTypes()[0],
            lineitem_table.GetColumns().GetColumnTypes()[2],
            lineitem_table.GetColumns().GetColumnTypes()[5],
            lineitem_table.GetColumns().GetColumnTypes()[6]
        };
        while (true) {
            DataChunk chunk; chunk.Initialize(context.client, types);
            lineitem_table.GetStorage().Scan(tx, chunk, ss);
            if (chunk.size() == 0) break;
            auto ok = FlatVector::GetData<int64_t>(chunk.data[0]);
            auto sk = FlatVector::GetData<int64_t>(chunk.data[1]);
            auto pr = FlatVector::GetData<int64_t>(chunk.data[2]);
            auto dc = FlatVector::GetData<int64_t>(chunk.data[3]);
            for (idx_t i = 0; i < chunk.size(); i++) {
                col_okey.push_back(ok[i]);
                col_skey.push_back(sk[i]);
                col_price.push_back(pr[i]);
                col_disc.push_back(dc[i]);
            }
        }
    }
    size_t num_rows = col_okey.size();
    auto b1 = clock::now();
    double phase_b_setup_ms = q5_ms(b0, b1);
    std::cout << "  [Phase B0] lineitem scan (" << num_rows
              << " rows): " << phase_b_setup_ms << " ms" << std::endl;

    bm_index::IndexedComBit   idx_cb_okey,  idx_cb_skey;
    bm_index::IndexedCRoaring idx_cr_okey,  idx_cr_skey;
    bm_index::IndexedCRoaring idx_crr_okey, idx_crr_skey;
    bm_index::IndexedWAH      idx_wah_okey, idx_wah_skey;
    bm_index::IndexedEWAH     idx_ew_okey,  idx_ew_skey;
    bm_index::IndexedConcise  idx_con_okey, idx_con_skey;

    double cb_build_ms = 0, cr_build_ms = 0, crr_build_ms = 0,
           wah_build_ms = 0, ew_build_ms = 0, con_build_ms = 0;

    if (run_cb()) {
        auto t0 = clock::now();
        idx_cb_okey.build(col_okey, num_rows);
        idx_cb_skey.build(col_skey, num_rows);
        cb_build_ms = q5_ms(t0, clock::now());
        std::cout << "  [Phase B1] ComBit index build: " << cb_build_ms << " ms ("
                  << idx_cb_okey.num_keys() << " orderkeys, "
                  << idx_cb_skey.num_keys() << " suppkeys, "
                  << (idx_cb_okey.storage_bytes() + idx_cb_skey.storage_bytes()) / 1e6 << " MB)" << std::endl;
    }
    if (run_cr()) {
        auto t0 = clock::now();
        idx_cr_okey.build(col_okey, num_rows, false);
        idx_cr_skey.build(col_skey, num_rows, false);
        cr_build_ms = q5_ms(t0, clock::now());
        std::cout << "  [Phase B1] CRoaring index build: " << cr_build_ms << " ms" << std::endl;
    }
    if (run_crr()) {
        auto t0 = clock::now();
        idx_crr_okey.build(col_okey, num_rows, true);
        idx_crr_skey.build(col_skey, num_rows, true);
        crr_build_ms = q5_ms(t0, clock::now());
        std::cout << "  [Phase B1] CRoaring+Run index build: " << crr_build_ms << " ms" << std::endl;
    }
    if (run_wah()) {
        auto t0 = clock::now();
        idx_wah_okey.build(col_okey, num_rows);
        idx_wah_skey.build(col_skey, num_rows);
        wah_build_ms = q5_ms(t0, clock::now());
        std::cout << "  [Phase B1] WAH index build: " << wah_build_ms << " ms" << std::endl;
    }
    if (run_ew()) {
        auto t0 = clock::now();
        idx_ew_okey.build(col_okey, num_rows);
        idx_ew_skey.build(col_skey, num_rows);
        ew_build_ms = q5_ms(t0, clock::now());
        std::cout << "  [Phase B1] EWAH index build: " << ew_build_ms << " ms" << std::endl;
    }
    if (run_con()) {
        auto t0 = clock::now();
        idx_con_okey.build(col_okey, num_rows);
        idx_con_skey.build(col_skey, num_rows);
        con_build_ms = q5_ms(t0, clock::now());
        std::cout << "  [Phase B1] Concise index build: " << con_build_ms << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Phase C: bitmap multi-OR + AND join, per-backend timed
    // -----------------------------------------------------------------------
    std::vector<double> cb_or_t, cb_and_t, cb_agg_t, cb_tot_t;
    std::vector<double> cr_or_t, cr_and_t, cr_agg_t, cr_tot_t;
    std::vector<double> crr_or_t, crr_and_t, crr_agg_t, crr_tot_t;
    std::vector<double> wah_or_t, wah_and_t, wah_agg_t, wah_tot_t;
    std::vector<double> ew_or_t, ew_and_t, ew_agg_t, ew_tot_t;
    std::vector<double> con_or_t, con_and_t, con_agg_t, con_tot_t;

    std::map<std::string, int64_t> cb_ans, cr_ans, crr_ans, wah_ans, ew_ans, con_ans;
    size_t cb_rows = 0, cr_rows = 0, crr_rows = 0, wah_rows = 0, ew_rows = 0, con_rows = 0;

    for (int iter = 0; iter < Q5_ITERATIONS; iter++) {
        bool warm = iter < Q5_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q5_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        if (run_cb()) {
            auto t0 = clock::now();
            std::vector<bool> empty_bits(num_rows, false);
            ComBit btv_or  = ComBit::compress(empty_bits, false, idx_cb_okey.segment_bits());
            ComBit btv_supp= ComBit::compress(empty_bits, false, idx_cb_skey.segment_bits());
            for (auto& [okey, _] : order_nation_map) idx_cb_okey.apply_or_to(btv_or, okey);
            auto t1 = clock::now();
            for (auto& [skey, _] : supp_nation_map) idx_cb_skey.apply_or_to(btv_supp, skey);
            btv_or &= btv_supp;
            auto t2 = clock::now();
            std::map<std::string, int64_t> ans;
            size_t cnt = 0;
            btv_or.for_each_literal([&](uint32_t word_pos, uint8_t val) {
                size_t rbase = static_cast<size_t>(word_pos) * 8;
                const auto& e = bm_bench::byte_lut[val];
                for (int k = 0; k < e.count; k++) {
                    size_t r = rbase + e.pos[k];
                    if (r >= num_rows) break;
                    auto on_it = order_nation_map.find(col_okey[r]);
                    auto sn_it = supp_nation_map.find(col_skey[r]);
                    if (on_it == order_nation_map.end() ||
                        sn_it == supp_nation_map.end() ||
                        on_it->second != sn_it->second) continue;
                    ans[n_name_map[on_it->second]] +=
                        col_price[r] * (100 - col_disc[r]);
                    cnt++;
                }
            });
            auto t3 = clock::now();
            cb_ans = ans;  cb_rows = cnt;
            double d_or = q5_ms(t0, t1), d_and = q5_ms(t1, t2),
                   d_agg = q5_ms(t2, t3),
                   d_tot = q5_ms(t0, t3) + phase_a_ms + phase_b_setup_ms + cb_build_ms;
            std::cout << "  CB:   OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warm) { cb_or_t.push_back(d_or); cb_and_t.push_back(d_and);
                          cb_agg_t.push_back(d_agg); cb_tot_t.push_back(d_tot); }
        }

        if (run_cr()) {
            auto t0 = clock::now();
            roaring::Roaring btv_or, btv_supp;
            for (auto& [okey, _] : order_nation_map) idx_cr_okey.apply_or_to(btv_or, okey);
            auto t1 = clock::now();
            for (auto& [skey, _] : supp_nation_map) idx_cr_skey.apply_or_to(btv_supp, skey);
            roaring::Roaring filt = btv_or & btv_supp;
            auto t2 = clock::now();
            std::map<std::string, int64_t> ans;
            size_t cnt = 0;
            for (auto rit = filt.begin(); rit != filt.end(); ++rit) {
                size_t r = *rit;
                if (r >= num_rows) break;
                auto on_it = order_nation_map.find(col_okey[r]);
                auto sn_it = supp_nation_map.find(col_skey[r]);
                if (on_it == order_nation_map.end() ||
                    sn_it == supp_nation_map.end() ||
                    on_it->second != sn_it->second) continue;
                ans[n_name_map[on_it->second]] +=
                    col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t3 = clock::now();
            cr_ans = ans;  cr_rows = cnt;
            double d_or = q5_ms(t0, t1), d_and = q5_ms(t1, t2),
                   d_agg = q5_ms(t2, t3),
                   d_tot = q5_ms(t0, t3) + phase_a_ms + phase_b_setup_ms + cr_build_ms;
            std::cout << "  CR:   OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warm) { cr_or_t.push_back(d_or); cr_and_t.push_back(d_and);
                          cr_agg_t.push_back(d_agg); cr_tot_t.push_back(d_tot); }
        }

        if (run_crr()) {
            auto t0 = clock::now();
            roaring::Roaring btv_or, btv_supp;
            for (auto& [okey, _] : order_nation_map) idx_crr_okey.apply_or_to(btv_or, okey);
            auto t1 = clock::now();
            for (auto& [skey, _] : supp_nation_map) idx_crr_skey.apply_or_to(btv_supp, skey);
            roaring::Roaring filt = btv_or & btv_supp;
            auto t2 = clock::now();
            std::map<std::string, int64_t> ans;
            size_t cnt = 0;
            for (auto rit = filt.begin(); rit != filt.end(); ++rit) {
                size_t r = *rit;
                if (r >= num_rows) break;
                auto on_it = order_nation_map.find(col_okey[r]);
                auto sn_it = supp_nation_map.find(col_skey[r]);
                if (on_it == order_nation_map.end() ||
                    sn_it == supp_nation_map.end() ||
                    on_it->second != sn_it->second) continue;
                ans[n_name_map[on_it->second]] +=
                    col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t3 = clock::now();
            crr_ans = ans;  crr_rows = cnt;
            double d_or = q5_ms(t0, t1), d_and = q5_ms(t1, t2),
                   d_agg = q5_ms(t2, t3),
                   d_tot = q5_ms(t0, t3) + phase_a_ms + phase_b_setup_ms + crr_build_ms;
            std::cout << "  CRR:  OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warm) { crr_or_t.push_back(d_or); crr_and_t.push_back(d_and);
                         crr_agg_t.push_back(d_agg); crr_tot_t.push_back(d_tot); }
        }

        if (run_wah()) {
            auto t0 = clock::now();
            ibis::bitvector btv_or, btv_supp;
            for (auto& [okey, _] : order_nation_map) idx_wah_okey.apply_or_to(btv_or, okey);
            auto t1 = clock::now();
            for (auto& [skey, _] : supp_nation_map) idx_wah_skey.apply_or_to(btv_supp, skey);
            ibis::bitvector filt; filt.copy(btv_or); filt &= btv_supp;
            auto t2 = clock::now();
            std::map<std::string, int64_t> ans;
            size_t cnt = 0;
            ibis::bitvector::pit pit(filt);
            while (*pit != 0xFFFFFFFFU) {
                size_t r = *pit;
                if (r < num_rows) {
                    auto on_it = order_nation_map.find(col_okey[r]);
                    auto sn_it = supp_nation_map.find(col_skey[r]);
                    if (on_it != order_nation_map.end() &&
                        sn_it != supp_nation_map.end() &&
                        on_it->second == sn_it->second) {
                        ans[n_name_map[on_it->second]] +=
                            col_price[r] * (100 - col_disc[r]);
                        cnt++;
                    }
                }
                pit.next();
            }
            auto t3 = clock::now();
            wah_ans = ans;  wah_rows = cnt;
            double d_or = q5_ms(t0, t1), d_and = q5_ms(t1, t2),
                   d_agg = q5_ms(t2, t3),
                   d_tot = q5_ms(t0, t3) + phase_a_ms + phase_b_setup_ms + wah_build_ms;
            std::cout << "  WAH:  OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warm) { wah_or_t.push_back(d_or); wah_and_t.push_back(d_and);
                         wah_agg_t.push_back(d_agg); wah_tot_t.push_back(d_tot); }
        }

        if (run_ew()) {
            auto t0 = clock::now();
            ewah::EWAHBoolArray<uint64_t> btv_or, btv_supp;
            for (auto& [okey, _] : order_nation_map) idx_ew_okey.apply_or_to(btv_or, okey);
            auto t1 = clock::now();
            for (auto& [skey, _] : supp_nation_map) idx_ew_skey.apply_or_to(btv_supp, skey);
            ewah::EWAHBoolArray<uint64_t> filt;
            btv_or.logicaland(btv_supp, filt);
            auto t2 = clock::now();
            std::map<std::string, int64_t> ans;
            size_t cnt = 0;
            for (auto rit = filt.begin(); rit != filt.end(); ++rit) {
                size_t r = *rit;
                if (r >= num_rows) break;
                auto on_it = order_nation_map.find(col_okey[r]);
                auto sn_it = supp_nation_map.find(col_skey[r]);
                if (on_it == order_nation_map.end() ||
                    sn_it == supp_nation_map.end() ||
                    on_it->second != sn_it->second) continue;
                ans[n_name_map[on_it->second]] +=
                    col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t3 = clock::now();
            ew_ans = ans;  ew_rows = cnt;
            double d_or = q5_ms(t0, t1), d_and = q5_ms(t1, t2),
                   d_agg = q5_ms(t2, t3),
                   d_tot = q5_ms(t0, t3) + phase_a_ms + phase_b_setup_ms + ew_build_ms;
            std::cout << "  EW:   OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warm) { ew_or_t.push_back(d_or); ew_and_t.push_back(d_and);
                         ew_agg_t.push_back(d_agg); ew_tot_t.push_back(d_tot); }
        }

        if (run_con()) {
            auto t0 = clock::now();
            ConciseSet<false> btv_or, btv_supp;
            for (auto& [okey, _] : order_nation_map) idx_con_okey.apply_or_to(btv_or, okey);
            auto t1 = clock::now();
            for (auto& [skey, _] : supp_nation_map) idx_con_skey.apply_or_to(btv_supp, skey);
            ConciseSet<false> filt = btv_or.logicaland(btv_supp);
            auto t2 = clock::now();
            std::map<std::string, int64_t> ans;
            size_t cnt = 0;
            for (auto rit = filt.begin(); rit != filt.end(); ++rit) {
                size_t r = *rit;
                if (r >= num_rows) break;
                auto on_it = order_nation_map.find(col_okey[r]);
                auto sn_it = supp_nation_map.find(col_skey[r]);
                if (on_it == order_nation_map.end() ||
                    sn_it == supp_nation_map.end() ||
                    on_it->second != sn_it->second) continue;
                ans[n_name_map[on_it->second]] +=
                    col_price[r] * (100 - col_disc[r]);
                cnt++;
            }
            auto t3 = clock::now();
            con_ans = ans;  con_rows = cnt;
            double d_or = q5_ms(t0, t1), d_and = q5_ms(t1, t2),
                   d_agg = q5_ms(t2, t3),
                   d_tot = q5_ms(t0, t3) + phase_a_ms + phase_b_setup_ms + con_build_ms;
            std::cout << "  CON:  OR=" << d_or << "  AND=" << d_and
                      << "  Agg=" << d_agg << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warm) { con_or_t.push_back(d_or); con_and_t.push_back(d_and);
                         con_agg_t.push_back(d_agg); con_tot_t.push_back(d_tot); }
        }
    }

    // -----------------------------------------------------------------------
    // Validate vs DuckDB SQL
    // -----------------------------------------------------------------------
    bool gt_ok = false;
    std::map<std::string, double> gt_revenue;
    {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT n.n_name, "
            "       CAST(sum(l.l_extendedprice * (1 - l.l_discount)) AS DOUBLE) AS revenue "
            "FROM   customer c, orders o, lineitem l, supplier s, nation n, region r "
            "WHERE  c.c_custkey  = o.o_custkey AND l.l_orderkey = o.o_orderkey "
            "  AND  l.l_suppkey  = s.s_suppkey AND c.c_nationkey = s.s_nationkey "
            "  AND  s.s_nationkey= n.n_nationkey AND n.n_regionkey = r.r_regionkey "
            "  AND  r.r_name = 'ASIA' "
            "  AND  o.o_orderdate >= DATE '1994-01-01' "
            "  AND  o.o_orderdate <  DATE '1995-01-01' "
            "GROUP BY n.n_name ORDER BY revenue DESC";
        auto r = con.Query(sql);
        if (r && !r->HasError()) {
            for (idx_t i = 0; i < r->RowCount(); i++) {
                gt_revenue[r->GetValue(0, i).GetValue<std::string>()] =
                    r->GetValue(1, i).GetValue<double>();
            }
            gt_ok = true;
        }
    }

    auto check_match = [&](const char* tag, const std::map<std::string, int64_t>& ans, bool active) {
        if (!active || !gt_ok) return;
        if (ans.size() != gt_revenue.size()) {
            std::ostringstream e;
            e << "[FAIL] Q5 " << tag << " produced " << ans.size()
              << " nations, SQL has " << gt_revenue.size();
            throw std::runtime_error(e.str());
        }
        for (auto& [name, rev_fp] : ans) {
            double our = double(rev_fp) / 10000;
            auto it = gt_revenue.find(name);
            if (it == gt_revenue.end() || std::fabs(our - it->second) > 0.01) {
                std::ostringstream e;
                e << "[FAIL] Q5 " << tag << " nation=" << name
                  << " our=" << our
                  << " sql=" << (it == gt_revenue.end() ? -1 : it->second);
                throw std::runtime_error(e.str());
            }
        }
    };
    check_match("CB",  cb_ans,  run_cb());
    check_match("CR",  cr_ans,  run_cr());
    check_match("CRR", crr_ans, run_crr());
    check_match("WAH", wah_ans, run_wah());
    check_match("EW",  ew_ans,  run_ew());
    check_match("CON", con_ans, run_con());
    if (gt_ok)
        std::cout << "\n[OK] all active backends match DuckDB SQL ground truth ("
                  << gt_revenue.size() << " nations)." << std::endl;

    const std::map<std::string, int64_t>* canonical = nullptr;
    const char* canonical_label = "";
    if      (!cb_ans.empty())  { canonical = &cb_ans;  canonical_label = "ComBit"; }
    else if (!cr_ans.empty())  { canonical = &cr_ans;  canonical_label = "CRoaring"; }
    else if (!crr_ans.empty()) { canonical = &crr_ans; canonical_label = "CRoaring+Run"; }
    else if (!wah_ans.empty()) { canonical = &wah_ans; canonical_label = "WAH"; }
    else if (!ew_ans.empty())  { canonical = &ew_ans;  canonical_label = "EWAH"; }
    else if (!con_ans.empty()) { canonical = &con_ans; canonical_label = "Concise"; }
    if (canonical && !canonical->empty()) {
        std::vector<std::pair<int64_t, std::string>> rows;
        for (auto& [name, rev] : *canonical) rows.push_back({rev, name});
        std::sort(rows.rbegin(), rows.rend());
        std::cout << "\n  Q5 Results (revenue DESC, source=" << canonical_label << "):" << std::endl;
        std::cout << "  nation                     revenue" << std::endl;
        for (auto& [rev, name] : rows)
            std::cout << "  " << std::left << std::setw(20) << name
                      << std::fixed << std::setprecision(4)
                      << (double(rev) / 10000) << std::endl;
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto cb_or_s=stats(cb_or_t), cb_and_s=stats(cb_and_t), cb_agg_s=stats(cb_agg_t), cb_tot_s=stats(cb_tot_t);
    auto cr_or_s=stats(cr_or_t), cr_and_s=stats(cr_and_t), cr_agg_s=stats(cr_agg_t), cr_tot_s=stats(cr_tot_t);
    auto crr_or_s=stats(crr_or_t),crr_and_s=stats(crr_and_t),crr_agg_s=stats(crr_agg_t),crr_tot_s=stats(crr_tot_t);
    auto wah_or_s=stats(wah_or_t),wah_and_s=stats(wah_and_t),wah_agg_s=stats(wah_agg_t),wah_tot_s=stats(wah_tot_t);
    auto ew_or_s=stats(ew_or_t), ew_and_s=stats(ew_and_t), ew_agg_s=stats(ew_agg_t), ew_tot_s=stats(ew_tot_t);
    auto con_or_s=stats(con_or_t),con_and_s=stats(con_and_t),con_agg_s=stats(con_agg_t),con_tot_s=stats(con_tot_t);
    int measured = std::max(0, Q5_ITERATIONS - Q5_WARMUP);

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q5 RESULTS (" << measured << " measured iter, median +/- stddev). Total includes Phases A+B+C." << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "                  CB              CR              CRR             WAH             EW              CON" << std::endl;
    auto pr = [](const char* l, bm_bench::Stats& a, bm_bench::Stats& b, bm_bench::Stats& c,
                                bm_bench::Stats& d, bm_bench::Stats& e, bm_bench::Stats& f) {
        std::cout << "  " << std::left << std::setw(8) << l << std::right
                  << std::setw(10) << a.median << " +/- " << std::setw(5) << a.stddev
                  << "  " << std::setw(10) << b.median << " +/- " << std::setw(5) << b.stddev
                  << "  " << std::setw(10) << c.median << " +/- " << std::setw(5) << c.stddev
                  << "  " << std::setw(10) << d.median << " +/- " << std::setw(5) << d.stddev
                  << "  " << std::setw(10) << e.median << " +/- " << std::setw(5) << e.stddev
                  << "  " << std::setw(10) << f.median << " +/- " << std::setw(5) << f.stddev
                  << std::endl;
    };
    pr("OR",    cb_or_s,  cr_or_s,  crr_or_s,  wah_or_s,  ew_or_s,  con_or_s);
    pr("AND",   cb_and_s, cr_and_s, crr_and_s, wah_and_s, ew_and_s, con_and_s);
    pr("Agg",   cb_agg_s, cr_agg_s, crr_agg_s, wah_agg_s, ew_agg_s, con_agg_s);
    pr("TOTAL", cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s, con_tot_s);
    std::cout << "================================================================\n" << std::endl;

    std::string sf = q5_get_sf_label();
    std::ofstream csv("q5_results_" + sf + ".csv");
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
        bm_bench::Stats bs_s{0,0,0,0}, bsa_s{0,0,0,0};
        auto row = [&](const std::string& op,
                       bm_bench::Stats& w, bm_bench::Stats& c, bm_bench::Stats& cr,
                       bm_bench::Stats& crr, bm_bench::Stats& e,
                       bm_bench::Stats& con) {
            auto sp = [](double a, double b) { return b > 0 ? a/b : 0.0; };
            csv << sf << "," << op << ","
                << w.median <<","<< w.stddev <<","<< w.min_val <<","<< w.max_val <<","
                << c.median <<","<< c.stddev <<","<< c.min_val <<","<< c.max_val <<","
                << cr.median<<","<< cr.stddev<<","<< cr.min_val<<","<< cr.max_val<<","
                << crr.median<<","<< crr.stddev<<","<< crr.min_val<<","<< crr.max_val<<","
                << e.median <<","<< e.stddev <<","<< e.min_val <<","<< e.max_val <<","
                << bs_s.median<<","<< bs_s.stddev<<","<< bs_s.min_val<<","<< bs_s.max_val<<","
                << bsa_s.median<<","<< bsa_s.stddev<<","<< bsa_s.min_val<<","<< bsa_s.max_val<<","
                << con.median<<","<< con.stddev<<","<< con.min_val<<","<< con.max_val<<","
                << sp(w.median, c.median) <<","<< sp(w.median, cr.median) <<","
                << sp(w.median, crr.median) <<","<< sp(w.median, e.median) <<","
                << "0,0,"
                << sp(w.median, con.median) << "\n";
        };
        row("OR",    wah_or_s,  cb_or_s,  cr_or_s,  crr_or_s,  ew_or_s,  con_or_s);
        row("AND",   wah_and_s, cb_and_s, cr_and_s, crr_and_s, ew_and_s, con_and_s);
        row("Agg",   wah_agg_s, cb_agg_s, cr_agg_s, crr_agg_s, ew_agg_s, con_agg_s);
        row("TOTAL", wah_tot_s, cb_tot_s, cr_tot_s, crr_tot_s, ew_tot_s, con_tot_s);
        std::cout << "  [CSV] q5_results_" << sf << ".csv" << std::endl;
    }

    });  // end call_once
}

} // namespace duckdb
