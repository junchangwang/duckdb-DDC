

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

#include "ddc_adapter.h"
#include "ddc/include/ddc.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"
#include "Concise/concise.h"

#include "execution/tpch/bm_baseline_loaders.hpp"
#include "execution/tpch/bm_bench_common.hpp"
#include "execution/tpch/indexed_bitmap.hpp"
#include "execution/tpch/bsr_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

using Q5BmType = bm_bench::Backend;
static const Q5BmType Q5_BM = bm_bench::parse_backend("Q5_BM");
static const char* q5_bm_label() { return bm_bench::backend_label(Q5_BM); }
static std::string q5_get_sf_label() { return bm_bench::sf_label(); }

static const int Q5_ITERATIONS = bm_bench::iter_count(5);
static const int Q5_WARMUP     = bm_bench::warmup_count(1);

static std::once_flag q5_once_flag;

static void get_rowids(const DDC& btv, std::vector<row_t>& out) {
    out.clear();
    btv.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out.push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}

static void get_rowids(const roaring::Roaring& btv, std::vector<row_t>& out) {
    out.clear();
    out.reserve(btv.cardinality());
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out.push_back(static_cast<row_t>(*it));
}

static void get_rowids(const ibis::bitvector& btv, std::vector<row_t>& out) {
    out.clear();
    ibis::bitvector::pit pit(btv);
    while (*pit != 0xFFFFFFFFU) {
        out.push_back(static_cast<row_t>(*pit));
        pit.next();
    }
}

static void get_rowids(const ::bitset::BitsetVector& btv, std::vector<row_t>& out) {
    out.clear();
    const uint64_t* w = btv.words();
    const uint64_t nb = btv.num_bits();
    for (size_t i = 0; i < btv.words_cnt(); ++i) {
        uint64_t x = w[i];
        const uint64_t base = static_cast<uint64_t>(i) * 64;
        while (x) {
            const uint64_t pos = base + __builtin_ctzll(x);
            if (pos >= nb) return;
            out.push_back(static_cast<row_t>(pos));
            x &= x - 1;
        }
    }
}

static void get_rowids(const ewah::EWAHBoolArray<uint64_t>& btv, std::vector<row_t>& out) {
    out.clear();
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out.push_back(static_cast<row_t>(*it));
}

static void get_rowids(const ConciseSet<false>& btv, std::vector<row_t>& out) {
    out.clear();
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out.push_back(static_cast<row_t>(*it));
}

static void run_q5_aggregate(
    ClientContext& ctx,
    TableCatalogEntry& lineitem_table,
    DuckTransaction& tx,
    const std::vector<row_t>& ids,
    const std::unordered_map<int64_t, int32_t>& order_nation_map,
    const std::unordered_map<int64_t, int32_t>& supp_nation_map,
    const std::unordered_map<int32_t, std::string>& n_name_map,
    std::map<std::string, int64_t>& ans,
    size_t& matched_rows)
{
    matched_rows = 0;
    if (ids.empty()) return;

    vector<StorageIndex> col_ids = {
        StorageIndex(0),
        StorageIndex(2),
        StorageIndex(5),
        StorageIndex(6),
    };
    vector<LogicalType> types = {
        lineitem_table.GetColumns().GetColumnTypes()[0],
        lineitem_table.GetColumns().GetColumnTypes()[2],
        lineitem_table.GetColumns().GetColumnTypes()[5],
        lineitem_table.GetColumns().GetColumnTypes()[6],
    };

    idx_t cursor = 0;
    idx_t num_idlist = ids.size();
    while (cursor < ids.size()) {
        DataChunk result; result.Initialize(ctx, types);
        ColumnFetchState column_fetch_state;

        data_ptr_t row_ids_data = (data_ptr_t)&ids[cursor];
        Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
        idx_t fetch_count = 2048;
        if (cursor + fetch_count > ids.size())
            fetch_count = ids.size() - cursor;

        lineitem_table.GetStorage().BMFetch(
            tx, result, col_ids, row_ids_vec, fetch_count,
            column_fetch_state, num_idlist);
        cursor += fetch_count;

        auto okey = FlatVector::GetData<int64_t>(result.data[0]);
        auto skey = FlatVector::GetData<int64_t>(result.data[1]);
        auto pr   = FlatVector::GetData<int64_t>(result.data[2]);
        auto dc   = FlatVector::GetData<int64_t>(result.data[3]);
        for (idx_t i = 0; i < result.size(); i++) {
            auto on_it = order_nation_map.find(okey[i]);
            auto sn_it = supp_nation_map.find(skey[i]);
            if (on_it == order_nation_map.end() ||
                sn_it == supp_nation_map.end() ||
                on_it->second != sn_it->second) continue;
            auto nm = n_name_map.find(on_it->second);
            if (nm == n_name_map.end()) continue;
            ans[nm->second] += pr[i] * (100 - dc[i]);
            matched_rows++;
        }
    }
}

void BMTableScan::BMTPCH_Q5(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q5_once_flag, [&]() {

    bm_bench::warn_if_sf1();

    auto& region_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "region");
    auto& nation_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "nation");
    auto& customer_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "customer");
    auto& orders_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
    auto& supplier_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "supplier");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q5 (BitEngine pattern, BMFetch) — " << q5_bm_label()
              << " (" << q5_get_sf_label() << ")" << std::endl;
    std::cout << "  Pre-loaded auxiliary structures: bitmap_orderkey + bitmap_suppkey" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_okey_base = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_orderkey);
    auto* idx_skey_base = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_suppkey);
    if (!idx_okey_base || !idx_skey_base) {
        std::cerr << "[Q5] ERROR: bitmap_orderkey or bitmap_suppkey not loaded.\n"
                     "          Run PRAGMA load_bitmap('orderkey'); load_bitmap('suppkey'); first.\n";
        return;
    }

    std::vector<double> tA_t, tB_t, tC_t, tot_t;
    std::map<std::string, int64_t> last_ans;
    size_t last_rows = 0;
    std::unordered_map<int32_t, std::string> n_name_map_capture;

    for (int iter = 0; iter < Q5_ITERATIONS; iter++) {
        bool warm = iter < Q5_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q5_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t_total_start = clk::now();

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

        auto t_phaseA_end = clk::now();

        if (order_nation_map.empty() || supp_nation_map.empty()) {
            std::cout << "  (no qualifying rows)" << std::endl;
            continue;
        }

        std::map<std::string, int64_t> ans;
        size_t matched = 0;
        size_t num_rows_lineitem = idx_okey_base->num_rows();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        std::vector<row_t> ids;

        std::vector<int64_t> okeys, skeys;
        okeys.reserve(order_nation_map.size());
        skeys.reserve(supp_nation_map.size());
        for (auto& [k, _] : order_nation_map) okeys.push_back(k);
        for (auto& [k, _] : supp_nation_map ) skeys.push_back(k);

        if (auto* bs_okey = dynamic_cast<bm_index::IndexedBSR*>(idx_okey_base)) {
            auto* bs_skey = dynamic_cast<bm_index::IndexedBSR*>(idx_skey_base);
            if (!bs_skey) { std::cerr << "[Q5] ERROR: orderkey is BSR, suppkey isn't.\n"; return; }
            bm_index::BsrSet btv_or   = bs_okey->or_many(okeys);
            bm_index::BsrSet btv_supp = bs_skey->or_many(skeys);
            bm_index::BsrSet r = bm_index::bsr_and(btv_or, btv_supp);
            r.get_rowids(&ids);
        }
        else if (auto* cb_okey = dynamic_cast<bm_index::IndexedDDC*>(idx_okey_base)) {
            auto* cb_skey = dynamic_cast<bm_index::IndexedDDC*>(idx_skey_base);
            if (!cb_skey) { std::cerr << "[Q5] ERROR: orderkey is DDC, suppkey isn't.\n"; return; }
            DDC btv_or   = cb_okey->or_many(okeys);
            DDC btv_supp = cb_skey->or_many(skeys);
            btv_or &= btv_supp;
            get_rowids(btv_or, ids);
        }

        else if (auto* cr_okey = dynamic_cast<bm_index::IndexedCRoaring*>(idx_okey_base)) {
            auto* cr_skey = dynamic_cast<bm_index::IndexedCRoaring*>(idx_skey_base);
            if (!cr_skey) { std::cerr << "[Q5] suppkey type mismatch.\n"; return; }
            roaring::Roaring btv_or, btv_supp;

            btv_or   = cr_okey->or_many(okeys);
            btv_supp = cr_skey->or_many(skeys);
            btv_or &= btv_supp;
            get_rowids(btv_or, ids);
        }

        else if (auto* bs_okey = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_okey_base)) {
            auto* bs_skey = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_skey_base);
            if (!bs_skey) { std::cerr << "[Q5] suppkey type mismatch.\n"; return; }
            ::bitset::BitsetVector btv_or   = bs_okey->or_many(okeys);
            ::bitset::BitsetVector btv_supp = bs_skey->or_many(skeys);
            ::bitset::BitsetVector::word_and_inplace(btv_or, btv_supp, bs_okey->use_simd());
            get_rowids(btv_or, ids);
        }
        else if (auto* wah_okey = dynamic_cast<bm_index::IndexedWAH*>(idx_okey_base)) {
            auto* wah_skey = dynamic_cast<bm_index::IndexedWAH*>(idx_skey_base);
            if (!wah_skey) { std::cerr << "[Q5] suppkey type mismatch.\n"; return; }
            ibis::bitvector btv_res     = wah_okey->or_many(okeys);
            ibis::bitvector btv_suppkey = wah_skey->or_many(skeys);
            btv_res &= btv_suppkey;
            get_rowids(btv_res, ids);
        }

        else if (auto* ew_okey = dynamic_cast<bm_index::IndexedEWAH*>(idx_okey_base)) {
            auto* ew_skey = dynamic_cast<bm_index::IndexedEWAH*>(idx_skey_base);
            if (!ew_skey) { std::cerr << "[Q5] suppkey type mismatch.\n"; return; }
            auto btv_or   = ew_okey->or_many(okeys);
            auto btv_supp = ew_skey->or_many(skeys);
            ewah::EWAHBoolArray<uint64_t> filt;
            btv_or.logicaland(btv_supp, filt);
            get_rowids(filt, ids);
        }

        else if (auto* con_okey = dynamic_cast<bm_index::IndexedConcise*>(idx_okey_base)) {
            auto* con_skey = dynamic_cast<bm_index::IndexedConcise*>(idx_skey_base);
            if (!con_skey) { std::cerr << "[Q5] suppkey type mismatch.\n"; return; }
            auto btv_or   = con_okey->or_many(okeys);
            auto btv_supp = con_skey->or_many(skeys);
            ConciseSet<false> filt = btv_or.logicaland(btv_supp);
            get_rowids(filt, ids);
        }
        else {
            std::cerr << "[Q5] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }

        auto t_phaseB_end = clk::now();

        run_q5_aggregate(context.client, lineitem_table, lineitem_tx, ids,
                         order_nation_map, supp_nation_map, n_name_map,
                         ans, matched);

        auto t_phaseC_end = clk::now();

        last_ans = ans; last_rows = matched; n_name_map_capture = n_name_map;

        double tA  = ms(t_total_start, t_phaseA_end);
        double tB  = ms(t_phaseA_end, t_phaseB_end);
        double tC  = ms(t_phaseB_end, t_phaseC_end);
        double tot = ms(t_total_start, t_phaseC_end);
        std::cout << "  " << idx_okey_base->backend_name()
                  << ":  PhaseA=" << tA << "  PhaseB=" << tB
                  << "  PhaseC=" << tC << "  Total=" << tot
                  << "  rows=" << matched << "  ids=" << ids.size() << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB);
            tC_t.push_back(tC); tot_t.push_back(tot);
        }
    }

    // SQL validation
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
            for (idx_t i = 0; i < r->RowCount(); i++)
                gt_revenue[r->GetValue(0, i).GetValue<std::string>()] =
                    r->GetValue(1, i).GetValue<double>();
            gt_ok = true;
        }
    }
    if (gt_ok) {
        bool ok = (last_ans.size() == gt_revenue.size());
        if (ok) {
            for (auto& [name, rev_fp] : last_ans) {
                double our = double(rev_fp) / 10000;
                auto it = gt_revenue.find(name);
                if (it == gt_revenue.end() || std::fabs(our - it->second) > 0.01) {
                    ok = false; break;
                }
            }
        }
        if (ok)
            std::cout << "\n[OK] " << idx_okey_base->backend_name()
                      << " matches DuckDB SQL ground truth ("
                      << gt_revenue.size() << " nations).\n";
        else
            std::cerr << "\n[FAIL] mismatch vs SQL ground truth!\n";
    }

    std::vector<std::pair<int64_t, std::string>> rows;
    for (auto& [name, rev] : last_ans) rows.push_back({rev, name});
    std::sort(rows.rbegin(), rows.rend());
    std::cout << "\n  Q5 Top results (revenue DESC):\n  nation                     revenue\n";
    for (auto& [rev, name] : rows)
        std::cout << "  " << std::left << std::setw(20) << name
                  << std::fixed << std::setprecision(4)
                  << (double(rev) / 10000) << std::endl;

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t), sT = stats(tot_t);
    int measured = std::max(0, Q5_ITERATIONS - Q5_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q5 RESULTS — " << idx_okey_base->backend_name()
              << " only (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (small joins) : " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (bitmap OR/AND): " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (BMFetch+Agg) : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  TOTAL                : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "  rows: " << last_rows << "\n";
    std::cout << "================================================================\n\n";

    std::string sf = q5_get_sf_label();
    std::ofstream csv("q5_results_" + sf + ".csv");
    if (csv) {
        csv << std::fixed << std::setprecision(4);
        csv << "sf,operation,"
            << "wah_median_ms,wah_stddev_ms,wah_min_ms,wah_max_ms,"
            << "ddc_median_ms,ddc_stddev_ms,ddc_min_ms,ddc_max_ms,"
            << "croaring_median_ms,croaring_stddev_ms,croaring_min_ms,croaring_max_ms,"
            << "croaring_run_median_ms,croaring_run_stddev_ms,croaring_run_min_ms,croaring_run_max_ms,"
            << "ewah_median_ms,ewah_stddev_ms,ewah_min_ms,ewah_max_ms,"
            << "bs_median_ms,bs_stddev_ms,bs_min_ms,bs_max_ms,"
            << "bsa_median_ms,bsa_stddev_ms,bsa_min_ms,bsa_max_ms,"
            << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms,"
            << "cb_vs_wah,cr_vs_wah,crr_vs_wah,ew_vs_wah,bs_vs_wah,bsa_vs_wah,con_vs_wah\n";
        bm_bench::Stats z{0,0,0,0};
        std::string bn = idx_okey_base->backend_name();
        auto put = [&](const std::string& op, bm_bench::Stats s) {

            csv << sf << "," << op << ",";
            auto cell = [&](bm_bench::Stats v) {
                csv << v.median << "," << v.stddev << "," << v.min_val << "," << v.max_val << ",";
            };

            cell(bn == "WAH" ? s : z);
            cell(bn == "DDC" ? s : z);
            cell(bn == "CRoaring" ? s : z);
            cell(bn == "CRoaringRun" ? s : z);
            cell(bn == "EWAH" ? s : z);
            cell(bn == "Bitset" ? s : z);
            cell(bn == "Bitset+AVX512" ? s : z);
            cell(bn == "Concise" ? s : z);
            csv << "0,0,0,0,0,0,0\n";
        };
        put("PhaseA", sA);
        put("PhaseB", sB);
        put("PhaseC", sC);
        put("TOTAL",  sT);
        std::cout << "  [CSV] q5_results_" << sf << ".csv\n";
    }

    });
}

}
