

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "execution/tpch/bitmap_table_scan.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q10_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q10_ITERATIONS = bm_bench::iter_count(5);
static const int Q10_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q10_once_flag_new;

static constexpr int64_t Q10_DATE_LO = 8674;
static constexpr int64_t Q10_DATE_HI = 8766;

static constexpr int64_t Q10_RETURNFLAG_R = 'R';

// rowid decode per backend
template <typename Btv>
static void q10_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q10_get_rowids<DDC>(const DDC& b, std::vector<row_t>* out) {
    // DDC literal walk
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++) out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q10_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q10_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q10_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q10_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}

void BMTableScan::BMTPCH_Q10(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q10_once_flag_new, [&]() {
    bm_bench::warn_if_sf1();

    auto& nation_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "nation");
    auto& customer_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "customer");
    auto& orders_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q10 (BitEngine pattern, runtime semi-joins)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_orderkey + bitmap_returnflag" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_okey = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_orderkey);
    auto* idx_rf   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_returnflag);
    if (!idx_okey || !idx_rf) {
        std::cerr << "[Q10] ERROR: bitmap_orderkey or bitmap_returnflag not loaded.\n";
        return;
    }

    std::vector<double> tA_t, tB_t, tC_t, tD_t, tE_t, tF_t, tG_t, tot_t;
    int64_t last_top1_revenue = 0;
    int64_t last_top1_custkey = -1;

    for (int iter = 0; iter < Q10_ITERATIONS; iter++) {
        bool warm = iter < Q10_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q10_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        // nation scan
        std::unordered_map<int32_t, std::string> nation_map;
        nation_map.reserve(64);
        {
            auto& tx = DuckTransaction::Get(context.client, nation_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(1)};
            nation_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types = {
                nation_table.GetColumns().GetColumnTypes()[0],
                nation_table.GetColumns().GetColumnTypes()[1],
            };
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                nation_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                auto nk = FlatVector::GetData<int32_t>(chunk.data[0]);
                auto nn = reinterpret_cast<string_t *>(chunk.data[1].GetData());
                for (idx_t i = 0; i < chunk.size(); i++)
                    nation_map[nk[i]] = nn[i].GetString();
            }
        }

        // customer->nation
        std::unordered_map<int64_t, int32_t> customer_nation;
        customer_nation.reserve(2000000);
        {
            auto& tx = DuckTransaction::Get(context.client, customer_table.catalog);
            TableScanState ss;

            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(3)};
            customer_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types = {
                customer_table.GetColumns().GetColumnTypes()[0],
                customer_table.GetColumns().GetColumnTypes()[3],
            };
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                customer_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                auto ck = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto nk = FlatVector::GetData<int32_t>(chunk.data[1]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (nation_map.count(nk[i]))
                        customer_nation[ck[i]] = nk[i];
                }
            }
        }

        std::unordered_map<int64_t, int64_t> okey_to_custkey;
        okey_to_custkey.reserve(2000000);
        std::vector<int64_t> matched_okeys;
        matched_okeys.reserve(2000000);
        {
            auto& tx = DuckTransaction::Get(context.client, orders_table.catalog);
            TableScanState ss;

            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(1), StorageIndex(4)};
            orders_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types;
            for (int c : {0, 1, 4})
                types.push_back(orders_table.GetColumns().GetColumnTypes()[c]);
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                orders_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto ck   = FlatVector::GetData<int64_t>(chunk.data[1]);
                auto od   = FlatVector::GetData<int32_t>(chunk.data[2]);
                // date + nation filter
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (od[i] < Q10_DATE_LO || od[i] >= Q10_DATE_HI) continue;
                    if (!customer_nation.count(ck[i])) continue;
                    okey_to_custkey.emplace(okey[i], ck[i]);
                    matched_okeys.push_back(okey[i]);
                }
            }
        }
        auto t_a = clk::now();

        size_t num_rows = idx_okey->num_rows();
        std::vector<row_t> ids;

        clk::time_point t_b, t_c, t_d;
        // DDC backend
        if (auto* cb_okey = dynamic_cast<bm_index::IndexedDDC*>(idx_okey)) {
            auto* cb_rf = dynamic_cast<bm_index::IndexedDDC*>(idx_rf);
            if (!cb_rf) { std::cerr << "[Q10] type mismatch.\n"; return; }
            DDC okey_filter = cb_okey->or_many(matched_okeys);  // orderkey OR
            t_b = clk::now();
            DDC rf_filter = DDC::from_sparse_positions({}, num_rows, cb_rf->segment_bits());
            cb_rf->apply_or_to(rf_filter, Q10_RETURNFLAG_R);  // returnflag OR
            t_c = clk::now();
            okey_filter &= rf_filter;  // AND
            q10_get_rowids(okey_filter, &ids);
            t_d = clk::now();
        // baselines: same OR/AND shape
        } else if (auto* cr_okey = dynamic_cast<bm_index::IndexedCRoaring*>(idx_okey)) {
            auto* cr_rf = dynamic_cast<bm_index::IndexedCRoaring*>(idx_rf);
            if (!cr_rf) { std::cerr << "[Q10] type mismatch.\n"; return; }

            roaring::Roaring okey_filter = cr_okey->or_many(matched_okeys);
            t_b = clk::now();
            roaring::Roaring rf_filter;
            cr_rf->apply_or_to(rf_filter, Q10_RETURNFLAG_R);
            t_c = clk::now();
            okey_filter &= rf_filter;
            q10_get_rowids(okey_filter, &ids);
            t_d = clk::now();
        } else if (auto* wah_okey = dynamic_cast<bm_index::IndexedWAH*>(idx_okey)) {
            auto* wah_rf = dynamic_cast<bm_index::IndexedWAH*>(idx_rf);
            if (!wah_rf) { std::cerr << "[Q10] type mismatch.\n"; return; }
            ibis::bitvector okey_filter = wah_okey->or_many(matched_okeys);
            t_b = clk::now();
            ibis::bitvector rf_filter;
            wah_rf->apply_or_to(rf_filter, Q10_RETURNFLAG_R);
            t_c = clk::now();
            okey_filter &= rf_filter;
            q10_get_rowids(okey_filter, &ids);
            t_d = clk::now();
        } else if (auto* ew_okey = dynamic_cast<bm_index::IndexedEWAH*>(idx_okey)) {
            auto* ew_rf = dynamic_cast<bm_index::IndexedEWAH*>(idx_rf);
            if (!ew_rf) { std::cerr << "[Q10] type mismatch.\n"; return; }
            ewah::EWAHBoolArray<uint64_t> okey_filter = ew_okey->or_many(matched_okeys);
            t_b = clk::now();
            ewah::EWAHBoolArray<uint64_t> rf_filter;
            ew_rf->apply_or_to(rf_filter, Q10_RETURNFLAG_R);
            t_c = clk::now();
            ewah::EWAHBoolArray<uint64_t> tmp;
            okey_filter.logicaland(rf_filter, tmp);
            q10_get_rowids(tmp, &ids);
            t_d = clk::now();
        } else if (auto* con_okey = dynamic_cast<bm_index::IndexedConcise*>(idx_okey)) {
            auto* con_rf = dynamic_cast<bm_index::IndexedConcise*>(idx_rf);
            if (!con_rf) { std::cerr << "[Q10] type mismatch.\n"; return; }
            ConciseSet<false> okey_filter = con_okey->or_many(matched_okeys);
            t_b = clk::now();
            ConciseSet<false> rf_filter;
            con_rf->apply_or_to(rf_filter, Q10_RETURNFLAG_R);
            t_c = clk::now();
            ConciseSet<false> result = okey_filter.logicaland(rf_filter);
            q10_get_rowids(result, &ids);
            t_d = clk::now();
        } else {
            std::cerr << "[Q10] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }

        // BMFetch + revenue agg
        std::unordered_map<int64_t, int64_t> revenue_by_custkey;
        revenue_by_custkey.reserve(600000);
        if (!ids.empty()) {
            vector<StorageIndex> col_ids = {
                StorageIndex(0), StorageIndex(5), StorageIndex(6)};
            vector<LogicalType> types;
            for (int c : {0, 5, 6})
                types.push_back(lineitem_table.GetColumns().GetColumnTypes()[c]);
            idx_t cursor = 0;
            idx_t num_idlist = ids.size();

            int64_t prev_okey = -1;
            int64_t* rev_slot = nullptr;

            while (cursor < ids.size()) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                ColumnFetchState column_fetch_state;
                data_ptr_t row_ids_data = (data_ptr_t)&ids[cursor];
                Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
                idx_t fetch_count = 2048;
                if (cursor + fetch_count > ids.size()) fetch_count = ids.size() - cursor;

                // fetch by rowid
                lineitem_table.GetStorage().Fetch(lineitem_tx, chunk, col_ids, row_ids_vec,
                                                  fetch_count, column_fetch_state);
                (void)num_idlist;
                cursor += fetch_count;
                auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto pr   = FlatVector::GetData<int64_t>(chunk.data[1]);
                auto dc   = FlatVector::GetData<int64_t>(chunk.data[2]);
                // accumulate, cache custkey slot
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (okey[i] != prev_okey) {
                        prev_okey = okey[i];
                        auto it = okey_to_custkey.find(okey[i]);
                        if (it == okey_to_custkey.end()) {
                            rev_slot = nullptr;
                        } else {

                            rev_slot = &revenue_by_custkey[it->second];
                        }
                    }
                    if (rev_slot)
                        *rev_slot += pr[i] * (100 - dc[i]);
                }
            }
        }
        auto t_e = clk::now();

        struct Entry {
            int64_t custkey;
            int64_t revenue;
        };

        auto cmp = [](const Entry& a, const Entry& b) {
            if (a.revenue != b.revenue) return a.revenue > b.revenue;
            return a.custkey < b.custkey;
        };
        // top-20 heap
        std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);
        for (auto& [ck, rev] : revenue_by_custkey) {
            if (heap.size() < 20) heap.push({ck, rev});
            else {
                const auto& top = heap.top();
                if (rev > top.revenue || (rev == top.revenue && ck < top.custkey)) {
                    heap.pop();
                    heap.push({ck, rev});
                }
            }
        }
        std::vector<Entry> top20;
        while (!heap.empty()) { top20.push_back(heap.top()); heap.pop(); }
        std::reverse(top20.begin(), top20.end());
        auto t_f = clk::now();

        struct Q10Attrs {
            std::string c_name;
            std::string c_address;
            std::string c_phone;
            std::string c_comment;
            int64_t     c_acctbal   = 0;
            int32_t     c_nationkey = 0;
        };
        std::unordered_set<int64_t> topk_custkeys;
        topk_custkeys.reserve(64);
        for (auto& e : top20) topk_custkeys.insert(e.custkey);

        // fetch top-20 cust attrs
        std::unordered_map<int64_t, Q10Attrs> cust_attr;
        cust_attr.reserve(64);
        {
            auto& tx = DuckTransaction::Get(context.client, customer_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {
                StorageIndex(0), StorageIndex(1), StorageIndex(2),
                StorageIndex(3), StorageIndex(4), StorageIndex(5),
                StorageIndex(7),
            };
            customer_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types;
            for (int c : {0, 1, 2, 3, 4, 5, 7})
                types.push_back(customer_table.GetColumns().GetColumnTypes()[c]);
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                customer_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                auto ck    = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto name  = reinterpret_cast<string_t *>(chunk.data[1].GetData());
                auto addr  = reinterpret_cast<string_t *>(chunk.data[2].GetData());
                auto nk    = FlatVector::GetData<int32_t>(chunk.data[3]);
                auto phone = reinterpret_cast<string_t *>(chunk.data[4].GetData());
                auto acct  = FlatVector::GetData<int64_t>(chunk.data[5]);
                auto comm  = reinterpret_cast<string_t *>(chunk.data[6].GetData());
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (topk_custkeys.count(ck[i])) {
                        auto& a = cust_attr[ck[i]];
                        a.c_name      = name[i].GetString();
                        a.c_address   = addr[i].GetString();
                        a.c_phone     = phone[i].GetString();
                        a.c_comment   = comm[i].GetString();
                        a.c_acctbal   = acct[i];
                        a.c_nationkey = nk[i];
                    }
                }
            }
        }
        auto t_g = clk::now();

        double tA = q10_ms(t0, t_a);
        double tB = q10_ms(t_a, t_b);
        double tC = q10_ms(t_b, t_c);
        double tD = q10_ms(t_c, t_d);
        double tE = q10_ms(t_d, t_e);
        double tF = q10_ms(t_e, t_f);
        double tG = q10_ms(t_f, t_g);
        double tot = q10_ms(t0, t_g);

        std::cout << "  " << idx_okey->backend_name()
                  << ":  PhaseA(orders)=" << tA
                  << "  PhaseB(okey_OR)=" << tB
                  << "  PhaseC(rf_OR)=" << tC
                  << "  PhaseD(AND+rids)=" << tD
                  << "  PhaseE(BMFetch+agg)=" << tE
                  << "  PhaseF(top20)=" << tF
                  << "  PhaseG(cust_attrs)=" << tG
                  << "  Total=" << tot
                  << "  rows=" << ids.size()
                  << "  custkeys=" << revenue_by_custkey.size();
        if (!top20.empty())
            std::cout << "  top1=" << top20[0].custkey << "/" << top20[0].revenue;
        std::cout << std::endl;

        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC);
            tD_t.push_back(tD); tE_t.push_back(tE); tF_t.push_back(tF);
            tG_t.push_back(tG);
            tot_t.push_back(tot);
        }
        if (!top20.empty()) {
            last_top1_custkey = top20[0].custkey;
            last_top1_revenue = top20[0].revenue;
        }
    }

    // SQL ground-truth check
    {
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT c_custkey, "
            "       sum(l_extendedprice * (1 - l_discount)) AS revenue "
            "FROM customer, orders, lineitem "
            "WHERE c_custkey = o_custkey "
            "  AND l_orderkey = o_orderkey "
            "  AND o_orderdate >= DATE '1993-10-01' "
            "  AND o_orderdate <  DATE '1994-01-01' "
            "  AND l_returnflag = 'R' "
            "GROUP BY c_custkey "
            "ORDER BY revenue DESC, c_custkey ASC LIMIT 1");
        if (r && !r->HasError() && r->RowCount() == 1) {
            int64_t gt_ck = r->GetValue(0, 0).GetValue<int64_t>();
            double  gt_rv = r->GetValue(1, 0).GetValue<double>();

            double ours = static_cast<double>(last_top1_revenue) / 10000.0;
            bool ok = gt_ck == last_top1_custkey && std::fabs(gt_rv - ours) < 0.01;
            if (ok) {
                std::cout << "\n[OK] " << idx_okey->backend_name()
                          << " matches DuckDB SQL ground truth "
                          << "(top-1 custkey=" << gt_ck
                          << " revenue=" << gt_rv << ").\n";
            } else {
                std::cerr << "\n[FAIL] mismatch: ours custkey=" << last_top1_custkey
                          << " revenue=" << ours
                          << " gt custkey=" << gt_ck << " revenue=" << gt_rv << "\n";
            }
        }
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t),
         sD = stats(tD_t), sE = stats(tE_t), sF = stats(tF_t),
         sG = stats(tG_t), sT = stats(tot_t);
    int measured = std::max(0, Q10_ITERATIONS - Q10_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q10 RESULTS — " << idx_okey->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (orders scan)       : " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (orderkey OR)       : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (returnflag OR)     : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  PhaseD (AND + get_rowids)  : " << sD.median << " +/- " << sD.stddev << " ms\n";
    std::cout << "  PhaseE (BMFetch + revenue) : " << sE.median << " +/- " << sE.stddev << " ms\n";
    std::cout << "  PhaseF (top-20 heap)       : " << sF.median << " +/- " << sF.stddev << " ms\n";
    std::cout << "  PhaseG (top-20 cust attrs) : " << sG.median << " +/- " << sG.stddev << " ms\n";
    std::cout << "  TOTAL                      : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q10_results_" + sf + ".csv");
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
        std::string bn = idx_okey->backend_name();
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
            cell(z); cell(z);
            cell(bn == "Concise" ? s : z);
            csv << "0,0,0,0,0,0,0\n";
        };
        put("PhaseA_orders", sA);
        put("PhaseB_okey",   sB);
        put("PhaseC_rf",     sC);
        put("PhaseD_AND",    sD);
        put("PhaseE_agg",    sE);
        put("PhaseF_top20",  sF);
        put("PhaseG_attrs",  sG);
        put("TOTAL", sT);
        std::cout << "  [CSV] q10_results_" << sf << ".csv\n";
    }

    });
}

}
