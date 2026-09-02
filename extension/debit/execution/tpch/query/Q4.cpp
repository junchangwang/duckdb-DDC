

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
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q4_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q4_ITERATIONS = bm_bench::iter_count(5);
static const int Q4_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q4_once_flag_new;

static constexpr int64_t Q4_DATE_LO = 8582;
static constexpr int64_t Q4_DATE_HI = 8674;

// rowids per backend
template <typename Btv>
static void q4_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q4_get_rowids<DDC>(const DDC& b, std::vector<row_t>* out) {
    out->clear();
    // expand literals
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++) out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q4_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q4_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q4_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q4_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q4_get_rowids<::bitset::BitsetVector>(const ::bitset::BitsetVector& b, std::vector<row_t>* out) {
    out->clear();
    const uint64_t* w = b.words();
    const uint64_t nb = b.num_bits();
    for (size_t i = 0; i < b.words_cnt(); ++i) {
        uint64_t x = w[i];
        const uint64_t base = static_cast<uint64_t>(i) * 64;
        while (x) {
            const uint64_t pos = base + __builtin_ctzll(x);
            if (pos >= nb) return;
            out->push_back(static_cast<row_t>(pos));
            x &= x - 1;
        }
    }
}


// Q4 entry
void BMTableScan::BMTPCH_Q4(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q4_once_flag_new, [&]() {
    bm_bench::warn_if_sf1();

    auto& orders_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q4 (BitEngine pattern, runtime EXISTS via BMFetch)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_orderkey" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_okey = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_orderkey);
    if (!idx_okey) { std::cerr << "[Q4] ERROR: bitmap_orderkey not loaded.\n"; return; }

    std::vector<double> tA_t, tB_t, tC_t, tD_t, tot_t;
    std::map<std::string, int64_t> last_priority_count;

    for (int iter = 0; iter < Q4_ITERATIONS; iter++) {
        bool warm = iter < Q4_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q4_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        // Orders filter
        std::unordered_map<int64_t, std::string> orderkey_priority;
        orderkey_priority.reserve(600000);
        {
            auto& tx = DuckTransaction::Get(context.client, orders_table.catalog);
            TableScanState ss;

            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(4), StorageIndex(5)};
            orders_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types;
            for (int c : {0, 4, 5}) types.push_back(orders_table.GetColumns().GetColumnTypes()[c]);
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                orders_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[2].Flatten(chunk.size());
                auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto od   = FlatVector::GetData<int32_t>(chunk.data[1]);
                auto op_  = FlatVector::GetData<string_t>(chunk.data[2]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (od[i] >= Q4_DATE_LO && od[i] < Q4_DATE_HI)
                        orderkey_priority[okey[i]] = op_[i].GetString();
                }
            }
        }
        auto t_a = clk::now();

        // Bitmap filter
        std::vector<row_t> ids;
        size_t num_rows = idx_okey->num_rows();
        std::vector<int64_t> keys;
        keys.reserve(orderkey_priority.size());
        for (auto& [k, _] : orderkey_priority) keys.push_back(k);
        if (auto* cb = dynamic_cast<bm_index::IndexedDDC*>(idx_okey)) {
            DDC btv = cb->or_many(keys);
            q4_get_rowids(btv, &ids);
        } else if (auto* cr = dynamic_cast<bm_index::IndexedCRoaring*>(idx_okey)) {

            roaring::Roaring btv = cr->or_many(keys);
            q4_get_rowids(btv, &ids);
        } else if (auto* bs = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_okey)) {
            ::bitset::BitsetVector btv = bs->or_many(keys);
            q4_get_rowids(btv, &ids);
        } else if (auto* wah = dynamic_cast<bm_index::IndexedWAH*>(idx_okey)) {
            ibis::bitvector btv = wah->or_many(keys);
            q4_get_rowids(btv, &ids);
        } else if (auto* ew = dynamic_cast<bm_index::IndexedEWAH*>(idx_okey)) {
            ewah::EWAHBoolArray<uint64_t> btv = ew->or_many(keys);
            q4_get_rowids(btv, &ids);
        } else if (auto* con = dynamic_cast<bm_index::IndexedConcise*>(idx_okey)) {
            ConciseSet<false> btv = con->or_many(keys);
            q4_get_rowids(btv, &ids);
        } else {
            std::cerr << "[Q4] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }
        auto t_b = clk::now();

        // Lineitem probe
        std::unordered_set<int64_t> orderkey_late;
        orderkey_late.reserve(600000);
        if (!ids.empty()) {

            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(11), StorageIndex(12)};
            vector<LogicalType> types;
            for (int c : {0, 11, 12}) types.push_back(lineitem_table.GetColumns().GetColumnTypes()[c]);
            idx_t cursor = 0;
            idx_t num_idlist = ids.size();
            while (cursor < ids.size()) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                ColumnFetchState column_fetch_state;
                data_ptr_t row_ids_data = (data_ptr_t)&ids[cursor];
                Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
                idx_t fetch_count = 2048;
                if (cursor + fetch_count > ids.size()) fetch_count = ids.size() - cursor;
                // fetch by rowid
                lineitem_table.GetStorage().BMFetch(lineitem_tx, chunk, col_ids, row_ids_vec,
                                                    fetch_count, column_fetch_state, num_idlist);
                cursor += fetch_count;
                auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto cd   = FlatVector::GetData<int32_t>(chunk.data[1]);
                auto rd   = FlatVector::GetData<int32_t>(chunk.data[2]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (cd[i] < rd[i]) orderkey_late.insert(okey[i]);
                }
            }
        }
        auto t_c = clk::now();

        // Priority count
        std::map<std::string, int64_t> priority_count;
        for (int64_t okey : orderkey_late) {
            auto it = orderkey_priority.find(okey);
            if (it != orderkey_priority.end()) priority_count[it->second]++;
        }
        auto t_d = clk::now();

        double tA  = q4_ms(t0, t_a);
        double tB  = q4_ms(t_a, t_b);
        double tC  = q4_ms(t_b, t_c);
        double tD  = q4_ms(t_c, t_d);
        double tot = q4_ms(t0, t_d);
        std::cout << "  " << idx_okey->backend_name()
                  << ":  PhaseA(orders_scan)=" << tA
                  << "  PhaseB(orderkey_OR+ids)=" << tB
                  << "  PhaseC(BMFetch+EXISTS)=" << tC
                  << "  PhaseD(count_by_priority)=" << tD
                  << "  Total=" << tot
                  << "  rows=" << ids.size()
                  << "  groups=" << priority_count.size()
                  << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC);
            tD_t.push_back(tD); tot_t.push_back(tot);
        }
        last_priority_count = priority_count;
    }

    // SQL validation
    {
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT o_orderpriority, count(*) AS order_count "
            "FROM orders "
            "WHERE o_orderdate >= DATE '1993-07-01' "
            "  AND o_orderdate <  DATE '1993-07-01' + INTERVAL '3' MONTH "
            "  AND EXISTS (SELECT * FROM lineitem "
            "              WHERE l_orderkey = o_orderkey "
            "                AND l_commitdate < l_receiptdate) "
            "GROUP BY o_orderpriority ORDER BY o_orderpriority");
        if (r && !r->HasError()) {
            bool ok = (r->RowCount() == last_priority_count.size());
            for (idx_t i = 0; i < r->RowCount(); i++) {
                std::string pri = r->GetValue(0, i).GetValue<std::string>();
                int64_t cnt = r->GetValue(1, i).GetValue<int64_t>();
                auto it = last_priority_count.find(pri);
                if (it == last_priority_count.end() || it->second != cnt) {
                    ok = false;
                    std::cerr << "[FAIL] priority=" << pri << " our="
                              << (it == last_priority_count.end() ? -1 : it->second)
                              << " gt=" << cnt << "\n";
                }
            }
            if (ok)
                std::cout << "\n[OK] " << idx_okey->backend_name()
                          << " matches DuckDB SQL ground truth ("
                          << r->RowCount() << " priorities).\n";
        }
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t),
         sD = stats(tD_t), sT = stats(tot_t);
    int measured = std::max(0, Q4_ITERATIONS - Q4_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q4 RESULTS — " << idx_okey->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (orders scan)        : " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (orderkey OR + ids)  : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (BMFetch + EXISTS)   : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  PhaseD (count by priority)  : " << sD.median << " +/- " << sD.stddev << " ms\n";
    std::cout << "  TOTAL                       : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    // emit CSV
    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q4_results_" + sf + ".csv");
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
            cell(bn == "Bitset" ? s : z);
            cell(bn == "Bitset+AVX512" ? s : z);
            cell(bn == "Concise" ? s : z);
            csv << "0,0,0,0,0,0,0\n";
        };
        put("PhaseA_orders", sA);
        put("PhaseB_okey",   sB);
        put("PhaseC_exists", sC);
        put("PhaseD_count",  sD);
        put("TOTAL",         sT);
        std::cout << "  [CSV] q4_results_" << sf << ".csv\n";
    }

    });
}

}
