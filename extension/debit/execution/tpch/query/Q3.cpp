

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "execution/tpch/bitmap_table_scan.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/date.hpp"

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
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q3_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q3_ITERATIONS = bm_bench::iter_count(5);
static const int Q3_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q3_once_flag_new;

static constexpr int64_t Q3_CUTOFF_DATE = 9204;

static const std::vector<int64_t>& Q3_GE_AFTER_MONTHS() {
    static const std::vector<int64_t> v = []() {
        std::vector<int64_t> k;

        for (int y = 1995; y <= 1998; y++) {
            int m_lo = (y == 1995) ? 4 : 1;
            int m_hi = 12;
            for (int m = m_lo; m <= m_hi; m++) k.push_back(static_cast<int64_t>(y) * 100 + m);
        }
        return k;
    }();
    return v;
}

struct Q3Acc {
    int64_t revenue_raw = 0;
    int32_t orderdate   = 0;
    int32_t shippriority= 0;
};

// per-backend rowid extract
template <typename ResultBitmap>
static void q3_get_rowids_t(const ResultBitmap& btv, std::vector<row_t>* out);
template <>
void q3_get_rowids_t<DDC>(const DDC& btv, std::vector<row_t>* out) {
    out->clear();
    btv.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q3_get_rowids_t<roaring::Roaring>(const roaring::Roaring& btv, std::vector<row_t>* out) {
    out->clear(); out->reserve(btv.cardinality());
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out->push_back(static_cast<row_t>(*it));
}
template <>
void q3_get_rowids_t<ibis::bitvector>(const ibis::bitvector& btv, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(btv);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q3_get_rowids_t<ewah::EWAHBoolArray<uint64_t>>(const ewah::EWAHBoolArray<uint64_t>& btv,
                                                    std::vector<row_t>* out) {
    out->clear();
    for (auto it = btv.begin(); it != btv.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q3_get_rowids_t<ConciseSet<false>>(const ConciseSet<false>& btv, std::vector<row_t>* out) {
    out->clear();
    for (auto it = btv.begin(); it != btv.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q3_get_rowids_t<::bitset::BitsetVector>(const ::bitset::BitsetVector& btv, std::vector<row_t>* out) {
    out->clear();
    const uint64_t* w = btv.words();
    const uint64_t nb = btv.num_bits();
    for (size_t i = 0; i < btv.words_cnt(); ++i) {
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


static void q3_aggregate(
    ClientContext& ctx,
    TableCatalogEntry& lineitem_table,
    DuckTransaction& tx,
    const std::vector<row_t>& ids,
    std::unordered_map<int64_t, Q3Acc>& l_orderkey_map)
{
    // Fetch aggregate
    if (ids.empty()) return;
    vector<StorageIndex> col_ids = {
        StorageIndex(0),
        StorageIndex(5),
        StorageIndex(6),
    };
    vector<LogicalType> types = {
        lineitem_table.GetColumns().GetColumnTypes()[0],
        lineitem_table.GetColumns().GetColumnTypes()[5],
        lineitem_table.GetColumns().GetColumnTypes()[6],
    };
    idx_t cursor = 0;
    idx_t num_idlist = ids.size();
    while (cursor < ids.size()) {
        // 2048 fetch batch
        DataChunk chunk; chunk.Initialize(ctx, types);
        ColumnFetchState column_fetch_state;
        data_ptr_t row_ids_data = (data_ptr_t)&ids[cursor];
        Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
        idx_t fetch_count = 2048;
        if (cursor + fetch_count > ids.size()) fetch_count = ids.size() - cursor;
        lineitem_table.GetStorage().BMFetch(tx, chunk, col_ids, row_ids_vec, fetch_count,
                                            column_fetch_state, num_idlist);
        cursor += fetch_count;
        auto okey  = FlatVector::GetData<int64_t>(chunk.data[0]);
        auto price = FlatVector::GetData<int64_t>(chunk.data[1]);
        auto disc  = FlatVector::GetData<int64_t>(chunk.data[2]);
        for (idx_t i = 0; i < chunk.size(); i++) {
            auto it = l_orderkey_map.find(okey[i]);
            if (it != l_orderkey_map.end())
                it->second.revenue_raw += price[i] * (100 - disc[i]);
        }
    }
}

void BMTableScan::BMTPCH_Q3(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q3_once_flag_new, [&]() {

    bm_bench::warn_if_sf1();

    auto& customer_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "customer");
    auto& orders_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q3 (BitEngine pattern, runtime semi-joins)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_orderkey + bitmap_shipdate" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_okey = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_orderkey);

    auto* idx_ge_m = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_GE_month);
    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    auto* idx_ship_use = idx_ge_m ? idx_ge_m : idx_ship;
    if (!idx_okey || !idx_ship_use) {
        std::cerr << "[Q3] ERROR: bitmap_orderkey or bitmap_shipdate(_GE_month) not loaded.\n"
                     "      PRAGMA load_bitmap('orderkey'); load_bitmap('shipdate_GE_month'); first.\n";
        return;
    }
    const bool use_month_ge_q3 = (idx_ge_m != nullptr);
    if (!use_month_ge_q3 &&
        std::string(idx_okey->backend_name()) != idx_ship->backend_name()) {
        std::cerr << "[Q3] ERROR: backend mismatch (orderkey vs shipdate).\n";
        return;
    }

    std::vector<double> tA_t, tB_t, tC_t, tD_t, tE_t, tot_t;
    std::map<int64_t, Q3Acc> last_top10_map;

    for (int iter = 0; iter < Q3_ITERATIONS; iter++) {
        bool warm = iter < Q3_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q3_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();

        // PhaseA: customer scan
        std::unordered_set<int64_t> c_custkey_set;
        c_custkey_set.reserve(300000);
        {
            auto& tx = DuckTransaction::Get(context.client, customer_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(6)};
            customer_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types = {
                customer_table.GetColumns().GetColumnTypes()[0],
                customer_table.GetColumns().GetColumnTypes()[6],
            };
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                customer_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[1].Flatten(chunk.size());
                auto ck = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto mk = FlatVector::GetData<string_t>(chunk.data[1]);
                auto& valid = FlatVector::Validity(chunk.data[1]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (!valid.RowIsValid(i)) continue;
                    if (mk[i].GetSize() == 8 &&
                        std::memcmp(mk[i].GetData(), "BUILDING", 8) == 0)
                        c_custkey_set.insert(ck[i]);
                }
            }
        }
        auto t_a = clk::now();

        std::unordered_map<int64_t, Q3Acc> l_orderkey_map;
        l_orderkey_map.reserve(2000000);

        auto* cb_okey = dynamic_cast<bm_index::IndexedDDC*>(idx_okey);
        auto* cr_okey = dynamic_cast<bm_index::IndexedCRoaring*>(idx_okey);
        auto* wah_okey= dynamic_cast<bm_index::IndexedWAH*>(idx_okey);
        auto* ew_okey = dynamic_cast<bm_index::IndexedEWAH*>(idx_okey);
        auto* con_okey= dynamic_cast<bm_index::IndexedConcise*>(idx_okey);
        auto* bs_okey = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_okey);

        size_t num_rows = idx_okey->num_rows();
        DDC cb_btv_res;
        roaring::Roaring cr_btv_res;
        ibis::bitvector wah_btv_res;
        ewah::EWAHBoolArray<uint64_t> ew_btv_res;
        ConciseSet<false> con_btv_res;
        ::bitset::BitsetVector bs_btv_res;
        if (cb_okey) cb_btv_res = DDC::from_sparse_positions({}, num_rows, cb_okey->segment_bits());

        // PhaseB: orders semi-join
        std::vector<int64_t> matched_okeys;
        matched_okeys.reserve(2000000);
        {
            auto& tx = DuckTransaction::Get(context.client, orders_table.catalog);
            TableScanState ss;

            vector<StorageIndex> col_ids = {
                StorageIndex(0), StorageIndex(1), StorageIndex(4), StorageIndex(7)};
            orders_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types;
            for (int c : {0, 1, 4, 7})
                types.push_back(orders_table.GetColumns().GetColumnTypes()[c]);
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                orders_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto ck   = FlatVector::GetData<int64_t>(chunk.data[1]);
                auto od   = FlatVector::GetData<int32_t>(chunk.data[2]);
                auto sp   = FlatVector::GetData<int32_t>(chunk.data[3]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (od[i] >= Q3_CUTOFF_DATE) continue;
                    if (!c_custkey_set.count(ck[i])) continue;
                    Q3Acc acc; acc.orderdate = od[i]; acc.shippriority = sp[i];
                    l_orderkey_map.emplace(okey[i], acc);
                    matched_okeys.push_back(okey[i]);
                }
            }
        }

        // OR matched orderkeys
        if (cb_okey) {
            cb_btv_res = cb_okey->or_many(matched_okeys);
        } else if (cr_okey) {

            cr_btv_res = cr_okey->or_many(matched_okeys);
        } else if (bs_okey) {
            bs_btv_res = bs_okey->or_many(matched_okeys);
        } else if (wah_okey) {
            wah_btv_res = wah_okey->or_many(matched_okeys);
        } else if (ew_okey) {
            ew_btv_res = ew_okey->or_many(matched_okeys);
        } else if (con_okey) {
            con_btv_res = con_okey->or_many(matched_okeys);
        }
        auto t_b = clk::now();

        auto t_c = t_b;
        auto t_d = t_b;
        std::vector<row_t> ids;
        ids.reserve(2000000);

        if (cb_okey) {

            auto* cbge_ship = dynamic_cast<bm_index::IndexedDDCGE*>(idx_ship_use);
            auto* cb_ship   = dynamic_cast<bm_index::IndexedDDC*>(idx_ship_use);
            DDC ship_filter;
            if (cbge_ship) {
                // month-GE fast path
                ship_filter = cbge_ship->or_many(Q3_GE_AFTER_MONTHS());
                t_c = clk::now();
                cb_btv_res &= ship_filter;
                t_d = clk::now();
                q3_get_rowids_t(cb_btv_res, &ids);
                auto t_e1 = clk::now();
                auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);
                q3_aggregate(context.client, lineitem_table, lineitem_tx, ids, l_orderkey_map);
                auto t_e2 = clk::now();

                struct HeapEntry {
                    int64_t orderkey; int64_t revenue; int32_t orderdate; int32_t shippriority;
                };
                auto cmp = [](const HeapEntry& a, const HeapEntry& b) {
                    if (a.revenue != b.revenue) return a.revenue > b.revenue;
                    return a.orderdate < b.orderdate;
                };
                // top-10 heap
                std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> heap(cmp);
                for (auto& [k, acc] : l_orderkey_map) {
                    HeapEntry e{k, acc.revenue_raw, acc.orderdate, acc.shippriority};
                    if (heap.size() < 10) heap.push(e);
                    else if (heap.top().revenue < e.revenue ||
                             (heap.top().revenue == e.revenue && heap.top().orderdate > e.orderdate)) {
                        heap.pop(); heap.push(e);
                    }
                }
                std::map<int64_t, Q3Acc> top10_map;
                while (!heap.empty()) {
                    const HeapEntry& e = heap.top();
                    top10_map[e.orderkey] = {e.revenue, e.orderdate, e.shippriority};
                    heap.pop();
                }
                last_top10_map = top10_map;

                double tA  = q3_ms(t0, t_a);
                double tB  = q3_ms(t_a, t_b);
                double tC  = q3_ms(t_b, t_c);
                double tD  = q3_ms(t_c, t_e1);
                double tE  = q3_ms(t_e1, t_e2);
                double tot = q3_ms(t0, t_e2);
                std::cout << "  " << idx_okey->backend_name()
                          << ":  PhaseA(cust_scan)=" << tA
                          << "  PhaseB(orders+okey_OR)=" << tB
                          << "  PhaseC(shipdate_OR)=" << tC
                          << "  PhaseD(AND+rowids)=" << tD
                          << "  PhaseE(BMFetch+agg)=" << tE
                          << "  Total=" << tot
                          << "  rows=" << ids.size() << std::endl;
                if (!warm) {
                    tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC);
                    tD_t.push_back(tD); tE_t.push_back(tE); tot_t.push_back(tot);
                }
                continue;
            }

            auto* cb_ship_bpe = dynamic_cast<bm_index::IndexedDDCBPE*>(
                static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_BPE));

            auto* cb_ship_ge = dynamic_cast<bm_index::IndexedDDCGE*>(
                static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_GE));
            ship_filter = DDC::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
            if (cb_ship_bpe) {                 // BPE prefix range
                int B    = cb_ship_bpe->bucket_of(Q3_CUTOFF_DATE);
                int LAST = static_cast<int>(cb_ship_bpe->num_keys()) - 1;
                ship_filter = *cb_ship_bpe->prefix_at_bucket(LAST);
                ship_filter &= ~(*cb_ship_bpe->prefix_at_bucket(B));
                int64_t bmax = cb_ship_bpe->bucket_max(B);
                if (Q3_CUTOFF_DATE + 1 <= bmax) {
                    DDC boundary = DDC::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                    cb_ship->apply_or_range_to(boundary, Q3_CUTOFF_DATE + 1, bmax);
                    ship_filter |= boundary;
                }
            } else if (cb_ship_ge) {

                ship_filter = cb_ship_ge->or_many(std::vector<int64_t>{1996, 1997, 1998});
                DDC boundary = DDC::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                cb_ship->apply_or_range_to(boundary, Q3_CUTOFF_DATE + 1, 9495);
                ship_filter |= boundary;
            } else {
                // per-key OR fallback
                std::vector<int64_t> in_range;
                cb_ship->for_each_key([&](int64_t k) {
                    if (k > Q3_CUTOFF_DATE) in_range.push_back(k);
                });
                ship_filter = cb_ship->or_many(in_range);
            }
            t_c = clk::now();
            cb_btv_res &= ship_filter;   // AND
            t_d = clk::now();
            q3_get_rowids_t(cb_btv_res, &ids);
        } else if (cr_okey) {
            auto* cr_ship = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ship_use);

            auto* cr_ship_bpe = dynamic_cast<bm_index::IndexedCRoaringBPE*>(
                static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_BPE));
            roaring::Roaring ship_filter;
            if (use_month_ge_q3) {
                ship_filter = cr_ship->or_many(Q3_GE_AFTER_MONTHS());
            } else if (cr_ship_bpe) {
                int B    = cr_ship_bpe->bucket_of(Q3_CUTOFF_DATE);
                int LAST = static_cast<int>(cr_ship_bpe->num_keys()) - 1;
                ship_filter = *cr_ship_bpe->prefix_at_bucket(LAST);
                ship_filter -= *cr_ship_bpe->prefix_at_bucket(B);
                int64_t bmax = cr_ship_bpe->bucket_max(B);
                if (Q3_CUTOFF_DATE + 1 <= bmax) {
                    roaring::Roaring boundary;
                    if (cr_ship->run_optimized()) boundary = cr_ship->or_range(Q3_CUTOFF_DATE + 1, bmax);
                    else cr_ship->apply_or_range_to(boundary, Q3_CUTOFF_DATE + 1, bmax);
                    ship_filter |= boundary;
                }
            } else if (cr_ship->run_optimized()) {
                // run-container range OR
                ship_filter = cr_ship->or_range(Q3_CUTOFF_DATE + 1, INT64_MAX);
            } else {

                cr_ship->apply_or_range_to(ship_filter, Q3_CUTOFF_DATE + 1, INT64_MAX);
            }
            t_c = clk::now();
            cr_btv_res &= ship_filter;
            t_d = clk::now();
            q3_get_rowids_t(cr_btv_res, &ids);
        } else if (bs_okey) {
            auto* bs_ship = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_ship_use);
            ::bitset::BitsetVector ship_filter = bs_ship->make_empty();
            if (use_month_ge_q3) ship_filter = bs_ship->or_many(Q3_GE_AFTER_MONTHS());
            else bs_ship->apply_or_range_to(ship_filter, Q3_CUTOFF_DATE + 1, INT64_MAX);
            t_c = clk::now();
            ::bitset::BitsetVector::word_and_inplace(bs_btv_res, ship_filter, bs_okey->use_simd());
            t_d = clk::now();
            q3_get_rowids_t(bs_btv_res, &ids);
        } else if (wah_okey) {

            auto* wah_ship = dynamic_cast<bm_index::IndexedWAH*>(idx_ship_use);
            ibis::bitvector ship_filter = use_month_ge_q3
                ? wah_ship->or_many(Q3_GE_AFTER_MONTHS())
                : wah_ship->or_range(Q3_CUTOFF_DATE + 1, INT64_MAX);
            t_c = clk::now();
            wah_btv_res &= ship_filter;
            t_d = clk::now();
            q3_get_rowids_t(wah_btv_res, &ids);
        } else if (ew_okey) {

            auto* ew_ship = dynamic_cast<bm_index::IndexedEWAH*>(idx_ship_use);
            ewah::EWAHBoolArray<uint64_t> ship_filter = use_month_ge_q3
                ? ew_ship->or_many(Q3_GE_AFTER_MONTHS())
                : ew_ship->or_range(Q3_CUTOFF_DATE + 1, INT64_MAX);
            t_c = clk::now();
            ewah::EWAHBoolArray<uint64_t> tmp;
            ew_btv_res.logicaland(ship_filter, tmp);
            ew_btv_res = std::move(tmp);
            t_d = clk::now();
            q3_get_rowids_t(ew_btv_res, &ids);
        } else if (con_okey) {

            auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_ship_use);
            ConciseSet<false> ship_filter = use_month_ge_q3
                ? con_ship->or_many(Q3_GE_AFTER_MONTHS())
                : con_ship->or_range(Q3_CUTOFF_DATE + 1, INT64_MAX);
            t_c = clk::now();
            con_btv_res = con_btv_res.logicaland(ship_filter);
            t_d = clk::now();
            q3_get_rowids_t(con_btv_res, &ids);
        } else {
            std::cerr << "[Q3] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }
        auto t_e1 = clk::now();

        // Fetch aggregate
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);
        q3_aggregate(context.client, lineitem_table, lineitem_tx, ids, l_orderkey_map);
        auto t_e2 = clk::now();

        struct HeapEntry {
            int64_t orderkey;
            int64_t revenue;
            int32_t orderdate;
            int32_t shippriority;
        };
        auto cmp = [](const HeapEntry& a, const HeapEntry& b) {

            if (a.revenue != b.revenue) return a.revenue > b.revenue;
            return a.orderdate < b.orderdate;
        };
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> heap(cmp);
        for (auto& [k, acc] : l_orderkey_map) {
            HeapEntry e{k, acc.revenue_raw, acc.orderdate, acc.shippriority};
            if (heap.size() < 10) heap.push(e);
            else if (heap.top().revenue < e.revenue ||
                     (heap.top().revenue == e.revenue && heap.top().orderdate > e.orderdate)) {
                heap.pop(); heap.push(e);
            }
        }
        std::map<int64_t, Q3Acc> top10_map;
        while (!heap.empty()) {
            const HeapEntry& e = heap.top();
            top10_map[e.orderkey] = {e.revenue, e.orderdate, e.shippriority};
            heap.pop();
        }
        last_top10_map = top10_map;

        double tA  = q3_ms(t0, t_a);
        double tB  = q3_ms(t_a, t_b);
        double tC  = q3_ms(t_b, t_c);
        double tD  = q3_ms(t_c, t_e1);
        double tE  = q3_ms(t_e1, t_e2);
        double tot = q3_ms(t0, t_e2);
        std::cout << "  " << idx_okey->backend_name()
                  << ":  PhaseA(cust_scan)=" << tA
                  << "  PhaseB(orders+okey_OR)=" << tB
                  << "  PhaseC(shipdate_OR)=" << tC
                  << "  PhaseD(AND+rowids)=" << tD
                  << "  PhaseE(BMFetch+agg)=" << tE
                  << "  Total=" << tot
                  << "  rows=" << ids.size() << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC);
            tD_t.push_back(tD); tE_t.push_back(tE); tot_t.push_back(tot);
        }
    }

    // SQL validation
    {
        Connection con(*context.client.db);

        const char* ship_cut = use_month_ge_q3
            ? "DATE '1995-04-01'"
            : "DATE '1995-03-15'";
        const char* ship_op = use_month_ge_q3 ? ">=" : ">";
        const std::string sql = std::string(
            "SELECT l_orderkey, "
            "       sum(l_extendedprice * (1 - l_discount)) AS revenue, "
            "       o_orderdate, o_shippriority "
            "FROM   customer, orders, lineitem "
            "WHERE  c_mktsegment = 'BUILDING' "
            "  AND  c_custkey  = o_custkey "
            "  AND  l_orderkey = o_orderkey "
            "  AND  o_orderdate < DATE '1995-03-15' "
            "  AND  l_shipdate  ") + ship_op + " " + ship_cut +
            " GROUP BY l_orderkey, o_orderdate, o_shippriority "
            " ORDER BY revenue DESC, o_orderdate "
            " LIMIT 10";
        auto r = con.Query(sql);
        if (r && !r->HasError()) {
            bool ok = (r->RowCount() == last_top10_map.size());
            std::map<int64_t, double> gt;
            for (idx_t i = 0; i < r->RowCount(); i++) {
                int64_t okey = r->GetValue(0, i).GetValue<int64_t>();
                double rev = r->GetValue(1, i).GetValue<double>();
                gt[okey] = rev;
            }

            for (auto& [okey, acc] : last_top10_map) {
                auto it = gt.find(okey);
                if (it == gt.end()) { ok = false; break; }
                double our = double(acc.revenue_raw) / 10000;
                if (std::fabs(our - it->second) > 0.5) {
                    ok = false;
                    std::cerr << "[FAIL] orderkey=" << okey << " our=" << our
                              << " gt=" << it->second << "\n";
                    break;
                }
            }
            if (ok)
                std::cout << "\n[OK] " << idx_okey->backend_name()
                          << " matches DuckDB SQL ground truth (top "
                          << gt.size() << ").\n";
            else
                std::cerr << "\n[FAIL] mismatch vs SQL ground truth!\n";
        }
    }

    std::cout << "\n  Q3 top-10 (orderkey, revenue, orderdate, shippriority):\n";
    std::vector<std::pair<int64_t, Q3Acc>> sorted(last_top10_map.begin(), last_top10_map.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.second.revenue_raw != b.second.revenue_raw)
            return a.second.revenue_raw > b.second.revenue_raw;
        return a.second.orderdate < b.second.orderdate;
    });
    for (auto& [okey, acc] : sorted) {
        date_t dt = Date::EpochDaysToDate(acc.orderdate);
        std::cout << "    " << okey << "  "
                  << std::fixed << std::setprecision(4) << double(acc.revenue_raw) / 10000
                  << "  " << Date::ToString(dt) << "  " << acc.shippriority << "\n";
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t),
         sD = stats(tD_t), sE = stats(tE_t), sT = stats(tot_t);
    int measured = std::max(0, Q3_ITERATIONS - Q3_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q3 RESULTS — " << idx_okey->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (customer scan)    : " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (orders+orderkey OR): " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (shipdate range OR): " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  PhaseD (AND + get rowids) : " << sD.median << " +/- " << sD.stddev << " ms\n";
    std::cout << "  PhaseE (BMFetch + agg)    : " << sE.median << " +/- " << sE.stddev << " ms\n";
    std::cout << "  TOTAL                     : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q3_results_" + sf + ".csv");
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
        put("PhaseA_cust", sA);
        put("PhaseB_okey", sB);
        put("PhaseC_ship", sC);
        put("PhaseD_AND",  sD);
        put("PhaseE_agg",  sE);
        put("TOTAL", sT);
        std::cout << "  [CSV] q3_results_" << sf << ".csv\n";
    }

    });
}

}
