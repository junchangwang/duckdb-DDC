// TPC-H Q8 — National Market Share (spec v3.0.1 §2.4.8)
//
//   SELECT o_year,
//          SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume)
//            AS mkt_share
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
// TPC-H 1.5.7 compliance: bitmap_partkey references l_partkey (FK,
// strict-compliant); bitmap_orderkey references l_orderkey (FK,
// strict-compliant).  All multi-table joins (region/nation/customer/
// supplier/orders) and the o_orderdate range are evaluated at
// query-time on fresh table scans — there is no pre-built
// lineitem×orders×part bitmap (the prior in_america_part_match aux
// joined three tables and violated 1.5.7 §5).
//
// Pipeline:
//   Phase A: build america_customers (region→nation→customer chain) +
//            brazil_suppliers (nation BRAZIL → supplier set) +
//            part_set (p_type='ECONOMY ANODIZED STEEL').
//   Phase B: scan orders → orderdate IN [1995,1996] AND custkey ∈
//            america_customers → orderkey_year map.
//   Phase C: bitmap_orderkey OR (orderkey_year.keys) → orderkey_filter.
//   Phase D: bitmap_partkey OR (part_set) → part_filter.
//   Phase E: orderkey_filter & part_filter → mask; get_rowids.
//   Phase F: BMFetch lineitem(orderkey, partkey, suppkey, extprice,
//            discount); per-row look up year via orderkey, accumulate
//            volume per year (denominator) and brazil-only volume per
//            year (numerator).
//   Phase G: compute mkt_share = numer / denom per year.

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

#include "combit_adapter.h"
#include "combit/include/combit.h"
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q8_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q8_ITERATIONS = bm_bench::iter_count(5);
static const int Q8_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q8_once_flag_new;

// 1995-01-01 / 1997-01-01 epoch days.  Q8 spec is BETWEEN 1995-01-01
// AND 1996-12-31 inclusive; we compare with `< Q8_DATE_HI`.
static constexpr int64_t Q8_DATE_LO = 9131;   // 1995-01-01
static constexpr int64_t Q8_DATE_HI = 9862;   // 1997-01-01

static constexpr int Q8_YEAR_LO = 1995;
static constexpr int Q8_YEAR_HI = 1996;

// Convert an epoch-day to its calendar year — copy of the closed-form
// used in debit_extension.cpp.  Hand-rolled here to avoid pulling in
// the full epoch_day_to_year helper.
static inline int q8_day_to_year(int32_t day) {
    if (day < 8401)  return 1992;   // 1992-01-01..1992-12-31
    if (day < 8766)  return 1993;
    if (day < 9131)  return 1994;
    if (day < 9496)  return 1995;
    if (day < 9862)  return 1996;
    if (day < 10227) return 1997;
    return 1998;
}

template <typename Btv>
static void q8_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q8_get_rowids<ComBit>(const ComBit& b, std::vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++) out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q8_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q8_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q8_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q8_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}

void BMTableScan::BMTPCH_Q8(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q8_once_flag_new, [&]() {
    bm_bench::warn_if_sf1();

    auto& region_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "region");
    auto& nation_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "nation");
    auto& customer_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "customer");
    auto& supplier_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "supplier");
    auto& part_table     = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "part");
    auto& orders_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q8 (BitEngine pattern, runtime semi-joins)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_partkey + bitmap_orderkey" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_okey = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_orderkey);
    auto* idx_pk   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_partkey);
    if (!idx_okey || !idx_pk) {
        std::cerr << "[Q8] ERROR: bitmap_orderkey or bitmap_partkey not loaded.\n";
        return;
    }

    std::vector<double> tA_t, tB_t, tCD_t, tE_t, tF_t, tot_t;
    double last_share_1995 = 0, last_share_1996 = 0;

    for (int iter = 0; iter < Q8_ITERATIONS; iter++) {
        bool warm = iter < Q8_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q8_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        // ===== Phase A: build america_customers + brazil_suppliers + part_set =====
        // NOTE: region/nation key columns are INTEGER (int32_t) in the
        // TPC-H schema, NOT BIGINT.  Reading them as int64 produces
        // garbage (two int32 values smushed into 64 bits).
        // A1: region scan -> AMERICA region key
        int32_t america_regionkey = -1;
        {
            auto& tx = DuckTransaction::Get(context.client, region_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(1)};
            region_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types = {
                region_table.GetColumns().GetColumnTypes()[0],
                region_table.GetColumns().GetColumnTypes()[1],
            };
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                region_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[1].Flatten(chunk.size());
                auto rk = FlatVector::GetData<int32_t>(chunk.data[0]);
                auto rn = FlatVector::GetData<string_t>(chunk.data[1]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    auto sv = rn[i];
                    if (sv.GetSize() == 7 && std::memcmp(sv.GetData(), "AMERICA", 7) == 0)
                        america_regionkey = rk[i];
                }
            }
        }

        // A2: nation scan -> america_nationkeys + brazil_nationkey
        std::unordered_set<int32_t> america_nationkeys;
        int32_t brazil_nationkey = -1;
        {
            auto& tx = DuckTransaction::Get(context.client, nation_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(1), StorageIndex(2)};
            nation_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types;
            for (int c : {0, 1, 2})
                types.push_back(nation_table.GetColumns().GetColumnTypes()[c]);
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                nation_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[1].Flatten(chunk.size());
                auto nk = FlatVector::GetData<int32_t>(chunk.data[0]);
                auto nn = FlatVector::GetData<string_t>(chunk.data[1]);
                auto rk = FlatVector::GetData<int32_t>(chunk.data[2]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (rk[i] == america_regionkey)
                        america_nationkeys.insert(nk[i]);
                    auto sv = nn[i];
                    if (sv.GetSize() == 6 && std::memcmp(sv.GetData(), "BRAZIL", 6) == 0)
                        brazil_nationkey = nk[i];
                }
            }
        }

        // A3: customer scan -> america_customers (custkeys whose nation in AMERICA region)
        std::unordered_set<int64_t> america_customers;
        america_customers.reserve(300000);
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
                auto cn = FlatVector::GetData<int32_t>(chunk.data[1]);  // INTEGER, not BIGINT
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (america_nationkeys.count(cn[i]))
                        america_customers.insert(ck[i]);
                }
            }
        }

        // A4: supplier scan -> suppkey -> nationkey map (for BRAZIL check at agg time)
        std::unordered_map<int64_t, int32_t> supp_to_nation;
        supp_to_nation.reserve(150000);
        {
            auto& tx = DuckTransaction::Get(context.client, supplier_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(3)};
            supplier_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types = {
                supplier_table.GetColumns().GetColumnTypes()[0],
                supplier_table.GetColumns().GetColumnTypes()[3],
            };
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                supplier_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                auto sk = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto sn = FlatVector::GetData<int32_t>(chunk.data[1]);  // INTEGER
                for (idx_t i = 0; i < chunk.size(); i++)
                    supp_to_nation.emplace(sk[i], sn[i]);
            }
        }

        // A5: part scan -> part_set where p_type = 'ECONOMY ANODIZED STEEL'
        std::vector<int64_t> part_set;
        part_set.reserve(2000);
        {
            auto& tx = DuckTransaction::Get(context.client, part_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(4)};
            part_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types = {
                part_table.GetColumns().GetColumnTypes()[0],
                part_table.GetColumns().GetColumnTypes()[4],
            };
            const char* TARGET = "ECONOMY ANODIZED STEEL";
            const size_t TARGET_LEN = 22;
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                part_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[1].Flatten(chunk.size());
                auto pk = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto pt = FlatVector::GetData<string_t>(chunk.data[1]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    auto sv = pt[i];
                    if (sv.GetSize() == TARGET_LEN
                        && std::memcmp(sv.GetData(), TARGET, TARGET_LEN) == 0)
                        part_set.push_back(pk[i]);
                }
            }
        }
        auto t_a = clk::now();

        // ===== Phase B: orders scan -> orderkey -> year map =====
        // Filter o_orderdate ∈ [1995-01-01, 1997-01-01) AND custkey ∈ america.
        std::unordered_map<int64_t, int> okey_to_year;
        okey_to_year.reserve(2000000);
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
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (od[i] < Q8_DATE_LO || od[i] >= Q8_DATE_HI) continue;
                    if (!america_customers.count(ck[i])) continue;
                    okey_to_year.emplace(okey[i], q8_day_to_year(od[i]));
                    matched_okeys.push_back(okey[i]);
                }
            }
        }
        auto t_b = clk::now();

        // ===== Phase C: orderkey OR + Phase D: partkey OR + Phase E: AND + get_rowids =====
        size_t num_rows = idx_okey->num_rows();
        std::vector<row_t> ids;

        // Plan A — each backend uses its natural multi-key OR primitive.
        clk::time_point t_cd;
        if (auto* cb_okey = dynamic_cast<bm_index::IndexedComBit*>(idx_okey)) {
            auto* cb_pk = dynamic_cast<bm_index::IndexedComBit*>(idx_pk);
            if (!cb_pk) { std::cerr << "[Q8] type mismatch.\n"; return; }
            ComBit okey_filter = cb_okey->or_many(matched_okeys);  // ComBit: sca
            ComBit part_filter = cb_pk->or_many(part_set);
            okey_filter &= part_filter;
            q8_get_rowids(okey_filter, &ids);
            t_cd = clk::now();
        } else if (auto* cr_okey = dynamic_cast<bm_index::IndexedCRoaring*>(idx_okey)) {
            auto* cr_pk = dynamic_cast<bm_index::IndexedCRoaring*>(idx_pk);
            if (!cr_pk) { std::cerr << "[Q8] type mismatch.\n"; return; }
            // Both CR & CRR use Roaring fastunion (k-way primitive).
            roaring::Roaring okey_filter = cr_okey->or_many(matched_okeys);
            roaring::Roaring part_filter = cr_pk->or_many(part_set);
            okey_filter &= part_filter;
            q8_get_rowids(okey_filter, &ids);
            t_cd = clk::now();
        } else if (auto* wah_okey = dynamic_cast<bm_index::IndexedWAH*>(idx_okey)) {
            auto* wah_pk = dynamic_cast<bm_index::IndexedWAH*>(idx_pk);
            if (!wah_pk) { std::cerr << "[Q8] type mismatch.\n"; return; }
            ibis::bitvector okey_filter = wah_okey->or_many(matched_okeys);  // WAH: copy+decompress+|=
            ibis::bitvector part_filter = wah_pk->or_many(part_set);
            okey_filter &= part_filter;
            q8_get_rowids(okey_filter, &ids);
            t_cd = clk::now();
        } else if (auto* ew_okey = dynamic_cast<bm_index::IndexedEWAH*>(idx_okey)) {
            auto* ew_pk = dynamic_cast<bm_index::IndexedEWAH*>(idx_pk);
            if (!ew_pk) { std::cerr << "[Q8] type mismatch.\n"; return; }
            ewah::EWAHBoolArray<uint64_t> okey_filter = ew_okey->or_many(matched_okeys);  // EW: fast_logicalor
            ewah::EWAHBoolArray<uint64_t> part_filter = ew_pk->or_many(part_set);
            ewah::EWAHBoolArray<uint64_t> tmp;
            okey_filter.logicaland(part_filter, tmp);
            q8_get_rowids(tmp, &ids);
            t_cd = clk::now();
        } else if (auto* con_okey = dynamic_cast<bm_index::IndexedConcise*>(idx_okey)) {
            auto* con_pk = dynamic_cast<bm_index::IndexedConcise*>(idx_pk);
            if (!con_pk) { std::cerr << "[Q8] type mismatch.\n"; return; }
            ConciseSet<false> okey_filter = con_okey->or_many(matched_okeys);  // CON: fast_logicalor
            ConciseSet<false> part_filter = con_pk->or_many(part_set);
            ConciseSet<false> result = okey_filter.logicaland(part_filter);
            q8_get_rowids(result, &ids);
            t_cd = clk::now();
        } else {
            std::cerr << "[Q8] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }

        // ===== Phase F: BMFetch lineitem + per-year volume + per-year brazil volume =====
        // l_orderkey (0), l_partkey (1) [unused — already filtered],
        // l_suppkey (2), l_extendedprice (5), l_discount (6).
        std::unordered_map<int, int64_t> volume_by_year;
        std::unordered_map<int, int64_t> brazil_by_year;
        if (!ids.empty()) {
            vector<StorageIndex> col_ids = {
                StorageIndex(0), StorageIndex(2), StorageIndex(5), StorageIndex(6)};
            vector<LogicalType> types;
            for (int c : {0, 2, 5, 6})
                types.push_back(lineitem_table.GetColumns().GetColumnTypes()[c]);
            idx_t cursor = 0;
            idx_t num_idlist = ids.size();
            while (cursor < ids.size()) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                ColumnFetchState column_fetch_state;
                data_ptr_t row_ids_data = (data_ptr_t)&ids[cursor];
                Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
                idx_t fetch_count = 2048;
                if (cursor + fetch_count > ids.size()) fetch_count = ids.size() - cursor;
                lineitem_table.GetStorage().BMFetch(lineitem_tx, chunk, col_ids, row_ids_vec,
                                                    fetch_count, column_fetch_state, num_idlist);
                cursor += fetch_count;
                auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto sk   = FlatVector::GetData<int64_t>(chunk.data[1]);
                auto pr   = FlatVector::GetData<int64_t>(chunk.data[2]);
                auto dc   = FlatVector::GetData<int64_t>(chunk.data[3]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    auto yi = okey_to_year.find(okey[i]);
                    if (yi == okey_to_year.end()) continue;
                    int64_t v = pr[i] * (100 - dc[i]);
                    volume_by_year[yi->second] += v;
                    auto si = supp_to_nation.find(sk[i]);
                    if (si != supp_to_nation.end() && si->second == brazil_nationkey)
                        brazil_by_year[yi->second] += v;
                }
            }
        }
        auto t_e = clk::now();

        // ===== Phase G: compute mkt_share per year =====
        double share_1995 = 0, share_1996 = 0;
        {
            auto d95 = volume_by_year[1995];
            auto n95 = brazil_by_year[1995];
            if (d95 > 0) share_1995 = static_cast<double>(n95) / static_cast<double>(d95);
            auto d96 = volume_by_year[1996];
            auto n96 = brazil_by_year[1996];
            if (d96 > 0) share_1996 = static_cast<double>(n96) / static_cast<double>(d96);
        }
        auto t_f = clk::now();

        double tA  = q8_ms(t0, t_a);
        double tB  = q8_ms(t_a, t_b);
        double tCD = q8_ms(t_b, t_cd);
        double tE  = q8_ms(t_cd, t_e);
        double tF  = q8_ms(t_e, t_f);
        double tot = q8_ms(t0, t_f);

        std::cout << "  " << idx_okey->backend_name()
                  << ":  PhaseA(side_tables)=" << tA
                  << "  PhaseB(orders)=" << tB
                  << "  PhaseCDE(okey/part_OR+AND)=" << tCD
                  << "  PhaseF(BMFetch+agg)=" << tE
                  << "  PhaseG(share)=" << tF
                  << "  Total=" << tot
                  << "  rows=" << ids.size()
                  << "  share_1995=" << std::fixed << std::setprecision(6) << share_1995
                  << "  share_1996=" << share_1996
                  << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tCD_t.push_back(tCD);
            tE_t.push_back(tE); tF_t.push_back(tF); tot_t.push_back(tot);
        }
        last_share_1995 = share_1995;
        last_share_1996 = share_1996;
    }

    {
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT o_year, "
            "       SUM(CASE WHEN nation='BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share "
            "FROM ("
            "  SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year, "
            "         l_extendedprice * (1 - l_discount) AS volume, "
            "         n2.n_name AS nation "
            "  FROM part, supplier, lineitem, orders, customer, "
            "       nation n1, region, nation n2 "
            "  WHERE p_partkey = l_partkey "
            "    AND s_suppkey = l_suppkey "
            "    AND l_orderkey = o_orderkey "
            "    AND o_custkey = c_custkey "
            "    AND c_nationkey = n1.n_nationkey "
            "    AND n1.n_regionkey = r_regionkey "
            "    AND r_name = 'AMERICA' "
            "    AND s_nationkey = n2.n_nationkey "
            "    AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31' "
            "    AND p_type = 'ECONOMY ANODIZED STEEL'"
            ") AS all_nations "
            "GROUP BY o_year ORDER BY o_year");
        if (r && !r->HasError() && r->RowCount() == 2) {
            double gt95 = r->GetValue(1, 0).GetValue<double>();
            double gt96 = r->GetValue(1, 1).GetValue<double>();
            bool ok = std::fabs(gt95 - last_share_1995) < 0.0001
                   && std::fabs(gt96 - last_share_1996) < 0.0001;
            if (ok) {
                std::cout << "\n[OK] " << idx_okey->backend_name()
                          << " matches DuckDB SQL ground truth "
                          << "(1995=" << gt95 << ", 1996=" << gt96 << ").\n";
            } else {
                std::cerr << "\n[FAIL] mismatch: ours 1995=" << last_share_1995
                          << " 1996=" << last_share_1996
                          << " gt 1995=" << gt95 << " 1996=" << gt96 << "\n";
            }
        }
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sCD = stats(tCD_t),
         sE = stats(tE_t), sF = stats(tF_t), sT = stats(tot_t);
    int measured = std::max(0, Q8_ITERATIONS - Q8_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q8 RESULTS — " << idx_okey->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (side tables)         : " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (orders scan)         : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseCDE (okey + part OR+AND): " << sCD.median << " +/- " << sCD.stddev << " ms\n";
    std::cout << "  PhaseF (BMFetch + agg)       : " << sE.median << " +/- " << sE.stddev << " ms\n";
    std::cout << "  PhaseG (share calc)          : " << sF.median << " +/- " << sF.stddev << " ms\n";
    std::cout << "  TOTAL                        : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q8_results_" + sf + ".csv");
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
        bm_bench::Stats z{0,0,0,0};
        std::string bn = idx_okey->backend_name();
        auto put = [&](const std::string& op, bm_bench::Stats s) {
            csv << sf << "," << op << ",";
            auto cell = [&](bm_bench::Stats v) {
                csv << v.median << "," << v.stddev << "," << v.min_val << "," << v.max_val << ",";
            };
            cell(bn == "WAH" ? s : z);
            cell(bn == "ComBit" ? s : z);
            cell(bn == "CRoaring" ? s : z);
            cell(bn == "CRoaringRun" ? s : z);
            cell(bn == "EWAH" ? s : z);
            cell(z); cell(z);
            cell(bn == "Concise" ? s : z);
            csv << "0,0,0,0,0,0,0\n";
        };
        put("PhaseA_side",  sA);
        put("PhaseB_orders", sB);
        put("PhaseCDE_OR_AND", sCD);
        put("PhaseF_agg",   sE);
        put("PhaseG_share", sF);
        put("TOTAL", sT);
        std::cout << "  [CSV] q8_results_" << sf << ".csv\n";
    }

    });  // end call_once
}

}  // namespace duckdb
