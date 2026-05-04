// TPC-H Q12 — Shipping Modes and Order Priority (spec v3.0.1 §2.4.12)
//
//   SELECT l_shipmode,
//          SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH')
//                   THEN 1 ELSE 0 END) AS high_line_count,
//          SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH')
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
// TPC-H 1.5.7 compliance: bitmap_shipmode references l_shipmode (VARCHAR
// — kept per teacher's BitEngine convention, same caveat as Q1/Q19);
// bitmap_receiptdate references l_receiptdate (date, strict-compliant).
// l_commitdate < l_receiptdate and l_shipdate < l_commitdate are
// multi-column predicates and must be evaluated at query-time on the
// BMFetch'd columns — they cannot be pre-derived without breaking the
// single-column rule (the prior commit_lt_receipt / ship_lt_commit aux
// structures violated 1.5.7 §5).
//
// Pipeline:
//   Phase A: scan orders → orderkey -> orderpriority class map.
//   Phase B: shipmode OR ('MAIL','SHIP') → shipmode_filter.
//   Phase C: receiptdate range OR ([1994-01-01, 1995-01-01)) → date_filter.
//   Phase D: shipmode_filter & date_filter → mask; get_rowids.
//   Phase E: BMFetch lineitem(orderkey, commitdate, receiptdate, shipdate,
//            shipmode); per-row check ship<commit<receipt; lookup
//            orderkey priority class; accumulate (shipmode, class) counts.

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
static inline double q12_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q12_ITERATIONS = bm_bench::iter_count(5);
static const int Q12_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q12_once_flag_new;

// 1994-01-01 / 1995-01-01 epoch days (relative to 1970-01-01).
static constexpr int64_t Q12_DATE_LO = 8766;
static constexpr int64_t Q12_DATE_HI = 9131;

// l_shipmode IN ('MAIL', 'SHIP').  bitmap_shipmode encodes first ASCII
// byte (see scan_column / debit_extension.cpp).  M = 77, S = 83.  Both
// are unique first letters in TPC-H's shipmode domain.
static constexpr int64_t Q12_SHIPMODE_MAIL = 'M';
static constexpr int64_t Q12_SHIPMODE_SHIP = 'S';

template <typename Btv>
static void q12_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q12_get_rowids<ComBit>(const ComBit& b, std::vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++) out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q12_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q12_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q12_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q12_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}

// orderkey -> orderpriority class (true = high {1-URGENT,2-HIGH}, false = low).
static std::unordered_map<int64_t, bool> q12_load_orderkey_priority(
        ClientContext& ctx, TableCatalogEntry& orders_table) {
    std::unordered_map<int64_t, bool> m;
    m.reserve(15000000);
    auto& tx = DuckTransaction::Get(ctx, orders_table.catalog);
    TableScanState ss;
    // o_orderkey, o_orderpriority
    vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(5)};
    orders_table.GetStorage().InitializeScan(ctx, tx, ss, col_ids);
    vector<LogicalType> types = {
        orders_table.GetColumns().GetColumnTypes()[0],
        orders_table.GetColumns().GetColumnTypes()[5],
    };
    while (true) {
        DataChunk chunk; chunk.Initialize(ctx, types);
        orders_table.GetStorage().Scan(tx, chunk, ss);
        if (chunk.size() == 0) break;
        chunk.data[1].Flatten(chunk.size());
        auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
        auto opri = FlatVector::GetData<string_t>(chunk.data[1]);
        for (idx_t i = 0; i < chunk.size(); i++) {
            auto sv = opri[i];
            // First char is the priority digit ('1'..'5').  '1' / '2' = high.
            const char c = sv.GetSize() > 0 ? sv.GetData()[0] : '0';
            m[okey[i]] = (c == '1' || c == '2');
        }
    }
    return m;
}

void BMTableScan::BMTPCH_Q12(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q12_once_flag_new, [&]() {
    bm_bench::warn_if_sf1();

    auto& orders_table   = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q12 (BitEngine pattern, runtime semi-joins)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_shipmode + bitmap_receiptdate" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_sm   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipmode);
    auto* idx_rdat = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_receiptdate);
    if (!idx_sm || !idx_rdat) {
        std::cerr << "[Q12] ERROR: bitmap_shipmode or bitmap_receiptdate not loaded.\n";
        return;
    }

    std::vector<double> tA_t, tB_t, tC_t, tD_t, tE_t, tot_t;
    int64_t last_high_mail = 0, last_low_mail = 0, last_high_ship = 0, last_low_ship = 0;

    for (int iter = 0; iter < Q12_ITERATIONS; iter++) {
        bool warm = iter < Q12_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q12_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        // ===== Phase A: orders scan -> orderkey priority map =====
        auto orderkey_priority = q12_load_orderkey_priority(context.client, orders_table);
        auto t_a = clk::now();

        // ===== Phase B: shipmode OR ('MAIL','SHIP') =====
        // ===== Phase C: receiptdate range OR =====
        // ===== Phase D: AND + get_rowids =====
        size_t num_rows = idx_sm->num_rows();
        std::vector<row_t> ids;
        std::vector<int64_t> sm_keys = {Q12_SHIPMODE_MAIL, Q12_SHIPMODE_SHIP};

        clk::time_point t_b, t_c, t_d;
        if (auto* cb_sm = dynamic_cast<bm_index::IndexedComBit*>(idx_sm)) {
            auto* cb_rdat = dynamic_cast<bm_index::IndexedComBit*>(idx_rdat);
            if (!cb_rdat) { std::cerr << "[Q12] type mismatch.\n"; return; }
            ComBit sm_filter = ComBit::from_sparse_positions({}, num_rows, cb_sm->segment_bits());
            for (auto k : sm_keys) cb_sm->apply_or_to(sm_filter, k);
            t_b = clk::now();
            ComBit date_filter = ComBit::from_sparse_positions({}, num_rows, cb_rdat->segment_bits());
            cb_rdat->apply_or_range_to(date_filter, Q12_DATE_LO, Q12_DATE_HI - 1);
            t_c = clk::now();
            sm_filter &= date_filter;
            q12_get_rowids(sm_filter, &ids);
            t_d = clk::now();
        } else if (auto* cr_sm = dynamic_cast<bm_index::IndexedCRoaring*>(idx_sm)) {
            auto* cr_rdat = dynamic_cast<bm_index::IndexedCRoaring*>(idx_rdat);
            if (!cr_rdat) { std::cerr << "[Q12] type mismatch.\n"; return; }
            roaring::Roaring sm_filter;
            if (cr_sm->run_optimized()) sm_filter = cr_sm->or_many(sm_keys);
            else for (auto k : sm_keys) cr_sm->apply_or_to(sm_filter, k);
            t_b = clk::now();
            roaring::Roaring date_filter;
            if (cr_rdat->run_optimized()) date_filter = cr_rdat->or_range(Q12_DATE_LO, Q12_DATE_HI - 1);
            else cr_rdat->apply_or_range_to(date_filter, Q12_DATE_LO, Q12_DATE_HI - 1);
            t_c = clk::now();
            sm_filter &= date_filter;
            q12_get_rowids(sm_filter, &ids);
            t_d = clk::now();
        } else if (auto* wah_sm = dynamic_cast<bm_index::IndexedWAH*>(idx_sm)) {
            auto* wah_rdat = dynamic_cast<bm_index::IndexedWAH*>(idx_rdat);
            if (!wah_rdat) { std::cerr << "[Q12] type mismatch.\n"; return; }
            ibis::bitvector sm_filter = wah_sm->or_many(sm_keys);
            t_b = clk::now();
            ibis::bitvector date_filter = wah_rdat->or_range(Q12_DATE_LO, Q12_DATE_HI - 1);
            t_c = clk::now();
            sm_filter &= date_filter;
            q12_get_rowids(sm_filter, &ids);
            t_d = clk::now();
        } else if (auto* ew_sm = dynamic_cast<bm_index::IndexedEWAH*>(idx_sm)) {
            auto* ew_rdat = dynamic_cast<bm_index::IndexedEWAH*>(idx_rdat);
            if (!ew_rdat) { std::cerr << "[Q12] type mismatch.\n"; return; }
            ewah::EWAHBoolArray<uint64_t> sm_filter = ew_sm->or_many(sm_keys);
            t_b = clk::now();
            ewah::EWAHBoolArray<uint64_t> date_filter = ew_rdat->or_range(Q12_DATE_LO, Q12_DATE_HI - 1);
            t_c = clk::now();
            ewah::EWAHBoolArray<uint64_t> tmp;
            sm_filter.logicaland(date_filter, tmp);
            q12_get_rowids(tmp, &ids);
            t_d = clk::now();
        } else if (auto* con_sm = dynamic_cast<bm_index::IndexedConcise*>(idx_sm)) {
            auto* con_rdat = dynamic_cast<bm_index::IndexedConcise*>(idx_rdat);
            if (!con_rdat) { std::cerr << "[Q12] type mismatch.\n"; return; }
            ConciseSet<false> sm_filter = con_sm->or_many(sm_keys);
            t_b = clk::now();
            ConciseSet<false> date_filter = con_rdat->or_range(Q12_DATE_LO, Q12_DATE_HI - 1);
            t_c = clk::now();
            ConciseSet<false> result = sm_filter.logicaland(date_filter);
            q12_get_rowids(result, &ids);
            t_d = clk::now();
        } else {
            std::cerr << "[Q12] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }

        // ===== Phase E: BMFetch lineitem + per-row predicate + group =====
        // Need l_orderkey (0), l_shipdate (10), l_commitdate (11),
        // l_receiptdate (12), l_shipmode (14).
        int64_t high_mail = 0, low_mail = 0, high_ship = 0, low_ship = 0;
        if (!ids.empty()) {
            vector<StorageIndex> col_ids = {
                StorageIndex(0), StorageIndex(10), StorageIndex(11),
                StorageIndex(12), StorageIndex(14)};
            vector<LogicalType> types;
            for (int c : {0, 10, 11, 12, 14})
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
                chunk.data[4].Flatten(chunk.size());
                auto okey = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto sd   = FlatVector::GetData<int32_t>(chunk.data[1]);
                auto cd   = FlatVector::GetData<int32_t>(chunk.data[2]);
                auto rd   = FlatVector::GetData<int32_t>(chunk.data[3]);
                auto sm   = FlatVector::GetData<string_t>(chunk.data[4]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (!(cd[i] < rd[i])) continue;
                    if (!(sd[i] < cd[i])) continue;
                    auto sv = sm[i];
                    if (sv.GetSize() == 0) continue;
                    char first = sv.GetData()[0];
                    auto it = orderkey_priority.find(okey[i]);
                    bool high = (it != orderkey_priority.end()) && it->second;
                    if (first == 'M') {
                        if (high) high_mail++; else low_mail++;
                    } else if (first == 'S') {
                        if (high) high_ship++; else low_ship++;
                    }
                }
            }
        }
        auto t_e = clk::now();

        double tA  = q12_ms(t0, t_a);
        double tB  = q12_ms(t_a, t_b);
        double tC  = q12_ms(t_b, t_c);
        double tD  = q12_ms(t_c, t_d);
        double tE  = q12_ms(t_d, t_e);
        double tot = q12_ms(t0, t_e);

        std::cout << "  " << idx_sm->backend_name()
                  << ":  PhaseA(orders)=" << tA
                  << "  PhaseB(shipmode_OR)=" << tB
                  << "  PhaseC(rdate_OR)=" << tC
                  << "  PhaseD(AND+rids)=" << tD
                  << "  PhaseE(BMFetch+agg)=" << tE
                  << "  Total=" << tot
                  << "  rows=" << ids.size()
                  << "  MAIL=" << high_mail << "/" << low_mail
                  << "  SHIP=" << high_ship << "/" << low_ship
                  << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC);
            tD_t.push_back(tD); tE_t.push_back(tE); tot_t.push_back(tot);
        }
        last_high_mail = high_mail; last_low_mail = low_mail;
        last_high_ship = high_ship; last_low_ship = low_ship;
    }

    {
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT l_shipmode, "
            "       SUM(CASE WHEN o_orderpriority='1-URGENT' OR o_orderpriority='2-HIGH' THEN 1 ELSE 0 END) AS high_line_count, "
            "       SUM(CASE WHEN o_orderpriority<>'1-URGENT' AND o_orderpriority<>'2-HIGH' THEN 1 ELSE 0 END) AS low_line_count "
            "FROM   orders, lineitem "
            "WHERE  o_orderkey = l_orderkey "
            "  AND  l_shipmode IN ('MAIL','SHIP') "
            "  AND  l_commitdate < l_receiptdate "
            "  AND  l_shipdate   < l_commitdate "
            "  AND  l_receiptdate >= DATE '1994-01-01' "
            "  AND  l_receiptdate <  DATE '1995-01-01' "
            "GROUP BY l_shipmode ORDER BY l_shipmode");
        if (r && !r->HasError() && r->RowCount() == 2) {
            int64_t mail_h = r->GetValue(1, 0).GetValue<int64_t>();
            int64_t mail_l = r->GetValue(2, 0).GetValue<int64_t>();
            int64_t ship_h = r->GetValue(1, 1).GetValue<int64_t>();
            int64_t ship_l = r->GetValue(2, 1).GetValue<int64_t>();
            bool ok = mail_h == last_high_mail && mail_l == last_low_mail
                   && ship_h == last_high_ship && ship_l == last_low_ship;
            if (ok) {
                std::cout << "\n[OK] " << idx_sm->backend_name()
                          << " matches DuckDB SQL ground truth "
                          << "(MAIL " << mail_h << "/" << mail_l
                          << ", SHIP " << ship_h << "/" << ship_l << ").\n";
            } else {
                std::cerr << "\n[FAIL] mismatch: ours MAIL=" << last_high_mail << "/" << last_low_mail
                          << " SHIP=" << last_high_ship << "/" << last_low_ship
                          << " gt MAIL=" << mail_h << "/" << mail_l
                          << " SHIP=" << ship_h << "/" << ship_l << "\n";
            }
        }
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t),
         sD = stats(tD_t), sE = stats(tE_t), sT = stats(tot_t);
    int measured = std::max(0, Q12_ITERATIONS - Q12_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q12 RESULTS — " << idx_sm->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (orders scan)         : " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (shipmode OR)         : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (receiptdate range OR): " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  PhaseD (AND + get_rowids)    : " << sD.median << " +/- " << sD.stddev << " ms\n";
    std::cout << "  PhaseE (BMFetch + agg)       : " << sE.median << " +/- " << sE.stddev << " ms\n";
    std::cout << "  TOTAL                        : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q12_results_" + sf + ".csv");
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
        std::string bn = idx_sm->backend_name();
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
        put("PhaseA_orders",   sA);
        put("PhaseB_shipmode", sB);
        put("PhaseC_rdate",    sC);
        put("PhaseD_AND",      sD);
        put("PhaseE_agg",      sE);
        put("TOTAL", sT);
        std::cout << "  [CSV] q12_results_" << sf << ".csv\n";
    }

    });  // end call_once
}

}  // namespace duckdb
