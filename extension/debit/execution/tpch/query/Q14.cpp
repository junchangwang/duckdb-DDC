// TPC-H Q14 — Promotion Effect Query (spec v3.0.1 §2.4.14)
//
// Verbatim port of teacher's BitEngine BMTPCH_Q14: shipdate-month
// filter via single-column bitmap on lineitem.l_shipdate, then
// runtime semi-join on part (single-column scan of p_type LIKE 'PROMO%').
// TPC-H 1.5.7-compliant — no pre-built lineitem×part bitmap (the prior
// is_promo/0.bm violated the single-base-table rule).
//
// Pipeline:
//   Phase A: shipdate range OR (l_shipdate ∈ [DATE, DATE + 1 month))
//            via bitmap_shipdate (per-day, single column).
//   Phase B: scan part — collect partkeys where p_type LIKE 'PROMO%'.
//   Phase C: get_rowids on shipdate filter; BMFetch lineitem
//            (l_partkey, l_extendedprice, l_discount); accumulate
//            total_rev and (if l_partkey ∈ promo_set) promo_rev.
// Result: 100 * promo_rev / total_rev.

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
#include <unordered_set>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q14_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q14_ITERATIONS = bm_bench::iter_count(5);
static const int Q14_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q14_once_flag_new;

static constexpr int64_t Q14_DATE_LO = 9374;   // 1995-09-01 epoch days
static constexpr int64_t Q14_DATE_HI = 9404;   // 1995-10-01 (excl)

template <typename Btv>
static void q14_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q14_get_rowids<ComBit>(const ComBit& b, std::vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++) out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q14_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q14_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q14_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q14_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}

void BMTableScan::BMTPCH_Q14(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q14_once_flag_new, [&]() {
    bm_bench::warn_if_sf1();

    auto& part_table     = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "part");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q14 (BitEngine pattern, runtime part-join)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_shipdate" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    if (!idx_ship) { std::cerr << "[Q14] ERROR: bitmap_shipdate not loaded.\n"; return; }

    std::vector<double> tA_t, tB_t, tC_t, tot_t;
    double last_promo_revenue = 0;

    for (int iter = 0; iter < Q14_ITERATIONS; iter++) {
        bool warm = iter < Q14_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q14_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        // ===== Phase A: shipdate range filter =====
        size_t num_rows = idx_ship->num_rows();
        // Plan A — each backend uses its natural multi-key OR primitive
        // for the ~30-day shipdate range.
        std::vector<row_t> ids;
        if (auto* cb_ship = dynamic_cast<bm_index::IndexedComBit*>(idx_ship)) {
            // ComBit: gather in-range keys + sca or_many.
            std::vector<int64_t> in_range;
            cb_ship->for_each_key([&](int64_t k) {
                if (k >= Q14_DATE_LO && k < Q14_DATE_HI) in_range.push_back(k);
            });
            ComBit ship_filter = cb_ship->or_many(in_range);
            q14_get_rowids(ship_filter, &ids);
        } else if (auto* cr_ship = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ship)) {
            roaring::Roaring ship_filter;
            if (cr_ship->run_optimized()) {
                ship_filter = cr_ship->or_range(Q14_DATE_LO, Q14_DATE_HI - 1);  // CRR: fastunion
            } else {
                cr_ship->apply_or_range_to(ship_filter, Q14_DATE_LO, Q14_DATE_HI - 1);  // CR: pairwise
            }
            q14_get_rowids(ship_filter, &ids);
        } else if (auto* wah_ship = dynamic_cast<bm_index::IndexedWAH*>(idx_ship)) {
            ibis::bitvector ship_filter = wah_ship->or_range(Q14_DATE_LO, Q14_DATE_HI - 1);
            q14_get_rowids(ship_filter, &ids);
        } else if (auto* ew_ship = dynamic_cast<bm_index::IndexedEWAH*>(idx_ship)) {
            ewah::EWAHBoolArray<uint64_t> ship_filter =
                ew_ship->or_range(Q14_DATE_LO, Q14_DATE_HI - 1);  // EW: fast_logicalor
            q14_get_rowids(ship_filter, &ids);
        } else if (auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_ship)) {
            ConciseSet<false> ship_filter =
                con_ship->or_range(Q14_DATE_LO, Q14_DATE_HI - 1);  // CON: fast_logicalor
            q14_get_rowids(ship_filter, &ids);
        } else {
            std::cerr << "[Q14] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }
        auto t_a = clk::now();

        // ===== Phase B: part scan → promo_partkeys =====
        std::unordered_set<int64_t> promo_partkeys;
        promo_partkeys.reserve(50000);
        {
            auto& tx = DuckTransaction::Get(context.client, part_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(4)};
            part_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types = {
                part_table.GetColumns().GetColumnTypes()[0],
                part_table.GetColumns().GetColumnTypes()[4],
            };
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                part_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[1].Flatten(chunk.size());
                auto pk = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto pt = FlatVector::GetData<string_t>(chunk.data[1]);
                auto& valid = FlatVector::Validity(chunk.data[1]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (!valid.RowIsValid(i)) continue;
                    auto sz = pt[i].GetSize();
                    if (sz >= 5 && std::memcmp(pt[i].GetData(), "PROMO", 5) == 0)
                        promo_partkeys.insert(pk[i]);
                }
            }
        }
        auto t_b = clk::now();

        // ===== Phase C: BMFetch lineitem + agg =====
        int64_t total_rev = 0, promo_rev = 0;
        if (!ids.empty()) {
            vector<StorageIndex> col_ids = {StorageIndex(1), StorageIndex(5), StorageIndex(6)};
            vector<LogicalType> types = {
                lineitem_table.GetColumns().GetColumnTypes()[1],
                lineitem_table.GetColumns().GetColumnTypes()[5],
                lineitem_table.GetColumns().GetColumnTypes()[6],
            };
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
                auto pk = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto pr = FlatVector::GetData<int64_t>(chunk.data[1]);
                auto dc = FlatVector::GetData<int64_t>(chunk.data[2]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    int64_t v = pr[i] * (100 - dc[i]);
                    total_rev += v;
                    if (promo_partkeys.count(pk[i])) promo_rev += v;
                }
            }
        }
        auto t_c = clk::now();

        double tA  = q14_ms(t0, t_a);
        double tB  = q14_ms(t_a, t_b);
        double tC  = q14_ms(t_b, t_c);
        double tot = q14_ms(t0, t_c);
        double promo_pct = total_rev > 0 ? 100.0 * double(promo_rev) / double(total_rev) : 0.0;

        std::cout << "  " << idx_ship->backend_name()
                  << ":  PhaseA(shipdate_OR)=" << tA
                  << "  PhaseB(part_scan)=" << tB
                  << "  PhaseC(BMFetch+agg)=" << tC
                  << "  Total=" << tot
                  << "  rows=" << ids.size()
                  << "  promo_pct=" << std::fixed << std::setprecision(6) << promo_pct
                  << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot);
        }
        last_promo_revenue = promo_pct;
    }

    {
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' "
            "                          THEN l_extendedprice * (1 - l_discount) "
            "                          ELSE 0 END) "
            "       / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue "
            "FROM lineitem, part "
            "WHERE l_partkey = p_partkey "
            "  AND l_shipdate >= DATE '1995-09-01' "
            "  AND l_shipdate <  DATE '1995-09-01' + INTERVAL '1' MONTH");
        if (r && !r->HasError() && r->RowCount() == 1) {
            double gt = r->GetValue(0, 0).GetValue<double>();
            if (std::fabs(gt - last_promo_revenue) < 0.001)
                std::cout << "\n[OK] " << idx_ship->backend_name()
                          << " matches DuckDB SQL (promo_revenue = " << gt << ").\n";
            else
                std::cerr << "\n[FAIL] mismatch: ours=" << last_promo_revenue
                          << " gt=" << gt << "\n";
        }
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t), sT = stats(tot_t);
    int measured = std::max(0, Q14_ITERATIONS - Q14_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q14 RESULTS — " << idx_ship->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (shipdate range OR): " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (part scan PROMO)  : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (BMFetch + agg)    : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  TOTAL                     : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q14_results_" + sf + ".csv");
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
        std::string bn = idx_ship->backend_name();
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
        put("PhaseA_ship", sA);
        put("PhaseB_part", sB);
        put("PhaseC_agg",  sC);
        put("TOTAL", sT);
        std::cout << "  [CSV] q14_results_" << sf << ".csv\n";
    }

    });  // end call_once
}

}  // namespace duckdb
