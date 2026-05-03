// TPC-H Q1 — Pricing Summary Report
// Verbatim port of teacher's BitEngine BMTPCH_Q1 "normal path" branch.
//
// Logic (mirrors teacher 1:1):
//   Phase A — multi-OR over rabit_shipdate->Btvs[d] for d > right_days,
//             then NOT to flip into shipdate_filter = (l_shipdate <= cutoff).
//             [Equivalent to OR over d <= right_days; we use that form
//              directly via apply_or_range_to(0, right_days).]
//
//   Phase B — for each (linestatus_v, returnflag_v) combo (5 valid
//             groups: 'N'+'O', 'N'+'F', 'A'+'F', 'R'+'F', 'A'+'O'):
//                  group_btv = linestatus_bm[ls] AND returnflag_bm[rf]
//                              AND shipdate_filter
//             Extract row IDs from group_btv.
//
//   Phase C — BMFetch lineitem(quantity, extendedprice, discount, tax)
//             for each group's row IDs, accumulate into Q1Group:
//               sum_qty, sum_base_price, sum_disc_price, sum_charge,
//               sum_discount, count_order.
//
// Pre-loaded auxiliary structures (PRAGMA load_bitmap, NOT in latency):
//   - bitmap_shipdate    (per-day IndexedX over l_shipdate)
//   - bitmap_linestatus  (per-char IndexedX over l_linestatus, key=ASCII)
//   - bitmap_returnflag  (per-char IndexedX over l_returnflag, key=ASCII)
//
// All Phase A + B + C cost is the timed "Q1 latency".

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
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q1_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

using Q1BmType = bm_bench::Backend;
static const Q1BmType Q1_BM_NEW = bm_bench::parse_backend("Q1_BM");
static const char* q1_bm_label_new() { return bm_bench::backend_label(Q1_BM_NEW); }
static std::string q1_get_sf_label() { return bm_bench::sf_label(); }

static const int Q1_ITERATIONS = bm_bench::iter_count(5);
static const int Q1_WARMUP     = bm_bench::warmup_count(1);

static std::once_flag q1_once_flag_new;

// 1998-09-02 — TPC-H Q1 cutoff day (epoch days since 1970-01-01).
// (Same value as teacher's `right_days = 10471`.)
static constexpr int64_t Q1_SHIPDATE_CUTOFF = 10471;

// Q1 aggregation result per (returnflag, linestatus) group
struct Q1GroupAns {
    int64_t sum_qty        = 0;
    int64_t sum_base_price = 0;
    int64_t sum_disc_price = 0;
    int64_t sum_charge     = 0;
    int64_t sum_discount   = 0;
    int64_t count_order    = 0;
};

// -----------------------------------------------------------------------
// q1_get_rowids — extract sorted ascending row IDs from a backend's
// per-group filter bitmap into a `vector<row_t>` consumable by
// DataTable::BMFetch (mirror of Q5's get_rowids).
// -----------------------------------------------------------------------
static void q1_get_rowids(const ComBit& btv, std::vector<row_t>& out) {
    out.clear();
    btv.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out.push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}

static void q1_get_rowids(const roaring::Roaring& btv, std::vector<row_t>& out) {
    out.clear();
    out.reserve(btv.cardinality());
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out.push_back(static_cast<row_t>(*it));
}

static void q1_get_rowids(const ibis::bitvector& btv, std::vector<row_t>& out) {
    out.clear();
    ibis::bitvector::pit pit(btv);
    while (*pit != 0xFFFFFFFFU) {
        out.push_back(static_cast<row_t>(*pit));
        pit.next();
    }
}

static void q1_get_rowids(const ewah::EWAHBoolArray<uint64_t>& btv, std::vector<row_t>& out) {
    out.clear();
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out.push_back(static_cast<row_t>(*it));
}

static void q1_get_rowids(const ConciseSet<false>& btv, std::vector<row_t>& out) {
    out.clear();
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out.push_back(static_cast<row_t>(*it));
}

// -----------------------------------------------------------------------
// q1_aggregate_group — given a sorted ids list (= rows that pass the
// (linestatus, returnflag, shipdate<=cutoff) filter for one group),
// BMFetch lineitem (qty, extendedprice, discount, tax) in 2048-row
// chunks and accumulate into Q1GroupAns.  Mirrors teacher's per-row
// q1_data accumulator.
// -----------------------------------------------------------------------
static void q1_aggregate_group(
    ClientContext& ctx,
    TableCatalogEntry& lineitem_table,
    DuckTransaction& tx,
    const std::vector<row_t>& ids,
    Q1GroupAns& g)
{
    if (ids.empty()) return;
    vector<StorageIndex> col_ids = {
        StorageIndex(4),  // l_quantity
        StorageIndex(5),  // l_extendedprice
        StorageIndex(6),  // l_discount
        StorageIndex(7),  // l_tax
    };
    vector<LogicalType> types = {
        lineitem_table.GetColumns().GetColumnTypes()[4],
        lineitem_table.GetColumns().GetColumnTypes()[5],
        lineitem_table.GetColumns().GetColumnTypes()[6],
        lineitem_table.GetColumns().GetColumnTypes()[7],
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

        auto qty   = FlatVector::GetData<int64_t>(result.data[0]);
        auto price = FlatVector::GetData<int64_t>(result.data[1]);
        auto disc  = FlatVector::GetData<int64_t>(result.data[2]);
        auto tax   = FlatVector::GetData<int64_t>(result.data[3]);
        for (idx_t i = 0; i < result.size(); i++) {
            int64_t dp = price[i] * (100 - disc[i]);
            g.sum_qty        += qty[i];
            g.sum_base_price += price[i];
            g.sum_disc_price += dp;
            g.sum_charge     += dp * (100 + tax[i]);
            g.sum_discount   += disc[i];
            g.count_order    += 1;
        }
    }
}

// =========================================================================
// BMTPCH_Q1 — main entry
// =========================================================================
void BMTableScan::BMTPCH_Q1(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q1_once_flag_new, [&]() {

    bm_bench::warn_if_sf1();

    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q1 (BitEngine pattern, BMFetch) — " << q1_bm_label_new()
              << " (" << q1_get_sf_label() << ")" << std::endl;
    std::cout << "  Pre-loaded auxiliary structures: bitmap_shipdate + bitmap_linestatus + bitmap_returnflag" << std::endl;
    std::cout << "================================================================" << std::endl;

    // ----- Pull pre-built indexed bitmaps from client_context -----
    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    auto* idx_ls   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_linestatus);
    auto* idx_rf   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_returnflag);
    if (!idx_ship || !idx_ls || !idx_rf) {
        std::cerr << "[Q1] ERROR: required bitmaps not loaded.\n"
                     "          Run PRAGMA load_bitmap('shipdate'); load_bitmap('linestatus'); load_bitmap('returnflag'); first.\n";
        return;
    }
    if (std::string(idx_ship->backend_name()) != idx_ls->backend_name() ||
        std::string(idx_ship->backend_name()) != idx_rf->backend_name()) {
        std::cerr << "[Q1] ERROR: backend mismatch among shipdate/linestatus/returnflag.\n";
        return;
    }
    size_t num_rows = idx_ship->num_rows();

    // -----------------------------------------------------------------------
    // 5 valid TPC-H Q1 groups (per spec): the only (rf, ls) pairs that
    // actually appear in lineitem are ('N','O'), ('N','F'), ('A','F'),
    // ('R','F'), ('A','O').  We still iterate cartesian and skip empties.
    // -----------------------------------------------------------------------
    static const char RF_CHARS[] = {'A', 'N', 'R'};
    static const char LS_CHARS[] = {'F', 'O'};
    static const int  RF_COUNT   = 3;
    static const int  LS_COUNT   = 2;

    std::vector<double> tA_t, tB_t, tC_t, tot_t;
    std::map<std::pair<char,char>, Q1GroupAns> last_ans;

    for (int iter = 0; iter < Q1_ITERATIONS; iter++) {
        bool warm = iter < Q1_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q1_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        std::map<std::pair<char,char>, Q1GroupAns> ans;
        std::map<std::pair<char,char>, std::vector<row_t>> ids_per_group;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        // ===== Phase A: shipdate range OR (l_shipdate <= cutoff) =====
        // Phase B inlined per backend after.

        // ---- ComBit ----
        if (auto* cb_ship = dynamic_cast<bm_index::IndexedComBit*>(idx_ship)) {
            auto* cb_ls_x = dynamic_cast<bm_index::IndexedComBit*>(idx_ls);
            auto* cb_rf_x = dynamic_cast<bm_index::IndexedComBit*>(idx_rf);
            if (!cb_ls_x || !cb_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            // empty_bits no longer used here (replaced by from_sparse_positions).
            auto t_a0 = clk::now();
            ComBit shipdate_filter = ComBit::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
            cb_ship->apply_or_range_to(shipdate_filter, INT64_MIN, Q1_SHIPDATE_CUTOFF);
            auto t_a1 = clk::now();

            // Pre-cache linestatus / returnflag base bitmaps.
            std::map<char, ComBit> ls_btv, rf_btv;
            for (int li = 0; li < LS_COUNT; li++) {
                ComBit b = ComBit::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                cb_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                ls_btv[LS_CHARS[li]] = std::move(b);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ComBit b = ComBit::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                cb_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
                rf_btv[RF_CHARS[ri]] = std::move(b);
            }
            auto t_b0 = clk::now();

            // For each (rf, ls) combo, AND linestatus & returnflag & shipdate_filter.
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    char rf = RF_CHARS[ri], ls = LS_CHARS[li];
                    ComBit grp = rf_btv[rf];
                    grp &= ls_btv[ls];
                    grp &= shipdate_filter;
                    auto& ids = ids_per_group[{rf, ls}];
                    q1_get_rowids(grp, ids);
                }
            }
            auto t_b1 = clk::now();

            // Phase C: per-group BMFetch + aggregate.
            for (auto& [k, ids] : ids_per_group) {
                if (ids.empty()) continue;
                q1_aggregate_group(context.client, lineitem_table, lineitem_tx, ids, ans[k]);
            }
            auto t_c1 = clk::now();
            double tA  = q1_ms(t_a0, t_b0);
            double tB  = q1_ms(t_b0, t_b1);
            double tC  = q1_ms(t_b1, t_c1);
            double tot = q1_ms(t0, t_c1);
            std::cout << "  ComBit:  PhaseA=" << tA << "  PhaseB=" << tB
                      << "  PhaseC=" << tC << "  Total=" << tot << std::endl;
            if (!warm) { tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot); }
        }
        // ---- CRoaring (pairwise) / CRoaringRun (fastunion) ----
        else if (auto* cr_ship = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ship)) {
            auto* cr_ls_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ls);
            auto* cr_rf_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_rf);
            if (!cr_ls_x || !cr_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }
            const bool use_fastunion = cr_ship->run_optimized();

            auto t_a0 = clk::now();
            roaring::Roaring shipdate_filter;
            if (use_fastunion) {
                shipdate_filter = cr_ship->or_range(INT64_MIN, Q1_SHIPDATE_CUTOFF);
            } else {
                cr_ship->apply_or_range_to(shipdate_filter, INT64_MIN, Q1_SHIPDATE_CUTOFF);
            }
            auto t_a1 = clk::now();

            std::map<char, roaring::Roaring> ls_btv, rf_btv;
            for (int li = 0; li < LS_COUNT; li++) {
                roaring::Roaring b;
                cr_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                ls_btv[LS_CHARS[li]] = std::move(b);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                roaring::Roaring b;
                cr_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
                rf_btv[RF_CHARS[ri]] = std::move(b);
            }
            auto t_b0 = clk::now();

            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    char rf = RF_CHARS[ri], ls = LS_CHARS[li];
                    roaring::Roaring grp = rf_btv[rf];
                    grp &= ls_btv[ls];
                    grp &= shipdate_filter;
                    auto& ids = ids_per_group[{rf, ls}];
                    q1_get_rowids(grp, ids);
                }
            }
            auto t_b1 = clk::now();

            for (auto& [k, ids] : ids_per_group) {
                if (ids.empty()) continue;
                q1_aggregate_group(context.client, lineitem_table, lineitem_tx, ids, ans[k]);
            }
            auto t_c1 = clk::now();
            double tA  = q1_ms(t_a0, t_b0);
            double tB  = q1_ms(t_b0, t_b1);
            double tC  = q1_ms(t_b1, t_c1);
            double tot = q1_ms(t0, t_c1);
            std::cout << "  " << idx_ship->backend_name()
                      << ":  PhaseA=" << tA << "  PhaseB=" << tB
                      << "  PhaseC=" << tC << "  Total=" << tot << std::endl;
            if (!warm) { tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot); }
        }
        // ---- WAH ----
        else if (auto* wah_ship = dynamic_cast<bm_index::IndexedWAH*>(idx_ship)) {
            auto* wah_ls_x = dynamic_cast<bm_index::IndexedWAH*>(idx_ls);
            auto* wah_rf_x = dynamic_cast<bm_index::IndexedWAH*>(idx_rf);
            if (!wah_ls_x || !wah_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ibis::bitvector shipdate_filter;
            wah_ship->apply_or_range_to(shipdate_filter, INT64_MIN, Q1_SHIPDATE_CUTOFF);
            auto t_a1 = clk::now();

            std::map<char, ibis::bitvector> ls_btv, rf_btv;
            for (int li = 0; li < LS_COUNT; li++) {
                ibis::bitvector b;
                wah_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                ls_btv[LS_CHARS[li]] = std::move(b);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ibis::bitvector b;
                wah_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
                rf_btv[RF_CHARS[ri]] = std::move(b);
            }
            auto t_b0 = clk::now();

            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    char rf = RF_CHARS[ri], ls = LS_CHARS[li];
                    ibis::bitvector grp; grp.copy(rf_btv[rf]);
                    grp &= ls_btv[ls];
                    grp &= shipdate_filter;
                    auto& ids = ids_per_group[{rf, ls}];
                    q1_get_rowids(grp, ids);
                }
            }
            auto t_b1 = clk::now();

            for (auto& [k, ids] : ids_per_group) {
                if (ids.empty()) continue;
                q1_aggregate_group(context.client, lineitem_table, lineitem_tx, ids, ans[k]);
            }
            auto t_c1 = clk::now();
            double tA  = q1_ms(t_a0, t_b0);
            double tB  = q1_ms(t_b0, t_b1);
            double tC  = q1_ms(t_b1, t_c1);
            double tot = q1_ms(t0, t_c1);
            std::cout << "  WAH:  PhaseA=" << tA << "  PhaseB=" << tB
                      << "  PhaseC=" << tC << "  Total=" << tot << std::endl;
            if (!warm) { tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot); }
        }
        // ---- EWAH (always fast_logicalor) ----
        else if (auto* ew_ship = dynamic_cast<bm_index::IndexedEWAH*>(idx_ship)) {
            auto* ew_ls_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_ls);
            auto* ew_rf_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_rf);
            if (!ew_ls_x || !ew_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ewah::EWAHBoolArray<uint64_t> shipdate_filter =
                ew_ship->or_range(INT64_MIN, Q1_SHIPDATE_CUTOFF);
            auto t_a1 = clk::now();

            std::map<char, ewah::EWAHBoolArray<uint64_t>> ls_btv, rf_btv;
            for (int li = 0; li < LS_COUNT; li++) {
                ewah::EWAHBoolArray<uint64_t> b;
                ew_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                ls_btv[LS_CHARS[li]] = std::move(b);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ewah::EWAHBoolArray<uint64_t> b;
                ew_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
                rf_btv[RF_CHARS[ri]] = std::move(b);
            }
            auto t_b0 = clk::now();

            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    char rf = RF_CHARS[ri], ls = LS_CHARS[li];
                    ewah::EWAHBoolArray<uint64_t> tmp1, grp;
                    rf_btv[rf].logicaland(ls_btv[ls], tmp1);
                    tmp1.logicaland(shipdate_filter, grp);
                    auto& ids = ids_per_group[{rf, ls}];
                    q1_get_rowids(grp, ids);
                }
            }
            auto t_b1 = clk::now();

            for (auto& [k, ids] : ids_per_group) {
                if (ids.empty()) continue;
                q1_aggregate_group(context.client, lineitem_table, lineitem_tx, ids, ans[k]);
            }
            auto t_c1 = clk::now();
            double tA  = q1_ms(t_a0, t_b0);
            double tB  = q1_ms(t_b0, t_b1);
            double tC  = q1_ms(t_b1, t_c1);
            double tot = q1_ms(t0, t_c1);
            std::cout << "  EWAH:  PhaseA=" << tA << "  PhaseB=" << tB
                      << "  PhaseC=" << tC << "  Total=" << tot << std::endl;
            if (!warm) { tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot); }
        }
        // ---- Concise (always fast_logicalor) ----
        else if (auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_ship)) {
            auto* con_ls_x = dynamic_cast<bm_index::IndexedConcise*>(idx_ls);
            auto* con_rf_x = dynamic_cast<bm_index::IndexedConcise*>(idx_rf);
            if (!con_ls_x || !con_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ConciseSet<false> shipdate_filter =
                con_ship->or_range(INT64_MIN, Q1_SHIPDATE_CUTOFF);
            auto t_a1 = clk::now();

            std::map<char, ConciseSet<false>> ls_btv, rf_btv;
            for (int li = 0; li < LS_COUNT; li++) {
                ConciseSet<false> b;
                con_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                ls_btv[LS_CHARS[li]] = std::move(b);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ConciseSet<false> b;
                con_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
                rf_btv[RF_CHARS[ri]] = std::move(b);
            }
            auto t_b0 = clk::now();

            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    char rf = RF_CHARS[ri], ls = LS_CHARS[li];
                    ConciseSet<false> grp = rf_btv[rf].logicaland(ls_btv[ls]).logicaland(shipdate_filter);
                    auto& ids = ids_per_group[{rf, ls}];
                    q1_get_rowids(grp, ids);
                }
            }
            auto t_b1 = clk::now();

            for (auto& [k, ids] : ids_per_group) {
                if (ids.empty()) continue;
                q1_aggregate_group(context.client, lineitem_table, lineitem_tx, ids, ans[k]);
            }
            auto t_c1 = clk::now();
            double tA  = q1_ms(t_a0, t_b0);
            double tB  = q1_ms(t_b0, t_b1);
            double tC  = q1_ms(t_b1, t_c1);
            double tot = q1_ms(t0, t_c1);
            std::cout << "  Concise:  PhaseA=" << tA << "  PhaseB=" << tB
                      << "  PhaseC=" << tC << "  Total=" << tot << std::endl;
            if (!warm) { tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot); }
        }
        else {
            std::cerr << "[Q1] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }

        last_ans = ans;
    }

    // ----- Validate vs DuckDB SQL -----
    bool gt_ok = false;
    std::map<std::pair<char,char>, std::vector<double>> gt;
    {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT l_returnflag, l_linestatus, "
            "       sum(l_quantity), sum(l_extendedprice), "
            "       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
            "       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
            "       sum(l_discount), count(*) "
            "FROM   lineitem "
            "WHERE  l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY "
            "GROUP BY l_returnflag, l_linestatus "
            "ORDER BY l_returnflag, l_linestatus";
        auto r = con.Query(sql);
        if (r && !r->HasError()) {
            for (idx_t i = 0; i < r->RowCount(); i++) {
                char rf = r->GetValue(0, i).GetValue<std::string>()[0];
                char ls = r->GetValue(1, i).GetValue<std::string>()[0];
                std::vector<double> vals(6);
                vals[0] = r->GetValue(2, i).GetValue<double>();
                vals[1] = r->GetValue(3, i).GetValue<double>();
                vals[2] = r->GetValue(4, i).GetValue<double>();
                vals[3] = r->GetValue(5, i).GetValue<double>();
                vals[4] = r->GetValue(6, i).GetValue<double>();
                vals[5] = double(r->GetValue(7, i).GetValue<int64_t>());
                gt[{rf, ls}] = vals;
            }
            gt_ok = true;
        }
    }
    if (gt_ok) {
        bool ok = (last_ans.size() == gt.size());
        if (ok) {
            for (auto& [k, g] : last_ans) {
                auto it = gt.find(k);
                if (it == gt.end()) { ok = false; break; }
                // qty stored as int64 raw; SQL returns DECIMAL → double.
                // Our raw values are 100x the real (cents).
                double our_qty   = double(g.sum_qty)        / 100;
                double our_price = double(g.sum_base_price) / 100;
                double our_disc_price = double(g.sum_disc_price) / 10000;
                double our_charge     = double(g.sum_charge)     / 1000000;
                double our_discount   = double(g.sum_discount)   / 100;
                double our_count      = double(g.count_order);
                if (std::fabs(our_qty   - it->second[0]) > 1.0 ||
                    std::fabs(our_price - it->second[1]) > 1.0 ||
                    std::fabs(our_disc_price - it->second[2]) > 1.0 ||
                    std::fabs(our_charge     - it->second[3]) > 1.0 ||
                    std::fabs(our_discount   - it->second[4]) > 1.0 ||
                    std::fabs(our_count      - it->second[5]) > 0.5) {
                    ok = false;
                    std::cerr << "[FAIL] (" << k.first << "," << k.second << "): "
                              << "ours=(" << our_qty << "," << our_price << "," << our_disc_price
                              << "," << our_charge << "," << our_discount << "," << our_count << ")  "
                              << "gt=(" << it->second[0] << "," << it->second[1] << "," << it->second[2]
                              << "," << it->second[3] << "," << it->second[4] << "," << it->second[5] << ")\n";
                }
            }
        }
        if (ok)
            std::cout << "\n[OK] " << idx_ship->backend_name()
                      << " matches DuckDB SQL ground truth ("
                      << gt.size() << " (rf,ls) groups).\n";
        else
            std::cerr << "\n[FAIL] mismatch vs SQL ground truth!\n";
    }

    // ----- Print final results -----
    std::cout << "\n  Q1 group results (rf,ls): qty / base_price / disc_price / charge / discount / count\n";
    for (auto& [k, g] : last_ans) {
        std::cout << "  (" << k.first << "," << k.second << "): "
                  << std::fixed << std::setprecision(2)
                  << double(g.sum_qty) / 100        << "  "
                  << double(g.sum_base_price) / 100 << "  "
                  << double(g.sum_disc_price) / 10000 << "  "
                  << double(g.sum_charge) / 1000000 << "  "
                  << double(g.sum_discount) / 100   << "  "
                  << g.count_order << std::endl;
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t), sT = stats(tot_t);
    int measured = std::max(0, Q1_ITERATIONS - Q1_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q1 RESULTS — " << idx_ship->backend_name()
              << " only (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (shipdate range OR): " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (per-group AND)    : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (BMFetch+Agg)      : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  TOTAL                     : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    // CSV row (Schema-A: same columns as Q5, only active backend has data)
    std::string sf = q1_get_sf_label();
    std::ofstream csv("q1_results_" + sf + ".csv");
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
            cell(z); cell(z);  // BS, BSA — N/A for Q1 new path
            cell(bn == "Concise" ? s : z);
            csv << "0,0,0,0,0,0,0\n";
        };
        put("PhaseA", sA);
        put("PhaseB", sB);
        put("PhaseC", sC);
        put("TOTAL",  sT);
        std::cout << "  [CSV] q1_results_" << sf << ".csv\n";
    }

    });  // end call_once
}

} // namespace duckdb
