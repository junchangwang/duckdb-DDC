// TPC-H Q6 — Forecasting Revenue Change Query
// Verbatim port of teacher's BitEngine BMTPCH_Q6 + TPCH_Q6_Lineitem_GetRowIds
// pattern.  Uses Group Encoding for shipdate (one bitmap per year) +
// per-value indexes for discount and quantity (range OR).
//
// Logic (mirrors teacher 1:1):
//   Phase A — bitmap_shipdate_GE.apply_or_to(dst, 1994)            (1 OR)
//   Phase B — bitmap_discount.apply_or_range_to(dst, 5, 7)        (3 ORs)
//   Phase C — bitmap_quantity.apply_or_range_to(dst, 0, 2399)     (range OR)
//   Phase D — AND all three → row_ids
//   Phase E — DuckDB consumes row_ids via BMFetch(l_discount, l_extendedprice)
//             and aggregates SUM(l_extendedprice * l_discount).
//
// Pre-loaded auxiliary structures (PRAGMA load_bitmap, NOT in latency):
//   - bitmap_shipdate_GE  (per-year  IndexedComBitGE / IndexedX)
//   - bitmap_discount     (per-value IndexedX over l_discount raw int64)
//   - bitmap_quantity     (per-value IndexedX over l_quantity raw int64)

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

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace duckdb {

// TPC-H Q6 spec parameters (translated for our raw int64 storage scaling):
// l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'  → year == 1994
// l_discount BETWEEN 0.05 AND 0.07                          → raw [5, 7]
// l_quantity <  24                                          → raw < 2400
static constexpr int64_t Q6_SHIPDATE_YEAR     = 1994;
static constexpr int64_t Q6_DISCOUNT_LO       = 5;
static constexpr int64_t Q6_DISCOUNT_HI       = 7;
static constexpr int64_t Q6_QUANTITY_LO       = 0;
static constexpr int64_t Q6_QUANTITY_HI_EXCL  = 2400;  // raw int64 < 2400 means value < 24.00

static std::once_flag q6_once_flag_new;
static std::mutex     q6_build_mutex;
static std::atomic<bool> q6_built{false};

// -----------------------------------------------------------------------
// q6_emit_csv — write Schema-A CSV for the just-finished backend run.
// Per-iteration timings are unavailable (Q6 is a pull-pipeline source,
// not a controlled iteration loop), so stddev=0 / min=median / max=median.
// bench_suite.py merges per-backend CSV captures into one combined file.
// -----------------------------------------------------------------------
struct Q6Phase { std::string op; double median; };

static void q6_emit_csv(const char* backend_name,
                        const std::vector<Q6Phase>& phases)
{
    std::string sf = bm_bench::sf_label();
    std::string path = "q6_results_" + sf + ".csv";
    std::ofstream csv(path);
    if (!csv) return;
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
    auto cell_z = [&]() { csv << "0,0,0,0,"; };
    auto cell_m = [&](double m) { csv << m << ",0," << m << "," << m << ","; };
    auto cell_pick = [&](const std::string& bn, const std::string& match, double m) {
        if (bn == match) cell_m(m); else cell_z();
    };
    std::string bn = backend_name;
    for (auto& ph : phases) {
        csv << sf << "," << ph.op << ",";
        cell_pick(bn, "WAH",         ph.median);
        cell_pick(bn, "ComBit",      ph.median);
        cell_pick(bn, "CRoaring",    ph.median);
        cell_pick(bn, "CRoaringRun", ph.median);
        cell_pick(bn, "EWAH",        ph.median);
        cell_z();   // BS  — N/A
        cell_z();   // BSA — N/A
        cell_pick(bn, "Concise",     ph.median);
        csv << "0,0,0,0,0,0,0\n";
    }
}

// -----------------------------------------------------------------------
// q6_get_rowids — extract sorted ascending row IDs from a backend's
// final filter bitmap into a `vector<row_t>` consumable by DataTable
// ::BMFetch (mirror of Q5's get_rowids).
// -----------------------------------------------------------------------
static void q6_get_rowids(const ComBit& btv, std::vector<row_t>* out) {
    out->clear();
    btv.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}

static void q6_get_rowids(const roaring::Roaring& btv, std::vector<row_t>* out) {
    out->clear();
    out->reserve(btv.cardinality());
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out->push_back(static_cast<row_t>(*it));
}

static void q6_get_rowids(const ibis::bitvector& btv, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(btv);
    while (*pit != 0xFFFFFFFFU) {
        out->push_back(static_cast<row_t>(*pit));
        pit.next();
    }
}

static void q6_get_rowids(const ewah::EWAHBoolArray<uint64_t>& btv, std::vector<row_t>* out) {
    out->clear();
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out->push_back(static_cast<row_t>(*it));
}

static void q6_get_rowids(const ConciseSet<false>& btv, std::vector<row_t>* out) {
    out->clear();
    for (auto it = btv.begin(); it != btv.end(); ++it)
        out->push_back(static_cast<row_t>(*it));
}

// =========================================================================
// TPCH_Q6_Lineitem_GetRowIds — produce row IDs into `row_ids` for DuckDB's
// downstream BMFetch.  Mirrors teacher's homonymous function: dynamic_cast
// the per-backend index, multi-OR the predicate columns, AND, get row IDs.
// =========================================================================
void BMTableScan::TPCH_Q6_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids)
{
    // Caller (BMTPCH_Q6) already holds q6_build_mutex, so this body
    // runs single-threaded; q6_built guards against double-build when
    // BMTPCH_Q6 is called multiple times in a row (e.g. successive
    // queries reusing the same client context).
    if (q6_built.load()) return;

    std::call_once(q6_once_flag_new, [&]() {
        bm_bench::warn_if_sf1();

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  TPC-H Q6 (BitEngine pattern, GE shipdate + range disc/qty)" << std::endl;
        std::cout << "  Pre-loaded: bitmap_shipdate_GE + bitmap_discount + bitmap_quantity" << std::endl;
        std::cout << "================================================================" << std::endl;
    });

    auto* idx_ship_ge = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_GE);
    auto* idx_disc    = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_discount);
    auto* idx_qty     = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_quantity);
    if (!idx_ship_ge || !idx_disc || !idx_qty) {
        std::cerr << "[Q6] ERROR: required bitmaps not loaded.\n"
                     "          Run PRAGMA load_bitmap('shipdate_GE'); load_bitmap('discount'); load_bitmap('quantity'); first.\n";
        return;
    }

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    size_t num_rows = idx_disc->num_rows();

    // ---- ComBit (teacher's Q6 logic: GE shipdate single-key OR + range
    // OR for discount/quantity).  GE step = single apply_or_to(year).
    // Range OR uses sca or_many (ComBit's design contribution).  No BPE.
    if (auto* cb_ship = dynamic_cast<bm_index::IndexedComBitGE*>(idx_ship_ge)) {
        auto* cb_disc_x = dynamic_cast<bm_index::IndexedComBit*>(idx_disc);
        auto* cb_qty_x  = dynamic_cast<bm_index::IndexedComBit*>(idx_qty);
        if (!cb_disc_x || !cb_qty_x) { std::cerr << "[Q6] ComBit type mismatch.\n"; return; }

        ComBit btv_ship = ComBit::from_sparse_positions({}, num_rows, cb_ship->segment_bits());

        auto t_a = clk::now();
        cb_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        // Gather in-range disc keys + sca or_many.
        std::vector<int64_t> disc_keys;
        cb_disc_x->for_each_key([&](int64_t k) {
            if (k >= Q6_DISCOUNT_LO && k <= Q6_DISCOUNT_HI) disc_keys.push_back(k);
        });
        ComBit btv_disc = cb_disc_x->or_many(disc_keys);
        auto t_c = clk::now();
        std::vector<int64_t> qty_keys;
        cb_qty_x->for_each_key([&](int64_t k) {
            if (k >= Q6_QUANTITY_LO && k <= Q6_QUANTITY_HI_EXCL - 1) qty_keys.push_back(k);
        });
        ComBit btv_qty = cb_qty_x->or_many(qty_keys);
        auto t_d = clk::now();
        // Reordered AND: (disc & qty) then & ship.  ship contains a
        // SparseComBit-OR'd result that triggers an AVX-512 memcpy
        // crash when used as LHS of `&=` against a SparseComBit-OR'd
        // RHS; reordering materialises the dense (disc & qty) first
        // (segments transition to Decompressed via the binary path),
        // then `& ship` runs the in-place Decompressed `&=` which is
        // crash-free.  Pre-reorder commit: 1336074392.
        btv_disc &= btv_qty;
        btv_disc &= btv_ship;
        // Final result lives in btv_disc — move into btv_ship for
        // the downstream get_rowids call.
        btv_ship = std::move(btv_disc);
        auto t_e = clk::now();
        q6_get_rowids(btv_ship, row_ids);
        auto t_f = clk::now();

        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        std::cout << "  [Q6 ComBit] ship_ge=" << ms(t_a, t_b)
                  << "  disc_or=" << ms(t_b, t_c)
                  << "  qty_or="  << ms(t_c, t_d)
                  << "  AND="     << ms(t_d, t_e)
                  << "  rowids="  << ms(t_e, t_f)
                  << "  total="   << ms(t0, t_f)
                  << "  rows="    << row_ids->size() << std::endl;
        q6_emit_csv("ComBit", {
            {"ship_GE",     ms(t_a, t_b)},
            {"OR_discount", ms(t_b, t_c)},
            {"OR_quantity", ms(t_c, t_d)},
            {"AND",         ms(t_d, t_e)},
            {"GetRowIds",   ms(t_e, t_f)},
            {"TOTAL",       ms(t0, t_f)},
        });
        q6_built.store(true);
        return;
    }
    // ---- CR (pairwise per-value |=) / CRR (fastunion via or_range) ----
    if (auto* cr_ship = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ship_ge)) {
        auto* cr_disc_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_disc);
        auto* cr_qty_x  = dynamic_cast<bm_index::IndexedCRoaring*>(idx_qty);
        if (!cr_disc_x || !cr_qty_x) { std::cerr << "[Q6] CR type mismatch.\n"; return; }
        const bool ro = cr_ship->run_optimized();

        roaring::Roaring btv_ship, btv_disc, btv_qty;
        auto t_a = clk::now();
        cr_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        if (ro) btv_disc = cr_disc_x->or_range(Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        else    cr_disc_x->apply_or_range_to(btv_disc, Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        if (ro) btv_qty = cr_qty_x->or_range(Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
        else    cr_qty_x->apply_or_range_to(btv_qty, Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
        auto t_d = clk::now();
        btv_ship &= btv_disc;
        btv_ship &= btv_qty;
        auto t_e = clk::now();
        q6_get_rowids(btv_ship, row_ids);
        auto t_f = clk::now();
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        std::cout << "  [Q6 " << idx_disc->backend_name() << "] ship_ge=" << ms(t_a, t_b)
                  << "  disc_or=" << ms(t_b, t_c)
                  << "  qty_or="  << ms(t_c, t_d)
                  << "  AND="     << ms(t_d, t_e)
                  << "  rowids="  << ms(t_e, t_f)
                  << "  total="   << ms(t0, t_f)
                  << "  rows="    << row_ids->size() << std::endl;
        q6_emit_csv(idx_disc->backend_name(), {
            {"ship_GE",     ms(t_a, t_b)},
            {"OR_discount", ms(t_b, t_c)},
            {"OR_quantity", ms(t_c, t_d)},
            {"AND",         ms(t_d, t_e)},
            {"GetRowIds",   ms(t_e, t_f)},
            {"TOTAL",       ms(t0, t_f)},
        });
        q6_built.store(true);
        return;
    }
    // ---- WAH ----
    if (auto* wah_ship = dynamic_cast<bm_index::IndexedWAH*>(idx_ship_ge)) {
        auto* wah_disc_x = dynamic_cast<bm_index::IndexedWAH*>(idx_disc);
        auto* wah_qty_x  = dynamic_cast<bm_index::IndexedWAH*>(idx_qty);
        if (!wah_disc_x || !wah_qty_x) { std::cerr << "[Q6] WAH type mismatch.\n"; return; }

        ibis::bitvector btv_ship, btv_disc, btv_qty;
        auto t_a = clk::now();
        wah_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        wah_disc_x->apply_or_range_to(btv_disc, Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        wah_qty_x->apply_or_range_to(btv_qty, Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
        auto t_d = clk::now();
        btv_ship &= btv_disc;
        btv_ship &= btv_qty;
        auto t_e = clk::now();
        q6_get_rowids(btv_ship, row_ids);
        auto t_f = clk::now();
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        std::cout << "  [Q6 WAH] ship_ge=" << ms(t_a, t_b)
                  << "  disc_or=" << ms(t_b, t_c)
                  << "  qty_or="  << ms(t_c, t_d)
                  << "  AND="     << ms(t_d, t_e)
                  << "  rowids="  << ms(t_e, t_f)
                  << "  total="   << ms(t0, t_f)
                  << "  rows="    << row_ids->size() << std::endl;
        q6_emit_csv("WAH", {
            {"ship_GE",     ms(t_a, t_b)},
            {"OR_discount", ms(t_b, t_c)},
            {"OR_quantity", ms(t_c, t_d)},
            {"AND",         ms(t_d, t_e)},
            {"GetRowIds",   ms(t_e, t_f)},
            {"TOTAL",       ms(t0, t_f)},
        });
        q6_built.store(true);
        return;
    }
    // ---- EWAH (always fast_logicalor) ----
    if (auto* ew_ship = dynamic_cast<bm_index::IndexedEWAH*>(idx_ship_ge)) {
        auto* ew_disc_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_disc);
        auto* ew_qty_x  = dynamic_cast<bm_index::IndexedEWAH*>(idx_qty);
        if (!ew_disc_x || !ew_qty_x) { std::cerr << "[Q6] EWAH type mismatch.\n"; return; }

        ewah::EWAHBoolArray<uint64_t> btv_ship, btv_disc, btv_qty;
        auto t_a = clk::now();
        ew_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        btv_disc = ew_disc_x->or_range(Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        btv_qty  = ew_qty_x->or_range(Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
        auto t_d = clk::now();
        ewah::EWAHBoolArray<uint64_t> tmp1, btv_res;
        btv_ship.logicaland(btv_disc, tmp1);
        tmp1.logicaland(btv_qty, btv_res);
        auto t_e = clk::now();
        q6_get_rowids(btv_res, row_ids);
        auto t_f = clk::now();
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        std::cout << "  [Q6 EWAH] ship_ge=" << ms(t_a, t_b)
                  << "  disc_or=" << ms(t_b, t_c)
                  << "  qty_or="  << ms(t_c, t_d)
                  << "  AND="     << ms(t_d, t_e)
                  << "  rowids="  << ms(t_e, t_f)
                  << "  total="   << ms(t0, t_f)
                  << "  rows="    << row_ids->size() << std::endl;
        q6_emit_csv("EWAH", {
            {"ship_GE",     ms(t_a, t_b)},
            {"OR_discount", ms(t_b, t_c)},
            {"OR_quantity", ms(t_c, t_d)},
            {"AND",         ms(t_d, t_e)},
            {"GetRowIds",   ms(t_e, t_f)},
            {"TOTAL",       ms(t0, t_f)},
        });
        q6_built.store(true);
        return;
    }
    // ---- Concise (always fast_logicalor) ----
    if (auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_ship_ge)) {
        auto* con_disc_x = dynamic_cast<bm_index::IndexedConcise*>(idx_disc);
        auto* con_qty_x  = dynamic_cast<bm_index::IndexedConcise*>(idx_qty);
        if (!con_disc_x || !con_qty_x) { std::cerr << "[Q6] Concise type mismatch.\n"; return; }

        ConciseSet<false> btv_ship, btv_disc, btv_qty;
        auto t_a = clk::now();
        con_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        btv_disc = con_disc_x->or_range(Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        btv_qty  = con_qty_x->or_range(Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
        auto t_d = clk::now();
        ConciseSet<false> btv_res = btv_ship.logicaland(btv_disc).logicaland(btv_qty);
        auto t_e = clk::now();
        q6_get_rowids(btv_res, row_ids);
        auto t_f = clk::now();
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
        };
        std::cout << "  [Q6 Concise] ship_ge=" << ms(t_a, t_b)
                  << "  disc_or=" << ms(t_b, t_c)
                  << "  qty_or="  << ms(t_c, t_d)
                  << "  AND="     << ms(t_d, t_e)
                  << "  rowids="  << ms(t_e, t_f)
                  << "  total="   << ms(t0, t_f)
                  << "  rows="    << row_ids->size() << std::endl;
        q6_emit_csv("Concise", {
            {"ship_GE",     ms(t_a, t_b)},
            {"OR_discount", ms(t_b, t_c)},
            {"OR_quantity", ms(t_c, t_d)},
            {"AND",         ms(t_d, t_e)},
            {"GetRowIds",   ms(t_e, t_f)},
            {"TOTAL",       ms(t0, t_f)},
        });
        q6_built.store(true);
        return;
    }

    std::cerr << "[Q6] ERROR: unrecognised IBitmapIndex backend.\n";
}

// =========================================================================
// BMTPCH_Q6 — DuckDB pull-style entry point.  On the first call, builds
// the row_ids list via TPCH_Q6_Lineitem_GetRowIds; on subsequent calls,
// streams 2048-row BMFetch chunks of (l_discount, l_extendedprice) until
// exhausted.  DuckDB's normal pipeline applies the SUM(price * discount)
// aggregate downstream.
// =========================================================================
SourceResultType BMTableScan::BMTPCH_Q6(ExecutionContext &context, DataChunk &chunk,
                                        const TableScanBindData &bind_data)
{
    // Single global lock for the whole call: BMFetch's RowGroupCollection
    // path uses a global `col_states` vector that is NOT thread-safe (see
    // src/storage/table/row_group_collection.cpp BMFetchColState).  Lock
    // also serialises the build-once and the cursor advancement.  We
    // never reset q6_built within a process — DuckDB's parallel scan
    // would otherwise see multiple workers re-enter BMTPCH_Q6 after one
    // worker reset state, double-build, and double-stream the row_ids
    // (manifests as the aggregate being summed twice).
    std::lock_guard<std::mutex> lock(q6_build_mutex);

    if (*cursor == 0 && !q6_built.load()) {
        TPCH_Q6_Lineitem_GetRowIds(context, row_ids);
        num_idlist = row_ids->size();
    }

    if (*cursor < row_ids->size()) {
        vector<StorageIndex> storage_column_ids;
        storage_column_ids.push_back(StorageIndex(6));  // l_discount
        storage_column_ids.push_back(StorageIndex(5));  // l_extendedprice

        TableScanState local_storage_state;
        local_storage_state.Initialize(storage_column_ids);
        ColumnFetchState column_fetch_state;

        auto &table_bind_data = bind_data;
        auto &transaction = DuckTransaction::Get(context.client, table_bind_data.table.catalog);

        data_ptr_t row_ids_data = (data_ptr_t)&((*row_ids)[*cursor]);
        Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
        idx_t fetch_count = 2048;
        if (*cursor + fetch_count > row_ids->size())
            fetch_count = row_ids->size() - *cursor;

        table_bind_data.table.GetStorage().BMFetch(transaction, chunk, storage_column_ids,
                                                   row_ids_vec, fetch_count,
                                                   column_fetch_state, num_idlist);
        *cursor += fetch_count;
        return SourceResultType::HAVE_MORE_OUTPUT;
    }
    // Stream exhausted.  Keep q6_built=true so any straggler thread
    // that races into BMTPCH_Q6 sees `*cursor >= row_ids->size()` and
    // returns FINISHED without rebuilding the row_ids vector.  We do
    // NOT reset query_source here either — that would route stragglers
    // through DuckDB's normal scan path and emit the entire lineitem
    // table as if no filter ran.
    return SourceResultType::FINISHED;
}

}  // namespace duckdb
