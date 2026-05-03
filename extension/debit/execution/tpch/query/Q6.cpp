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
#include <iostream>
#include <mutex>
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

    // ---- ComBit (with GE shipdate) ----
    if (auto* cb_ship = dynamic_cast<bm_index::IndexedComBitGE*>(idx_ship_ge)) {
        auto* cb_disc_x = dynamic_cast<bm_index::IndexedComBit*>(idx_disc);
        auto* cb_qty_x  = dynamic_cast<bm_index::IndexedComBit*>(idx_qty);
        if (!cb_disc_x || !cb_qty_x) { std::cerr << "[Q6] ComBit type mismatch.\n"; return; }

        std::vector<bool> empty_bits(num_rows, false);
        ComBit btv_ship = ComBit::compress(empty_bits, false, cb_ship->segment_bits());
        ComBit btv_disc = ComBit::compress(empty_bits, false, cb_disc_x->segment_bits());
        ComBit btv_qty  = ComBit::compress(empty_bits, false, cb_qty_x->segment_bits());

        auto t_a = clk::now();
        cb_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        cb_disc_x->apply_or_range_to(btv_disc, Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        cb_qty_x->apply_or_range_to(btv_qty, Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
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
        if (!row_ids->empty()) {
            row_t mx = (*row_ids)[0], mn = (*row_ids)[0];
            for (auto r : *row_ids) { if (r > mx) mx = r; if (r < mn) mn = r; }
            std::cout << "  [Q6 ComBit] row_id range: [" << mn << ", " << mx
                      << "] num_rows=" << num_rows << std::endl;
        }
        q6_built.store(true);
        return;
    }
    // ---- CRoaring / CRoaringRun ----
    if (auto* cr_ship = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ship_ge)) {
        auto* cr_disc_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_disc);
        auto* cr_qty_x  = dynamic_cast<bm_index::IndexedCRoaring*>(idx_qty);
        if (!cr_disc_x || !cr_qty_x) { std::cerr << "[Q6] CR type mismatch.\n"; return; }

        roaring::Roaring btv_ship, btv_disc, btv_qty;
        auto t_a = clk::now();
        cr_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        cr_disc_x->apply_or_range_to(btv_disc, Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        cr_qty_x->apply_or_range_to(btv_qty, Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
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
        q6_built.store(true);
        return;
    }
    // ---- EWAH ----
    if (auto* ew_ship = dynamic_cast<bm_index::IndexedEWAH*>(idx_ship_ge)) {
        auto* ew_disc_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_disc);
        auto* ew_qty_x  = dynamic_cast<bm_index::IndexedEWAH*>(idx_qty);
        if (!ew_disc_x || !ew_qty_x) { std::cerr << "[Q6] EWAH type mismatch.\n"; return; }

        ewah::EWAHBoolArray<uint64_t> btv_ship, btv_disc, btv_qty;
        auto t_a = clk::now();
        ew_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        ew_disc_x->apply_or_range_to(btv_disc, Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        ew_qty_x->apply_or_range_to(btv_qty, Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
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
        q6_built.store(true);
        return;
    }
    // ---- Concise ----
    if (auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_ship_ge)) {
        auto* con_disc_x = dynamic_cast<bm_index::IndexedConcise*>(idx_disc);
        auto* con_qty_x  = dynamic_cast<bm_index::IndexedConcise*>(idx_qty);
        if (!con_disc_x || !con_qty_x) { std::cerr << "[Q6] Concise type mismatch.\n"; return; }

        ConciseSet<false> btv_ship, btv_disc, btv_qty;
        auto t_a = clk::now();
        con_ship->apply_or_to(btv_ship, Q6_SHIPDATE_YEAR);
        auto t_b = clk::now();
        con_disc_x->apply_or_range_to(btv_disc, Q6_DISCOUNT_LO, Q6_DISCOUNT_HI);
        auto t_c = clk::now();
        con_qty_x->apply_or_range_to(btv_qty, Q6_QUANTITY_LO, Q6_QUANTITY_HI_EXCL - 1);
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
    // src/storage/table/row_group_collection.cpp BMFetchColState), so all
    // BMTPCH_Q6 calls must serialise.  Mirror of teacher's pattern.
    std::lock_guard<std::mutex> lock(q6_build_mutex);

    if (*cursor == 0) {
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
    } else {
        row_ids->clear();
        *cursor = 0;
        num_idlist = 0;
        q6_built.store(false);
        context.client.query_source = "tpch";
        return SourceResultType::FINISHED;
    }
}

}  // namespace duckdb
