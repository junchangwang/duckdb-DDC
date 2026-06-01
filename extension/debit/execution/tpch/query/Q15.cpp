// TPC-H Q15 - Top Supplier Query (spec v3.0.1 §2.4.15)
//
// Streaming variant - mirrors BitEngine paper Q15 methodology.
//
// Pipeline:
//   - First lineitem table-scan call: build row_ids via shipdate filter
//     (month-GE 3-key OR for cb_ge / cr+ge, per-day range OR otherwise).
//   - Subsequent calls: BMFetch 2048-row chunk of
//       (l_suppkey, l_extendedprice, l_discount)
//     and return to DuckDB upstream.
//   - DuckDB upstream pipeline does the actual TPC-H Q15 work:
//       GROUP BY l_suppkey + SUM(l_extendedprice*(1-l_discount))
//       MAX(total_revenue)
//       JOIN supplier ON s_suppkey = supplier_no
//       ORDER BY s_suppkey
//
// BitEngine paper Q15 uses BMFetch (fast path); we match that.

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
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

#include <atomic>
#include <iostream>
#include <mutex>
#include <vector>

namespace duckdb {

static std::mutex     q15_build_mutex;
static std::atomic<bool> q15_built{false};

// TPC-H Q15 spec: l_shipdate in [1996-01-01, 1996-04-01).  Per-day epoch
// keys [9496, 9586] (91 keys), or month-GE 3-key OR {199601, 199602, 199603}.
static constexpr int Q15_DATE_START = 9496;
static constexpr int Q15_DATE_END   = 9586;
static constexpr int64_t Q15_MONTH_KEYS[3] = {199601, 199602, 199603};

// Per-backend row-id extractors.
template <typename Btv>
static void q15_get_rowids(const Btv& b, vector<row_t>* out);
template <>
void q15_get_rowids<ComBit>(const ComBit& b, vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q15_get_rowids<roaring::Roaring>(const roaring::Roaring& b, vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q15_get_rowids<ibis::bitvector>(const ibis::bitvector& b, vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q15_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q15_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}

// =====================================================================
// TPCH_Q15_Lineitem_GetRowIds - shipdate filter (per active backend) ->
// row_ids.  Prefers month-GE 3-key OR when bitmap_shipdate_GE_month is
// loaded; else per-day 91-key range OR.
// =====================================================================
void BMTableScan::TPCH_Q15_Lineitem_GetRowIds(ExecutionContext &context,
                                              vector<row_t> *row_ids)
{
    if (q15_built.load()) return;

    auto* idx_ge_m = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_GE_month);
    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    auto* idx_use  = idx_ge_m ? idx_ge_m : idx_ship;
    if (!idx_use) {
        std::cerr << "[Q15] ERROR: neither bitmap_shipdate_GE_month nor bitmap_shipdate loaded.\n";
        row_ids->clear();
        q15_built.store(true);
        return;
    }
    const bool use_month_ge = (idx_ge_m != nullptr);
    const std::vector<int64_t> q15_ge_keys(Q15_MONTH_KEYS, Q15_MONTH_KEYS + 3);

    row_ids->clear();

    if (auto* cbge = dynamic_cast<bm_index::IndexedComBitGE*>(idx_use)) {
        ComBit f = cbge->or_many(q15_ge_keys);
        q15_get_rowids(f, row_ids);
    } else if (auto* cb = dynamic_cast<bm_index::IndexedComBit*>(idx_use)) {
        ComBit f = ComBit::from_sparse_positions({}, cb->num_rows(), cb->segment_bits());
        cb->apply_or_range_to(f, Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* cr = dynamic_cast<bm_index::IndexedCRoaring*>(idx_use)) {
        roaring::Roaring f = use_month_ge
            ? cr->or_many(q15_ge_keys)
            : cr->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* wah = dynamic_cast<bm_index::IndexedWAH*>(idx_use)) {
        ibis::bitvector f = use_month_ge
            ? wah->or_many(q15_ge_keys)
            : wah->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* ew = dynamic_cast<bm_index::IndexedEWAH*>(idx_use)) {
        ewah::EWAHBoolArray<uint64_t> f = use_month_ge
            ? ew->or_many(q15_ge_keys)
            : ew->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* con = dynamic_cast<bm_index::IndexedConcise*>(idx_use)) {
        ConciseSet<false> f = use_month_ge
            ? con->or_many(q15_ge_keys)
            : con->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else {
        std::cerr << "[Q15] ERROR: unrecognised IBitmapIndex backend.\n";
    }

    q15_built.store(true);
}

// =====================================================================
// BMTPCH_Q15 - DuckDB pull-style streaming entry point.  Same pattern
// as BMTPCH_Q6/Q19: build row_ids on first call, BMFetch chunks
// subsequently.  Reset state on FINISHED for multi-iter benchmarking
// (must be run single-threaded; see Q19.cpp comment).
// =====================================================================
SourceResultType BMTableScan::BMTPCH_Q15(ExecutionContext &context,
                                          DataChunk &chunk,
                                          const TableScanBindData &bind_data)
{
    std::lock_guard<std::mutex> lock(q15_build_mutex);

    if (*cursor == 0 && !q15_built.load()) {
        TPCH_Q15_Lineitem_GetRowIds(context, row_ids);
        num_idlist = row_ids->size();
    }

    if (*cursor < row_ids->size()) {
        // 3 columns: l_suppkey (2), l_extendedprice (5), l_discount (6).
        vector<StorageIndex> storage_column_ids;
        storage_column_ids.push_back(StorageIndex(2));
        storage_column_ids.push_back(StorageIndex(5));
        storage_column_ids.push_back(StorageIndex(6));

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

        // Matches BitEngine paper Q15: BMFetch (fast batched path).
        table_bind_data.table.GetStorage().BMFetch(transaction, chunk,
                                                   storage_column_ids, row_ids_vec,
                                                   fetch_count, column_fetch_state,
                                                   num_idlist);
        *cursor += fetch_count;
        return SourceResultType::HAVE_MORE_OUTPUT;
    }

    // Exhausted - reset state for next PRAGMA bm_tpch(15) call.
    row_ids->clear();
    *cursor = 0;
    num_idlist = 0;
    q15_built.store(false);
    context.client.query_source = "tpch";

    return SourceResultType::FINISHED;
}

}  // namespace duckdb
