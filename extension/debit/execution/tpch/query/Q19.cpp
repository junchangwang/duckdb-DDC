// TPC-H Q19 - Discounted Revenue Query (spec v3.0.1 §2.4.19)
//
// Streaming variant - mirrors BitEngine paper Q19 methodology.
//
// Pipeline:
//   - First lineitem table-scan call: build row_ids via the loaded
//     bitmap_shipmode AND bitmap_shipinstr filter.
//   - Subsequent calls: Fetch one 2048-row chunk of
//       (l_partkey, l_quantity, l_shipmode, l_extendedprice, l_discount)
//     and return it to DuckDB's upstream operators.
//   - DuckDB upstream pipeline does the actual TPC-H Q19 work:
//       hash join with part on l_partkey = p_partkey
//       3-OR predicate filter (Brand#12/SM*/qty1-11/size1-5, Brand#23/MED*,
//                              Brand#34/LG*) AND shipmode/shipinstruct
//       SUM(l_extendedprice * (1 - l_discount))
//
// The end-to-end query latency observed by the user (printed by DuckDB
// CLI .timer) is the comparable measurement against BitEngine paper.

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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <vector>

namespace duckdb {

// Per-iteration build state.  We RESET on FINISHED so each subsequent
// PRAGMA bm_tpch(19) call re-builds row_ids (re-measures bitmap phase).
static std::mutex     q19_build_mutex;
static std::atomic<bool> q19_built{false};

// Per-backend row-id extractors (sorted output for BMFetch/Fetch).
template <typename Btv>
static void q19_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q19_get_rowids<ComBit>(const ComBit& b, std::vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q19_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q19_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q19_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q19_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}

// =====================================================================
// TPCH_Q19_Lineitem_GetRowIds - bitmap filter (per active backend) ->
// row_ids vector.  Single-threaded under q19_build_mutex (held by
// BMTPCH_Q19 caller).  Mirrors BitEngine paper exactly: shipmode 'A'
// (= AIR or AIR REG) AND shipinstruct 'D' (= DELIVER IN PERSON).
// =====================================================================
void BMTableScan::TPCH_Q19_Lineitem_GetRowIds(ExecutionContext &context,
                                              vector<row_t> *row_ids)
{
    if (q19_built.load()) return;

    auto* idx_sm = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipmode);
    auto* idx_si = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipinstr);
    if (!idx_sm || !idx_si) {
        std::cerr << "[Q19] ERROR: bitmap_shipmode or bitmap_shipinstr not loaded.\n";
        row_ids->clear();
        q19_built.store(true);
        return;
    }

    size_t num_rows = idx_sm->num_rows();
    row_ids->clear();

    if (auto* cb_sm = dynamic_cast<bm_index::IndexedComBit*>(idx_sm)) {
        auto* cb_si = dynamic_cast<bm_index::IndexedComBit*>(idx_si);
        if (!cb_si) { std::cerr << "[Q19] CB type mismatch.\n"; return; }
        ComBit bm_sm = ComBit::from_sparse_positions({}, num_rows, cb_sm->segment_bits());
        ComBit bm_si = ComBit::from_sparse_positions({}, num_rows, cb_si->segment_bits());
        cb_sm->apply_or_to(bm_sm, 'A');
        cb_si->apply_or_to(bm_si, 'D');
        bm_sm &= bm_si;
        q19_get_rowids(bm_sm, row_ids);
    } else if (auto* cr_sm = dynamic_cast<bm_index::IndexedCRoaring*>(idx_sm)) {
        auto* cr_si = dynamic_cast<bm_index::IndexedCRoaring*>(idx_si);
        if (!cr_si) { std::cerr << "[Q19] CR type mismatch.\n"; return; }
        roaring::Roaring bm_sm, bm_si;
        cr_sm->apply_or_to(bm_sm, 'A');
        cr_si->apply_or_to(bm_si, 'D');
        bm_sm &= bm_si;
        q19_get_rowids(bm_sm, row_ids);
    } else if (auto* wah_sm = dynamic_cast<bm_index::IndexedWAH*>(idx_sm)) {
        auto* wah_si = dynamic_cast<bm_index::IndexedWAH*>(idx_si);
        if (!wah_si) { std::cerr << "[Q19] WAH type mismatch.\n"; return; }
        ibis::bitvector bm_sm, bm_si;
        wah_sm->apply_or_to(bm_sm, 'A');
        wah_si->apply_or_to(bm_si, 'D');
        bm_sm &= bm_si;
        q19_get_rowids(bm_sm, row_ids);
    } else if (auto* ew_sm = dynamic_cast<bm_index::IndexedEWAH*>(idx_sm)) {
        auto* ew_si = dynamic_cast<bm_index::IndexedEWAH*>(idx_si);
        if (!ew_si) { std::cerr << "[Q19] EW type mismatch.\n"; return; }
        ewah::EWAHBoolArray<uint64_t> bm_sm, bm_si;
        ew_sm->apply_or_to(bm_sm, 'A');
        ew_si->apply_or_to(bm_si, 'D');
        ewah::EWAHBoolArray<uint64_t> tmp;
        bm_sm.logicaland(bm_si, tmp);
        q19_get_rowids(tmp, row_ids);
    } else if (auto* con_sm = dynamic_cast<bm_index::IndexedConcise*>(idx_sm)) {
        auto* con_si = dynamic_cast<bm_index::IndexedConcise*>(idx_si);
        if (!con_si) { std::cerr << "[Q19] CON type mismatch.\n"; return; }
        ConciseSet<false> bm_sm, bm_si;
        con_sm->apply_or_to(bm_sm, 'A');
        con_si->apply_or_to(bm_si, 'D');
        ConciseSet<false> tmp = bm_sm.logicaland(bm_si);
        q19_get_rowids(tmp, row_ids);
    } else {
        std::cerr << "[Q19] ERROR: unrecognised IBitmapIndex backend.\n";
    }

    q19_built.store(true);
}

// =====================================================================
// BMTPCH_Q19 - DuckDB pull-style streaming entry point.  First call
// (cursor==0) builds row_ids via the bitmap filter; subsequent calls
// Fetch a 2048-row chunk and return SourceResultType::HAVE_MORE_OUTPUT.
// On exhaustion, state is reset so the NEXT PRAGMA bm_tpch(19) re-runs
// the full pipeline (allowing multi-iter benchmarking).
// =====================================================================
SourceResultType BMTableScan::BMTPCH_Q19(ExecutionContext &context,
                                          DataChunk &chunk,
                                          const TableScanBindData &bind_data)
{
    std::lock_guard<std::mutex> lock(q19_build_mutex);

    if (*cursor == 0 && !q19_built.load()) {
        TPCH_Q19_Lineitem_GetRowIds(context, row_ids);
        num_idlist = row_ids->size();
    }

    if (*cursor < row_ids->size()) {
        // 5 columns matching BitEngine paper Q19 BMTPCH_Q19:
        //   l_partkey (1), l_quantity (4), l_shipmode (14),
        //   l_extendedprice (5), l_discount (6).
        vector<StorageIndex> storage_column_ids;
        storage_column_ids.push_back(StorageIndex(1));
        storage_column_ids.push_back(StorageIndex(4));
        storage_column_ids.push_back(StorageIndex(14));
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

        // Matches BitEngine paper Q19: vanilla Fetch (slow path).
        table_bind_data.table.GetStorage().Fetch(transaction, chunk,
                                                 storage_column_ids, row_ids_vec,
                                                 fetch_count, column_fetch_state);
        *cursor += fetch_count;
        return SourceResultType::HAVE_MORE_OUTPUT;
    }

    // Exhausted - reset state for the NEXT PRAGMA bm_tpch(19) call so
    // we re-measure the bitmap-filter + Fetch + upstream pipeline fresh.
    row_ids->clear();
    *cursor = 0;
    num_idlist = 0;
    q19_built.store(false);
    context.client.query_source = "tpch";   // resume normal scan path

    return SourceResultType::FINISHED;
}

}  // namespace duckdb
