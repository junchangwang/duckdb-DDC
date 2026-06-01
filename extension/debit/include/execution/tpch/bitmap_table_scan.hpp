#pragma once

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"

#include <cstdint>

namespace duckdb {

// BMTableScan is the dispatch shim that PhysicalTableScan delegates to
// when the current query was produced by `PRAGMA bm_tpch(N)`.  It owns
// the per-connection scratch state (row id buffer + cursor) used by Q6
// to feed materialised row IDs back into DuckDB's storage scanner; the
// other 11 queries run their entire pipeline inside their BMTPCH_Q<N>
// entry point and return without touching DuckDB storage.
class BMTableScan {
public:
    BMTableScan();
    ~BMTableScan();

    // Q6-specific scratch state (used only by BMTPCH_Q6 / BMFetch).
    vector<row_t> *row_ids;
    idx_t *cursor;
    idx_t num_idlist;

    // 12 modernised TPC-H entry points.  Each runs the full per-backend
    // benchmark sweep (ComBit / WAH / CRoaring / CRoaring+Run / EWAH /
    // Bitset / Bitset+AVX512 / Concise) and validates against a DuckDB
    // native SQL ground truth.  Q6 is the only one that returns rows to
    // the caller; the rest are sinks (results printed + asserted).
    void BMTPCH_Q1 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q3 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q4 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q5 (ExecutionContext &context, const PhysicalTableScan &op);
    void TPCH_Q6_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids);
    SourceResultType BMTPCH_Q6(ExecutionContext &context, DataChunk &chunk,
                               const TableScanBindData &bind_data);
    void BMTPCH_Q8 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q10(ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q12(ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q14(ExecutionContext &context, const PhysicalTableScan &op);
    // Q15: streaming pattern (mirrors BitEngine paper).  First call builds
    // row_ids via shipdate range / month-GE OR; each subsequent call BMFetches
    // a 2048-row chunk of (l_suppkey, l_extendedprice, l_discount) for DuckDB
    // upstream's GROUP BY l_suppkey + MAX + JOIN supplier.
    void TPCH_Q15_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids);
    SourceResultType BMTPCH_Q15(ExecutionContext &context, DataChunk &chunk,
                                const TableScanBindData &bind_data);
    void BMTPCH_Q17(ExecutionContext &context, const PhysicalTableScan &op);
    // Q19: streaming pattern (mirrors BitEngine paper).  First call into the
    // lineitem table-scan builds row_ids via shipmode AND shipinstruct; each
    // subsequent call Fetches one 2048-row chunk and returns it to DuckDB's
    // upstream (hash join with part + 3-OR predicate filter + SUM agg).
    void TPCH_Q19_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids);
    SourceResultType BMTPCH_Q19(ExecutionContext &context, DataChunk &chunk,
                                const TableScanBindData &bind_data);
};

}