#pragma once

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"

#include <cstdint>

namespace duckdb {

class BMTableScan {
public:
    BMTableScan();
    ~BMTableScan();

    vector<row_t> *row_ids;
    idx_t *cursor;          // emit cursor
    idx_t num_idlist;

    void BMTPCH_Q1 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q3 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q4 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q5 (ExecutionContext &context, const PhysicalTableScan &op);
    // row-id list build
    void TPCH_Q6_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids);
    SourceResultType BMTPCH_Q6(ExecutionContext &context, DataChunk &chunk,
                               const TableScanBindData &bind_data);
    void BMTPCH_Q8 (ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q10(ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q12(ExecutionContext &context, const PhysicalTableScan &op);
    void BMTPCH_Q14(ExecutionContext &context, const PhysicalTableScan &op);

    void TPCH_Q15_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids);
    SourceResultType BMTPCH_Q15(ExecutionContext &context, DataChunk &chunk,
                                const TableScanBindData &bind_data);
    void BMTPCH_Q17(ExecutionContext &context, const PhysicalTableScan &op);

    void TPCH_Q19_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids);
    SourceResultType BMTPCH_Q19(ExecutionContext &context, DataChunk &chunk,
                                const TableScanBindData &bind_data);
};

}