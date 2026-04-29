#define DUCKDB_EXTENSION_MAIN

#include "debit_extension.hpp"
#include "bm_dbgen.hpp"

#ifndef DUCKDB_AMALGAMATION
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/main/client_context.hpp"
#endif

namespace duckdb {

// PRAGMA bm_tpch(N) — single entry point used by all 12 modernised TPC-H
// benchmarks (Q1, Q3, Q4, Q5, Q6, Q8, Q10, Q12, Q14, Q15, Q17, Q19).  The
// query string returned here is recognised by the patched
// PhysicalTableScan dispatcher in src/execution/operator/scan/, which
// routes it to BMTableScan::BMTPCH_Q<N>(...) defined under
// extension/debit/execution/tpch/query/.
static string PragmaTpchQuery(ClientContext &context, const FunctionParameters &parameters) {
    context.query_source = "bm_tpch";
    auto index = parameters.values[0].GetValue<int32_t>();
    return bmtpch::DBGenWrapper::GetQuery(index);
}

static void LoadInternal(DuckDB &db) {
    auto &db_instance = *db.instance;

    auto bmtpch_func = PragmaFunction::PragmaCall("bm_tpch", PragmaTpchQuery, {LogicalType::BIGINT});
    ExtensionUtil::RegisterFunction(db_instance, bmtpch_func);
}

void DebitExtension::Load(DuckDB &db) {
    LoadInternal(db);
}

std::string DebitExtension::Name() {
    return "debit";
}

std::string DebitExtension::Version() const {
    return "1.0.0";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void debit_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    duckdb::LoadInternal(db_wrapper);
}

DUCKDB_EXTENSION_API const char *debit_version() {
    return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif