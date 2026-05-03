#define DUCKDB_EXTENSION_MAIN

#include "debit_extension.hpp"
#include "bm_dbgen.hpp"

#ifndef DUCKDB_AMALGAMATION
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#endif

#include "execution/tpch/indexed_bitmap.hpp"
#include "execution/tpch/bm_bench_common.hpp"
#include <chrono>
#include <iostream>
#include <memory>

namespace duckdb {

// PRAGMA bm_tpch(N) — single entry point used by all 12 modernised TPC-H
// benchmarks.
static string PragmaTpchQuery(ClientContext &context, const FunctionParameters &parameters) {
    context.query_source = "bm_tpch";
    auto index = parameters.values[0].GetValue<int32_t>();
    return bmtpch::DBGenWrapper::GetQuery(index);
}

// ---------------------------------------------------------------------------
// PRAGMA load_bitmap(col_name) — pre-build a value-indexed bitmap auxiliary
// structure for the named column, store it in `context.client.bitmap_<col>`
// for BMTPCH_Q* handlers to consume.  The backend variant (ComBit /
// CRoaring / CRoaring+Run / WAH / EWAH / Concise) is selected via the
// DEBIT_BM environment variable, mirroring the teacher's BitEngine
// pattern of dynamic_cast<rabit::Rabit*>(context.client.bitmap_*).
//
// TPC-H 1.5.7 compliance: each load_bitmap call references EXACTLY ONE
// base-table column (PK / FK / date), so the auxiliary structure is
// permitted under the spec.  Build cost happens here (auxiliary
// construction), NOT inside the timed BMTPCH_Q* execution.
// ---------------------------------------------------------------------------

namespace {

struct BitmapColumnSpec {
    const char* name;
    const char* table;
    int storage_index;
    bool is_int64;        // l_orderkey/l_suppkey/etc. are int64; l_shipdate/o_orderdate are int32
    void** context_field; // not used directly; resolved per-context
};

// Returns the column spec by name + the field-pointer offset, set by caller
// based on the column.
struct ColInfo {
    std::string table;
    int storage_index;
    bool is_int64;
};

static ColInfo resolve_column(const std::string& col) {
    // (table, column index) per duckdb-dev TPC-H schema.
    if (col == "orderkey")    return {"lineitem", 0, true};
    if (col == "suppkey")     return {"lineitem", 2, true};
    if (col == "partkey")     return {"lineitem", 1, true};
    if (col == "shipdate")    return {"lineitem", 10, false};
    if (col == "shipdate_GE") return {"lineitem", 10, false};
    if (col == "orderdate")   return {"orders", 4, false};
    if (col == "linestatus")  return {"lineitem", 9, false};
    if (col == "returnflag")  return {"lineitem", 8, false};
    if (col == "discount")    return {"lineitem", 6, false};
    if (col == "quantity")    return {"lineitem", 4, false};
    if (col == "shipmode")    return {"lineitem", 14, false};
    if (col == "shipinstruct")return {"lineitem", 13, false};
    return {"", -1, false};
}

static void** resolve_context_field(ClientContext& ctx, const std::string& col) {
    if (col == "orderkey")    return &ctx.bitmap_orderkey;
    if (col == "suppkey")     return &ctx.bitmap_suppkey;
    if (col == "partkey")     return &ctx.bitmap_partkey;
    if (col == "shipdate")    return &ctx.bitmap_shipdate;
    if (col == "shipdate_GE") return &ctx.bitmap_shipdate_GE;
    if (col == "orderdate")   return &ctx.bitmap_orderdate;
    if (col == "linestatus")  return &ctx.bitmap_linestatus;
    if (col == "returnflag")  return &ctx.bitmap_returnflag;
    if (col == "discount")    return &ctx.bitmap_discount;
    if (col == "quantity")    return &ctx.bitmap_quantity;
    if (col == "shipmode")    return &ctx.bitmap_shipmode;
    if (col == "shipinstruct")return &ctx.bitmap_shipinstr;
    return nullptr;
}

// Scan column values into vector<int64_t>.  Promotes int32 to int64 (e.g.
// shipdate / linestatus / returnflag).  Used by load_bitmap to feed
// IndexedBitmap::build().
static std::vector<int64_t> scan_column(ClientContext& ctx, const ColInfo& info) {
    auto& tbl = Catalog::GetEntry<TableCatalogEntry>(ctx, "", "", info.table);
    auto& tx  = DuckTransaction::Get(ctx, tbl.catalog);
    TableScanState ss;
    vector<StorageIndex> col_ids = {StorageIndex(info.storage_index)};
    tbl.GetStorage().InitializeScan(ctx, tx, ss, col_ids);
    vector<LogicalType> types = {tbl.GetColumns().GetColumnTypes()[info.storage_index]};
    std::vector<int64_t> out;
    while (true) {
        DataChunk chunk; chunk.Initialize(ctx, types);
        tbl.GetStorage().Scan(tx, chunk, ss);
        if (chunk.size() == 0) break;
        if (info.is_int64) {
            auto p = FlatVector::GetData<int64_t>(chunk.data[0]);
            out.insert(out.end(), p, p + chunk.size());
        } else {
            auto p = FlatVector::GetData<int32_t>(chunk.data[0]);
            for (idx_t i = 0; i < chunk.size(); i++)
                out.push_back(static_cast<int64_t>(p[i]));
        }
    }
    return out;
}

}  // anonymous namespace

static string PragmaLoadBitmap(ClientContext &context, const FunctionParameters &parameters) {
    auto col = parameters.values[0].GetValue<std::string>();
    auto info = resolve_column(col);
    if (info.storage_index < 0) {
        std::cerr << "[load_bitmap] unknown column: " << col << std::endl;
        return "";
    }
    void** field = resolve_context_field(context, col);
    if (!field) {
        std::cerr << "[load_bitmap] no context field for: " << col << std::endl;
        return "";
    }
    if (*field) {
        std::cerr << "[load_bitmap] " << col << " already loaded; skipping." << std::endl;
        return "";
    }
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    auto values = scan_column(context, info);
    auto t1 = clock::now();
    std::cerr << "[load_bitmap] " << col << ": scanned " << values.size()
              << " rows in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms" << std::endl;

    // Backend selection via DEBIT_BM env (mirrors BMTPCH gating).
    auto backend = bm_bench::parse_backend("DEBIT_BM");
    bm_index::IBitmapIndex* idx = nullptr;
    using B = bm_bench::Backend;
    switch (backend) {
        case B::CB: {
            auto* x = new bm_index::IndexedComBit();
            x->build(values, values.size());
            idx = x;
            break;
        }
        case B::CR: {
            auto* x = new bm_index::IndexedCRoaring();
            x->build(values, values.size(), false);
            idx = x;
            break;
        }
        case B::CRR: {
            auto* x = new bm_index::IndexedCRoaring();
            x->build(values, values.size(), true);
            idx = x;
            break;
        }
        case B::WAH: {
            auto* x = new bm_index::IndexedWAH();
            x->build(values, values.size());
            idx = x;
            break;
        }
        case B::EW: {
            auto* x = new bm_index::IndexedEWAH();
            x->build(values, values.size());
            idx = x;
            break;
        }
        case B::CON: {
            auto* x = new bm_index::IndexedConcise();
            x->build(values, values.size());
            idx = x;
            break;
        }
        case B::ALL:
        default:
            // Default to ComBit when ALL/unspecified — for "ALL" semantics
            // we'd need separate fields per backend; keep simple for now.
            auto* x = new bm_index::IndexedComBit();
            x->build(values, values.size());
            idx = x;
            break;
    }
    auto t2 = clock::now();
    *field = idx;
    std::cerr << "[load_bitmap] " << col << ": built "
              << idx->backend_name() << " index ("
              << idx->num_keys() << " keys, "
              << idx->storage_bytes() / 1e6 << " MB) in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms" << std::endl;
    return "";
}

static void LoadInternal(DuckDB &db) {
    auto &db_instance = *db.instance;

    auto bmtpch_func = PragmaFunction::PragmaCall("bm_tpch", PragmaTpchQuery, {LogicalType::BIGINT});
    ExtensionUtil::RegisterFunction(db_instance, bmtpch_func);

    auto load_bitmap_func = PragmaFunction::PragmaCall("load_bitmap", PragmaLoadBitmap,
                                                       {LogicalType::VARCHAR});
    ExtensionUtil::RegisterFunction(db_instance, load_bitmap_func);
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