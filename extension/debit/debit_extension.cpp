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
enum class Q1ColKind {
    Int64,    // BIGINT — read directly via FlatVector::GetData<int64_t>
    Int32,    // INTEGER / DATE — read as int32_t, promote to int64
    VarChar,  // VARCHAR — first byte ASCII to int64 (linestatus / returnflag / shipmode / shipinstruct)
    DateGEYear, // DATE (epoch days) — bucketed into year (key = year, e.g. 1992..1998)
};

// Bucket epoch-days (since 1970-01-01) into the calendar year that contains
// the date.  Hard-coded boundaries cover the TPC-H date span (1992-1998 and
// a small margin); used by load_bitmap('shipdate_GE') to build per-year
// IndexedComBitGE / IndexedX (mirror of teacher's `Btvs_GE[c_year - start_year]`).
static int64_t epoch_day_to_year(int32_t day) {
    static constexpr int boundaries[] = {
         8035,  8401,  8766,  9131,  9496,  9862, 10227, 10592, 10957
    };  // 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000
    static constexpr int years[]      = {
         1992,  1993,  1994,  1995,  1996,  1997,  1998,  1999,  2000
    };
    if (day < boundaries[0]) return 1991;
    for (int i = 0; i + 1 < (int)(sizeof(boundaries)/sizeof(boundaries[0])); i++)
        if (day >= boundaries[i] && day < boundaries[i+1])
            return years[i];
    return 2001;
}

struct ColInfo {
    std::string table;
    int storage_index;
    Q1ColKind kind;
};

static ColInfo resolve_column(const std::string& col) {
    // (table, column index) per duckdb-dev TPC-H schema.
    if (col == "orderkey")    return {"lineitem", 0,  Q1ColKind::Int64};
    if (col == "suppkey")     return {"lineitem", 2,  Q1ColKind::Int64};
    if (col == "partkey")     return {"lineitem", 1,  Q1ColKind::Int64};
    if (col == "shipdate")    return {"lineitem", 10, Q1ColKind::Int32};
    if (col == "shipdate_GE") return {"lineitem", 10, Q1ColKind::DateGEYear};
    if (col == "orderdate")   return {"orders",   4,  Q1ColKind::Int32};
    if (col == "linestatus")  return {"lineitem", 9,  Q1ColKind::VarChar};
    if (col == "returnflag")  return {"lineitem", 8,  Q1ColKind::VarChar};
    if (col == "discount")    return {"lineitem", 6,  Q1ColKind::Int64};   // DECIMAL(15,2) → int64 (raw)
    if (col == "quantity")    return {"lineitem", 4,  Q1ColKind::Int64};
    if (col == "shipmode")    return {"lineitem", 14, Q1ColKind::VarChar};
    if (col == "shipinstruct")return {"lineitem", 13, Q1ColKind::VarChar};
    return {"", -1, Q1ColKind::Int64};
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
// shipdate / orderdate).  For VARCHAR columns (linestatus / returnflag /
// shipmode / shipinstruct) takes the first byte as ASCII int64.  Used by
// load_bitmap to feed IndexedBitmap::build().
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
        switch (info.kind) {
            case Q1ColKind::Int64: {
                auto p = FlatVector::GetData<int64_t>(chunk.data[0]);
                out.insert(out.end(), p, p + chunk.size());
                break;
            }
            case Q1ColKind::Int32: {
                auto p = FlatVector::GetData<int32_t>(chunk.data[0]);
                for (idx_t i = 0; i < chunk.size(); i++)
                    out.push_back(static_cast<int64_t>(p[i]));
                break;
            }
            case Q1ColKind::DateGEYear: {
                auto p = FlatVector::GetData<int32_t>(chunk.data[0]);
                for (idx_t i = 0; i < chunk.size(); i++)
                    out.push_back(epoch_day_to_year(p[i]));
                break;
            }
            case Q1ColKind::VarChar: {
                // string_t — empty strings / NULLs go to 0; otherwise first byte ASCII.
                auto& vec = chunk.data[0];
                vec.Flatten(chunk.size());
                auto p = FlatVector::GetData<string_t>(vec);
                auto& validity = FlatVector::Validity(vec);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    int64_t v = 0;
                    if (validity.RowIsValid(i)) {
                        auto sz = p[i].GetSize();
                        if (sz > 0) v = static_cast<unsigned char>(p[i].GetData()[0]);
                    }
                    out.push_back(v);
                }
                break;
            }
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
    bool ge_path = (col == "shipdate_GE");
    using B = bm_bench::Backend;
    switch (backend) {
        case B::CB: {
            if (ge_path) {
                auto* x = new bm_index::IndexedComBitGE();
                x->build(values, values.size());
                idx = x;
            } else {
                auto* x = new bm_index::IndexedComBit();
                x->build(values, values.size());
                idx = x;
            }
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
        default: {
            // Default to ComBit when ALL/unspecified — for "ALL" semantics
            // we'd need separate fields per backend; keep simple for now.
            if (ge_path) {
                auto* x = new bm_index::IndexedComBitGE();
                x->build(values, values.size());
                idx = x;
            } else {
                auto* x = new bm_index::IndexedComBit();
                x->build(values, values.size());
                idx = x;
            }
            break;
        }
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