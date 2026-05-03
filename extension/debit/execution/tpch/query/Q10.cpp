// TPC-H Q10 — Returned Item Reporting Query
//
// Bitmap-accelerated implementation of TPC-H Q10 (spec v3.0.1 §2.4.10).
//
// Query semantics (verbatim from the spec, with the standard substitution
// parameter DATE = 1993-10-01):
//
//   SELECT c_custkey, c_name,
//          sum(l_extendedprice * (1 - l_discount)) AS revenue,
//          c_acctbal, n_name, c_address, c_phone, c_comment
//     FROM customer, orders, lineitem, nation
//    WHERE c_custkey  = o_custkey
//      AND l_orderkey = o_orderkey
//      AND o_orderdate >= DATE '1993-10-01'
//      AND o_orderdate <  DATE '1993-10-01' + INTERVAL '3' MONTH
//      AND l_returnflag = 'R'
//      AND c_nationkey = n_nationkey
//    GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
//    ORDER BY revenue DESC
//    LIMIT 20;
//
// Bitmap pipeline (per backend, per iteration):
//
//   1. OR 92 lineitem-level orderdate bitmaps {day_639..day_730}
//      → date_mask      (92 days span 1993-10-01 .. 1993-12-31 inclusive,
//                        1994-01-01 is the exclusive upper bound)
//   2. AND lineitem-level returnflag='R' bitmap  → filter_mask
//   3. Decode filter_mask  → walk set bits (row_ids)
//   4. Aggregate: for each qualifying row,
//        custkey = orders_custkey[l_orderkey[row]]
//        revenue_by_custkey[custkey] += l_price[row] * (100 - l_discount[row])
//   5. Heap top-20 by revenue DESC, tie-break by c_custkey ASC.
//
// Bitmap reuse strategy:
//   * orderdate bitmaps are reused from the Q5 pre-joined directory
//     (tpch_q5{_sfN}_{cb,wah,cr,ew}/orderdate/{639..730}.bm).  They are
//     lineitem-level, pre-joined on o_orderdate, so no join with the
//     orders table is needed to evaluate the date predicate.
//   * returnflag bitmaps are reused from the Q1 directory
//     (tpch_q1{_sfN}_{cb,wah,cr,ew}/returnflag/2.bm).  Under the Q1
//     builder's encoding (A=0, N=1, R=2), index 2 is 'R'.  This
//     intentionally differs from the legacy Rabit Q10 implementation,
//     which used a different (Rabit-specific) encoding.
//
// Side tables (orders / customer / nation) are loaded from DuckDB
// storage exactly once, outside the measured iteration loop, so that
// bitmap timings are not polluted by DuckDB-level I/O or decompression.
// DuckDB's native Q10 SQL is executed after the iteration loop to
// provide the ground-truth Top-20 that every active backend is checked
// against.
//
// Environment variables (shared with Q1/Q5/Q6 via bm_bench_common.hpp):
//   DEBIT_BM          = all|wah|cb|cr|crr|ew           (default = all)
//   DEBIT_BITMAP_DIR  = absolute base for tpch_q*_*/   (default = cwd)
//   DEBIT_ITER        = measured iterations            (default = 10)
//   DEBIT_WARMUP      = warm-up iterations             (default = 2)
//   TPCH_SF           = scale factor                   (default = 10)
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
#include "duckdb/common/types/string_type.hpp"

#include "combit_adapter.h"
#include "combit/include/combit.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"

// Uncompressed bitset (BS / BSA) + Concise (CON) baselines and their
// shared from-CRoaring loaders.
#include "bitset_simple.h"
#include "Concise/concise.h"
#include "execution/tpch/bm_baseline_loaders.hpp"

#include "execution/tpch/bm_bench_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {

// ===========================================================================
// Bitmap directory resolution
// ===========================================================================
// Q10 reuses Q5's orderdate/ and Q1's returnflag/ directories unchanged.
// DEBIT_BITMAP_DIR (if set) is prepended so the benchmark is cwd-independent.
static const std::string Q10_SF         = bm_bench::sf_suffix();

static const std::string Q10_Q5_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q5" + Q10_SF + "_combit");
static const std::string Q10_Q5_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q5" + Q10_SF + "_wah");
static const std::string Q10_Q5_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q5" + Q10_SF + "_croaring");
static const std::string Q10_Q5_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q5" + Q10_SF + "_ewah");

static const std::string Q10_Q1_CB_DIR  = bm_bench::resolve_bitmap_dir("tpch_q1" + Q10_SF + "_combit");
static const std::string Q10_Q1_WAH_DIR = bm_bench::resolve_bitmap_dir("tpch_q1" + Q10_SF + "_wah");
static const std::string Q10_Q1_CR_DIR  = bm_bench::resolve_bitmap_dir("tpch_q1" + Q10_SF + "_croaring");
static const std::string Q10_Q1_EW_DIR  = bm_bench::resolve_bitmap_dir("tpch_q1" + Q10_SF + "_ewah");

// ===========================================================================
// Backend selection (DEBIT_BM shared with Q1/Q5/Q6; legacy Q10_BM honoured)
// ===========================================================================
using Q10BmType = bm_bench::Backend;
static const Q10BmType Q10_BM = bm_bench::parse_backend("Q10_BM");

static bool run_all() { return Q10_BM == Q10BmType::ALL; }
static bool run_wah() { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::WAH; }
static bool run_cb()  { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::CB;  }
static bool run_cr()  { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::CR;  }
static bool run_crr() { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::CRR; }
static bool run_ew()  { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::EW;  }
static bool run_bs()  { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::BS;  }
static bool run_bsa() { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::BSA; }
static bool run_con() { return Q10_BM == Q10BmType::ALL || Q10_BM == Q10BmType::CON; }

static const char* q10_bm_label() { return bm_bench::backend_label(Q10_BM); }
static std::string q10_sf_label() { return bm_bench::sf_label(); }

// ===========================================================================
// Query constants (spec §2.4.10)
// ===========================================================================
// o_orderdate window: [1993-10-01, 1994-01-01)
//   In days-since-1992-01-01 (the encoding used by the Q5 builder):
//     1992-01-01 .. 1993-10-01  = 366 (1992, leap) + 273 days = 639
//     1994-01-01                = 366 + 365           = 731  (exclusive)
//   So the inclusive day-index range is [639, 730] == 92 bitmaps.
static const int Q10_DATE_START = 639;
static const int Q10_DATE_END   = 730;

// l_returnflag = 'R'
//   Q1 builder encoding: A=0, N=1, R=2 (see Q1.cpp:71).  Hence the
//   'R' bitmap file is returnflag/2.bm.
static const int Q10_RF_R_IDX   = 2;

// LIMIT 20 (spec)
static constexpr int Q10_TOPK   = 20;

// Iteration control (DEBIT_ITER / DEBIT_WARMUP override)
static const int Q10_ITERATIONS = bm_bench::iter_count(10);
static const int Q10_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q10_once_flag;

// ===========================================================================
// ComBit byte-LUT: byte value (0..255) → set bit positions
// ===========================================================================
// Used by the ComBit post-AND decode to extract row indices from each
// literal AND byte.  Pattern shared with Q5/Q6.
// ===========================================================================
// Bitmap loaders (same binary formats as Q1/Q5/Q6)
// ===========================================================================
static ComBit q10_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: cannot open " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q10_load_cr(const std::string& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) return roaring::Roaring();
    auto sz = in.tellg(); in.seekg(0);
    uint32_t ls; in.read(reinterpret_cast<char*>(&ls), 4);
    auto bsz = sz - 4;
    if (bsz > 0) {
        std::vector<char> buf(bsz);
        in.read(buf.data(), bsz);
        return roaring::Roaring::readSafe(buf.data(), bsz);
    }
    return roaring::Roaring();
}
static ibis::bitvector q10_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q10_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// ===========================================================================
// Side-table row types
// ===========================================================================
// All 6 non-aggregate group-by columns (excluding c_custkey itself) are
// stored here.  Looked up only for the 20 custkeys that survive Top-K,
// so a hash map is fine.  c_acctbal is DECIMAL(15,2) stored by DuckDB as
// int64 × 100.
struct Q10Customer {
    std::string c_name;
    std::string c_address;
    std::string c_phone;
    std::string c_comment;
    int64_t     c_acctbal_fp = 0;   // decimal(15,2) × 100
    int32_t     c_nationkey  = 0;
};

// ===========================================================================
// Top-K heap element
// ===========================================================================
// revenue_fp is fixed-point: sum(l_extendedprice_fp * (100 - l_discount_fp)).
// Since DuckDB stores DECIMAL(15,2) as int64 × 100, the product is already
// scaled by 10^4 — print as double(rev_fp) / 10000.0 to match spec output.
struct Q10Row {
    int64_t c_custkey;
    int64_t revenue_fp;
};
struct Q10MinHeapCmp {
    // Min-heap: top element is the *smallest* revenue, so we can pop it when
    // a larger candidate arrives.  Tie-break: larger custkey on top so that
    // among equal-revenue rows, smaller custkey is retained (spec-friendly).
    bool operator()(const Q10Row& a, const Q10Row& b) const {
        if (a.revenue_fp != b.revenue_fp) return a.revenue_fp > b.revenue_fp;
        return a.c_custkey < b.c_custkey;
    }
};

// ===========================================================================
// ms() — microsecond-precision duration in milliseconds (double)
// ===========================================================================
static inline double q10_ms(std::chrono::high_resolution_clock::time_point a,
                            std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

// ===========================================================================
// BMTPCH_Q10 — main benchmark entry point
// ===========================================================================
// Called once per connection via std::call_once.  The ExecutionContext is
// used only to obtain a DuckDB catalog handle for storage-level scans
// (orders, customer, nation, lineitem) and for the ground-truth SQL.
void BMTableScan::BMTPCH_Q10(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q10_once_flag, [&]() {

    using clock = std::chrono::high_resolution_clock;

    bm_bench::warn_if_sf1();

    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q10 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << q10_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q10 Benchmark — " << q10_bm_label() << " only ("
                  << q10_sf_label() << ")" << std::endl;
    }
    std::cout << "  OR orderdate days " << Q10_DATE_START << ".." << Q10_DATE_END
              << " (" << (Q10_DATE_END - Q10_DATE_START + 1) << " bitmaps)" << std::endl;
    std::cout << "  Then AND returnflag='R' (index " << Q10_RF_R_IDX
              << ") -> aggregate revenue per c_custkey -> Top-" << Q10_TOPK << std::endl;
    std::cout << "  TPC-H params: o_orderdate [1993-10-01, 1994-01-01), l_returnflag = 'R'" << std::endl;
    std::cout << "  Bitmap dirs:" << std::endl;
    if (run_cb())              std::cout << "    orderdate: " << Q10_Q5_CB_DIR  << "  returnflag: " << Q10_Q1_CB_DIR  << std::endl;
    if (run_cr() || run_crr()) std::cout << "    orderdate: " << Q10_Q5_CR_DIR  << "  returnflag: " << Q10_Q1_CR_DIR  << std::endl;
    if (run_wah())             std::cout << "    orderdate: " << Q10_Q5_WAH_DIR << "  returnflag: " << Q10_Q1_WAH_DIR << std::endl;
    if (run_ew())              std::cout << "    orderdate: " << Q10_Q5_EW_DIR  << "  returnflag: " << Q10_Q1_EW_DIR  << std::endl;
    std::cout << "  Iterations: " << Q10_ITERATIONS << " (first " << Q10_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // -----------------------------------------------------------------------
    // 1. Side-table load: nation (25 rows)
    // -----------------------------------------------------------------------
    //   nation table cols used: 0 = n_nationkey (int32), 1 = n_name (varchar)
    std::unordered_map<int32_t, std::string> nation_name;
    {
        auto t0 = clock::now();
        auto &tbl = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "nation");
        auto &txn = DuckTransaction::Get(context.client, tbl.catalog);
        TableScanState st;
        vector<StorageIndex> cols{ StorageIndex(0), StorageIndex(1) };
        tbl.GetStorage().InitializeScan(context.client, txn, st, cols);
        vector<LogicalType> types{
            tbl.GetColumns().GetColumnTypes()[0],
            tbl.GetColumns().GetColumnTypes()[1] };
        while (true) {
            DataChunk ch;
            ch.Initialize(context.client, types);
            tbl.GetStorage().Scan(txn, ch, st);
            if (ch.size() == 0) break;
            auto nk = FlatVector::GetData<int32_t>(ch.data[0]);
            auto nn = FlatVector::GetData<string_t>(ch.data[1]);
            for (idx_t i = 0; i < ch.size(); i++) {
                nation_name[nk[i]] = nn[i].GetString();
            }
        }
        std::cout << "\n[Side-load] nation: " << nation_name.size()
                  << " rows in " << q10_ms(t0, clock::now()) << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. Side-table load: customer (1.5M × SF rows)
    // -----------------------------------------------------------------------
    //   customer cols: 0=c_custkey, 1=c_name, 2=c_address, 3=c_nationkey,
    //                  4=c_phone, 5=c_acctbal, 7=c_comment  (6=c_mktsegment skipped)
    std::unordered_map<int64_t, Q10Customer> customer_info;
    int64_t max_custkey = 0;
    {
        auto t0 = clock::now();
        auto &tbl = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "customer");
        auto &txn = DuckTransaction::Get(context.client, tbl.catalog);
        TableScanState st;
        vector<StorageIndex> cols{
            StorageIndex(0), StorageIndex(1), StorageIndex(2), StorageIndex(3),
            StorageIndex(4), StorageIndex(5), StorageIndex(7) };
        tbl.GetStorage().InitializeScan(context.client, txn, st, cols);
        vector<LogicalType> types;
        for (auto c : {0, 1, 2, 3, 4, 5, 7}) types.push_back(tbl.GetColumns().GetColumnTypes()[c]);

        customer_info.reserve(2'000'000);

        while (true) {
            DataChunk ch;
            ch.Initialize(context.client, types);
            tbl.GetStorage().Scan(txn, ch, st);
            if (ch.size() == 0) break;
            auto ck   = FlatVector::GetData<int64_t>(ch.data[0]);
            auto nm   = FlatVector::GetData<string_t>(ch.data[1]);
            auto addr = FlatVector::GetData<string_t>(ch.data[2]);
            auto nk   = FlatVector::GetData<int32_t>(ch.data[3]);
            auto ph   = FlatVector::GetData<string_t>(ch.data[4]);
            auto ab   = FlatVector::GetData<int64_t>(ch.data[5]);   // DECIMAL(15,2) × 100
            auto cm   = FlatVector::GetData<string_t>(ch.data[6]);

            for (idx_t i = 0; i < ch.size(); i++) {
                Q10Customer& c = customer_info[ck[i]];
                c.c_name       = nm[i].GetString();
                c.c_address    = addr[i].GetString();
                c.c_phone      = ph[i].GetString();
                c.c_comment    = cm[i].GetString();
                c.c_acctbal_fp = ab[i];
                c.c_nationkey  = nk[i];
                if (ck[i] > max_custkey) max_custkey = ck[i];
            }
        }
        std::cout << "[Side-load] customer: " << customer_info.size()
                  << " rows (max_custkey=" << max_custkey << ") in "
                  << q10_ms(t0, clock::now()) << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 3. Side-table load: orders — build orderkey → custkey flat vector
    // -----------------------------------------------------------------------
    //   orders cols: 0=o_orderkey, 1=o_custkey
    //   The orderdate predicate is already applied by the date bitmap, so we
    //   do NOT filter on o_orderdate here.  Flat vector is much faster than
    //   unordered_map for the per-row hot lookup in the iteration loop.
    //   Entry 0 is unused (TPC-H orderkeys are 1-based); 0 in a slot means
    //   "no order at that orderkey" (HPRB gaps).
    std::vector<int64_t> orders_custkey;
    int64_t max_orderkey = 0;
    int64_t orders_rows  = 0;
    {
        auto t0 = clock::now();

        auto &tbl = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "orders");
        auto &txn = DuckTransaction::Get(context.client, tbl.catalog);

        // Pass 1: discover max_orderkey via a tiny SELECT max(o_orderkey) query.
        // (An explicit scan would cost the same as the main load; SQL keeps this
        // setup code compact and uses DuckDB's own min/max metadata.)
        {
            Connection con(*context.client.db);
            auto r = con.Query("SELECT max(o_orderkey) FROM orders");
            if (r && !r->HasError() && r->RowCount() == 1) {
                max_orderkey = r->GetValue(0, 0).GetValue<int64_t>();
            }
        }
        if (max_orderkey <= 0) {
            std::cerr << "Error: max(o_orderkey) query failed — cannot size orders_custkey" << std::endl;
            return;
        }
        orders_custkey.assign(max_orderkey + 1, 0);

        // Pass 2: full scan, populate orders_custkey[ok] = ck.
        TableScanState st;
        vector<StorageIndex> cols{ StorageIndex(0), StorageIndex(1) };
        tbl.GetStorage().InitializeScan(context.client, txn, st, cols);
        vector<LogicalType> types{
            tbl.GetColumns().GetColumnTypes()[0],
            tbl.GetColumns().GetColumnTypes()[1] };
        while (true) {
            DataChunk ch;
            ch.Initialize(context.client, types);
            tbl.GetStorage().Scan(txn, ch, st);
            if (ch.size() == 0) break;
            auto ok = FlatVector::GetData<int64_t>(ch.data[0]);
            auto ck = FlatVector::GetData<int64_t>(ch.data[1]);
            for (idx_t i = 0; i < ch.size(); i++) orders_custkey[ok[i]] = ck[i];
            orders_rows += ch.size();
        }
        std::cout << "[Side-load] orders:   " << orders_rows
                  << " rows (max_orderkey=" << max_orderkey << "; flat vector "
                  << (orders_custkey.size() * sizeof(int64_t)) / (1024ULL * 1024ULL)
                  << " MiB) in " << q10_ms(t0, clock::now()) << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 4. Pre-load lineitem (l_orderkey, l_extendedprice, l_discount)
    // -----------------------------------------------------------------------
    //   lineitem cols: 0 = l_orderkey (int64), 5 = l_extendedprice (DEC15,2 ×100),
    //                  6 = l_discount (DEC15,2 ×100)
    //   Loaded once into flat int64 arrays indexed by row id; the iteration
    //   loop then operates entirely on these arrays, matching Q5.
    size_t num_lineitem_rows = 0;
    {
        std::ifstream meta(Q10_Q5_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_lineitem_rows = std::stoull(line.substr(9));
        }
    }
    if (num_lineitem_rows == 0) {
        std::cerr << "Error: could not read num_rows from "
                  << Q10_Q5_CB_DIR << "/done.txt — aborting Q10 benchmark."
                  << std::endl;
        return;
    }

    std::vector<int64_t> col_ok(num_lineitem_rows);
    std::vector<int64_t> col_price(num_lineitem_rows);
    std::vector<int64_t> col_disc(num_lineitem_rows);
    {
        auto t0 = clock::now();
        auto &tbl = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");
        auto &txn = DuckTransaction::Get(context.client, tbl.catalog);
        TableScanState st;
        vector<StorageIndex> cols{ StorageIndex(0), StorageIndex(5), StorageIndex(6) };
        tbl.GetStorage().InitializeScan(context.client, txn, st, cols);
        vector<LogicalType> types{
            tbl.GetColumns().GetColumnTypes()[0],
            tbl.GetColumns().GetColumnTypes()[5],
            tbl.GetColumns().GetColumnTypes()[6] };

        size_t row_offset = 0;
        while (true) {
            DataChunk ch;
            ch.Initialize(context.client, types);
            tbl.GetStorage().Scan(txn, ch, st);
            if (ch.size() == 0) break;
            auto ok = FlatVector::GetData<int64_t>(ch.data[0]);
            auto pp = FlatVector::GetData<int64_t>(ch.data[1]);
            auto dd = FlatVector::GetData<int64_t>(ch.data[2]);
            std::memcpy(col_ok.data()    + row_offset, ok, ch.size() * sizeof(int64_t));
            std::memcpy(col_price.data() + row_offset, pp, ch.size() * sizeof(int64_t));
            std::memcpy(col_disc.data()  + row_offset, dd, ch.size() * sizeof(int64_t));
            row_offset += ch.size();
        }
        std::cout << "[Pre-load] lineitem:  " << row_offset
                  << " rows (l_orderkey, l_extendedprice, l_discount) in "
                  << q10_ms(t0, clock::now()) << " ms" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 4.5 Build orderdate bitmaps dynamically (TPC-H 1.5.7).
    //     Per-lineitem orderdate is multi-table (lineitem⋈orders) so we
    //     can no longer load it from disk.  SQL JOIN at startup, bucket
    //     into per-day position lists, encode for each backend.
    // -----------------------------------------------------------------------
    std::vector<std::vector<uint32_t>> q10_date_pos(Q10_DATE_END + 1);
    double q10_join_setup_ms = 0;
    {
        auto t0 = clock::now();
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT (o.o_orderdate - DATE '1992-01-01')::INT "
            "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
            "ORDER BY l.rowid");
        if (r && !r->HasError()) {
            size_t pos = 0;
            while (auto chunk = r->Fetch()) {
                auto day = FlatVector::GetData<int32_t>(chunk->data[0]);
                for (idx_t j = 0; j < chunk->size(); j++, pos++) {
                    int d = day[j];
                    if (d >= Q10_DATE_START && d <= Q10_DATE_END)
                        q10_date_pos[d].push_back(static_cast<uint32_t>(pos));
                }
            }
            q10_join_setup_ms = q10_ms(t0, clock::now());
            std::cout << "[Build Q10 orderdate] " << pos << " lineitems scanned in "
                      << q10_join_setup_ms << " ms (counted in Total)" << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // 5. Load bitmaps (per-backend gated + timed)
    // -----------------------------------------------------------------------
    std::cout << "\n[Load] Loading bitmaps (mode=" << q10_bm_label() << ")..." << std::endl;

    const int n_dates = Q10_DATE_END - Q10_DATE_START + 1;

    // ComBit
    std::vector<ComBit> cb_date;
    ComBit cb_rf;
    std::vector<const ComBit*> cb_date_ptrs;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = clock::now();
        cb_date.resize(Q10_DATE_END + 1);
        std::vector<bool> mask(num_lineitem_rows, false);
        std::vector<uint32_t>* prev = nullptr;
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++) {
            if (prev) for (uint32_t p : *prev) mask[p] = false;
            for (uint32_t p : q10_date_pos[d]) mask[p] = true;
            cb_date[d] = ComBit::compress(mask, false);
            prev = &q10_date_pos[d];
        }
        cb_rf = q10_load_cb(Q10_Q1_CB_DIR + "/returnflag/" + std::to_string(Q10_RF_R_IDX) + ".bm");

        cb_date_ptrs.reserve(n_dates);
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++) cb_date_ptrs.push_back(&cb_date[d]);
        cb_load_ms = q10_ms(t0, clock::now());
    }

    // CRoaring (unoptimized)
    std::vector<roaring::Roaring> cr_date;
    roaring::Roaring cr_rf;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = clock::now();
        cr_date.resize(Q10_DATE_END + 1);
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++) {
            auto& pos = q10_date_pos[d];
            if (!pos.empty()) cr_date[d].addMany(pos.size(), pos.data());
        }
        cr_rf = q10_load_cr(Q10_Q1_CR_DIR + "/returnflag/" + std::to_string(Q10_RF_R_IDX) + ".bm");
        cr_load_ms = q10_ms(t0, clock::now());
    }

    // CRoaring+Run (fresh load + runOptimize so this backend stands alone)
    std::vector<roaring::Roaring> crr_date;
    roaring::Roaring crr_rf;
    std::vector<const roaring::Roaring*> crr_date_ptrs;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = clock::now();
        crr_date.resize(Q10_DATE_END + 1);
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++) {
            auto& pos = q10_date_pos[d];
            if (!pos.empty()) crr_date[d].addMany(pos.size(), pos.data());
            crr_date[d].runOptimize();
        }
        crr_rf = q10_load_cr(Q10_Q1_CR_DIR + "/returnflag/" + std::to_string(Q10_RF_R_IDX) + ".bm");
        crr_rf.runOptimize();

        crr_date_ptrs.reserve(n_dates);
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++) crr_date_ptrs.push_back(&crr_date[d]);
        crr_load_ms = q10_ms(t0, clock::now());
    }

    // WAH
    std::vector<ibis::bitvector> wah_date;
    ibis::bitvector wah_rf;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = clock::now();
        wah_date.resize(Q10_DATE_END + 1);
        auto wah_build = [&](ibis::bitvector& w, const std::vector<uint32_t>& pos) {
            size_t k = 0;
            for (size_t i = 0; i < num_lineitem_rows; i++) {
                bool b = (k < pos.size() && pos[k] == i);
                if (b) k++;
                w += (b ? 1 : 0);
            }
            w.compress();
        };
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++)
            wah_build(wah_date[d], q10_date_pos[d]);
        wah_rf = q10_load_wah(Q10_Q1_WAH_DIR + "/returnflag/" + std::to_string(Q10_RF_R_IDX) + ".bm");
        wah_load_ms = q10_ms(t0, clock::now());
    }

    // EWAH.  Pre-build date pointer array so OR phase can use
    // ewah::fast_logicalor (priority-queue k-way merge) — matches
    // CRR fastunion / Concise fast_logicalor.
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_date;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_date_ptrs;
    ewah::EWAHBoolArray<uint64_t> ew_rf;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = clock::now();
        ew_date.resize(Q10_DATE_END + 1);
        auto ew_build = [&](ewah::EWAHBoolArray<uint64_t>& e, const std::vector<uint32_t>& pos) {
            for (uint32_t p : pos) e.set(p);
            if (e.sizeInBits() < num_lineitem_rows) e.padWithZeroes(num_lineitem_rows);
        };
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++)
            ew_build(ew_date[d], q10_date_pos[d]);
        ew_rf = q10_load_ew(Q10_Q1_EW_DIR + "/returnflag/" + std::to_string(Q10_RF_R_IDX) + ".bm");
        ew_date_ptrs.reserve(Q10_DATE_END - Q10_DATE_START + 1);
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++)
            ew_date_ptrs.push_back(&ew_date[d]);
        ew_load_ms = q10_ms(t0, clock::now());
    }

    // Bitset (uncompressed; shared by BS / BSA)
    std::vector<bs::Bitmap> bs_date;
    bs::Bitmap bs_rf;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = clock::now();
        bs_date.resize(Q10_DATE_END + 1);
        auto bs_build = [&](bs::Bitmap& b, const std::vector<uint32_t>& pos) {
            b.alloc_for_bits(num_lineitem_rows);
            for (uint32_t p : pos) b.words[p / 64] |= uint64_t(1) << (p % 64);
        };
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++)
            bs_build(bs_date[d], q10_date_pos[d]);
        bs_rf = bm_bench::load_bitmap_from_croaring(Q10_Q1_CR_DIR + "/returnflag/" + std::to_string(Q10_RF_R_IDX) + ".bm");
        bs_load_ms = q10_ms(t0, clock::now());
    }

    // Concise
    std::vector<ConciseSet<false>> con_date;
    ConciseSet<false> con_rf;
    std::vector<const ConciseSet<false>*> con_date_ptrs;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = clock::now();
        con_date.resize(Q10_DATE_END + 1);
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++)
            for (uint32_t p : q10_date_pos[d]) con_date[d].add(p);
        con_rf = bm_bench::load_concise_from_croaring(Q10_Q1_CR_DIR + "/returnflag/" + std::to_string(Q10_RF_R_IDX) + ".bm");
        con_date_ptrs.reserve(n_dates);
        for (int d = Q10_DATE_START; d <= Q10_DATE_END; d++) con_date_ptrs.push_back(&con_date[d]);
        con_load_ms = q10_ms(t0, clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:      " << wah_load_ms << " ms" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:   " << cb_load_ms  << " ms" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load: " << cr_load_ms  << " ms" << std::endl;
    if (run_crr()) std::cout << "  CRR load:      " << crr_load_ms << " ms" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:     " << ew_load_ms  << " ms" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:   " << bs_load_ms  << " ms (shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:  " << con_load_ms << " ms" << std::endl;

    // On-disk footprint of the shared directories (orderdate + returnflag).
    std::cout << std::fixed << std::setprecision(2);
    if (run_wah())
        std::cout << "  WAH      on disk: "
                  << bm_bench::mib(bm_bench::dir_size_bytes(Q10_Q5_WAH_DIR)
                                 + bm_bench::dir_size_bytes(Q10_Q1_WAH_DIR))
                  << " MiB (q5+q1)" << std::endl;
    if (run_cb())
        std::cout << "  ComBit   on disk: "
                  << bm_bench::mib(bm_bench::dir_size_bytes(Q10_Q5_CB_DIR)
                                 + bm_bench::dir_size_bytes(Q10_Q1_CB_DIR))
                  << " MiB (q5+q1)" << std::endl;
    if (run_cr() || run_crr())
        std::cout << "  CRoaring on disk: "
                  << bm_bench::mib(bm_bench::dir_size_bytes(Q10_Q5_CR_DIR)
                                 + bm_bench::dir_size_bytes(Q10_Q1_CR_DIR))
                  << " MiB (q5+q1, shared by CR / CRR)" << std::endl;
    if (run_ew())
        std::cout << "  EWAH     on disk: "
                  << bm_bench::mib(bm_bench::dir_size_bytes(Q10_Q5_EW_DIR)
                                 + bm_bench::dir_size_bytes(Q10_Q1_EW_DIR))
                  << " MiB (q5+q1)" << std::endl;
    if (run_bs() || run_bsa()) {
        size_t b_bytes = 0;
        for (auto& b : bs_date) b_bytes += b.nwords * sizeof(uint64_t);
        b_bytes += bs_rf.nwords * sizeof(uint64_t);
        std::cout << "  Bitset in mem:    " << bm_bench::mib(b_bytes) << " MiB (rebuilt from CRoaring at load)" << std::endl;
    }
    if (run_con()) {
        size_t c_bytes = con_rf.sizeInBytes();
        for (auto& c : con_date) c_bytes += c.sizeInBytes();
        std::cout << "  Concise in mem:   " << bm_bench::mib(c_bytes) << " MiB (rebuilt from CRoaring at load)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 6. Per-iteration buffers (reused across iterations to avoid alloc noise)
    // -----------------------------------------------------------------------
    // revenue_by_custkey is a flat aggregation buffer indexed by c_custkey.
    // Size = max_custkey + 1 (8 bytes each).  For SF10 (~1.5 M customers)
    // this is ~12 MiB; reset via std::fill at the start of each iteration.
    const int64_t revenue_buf_size = max_custkey + 1;
    std::vector<int64_t> revenue_by_custkey(revenue_buf_size, 0);

    // -----------------------------------------------------------------------
    // 7. Per-backend phase timings + final Top-20 snapshots
    // -----------------------------------------------------------------------
    std::vector<double> cb_or_times,  cb_and_times,  cb_topk_times,  cb_total_times;
    std::vector<double> cr_or_times,  cr_and_times,  cr_topk_times,  cr_total_times;
    std::vector<double> crr_or_times, crr_and_times, crr_topk_times, crr_total_times;
    std::vector<double> wah_or_times, wah_and_times, wah_topk_times, wah_total_times;
    std::vector<double> ew_or_times,  ew_and_times,  ew_topk_times,  ew_total_times;
    std::vector<double> bs_or_times,  bs_and_times,  bs_topk_times,  bs_total_times;
    std::vector<double> bsa_or_times, bsa_and_times, bsa_topk_times, bsa_total_times;
    std::vector<double> con_or_times, con_and_times, con_topk_times, con_total_times;

    std::vector<Q10Row> cb_top, cr_top, crr_top, wah_top, ew_top;
    std::vector<Q10Row> bs_top, bsa_top, con_top;
    size_t cb_rows = 0, cr_rows = 0, crr_rows = 0, wah_rows = 0, ew_rows = 0;
    size_t bs_rows = 0, bsa_rows = 0, con_rows = 0;

    // Helper: flat scan of revenue_by_custkey → Top-K min-heap → sorted DESC.
    // Lives outside the measured OR/AND phases; TopK time is reported separately.
    auto extract_topk = [&](std::vector<int64_t>& rev_buf,
                            std::vector<Q10Row>& out) {
        std::priority_queue<Q10Row, std::vector<Q10Row>, Q10MinHeapCmp> heap;
        for (int64_t ck = 1; ck < revenue_buf_size; ck++) {
            int64_t rev = rev_buf[ck];
            if (rev == 0) continue;
            if ((int)heap.size() < Q10_TOPK) {
                heap.push({ck, rev});
            } else {
                // Replace heap top iff the new row ranks higher under the
                // same tie-break the SQL ground truth uses
                //   (revenue DESC, c_custkey ASC).
                const auto& top = heap.top();
                if (rev > top.revenue_fp ||
                    (rev == top.revenue_fp && ck < top.c_custkey)) {
                    heap.pop();
                    heap.push({ck, rev});
                }
            }
        }
        out.clear();
        out.reserve(heap.size());
        while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
        // Final stable sort: revenue DESC, c_custkey ASC (matches ground-truth).
        std::sort(out.begin(), out.end(), [](const Q10Row& a, const Q10Row& b) {
            if (a.revenue_fp != b.revenue_fp) return a.revenue_fp > b.revenue_fp;
            return a.c_custkey < b.c_custkey;
        });
    };

    // -----------------------------------------------------------------------
    // 8. Benchmark loop
    // -----------------------------------------------------------------------
    for (int iter = 0; iter < Q10_ITERATIONS; iter++) {
        bool warmup = (iter < Q10_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q10_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ================= ComBit =================
        // Path mirrors the other four backends for a fair, apples-to-apples
        // comparison: OR_many -> explicit AND -> iterate the sparse result.
        //
        //   1. OR_many(92 dates) returns a Decompressed ComBit (cb_or).
        //   2. cb_or &= cb_rf uses ComBit's AVX-512 VBMI2 operator&=
        //      (per-region L3 bypass, mask_expandloadu on RHS literals).
        //      The LHS stays Decompressed, with l1_literals_ holding the
        //      per-row AND result.
        //   3. cb_or.for_each_literal delivers (word_pos, val) for every
        //      non-zero result byte; the byte-LUT scatters its set bits
        //      directly into the revenue aggregation buffer.  The library
        //      picks its AVX-512 walk strategy based on the segment's
        //      state — no caller-side compaction is needed.
        //
        // Note: an alternative "fused" formulation ran for_each_literal on
        // the dense cb_rf (~25% density) and AND-ed with a flat cb_or
        // buffer inside the callback.  That pattern is optimal when the
        // walked side is sparse (Q5's 4%-density per-nation bitmaps), but
        // produces an excessive callback count for Q10's denser
        // returnflag='R' bitmap.  Doing the AND up-front keeps the
        // post-AND iteration at the true qualifying-row count and removes
        // a ComBit-specific code path from the benchmark comparison.
        if (run_cb()) {
            auto t0 = clock::now();

            ComBit cb_or = ComBit::OR_many(cb_date_ptrs.size(), cb_date_ptrs.data());
            auto t1 = clock::now();

            cb_or &= cb_rf;

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ok    = col_ok.data();
            const int64_t* pp    = col_price.data();
            const int64_t* dp    = col_disc.data();
            const int64_t* ocust = orders_custkey.data();
            int64_t*       rev   = revenue_by_custkey.data();

            cb_or.for_each_literal(
                [&](uint32_t word_pos, uint8_t val) {
                    size_t rbase = static_cast<size_t>(word_pos) * 8;
                    const auto& e = bm_bench::byte_lut[val];
                    for (int k = 0; k < e.count; k++) {
                        size_t row = rbase + e.pos[k];
                        int64_t custkey = ocust[ok[row]];
                        rev[custkey] += pp[row] * (100 - dp[row]);
                        cnt++;
                    }
                });
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            cb_top  = top;
            cb_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  CB:   OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                cb_or_times.push_back(d_or);
                cb_and_times.push_back(d_and);
                cb_topk_times.push_back(d_tk);
                cb_total_times.push_back(d_tot);
            }
        }

        // ================= CRoaring (unoptimized) =================
        if (run_cr()) {
            auto t0 = clock::now();

            roaring::Roaring cr_filt = cr_date[Q10_DATE_START];
            for (int d = Q10_DATE_START + 1; d <= Q10_DATE_END; d++)
                cr_filt |= cr_date[d];
            auto t1 = clock::now();

            cr_filt &= cr_rf;

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ocust = orders_custkey.data();
            for (auto it = cr_filt.begin(); it != cr_filt.end(); ++it) {
                size_t row = *it;
                int64_t custkey = ocust[col_ok[row]];
                revenue_by_custkey[custkey] += col_price[row] * (100 - col_disc[row]);
                cnt++;
            }
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            cr_top  = top;
            cr_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  CR:   OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                cr_or_times.push_back(d_or);
                cr_and_times.push_back(d_and);
                cr_topk_times.push_back(d_tk);
                cr_total_times.push_back(d_tot);
            }
        }

        // ================= CRoaring+Run (fastunion) =================
        if (run_crr()) {
            auto t0 = clock::now();

            roaring::Roaring crr_filt = roaring::Roaring::fastunion(
                crr_date_ptrs.size(), crr_date_ptrs.data());
            auto t1 = clock::now();

            crr_filt &= crr_rf;

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ocust = orders_custkey.data();
            for (auto it = crr_filt.begin(); it != crr_filt.end(); ++it) {
                size_t row = *it;
                int64_t custkey = ocust[col_ok[row]];
                revenue_by_custkey[custkey] += col_price[row] * (100 - col_disc[row]);
                cnt++;
            }
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            crr_top  = top;
            crr_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  CRR:  OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                crr_or_times.push_back(d_or);
                crr_and_times.push_back(d_and);
                crr_topk_times.push_back(d_tk);
                crr_total_times.push_back(d_tot);
            }
        }

        // ================= WAH =================
        if (run_wah()) {
            auto t0 = clock::now();

            ibis::bitvector wah_filt = wah_date[Q10_DATE_START];
            wah_filt.decompress();
            for (int d = Q10_DATE_START + 1; d <= Q10_DATE_END; d++)
                wah_filt |= wah_date[d];
            auto t1 = clock::now();

            wah_filt &= wah_rf;

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ocust = orders_custkey.data();
            ibis::bitvector::pit pit(wah_filt);
            while (*pit != 0xFFFFFFFFU) {
                size_t row = *pit;
                int64_t custkey = ocust[col_ok[row]];
                revenue_by_custkey[custkey] += col_price[row] * (100 - col_disc[row]);
                cnt++;
                pit.next();
            }
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            wah_top  = top;
            wah_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  WAH:  OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                wah_or_times.push_back(d_or);
                wah_and_times.push_back(d_and);
                wah_topk_times.push_back(d_tk);
                wah_total_times.push_back(d_tot);
            }
        }

        // ================= EWAH =================
        if (run_ew()) {
            // fast_logicalor: priority-queue k-way merge (ewah-inl.h:1129).
            auto t0 = clock::now();

            ewah::EWAHBoolArray<uint64_t> ew_filt = ewah::fast_logicalor(
                ew_date_ptrs.size(), ew_date_ptrs.data());
            auto t1 = clock::now();

            ewah::EWAHBoolArray<uint64_t> ew_final;
            ew_filt.logicaland(ew_rf, ew_final);

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ocust = orders_custkey.data();
            for (auto it = ew_final.begin(); it != ew_final.end(); ++it) {
                size_t row = *it;
                int64_t custkey = ocust[col_ok[row]];
                revenue_by_custkey[custkey] += col_price[row] * (100 - col_disc[row]);
                cnt++;
            }
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            ew_top  = top;
            ew_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  EW:   OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                ew_or_times.push_back(d_or);
                ew_and_times.push_back(d_and);
                ew_topk_times.push_back(d_tk);
                ew_total_times.push_back(d_tot);
            }
        }

        // ================= Bitset (scalar) =================
        // Pure baseline: clone first orderdate, scalar-OR the rest in,
        // scalar-AND with returnflag='R', then walk set bits via ctz to
        // aggregate revenue per c_custkey.  TopK is the shared lambda.
        if (run_bs()) {
            auto t0 = clock::now();

            bs::Bitmap bs_filt = bs_date[Q10_DATE_START].clone();
            for (int d = Q10_DATE_START + 1; d <= Q10_DATE_END; d++)
                bs::or_inplace(bs_filt, bs_date[d], false);
            auto t1 = clock::now();

            bs::and_inplace(bs_filt, bs_rf, false);

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ocust = orders_custkey.data();
            for (size_t i = 0; i < bs_filt.nwords; ++i) {
                uint64_t w = bs_filt.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t row = base + __builtin_ctzll(w);
                    if (row >= bs_filt.nbits) break;
                    int64_t custkey = ocust[col_ok[row]];
                    revenue_by_custkey[custkey] += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                    w &= w - 1;
                }
            }
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            bs_top  = top;
            bs_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  BS:   OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                bs_or_times.push_back(d_or);
                bs_and_times.push_back(d_and);
                bs_topk_times.push_back(d_tk);
                bs_total_times.push_back(d_tot);
            }
        }

        // ================= Bitset + AVX-512 =================
        // Same data as BS; simd flag flipped on for OR / AND.  Aggregation
        // walk is identical (ctz over result words).
        if (run_bsa()) {
            auto t0 = clock::now();

            bs::Bitmap bsa_filt = bs_date[Q10_DATE_START].clone();
            for (int d = Q10_DATE_START + 1; d <= Q10_DATE_END; d++)
                bs::or_inplace(bsa_filt, bs_date[d], true);
            auto t1 = clock::now();

            bs::and_inplace(bsa_filt, bs_rf, true);

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ocust = orders_custkey.data();
            for (size_t i = 0; i < bsa_filt.nwords; ++i) {
                uint64_t w = bsa_filt.words[i];
                const size_t base = i * 64;
                while (w) {
                    size_t row = base + __builtin_ctzll(w);
                    if (row >= bsa_filt.nbits) break;
                    int64_t custkey = ocust[col_ok[row]];
                    revenue_by_custkey[custkey] += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                    w &= w - 1;
                }
            }
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            bsa_top  = top;
            bsa_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  BSA:  OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                bsa_or_times.push_back(d_or);
                bsa_and_times.push_back(d_and);
                bsa_topk_times.push_back(d_tk);
                bsa_total_times.push_back(d_tot);
            }
        }

        // ================= Concise =================
        // OR via fast_logicalor (priority-queue k-way merge), AND via
        // logicaland (returns fresh container — Concise's *ToContainer
        // variants don't reset their `res` first).
        if (run_con()) {
            auto t0 = clock::now();

            ConciseSet<false> con_filt = ConciseSet<false>::fast_logicalor(
                con_date_ptrs.size(), con_date_ptrs.data());
            auto t1 = clock::now();

            ConciseSet<false> con_final = con_filt.logicaland(con_rf);

            std::fill(revenue_by_custkey.begin(), revenue_by_custkey.end(), 0);
            size_t cnt = 0;
            const int64_t* ocust = orders_custkey.data();
            for (auto it = con_final.begin(); it != con_final.end(); ++it) {
                size_t row = *it;
                int64_t custkey = ocust[col_ok[row]];
                revenue_by_custkey[custkey] += col_price[row] * (100 - col_disc[row]);
                cnt++;
            }
            auto t2 = clock::now();

            std::vector<Q10Row> top;
            extract_topk(revenue_by_custkey, top);
            auto t3 = clock::now();

            con_top  = top;
            con_rows = cnt;

            double d_or = q10_ms(t0,t1), d_and = q10_ms(t1,t2),
                   d_tk = q10_ms(t2,t3), d_tot = q10_ms(t0,t3) + q10_join_setup_ms;
            std::cout << "  CON:  OR=" << d_or << "  AND+Agg=" << d_and
                      << "  TopK=" << d_tk  << "  Total=" << d_tot
                      << "  rows=" << cnt << std::endl;
            if (!warmup) {
                con_or_times.push_back(d_or);
                con_and_times.push_back(d_and);
                con_topk_times.push_back(d_tk);
                con_total_times.push_back(d_tot);
            }
        }
    } // end iterations

    // -----------------------------------------------------------------------
    // 9. DuckDB native SQL ground truth — run AFTER iterations to avoid
    //    polluting CPU cache mid-measurement.
    // -----------------------------------------------------------------------
    //   The SQL below is the verbatim spec §2.4.10 query with DATE =
    //   1993-10-01.  Ordering is deterministic: revenue DESC, then
    //   c_custkey ASC (the spec's ORDER BY is underspecified for ties, so
    //   we pin the tie-break here to match our Top-K extractor).
    struct Q10GroundTruth {
        int64_t     c_custkey;
        std::string c_name;
        double      revenue;
        double      c_acctbal;
        std::string n_name;
        std::string c_address;
        std::string c_phone;
        std::string c_comment;
    };
    std::vector<Q10GroundTruth> gt;
    double gt_sql_ms = 0.0;
    bool   gt_ok     = false;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT c_custkey, c_name, "
            "       CAST(sum(l_extendedprice * (1 - l_discount)) AS DOUBLE) AS revenue, "
            "       CAST(c_acctbal AS DOUBLE) AS c_acctbal, "
            "       n_name, c_address, c_phone, c_comment "
            "FROM customer, orders, lineitem, nation "
            "WHERE c_custkey = o_custkey "
            "  AND l_orderkey = o_orderkey "
            "  AND o_orderdate >= DATE '1993-10-01' "
            "  AND o_orderdate <  DATE '1994-01-01' "
            "  AND l_returnflag = 'R' "
            "  AND c_nationkey = n_nationkey "
            "GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
            "ORDER BY revenue DESC, c_custkey ASC "
            "LIMIT " + std::to_string(Q10_TOPK);
        auto t0 = clock::now();
        auto result = con.Query(sql);
        auto t1 = clock::now();
        if (result && !result->HasError()) {
            gt_sql_ms = q10_ms(t0, t1);
            gt.reserve(result->RowCount());
            for (idx_t r = 0; r < result->RowCount(); r++) {
                Q10GroundTruth g;
                g.c_custkey = result->GetValue(0, r).GetValue<int64_t>();
                g.c_name    = result->GetValue(1, r).ToString();
                g.revenue   = result->GetValue(2, r).GetValue<double>();
                g.c_acctbal = result->GetValue(3, r).GetValue<double>();
                g.n_name    = result->GetValue(4, r).ToString();
                g.c_address = result->GetValue(5, r).ToString();
                g.c_phone   = result->GetValue(6, r).ToString();
                g.c_comment = result->GetValue(7, r).ToString();
                gt.push_back(g);
            }
            gt_ok = true;
        } else if (result && result->HasError()) {
            std::cerr << "[Baseline] SQL error: " << result->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }

    std::cout << std::fixed << std::setprecision(2);
    if (gt_ok) {
        std::cout << "\n[Baseline] DuckDB native SQL Top-" << gt.size()
                  << "  (single run: " << gt_sql_ms << " ms)" << std::endl;
    } else {
        std::cout << "\n[Baseline] DuckDB SQL ground truth unavailable — skipping assert."
                  << std::endl;
    }

    // -----------------------------------------------------------------------
    // 10. Correctness: compare each active backend's Top-K against SQL
    // -----------------------------------------------------------------------
    //   We compare on (c_custkey, revenue) since the other six columns
    //   are pure customer-table lookups that cannot differ across bitmap
    //   backends (no bitmap code touches those fields).  A tolerance of
    //   0.01 on revenue accounts for DECIMAL(15,2) × DECIMAL(15,2) →
    //   DECIMAL(30,4) rounding differences between our integer
    //   accumulator and DuckDB's decimal pipeline.
    auto check_topk = [&](const char* label, bool active,
                          const std::vector<Q10Row>& our) {
        if (!active || !gt_ok) return;
        if (our.size() != gt.size()) {
            std::ostringstream oss;
            oss << "[FAIL] Q10 " << label << " produced " << our.size()
                << " rows, SQL ground truth has " << gt.size()
                << " — bitmap pipeline is incorrect";
            throw std::runtime_error(oss.str());
        }
        for (size_t i = 0; i < our.size(); i++) {
            double our_rev = static_cast<double>(our[i].revenue_fp) / 10000.0;
            double dr      = std::fabs(our_rev - gt[i].revenue);
            if (our[i].c_custkey != gt[i].c_custkey || dr > 0.01) {
                std::ostringstream oss;
                oss << "[FAIL] Q10 " << label << " row " << i
                    << ": got (c_custkey=" << our[i].c_custkey
                    << ", revenue=" << our_rev << ")"
                    << " vs SQL (c_custkey=" << gt[i].c_custkey
                    << ", revenue=" << gt[i].revenue << ")"
                    << " — bitmap pipeline is incorrect";
                throw std::runtime_error(oss.str());
            }
        }
    };
    if (gt_ok) {
        check_topk("WAH",           run_wah(), wah_top);
        check_topk("ComBit",        run_cb(),  cb_top);
        check_topk("CRoaring",      run_cr(),  cr_top);
        check_topk("CRoaring+Run",  run_crr(), crr_top);
        check_topk("EWAH",          run_ew(),  ew_top);
        check_topk("Bitset",        run_bs(),  bs_top);
        check_topk("Bitset+AVX512", run_bsa(), bsa_top);
        check_topk("Concise",       run_con(), con_top);
        std::cout << "[OK] all active backends match DuckDB SQL ground truth (Top-"
                  << gt.size() << ", revenue within 0.01)." << std::endl;
    }

    // -----------------------------------------------------------------------
    // 11. Printable Top-20 (from whichever backend is considered canonical)
    // -----------------------------------------------------------------------
    const std::vector<Q10Row>* canonical = nullptr;
    const char* canonical_label = "";
    if      (run_cb())   { canonical = &cb_top;  canonical_label = "ComBit"; }
    else if (run_cr())   { canonical = &cr_top;  canonical_label = "CRoaring"; }
    else if (run_crr())  { canonical = &crr_top; canonical_label = "CRoaring+Run"; }
    else if (run_wah())  { canonical = &wah_top; canonical_label = "WAH"; }
    else if (run_ew())   { canonical = &ew_top;  canonical_label = "EWAH"; }
    else if (run_bs())   { canonical = &bs_top;  canonical_label = "Bitset"; }
    else if (run_bsa())  { canonical = &bsa_top; canonical_label = "Bitset+AVX512"; }
    else if (run_con())  { canonical = &con_top; canonical_label = "Concise"; }

    if (canonical && !canonical->empty()) {
        std::cout << "\n  Q10 Top-" << Q10_TOPK
                  << " (revenue = SUM(l_extendedprice * (1 - l_discount)), sorted DESC, source=" << canonical_label << "):"
                  << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  " << std::left << std::setw(10) << "c_custkey"
                  << std::setw(22) << "c_name"
                  << std::right << std::setw(16) << "revenue"
                  << std::setw(14) << "c_acctbal"
                  << "  " << std::left << std::setw(16) << "n_name"
                  << std::setw(42) << "c_address"
                  << std::setw(17) << "c_phone"
                  << "  c_comment"
                  << std::endl;
        for (const auto& r : *canonical) {
            auto it = customer_info.find(r.c_custkey);
            const Q10Customer* c = (it == customer_info.end()) ? nullptr : &it->second;
            std::string nn = (c && nation_name.count(c->c_nationkey))
                             ? nation_name[c->c_nationkey] : "?";
            std::cout << "  " << std::left << std::setw(10) << r.c_custkey
                      << std::setw(22) << (c ? c->c_name : std::string("?"))
                      << std::right << std::setw(16)
                      << (static_cast<double>(r.revenue_fp) / 10000.0)
                      << std::setw(14)
                      << (c ? static_cast<double>(c->c_acctbal_fp) / 100.0 : 0.0)
                      << "  " << std::left << std::setw(16) << nn
                      << std::setw(42) << (c ? c->c_address : std::string("?"))
                      << std::setw(17) << (c ? c->c_phone   : std::string("?"))
                      << "  "          << (c ? c->c_comment : std::string("?"))
                      << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // 12. Statistics summary — same table layout as Q5
    // -----------------------------------------------------------------------
    auto cb_or_s   = bm_bench::compute_stats(cb_or_times);
    auto cb_and_s  = bm_bench::compute_stats(cb_and_times);
    auto cb_tk_s   = bm_bench::compute_stats(cb_topk_times);
    auto cb_tot_s  = bm_bench::compute_stats(cb_total_times);

    auto cr_or_s   = bm_bench::compute_stats(cr_or_times);
    auto cr_and_s  = bm_bench::compute_stats(cr_and_times);
    auto cr_tk_s   = bm_bench::compute_stats(cr_topk_times);
    auto cr_tot_s  = bm_bench::compute_stats(cr_total_times);

    auto crr_or_s  = bm_bench::compute_stats(crr_or_times);
    auto crr_and_s = bm_bench::compute_stats(crr_and_times);
    auto crr_tk_s  = bm_bench::compute_stats(crr_topk_times);
    auto crr_tot_s = bm_bench::compute_stats(crr_total_times);

    auto wah_or_s  = bm_bench::compute_stats(wah_or_times);
    auto wah_and_s = bm_bench::compute_stats(wah_and_times);
    auto wah_tk_s  = bm_bench::compute_stats(wah_topk_times);
    auto wah_tot_s = bm_bench::compute_stats(wah_total_times);

    auto ew_or_s   = bm_bench::compute_stats(ew_or_times);
    auto ew_and_s  = bm_bench::compute_stats(ew_and_times);
    auto ew_tk_s   = bm_bench::compute_stats(ew_topk_times);
    auto ew_tot_s  = bm_bench::compute_stats(ew_total_times);

    auto bs_or_s   = bm_bench::compute_stats(bs_or_times);
    auto bs_and_s  = bm_bench::compute_stats(bs_and_times);
    auto bs_tk_s   = bm_bench::compute_stats(bs_topk_times);
    auto bs_tot_s  = bm_bench::compute_stats(bs_total_times);

    auto bsa_or_s  = bm_bench::compute_stats(bsa_or_times);
    auto bsa_and_s = bm_bench::compute_stats(bsa_and_times);
    auto bsa_tk_s  = bm_bench::compute_stats(bsa_topk_times);
    auto bsa_tot_s = bm_bench::compute_stats(bsa_total_times);

    auto con_or_s  = bm_bench::compute_stats(con_or_times);
    auto con_and_s = bm_bench::compute_stats(con_and_times);
    auto con_tk_s  = bm_bench::compute_stats(con_topk_times);
    auto con_tot_s = bm_bench::compute_stats(con_total_times);

    int measured = Q10_ITERATIONS - Q10_WARMUP;

    if (run_all()) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q10 RESULTS (" << measured << " measured iterations, median +/- stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  CB (ms)         CR (ms)        CRR (ms)        WAH (ms)        EW (ms)" << std::endl;
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;

        auto print_row = [](const char* label, bm_bench::Stats& cb, bm_bench::Stats& cr,
                            bm_bench::Stats& crr, bm_bench::Stats& wah, bm_bench::Stats& ew) {
            std::cout << "  " << std::left << std::setw(14) << label
                      << std::right
                      << std::setw(8) << cb.median  << " +/- " << std::setw(5) << cb.stddev
                      << "  " << std::setw(8) << cr.median  << " +/- " << std::setw(5) << cr.stddev
                      << "  " << std::setw(8) << crr.median << " +/- " << std::setw(5) << crr.stddev
                      << "  " << std::setw(8) << wah.median << " +/- " << std::setw(5) << wah.stddev
                      << "  " << std::setw(8) << ew.median  << " +/- " << std::setw(5) << ew.stddev
                      << std::endl;
        };

        print_row("OR_date", cb_or_s,  cr_or_s,  crr_or_s,  wah_or_s,  ew_or_s);
        print_row("AND+Agg", cb_and_s, cr_and_s, crr_and_s, wah_and_s, ew_and_s);
        print_row("TopK",    cb_tk_s,  cr_tk_s,  crr_tk_s,  wah_tk_s,  ew_tk_s);
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL",   cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s);

        std::cout << "\n  CB rows:  " << cb_rows
                  << "  CR rows:  " << cr_rows
                  << "  CRR rows: " << crr_rows
                  << "  WAH rows: " << wah_rows
                  << "  EW rows:  " << ew_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;

        // ===========================================================
        // ALL mode: baseline backends table (BS / BSA / Concise).
        // Printed separately so the 5-way table above stays readable;
        // speedups anchor on WAH for parity with that table.
        // ===========================================================
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q10 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  BS (ms)         BSA (ms)        Concise (ms)     BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;

        auto print_baseline_row = [](const char* label, bm_bench::Stats& w,
                                     bm_bench::Stats& b, bm_bench::Stats& ba, bm_bench::Stats& c) {
            double bs_sp  = (b.median  > 0) ? w.median / b.median  : 0;
            double bsa_sp = (ba.median > 0) ? w.median / ba.median : 0;
            double con_sp = (c.median  > 0) ? w.median / c.median  : 0;
            std::cout << "  " << std::left << std::setw(14) << label
                      << std::right
                      << std::setw(8) << b.median  << " +/- " << std::setw(5) << b.stddev
                      << "  " << std::setw(8) << ba.median << " +/- " << std::setw(5) << ba.stddev
                      << "  " << std::setw(8) << c.median  << " +/- " << std::setw(5) << c.stddev
                      << "     " << std::setw(5) << bs_sp  << "x"
                      << "     " << std::setw(5) << bsa_sp << "x"
                      << "     " << std::setw(5) << con_sp << "x" << std::endl;
        };
        print_baseline_row("OR_date", wah_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
        print_baseline_row("AND+Agg", wah_and_s, bs_and_s, bsa_and_s, con_and_s);
        print_baseline_row("TopK",    wah_tk_s,  bs_tk_s,  bsa_tk_s,  con_tk_s);
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        print_baseline_row("TOTAL",   wah_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);

        std::cout << "\n  BS rows:  " << bs_rows
                  << "  BSA rows: " << bsa_rows
                  << "  CON rows: " << con_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;
    } else {
        bm_bench::Stats *sel_or = nullptr, *sel_and = nullptr, *sel_tk = nullptr, *sel_tot = nullptr;
        size_t sel_rows = 0;
        switch (Q10_BM) {
            case Q10BmType::WAH: sel_or = &wah_or_s; sel_and = &wah_and_s;
                                 sel_tk = &wah_tk_s; sel_tot = &wah_tot_s;
                                 sel_rows = wah_rows; break;
            case Q10BmType::CB:  sel_or = &cb_or_s;  sel_and = &cb_and_s;
                                 sel_tk = &cb_tk_s;  sel_tot = &cb_tot_s;
                                 sel_rows = cb_rows; break;
            case Q10BmType::CR:  sel_or = &cr_or_s;  sel_and = &cr_and_s;
                                 sel_tk = &cr_tk_s;  sel_tot = &cr_tot_s;
                                 sel_rows = cr_rows; break;
            case Q10BmType::CRR: sel_or = &crr_or_s; sel_and = &crr_and_s;
                                 sel_tk = &crr_tk_s; sel_tot = &crr_tot_s;
                                 sel_rows = crr_rows; break;
            case Q10BmType::EW:  sel_or = &ew_or_s;  sel_and = &ew_and_s;
                                 sel_tk = &ew_tk_s;  sel_tot = &ew_tot_s;
                                 sel_rows = ew_rows; break;
            case Q10BmType::BS:  sel_or = &bs_or_s;  sel_and = &bs_and_s;
                                 sel_tk = &bs_tk_s;  sel_tot = &bs_tot_s;
                                 sel_rows = bs_rows; break;
            case Q10BmType::BSA: sel_or = &bsa_or_s; sel_and = &bsa_and_s;
                                 sel_tk = &bsa_tk_s; sel_tot = &bsa_tot_s;
                                 sel_rows = bsa_rows; break;
            case Q10BmType::CON: sel_or = &con_or_s; sel_and = &con_and_s;
                                 sel_tk = &con_tk_s; sel_tot = &con_tot_s;
                                 sel_rows = con_rows; break;
            case Q10BmType::ALL: break;  // unreachable
        }

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q10 RESULTS — " << q10_bm_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;
        auto print_single = [](const char* label, bm_bench::Stats& s) {
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(9)  << s.median
                      << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val
                      << std::setw(10) << s.max_val << std::endl;
        };
        if (sel_or) {
            print_single("OR_date",    *sel_or);
            print_single("AND+Agg",    *sel_and);
            print_single("TopK",       *sel_tk);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            print_single("TOTAL",      *sel_tot);
            std::cout << "\n  " << q10_bm_label() << " rows: " << sel_rows << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 13. CSV export — ALL mode only (same schema convention as Q5/Q6)
    // -----------------------------------------------------------------------
    if (run_all()) {
        std::string sf_label = q10_sf_label();
        std::string csv_path = "q10_results_" + sf_label + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << std::fixed << std::setprecision(4);
            csv << "sf,operation,"
                << "cb_median_ms,cb_stddev_ms,cb_min_ms,cb_max_ms,"
                << "cr_median_ms,cr_stddev_ms,cr_min_ms,cr_max_ms,"
                << "crr_median_ms,crr_stddev_ms,crr_min_ms,crr_max_ms,"
                << "wah_median_ms,wah_stddev_ms,wah_min_ms,wah_max_ms,"
                << "ew_median_ms,ew_stddev_ms,ew_min_ms,ew_max_ms,"
                << "bs_median_ms,bs_stddev_ms,bs_min_ms,bs_max_ms,"
                << "bsa_median_ms,bsa_stddev_ms,bsa_min_ms,bsa_max_ms,"
                << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms\n";

            auto csv_row = [&](const std::string& op,
                               bm_bench::Stats& cb, bm_bench::Stats& cr, bm_bench::Stats& crr,
                               bm_bench::Stats& wah, bm_bench::Stats& ew,
                               bm_bench::Stats& b, bm_bench::Stats& ba, bm_bench::Stats& co) {
                csv << sf_label << "," << op << ","
                    << cb.median  << "," << cb.stddev  << "," << cb.min_val  << "," << cb.max_val  << ","
                    << cr.median  << "," << cr.stddev  << "," << cr.min_val  << "," << cr.max_val  << ","
                    << crr.median << "," << crr.stddev << "," << crr.min_val << "," << crr.max_val << ","
                    << wah.median << "," << wah.stddev << "," << wah.min_val << "," << wah.max_val << ","
                    << ew.median  << "," << ew.stddev  << "," << ew.min_val  << "," << ew.max_val  << ","
                    << b.median   << "," << b.stddev   << "," << b.min_val   << "," << b.max_val   << ","
                    << ba.median  << "," << ba.stddev  << "," << ba.min_val  << "," << ba.max_val  << ","
                    << co.median  << "," << co.stddev  << "," << co.min_val  << "," << co.max_val  << "\n";
            };

            csv_row("OR_date", cb_or_s,  cr_or_s,  crr_or_s,  wah_or_s,  ew_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
            csv_row("AND+Agg", cb_and_s, cr_and_s, crr_and_s, wah_and_s, ew_and_s, bs_and_s, bsa_and_s, con_and_s);
            csv_row("TopK",    cb_tk_s,  cr_tk_s,  crr_tk_s,  wah_tk_s,  ew_tk_s,  bs_tk_s,  bsa_tk_s,  con_tk_s);
            csv_row("TOTAL",   cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);

            csv.close();
            std::cout << "  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
