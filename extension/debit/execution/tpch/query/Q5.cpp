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

// Shared benchmark helpers (DEBIT_BM / DEBIT_BITMAP_DIR / DEBIT_ITER /
// DEBIT_WARMUP env dispatch, stats, SF warnings).
#include "execution/tpch/bm_bench_common.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace duckdb {

// --- Q5 bitmap directories ---
// DEBIT_BITMAP_DIR (if set) supplies an absolute base; unset = cwd-relative.
// TPCH_SF selects the scale factor suffix (default: 10 = no suffix).
static const std::string Q5_SF       = bm_bench::sf_suffix();
static const std::string Q5_CB_DIR   = bm_bench::resolve_bitmap_dir("tpch_q5" + Q5_SF + "_combit");
static const std::string Q5_WAH_DIR  = bm_bench::resolve_bitmap_dir("tpch_q5" + Q5_SF + "_wah");
static const std::string Q5_CR_DIR   = bm_bench::resolve_bitmap_dir("tpch_q5" + Q5_SF + "_croaring");
static const std::string Q5_EW_DIR   = bm_bench::resolve_bitmap_dir("tpch_q5" + Q5_SF + "_ewah");

// --- Backend selection ---
// DEBIT_BM=all|wah|cb|cr|crr|ew  (legacy Q5_BM also honoured).
using Q5BmType = bm_bench::Backend;
static const Q5BmType Q5_BM = bm_bench::parse_backend("Q5_BM");

static bool run_all() { return Q5_BM == Q5BmType::ALL; }
static bool run_wah() { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::WAH; }
static bool run_cb()  { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CB;  }
static bool run_cr()  { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CR;  }
static bool run_crr() { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CRR; }
static bool run_ew()  { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::EW;  }
static bool run_bs()  { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::BS;  }
static bool run_bsa() { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::BSA; }
static bool run_con() { return Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CON; }

static const char* q5_bm_label()     { return bm_bench::backend_label(Q5_BM); }
static std::string q5_get_sf_label() { return bm_bench::sf_label(); }

// --- Q5 predicate (TPC-H spec §2.4.5) ---
// r_name = 'ASIA'
// o_orderdate >= '1994-01-01' AND o_orderdate < '1995-01-01'
// Orderdate encoded as days since 1992-01-01; 1994-01-01…1995-01-01 = days 731..1095 (inclusive).
static const int Q5_DATE_START = 731;
static const int Q5_DATE_END   = 1095;

// --- Iteration counts (override via DEBIT_ITER / DEBIT_WARMUP) ---
static const int Q5_ITERATIONS = bm_bench::iter_count(10);
static const int Q5_WARMUP     = bm_bench::warmup_count(2);

static std::once_flag q5_once_flag;

// ASIA nations
struct Q5Nation {
    int key;
    std::string name;
};

// Byte-LUT for ComBit aggregation
struct Q5ByteEntry { uint8_t count; uint8_t pos[8]; };
static Q5ByteEntry q5_byte_lut[256];
static bool q5_lut_init = []() {
    for (int v = 0; v < 256; v++) {
        uint8_t c = 0;
        for (int b = 7; b >= 0; b--)
            if (v & (1 << b)) q5_byte_lut[v].pos[c++] = 7 - b;
        q5_byte_lut[v].count = c;
    }
    return true;
}();

// Bitmap loaders
static ComBit q5_load_cb(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { std::cerr << "Error: " << p << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}
static roaring::Roaring q5_load_cr(const std::string& p) {
    std::ifstream in(p, std::ios::binary | std::ios::ate);
    if (!in) return roaring::Roaring();
    auto sz = in.tellg(); in.seekg(0);
    uint32_t ls; in.read(reinterpret_cast<char*>(&ls), 4);
    auto bsz = sz - 4;
    if (bsz > 0) { std::vector<char> buf(bsz); in.read(buf.data(), bsz); return roaring::Roaring::readSafe(buf.data(), bsz); }
    return roaring::Roaring();
}
static ibis::bitvector q5_load_wah(const std::string& p) {
    ibis::bitvector b; b.read(p.c_str()); return b;
}
static ewah::EWAHBoolArray<uint64_t> q5_load_ew(const std::string& p) {
    ewah::EWAHBoolArray<uint64_t> b;
    std::ifstream in(p, std::ios::binary);
    if (!in) return b;
    uint64_t bits; in.read(reinterpret_cast<char*>(&bits), 8);
    b.read(in);
    return b;
}

// Statistics helper
struct Q5Stats {
    double median, stddev, min_val, max_val;
};
static Q5Stats q5_compute_stats(std::vector<double>& v) {
    Q5Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    s.median  = (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
    s.min_val = v.front();
    s.max_val = v.back();
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double sq = 0;
    for (auto x : v) sq += (x - mean) * (x - mean);
    s.stddev = std::sqrt(sq / n);
    return s;
}

// Parse nation_names.csv
static std::vector<Q5Nation> q5_load_nations(const std::string& csv_path) {
    std::vector<Q5Nation> nations;
    std::ifstream in(csv_path);
    if (!in) { std::cerr << "Error: cannot open " << csv_path << std::endl; return nations; }
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find(',');
        if (pos == std::string::npos) continue;
        Q5Nation n;
        n.key  = std::stoi(line.substr(0, pos));
        n.name = line.substr(pos + 1);
        while (!n.name.empty() && (n.name.back() == '\r' || n.name.back() == '\n' || n.name.back() == ' '))
            n.name.pop_back();
        nations.push_back(n);
    }
    return nations;
}

// BMTPCH_Q5 — Main benchmark entry point
void BMTableScan::BMTPCH_Q5(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q5_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    // Loud warning if the user is running on the known-duplicated SF1 db.
    bm_bench::warn_if_sf1();

    // ============================================================
    // 0. Pre-load column data from DuckDB
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q5 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << q5_get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q5 Benchmark — " << q5_bm_label() << " only ("
                  << q5_get_sf_label() << ")" << std::endl;
    }
    std::cout << "  OR orderdate days " << Q5_DATE_START << ".." << Q5_DATE_END
              << " (" << (Q5_DATE_END - Q5_DATE_START + 1) << " bitmaps)" << std::endl;
    std::cout << "  Then AND each nation_join bitmap -> Aggregate revenue" << std::endl;
    std::cout << "  TPC-H params: r_name = 'ASIA', orderdate [1994-01-01, 1995-01-01)" << std::endl;
    std::cout << "  Bitmap dirs:";
    if (run_cb())              std::cout << " " << Q5_CB_DIR;
    if (run_cr() || run_crr()) std::cout << " " << Q5_CR_DIR;
    if (run_wah())             std::cout << " " << Q5_WAH_DIR;
    if (run_ew())              std::cout << " " << Q5_EW_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << Q5_ITERATIONS << " (first " << Q5_WARMUP << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto nations = q5_load_nations(Q5_CB_DIR + "/nation_names.csv");
    if (nations.empty()) {
        std::cerr << "Error: no nations loaded!" << std::endl;
        return;
    }
    std::cout << "  Nations: ";
    for (auto& n : nations) std::cout << n.name << "(" << n.key << ") ";
    std::cout << std::endl;

    size_t num_rows = 0;
    {
        std::ifstream meta(Q5_CB_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
        }
    }

    auto &lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");
    auto &lineitem_transaction = DuckTransaction::Get(context.client, lineitem_table.catalog);

    TableScanState scan_state;
    vector<StorageIndex> col_ids;
    col_ids.push_back(StorageIndex(5));  // l_extendedprice
    col_ids.push_back(StorageIndex(6));  // l_discount
    lineitem_table.GetStorage().InitializeScan(context.client, lineitem_transaction, scan_state, col_ids);

    vector<LogicalType> types;
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[5]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[6]);

    std::cout << "\n[Pre-load] Loading " << num_rows << " rows (l_extendedprice, l_discount) ..." << std::endl;
    auto t_preload = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> col_price(num_rows);
    std::vector<int64_t> col_disc(num_rows);
    size_t row_offset = 0;

    while (true) {
        DataChunk chunk;
        chunk.Initialize(context.client, types);
        lineitem_table.GetStorage().Scan(lineitem_transaction, chunk, scan_state);
        if (chunk.size() == 0) break;

        auto p = FlatVector::GetData<int64_t>(chunk.data[0]);
        auto d = FlatVector::GetData<int64_t>(chunk.data[1]);
        std::memcpy(col_price.data() + row_offset, p, chunk.size() * sizeof(int64_t));
        std::memcpy(col_disc.data()  + row_offset, d, chunk.size() * sizeof(int64_t));
        row_offset += chunk.size();
    }

    auto t_preload_done = std::chrono::high_resolution_clock::now();
    std::cout << "[Pre-load] Done in " << ms(t_preload, t_preload_done) << " ms" << std::endl;

    // ============================================================
    // 1. Load bitmaps (per-backend gated + timed)
    //
    // Each backend's load is wrapped in if(run_xxx()) and timed
    // separately so single-mode runs don't inflate totals with I/O
    // they didn't actually perform.  CRR loads fresh from Q5_CR_DIR
    // instead of copying from cr_date so Q5_BM=crr works stand-alone.
    // ============================================================
    std::cout << "\n[Load] Loading bitmaps (mode=" << q5_bm_label() << ")..." << std::endl;

    size_t n_nations = nations.size();

    // --- ComBit ---
    std::vector<ComBit> cb_date, cb_nation;
    std::vector<const ComBit*> cb_date_ptrs;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_date.resize(Q5_DATE_END + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            cb_date[d] = q5_load_cb(Q5_CB_DIR + "/orderdate/" + std::to_string(d) + ".bm");
        cb_nation.resize(n_nations);
        for (size_t i = 0; i < n_nations; i++)
            cb_nation[i] = q5_load_cb(Q5_CB_DIR + "/nation_join/" + std::to_string(nations[i].key) + ".bm");

        // Pre-build pointer vector for OR_many (reused across iterations;
        // setup-out-of-timed-region pattern).
        cb_date_ptrs.reserve(Q5_DATE_END - Q5_DATE_START + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            cb_date_ptrs.push_back(&cb_date[d]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring ---
    std::vector<roaring::Roaring> cr_date, cr_nation;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_date.resize(Q5_DATE_END + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            cr_date[d] = q5_load_cr(Q5_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm");
        cr_nation.resize(n_nations);
        for (size_t i = 0; i < n_nations; i++)
            cr_nation[i] = q5_load_cr(Q5_CR_DIR + "/nation_join/" + std::to_string(nations[i].key) + ".bm");
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring+Run ---
    // Loads fresh from Q5_CR_DIR and applies runOptimize so Q5_BM=crr
    // works stand-alone (doesn't require CR to be loaded).
    std::vector<roaring::Roaring> crr_date, crr_nation;
    std::vector<const roaring::Roaring*> crr_date_ptrs;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_date.resize(Q5_DATE_END + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++) {
            crr_date[d] = q5_load_cr(Q5_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm");
            crr_date[d].runOptimize();
        }
        crr_nation.resize(n_nations);
        for (size_t i = 0; i < n_nations; i++) {
            crr_nation[i] = q5_load_cr(Q5_CR_DIR + "/nation_join/" + std::to_string(nations[i].key) + ".bm");
            crr_nation[i].runOptimize();
        }

        // Pre-build pointer vector for fastunion (reused across iterations;
        // setup-out-of-timed-region pattern).
        crr_date_ptrs.reserve(Q5_DATE_END - Q5_DATE_START + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            crr_date_ptrs.push_back(&crr_date[d]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- WAH ---
    std::vector<ibis::bitvector> wah_date, wah_nation;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wah_date.resize(Q5_DATE_END + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            wah_date[d] = q5_load_wah(Q5_WAH_DIR + "/orderdate/" + std::to_string(d) + ".bm");
        wah_nation.resize(n_nations);
        for (size_t i = 0; i < n_nations; i++)
            wah_nation[i] = q5_load_wah(Q5_WAH_DIR + "/nation_join/" + std::to_string(nations[i].key) + ".bm");
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- EWAH ---
    // Pre-build date pointer array so OR phase can use
    // ewah::fast_logicalor (priority-queue k-way merge) — the direct
    // counterpart of CRR's fastunion and Concise's fast_logicalor.
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_date, ew_nation;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_date_ptrs;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ew_date.resize(Q5_DATE_END + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            ew_date[d] = q5_load_ew(Q5_EW_DIR + "/orderdate/" + std::to_string(d) + ".bm");
        ew_nation.resize(n_nations);
        for (size_t i = 0; i < n_nations; i++)
            ew_nation[i] = q5_load_ew(Q5_EW_DIR + "/nation_join/" + std::to_string(nations[i].key) + ".bm");
        ew_date_ptrs.reserve(Q5_DATE_END - Q5_DATE_START + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            ew_date_ptrs.push_back(&ew_date[d]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Bitset (uncompressed; shared by BS / BSA) ---
    std::vector<bs::Bitmap> bs_date, bs_nation;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_date.resize(Q5_DATE_END + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            bs_date[d] = bm_bench::load_bitmap_from_croaring(Q5_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm");
        bs_nation.resize(n_nations);
        for (size_t i = 0; i < n_nations; i++)
            bs_nation[i] = bm_bench::load_bitmap_from_croaring(Q5_CR_DIR + "/nation_join/" + std::to_string(nations[i].key) + ".bm");
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Concise ---
    std::vector<ConciseSet<false>> con_date, con_nation;
    std::vector<const ConciseSet<false>*> con_date_ptrs;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_date.resize(Q5_DATE_END + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            con_date[d] = bm_bench::load_concise_from_croaring(Q5_CR_DIR + "/orderdate/" + std::to_string(d) + ".bm");
        con_nation.resize(n_nations);
        for (size_t i = 0; i < n_nations; i++)
            con_nation[i] = bm_bench::load_concise_from_croaring(Q5_CR_DIR + "/nation_join/" + std::to_string(nations[i].key) + ".bm");
        con_date_ptrs.reserve(Q5_DATE_END - Q5_DATE_START + 1);
        for (int d = Q5_DATE_START; d <= Q5_DATE_END; d++)
            con_date_ptrs.push_back(&con_date[d]);
        con_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:      " << wah_load_ms << " ms" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:   " << cb_load_ms  << " ms" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load: " << cr_load_ms  << " ms" << std::endl;
    if (run_crr()) std::cout << "  CRR load:      " << crr_load_ms << " ms" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:     " << ew_load_ms  << " ms" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:   " << bs_load_ms  << " ms (shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:  " << con_load_ms << " ms" << std::endl;

    // Per-backend on-disk footprint (useful in every mode, not just ALL).
    std::cout << std::fixed << std::setprecision(2);
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q5_WAH_DIR)) << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q5_CB_DIR))  << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q5_CR_DIR))  << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(Q5_EW_DIR))  << " MiB" << std::endl;
    if (run_bs() || run_bsa()) {
        size_t b_bytes = 0;
        for (auto& b : bs_date)   b_bytes += b.nwords * sizeof(uint64_t);
        for (auto& b : bs_nation) b_bytes += b.nwords * sizeof(uint64_t);
        std::cout << "  Bitset in mem:    " << bm_bench::mib(b_bytes) << " MiB (rebuilt from CRoaring at load)" << std::endl;
    }
    if (run_con()) {
        size_t c_bytes = 0;
        for (auto& c : con_date)   c_bytes += c.sizeInBytes();
        for (auto& c : con_nation) c_bytes += c.sizeInBytes();
        std::cout << "  Concise in mem:   " << bm_bench::mib(c_bytes) << " MiB (rebuilt from CRoaring at load)" << std::endl;
    }

    // ============================================================
    // 2. Benchmark loop
    // ============================================================
    std::vector<double> cb_or_times, cb_and_times, cb_total_times;
    std::vector<double> cr_or_times, cr_and_times, cr_total_times;
    std::vector<double> crr_or_times, crr_and_times, crr_total_times;
    std::vector<double> wah_or_times, wah_and_times, wah_total_times;
    std::vector<double> ew_or_times, ew_and_times, ew_total_times;
    std::vector<double> bs_or_times, bs_and_times, bs_total_times;
    std::vector<double> bsa_or_times, bsa_and_times, bsa_total_times;
    std::vector<double> con_or_times, con_and_times, con_total_times;

    std::map<std::string, int64_t> cb_results, cr_results, crr_results, wah_results, ew_results;
    std::map<std::string, int64_t> bs_results, bsa_results, con_results;
    size_t cb_rows = 0, cr_rows = 0, crr_rows = 0, wah_rows = 0, ew_rows = 0;
    size_t bs_rows = 0, bsa_rows = 0, con_rows = 0;

    for (int iter = 0; iter < Q5_ITERATIONS; iter++) {
        bool warmup = (iter < Q5_WARMUP);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << Q5_ITERATIONS
                  << (warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ========== ComBit ==========
        // Uses the library's public OR_many + for_each_literal APIs.
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // Library-internal adaptive multi-way OR.
            // OR_many now returns a ComBit with Decompressed segments;
            // flatten into a contiguous byte buffer for downstream SIMD ops.
            ComBit cb_or = ComBit::OR_many(
                cb_date_ptrs.size(), cb_date_ptrs.data());
            auto cb_flat = combit_or_result_to_flat(cb_or);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            const int64_t* pp = col_price.data();
            const int64_t* dp = col_disc.data();
            const uint8_t* flat = cb_flat.data();

            for (size_t ni = 0; ni < n_nations; ni++) {
                int64_t revenue = 0;
                size_t cnt = 0;

                // Walk L3→L2→L1: skip zero regions efficiently.
                cb_nation[ni].for_each_literal(
                    [&](uint32_t word_pos, uint8_t val) {
                        uint8_t rb = val & flat[word_pos];
                        if (rb == 0) return;
                        size_t rbase = static_cast<size_t>(word_pos) * 8;
                        const auto& e = q5_byte_lut[rb];
                        for (int k = 0; k < e.count; k++) {
                            size_t row = rbase + e.pos[k];
                            revenue += pp[row] * (100 - dp[row]);
                            cnt++;
                        }
                    });

                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            cb_results = results;
            cb_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  CB:   OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                cb_or_times.push_back(d_or);
                cb_and_times.push_back(d_and);
                cb_total_times.push_back(d_total);
            }
        }

        // ========== CRoaring ==========
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            roaring::Roaring cr_filt = cr_date[Q5_DATE_START];
            for (int d = Q5_DATE_START + 1; d <= Q5_DATE_END; d++)
                cr_filt |= cr_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            for (size_t ni = 0; ni < n_nations; ni++) {
                roaring::Roaring grp = cr_filt & cr_nation[ni];

                int64_t revenue = 0;
                size_t cnt = 0;
                for (auto it = grp.begin(); it != grp.end(); ++it) {
                    size_t row = *it;
                    revenue += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }

                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            cr_results = results;
            cr_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  CR:   OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                cr_or_times.push_back(d_or);
                cr_and_times.push_back(d_and);
                cr_total_times.push_back(d_total);
            }
        }

        // ========== CRoaring+Run (fastunion) ==========
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            roaring::Roaring crr_filt = roaring::Roaring::fastunion(crr_date_ptrs.size(), crr_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            for (size_t ni = 0; ni < n_nations; ni++) {
                roaring::Roaring grp = crr_filt & crr_nation[ni];

                int64_t revenue = 0;
                size_t cnt = 0;
                for (auto it = grp.begin(); it != grp.end(); ++it) {
                    size_t row = *it;
                    revenue += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }

                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            crr_results = results;
            crr_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  CRR:  OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                crr_or_times.push_back(d_or);
                crr_and_times.push_back(d_and);
                crr_total_times.push_back(d_total);
            }
        }

        // ========== WAH ==========
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ibis::bitvector wah_filt = wah_date[Q5_DATE_START];
            wah_filt.decompress();
            for (int d = Q5_DATE_START + 1; d <= Q5_DATE_END; d++)
                wah_filt |= wah_date[d];
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            for (size_t ni = 0; ni < n_nations; ni++) {
                ibis::bitvector grp;
                grp.copy(wah_filt);
                grp &= wah_nation[ni];

                int64_t revenue = 0;
                size_t cnt = 0;
                ibis::bitvector::pit pit(grp);
                while (*pit != 0xFFFFFFFFU) {
                    size_t row = *pit;
                    revenue += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                    pit.next();
                }

                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            wah_results = results;
            wah_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  WAH:  OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                wah_or_times.push_back(d_or);
                wah_and_times.push_back(d_and);
                wah_total_times.push_back(d_total);
            }
        }

        // ========== EWAH ==========
        // fast_logicalor: priority-queue k-way merge (ewah-inl.h:1129),
        // matches CRR fastunion / Concise fast_logicalor.
        if (run_ew()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ewah::EWAHBoolArray<uint64_t> ew_filt = ewah::fast_logicalor(
                ew_date_ptrs.size(), ew_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            for (size_t ni = 0; ni < n_nations; ni++) {
                ewah::EWAHBoolArray<uint64_t> grp;
                ew_filt.logicaland(ew_nation[ni], grp);

                int64_t revenue = 0;
                size_t cnt = 0;
                for (auto it = grp.begin(); it != grp.end(); ++it) {
                    size_t row = *it;
                    revenue += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }

                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            ew_results = results;
            ew_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  EW:   OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                ew_or_times.push_back(d_or);
                ew_and_times.push_back(d_and);
                ew_total_times.push_back(d_total);
            }
        }

        // ========== Bitset (scalar) ==========
        // Pure baseline: clone the first orderdate bitmap, OR the rest in
        // (scalar word-loop), then for each nation clone the filter, AND
        // with nation_join, and aggregate revenue while walking set bits
        // via ctz.
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bs_filt = bs_date[Q5_DATE_START].clone();
            for (int d = Q5_DATE_START + 1; d <= Q5_DATE_END; d++)
                bs::or_inplace(bs_filt, bs_date[d], false);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            for (size_t ni = 0; ni < n_nations; ni++) {
                bs::Bitmap grp = bs_filt.clone();
                bs::and_inplace(grp, bs_nation[ni], false);

                int64_t revenue = 0;
                size_t cnt = 0;
                for (size_t i = 0; i < grp.nwords; ++i) {
                    uint64_t w = grp.words[i];
                    const size_t base = i * 64;
                    while (w) {
                        size_t row = base + __builtin_ctzll(w);
                        if (row >= grp.nbits) break;
                        revenue += col_price[row] * (100 - col_disc[row]);
                        cnt++;
                        w &= w - 1;
                    }
                }
                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            bs_results = results;
            bs_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  BS:   OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                bs_or_times.push_back(d_or);
                bs_and_times.push_back(d_and);
                bs_total_times.push_back(d_total);
            }
        }

        // ========== Bitset + AVX-512 ==========
        // Same data as BS, simd flag flipped on for OR/AND.  Aggregation
        // loop is identical (ctz walk over the result words).
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bsa_filt = bs_date[Q5_DATE_START].clone();
            for (int d = Q5_DATE_START + 1; d <= Q5_DATE_END; d++)
                bs::or_inplace(bsa_filt, bs_date[d], true);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            for (size_t ni = 0; ni < n_nations; ni++) {
                bs::Bitmap grp = bsa_filt.clone();
                bs::and_inplace(grp, bs_nation[ni], true);

                int64_t revenue = 0;
                size_t cnt = 0;
                for (size_t i = 0; i < grp.nwords; ++i) {
                    uint64_t w = grp.words[i];
                    const size_t base = i * 64;
                    while (w) {
                        size_t row = base + __builtin_ctzll(w);
                        if (row >= grp.nbits) break;
                        revenue += col_price[row] * (100 - col_disc[row]);
                        cnt++;
                        w &= w - 1;
                    }
                }
                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            bsa_results = results;
            bsa_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  BSA:  OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                bsa_or_times.push_back(d_or);
                bsa_and_times.push_back(d_and);
                bsa_total_times.push_back(d_total);
            }
        }

        // ========== Concise ==========
        // Uses fast_logicalor (priority-queue k-way merge) for OR — the
        // direct counterpart of CRR's fastunion — and logicaland (returns
        // a fresh container) per nation, since the *ToContainer variants
        // in the upstream library do not reset `res` before writing.
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_filt = ConciseSet<false>::fast_logicalor(
                con_date_ptrs.size(), con_date_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            std::map<std::string, int64_t> results;
            size_t total_rows = 0;
            for (size_t ni = 0; ni < n_nations; ni++) {
                ConciseSet<false> grp = con_filt.logicaland(con_nation[ni]);

                int64_t revenue = 0;
                size_t cnt = 0;
                for (auto it = grp.begin(); it != grp.end(); ++it) {
                    size_t row = *it;
                    revenue += col_price[row] * (100 - col_disc[row]);
                    cnt++;
                }
                results[nations[ni].name] = revenue;
                total_rows += cnt;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            con_results = results;
            con_rows = total_rows;

            double d_or = ms(t0,t1), d_and = ms(t1,t2), d_total = ms(t0,t2);
            std::cout << "  CON:  OR_date=" << d_or << "  AND+Agg=" << d_and
                      << "  Total=" << d_total << "  rows=" << total_rows << std::endl;
            if (!warmup) {
                con_or_times.push_back(d_or);
                con_and_times.push_back(d_and);
                con_total_times.push_back(d_total);
            }
        }
    } // end iterations

    // ============================================================
    // 3. Correctness validation
    //
    // Row counts and revenue columns are shown only for backends that
    // actually ran.  Cross-backend consistency check only makes sense
    // in ALL mode (need multiple backends to compare).
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q5 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "  Row counts:";
    if (run_wah()) std::cout << " WAH=" << wah_rows;
    if (run_cb())  std::cout << " CB="  << cb_rows;
    if (run_cr())  std::cout << " CR="  << cr_rows;
    if (run_crr()) std::cout << " CRR=" << crr_rows;
    if (run_ew())  std::cout << " EW="  << ew_rows;
    if (run_bs())  std::cout << " BS="  << bs_rows;
    if (run_bsa()) std::cout << " BSA=" << bsa_rows;
    if (run_con()) std::cout << " CON=" << con_rows;
    std::cout << std::endl;

    // Pick one backend's results as the "source" to sort rows by revenue
    // (ALL mode prefers ComBit; single mode uses the selected backend).
    const std::map<std::string, int64_t>* print_src = nullptr;
    const char* print_src_label = "";
    if      (Q5_BM == Q5BmType::ALL || Q5_BM == Q5BmType::CB)  { print_src = &cb_results;  print_src_label = "ComBit"; }
    else if (Q5_BM == Q5BmType::WAH)                           { print_src = &wah_results; print_src_label = "WAH";    }
    else if (Q5_BM == Q5BmType::CR)                            { print_src = &cr_results;  print_src_label = "CRoaring"; }
    else if (Q5_BM == Q5BmType::CRR)                           { print_src = &crr_results; print_src_label = "CRoaring+Run"; }
    else if (Q5_BM == Q5BmType::EW)                            { print_src = &ew_results;  print_src_label = "EWAH"; }
    else if (Q5_BM == Q5BmType::BS)                            { print_src = &bs_results;  print_src_label = "Bitset"; }
    else if (Q5_BM == Q5BmType::BSA)                           { print_src = &bsa_results; print_src_label = "Bitset+AVX512"; }
    else if (Q5_BM == Q5BmType::CON)                           { print_src = &con_results; print_src_label = "Concise"; }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n  Q5 Results (revenue = SUM(l_extendedprice * (1 - l_discount)), sorted by "
              << print_src_label << "):" << std::endl;
    std::cout << "  " << std::left << std::setw(15) << "nation" << std::right;
    if (run_cb())  std::cout << std::setw(22) << "CB_revenue";
    if (run_cr())  std::cout << std::setw(22) << "CR_revenue";
    if (run_crr()) std::cout << std::setw(22) << "CRR_revenue";
    if (run_wah()) std::cout << std::setw(22) << "WAH_revenue";
    if (run_ew())  std::cout << std::setw(22) << "EW_revenue";
    if (run_bs())  std::cout << std::setw(22) << "BS_revenue";
    if (run_bsa()) std::cout << std::setw(22) << "BSA_revenue";
    if (run_con()) std::cout << std::setw(22) << "CON_revenue";
    std::cout << std::endl;

    if (print_src) {
        std::vector<std::pair<std::string, int64_t>> sorted_results(print_src->begin(), print_src->end());
        std::sort(sorted_results.begin(), sorted_results.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (auto& [name, rev] : sorted_results) {
            std::cout << "  " << std::left << std::setw(15) << name << std::right;
            if (run_cb())  std::cout << std::setw(22) << (double)cb_results[name]  / 10000.0;
            if (run_cr())  std::cout << std::setw(22) << (double)cr_results[name]  / 10000.0;
            if (run_crr()) std::cout << std::setw(22) << (double)crr_results[name] / 10000.0;
            if (run_wah()) std::cout << std::setw(22) << (double)wah_results[name] / 10000.0;
            if (run_ew())  std::cout << std::setw(22) << (double)ew_results[name]  / 10000.0;
            if (run_bs())  std::cout << std::setw(22) << (double)bs_results[name]  / 10000.0;
            if (run_bsa()) std::cout << std::setw(22) << (double)bsa_results[name] / 10000.0;
            if (run_con()) std::cout << std::setw(22) << (double)con_results[name] / 10000.0;
            std::cout << std::endl;
        }
    }

    if (run_all()) {
        bool consistent = true;
        for (auto& [name, rev] : cb_results) {
            if (cr_results[name] != rev || crr_results[name] != rev ||
                wah_results[name] != rev || ew_results[name] != rev ||
                bs_results[name] != rev || bsa_results[name] != rev ||
                con_results[name] != rev) {
                consistent = false;
                std::cout << "  *** MISMATCH for " << name << " ***" << std::endl;
            }
        }
        std::cout << "  Consistency: " << (consistent ? "ALL MATCH" : "MISMATCH DETECTED") << std::endl;
    }

    // ============================================================
    // 4. Statistics summary
    //
    // compute_stats on empty vectors (skipped backends) returns all
    // zeros — the single-backend print path below picks its own source,
    // so zero entries never appear in the output.
    // ============================================================
    auto cb_or_s  = q5_compute_stats(cb_or_times);
    auto cb_and_s = q5_compute_stats(cb_and_times);
    auto cb_tot_s = q5_compute_stats(cb_total_times);

    auto cr_or_s  = q5_compute_stats(cr_or_times);
    auto cr_and_s = q5_compute_stats(cr_and_times);
    auto cr_tot_s = q5_compute_stats(cr_total_times);

    auto crr_or_s  = q5_compute_stats(crr_or_times);
    auto crr_and_s = q5_compute_stats(crr_and_times);
    auto crr_tot_s = q5_compute_stats(crr_total_times);

    auto wah_or_s  = q5_compute_stats(wah_or_times);
    auto wah_and_s = q5_compute_stats(wah_and_times);
    auto wah_tot_s = q5_compute_stats(wah_total_times);

    auto ew_or_s  = q5_compute_stats(ew_or_times);
    auto ew_and_s = q5_compute_stats(ew_and_times);
    auto ew_tot_s = q5_compute_stats(ew_total_times);

    auto bs_or_s   = q5_compute_stats(bs_or_times);
    auto bs_and_s  = q5_compute_stats(bs_and_times);
    auto bs_tot_s  = q5_compute_stats(bs_total_times);

    auto bsa_or_s  = q5_compute_stats(bsa_or_times);
    auto bsa_and_s = q5_compute_stats(bsa_and_times);
    auto bsa_tot_s = q5_compute_stats(bsa_total_times);

    auto con_or_s  = q5_compute_stats(con_or_times);
    auto con_and_s = q5_compute_stats(con_and_times);
    auto con_tot_s = q5_compute_stats(con_total_times);

    int measured = Q5_ITERATIONS - Q5_WARMUP;

    // --- DuckDB native SQL baseline + ground-truth row count ---
    // Runs AFTER the bitmap iteration loop so it does not pollute the
    // CPU cache and skew bitmap measurements.  TPC-H Q5 (§2.4.5)
    // row count over lineitem after all joins + filters.
    int64_t gt_rows = -1;
    double  gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT count(*) "
            "FROM lineitem l, orders o, customer c, supplier s, nation n, region r "
            "WHERE l.l_orderkey = o.o_orderkey "
            "  AND l.l_suppkey  = s.s_suppkey "
            "  AND c.c_custkey  = o.o_custkey "
            "  AND c.c_nationkey = s.s_nationkey "
            "  AND s.s_nationkey = n.n_nationkey "
            "  AND n.n_regionkey = r.r_regionkey "
            "  AND r.r_name      = 'ASIA' "
            "  AND o.o_orderdate >= DATE '1994-01-01' "
            "  AND o.o_orderdate <  DATE '1995-01-01'";
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = con.Query(sql);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (result && !result->HasError() && result->RowCount() == 1) {
            gt_rows = result->GetValue(0, 0).GetValue<int64_t>();
            gt_sql_ms = ms(t0, t1);
        } else if (result && result->HasError()) {
            std::cerr << "[Baseline] SQL error: " << result->GetError() << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "[Baseline] Connection/Query threw: " << e.what() << std::endl;
    }
    std::cout << std::fixed << std::setprecision(2);
    if (gt_rows >= 0) {
        std::cout << "\n[Baseline] DuckDB native SQL count(*) = " << gt_rows
                  << "  (single run: " << gt_sql_ms << " ms)" << std::endl;
    } else {
        std::cout << "\n[Baseline] DuckDB SQL ground truth unavailable — skipping assert." << std::endl;
    }

    // --- Ground-truth assert: every active backend's total rows must match SQL ---
    if (gt_rows >= 0) {
        auto check = [&](const char* label, bool active, size_t got) {
            if (!active) return;
            if (static_cast<int64_t>(got) == gt_rows) return;
            std::ostringstream oss;
            oss << "[FAIL] Q5 " << label << " total rows " << got
                << " != DuckDB SQL ground truth " << gt_rows
                << " — bitmap pipeline is incorrect";
            throw std::runtime_error(oss.str());
        };
        check("WAH",           run_wah(), wah_rows);
        check("ComBit",        run_cb(),  cb_rows);
        check("CRoaring",      run_cr(),  cr_rows);
        check("CRoaring+Run",  run_crr(), crr_rows);
        check("EWAH",          run_ew(),  ew_rows);
        check("Bitset",        run_bs(),  bs_rows);
        check("Bitset+AVX512", run_bsa(), bsa_rows);
        check("Concise",       run_con(), con_rows);
        std::cout << "[OK] all active backends match DuckDB SQL ground truth ("
                  << gt_rows << " rows)." << std::endl;
    }


    if (run_all()) {
        // -------- ALL mode: full 5-way comparison table --------
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q5 RESULTS (" << measured << " measured iterations, median +/- stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  CB (ms)         CR (ms)        CRR (ms)        WAH (ms)        EW (ms)" << std::endl;
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;

        auto print_row = [](const char* label, Q5Stats& cb, Q5Stats& cr,
                            Q5Stats& crr, Q5Stats& wah, Q5Stats& ew) {
            std::cout << "  " << std::left << std::setw(14) << label
                      << std::right
                      << std::setw(8) << cb.median << " +/- " << std::setw(5) << cb.stddev
                      << "  " << std::setw(8) << cr.median << " +/- " << std::setw(5) << cr.stddev
                      << "  " << std::setw(8) << crr.median << " +/- " << std::setw(5) << crr.stddev
                      << "  " << std::setw(8) << wah.median << " +/- " << std::setw(5) << wah.stddev
                      << "  " << std::setw(8) << ew.median << " +/- " << std::setw(5) << ew.stddev
                      << std::endl;
        };

        print_row("OR_date", cb_or_s, cr_or_s, crr_or_s, wah_or_s, ew_or_s);
        print_row("AND+Agg", cb_and_s, cr_and_s, crr_and_s, wah_and_s, ew_and_s);
        std::cout << "  -----------------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL", cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s);

        std::cout << "\n  CB rows:  " << cb_rows
                  << "  CR rows:  " << cr_rows
                  << "  CRR rows: " << crr_rows
                  << "  WAH rows: " << wah_rows
                  << "  EW rows:  " << ew_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;

        // ============================================================
        // ALL mode: baseline backends table (BS / BSA / Concise).
        // Printed separately so the 5-way compression-format table
        // above stays readable; speedups anchor on WAH for parity
        // with that table.
        // ============================================================
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q5 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  BS (ms)         BSA (ms)        Concise (ms)     BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;

        auto print_baseline_row = [](const char* label, Q5Stats& w,
                                     Q5Stats& b, Q5Stats& ba, Q5Stats& c) {
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
                      << "     " << std::setw(5) << con_sp << "x"
                      << std::endl;
        };
        print_baseline_row("OR_date", wah_or_s,  bs_or_s,  bsa_or_s,  con_or_s);
        print_baseline_row("AND+Agg", wah_and_s, bs_and_s, bsa_and_s, con_and_s);
        std::cout << "  ----------------------------------------------------------------------------------------------" << std::endl;
        print_baseline_row("TOTAL",   wah_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);

        std::cout << "\n  BS rows:  " << bs_rows
                  << "  BSA rows: " << bsa_rows
                  << "  CON rows: " << con_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;
    } else {
        // -------- Single-backend mode: 4-column (median/stddev/min/max) --------
        Q5Stats *sel_or = nullptr, *sel_and = nullptr, *sel_tot = nullptr;
        size_t sel_rows = 0;

        switch (Q5_BM) {
            case Q5BmType::WAH:
                sel_or = &wah_or_s; sel_and = &wah_and_s; sel_tot = &wah_tot_s;
                sel_rows = wah_rows; break;
            case Q5BmType::CB:
                sel_or = &cb_or_s;  sel_and = &cb_and_s;  sel_tot = &cb_tot_s;
                sel_rows = cb_rows;  break;
            case Q5BmType::CR:
                sel_or = &cr_or_s;  sel_and = &cr_and_s;  sel_tot = &cr_tot_s;
                sel_rows = cr_rows;  break;
            case Q5BmType::CRR:
                sel_or = &crr_or_s; sel_and = &crr_and_s; sel_tot = &crr_tot_s;
                sel_rows = crr_rows; break;
            case Q5BmType::EW:
                sel_or = &ew_or_s;  sel_and = &ew_and_s;  sel_tot = &ew_tot_s;
                sel_rows = ew_rows;  break;
            case Q5BmType::BS:
                sel_or = &bs_or_s;  sel_and = &bs_and_s;  sel_tot = &bs_tot_s;
                sel_rows = bs_rows;  break;
            case Q5BmType::BSA:
                sel_or = &bsa_or_s; sel_and = &bsa_and_s; sel_tot = &bsa_tot_s;
                sel_rows = bsa_rows; break;
            case Q5BmType::CON:
                sel_or = &con_or_s; sel_and = &con_and_s; sel_tot = &con_tot_s;
                sel_rows = con_rows; break;
            case Q5BmType::ALL: break;   // unreachable (handled above)
        }

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q5 RESULTS — " << q5_bm_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;

        auto print_single = [](const char* label, Q5Stats& s) {
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(9) << s.median
                      << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val
                      << std::setw(10) << s.max_val << std::endl;
        };

        if (sel_or) {
            print_single("OR_date",    *sel_or);
            print_single("AND+Agg",    *sel_and);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            print_single("TOTAL",      *sel_tot);
            std::cout << "\n  " << q5_bm_label() << " rows: " << sel_rows << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }

    // ============================================================
    // 5. CSV export — ALL mode only
    //
    // The CSV schema is fixed 20 columns (5 backends × 4 stats); in
    // single-backend mode 4 backends' columns would all be zero, which
    // is misleading for cross-SF reporting.  Single-backend runs are
    // iterative dev tools, not CSV report inputs.
    // ============================================================
    if (run_all()) {
        std::string sf_label = q5_get_sf_label();
        std::string csv_path = "q5_results_" + sf_label + ".csv";
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
                               Q5Stats& cb, Q5Stats& cr, Q5Stats& crr,
                               Q5Stats& wah, Q5Stats& ew,
                               Q5Stats& b, Q5Stats& ba, Q5Stats& co) {
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
            csv_row("TOTAL",   cb_tot_s, cr_tot_s, crr_tot_s, wah_tot_s, ew_tot_s, bs_tot_s, bsa_tot_s, con_tot_s);

            csv.close();
            std::cout << "  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
