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
#include <cctype>

namespace duckdb {

// --- Q1 bitmap directories ---
// DEBIT_BITMAP_DIR (if set) supplies an absolute base; unset = cwd-relative.
// TPCH_SF selects the scale factor suffix (default: 10 = no suffix).
static const std::string SF_SUFFIX    = bm_bench::sf_suffix();
static const std::string COMBIT_DIR   = bm_bench::resolve_bitmap_dir("tpch_q1" + SF_SUFFIX + "_combit");
static const std::string WAH_DIR      = bm_bench::resolve_bitmap_dir("tpch_q1" + SF_SUFFIX + "_wah");
static const std::string CROARING_DIR = bm_bench::resolve_bitmap_dir("tpch_q1" + SF_SUFFIX + "_croaring");
static const std::string EWAH_DIR     = bm_bench::resolve_bitmap_dir("tpch_q1" + SF_SUFFIX + "_ewah");

// --- Backend selection ---
// DEBIT_BM=all|wah|cb|cr|crr|ew  (legacy Q1_BM also honoured).
using Q1BmType = bm_bench::Backend;
static const Q1BmType Q1_BM = bm_bench::parse_backend("Q1_BM");

static bool run_all() { return Q1_BM == Q1BmType::ALL; }
static bool run_wah() { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::WAH; }
static bool run_cb()  { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::CB;  }
static bool run_cr()  { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::CR;  }
static bool run_crr() { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::CRR; }
static bool run_ew()  { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::EW;  }
static bool run_bs()  { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::BS;  }
static bool run_bsa() { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::BSA; }
static bool run_con() { return Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::CON; }

static const char* q1_bm_label()  { return bm_bench::backend_label(Q1_BM); }
static std::string get_sf_label() { return bm_bench::sf_label(); }

// --- Q1 predicate (TPC-H spec §2.4.1) ---
// l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY = 1998-09-02
// Shipdate encoded as days since 1992-01-01; 1998-09-02 ↔ day 2436.
static const int SHIPDATE_CUTOFF = 2436;
static int       SHIPDATE_MAX    = 0;  // discovered at load time

// returnflag: A=0, N=1, R=2;  linestatus: F=0, O=1.
static const char RF_CHARS[] = {'A', 'N', 'R'};
static const char LS_CHARS[] = {'F', 'O'};
static const int  RF_COUNT   = 3;
static const int  LS_COUNT   = 2;

// --- Iteration counts (override via DEBIT_ITER / DEBIT_WARMUP) ---
static const int NUM_ITERATIONS = bm_bench::iter_count(10);
static const int WARMUP_RUNS    = bm_bench::warmup_count(2);

static std::once_flag q1_once_flag;

// Q1 aggregation result per group
struct Q1Group {
    int64_t sum_qty        = 0;
    int64_t sum_base_price = 0;
    int64_t sum_disc_price = 0;
    int64_t sum_charge     = 0;
    int64_t sum_discount   = 0;
    int64_t count_order    = 0;
};

// Bitmap loaders (same as Q6)
static ibis::bitvector load_wah_bm(const std::string& path) {
    ibis::bitvector btv;
    btv.read(path.c_str());
    return btv;
}

static ComBit load_combit_bm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Error: cannot open " << path << std::endl; return ComBit(); }
    return ComBit::deserialize(in);
}

static roaring::Roaring load_croaring_bm(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::cerr << "Error: cannot open " << path << std::endl; return roaring::Roaring(); }
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    uint32_t logical_size;
    in.read(reinterpret_cast<char*>(&logical_size), sizeof(logical_size));
    std::streamsize bitmap_size = size - sizeof(logical_size);
    if (bitmap_size > 0) {
        std::vector<char> buffer(bitmap_size);
        in.read(buffer.data(), bitmap_size);
        return roaring::Roaring::readSafe(buffer.data(), bitmap_size);
    }
    return roaring::Roaring();
}

static ewah::EWAHBoolArray<uint64_t> load_ewah_bm(const std::string& path) {
    ewah::EWAHBoolArray<uint64_t> btv;
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Error: cannot open " << path << std::endl; return btv; }
    uint64_t current_bits;
    in.read(reinterpret_cast<char*>(&current_bits), sizeof(current_bits));
    btv.read(in);
    return btv;
}

// WAH flip (NOT) — in-place on decompressed bitvector
static void wah_flip(ibis::bitvector* btv) {
#if defined(__AVX512F__)
    ibis::bitvector::word_t *it = btv->m_vec.begin();
    while (it + 15 < btv->m_vec.end()) {
        _mm512_storeu_epi32(it, _mm512_andnot_epi32(
            _mm512_loadu_epi32(it), _mm512_set1_epi32(0x7fffffff)));
        it += 16;
    }
    for (; it < btv->m_vec.end(); it++)
        *it ^= ibis::bitvector::ALLONES;
    if (btv->active.nbits > 0)
        btv->active.val ^= ((1 << btv->active.nbits) - 1);
#else
    ibis::bitvector::word_t *it = btv->m_vec.begin();
    for (; it < btv->m_vec.end(); ++it)
        *it ^= ibis::bitvector::ALLONES;
    if (btv->active.nbits > 0)
        btv->active.val ^= ((1 << btv->active.nbits) - 1);
#endif
}

// Byte-LUT for MSB-first bit extraction (same as combit_adapter.cpp)
// 256 × 9 = 2304 bytes, permanently in L1d cache.
struct Q1ByteEntry { uint8_t count; uint8_t pos[8]; };
static Q1ByteEntry byte_lut[256];
static bool byte_lut_init = []() {
    for (int v = 0; v < 256; v++) {
        uint8_t c = 0;
        for (int b = 7; b >= 0; b--)
            if (v & (1 << b))
                byte_lut[v].pos[c++] = 7 - b;
        byte_lut[v].count = c;
    }
    return true;
}();

// Per-row aggregation macro (identical computation for all formats)
#define Q1_AGG_ROW(g, r, qty, price, disc, tax) do { \
    (g).sum_qty        += (qty)[(r)];                 \
    (g).sum_discount   += (disc)[(r)];                \
    (g).sum_base_price += (price)[(r)];               \
    int64_t _dp = (price)[(r)] * (100 - (disc)[(r)]); \
    (g).sum_disc_price += _dp;                        \
    (g).sum_charge     += _dp * (100 + (tax)[(r)]);   \
    (g).count_order++;                                \
} while (0)

// Statistics helper (same as Q6)
struct Q1TimingStats {
    double median, stddev, min_val, max_val;
};

static Q1TimingStats compute_stats(std::vector<double>& vals) {
    Q1TimingStats s{};
    if (vals.empty()) return s;
    std::sort(vals.begin(), vals.end());
    size_t n = vals.size();
    s.median  = (n % 2 == 0) ? (vals[n/2-1] + vals[n/2]) / 2.0 : vals[n/2];
    s.min_val = vals.front();
    s.max_val = vals.back();
    double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / n;
    double sq = 0;
    for (auto v : vals) sq += (v - mean) * (v - mean);
    s.stddev = std::sqrt(sq / n);
    return s;
}

// Detect max shipdate day from directory
static int detect_shipdate_max(const std::string& dir) {
    int mx = 0;
    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".bm") {
            try { mx = std::max(mx, std::stoi(entry.path().stem().string())); }
            catch (...) {}
        }
    }
    return mx;
}

// BMTPCH_Q1 — Main benchmark entry point
void BMTableScan::BMTPCH_Q1(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q1_once_flag, [&]() {

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
        std::cout << "  TPC-H Q1 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << get_sf_label() << ")" << std::endl;
    } else {
        std::cout << "  TPC-H Q1 Benchmark — " << q1_bm_label() << " only ("
                  << get_sf_label() << ")" << std::endl;
    }
    std::cout << "  Complement approach: OR days " << (SHIPDATE_CUTOFF+1) << "..max → NOT" << std::endl;
    std::cout << "  TPC-H params: l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY (= 1998-09-02)" << std::endl;
    std::cout << "  Bitmap dirs:";
    if (run_wah()) std::cout << " " << WAH_DIR;
    if (run_cb())  std::cout << " " << COMBIT_DIR;
    if (run_cr() || run_crr()) std::cout << " " << CROARING_DIR;
    if (run_ew())  std::cout << " " << EWAH_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << NUM_ITERATIONS << " (first " << WARMUP_RUNS << " = warm-up)" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto &lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");
    auto &lineitem_transaction = DuckTransaction::Get(context.client, lineitem_table.catalog);

    TableScanState scan_state;
    vector<StorageIndex> col_ids;
    col_ids.push_back(StorageIndex(4));  // l_quantity
    col_ids.push_back(StorageIndex(5));  // l_extendedprice
    col_ids.push_back(StorageIndex(6));  // l_discount
    col_ids.push_back(StorageIndex(7));  // l_tax
    lineitem_table.GetStorage().InitializeScan(context.client, lineitem_transaction, scan_state, col_ids);

    vector<LogicalType> types;
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[4]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[5]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[6]);
    types.push_back(lineitem_table.GetColumns().GetColumnTypes()[7]);

    // Read num_rows from the ComBit done.txt metadata file
    size_t num_rows = 0;
    {
        std::ifstream meta(COMBIT_DIR + "/done.txt");
        std::string line;
        while (std::getline(meta, line)) {
            if (line.rfind("num_rows=", 0) == 0)
                num_rows = std::stoull(line.substr(9));
        }
    }

    std::cout << "\n[Pre-load] Loading " << num_rows << " rows (4 columns) ..." << std::endl;
    auto t_preload = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> col_qty(num_rows);
    std::vector<int64_t> col_price(num_rows);
    std::vector<int64_t> col_disc(num_rows);
    std::vector<int64_t> col_tax(num_rows);
    size_t row_offset = 0;

    while (true) {
        DataChunk chunk;
        chunk.Initialize(context.client, types);
        lineitem_table.GetStorage().Scan(lineitem_transaction, chunk, scan_state);
        if (chunk.size() == 0) break;

        auto q = FlatVector::GetData<int64_t>(chunk.data[0]);
        auto p = FlatVector::GetData<int64_t>(chunk.data[1]);
        auto d = FlatVector::GetData<int64_t>(chunk.data[2]);
        auto t = FlatVector::GetData<int64_t>(chunk.data[3]);

        std::memcpy(col_qty.data()   + row_offset, q, chunk.size() * sizeof(int64_t));
        std::memcpy(col_price.data() + row_offset, p, chunk.size() * sizeof(int64_t));
        std::memcpy(col_disc.data()  + row_offset, d, chunk.size() * sizeof(int64_t));
        std::memcpy(col_tax.data()   + row_offset, t, chunk.size() * sizeof(int64_t));
        row_offset += chunk.size();
    }

    auto t_preload_done = std::chrono::high_resolution_clock::now();
    std::cout << "[Pre-load] Done in " << ms(t_preload, t_preload_done) << " ms" << std::endl;

    // ============================================================
    // 1. Load bitmaps (only for selected backends; per-backend
    //    timed separately so single-mode runs don't inflate totals
    //    with I/O they didn't actually perform)
    // ============================================================
    std::cout << "\n[Load] Loading bitmaps (mode=" << q1_bm_label() << ")..." << std::endl;

    SHIPDATE_MAX = detect_shipdate_max(COMBIT_DIR + "/shipdate");
    int ship_or_start = SHIPDATE_CUTOFF + 1;
    int ship_or_count = SHIPDATE_MAX - SHIPDATE_CUTOFF;
    std::cout << "  Shipdate: cutoff=" << SHIPDATE_CUTOFF << " max=" << SHIPDATE_MAX
              << " complement OR: " << ship_or_count << " bitmaps (days " << ship_or_start << ".." << SHIPDATE_MAX << ")" << std::endl;

    // --- ComBit ---
    std::vector<ComBit> cb_ship, cb_rf, cb_ls;
    std::vector<const ComBit*> cb_ship_ptrs;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_ship.resize(SHIPDATE_MAX + 1);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            cb_ship[d] = load_combit_bm(COMBIT_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cb_rf.resize(RF_COUNT);
        for (int i = 0; i < RF_COUNT; i++)
            cb_rf[i] = load_combit_bm(COMBIT_DIR + "/returnflag/" + std::to_string(i) + ".bm");
        cb_ls.resize(LS_COUNT);
        for (int i = 0; i < LS_COUNT; i++)
            cb_ls[i] = load_combit_bm(COMBIT_DIR + "/linestatus/" + std::to_string(i) + ".bm");
        cb_ship_ptrs.reserve(ship_or_count);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            cb_ship_ptrs.push_back(&cb_ship[d]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring ---
    std::vector<roaring::Roaring> cr_ship, cr_rf, cr_ls;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_ship.resize(SHIPDATE_MAX + 1);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            cr_ship[d] = load_croaring_bm(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cr_rf.resize(RF_COUNT);
        for (int i = 0; i < RF_COUNT; i++)
            cr_rf[i] = load_croaring_bm(CROARING_DIR + "/returnflag/" + std::to_string(i) + ".bm");
        cr_ls.resize(LS_COUNT);
        for (int i = 0; i < LS_COUNT; i++)
            cr_ls[i] = load_croaring_bm(CROARING_DIR + "/linestatus/" + std::to_string(i) + ".bm");
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- CRoaring+Run ---
    // Loads fresh from the croaring/ dir and applies runOptimize so
    // Q1_BM=crr works stand-alone (doesn't require CR to be loaded).
    std::vector<roaring::Roaring> crr_ship, crr_rf, crr_ls;
    std::vector<const roaring::Roaring*> crr_ship_ptrs;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_ship.resize(SHIPDATE_MAX + 1);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++) {
            crr_ship[d] = load_croaring_bm(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");
            crr_ship[d].runOptimize();
        }
        crr_rf.resize(RF_COUNT);
        for (int i = 0; i < RF_COUNT; i++) {
            crr_rf[i] = load_croaring_bm(CROARING_DIR + "/returnflag/" + std::to_string(i) + ".bm");
            crr_rf[i].runOptimize();
        }
        crr_ls.resize(LS_COUNT);
        for (int i = 0; i < LS_COUNT; i++) {
            crr_ls[i] = load_croaring_bm(CROARING_DIR + "/linestatus/" + std::to_string(i) + ".bm");
            crr_ls[i].runOptimize();
        }
        crr_ship_ptrs.reserve(ship_or_count);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            crr_ship_ptrs.push_back(&crr_ship[d]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- WAH ---
    std::vector<ibis::bitvector> wah_ship, wah_rf, wah_ls;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wah_ship.resize(SHIPDATE_MAX + 1);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            wah_ship[d] = load_wah_bm(WAH_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        wah_rf.resize(RF_COUNT);
        for (int i = 0; i < RF_COUNT; i++)
            wah_rf[i] = load_wah_bm(WAH_DIR + "/returnflag/" + std::to_string(i) + ".bm");
        wah_ls.resize(LS_COUNT);
        for (int i = 0; i < LS_COUNT; i++)
            wah_ls[i] = load_wah_bm(WAH_DIR + "/linestatus/" + std::to_string(i) + ".bm");
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- EWAH ---
    // Pre-build shipdate pointer array so OR phase can use
    // ewah::fast_logicalor (priority-queue k-way merge) — matches
    // CRR fastunion / Concise fast_logicalor.
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_ship, ew_rf, ew_ls;
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_ship_ptrs;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ew_ship.resize(SHIPDATE_MAX + 1);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            ew_ship[d] = load_ewah_bm(EWAH_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        ew_rf.resize(RF_COUNT);
        for (int i = 0; i < RF_COUNT; i++)
            ew_rf[i] = load_ewah_bm(EWAH_DIR + "/returnflag/" + std::to_string(i) + ".bm");
        ew_ls.resize(LS_COUNT);
        for (int i = 0; i < LS_COUNT; i++)
            ew_ls[i] = load_ewah_bm(EWAH_DIR + "/linestatus/" + std::to_string(i) + ".bm");
        ew_ship_ptrs.reserve(ship_or_count);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            ew_ship_ptrs.push_back(&ew_ship[d]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Bitset (uncompressed; shared by BS / BSA) ---
    std::vector<bs::Bitmap> bs_ship, bs_rf, bs_ls;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_ship.resize(SHIPDATE_MAX + 1);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            bs_ship[d] = bm_bench::load_bitmap_from_croaring(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        bs_rf.resize(RF_COUNT);
        for (int i = 0; i < RF_COUNT; i++)
            bs_rf[i] = bm_bench::load_bitmap_from_croaring(CROARING_DIR + "/returnflag/" + std::to_string(i) + ".bm");
        bs_ls.resize(LS_COUNT);
        for (int i = 0; i < LS_COUNT; i++)
            bs_ls[i] = bm_bench::load_bitmap_from_croaring(CROARING_DIR + "/linestatus/" + std::to_string(i) + ".bm");
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // --- Concise ---
    // The complement step (NOT future) is implemented as
    // `con_full.logicalandnot(future)`; con_full is the universe
    // [0, num_rows) and is built once at load.  Concise represents a
    // dense run as a single fill word, so con_full's storage cost is
    // O(1) words.
    std::vector<ConciseSet<false>> con_ship, con_rf, con_ls;
    std::vector<const ConciseSet<false>*> con_ship_ptrs;
    ConciseSet<false> con_full;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_ship.resize(SHIPDATE_MAX + 1);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            con_ship[d] = bm_bench::load_concise_from_croaring(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        con_rf.resize(RF_COUNT);
        for (int i = 0; i < RF_COUNT; i++)
            con_rf[i] = bm_bench::load_concise_from_croaring(CROARING_DIR + "/returnflag/" + std::to_string(i) + ".bm");
        con_ls.resize(LS_COUNT);
        for (int i = 0; i < LS_COUNT; i++)
            con_ls[i] = bm_bench::load_concise_from_croaring(CROARING_DIR + "/linestatus/" + std::to_string(i) + ".bm");
        con_ship_ptrs.reserve(ship_or_count);
        for (int d = ship_or_start; d <= SHIPDATE_MAX; d++)
            con_ship_ptrs.push_back(&con_ship[d]);
        // Universe over [0, num_rows) — monotone increasing adds hit the
        // Concise fast path that compresses straight into a fill word.
        for (uint32_t i = 0; i < num_rows; i++) con_full.add(i);
        con_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:      " << wah_load_ms << " ms" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:   " << cb_load_ms  << " ms" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load: " << cr_load_ms  << " ms" << std::endl;
    if (run_crr()) std::cout << "  CRR load:      " << crr_load_ms << " ms" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:     " << ew_load_ms  << " ms" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:   " << bs_load_ms  << " ms (shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:  " << con_load_ms << " ms (incl. universe build)" << std::endl;

    // Per-backend on-disk footprint (useful in every mode, not just ALL).
    std::cout << std::fixed << std::setprecision(2);
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(WAH_DIR))      << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(COMBIT_DIR))   << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(CROARING_DIR)) << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(EWAH_DIR))     << " MiB" << std::endl;
    if (run_bs() || run_bsa()) {
        size_t b_bytes = 0;
        for (auto& b : bs_ship) b_bytes += b.nwords * sizeof(uint64_t);
        for (auto& b : bs_rf)   b_bytes += b.nwords * sizeof(uint64_t);
        for (auto& b : bs_ls)   b_bytes += b.nwords * sizeof(uint64_t);
        std::cout << "  Bitset in mem:    " << bm_bench::mib(b_bytes) << " MiB (rebuilt from CRoaring at load)" << std::endl;
    }
    if (run_con()) {
        size_t c_bytes = con_full.sizeInBytes();
        for (auto& c : con_ship) c_bytes += c.sizeInBytes();
        for (auto& c : con_rf)   c_bytes += c.sizeInBytes();
        for (auto& c : con_ls)   c_bytes += c.sizeInBytes();
        std::cout << "  Concise in mem:   " << bm_bench::mib(c_bytes) << " MiB (rebuilt from CRoaring + universe)" << std::endl;
    }

    // ============================================================
    // 2. Benchmark loop
    // ============================================================
    // Per-format timing vectors
    std::vector<double> cb_or_times, cb_not_times, cb_agg_times, cb_total_times;
    std::vector<double> cr_or_times, cr_not_times, cr_agg_times, cr_total_times;
    std::vector<double> crr_or_times, crr_not_times, crr_agg_times, crr_total_times;
    std::vector<double> wah_or_times, wah_not_times, wah_agg_times, wah_total_times;
    std::vector<double> ew_or_times, ew_not_times, ew_agg_times, ew_total_times;
    std::vector<double> bs_or_times, bs_not_times, bs_agg_times, bs_total_times;
    std::vector<double> bsa_or_times, bsa_not_times, bsa_agg_times, bsa_total_times;
    std::vector<double> con_or_times, con_not_times, con_agg_times, con_total_times;

    std::map<std::pair<char,char>, Q1Group> cb_results, cr_results, crr_results, wah_results, ew_results;
    std::map<std::pair<char,char>, Q1Group> bs_results, bsa_results, con_results;
    size_t cb_total_rows = 0, cr_total_rows = 0, crr_total_rows = 0, wah_total_rows = 0, ew_total_rows = 0;
    size_t bs_total_rows = 0, bsa_total_rows = 0, con_total_rows = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        bool is_warmup = (iter < WARMUP_RUNS);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << NUM_ITERATIONS
                  << (is_warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // ========== WAH ==========
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ibis::bitvector wah_future = wah_ship[ship_or_start];
            wah_future.decompress();
            for (int d = ship_or_start + 1; d <= SHIPDATE_MAX; d++)
                wah_future |= wah_ship[d];
            auto t1 = std::chrono::high_resolution_clock::now();

            wah_flip(&wah_future);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    ibis::bitvector grp;
                    grp.copy(wah_future);
                    grp &= wah_rf[r];
                    grp &= wah_ls[l];

                    Q1Group g;
                    ibis::bitvector::pit pit(grp);
                    while (*pit != 0xFFFFFFFFU) {
                        Q1_AGG_ROW(g, *pit, col_qty.data(), col_price.data(),
                                   col_disc.data(), col_tax.data());
                        pit.next();
                    }

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            wah_results = groups;
            wah_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  WAH:  OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                wah_or_times.push_back(d_or);
                wah_not_times.push_back(d_not);
                wah_agg_times.push_back(d_and_agg);
                wah_total_times.push_back(d_total);
            }
        }

        // ========== ComBit ==========
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // OR shipdate complement: days (cutoff+1)..max via OR_many.
            // Pointer vector `cb_ship_ptrs` was pre-built outside the loop.
            ComBit cb_future = ComBit::OR_many(cb_ship_ptrs.size(),
                                               cb_ship_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            // NOT → shipdate filter
            ComBit cb_filter = ~cb_future;
            auto t2 = std::chrono::high_resolution_clock::now();

            // AND per group + direct byte-scan aggregate (zero-copy from expanded segments)
            // ComBit pipeline: copy Decompressed `cb_filter` (produced by ~) as the
            // per-cell accumulator, then two consecutive in-place `&=` calls.  This
            // is the canonical Q12/Q14 pattern: `ComBitBtv::operator&=` asserts
            // `state == Decompressed` and runs the dedicated AVX-512 VBMI2 in-place
            // AND fast path.  The free `operator&` works correctly on a Decompressed
            // LHS (the assert is FIXME and disabled in release) but incurs an extra
            // per-region L2/L1 `expandloadu` + `popcount` offset walk on the LHS
            // that the `&=` path skips — unnecessary work on an already-Decompressed
            // accumulator.  (Measured: Q14 went from AND_pro=3.35 to 1.60 ms with
            // the same fix.)
            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    ComBit grp = cb_filter;
                    grp &= cb_rf[r];
                    grp &= cb_ls[l];

                    Q1Group g;
                    const int64_t* qp = col_qty.data();
                    const int64_t* pp = col_price.data();
                    const int64_t* dp = col_disc.data();
                    const int64_t* tp = col_tax.data();
                    size_t row_base = 0;
                    for (size_t s = 0; s < grp.num_segments(); s++) {
                        const auto& seg = grp.segment(s);
                        const uint8_t* data = seg.l1_literal_data();
                        size_t n = seg.num_literals();
                        for (size_t bi = 0; bi < n; bi++) {
                            uint8_t b = data[bi];
                            if (b == 0) { row_base += 8; continue; }
                            const auto& entry = byte_lut[b];
                            for (int k = 0; k < entry.count; k++)
                                Q1_AGG_ROW(g, row_base + entry.pos[k], qp, pp, dp, tp);
                            row_base += 8;
                        }
                    }

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            cb_results = groups;
            cb_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  CB:   OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                cb_or_times.push_back(d_or);
                cb_not_times.push_back(d_not);
                cb_agg_times.push_back(d_and_agg);
                cb_total_times.push_back(d_total);
            }
        }

        // ========== CRoaring ==========
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            roaring::Roaring cr_future = cr_ship[ship_or_start];
            for (int d = ship_or_start + 1; d <= SHIPDATE_MAX; d++)
                cr_future |= cr_ship[d];
            auto t1 = std::chrono::high_resolution_clock::now();

            roaring::Roaring cr_filter = cr_future;
            cr_filter.flip(0, static_cast<uint64_t>(num_rows));
            auto t2 = std::chrono::high_resolution_clock::now();

            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    roaring::Roaring grp = cr_filter & cr_rf[r];
                    grp &= cr_ls[l];

                    Q1Group g;
                    for (auto it = grp.begin(); it != grp.end(); ++it)
                        Q1_AGG_ROW(g, *it, col_qty.data(), col_price.data(),
                                   col_disc.data(), col_tax.data());

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            cr_results = groups;
            cr_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  CR:   OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                cr_or_times.push_back(d_or);
                cr_not_times.push_back(d_not);
                cr_agg_times.push_back(d_and_agg);
                cr_total_times.push_back(d_total);
            }
        }

        // ========== CRoaring+Run (fastunion) ==========
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            roaring::Roaring crr_future = roaring::Roaring::fastunion(crr_ship_ptrs.size(), crr_ship_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            roaring::Roaring crr_filter = crr_future;
            crr_filter.flip(0, static_cast<uint64_t>(num_rows));
            auto t2 = std::chrono::high_resolution_clock::now();

            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    roaring::Roaring grp = crr_filter & crr_rf[r];
                    grp &= crr_ls[l];

                    Q1Group g;
                    for (auto it = grp.begin(); it != grp.end(); ++it)
                        Q1_AGG_ROW(g, *it, col_qty.data(), col_price.data(),
                                   col_disc.data(), col_tax.data());

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            crr_results = groups;
            crr_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  CRR:  OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                crr_or_times.push_back(d_or);
                crr_not_times.push_back(d_not);
                crr_agg_times.push_back(d_and_agg);
                crr_total_times.push_back(d_total);
            }
        }

        // ========== EWAH ==========
        if (run_ew()) {
            // fast_logicalor: priority-queue k-way merge (ewah-inl.h:1129).
            auto t0 = std::chrono::high_resolution_clock::now();

            ewah::EWAHBoolArray<uint64_t> ew_future = ewah::fast_logicalor(
                ew_ship_ptrs.size(), ew_ship_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ew_future.padWithZeroes(num_rows);
            ew_future.inplace_logicalnot();
            auto t2 = std::chrono::high_resolution_clock::now();

            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    ewah::EWAHBoolArray<uint64_t> tmp1, grp;
                    ew_future.logicaland(ew_rf[r], tmp1);
                    tmp1.logicaland(ew_ls[l], grp);

                    Q1Group g;
                    for (auto it = grp.begin(); it != grp.end(); ++it)
                        Q1_AGG_ROW(g, *it, col_qty.data(), col_price.data(),
                                   col_disc.data(), col_tax.data());

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            ew_results = groups;
            ew_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  EW:   OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                ew_or_times.push_back(d_or);
                ew_not_times.push_back(d_not);
                ew_agg_times.push_back(d_and_agg);
                ew_total_times.push_back(d_total);
            }
        }

        // ========== Bitset (scalar) ==========
        // Pure baseline: clone first shipdate bitmap, scalar-OR the rest in,
        // then scalar-NOT (with tail clear), per group clone+AND+AND+walk-agg.
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bs_future = bs_ship[ship_or_start].clone();
            for (int d = ship_or_start + 1; d <= SHIPDATE_MAX; d++)
                bs::or_inplace(bs_future, bs_ship[d], false);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::not_inplace(bs_future, false);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    bs::Bitmap grp = bs_future.clone();
                    bs::and_inplace(grp, bs_rf[r], false);
                    bs::and_inplace(grp, bs_ls[l], false);

                    Q1Group g;
                    const int64_t* qp = col_qty.data();
                    const int64_t* pp = col_price.data();
                    const int64_t* dp = col_disc.data();
                    const int64_t* tp = col_tax.data();
                    for (size_t i = 0; i < grp.nwords; ++i) {
                        uint64_t w = grp.words[i];
                        const size_t base = i * 64;
                        while (w) {
                            size_t row = base + __builtin_ctzll(w);
                            if (row >= grp.nbits) break;
                            Q1_AGG_ROW(g, row, qp, pp, dp, tp);
                            w &= w - 1;
                        }
                    }

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            bs_results = groups;
            bs_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  BS:   OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                bs_or_times.push_back(d_or);
                bs_not_times.push_back(d_not);
                bs_agg_times.push_back(d_and_agg);
                bs_total_times.push_back(d_total);
            }
        }

        // ========== Bitset + AVX-512 ==========
        // Same data as BS; simd flag flipped on for OR / NOT / AND.
        // Aggregation walk is unchanged (ctz over result words).
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bsa_future = bs_ship[ship_or_start].clone();
            for (int d = ship_or_start + 1; d <= SHIPDATE_MAX; d++)
                bs::or_inplace(bsa_future, bs_ship[d], true);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::not_inplace(bsa_future, true);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    bs::Bitmap grp = bsa_future.clone();
                    bs::and_inplace(grp, bs_rf[r], true);
                    bs::and_inplace(grp, bs_ls[l], true);

                    Q1Group g;
                    const int64_t* qp = col_qty.data();
                    const int64_t* pp = col_price.data();
                    const int64_t* dp = col_disc.data();
                    const int64_t* tp = col_tax.data();
                    for (size_t i = 0; i < grp.nwords; ++i) {
                        uint64_t w = grp.words[i];
                        const size_t base = i * 64;
                        while (w) {
                            size_t row = base + __builtin_ctzll(w);
                            if (row >= grp.nbits) break;
                            Q1_AGG_ROW(g, row, qp, pp, dp, tp);
                            w &= w - 1;
                        }
                    }

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            bsa_results = groups;
            bsa_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  BSA:  OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                bsa_or_times.push_back(d_or);
                bsa_not_times.push_back(d_not);
                bsa_agg_times.push_back(d_and_agg);
                bsa_total_times.push_back(d_total);
            }
        }

        // ========== Concise ==========
        // OR via fast_logicalor (priority-queue k-way merge).
        // NOT emulated as `con_full \ future` (Concise has no native flip);
        // the universe was built once at load.
        // AND + Agg per group uses logicaland (returns fresh container).
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_future = ConciseSet<false>::fast_logicalor(
                con_ship_ptrs.size(), con_ship_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_filter = con_full.logicalandnot(con_future);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::map<std::pair<char,char>, Q1Group> groups;
            size_t total_rows = 0;
            for (int r = 0; r < RF_COUNT; r++) {
                for (int l = 0; l < LS_COUNT; l++) {
                    ConciseSet<false> grp = con_filter.logicaland(con_rf[r])
                                                       .logicaland(con_ls[l]);

                    Q1Group g;
                    for (auto it = grp.begin(); it != grp.end(); ++it)
                        Q1_AGG_ROW(g, *it, col_qty.data(), col_price.data(),
                                   col_disc.data(), col_tax.data());

                    if (g.count_order == 0) continue;
                    auto key = std::make_pair(RF_CHARS[r], LS_CHARS[l]);
                    groups[key] = g;
                    total_rows += g.count_order;
                }
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            con_results = groups;
            con_total_rows = total_rows;

            double d_or = ms(t0,t1), d_not = ms(t1,t2), d_and_agg = ms(t2,t3), d_total = ms(t0,t3);
            std::cout << "  CON:  OR_ship=" << d_or << "  NOT=" << d_not
                      << "  AND+Agg=" << d_and_agg << "  Total=" << d_total
                      << "  rows=" << total_rows << std::endl;

            if (!is_warmup) {
                con_or_times.push_back(d_or);
                con_not_times.push_back(d_not);
                con_agg_times.push_back(d_and_agg);
                con_total_times.push_back(d_total);
            }
        }
    } // end iterations

    // ============================================================
    // 3. Correctness validation — print Q1 results
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Q1 Correctness Validation" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "  Row counts:";
    if (run_wah()) std::cout << " WAH=" << wah_total_rows;
    if (run_cb())  std::cout << " CB="  << cb_total_rows;
    if (run_cr())  std::cout << " CR="  << cr_total_rows;
    if (run_crr()) std::cout << " CRR=" << crr_total_rows;
    if (run_ew())  std::cout << " EW="  << ew_total_rows;
    if (run_bs())  std::cout << " BS="  << bs_total_rows;
    if (run_bsa()) std::cout << " BSA=" << bsa_total_rows;
    if (run_con()) std::cout << " CON=" << con_total_rows;
    std::cout << std::endl;

    // Pick a backend whose aggregated groups we'll print.  In ALL
    // mode we keep the historical "from ComBit"; otherwise we print
    // from the selected backend.
    const std::map<std::pair<char,char>, Q1Group>* print_src = nullptr;
    const char* print_src_label = "";
    if      (Q1_BM == Q1BmType::ALL || Q1_BM == Q1BmType::CB)  { print_src = &cb_results;  print_src_label = "ComBit"; }
    else if (Q1_BM == Q1BmType::WAH)                           { print_src = &wah_results; print_src_label = "WAH";    }
    else if (Q1_BM == Q1BmType::CR)                            { print_src = &cr_results;  print_src_label = "CRoaring"; }
    else if (Q1_BM == Q1BmType::CRR)                           { print_src = &crr_results; print_src_label = "CRoaring+Run"; }
    else if (Q1_BM == Q1BmType::EW)                            { print_src = &ew_results;  print_src_label = "EWAH"; }
    else if (Q1_BM == Q1BmType::BS)                            { print_src = &bs_results;  print_src_label = "Bitset"; }
    else if (Q1_BM == Q1BmType::BSA)                           { print_src = &bsa_results; print_src_label = "Bitset+AVX512"; }
    else if (Q1_BM == Q1BmType::CON)                           { print_src = &con_results; print_src_label = "Concise"; }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  Q1 Results (from " << print_src_label << "):" << std::endl;
    std::cout << "  rf ls  sum_qty       sum_base_price  sum_disc_price     sum_charge        avg_qty  avg_price  avg_disc  count" << std::endl;
    if (print_src) {
        for (auto& [key, g] : *print_src) {
            std::cout << "  " << key.first << "  " << key.second
                      << "  " << std::setw(12) << (double)g.sum_qty / 100
                      << "  " << std::setw(14) << (double)g.sum_base_price / 100
                      << "  " << std::setw(16) << (double)g.sum_disc_price / 10000
                      << "  " << std::setw(18) << (double)g.sum_charge / 1000000
                      << "  " << std::setw(8) << (double)g.sum_qty / g.count_order / 100
                      << "  " << std::setw(9) << (double)g.sum_base_price / g.count_order / 100
                      << "  " << std::setw(8) << (double)g.sum_discount / g.count_order / 100
                      << "  " << std::setw(6) << g.count_order
                      << std::endl;
        }
    }

    // Cross-backend consistency check (only meaningful in ALL mode —
    // single-backend mode has nothing to compare against).  Compares
    // every integer field of every group against ComBit's result.
    if (run_all()) {
        bool consistent = true;
        auto group_eq = [](const Q1Group& a, const Q1Group& b) {
            return a.sum_qty == b.sum_qty && a.sum_base_price == b.sum_base_price
                && a.sum_disc_price == b.sum_disc_price && a.sum_charge == b.sum_charge
                && a.sum_discount == b.sum_discount && a.count_order == b.count_order;
        };
        for (auto& [key, g] : cb_results) {
            if (!group_eq(g, wah_results[key]) || !group_eq(g, cr_results[key])
             || !group_eq(g, crr_results[key]) || !group_eq(g, ew_results[key])
             || !group_eq(g, bs_results[key])  || !group_eq(g, bsa_results[key])
             || !group_eq(g, con_results[key])) {
                consistent = false;
                std::cout << "  *** MISMATCH for (" << key.first << "," << key.second << ") ***" << std::endl;
            }
        }
        std::cout << "  Consistency: " << (consistent ? "ALL MATCH" : "MISMATCH DETECTED") << std::endl;
    }

    // ============================================================
    // 4. Statistics summary
    // ============================================================
    auto wah_or_s = compute_stats(wah_or_times);
    auto wah_not_s = compute_stats(wah_not_times);
    auto wah_agg_s = compute_stats(wah_agg_times);
    auto wah_total_s = compute_stats(wah_total_times);

    auto cb_or_s = compute_stats(cb_or_times);
    auto cb_not_s = compute_stats(cb_not_times);
    auto cb_agg_s = compute_stats(cb_agg_times);
    auto cb_total_s = compute_stats(cb_total_times);

    auto cr_or_s = compute_stats(cr_or_times);
    auto cr_not_s = compute_stats(cr_not_times);
    auto cr_agg_s = compute_stats(cr_agg_times);
    auto cr_total_s = compute_stats(cr_total_times);

    auto crr_or_s = compute_stats(crr_or_times);
    auto crr_not_s = compute_stats(crr_not_times);
    auto crr_agg_s = compute_stats(crr_agg_times);
    auto crr_total_s = compute_stats(crr_total_times);

    auto ew_or_s = compute_stats(ew_or_times);
    auto ew_not_s = compute_stats(ew_not_times);
    auto ew_agg_s = compute_stats(ew_agg_times);
    auto ew_total_s = compute_stats(ew_total_times);

    auto bs_or_s     = compute_stats(bs_or_times);
    auto bs_not_s    = compute_stats(bs_not_times);
    auto bs_agg_s    = compute_stats(bs_agg_times);
    auto bs_total_s  = compute_stats(bs_total_times);

    auto bsa_or_s    = compute_stats(bsa_or_times);
    auto bsa_not_s   = compute_stats(bsa_not_times);
    auto bsa_agg_s   = compute_stats(bsa_agg_times);
    auto bsa_total_s = compute_stats(bsa_total_times);

    auto con_or_s    = compute_stats(con_or_times);
    auto con_not_s   = compute_stats(con_not_times);
    auto con_agg_s   = compute_stats(con_agg_times);
    auto con_total_s = compute_stats(con_total_times);

    int measured = NUM_ITERATIONS - WARMUP_RUNS;

    // --- DuckDB native SQL baseline + ground-truth row count ---
    // Runs AFTER the bitmap iteration loop so it does not pollute the
    // CPU cache and skew bitmap measurements.  Separate Connection
    // avoids re-entrancy against the outer pragma.
    int64_t gt_rows = -1;
    double  gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT count(*) FROM lineitem "
            "WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY";
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

    // --- Ground-truth assert: every active backend's total_rows (sum over groups) must match SQL ---
    if (gt_rows >= 0) {
        auto check = [&](const char* label, bool active, size_t got) {
            if (!active) return;
            if (static_cast<int64_t>(got) == gt_rows) return;
            std::ostringstream oss;
            oss << "[FAIL] Q1 " << label << " total rows " << got
                << " != DuckDB SQL ground truth " << gt_rows
                << " — bitmap pipeline is incorrect";
            throw std::runtime_error(oss.str());
        };
        check("WAH",           run_wah(), wah_total_rows);
        check("ComBit",        run_cb(),  cb_total_rows);
        check("CRoaring",      run_cr(),  cr_total_rows);
        check("CRoaring+Run",  run_crr(), crr_total_rows);
        check("EWAH",          run_ew(),  ew_total_rows);
        check("Bitset",        run_bs(),  bs_total_rows);
        check("Bitset+AVX512", run_bsa(), bsa_total_rows);
        check("Concise",       run_con(), con_total_rows);
        std::cout << "[OK] all active backends match DuckDB SQL ground truth ("
                  << gt_rows << " rows)." << std::endl;
    }


    if (run_all()) {
        // ============================================================
        // ALL mode: full 5-column comparison table
        // ============================================================
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q1 RESULTS (" << measured << " measured iterations, median \u00b1 stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                       WAH (ms)              ComBit (ms)         CRoaring (ms)      CRoar+Run (ms)       EWAH (ms)          CB vs WAH   CR vs WAH  CRR vs WAH   EW vs WAH" << std::endl;
        std::cout << "  --------------------------------------------------------------------------------------------------------------------------------" << std::endl;

        auto print_row = [](const char* label, Q1TimingStats& w, Q1TimingStats& c,
                            Q1TimingStats& r, Q1TimingStats& rr, Q1TimingStats& e) {
            double cb_speedup  = (c.median  > 0) ? w.median / c.median  : 0;
            double cr_speedup  = (r.median  > 0) ? w.median / r.median  : 0;
            double crr_speedup = (rr.median > 0) ? w.median / rr.median : 0;
            double ew_speedup  = (e.median  > 0) ? w.median / e.median  : 0;
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(8) << w.median  << " \u00b1 " << std::setw(6) << w.stddev
                      << "     "   << std::setw(8) << c.median  << " \u00b1 " << std::setw(6) << c.stddev
                      << "     "   << std::setw(8) << r.median  << " \u00b1 " << std::setw(6) << r.stddev
                      << "     "   << std::setw(8) << rr.median << " \u00b1 " << std::setw(6) << rr.stddev
                      << "     "   << std::setw(8) << e.median  << " \u00b1 " << std::setw(6) << e.stddev
                      << "     "   << std::setw(5) << cb_speedup  << "x"
                      << "     "   << std::setw(5) << cr_speedup  << "x"
                      << "     "   << std::setw(5) << crr_speedup << "x"
                      << "     "   << std::setw(5) << ew_speedup  << "x" << std::endl;
        };

        print_row("OR_ship", wah_or_s, cb_or_s, cr_or_s, crr_or_s, ew_or_s);
        print_row("NOT",     wah_not_s, cb_not_s, cr_not_s, crr_not_s, ew_not_s);
        print_row("AND+Agg", wah_agg_s, cb_agg_s, cr_agg_s, crr_agg_s, ew_agg_s);
        std::cout << "  --------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL",   wah_total_s, cb_total_s, cr_total_s, crr_total_s, ew_total_s);

        std::cout << "\n  WAH rows:        " << wah_total_rows << std::endl;
        std::cout << "  ComBit rows:     " << cb_total_rows  << std::endl;
        std::cout << "  CRoaring rows:   " << cr_total_rows  << std::endl;
        std::cout << "  CRoar+Run rows:  " << crr_total_rows << std::endl;
        std::cout << "  EWAH rows:       " << ew_total_rows  << std::endl;
        std::cout << "================================================================\n" << std::endl;

        // ============================================================
        // ALL mode: baseline backends table (BS / BSA / Concise).
        // Printed separately so the 5-way table above stays readable;
        // speedups anchor on WAH for parity with that table.
        // ============================================================
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q1 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                       BS (ms)              BSA (ms)            Concise (ms)        BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  -------------------------------------------------------------------------------------------------------------" << std::endl;

        auto print_baseline_row = [](const char* label, Q1TimingStats& w,
                                     Q1TimingStats& b, Q1TimingStats& ba, Q1TimingStats& c) {
            double bs_sp  = (b.median  > 0) ? w.median / b.median  : 0;
            double bsa_sp = (ba.median > 0) ? w.median / ba.median : 0;
            double con_sp = (c.median  > 0) ? w.median / c.median  : 0;
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(8) << b.median  << " \u00b1 " << std::setw(6) << b.stddev
                      << "     " << std::setw(8) << ba.median << " \u00b1 " << std::setw(6) << ba.stddev
                      << "     " << std::setw(8) << c.median  << " \u00b1 " << std::setw(6) << c.stddev
                      << "     " << std::setw(5) << bs_sp  << "x"
                      << "     " << std::setw(5) << bsa_sp << "x"
                      << "     " << std::setw(5) << con_sp << "x" << std::endl;
        };
        print_baseline_row("OR_ship", wah_or_s,    bs_or_s,    bsa_or_s,    con_or_s);
        print_baseline_row("NOT",     wah_not_s,   bs_not_s,   bsa_not_s,   con_not_s);
        print_baseline_row("AND+Agg", wah_agg_s,   bs_agg_s,   bsa_agg_s,   con_agg_s);
        std::cout << "  -------------------------------------------------------------------------------------------------------------" << std::endl;
        print_baseline_row("TOTAL",   wah_total_s, bs_total_s, bsa_total_s, con_total_s);

        std::cout << "\n  Bitset rows:         " << bs_total_rows  << std::endl;
        std::cout << "  Bitset+AVX512 rows:  " << bsa_total_rows << std::endl;
        std::cout << "  Concise rows:        " << con_total_rows << std::endl;
        std::cout << "================================================================\n" << std::endl;
    } else {
        // ============================================================
        // Single-backend mode: focused per-phase summary for just
        // the selected backend (median / stddev / min / max).
        // ============================================================
        Q1TimingStats *sel_or = nullptr, *sel_not = nullptr, *sel_agg = nullptr, *sel_tot = nullptr;
        size_t sel_rows = 0;
        switch (Q1_BM) {
            case Q1BmType::WAH:
                sel_or = &wah_or_s; sel_not = &wah_not_s; sel_agg = &wah_agg_s; sel_tot = &wah_total_s;
                sel_rows = wah_total_rows; break;
            case Q1BmType::CB:
                sel_or = &cb_or_s;  sel_not = &cb_not_s;  sel_agg = &cb_agg_s;  sel_tot = &cb_total_s;
                sel_rows = cb_total_rows;  break;
            case Q1BmType::CR:
                sel_or = &cr_or_s;  sel_not = &cr_not_s;  sel_agg = &cr_agg_s;  sel_tot = &cr_total_s;
                sel_rows = cr_total_rows;  break;
            case Q1BmType::CRR:
                sel_or = &crr_or_s; sel_not = &crr_not_s; sel_agg = &crr_agg_s; sel_tot = &crr_total_s;
                sel_rows = crr_total_rows; break;
            case Q1BmType::EW:
                sel_or = &ew_or_s;  sel_not = &ew_not_s;  sel_agg = &ew_agg_s;  sel_tot = &ew_total_s;
                sel_rows = ew_total_rows;  break;
            case Q1BmType::BS:
                sel_or = &bs_or_s;  sel_not = &bs_not_s;  sel_agg = &bs_agg_s;  sel_tot = &bs_total_s;
                sel_rows = bs_total_rows;  break;
            case Q1BmType::BSA:
                sel_or = &bsa_or_s; sel_not = &bsa_not_s; sel_agg = &bsa_agg_s; sel_tot = &bsa_total_s;
                sel_rows = bsa_total_rows; break;
            case Q1BmType::CON:
                sel_or = &con_or_s; sel_not = &con_not_s; sel_agg = &con_agg_s; sel_tot = &con_total_s;
                sel_rows = con_total_rows; break;
            case Q1BmType::ALL: break;  // unreachable
        }

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q1 RESULTS — " << q1_bm_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;

        auto print_single = [](const char* label, Q1TimingStats& s) {
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(9) << s.median
                      << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val
                      << std::setw(10) << s.max_val << std::endl;
        };
        if (sel_or) {
            print_single("OR_ship",    *sel_or);
            print_single("NOT",        *sel_not);
            print_single("AND+Agg",    *sel_agg);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            print_single("TOTAL",      *sel_tot);
            std::cout << "\n  " << q1_bm_label() << " rows: " << sel_rows << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }

    // ============================================================
    // 5. CSV export — ALL mode only (the 26-column schema requires
    //    every backend's stats).  Single-backend runs are iterative
    //    dev tools, not inputs for cross-SF reporting.
    // ============================================================
    if (run_all()) {
        std::string sf_label = get_sf_label();
        std::string csv_path = "q1_results_" + sf_label + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << std::fixed << std::setprecision(4);
            csv << "sf,operation,"
                << "wah_median_ms,wah_stddev_ms,wah_min_ms,wah_max_ms,"
                << "combit_median_ms,combit_stddev_ms,combit_min_ms,combit_max_ms,"
                << "croaring_median_ms,croaring_stddev_ms,croaring_min_ms,croaring_max_ms,"
                << "croaring_run_median_ms,croaring_run_stddev_ms,croaring_run_min_ms,croaring_run_max_ms,"
                << "ewah_median_ms,ewah_stddev_ms,ewah_min_ms,ewah_max_ms,"
                << "bs_median_ms,bs_stddev_ms,bs_min_ms,bs_max_ms,"
                << "bsa_median_ms,bsa_stddev_ms,bsa_min_ms,bsa_max_ms,"
                << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms,"
                << "cb_vs_wah,cr_vs_wah,crr_vs_wah,ew_vs_wah,"
                << "bs_vs_wah,bsa_vs_wah,con_vs_wah\n";

            auto csv_row = [&](const std::string& op,
                               Q1TimingStats& w, Q1TimingStats& c, Q1TimingStats& r,
                               Q1TimingStats& rr, Q1TimingStats& e,
                               Q1TimingStats& b, Q1TimingStats& ba, Q1TimingStats& co) {
                double cb_speedup  = (c.median  > 0) ? w.median / c.median  : 0;
                double cr_speedup  = (r.median  > 0) ? w.median / r.median  : 0;
                double crr_speedup = (rr.median > 0) ? w.median / rr.median : 0;
                double ew_speedup  = (e.median  > 0) ? w.median / e.median  : 0;
                double bs_speedup  = (b.median  > 0) ? w.median / b.median  : 0;
                double bsa_speedup = (ba.median > 0) ? w.median / ba.median : 0;
                double con_speedup = (co.median > 0) ? w.median / co.median : 0;
                csv << sf_label << "," << op << ","
                    << w.median  << "," << w.stddev  << "," << w.min_val  << "," << w.max_val  << ","
                    << c.median  << "," << c.stddev  << "," << c.min_val  << "," << c.max_val  << ","
                    << r.median  << "," << r.stddev  << "," << r.min_val  << "," << r.max_val  << ","
                    << rr.median << "," << rr.stddev << "," << rr.min_val << "," << rr.max_val << ","
                    << e.median  << "," << e.stddev  << "," << e.min_val  << "," << e.max_val  << ","
                    << b.median  << "," << b.stddev  << "," << b.min_val  << "," << b.max_val  << ","
                    << ba.median << "," << ba.stddev << "," << ba.min_val << "," << ba.max_val << ","
                    << co.median << "," << co.stddev << "," << co.min_val << "," << co.max_val << ","
                    << cb_speedup << "," << cr_speedup << "," << crr_speedup << "," << ew_speedup << ","
                    << bs_speedup << "," << bsa_speedup << "," << con_speedup << "\n";
            };

            csv_row("OR_ship", wah_or_s,    cb_or_s,    cr_or_s,    crr_or_s,    ew_or_s,    bs_or_s,    bsa_or_s,    con_or_s);
            csv_row("NOT",     wah_not_s,   cb_not_s,   cr_not_s,   crr_not_s,   ew_not_s,   bs_not_s,   bsa_not_s,   con_not_s);
            csv_row("AND+Agg", wah_agg_s,   cb_agg_s,   cr_agg_s,   crr_agg_s,   ew_agg_s,   bs_agg_s,   bsa_agg_s,   con_agg_s);
            csv_row("TOTAL",   wah_total_s, cb_total_s, cr_total_s, crr_total_s, ew_total_s, bs_total_s, bsa_total_s, con_total_s);

            csv.close();
            std::cout << "  [CSV] Results written to: " << csv_path << std::endl;
        }
    }

    }); // end call_once
}

} // namespace duckdb
