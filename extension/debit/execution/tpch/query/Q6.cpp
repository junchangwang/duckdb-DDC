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
#include "duckdb/common/types/data_chunk.hpp"

// Direct ComBit and WAH (FastBit) includes — no Rabit dependency
#include "combit_adapter.h"
#include "combit/include/combit.h"
#include "fastbit/bitvector.h"

// CRoaring
#include "roaring.hh"

// EWAH
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
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cctype>

namespace duckdb {

// --- Q6 bitmap directories ---
// DEBIT_BITMAP_DIR (if set) supplies an absolute base; unset = cwd-relative.
// TPCH_SF selects the scale factor suffix (default: 10 = no suffix).
static const std::string SF_SUFFIX    = bm_bench::sf_suffix();
static const std::string COMBIT_DIR   = bm_bench::resolve_bitmap_dir("tpch_q6" + SF_SUFFIX + "_combit");
static const std::string WAH_DIR      = bm_bench::resolve_bitmap_dir("tpch_q6" + SF_SUFFIX + "_wah");
static const std::string CROARING_DIR = bm_bench::resolve_bitmap_dir("tpch_q6" + SF_SUFFIX + "_croaring");
static const std::string EWAH_DIR     = bm_bench::resolve_bitmap_dir("tpch_q6" + SF_SUFFIX + "_ewah");

// --- Q6 predicate parameters (TPC-H spec §2.4.6) ---
// l_discount BETWEEN 0.05 AND 0.07  (fixed-point: 5..7)
// l_quantity  <  24                 (fixed-point: 1..23)
// l_shipdate  >= '1994-01-01' AND l_shipdate < '1995-01-01'
// Shipdate encoding: days since 1992-01-01 (1992 = leap year).
static const int DISCOUNT_MIN       = 5;
static const int DISCOUNT_MAX       = 7;
static const int QUANTITY_MAX       = 23;
static const int SHIPDATE_DAY_START = 731;   // 1994-01-01
static const int SHIPDATE_DAY_END   = 1096;  // 1995-01-01 (exclusive)

// --- Iteration counts (override via DEBIT_ITER / DEBIT_WARMUP) ---
static const int NUM_ITERATIONS = bm_bench::iter_count(10);
static const int WARMUP_RUNS    = bm_bench::warmup_count(2);

// --- Backend selection ---
// DEBIT_BM=all|wah|cb|cr|crr|ew  (legacy Q6_BM also honoured).
using Q6BmType = bm_bench::Backend;
static const Q6BmType Q6_BM = bm_bench::parse_backend("Q6_BM");

static bool run_all() { return Q6_BM == Q6BmType::ALL; }
static bool run_wah() { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::WAH; }
static bool run_cb()  { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::CB;  }
static bool run_cr()  { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::CR;  }
static bool run_crr() { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::CRR; }
static bool run_ew()  { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::EW;  }
static bool run_bs()  { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::BS;  }
static bool run_bsa() { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::BSA; }
static bool run_con() { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::CON; }

// Which backend's bitmap feeds DuckDB's downstream row_ids scan.
static bool wah_primary() { return Q6_BM == Q6BmType::WAH; }
static bool cb_primary()  { return Q6_BM == Q6BmType::ALL || Q6_BM == Q6BmType::CB; }
static bool cr_primary()  { return Q6_BM == Q6BmType::CR;  }
static bool crr_primary() { return Q6_BM == Q6BmType::CRR; }
static bool ew_primary()  { return Q6_BM == Q6BmType::EW;  }
static bool bs_primary()  { return Q6_BM == Q6BmType::BS;  }
static bool bsa_primary() { return Q6_BM == Q6BmType::BSA; }
static bool con_primary() { return Q6_BM == Q6BmType::CON; }

static const char* q6_bm_label()  { return bm_bench::backend_label(Q6_BM); }
static std::string get_sf_label() { return bm_bench::sf_label(); }

// Thread-safety
static std::once_flag q6_once_flag;
static std::mutex q6_fetch_mutex;

// WAH: load ibis::bitvector from our WAH .bm file
static ibis::bitvector load_wah_bm(const std::string& path) {
    ibis::bitvector btv;
    btv.read(path.c_str());
    return btv;
}

// ComBit: load from our ComBit .bm file
static ComBit load_combit_bm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Error: cannot open " << path << std::endl;
        return ComBit();
    }
    return ComBit::deserialize(in);
}

// CRoaring: load from our CRoaring .bm file
// File format: [uint32_t logical_size][roaring serialized data]
static roaring::Roaring load_croaring_bm(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "Error: cannot open " << path << std::endl;
        return roaring::Roaring();
    }
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

// EWAH: load from our EWAH .bm file
// File format: [uint64_t current_bits][EWAH compressed data]
static ewah::EWAHBoolArray<uint64_t> load_ewah_bm(const std::string& path) {
    ewah::EWAHBoolArray<uint64_t> btv;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Error: cannot open " << path << std::endl;
        return btv;
    }
    uint64_t current_bits;
    in.read(reinterpret_cast<char*>(&current_bits), sizeof(current_bits));
    btv.read(in);
    return btv;
}

// Bitset: GetRowIds — write out positions of set bits.
static void GetRowidsBS(const bs::Bitmap& bm, std::vector<int64_t>* row_ids) {
    bs::decode(bm, *row_ids);
}

// Concise: GetRowIds — iterate bit positions via the forward iterator.
static void GetRowidsConcise(const ConciseSet<false>& cs, std::vector<int64_t>* row_ids) {
    row_ids->clear();
    row_ids->reserve(cs.size());
    for (auto it = cs.begin(); it != cs.end(); ++it)
        row_ids->push_back(static_cast<int64_t>(*it));
}

// WAH: GetRowIds from ibis::bitvector
static void GetRowidsWAH(const ibis::bitvector& btv, std::vector<int64_t>* row_ids) {
    row_ids->clear();
    row_ids->reserve(btv.cnt());
    ibis::bitvector::pit it(btv);
    while (*it != 0xFFFFFFFFU) {
        row_ids->push_back(static_cast<int64_t>(*it));
        it.next();
    }
}

// CRoaring: GetRowIds from roaring::Roaring
static void GetRowidsCRoaring(const roaring::Roaring& bm, std::vector<int64_t>* row_ids) {
    row_ids->clear();
    uint64_t card = bm.cardinality();
    row_ids->resize(card);
    std::vector<uint32_t> tmp(card);
    bm.toUint32Array(tmp.data());
    for (uint64_t i = 0; i < card; i++) {
        (*row_ids)[i] = static_cast<int64_t>(tmp[i]);
    }
}

// EWAH: GetRowIds from EWAHBoolArray
static void GetRowidsEWAH(const ewah::EWAHBoolArray<uint64_t>& btv, std::vector<int64_t>* row_ids) {
    row_ids->clear();
    auto positions = btv.toArray();
    row_ids->reserve(positions.size());
    for (auto pos : positions) {
        row_ids->push_back(static_cast<int64_t>(pos));
    }
}

// Statistics helper
struct TimingStats {
    double median;
    double stddev;
    double min_val;
    double max_val;
};

static TimingStats compute_stats(std::vector<double>& vals) {
    TimingStats s{};
    if (vals.empty()) return s;
    std::sort(vals.begin(), vals.end());
    size_t n = vals.size();
    s.median = (n % 2 == 0) ? (vals[n/2-1] + vals[n/2]) / 2.0 : vals[n/2];
    s.min_val = vals.front();
    s.max_val = vals.back();
    double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / n;
    double sq_sum = 0;
    for (auto v : vals) sq_sum += (v - mean) * (v - mean);
    s.stddev = std::sqrt(sq_sum / n);
    return s;
}

void BMTableScan::TPCH_Q6_Lineitem_GetRowIds(ExecutionContext &context, vector<row_t> *row_ids)
{
    std::call_once(q6_once_flag, [&]() {

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    // Loud warning if the user is running on the known-duplicated SF1 db.
    bm_bench::warn_if_sf1();

    // ============================================================
    // System Information (only printed in ALL mode — single-backend
    // mode skips the CPU/memory/compiler block per the user's
    // preference for a focused, backend-specific report.)
    // ============================================================
    if (run_all()) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  SYSTEM INFORMATION" << std::endl;
        std::cout << "================================================================" << std::endl;

        // CPU model
        {
            std::ifstream cpuinfo("/proc/cpuinfo");
            std::string line;
            while (std::getline(cpuinfo, line)) {
                if (line.find("model name") != std::string::npos) {
                    std::cout << "  CPU: " << line.substr(line.find(':') + 2) << std::endl;
                    break;
                }
            }
        }
        // CPU cores
        {
            std::ifstream cpuinfo("/proc/cpuinfo");
            std::string line;
            int cores = 0;
            while (std::getline(cpuinfo, line))
                if (line.find("processor") == 0) cores++;
            std::cout << "  CPU cores: " << cores << std::endl;
        }
        // Memory
        {
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal") != std::string::npos) {
                    std::istringstream iss(line);
                    std::string label, val, unit;
                    iss >> label >> val >> unit;
                    double gb = std::stod(val) / (1024.0 * 1024.0);
                    std::cout << "  Memory: " << std::fixed << std::setprecision(1) << gb << " GB" << std::endl;
                    break;
                }
            }
        }
        // Compiler
#ifdef __VERSION__
        std::cout << "  Compiler: " << __VERSION__ << std::endl;
#endif
#ifdef __OPTIMIZE__
        std::cout << "  Optimization: enabled (-O)" << std::endl;
#endif
#ifdef __AVX512VBMI2__
        std::cout << "  AVX-512 VBMI2: enabled" << std::endl;
#endif
        std::cout << "  Sizeof(void*): " << sizeof(void*) << " (" << (sizeof(void*)*8) << "-bit)" << std::endl;
    }

    // ============================================================
    // Benchmark header — always printed, content adapts to mode.
    // ============================================================
    std::cout << "\n================================================================" << std::endl;
    if (run_all()) {
        std::cout << "  TPC-H Q6 Benchmark — ComBit vs WAH vs CRoaring vs EWAH ("
                  << get_sf_label() << ", in DuckDB)" << std::endl;
    } else {
        std::cout << "  TPC-H Q6 Benchmark — " << q6_bm_label() << " only ("
                  << get_sf_label() << ", in DuckDB)" << std::endl;
    }
    std::cout << "  Equality encoding on ALL columns (publication-grade)" << std::endl;
    // Only print dirs for backends we're actually running.
    std::cout << "  Bitmap dirs:";
    if (run_wah()) std::cout << " " << WAH_DIR;
    if (run_cb())  std::cout << " " << COMBIT_DIR;
    if (run_cr() || run_crr()) std::cout << " " << CROARING_DIR;
    if (run_ew())  std::cout << " " << EWAH_DIR;
    std::cout << std::endl;
    std::cout << "  Iterations: " << NUM_ITERATIONS << " (first " << WARMUP_RUNS << " = warm-up)" << std::endl;
    std::cout << "  OR discount: " << (DISCOUNT_MAX - DISCOUNT_MIN + 1) << " bitmaps" << std::endl;
    std::cout << "  OR quantity: " << QUANTITY_MAX << " bitmaps" << std::endl;
    std::cout << "  OR shipdate: " << (SHIPDATE_DAY_END - SHIPDATE_DAY_START) << " bitmaps" << std::endl;
    std::cout << "  TPC-H params: discount [0.05, 0.07], quantity < 24, shipdate [1994-01-01, 1995-01-01)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // ============================================================
    // Pre-load bitmaps once.  Each backend's load block is gated on
    // run_xxx() so single-backend mode only touches the one dir.
    // Vectors are declared at outer scope (default-empty) so they
    // stay alive through the iteration loop; unloaded backends'
    // vectors remain empty and their gated iteration block is
    // skipped in lock-step.
    // ============================================================
    std::cout << "\n[Load] Loading bitmaps (mode=" << q6_bm_label() << ")..." << std::endl;
    const int files_per_backend = 3 + QUANTITY_MAX + (SHIPDATE_DAY_END - SHIPDATE_DAY_START);

    // WAH bitmaps
    std::vector<ibis::bitvector> wah_disc, wah_qty, wah_ship;
    double wah_load_ms = 0;
    if (run_wah()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wah_disc.resize(DISCOUNT_MAX + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            wah_disc[v] = load_wah_bm(WAH_DIR + "/discount/" + std::to_string(v) + ".bm");
        wah_qty.resize(QUANTITY_MAX + 1);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            wah_qty[v] = load_wah_bm(WAH_DIR + "/quantity/" + std::to_string(v) + ".bm");
        wah_ship.resize(SHIPDATE_DAY_END);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            wah_ship[d] = load_wah_bm(WAH_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        wah_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // ComBit bitmaps + pointer vectors for OR_many (rebuilt out of
    // the timed region for fairness with CRR's crr_*_ptrs).
    std::vector<ComBit> cb_disc, cb_qty, cb_ship;
    std::vector<const ComBit*> cb_disc_ptrs, cb_qty_ptrs, cb_ship_ptrs;
    double cb_load_ms = 0;
    if (run_cb()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cb_disc.resize(DISCOUNT_MAX + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            cb_disc[v] = load_combit_bm(COMBIT_DIR + "/discount/" + std::to_string(v) + ".bm");
        cb_qty.resize(QUANTITY_MAX + 1);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            cb_qty[v] = load_combit_bm(COMBIT_DIR + "/quantity/" + std::to_string(v) + ".bm");
        cb_ship.resize(SHIPDATE_DAY_END);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            cb_ship[d] = load_combit_bm(COMBIT_DIR + "/shipdate/" + std::to_string(d) + ".bm");

        cb_disc_ptrs.reserve(DISCOUNT_MAX - DISCOUNT_MIN + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            cb_disc_ptrs.push_back(&cb_disc[v]);
        cb_qty_ptrs.reserve(QUANTITY_MAX);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            cb_qty_ptrs.push_back(&cb_qty[v]);
        cb_ship_ptrs.reserve(SHIPDATE_DAY_END - SHIPDATE_DAY_START);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            cb_ship_ptrs.push_back(&cb_ship[d]);
        cb_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // CRoaring bitmaps (no runOptimize)
    std::vector<roaring::Roaring> cr_disc, cr_qty, cr_ship;
    double cr_load_ms = 0;
    if (run_cr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cr_disc.resize(DISCOUNT_MAX + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            cr_disc[v] = load_croaring_bm(CROARING_DIR + "/discount/" + std::to_string(v) + ".bm");
        cr_qty.resize(QUANTITY_MAX + 1);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            cr_qty[v] = load_croaring_bm(CROARING_DIR + "/quantity/" + std::to_string(v) + ".bm");
        cr_ship.resize(SHIPDATE_DAY_END);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            cr_ship[d] = load_croaring_bm(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        cr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // CRoaring+Run: load fresh from the same files and apply
    // runOptimize.  Independent of cr_* so Q6_BM=crr works even
    // when CRoaring-without-Run is disabled.  Pointer vectors are
    // pre-built here (outside the iteration loop) for fastunion.
    std::vector<roaring::Roaring> crr_disc, crr_qty, crr_ship;
    std::vector<const roaring::Roaring*> crr_disc_ptrs, crr_qty_ptrs, crr_ship_ptrs;
    double crr_load_ms = 0;
    if (run_crr()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        crr_disc.resize(DISCOUNT_MAX + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++) {
            crr_disc[v] = load_croaring_bm(CROARING_DIR + "/discount/" + std::to_string(v) + ".bm");
            crr_disc[v].runOptimize();
        }
        crr_qty.resize(QUANTITY_MAX + 1);
        for (int v = 1; v <= QUANTITY_MAX; v++) {
            crr_qty[v] = load_croaring_bm(CROARING_DIR + "/quantity/" + std::to_string(v) + ".bm");
            crr_qty[v].runOptimize();
        }
        crr_ship.resize(SHIPDATE_DAY_END);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++) {
            crr_ship[d] = load_croaring_bm(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");
            crr_ship[d].runOptimize();
        }

        crr_disc_ptrs.reserve(DISCOUNT_MAX - DISCOUNT_MIN + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            crr_disc_ptrs.push_back(&crr_disc[v]);
        crr_qty_ptrs.reserve(QUANTITY_MAX);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            crr_qty_ptrs.push_back(&crr_qty[v]);
        crr_ship_ptrs.reserve(SHIPDATE_DAY_END - SHIPDATE_DAY_START);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            crr_ship_ptrs.push_back(&crr_ship[d]);
        crr_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // EWAH bitmaps
    std::vector<ewah::EWAHBoolArray<uint64_t>> ew_disc, ew_qty, ew_ship;
    // Pre-build EWAH pointer arrays so all three OR phases can use
    // ewah::fast_logicalor (priority-queue k-way merge) — matches
    // CRR fastunion / Concise fast_logicalor.
    std::vector<const ewah::EWAHBoolArray<uint64_t>*> ew_disc_ptrs, ew_qty_ptrs, ew_ship_ptrs;
    double ew_load_ms = 0;
    if (run_ew()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ew_disc.resize(DISCOUNT_MAX + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            ew_disc[v] = load_ewah_bm(EWAH_DIR + "/discount/" + std::to_string(v) + ".bm");
        ew_qty.resize(QUANTITY_MAX + 1);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            ew_qty[v] = load_ewah_bm(EWAH_DIR + "/quantity/" + std::to_string(v) + ".bm");
        ew_ship.resize(SHIPDATE_DAY_END);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            ew_ship[d] = load_ewah_bm(EWAH_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        ew_disc_ptrs.reserve(DISCOUNT_MAX - DISCOUNT_MIN + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++) ew_disc_ptrs.push_back(&ew_disc[v]);
        ew_qty_ptrs.reserve(QUANTITY_MAX);
        for (int v = 1; v <= QUANTITY_MAX; v++) ew_qty_ptrs.push_back(&ew_qty[v]);
        ew_ship_ptrs.reserve(SHIPDATE_DAY_END - SHIPDATE_DAY_START);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++) ew_ship_ptrs.push_back(&ew_ship[d]);
        ew_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // Bitset baseline (uncompressed): same Bitmap type backs both BS
    // (scalar) and BSA (AVX-512), so we load once and let the runtime
    // dispatch on use_simd at op time.  The vectors are tagged "bs_*"
    // and shared between the BS and BSA pipelines.
    std::vector<bs::Bitmap> bs_disc, bs_qty, bs_ship;
    double bs_load_ms = 0;
    if (run_bs() || run_bsa()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        bs_disc.resize(DISCOUNT_MAX + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            bs_disc[v] = bm_bench::load_bitmap_from_croaring(CROARING_DIR + "/discount/" + std::to_string(v) + ".bm");
        bs_qty.resize(QUANTITY_MAX + 1);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            bs_qty[v] = bm_bench::load_bitmap_from_croaring(CROARING_DIR + "/quantity/" + std::to_string(v) + ".bm");
        bs_ship.resize(SHIPDATE_DAY_END);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            bs_ship[d] = bm_bench::load_bitmap_from_croaring(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");
        bs_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    // Concise (Colantonio & Di Pietro).  Pointer vectors mirror the
    // CRR setup so we can call ConciseSet::fast_logicalor (k-way merge)
    // — the apples-to-apples counterpart of CRoaring's fastunion.
    std::vector<ConciseSet<false>> con_disc, con_qty, con_ship;
    std::vector<const ConciseSet<false>*> con_disc_ptrs, con_qty_ptrs, con_ship_ptrs;
    double con_load_ms = 0;
    if (run_con()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        con_disc.resize(DISCOUNT_MAX + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            con_disc[v] = bm_bench::load_concise_from_croaring(CROARING_DIR + "/discount/" + std::to_string(v) + ".bm");
        con_qty.resize(QUANTITY_MAX + 1);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            con_qty[v] = bm_bench::load_concise_from_croaring(CROARING_DIR + "/quantity/" + std::to_string(v) + ".bm");
        con_ship.resize(SHIPDATE_DAY_END);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            con_ship[d] = bm_bench::load_concise_from_croaring(CROARING_DIR + "/shipdate/" + std::to_string(d) + ".bm");

        con_disc_ptrs.reserve(DISCOUNT_MAX - DISCOUNT_MIN + 1);
        for (int v = DISCOUNT_MIN; v <= DISCOUNT_MAX; v++)
            con_disc_ptrs.push_back(&con_disc[v]);
        con_qty_ptrs.reserve(QUANTITY_MAX);
        for (int v = 1; v <= QUANTITY_MAX; v++)
            con_qty_ptrs.push_back(&con_qty[v]);
        con_ship_ptrs.reserve(SHIPDATE_DAY_END - SHIPDATE_DAY_START);
        for (int d = SHIPDATE_DAY_START; d < SHIPDATE_DAY_END; d++)
            con_ship_ptrs.push_back(&con_ship[d]);
        con_load_ms = ms(t0, std::chrono::high_resolution_clock::now());
    }

    if (run_wah()) std::cout << "  WAH load:         " << wah_load_ms << " ms (" << files_per_backend << " files)" << std::endl;
    if (run_cb())  std::cout << "  ComBit load:      " << cb_load_ms  << " ms (" << files_per_backend << " files)" << std::endl;
    if (run_cr())  std::cout << "  CRoaring load:    " << cr_load_ms  << " ms (" << files_per_backend << " files)" << std::endl;
    if (run_crr()) std::cout << "  CRoar+Run load:   " << crr_load_ms << " ms (" << files_per_backend << " files)" << std::endl;
    if (run_ew())  std::cout << "  EWAH load:        " << ew_load_ms  << " ms (" << files_per_backend << " files)" << std::endl;
    if (run_bs() || run_bsa())
                    std::cout << "  Bitset load:      " << bs_load_ms  << " ms (" << files_per_backend << " files, shared by BS / BSA)" << std::endl;
    if (run_con()) std::cout << "  Concise load:     " << con_load_ms << " ms (" << files_per_backend << " files)" << std::endl;

    // Per-backend on-disk footprint (useful in every mode, not just ALL).
    std::cout << std::fixed << std::setprecision(2);
    if (run_wah()) std::cout << "  WAH      on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(WAH_DIR))      << " MiB" << std::endl;
    if (run_cb())  std::cout << "  ComBit   on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(COMBIT_DIR))   << " MiB" << std::endl;
    if (run_cr() || run_crr())
                    std::cout << "  CRoaring on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(CROARING_DIR)) << " MiB (shared by CR / CRR)" << std::endl;
    if (run_ew())  std::cout << "  EWAH     on disk: " << bm_bench::mib(bm_bench::dir_size_bytes(EWAH_DIR))     << " MiB" << std::endl;
    // Bitset/Concise: in-memory footprint (no separate on-disk format —
    // they're built on the fly from the CRoaring file at load).
    if (run_bs() || run_bsa()) {
        size_t bs_bytes = 0;
        for (auto& b : bs_disc) bs_bytes += b.nwords * sizeof(uint64_t);
        for (auto& b : bs_qty)  bs_bytes += b.nwords * sizeof(uint64_t);
        for (auto& b : bs_ship) bs_bytes += b.nwords * sizeof(uint64_t);
        std::cout << "  Bitset in mem:    " << bm_bench::mib(bs_bytes) << " MiB (rebuilt from CRoaring at load)" << std::endl;
    }
    if (run_con()) {
        size_t con_bytes = 0;
        for (auto& c : con_disc) con_bytes += c.sizeInBytes();
        for (auto& c : con_qty)  con_bytes += c.sizeInBytes();
        for (auto& c : con_ship) con_bytes += c.sizeInBytes();
        std::cout << "  Concise in mem:   " << bm_bench::mib(con_bytes) << " MiB (rebuilt from CRoaring at load)" << std::endl;
    }

    // ============================================================
    // Compression: scan all bitmap files once (shared by stdout
    // report below and CSV export later).  Only runs in ALL mode —
    // the compression report's 4-column layout is meaningful only
    // when all backends are present, and the scan needs cb_disc[]
    // loaded to read bit_count() for the raw-size denominator.
    // Runs outside the timed benchmark loop, so the extra stat()
    // calls don't affect Q6 timing measurements.
    // ============================================================
    struct Q6ColumnSizes {
        long count = 0;
        long raw = 0;
        long cb  = 0, wah = 0, cr = 0, ew = 0;
    };
    Q6ColumnSizes disc_sz, qty_sz, ship_sz, total_sz;

    if (run_all()) {
        auto file_size_of = [](const std::string& path) -> long {
            std::error_code ec;
            auto sz = std::filesystem::file_size(path, ec);
            return ec ? 0L : static_cast<long>(sz);
        };
        auto scan_column = [&](const std::string& subdir, int start, int end_excl,
                               size_t seg_bits) {
            Q6ColumnSizes s;
            s.count = end_excl - start;
            for (int v = start; v < end_excl; v++) {
                std::string suffix = "/" + subdir + "/" + std::to_string(v) + ".bm";
                s.cb  += file_size_of(COMBIT_DIR   + suffix);
                s.wah += file_size_of(WAH_DIR      + suffix);
                s.cr  += file_size_of(CROARING_DIR + suffix);
                s.ew  += file_size_of(EWAH_DIR     + suffix);
            }
            s.raw = s.count * (long)((seg_bits + 7) / 8);
            return s;
        };

        disc_sz = scan_column("discount", DISCOUNT_MIN, DISCOUNT_MAX + 1,
                              cb_disc[DISCOUNT_MIN].bit_count());
        qty_sz  = scan_column("quantity", 1, QUANTITY_MAX + 1,
                              cb_qty[1].bit_count());
        ship_sz = scan_column("shipdate", SHIPDATE_DAY_START, SHIPDATE_DAY_END,
                              cb_ship[SHIPDATE_DAY_START].bit_count());
        total_sz.count = disc_sz.count + qty_sz.count + ship_sz.count;
        total_sz.raw   = disc_sz.raw   + qty_sz.raw   + ship_sz.raw;
        total_sz.cb    = disc_sz.cb    + qty_sz.cb    + ship_sz.cb;
        total_sz.wah   = disc_sz.wah   + qty_sz.wah   + ship_sz.wah;
        total_sz.cr    = disc_sz.cr    + qty_sz.cr    + ship_sz.cr;
        total_sz.ew    = disc_sz.ew    + qty_sz.ew    + ship_sz.ew;

        // ============================================================
        // Compression Ratio Report (stdout)
        // ============================================================
        auto mb = [](long b) { return b / (1024.0 * 1024.0); };
        auto ratio = [](long compressed, long raw) -> double {
            return raw > 0 ? (double)compressed / raw : 0.0;
        };

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  COMPRESSION RATIO (Q6-relevant bitmaps only)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                 Count    Raw(MB)   ComBit(MB)  WAH(MB)  CRoar(MB)  EWAH(MB)  CB ratio  WAH ratio  CR ratio  EW ratio" << std::endl;
        std::cout << "  ---------------------------------------------------------------------------------------------------------------" << std::endl;

        auto print_cr = [&](const char* label, const Q6ColumnSizes& s) {
            std::cout << "  " << std::left << std::setw(14) << label
                      << std::right << std::setw(5) << s.count
                      << std::setw(11) << mb(s.raw)
                      << std::setw(12) << mb(s.cb)
                      << std::setw(9)  << mb(s.wah)
                      << std::setw(10) << mb(s.cr)
                      << std::setw(10) << mb(s.ew)
                      << std::setw(10) << ratio(s.cb,  s.raw)
                      << std::setw(10) << ratio(s.wah, s.raw)
                      << std::setw(10) << ratio(s.cr,  s.raw)
                      << std::setw(10) << ratio(s.ew,  s.raw) << std::endl;
        };

        print_cr("Discount", disc_sz);
        print_cr("Quantity", qty_sz);
        print_cr("Shipdate", ship_sz);
        std::cout << "  ---------------------------------------------------------------------------------------------------------------" << std::endl;
        print_cr("TOTAL", total_sz);
        std::cout << std::endl;
    }

    // ============================================================
    // Benchmark: multiple iterations
    // ============================================================
    // Per-iteration timing vectors (excluding warm-up)
    std::vector<double> wah_or_disc_times, wah_or_qty_times, wah_or_ship_times;
    std::vector<double> wah_and_times, wah_decode_times, wah_total_times;
    std::vector<double> cb_or_disc_times, cb_or_qty_times, cb_or_ship_times;
    std::vector<double> cb_and_times, cb_decode_times, cb_total_times;
    std::vector<double> cr_or_disc_times, cr_or_qty_times, cr_or_ship_times;
    std::vector<double> cr_and_times, cr_decode_times, cr_total_times;
    std::vector<double> crr_or_disc_times, crr_or_qty_times, crr_or_ship_times;
    std::vector<double> crr_and_times, crr_decode_times, crr_total_times;
    std::vector<double> ew_or_disc_times, ew_or_qty_times, ew_or_ship_times;
    std::vector<double> ew_and_times, ew_decode_times, ew_total_times;
    std::vector<double> bs_or_disc_times, bs_or_qty_times, bs_or_ship_times;
    std::vector<double> bs_and_times, bs_decode_times, bs_total_times;
    std::vector<double> bsa_or_disc_times, bsa_or_qty_times, bsa_or_ship_times;
    std::vector<double> bsa_and_times, bsa_decode_times, bsa_total_times;
    std::vector<double> con_or_disc_times, con_or_qty_times, con_or_ship_times;
    std::vector<double> con_and_times, con_decode_times, con_total_times;

    size_t wah_row_count = 0, cb_row_count = 0, cr_row_count = 0, crr_row_count = 0, ew_row_count = 0;
    size_t bs_row_count  = 0, bsa_row_count = 0, con_row_count = 0;

    // Reusable decode buffers, hoisted so every backend pays the same
    // one-off allocation cost during warm-up (removes an old
    // ComBit-only measurement asymmetry).
    std::vector<int64_t> wah_rowids, cb_decode_buf, cr_rowids, crr_rowids, ew_rowids;
    std::vector<int64_t> bs_rowids, bsa_rowids, con_rowids;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        bool is_warmup = (iter < WARMUP_RUNS);
        std::cout << "\n--- Iteration " << (iter + 1) << "/" << NUM_ITERATIONS
                  << (is_warmup ? " (warm-up)" : "") << " ---" << std::endl;

        // === WAH ===
        if (run_wah()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // OR discount
            ibis::bitvector wah_disc_or = wah_disc[DISCOUNT_MIN];
            for (int v = DISCOUNT_MIN + 1; v <= DISCOUNT_MAX; v++)
                wah_disc_or |= wah_disc[v];
            auto t1 = std::chrono::high_resolution_clock::now();

            // OR quantity
            ibis::bitvector wah_qty_or = wah_qty[1];
            for (int v = 2; v <= QUANTITY_MAX; v++)
                wah_qty_or |= wah_qty[v];
            auto t2 = std::chrono::high_resolution_clock::now();

            // OR shipdate (365 bitmaps!)
            ibis::bitvector wah_ship_or = wah_ship[SHIPDATE_DAY_START];
            for (int d = SHIPDATE_DAY_START + 1; d < SHIPDATE_DAY_END; d++)
                wah_ship_or |= wah_ship[d];
            auto t3 = std::chrono::high_resolution_clock::now();

            // AND
            ibis::bitvector wah_result = wah_disc_or;
            wah_result &= wah_qty_or;
            wah_result &= wah_ship_or;
            auto t4 = std::chrono::high_resolution_clock::now();

            // Decode (reuses hoisted buffer)
            GetRowidsWAH(wah_result, &wah_rowids);
            wah_row_count = wah_rowids.size();
            if (wah_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(wah_rowids);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  WAH:  OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << wah_row_count << std::endl;

            if (!is_warmup) {
                wah_or_disc_times.push_back(ms(t0, t1));
                wah_or_qty_times.push_back(ms(t1, t2));
                wah_or_ship_times.push_back(ms(t2, t3));
                wah_and_times.push_back(ms(t3, t4));
                wah_decode_times.push_back(ms(t4, t5));
                wah_total_times.push_back(ms(t0, t5));
            }
        }

        // === ComBit ===
        // Mirrors the WAH block above: three OR_many, chained &=, decode.
        // All bitwise work goes through the library's public operators;
        // no flat-byte detour, no hand-rolled SIMD.  operator&= accepts
        // Decompressed RHS (degenerates to per-region flat 64-byte AND),
        // so the OR_many results can be AND-chained directly.
        if (run_cb()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ComBit cb_disc_or = ComBit::OR_many(cb_disc_ptrs.size(),
                                                cb_disc_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ComBit cb_qty_or = ComBit::OR_many(cb_qty_ptrs.size(),
                                               cb_qty_ptrs.data());
            auto t2 = std::chrono::high_resolution_clock::now();

            ComBit cb_ship_or = ComBit::OR_many(cb_ship_ptrs.size(),
                                                cb_ship_ptrs.data());
            auto t3 = std::chrono::high_resolution_clock::now();

            cb_disc_or &= cb_qty_or;
            cb_disc_or &= cb_ship_or;
            auto t4 = std::chrono::high_resolution_clock::now();

            GetRowidsComBit(cb_disc_or, &cb_decode_buf);
            cb_row_count = cb_decode_buf.size();
            if (cb_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(cb_decode_buf);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  CB:   OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << cb_row_count << std::endl;

            if (!is_warmup) {
                cb_or_disc_times.push_back(ms(t0, t1));
                cb_or_qty_times.push_back(ms(t1, t2));
                cb_or_ship_times.push_back(ms(t2, t3));
                cb_and_times.push_back(ms(t3, t4));
                cb_decode_times.push_back(ms(t4, t5));
                cb_total_times.push_back(ms(t0, t5));
            }
        }

        // === CRoaring ===
        if (run_cr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // OR discount
            roaring::Roaring cr_disc_or = cr_disc[DISCOUNT_MIN];
            for (int v = DISCOUNT_MIN + 1; v <= DISCOUNT_MAX; v++)
                cr_disc_or |= cr_disc[v];
            auto t1 = std::chrono::high_resolution_clock::now();

            // OR quantity
            roaring::Roaring cr_qty_or = cr_qty[1];
            for (int v = 2; v <= QUANTITY_MAX; v++)
                cr_qty_or |= cr_qty[v];
            auto t2 = std::chrono::high_resolution_clock::now();

            // OR shipdate (365 bitmaps!)
            roaring::Roaring cr_ship_or = cr_ship[SHIPDATE_DAY_START];
            for (int d = SHIPDATE_DAY_START + 1; d < SHIPDATE_DAY_END; d++)
                cr_ship_or |= cr_ship[d];
            auto t3 = std::chrono::high_resolution_clock::now();

            // AND
            roaring::Roaring cr_result = cr_disc_or;
            cr_result &= cr_qty_or;
            cr_result &= cr_ship_or;
            auto t4 = std::chrono::high_resolution_clock::now();

            // Decode (reuses hoisted buffer)
            GetRowidsCRoaring(cr_result, &cr_rowids);
            cr_row_count = cr_rowids.size();
            if (cr_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(cr_rowids);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  CR:   OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << cr_row_count << std::endl;

            if (!is_warmup) {
                cr_or_disc_times.push_back(ms(t0, t1));
                cr_or_qty_times.push_back(ms(t1, t2));
                cr_or_ship_times.push_back(ms(t2, t3));
                cr_and_times.push_back(ms(t3, t4));
                cr_decode_times.push_back(ms(t4, t5));
                cr_total_times.push_back(ms(t0, t5));
            }
        }

        // === CRoaring+Run (with fastunion / or_many) ===
        if (run_crr()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // OR discount via fastunion (lazy OR + repair)
            roaring::Roaring crr_disc_or = roaring::Roaring::fastunion(crr_disc_ptrs.size(), crr_disc_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            // OR quantity via fastunion
            roaring::Roaring crr_qty_or = roaring::Roaring::fastunion(crr_qty_ptrs.size(), crr_qty_ptrs.data());
            auto t2 = std::chrono::high_resolution_clock::now();

            // OR shipdate via fastunion
            roaring::Roaring crr_ship_or = roaring::Roaring::fastunion(crr_ship_ptrs.size(), crr_ship_ptrs.data());
            auto t3 = std::chrono::high_resolution_clock::now();

            roaring::Roaring crr_result = crr_disc_or;
            crr_result &= crr_qty_or;
            crr_result &= crr_ship_or;
            auto t4 = std::chrono::high_resolution_clock::now();

            // Decode (reuses hoisted buffer)
            GetRowidsCRoaring(crr_result, &crr_rowids);
            crr_row_count = crr_rowids.size();
            if (crr_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(crr_rowids);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  CRR:  OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << crr_row_count << std::endl;

            if (!is_warmup) {
                crr_or_disc_times.push_back(ms(t0, t1));
                crr_or_qty_times.push_back(ms(t1, t2));
                crr_or_ship_times.push_back(ms(t2, t3));
                crr_and_times.push_back(ms(t3, t4));
                crr_decode_times.push_back(ms(t4, t5));
                crr_total_times.push_back(ms(t0, t5));
            }
        }

        // === EWAH ===
        if (run_ew()) {
            // fast_logicalor: priority-queue k-way merge (ewah-inl.h:1129).
            auto t0 = std::chrono::high_resolution_clock::now();

            // OR discount
            ewah::EWAHBoolArray<uint64_t> ew_disc_or = ewah::fast_logicalor(
                ew_disc_ptrs.size(), ew_disc_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            // OR quantity
            ewah::EWAHBoolArray<uint64_t> ew_qty_or = ewah::fast_logicalor(
                ew_qty_ptrs.size(), ew_qty_ptrs.data());
            auto t2 = std::chrono::high_resolution_clock::now();

            // OR shipdate (365 bitmaps!)
            ewah::EWAHBoolArray<uint64_t> ew_ship_or = ewah::fast_logicalor(
                ew_ship_ptrs.size(), ew_ship_ptrs.data());
            auto t3 = std::chrono::high_resolution_clock::now();

            // AND
            ewah::EWAHBoolArray<uint64_t> ew_tmp;
            ew_disc_or.logicaland(ew_qty_or, ew_tmp);
            ewah::EWAHBoolArray<uint64_t> ew_result;
            ew_tmp.logicaland(ew_ship_or, ew_result);
            auto t4 = std::chrono::high_resolution_clock::now();

            // Decode (reuses hoisted buffer)
            GetRowidsEWAH(ew_result, &ew_rowids);
            ew_row_count = ew_rowids.size();
            if (ew_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(ew_rowids);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  EW:   OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << ew_row_count << std::endl;

            if (!is_warmup) {
                ew_or_disc_times.push_back(ms(t0, t1));
                ew_or_qty_times.push_back(ms(t1, t2));
                ew_or_ship_times.push_back(ms(t2, t3));
                ew_and_times.push_back(ms(t3, t4));
                ew_decode_times.push_back(ms(t4, t5));
                ew_total_times.push_back(ms(t0, t5));
            }
        }

        // === Bitset (scalar) ===
        // Pure baseline: clone first bitmap of each column then OR the
        // rest into it via a non-vectorised scalar word-loop.  AND chains
        // also use the scalar kernel.  Decode walks set bits via ctz.
        if (run_bs()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bs_disc_or = bs_disc[DISCOUNT_MIN].clone();
            for (int v = DISCOUNT_MIN + 1; v <= DISCOUNT_MAX; v++)
                bs::or_inplace(bs_disc_or, bs_disc[v], false);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bs_qty_or = bs_qty[1].clone();
            for (int v = 2; v <= QUANTITY_MAX; v++)
                bs::or_inplace(bs_qty_or, bs_qty[v], false);
            auto t2 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bs_ship_or = bs_ship[SHIPDATE_DAY_START].clone();
            for (int d = SHIPDATE_DAY_START + 1; d < SHIPDATE_DAY_END; d++)
                bs::or_inplace(bs_ship_or, bs_ship[d], false);
            auto t3 = std::chrono::high_resolution_clock::now();

            bs::and_inplace(bs_disc_or, bs_qty_or,  false);
            bs::and_inplace(bs_disc_or, bs_ship_or, false);
            auto t4 = std::chrono::high_resolution_clock::now();

            GetRowidsBS(bs_disc_or, &bs_rowids);
            bs_row_count = bs_rowids.size();
            if (bs_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(bs_rowids);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  BS:   OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << bs_row_count << std::endl;

            if (!is_warmup) {
                bs_or_disc_times.push_back(ms(t0, t1));
                bs_or_qty_times.push_back(ms(t1, t2));
                bs_or_ship_times.push_back(ms(t2, t3));
                bs_and_times.push_back(ms(t3, t4));
                bs_decode_times.push_back(ms(t4, t5));
                bs_total_times.push_back(ms(t0, t5));
            }
        }

        // === Bitset + AVX-512 ===
        // Same Bitmap data + same kernels modulo simd flag = true.
        // Demonstrates the marginal gain of pure SIMD over scalar on the
        // uncompressed baseline (no compression algorithm to amortise).
        if (run_bsa()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bsa_disc_or = bs_disc[DISCOUNT_MIN].clone();
            for (int v = DISCOUNT_MIN + 1; v <= DISCOUNT_MAX; v++)
                bs::or_inplace(bsa_disc_or, bs_disc[v], true);
            auto t1 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bsa_qty_or = bs_qty[1].clone();
            for (int v = 2; v <= QUANTITY_MAX; v++)
                bs::or_inplace(bsa_qty_or, bs_qty[v], true);
            auto t2 = std::chrono::high_resolution_clock::now();

            bs::Bitmap bsa_ship_or = bs_ship[SHIPDATE_DAY_START].clone();
            for (int d = SHIPDATE_DAY_START + 1; d < SHIPDATE_DAY_END; d++)
                bs::or_inplace(bsa_ship_or, bs_ship[d], true);
            auto t3 = std::chrono::high_resolution_clock::now();

            bs::and_inplace(bsa_disc_or, bsa_qty_or,  true);
            bs::and_inplace(bsa_disc_or, bsa_ship_or, true);
            auto t4 = std::chrono::high_resolution_clock::now();

            GetRowidsBS(bsa_disc_or, &bsa_rowids);
            bsa_row_count = bsa_rowids.size();
            if (bsa_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(bsa_rowids);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  BSA:  OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << bsa_row_count << std::endl;

            if (!is_warmup) {
                bsa_or_disc_times.push_back(ms(t0, t1));
                bsa_or_qty_times.push_back(ms(t1, t2));
                bsa_or_ship_times.push_back(ms(t2, t3));
                bsa_and_times.push_back(ms(t3, t4));
                bsa_decode_times.push_back(ms(t4, t5));
                bsa_total_times.push_back(ms(t0, t5));
            }
        }

        // === Concise ===
        // Uses fast_logicalor (priority-queue k-way merge) for OR, the
        // direct counterpart of CRR's fastunion.  AND chains logicaland
        // (returns a fresh ConciseSet each call) — the *ToContainer
        // variants in the upstream library do NOT reset `res` before
        // writing into it, so reusing a scratch container layers stale
        // words on top of the new output.  Fresh-alloc is also what
        // CRR does via logicaland / operator&.
        if (run_con()) {
            auto t0 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_disc_or = ConciseSet<false>::fast_logicalor(
                con_disc_ptrs.size(), con_disc_ptrs.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_qty_or = ConciseSet<false>::fast_logicalor(
                con_qty_ptrs.size(), con_qty_ptrs.data());
            auto t2 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_ship_or = ConciseSet<false>::fast_logicalor(
                con_ship_ptrs.size(), con_ship_ptrs.data());
            auto t3 = std::chrono::high_resolution_clock::now();

            ConciseSet<false> con_result = con_disc_or.logicaland(con_qty_or);
            con_result = con_result.logicaland(con_ship_or);
            auto t4 = std::chrono::high_resolution_clock::now();

            GetRowidsConcise(con_result, &con_rowids);
            con_row_count = con_rowids.size();
            if (con_primary() && iter == NUM_ITERATIONS - 1)
                *row_ids = std::move(con_rowids);
            auto t5 = std::chrono::high_resolution_clock::now();

            std::cout << "  CON:  OR_disc=" << ms(t0,t1) << "  OR_qty=" << ms(t1,t2)
                      << "  OR_ship=" << ms(t2,t3) << "  AND=" << ms(t3,t4)
                      << "  Decode=" << ms(t4,t5) << "  Total=" << ms(t0,t5)
                      << "  rows=" << con_row_count << std::endl;

            if (!is_warmup) {
                con_or_disc_times.push_back(ms(t0, t1));
                con_or_qty_times.push_back(ms(t1, t2));
                con_or_ship_times.push_back(ms(t2, t3));
                con_and_times.push_back(ms(t3, t4));
                con_decode_times.push_back(ms(t4, t5));
                con_total_times.push_back(ms(t0, t5));
            }
        }
    }

    // ============================================================
    // Statistics Summary
    // ============================================================
    auto wah_or_disc_s = compute_stats(wah_or_disc_times);
    auto wah_or_qty_s  = compute_stats(wah_or_qty_times);
    auto wah_or_ship_s = compute_stats(wah_or_ship_times);
    auto wah_and_s     = compute_stats(wah_and_times);
    auto wah_decode_s  = compute_stats(wah_decode_times);
    auto wah_total_s   = compute_stats(wah_total_times);

    auto cb_or_disc_s = compute_stats(cb_or_disc_times);
    auto cb_or_qty_s  = compute_stats(cb_or_qty_times);
    auto cb_or_ship_s = compute_stats(cb_or_ship_times);
    auto cb_and_s     = compute_stats(cb_and_times);
    auto cb_decode_s  = compute_stats(cb_decode_times);
    auto cb_total_s   = compute_stats(cb_total_times);

    auto cr_or_disc_s = compute_stats(cr_or_disc_times);
    auto cr_or_qty_s  = compute_stats(cr_or_qty_times);
    auto cr_or_ship_s = compute_stats(cr_or_ship_times);
    auto cr_and_s     = compute_stats(cr_and_times);
    auto cr_decode_s  = compute_stats(cr_decode_times);
    auto cr_total_s   = compute_stats(cr_total_times);

    auto crr_or_disc_s = compute_stats(crr_or_disc_times);
    auto crr_or_qty_s  = compute_stats(crr_or_qty_times);
    auto crr_or_ship_s = compute_stats(crr_or_ship_times);
    auto crr_and_s     = compute_stats(crr_and_times);
    auto crr_decode_s  = compute_stats(crr_decode_times);
    auto crr_total_s   = compute_stats(crr_total_times);

    auto ew_or_disc_s = compute_stats(ew_or_disc_times);
    auto ew_or_qty_s  = compute_stats(ew_or_qty_times);
    auto ew_or_ship_s = compute_stats(ew_or_ship_times);
    auto ew_and_s     = compute_stats(ew_and_times);
    auto ew_decode_s  = compute_stats(ew_decode_times);
    auto ew_total_s   = compute_stats(ew_total_times);

    auto bs_or_disc_s  = compute_stats(bs_or_disc_times);
    auto bs_or_qty_s   = compute_stats(bs_or_qty_times);
    auto bs_or_ship_s  = compute_stats(bs_or_ship_times);
    auto bs_and_s      = compute_stats(bs_and_times);
    auto bs_decode_s   = compute_stats(bs_decode_times);
    auto bs_total_s    = compute_stats(bs_total_times);

    auto bsa_or_disc_s = compute_stats(bsa_or_disc_times);
    auto bsa_or_qty_s  = compute_stats(bsa_or_qty_times);
    auto bsa_or_ship_s = compute_stats(bsa_or_ship_times);
    auto bsa_and_s     = compute_stats(bsa_and_times);
    auto bsa_decode_s  = compute_stats(bsa_decode_times);
    auto bsa_total_s   = compute_stats(bsa_total_times);

    auto con_or_disc_s = compute_stats(con_or_disc_times);
    auto con_or_qty_s  = compute_stats(con_or_qty_times);
    auto con_or_ship_s = compute_stats(con_or_ship_times);
    auto con_and_s     = compute_stats(con_and_times);
    auto con_decode_s  = compute_stats(con_decode_times);
    auto con_total_s   = compute_stats(con_total_times);

    int measured = NUM_ITERATIONS - WARMUP_RUNS;

    // --- DuckDB native SQL baseline + ground-truth row count ---
    // Runs AFTER the bitmap iteration loop so it does not pollute the
    // CPU cache and skew bitmap measurements.  A fresh Connection on
    // the same DatabaseInstance uses a separate ClientContext, so no
    // re-entrancy against the outer pragma.
    int64_t gt_rows = -1;
    double  gt_sql_ms = 0.0;
    try {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT count(*) FROM lineitem "
            "WHERE l_shipdate >= DATE '1994-01-01' "
            "  AND l_shipdate <  DATE '1995-01-01' "
            "  AND l_discount BETWEEN 0.05 AND 0.07 "
            "  AND l_quantity <  24";
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

    // --- Ground-truth assert: every active backend must match the SQL count ---
    if (gt_rows >= 0) {
        auto check = [&](const char* label, bool active, size_t got) {
            if (!active) return;
            if (static_cast<int64_t>(got) == gt_rows) return;
            std::ostringstream oss;
            oss << "[FAIL] Q6 " << label << " row count " << got
                << " != DuckDB SQL ground truth " << gt_rows
                << " — bitmap pipeline is incorrect";
            throw std::runtime_error(oss.str());
        };
        check("WAH",           run_wah(), wah_row_count);
        check("ComBit",        run_cb(),  cb_row_count);
        check("CRoaring",      run_cr(),  cr_row_count);
        check("CRoaring+Run",  run_crr(), crr_row_count);
        check("EWAH",          run_ew(),  ew_row_count);
        check("Bitset",        run_bs(),  bs_row_count);
        check("Bitset+AVX512", run_bsa(), bsa_row_count);
        check("Concise",       run_con(), con_row_count);
        std::cout << "[OK] all active backends match DuckDB SQL ground truth ("
                  << gt_rows << " rows)." << std::endl;
    }


    if (run_all()) {
        // ============================================================
        // ALL mode: full 5-column comparison table
        // ============================================================
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  RESULTS (" << measured << " measured iterations, median ± stddev)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                       WAH (ms)              ComBit (ms)         CRoaring (ms)      CRoar+Run (ms)       EWAH (ms)          CB vs WAH   CR vs WAH  CRR vs WAH   EW vs WAH" << std::endl;
        std::cout << "  --------------------------------------------------------------------------------------------------------------------------------" << std::endl;

        auto print_row = [](const char* label, TimingStats& w, TimingStats& c, TimingStats& r, TimingStats& rr, TimingStats& e) {
            double cb_speedup  = (c.median > 0)  ? w.median / c.median  : 0;
            double cr_speedup  = (r.median > 0)  ? w.median / r.median  : 0;
            double crr_speedup = (rr.median > 0) ? w.median / rr.median : 0;
            double ew_speedup  = (e.median > 0)  ? w.median / e.median  : 0;
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(8) << w.median << " ± " << std::setw(6) << w.stddev
                      << "     " << std::setw(8) << c.median << " ± " << std::setw(6) << c.stddev
                      << "     " << std::setw(8) << r.median << " ± " << std::setw(6) << r.stddev
                      << "     " << std::setw(8) << rr.median << " ± " << std::setw(6) << rr.stddev
                      << "     " << std::setw(8) << e.median << " ± " << std::setw(6) << e.stddev
                      << "     " << std::setw(5) << cb_speedup << "x"
                      << "     " << std::setw(5) << cr_speedup << "x"
                      << "     " << std::setw(5) << crr_speedup << "x"
                      << "     " << std::setw(5) << ew_speedup << "x" << std::endl;
        };

        print_row("OR discount", wah_or_disc_s, cb_or_disc_s, cr_or_disc_s, crr_or_disc_s, ew_or_disc_s);
        print_row("OR quantity", wah_or_qty_s, cb_or_qty_s, cr_or_qty_s, crr_or_qty_s, ew_or_qty_s);
        print_row("OR shipdate", wah_or_ship_s, cb_or_ship_s, cr_or_ship_s, crr_or_ship_s, ew_or_ship_s);
        print_row("AND (3-way)", wah_and_s, cb_and_s, cr_and_s, crr_and_s, ew_and_s);
        print_row("Decode", wah_decode_s, cb_decode_s, cr_decode_s, crr_decode_s, ew_decode_s);
        std::cout << "  --------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL", wah_total_s, cb_total_s, cr_total_s, crr_total_s, ew_total_s);

        std::cout << "\n  WAH rows:        " << wah_row_count << std::endl;
        std::cout << "  ComBit rows:     " << cb_row_count << std::endl;
        std::cout << "  CRoaring rows:   " << cr_row_count << std::endl;
        std::cout << "  CRoar+Run rows:  " << crr_row_count << std::endl;
        std::cout << "  EWAH rows:       " << ew_row_count << std::endl;

        std::cout << "================================================================\n" << std::endl;

        // ============================================================
        // ALL mode: baseline backends table (BS / BSA / Concise).
        // Printed separately because adding 3 more backends to the
        // 5-way table above would push it well past 250 columns.  The
        // speedup column anchors against WAH for consistency with the
        // 5-way table.
        // ============================================================
        std::cout << "\n================================================================" << std::endl;
        std::cout << "  Q6 BASELINE BACKENDS (no compression / Concise)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                       BS (ms)              BSA (ms)            Concise (ms)        BS vs WAH   BSA vs WAH   CON vs WAH" << std::endl;
        std::cout << "  -------------------------------------------------------------------------------------------------------------" << std::endl;

        auto print_baseline_row = [&](const char* label, TimingStats& w,
                                      TimingStats& b, TimingStats& ba, TimingStats& c) {
            double bs_speedup  = (b.median  > 0) ? w.median / b.median  : 0;
            double bsa_speedup = (ba.median > 0) ? w.median / ba.median : 0;
            double con_speedup = (c.median  > 0) ? w.median / c.median  : 0;
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(8) << b.median  << " ± " << std::setw(6) << b.stddev
                      << "     " << std::setw(8) << ba.median << " ± " << std::setw(6) << ba.stddev
                      << "     " << std::setw(8) << c.median  << " ± " << std::setw(6) << c.stddev
                      << "     " << std::setw(5) << bs_speedup  << "x"
                      << "     " << std::setw(5) << bsa_speedup << "x"
                      << "     " << std::setw(5) << con_speedup << "x" << std::endl;
        };

        print_baseline_row("OR discount", wah_or_disc_s, bs_or_disc_s, bsa_or_disc_s, con_or_disc_s);
        print_baseline_row("OR quantity", wah_or_qty_s,  bs_or_qty_s,  bsa_or_qty_s,  con_or_qty_s);
        print_baseline_row("OR shipdate", wah_or_ship_s, bs_or_ship_s, bsa_or_ship_s, con_or_ship_s);
        print_baseline_row("AND (3-way)", wah_and_s,     bs_and_s,     bsa_and_s,     con_and_s);
        print_baseline_row("Decode",      wah_decode_s,  bs_decode_s,  bsa_decode_s,  con_decode_s);
        std::cout << "  -------------------------------------------------------------------------------------------------------------" << std::endl;
        print_baseline_row("TOTAL",       wah_total_s,   bs_total_s,   bsa_total_s,   con_total_s);

        std::cout << "\n  Bitset rows:         " << bs_row_count  << std::endl;
        std::cout << "  Bitset+AVX512 rows:  " << bsa_row_count << std::endl;
        std::cout << "  Concise rows:        " << con_row_count << std::endl;
        std::cout << "================================================================\n" << std::endl;
    } else {
        // ============================================================
        // Single-backend mode: focused per-phase summary for just
        // the selected backend (median ± stddev, min, max).  No
        // cross-backend speedup column — nothing to compare against.
        // ============================================================
        TimingStats *sel_disc = nullptr, *sel_qty = nullptr, *sel_ship = nullptr,
                    *sel_and  = nullptr, *sel_dec = nullptr, *sel_tot  = nullptr;
        size_t       sel_rows = 0;
        switch (Q6_BM) {
            case Q6BmType::WAH:
                sel_disc = &wah_or_disc_s; sel_qty = &wah_or_qty_s; sel_ship = &wah_or_ship_s;
                sel_and  = &wah_and_s;     sel_dec = &wah_decode_s; sel_tot  = &wah_total_s;
                sel_rows = wah_row_count; break;
            case Q6BmType::CB:
                sel_disc = &cb_or_disc_s;  sel_qty = &cb_or_qty_s;  sel_ship = &cb_or_ship_s;
                sel_and  = &cb_and_s;      sel_dec = &cb_decode_s;  sel_tot  = &cb_total_s;
                sel_rows = cb_row_count; break;
            case Q6BmType::CR:
                sel_disc = &cr_or_disc_s;  sel_qty = &cr_or_qty_s;  sel_ship = &cr_or_ship_s;
                sel_and  = &cr_and_s;      sel_dec = &cr_decode_s;  sel_tot  = &cr_total_s;
                sel_rows = cr_row_count; break;
            case Q6BmType::CRR:
                sel_disc = &crr_or_disc_s; sel_qty = &crr_or_qty_s; sel_ship = &crr_or_ship_s;
                sel_and  = &crr_and_s;     sel_dec = &crr_decode_s; sel_tot  = &crr_total_s;
                sel_rows = crr_row_count; break;
            case Q6BmType::EW:
                sel_disc = &ew_or_disc_s;  sel_qty = &ew_or_qty_s;  sel_ship = &ew_or_ship_s;
                sel_and  = &ew_and_s;      sel_dec = &ew_decode_s;  sel_tot  = &ew_total_s;
                sel_rows = ew_row_count; break;
            case Q6BmType::BS:
                sel_disc = &bs_or_disc_s;  sel_qty = &bs_or_qty_s;  sel_ship = &bs_or_ship_s;
                sel_and  = &bs_and_s;      sel_dec = &bs_decode_s;  sel_tot  = &bs_total_s;
                sel_rows = bs_row_count; break;
            case Q6BmType::BSA:
                sel_disc = &bsa_or_disc_s; sel_qty = &bsa_or_qty_s; sel_ship = &bsa_or_ship_s;
                sel_and  = &bsa_and_s;     sel_dec = &bsa_decode_s; sel_tot  = &bsa_total_s;
                sel_rows = bsa_row_count; break;
            case Q6BmType::CON:
                sel_disc = &con_or_disc_s; sel_qty = &con_or_qty_s; sel_ship = &con_or_ship_s;
                sel_and  = &con_and_s;     sel_dec = &con_decode_s; sel_tot  = &con_total_s;
                sel_rows = con_row_count; break;
            case Q6BmType::ALL: break;  // unreachable in this branch
        }

        std::cout << "\n================================================================" << std::endl;
        std::cout << "  RESULTS — " << q6_bm_label() << " only ("
                  << measured << " measured iterations)" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "                  median(ms)   stddev    min      max" << std::endl;
        std::cout << "  -------------------------------------------------------------" << std::endl;

        auto print_single = [](const char* label, TimingStats& s) {
            std::cout << "  " << std::left << std::setw(16) << label
                      << std::right << std::setw(9) << s.median
                      << std::setw(10) << s.stddev
                      << std::setw(10) << s.min_val
                      << std::setw(10) << s.max_val << std::endl;
        };
        if (sel_disc) {
            print_single("OR discount", *sel_disc);
            print_single("OR quantity", *sel_qty);
            print_single("OR shipdate", *sel_ship);
            print_single("AND (3-way)", *sel_and);
            print_single("Decode",      *sel_dec);
            std::cout << "  -------------------------------------------------------------" << std::endl;
            print_single("TOTAL",       *sel_tot);
            std::cout << "\n  " << q6_bm_label() << " rows: " << sel_rows << std::endl;
        }
        std::cout << "================================================================\n" << std::endl;
    }

    // ============================================================
    // CSV export — ALL mode only.  The full 8-backend CSV schema is
    // meaningful only when every backend ran; single-backend runs are
    // iterative dev tools and intentionally skip the export.
    // ============================================================
    if (run_all()) {
        std::string sf_label = get_sf_label();
        std::string csv_path = "q6_results_" + sf_label + ".csv";
        std::ofstream csv(csv_path);
        if (csv) {
            csv << std::fixed << std::setprecision(4);
            // Header
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
                               TimingStats& w, TimingStats& c, TimingStats& r,
                               TimingStats& rr, TimingStats& e,
                               TimingStats& b, TimingStats& ba, TimingStats& co) {
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

            csv_row("OR_discount", wah_or_disc_s, cb_or_disc_s, cr_or_disc_s, crr_or_disc_s, ew_or_disc_s, bs_or_disc_s, bsa_or_disc_s, con_or_disc_s);
            csv_row("OR_quantity", wah_or_qty_s,  cb_or_qty_s,  cr_or_qty_s,  crr_or_qty_s,  ew_or_qty_s,  bs_or_qty_s,  bsa_or_qty_s,  con_or_qty_s);
            csv_row("OR_shipdate", wah_or_ship_s, cb_or_ship_s, cr_or_ship_s, crr_or_ship_s, ew_or_ship_s, bs_or_ship_s, bsa_or_ship_s, con_or_ship_s);
            csv_row("AND",         wah_and_s,     cb_and_s,     cr_and_s,     crr_and_s,     ew_and_s,     bs_and_s,     bsa_and_s,     con_and_s);
            csv_row("Decode",      wah_decode_s,  cb_decode_s,  cr_decode_s,  crr_decode_s,  ew_decode_s,  bs_decode_s,  bsa_decode_s,  con_decode_s);
            csv_row("TOTAL",       wah_total_s,   cb_total_s,   cr_total_s,   crr_total_s,   ew_total_s,   bs_total_s,   bsa_total_s,   con_total_s);

            csv.close();
            std::cout << "  [CSV] Results written to: " << csv_path << std::endl;
        }

        // Also export compression ratio CSV (reuses the Q6ColumnSizes
        // data scanned once at the top of the function — avoids a
        // second round of ~400 stat() calls).
        std::string cr_csv_path = "q6_compression_" + sf_label + ".csv";
        std::ofstream cr_csv(cr_csv_path);
        if (cr_csv) {
            cr_csv << std::fixed << std::setprecision(4);
            cr_csv << "sf,column,count,raw_bytes,combit_bytes,wah_bytes,croaring_bytes,ewah_bytes,"
                   << "combit_ratio,wah_ratio,croaring_ratio,ewah_ratio\n";

            auto cr_row = [&](const std::string& col, const Q6ColumnSizes& s) {
                double cb_r  = s.raw > 0 ? (double)s.cb  / s.raw : 0;
                double wah_r = s.raw > 0 ? (double)s.wah / s.raw : 0;
                double cro_r = s.raw > 0 ? (double)s.cr  / s.raw : 0;
                double ew_r  = s.raw > 0 ? (double)s.ew  / s.raw : 0;
                cr_csv << sf_label << "," << col << "," << s.count << ","
                       << s.raw << "," << s.cb << "," << s.wah << "," << s.cr << "," << s.ew << ","
                       << cb_r << "," << wah_r << "," << cro_r << "," << ew_r << "\n";
            };

            cr_row("discount", disc_sz);
            cr_row("quantity", qty_sz);
            cr_row("shipdate", ship_sz);
            cr_row("TOTAL",    total_sz);

            cr_csv.close();
            std::cout << "  [CSV] Compression written to: " << cr_csv_path << std::endl;
        }
    }

    }); // end call_once
}

SourceResultType BMTableScan::BMTPCH_Q6(ExecutionContext &context, DataChunk &chunk, const TableScanBindData &bind_data)
{
    std::lock_guard<std::mutex> lock(q6_fetch_mutex);

    if(*cursor == 0) {
			TPCH_Q6_Lineitem_GetRowIds(context, row_ids);
			num_idlist = row_ids->size();
		}
		
		if(*cursor < row_ids->size()) {

			vector<StorageIndex> storage_column_ids;

			storage_column_ids.push_back(StorageIndex(6));	// For l_discount
			storage_column_ids.push_back(StorageIndex(5));	// For l_extendedprice

			TableScanState local_storage_state;
			local_storage_state.Initialize(storage_column_ids);
			ColumnFetchState column_fetch_state;

			auto &table_bind_data = bind_data;
			auto &transaction = DuckTransaction::Get(context.client, table_bind_data.table.catalog);

			data_ptr_t row_ids_data = nullptr;
			row_ids_data = (data_ptr_t)&((*row_ids)[*cursor]);
			Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
			idx_t fetch_count = 2048;
			if(*cursor + fetch_count > row_ids->size()) {
				fetch_count = row_ids->size() - *cursor;
			}

			table_bind_data.table.GetStorage().BMFetch(transaction, chunk, storage_column_ids, row_ids_vec, fetch_count,
                                                column_fetch_state, num_idlist);
			*cursor += fetch_count;
			return SourceResultType::HAVE_MORE_OUTPUT;
		}
		else {
			row_ids->clear();
            *cursor = 0;
            context.client.query_source = "tpch";

            return SourceResultType::FINISHED;
		}
}


}