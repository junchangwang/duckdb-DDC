// Shared helpers for the TPC-H bitmap benchmarks
// (Q1 / Q3 / Q4 / Q5 / Q6 / Q8 / Q10 / Q12).
// Header-only: no separate .cpp, no CMake changes needed.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <system_error>
#include <vector>

namespace duckdb {
namespace bm_bench {

// --- Scale factor ---

inline std::string sf_suffix() {
    const char* e = std::getenv("TPCH_SF");
    if (!e || !*e) return "";
    std::string s(e);
    return (s == "10") ? std::string() : ("_sf" + s);
}

inline std::string sf_label() {
    const char* e = std::getenv("TPCH_SF");
    return (e && *e) ? (std::string("SF") + e) : std::string("SF10");
}

inline int sf_value() {
    const char* e = std::getenv("TPCH_SF");
    return (e && *e) ? std::atoi(e) : 10;
}

// --- Bitmap directory resolution ---
//
// If DEBIT_BITMAP_DIR is exported, prepend it to each relative name so the
// benchmark works regardless of cwd.  If unset, return the name as-is,
// which preserves the legacy cwd-relative behaviour.
inline std::string resolve_bitmap_dir(const std::string& rel) {
    const char* base = std::getenv("DEBIT_BITMAP_DIR");
    if (!base || !*base) return rel;
    std::filesystem::path p(base);
    p /= rel;
    return p.string();
}

// --- Backend selection ---
//
// Unified DEBIT_BM selector.  Falls back to the legacy per-query variable
// (Q1_BM / Q5_BM / Q6_BM) so existing scripts keep working.
// ALL  = run every backend
// WAH  = WAH (FastBit / ibis::bitvector)
// CB   = ComBit
// CR   = CRoaring (no run-length opt)
// CRR  = CRoaring + runOptimize / fastunion
// EW   = EWAH
// BS   = uncompressed bitset, scalar (no algorithm + no SIMD baseline)
// BSA  = uncompressed bitset, AVX-512 (no algorithm + SIMD baseline)
// CON  = Concise (Colantonio & Di Pietro)
enum class Backend { ALL, WAH, CB, CR, CRR, EW, BS, BSA, CON };

inline Backend parse_backend(const char* legacy_env) {
    const char* env = std::getenv("DEBIT_BM");
    if (!env || !*env) env = std::getenv(legacy_env);
    if (!env || !*env) return Backend::ALL;

    std::string s(env);
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (s.empty() || s == "all")                                    return Backend::ALL;
    if (s == "wah")                                                 return Backend::WAH;
    if (s == "cb"  || s == "combit")                                return Backend::CB;
    if (s == "cr"  || s == "croaring")                              return Backend::CR;
    if (s == "crr" || s == "croaring_run" || s == "croaring+run")   return Backend::CRR;
    if (s == "ew"  || s == "ewah")                                  return Backend::EW;
    if (s == "bs"  || s == "bitset")                                return Backend::BS;
    if (s == "bsa" || s == "bitset_avx512" || s == "bitset_simd")   return Backend::BSA;
    if (s == "con" || s == "concise")                               return Backend::CON;

    std::cerr << "[DEBIT_BM] unknown value '" << env
              << "' — defaulting to ALL. Accepted: all|wah|cb|cr|crr|ew|bs|bsa|con\n";
    return Backend::ALL;
}

inline const char* backend_label(Backend b) {
    switch (b) {
        case Backend::ALL: return "ALL";
        case Backend::WAH: return "WAH";
        case Backend::CB:  return "ComBit";
        case Backend::CR:  return "CRoaring";
        case Backend::CRR: return "CRoaring+Run";
        case Backend::EW:  return "EWAH";
        case Backend::BS:  return "Bitset";
        case Backend::BSA: return "Bitset+AVX512";
        case Backend::CON: return "Concise";
    }
    return "?";
}

// --- Iterations / warmup ---

inline int iter_count(int fallback) {
    const char* e = std::getenv("DEBIT_ITER");
    if (e && *e) { int v = std::atoi(e); if (v > 0) return v; }
    return fallback;
}

inline int warmup_count(int fallback) {
    const char* e = std::getenv("DEBIT_WARMUP");
    if (e && *e) { int v = std::atoi(e); if (v >= 0) return v; }
    return fallback;
}

// --- Directory size in bytes ---
//
// Returns the sum of regular-file sizes under `dir`, or 0 if the path
// does not exist (or is unreadable).  Safe to call for every backend
// during load: errors are silently treated as "missing".
inline uint64_t dir_size_bytes(const std::string& dir) {
    std::error_code ec;
    std::filesystem::path p(dir);
    if (!std::filesystem::exists(p, ec)) return 0;
    uint64_t total = 0;
    for (auto it = std::filesystem::recursive_directory_iterator(p, ec);
         !ec && it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            auto sz = it->file_size(ec);
            if (!ec) total += sz;
        }
    }
    return total;
}

inline double mib(uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

// --- SF=1 duplicate-load warning ---
//
// The SF1 database checked into the repo is known to contain duplicate
// rows (it was loaded twice during setup).  Emit a loud warning so
// benchmark output is never silently taken as authoritative.
inline void warn_if_sf1() {
    const char* e = std::getenv("TPCH_SF");
    if (e && std::string(e) == "1") {
        std::cerr
            << "WARNING: TPCH_SF=1 — this database contains duplicate-loaded rows. "
            << "Aggregates will be ~2x the spec. Use TPCH_SF=10 for valid results."
            << std::endl;
    }
}

// --- Stats helper (median / stddev / min / max) ---

struct Stats { double median = 0, stddev = 0, min_val = 0, max_val = 0; };

inline Stats compute_stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    s.median  = (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
    s.min_val = v.front();
    s.max_val = v.back();
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double sq = 0;
    for (auto x : v) sq += (x - mean) * (x - mean);
    s.stddev = std::sqrt(sq / n);
    return s;
}

} // namespace bm_bench
} // namespace duckdb
