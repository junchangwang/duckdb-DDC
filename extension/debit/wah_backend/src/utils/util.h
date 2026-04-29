// Minimal compatibility header for the FastBit WAH bitvector (`bitvector.cpp`).
//
// The original CUBIT-d util.h shipped a sprawling Table_config (file paths,
// thread counts, encoding scheme, segment layout, ...) and a boost-based
// CLI parser, all of which were consumed exclusively by the rabit/ub/ucb/
// naive bitmap stores that this codebase no longer ships.  bitvector.cpp
// itself only touches two members of Table_config (`n_rows` and
// `enable_fence_pointer`) inside its `setBit` / `getBit` / `decode`
// virtual overrides, plus the global `enable_fence_pointer` toggle that
// gates fence-pointer maintenance in the OR fast paths.  Anything beyond
// that has been removed.
//
// The 12 modernised TPC-H benchmarks under extension/debit/execution/tpch/
// query/ never reach the setBit / getBit paths — they read pre-built WAH
// files from disk and operate on the resulting `ibis::bitvector` purely
// through the read-only OR / decompress / pit iterator API.  The
// definitions here therefore exist solely to satisfy the type checker;
// `Table_config` is never instantiated at runtime by any active code
// path in this tree.
#ifndef UTIL_H
#define UTIL_H

#include <cstdint>

class Table_config {
public:
    uint64_t n_rows = 0;
    bool enable_fence_pointer = false;
};

// Defined in utils/util.cpp — read by the OR fast paths in bitvector.cpp
// (e.g. `if (enable_fence_pointer) res.m_vec.reserve(...)`).  Kept as a
// process-wide flag for binary compatibility; never written by any code
// in this tree, so its value is always `false` in practice.
extern bool enable_fence_pointer;

#endif
