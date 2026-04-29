// bm_baseline_loaders.hpp
//
// Shared loaders for the two "no-compression" baselines (uncompressed
// bs::Bitmap and Concise) used by every TPC-H bitmap benchmark
// (Q1 / Q3 / Q4 / Q5 / Q6 / Q8 / Q10 / Q12).  Both rebuild from the
// CRoaring serialized .bm file because that's the only flavour that
// ships at SF10 in this checkout.  Cost is paid once at load and
// never re-measured.
#pragma once

#include "bitset_simple.h"
#include "Concise/concise.h"
#include "roaring.hh"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace duckdb {
namespace bm_bench {

// CRoaring file layout: [uint32_t logical_size][serialized roaring data].
// Helper extracts (logical_size, sorted ascending positions) from such a file.
inline bool decode_croaring_file(const std::string& path,
                                 uint32_t& logical_size,
                                 std::vector<uint32_t>& positions) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    std::streamsize fsize = in.tellg();
    in.seekg(0, std::ios::beg);
    logical_size = 0;
    in.read(reinterpret_cast<char*>(&logical_size), sizeof(logical_size));
    std::streamsize body = fsize - static_cast<std::streamsize>(sizeof(logical_size));
    positions.clear();
    if (body > 0) {
        std::vector<char> buf(body);
        in.read(buf.data(), body);
        roaring::Roaring r = roaring::Roaring::readSafe(buf.data(), body);
        positions.resize(r.cardinality());
        r.toUint32Array(positions.data());
    }
    return true;
}

inline bs::Bitmap load_bitmap_from_croaring(const std::string& path) {
    bs::Bitmap bm;
    uint32_t nbits = 0;
    std::vector<uint32_t> pos;
    if (!decode_croaring_file(path, nbits, pos)) return bm;
    bm.alloc_for_bits(nbits);
    for (uint32_t p : pos) bm.words[p / 64] |= uint64_t(1) << (p % 64);
    return bm;
}

inline ConciseSet<false> load_concise_from_croaring(const std::string& path) {
    ConciseSet<false> cs;
    uint32_t nbits = 0;
    std::vector<uint32_t> pos;
    if (!decode_croaring_file(path, nbits, pos)) return cs;
    // toUint32Array produces sorted ascending positions, which is the
    // monotone fast path of ConciseSet::add().
    for (uint32_t p : pos) cs.add(p);
    return cs;
}

} // namespace bm_bench
} // namespace duckdb
