

#pragma once

#include "bitset_simple.h"
#include "Concise/concise.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace duckdb {
namespace bm_bench {

// decode roaring file
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
    // set bits
    for (uint32_t p : pos) bm.words[p / 64] |= uint64_t(1) << (p % 64);
    return bm;
}

inline ConciseSet<false> load_concise_from_croaring(const std::string& path) {
    ConciseSet<false> cs;
    uint32_t nbits = 0;
    std::vector<uint32_t> pos;
    if (!decode_croaring_file(path, nbits, pos)) return cs;

    for (uint32_t p : pos) cs.add(p);
    return cs;
}

// WAH NOT
inline void wah_flip(ibis::bitvector* btv) {
#if defined(__AVX512F__)
    // SIMD flip
    auto* it = btv->m_vec.begin();
    while (it + 15 < btv->m_vec.end()) {
        _mm512_storeu_epi32(it, _mm512_andnot_epi32(
            _mm512_loadu_epi32(it), _mm512_set1_epi32(0x7fffffff)));
        it += 16;
    }
    // scalar tail
    for (; it < btv->m_vec.end(); it++) *it ^= ibis::bitvector::ALLONES;
    if (btv->active.nbits > 0)
        btv->active.val ^= ((1u << btv->active.nbits) - 1);
#else
    for (auto* it = btv->m_vec.begin(); it < btv->m_vec.end(); ++it)
        *it ^= ibis::bitvector::ALLONES;
    if (btv->active.nbits > 0)
        btv->active.val ^= ((1u << btv->active.nbits) - 1);
#endif
}

}
}
