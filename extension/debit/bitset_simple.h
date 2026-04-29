// bitset_simple.h
//
// Uncompressed bitmap baseline ("no algorithm") for the TPC-H bitmap
// benchmarks (Q1/Q5/Q6/Q10).  A `bool simd` flag selects between a
// strictly-scalar path (BS — pure baseline) and an AVX-512 path (BSA —
// uncompressed baseline + SIMD).  The same Bitmap type backs both so
// memory and load costs are identical; only the OR/AND/popcount kernels
// differ.
//
// File format = raw packed bits (LSB-first per byte), exactly the
// gen_bitmap output that compress_tpch_q* consumes.  Storage is 64-byte
// aligned and word count is rounded up to a multiple of 8 so AVX-512
// 512-bit loads never need tail handling.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace bs {

struct Bitmap {
    uint64_t* words  = nullptr;
    size_t    nwords = 0;        // padded to multiple of 8 (AVX-512 friendly)
    uint64_t  nbits  = 0;

    Bitmap() = default;
    Bitmap(const Bitmap&)            = delete;
    Bitmap& operator=(const Bitmap&) = delete;

    Bitmap(Bitmap&& o) noexcept
        : words(o.words), nwords(o.nwords), nbits(o.nbits) {
        o.words = nullptr; o.nwords = 0; o.nbits = 0;
    }
    Bitmap& operator=(Bitmap&& o) noexcept {
        if (this != &o) {
            std::free(words);
            words = o.words; nwords = o.nwords; nbits = o.nbits;
            o.words = nullptr; o.nwords = 0; o.nbits = 0;
        }
        return *this;
    }
    ~Bitmap() { std::free(words); }

    void alloc_for_bits(uint64_t bits) {
        std::free(words);
        nbits = bits;
        size_t need = (bits + 63) / 64;
        nwords = (need + 7) & ~size_t(7);
        words  = static_cast<uint64_t*>(std::aligned_alloc(64, nwords * sizeof(uint64_t)));
        std::memset(words, 0, nwords * sizeof(uint64_t));
    }

    Bitmap clone() const {
        Bitmap r;
        r.alloc_for_bits(nbits);
        std::memcpy(r.words, words, nwords * sizeof(uint64_t));
        return r;
    }
};

// ----------------------------------------------------------------------------
// AVX-512 kernels (only emitted when target supports it; runtime guarded).
// ----------------------------------------------------------------------------
#if defined(__AVX512F__)
__attribute__((target("avx512f")))
inline void or_simd(uint64_t* __restrict a, const uint64_t* __restrict b, size_t n) {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_si512(a + i, _mm512_or_si512(
            _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i)));
}
__attribute__((target("avx512f")))
inline void and_simd(uint64_t* __restrict a, const uint64_t* __restrict b, size_t n) {
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_si512(a + i, _mm512_and_si512(
            _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i)));
}
__attribute__((target("avx512f")))
inline void not_simd(uint64_t* __restrict a, size_t n) {
    const __m512i ones = _mm512_set1_epi64(-1);
    for (size_t i = 0; i + 8 <= n; i += 8)
        _mm512_storeu_si512(a + i, _mm512_xor_si512(_mm512_loadu_si512(a + i), ones));
}
#endif

// Scalar kernels: explicitly de-vectorised so BS measures a real "no
// algorithm + no SIMD" floor (otherwise GCC would auto-vectorise these
// trivial loops and there'd be no difference between BS and BSA).
__attribute__((optimize("no-tree-vectorize")))
inline void or_scalar(uint64_t* __restrict a, const uint64_t* __restrict b, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] |= b[i];
}
__attribute__((optimize("no-tree-vectorize")))
inline void and_scalar(uint64_t* __restrict a, const uint64_t* __restrict b, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] &= b[i];
}
__attribute__((optimize("no-tree-vectorize")))
inline void not_scalar(uint64_t* __restrict a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = ~a[i];
}

// ----------------------------------------------------------------------------
// Public ops: dispatch on simd flag.
// ----------------------------------------------------------------------------
inline void or_inplace(Bitmap& a, const Bitmap& b, bool simd) {
    size_t n = std::min(a.nwords, b.nwords);
#if defined(__AVX512F__)
    if (simd) { or_simd(a.words, b.words, n); return; }
#else
    (void)simd;
#endif
    or_scalar(a.words, b.words, n);
}

inline void and_inplace(Bitmap& a, const Bitmap& b, bool simd) {
    size_t n = std::min(a.nwords, b.nwords);
#if defined(__AVX512F__)
    if (simd) { and_simd(a.words, b.words, n); return; }
#else
    (void)simd;
#endif
    and_scalar(a.words, b.words, n);
}

// In-place complement.  Flips every word then clears the bits past
// nbits in the last logical word and zero-fills the padding tail, so
// downstream AND / decode see only the logical [0, nbits) range.
inline void not_inplace(Bitmap& bm, bool simd) {
#if defined(__AVX512F__)
    if (simd) not_simd(bm.words, bm.nwords);
    else      not_scalar(bm.words, bm.nwords);
#else
    (void)simd;
    not_scalar(bm.words, bm.nwords);
#endif
    // Clear bits past nbits in the last logical word (everything in
    // padding words stays 0 because nwords is a multiple of 8 and the
    // tail words were zeroed at alloc).
    const uint64_t tail = bm.nbits & 63;
    if (tail != 0 && bm.nbits > 0) {
        const size_t last = (bm.nbits - 1) / 64;
        bm.words[last] &= (uint64_t(1) << tail) - 1;
    }
    // Padding words [last+1 .. nwords) must stay 0 — they were 0 before
    // the flip (alloc memset), so flipping made them all-ones.  Reset.
    const size_t logical_words = (bm.nbits + 63) / 64;
    for (size_t i = logical_words; i < bm.nwords; ++i) bm.words[i] = 0;
}

// Popcount over the logical [0, nbits) range.  Padding words are kept
// zero by alloc / not_inplace tail-clear, so a flat sum over the whole
// words[] array is correct and avoids a per-word bounds check.
//   simd=false  → scalar __builtin_popcountll loop (BS baseline).
//   simd=true   → AVX-512 VPOPCNTDQ-accelerated reduction (BSA).
__attribute__((optimize("no-tree-vectorize")))
inline size_t popcount_scalar(const uint64_t* __restrict a, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) c += __builtin_popcountll(a[i]);
    return c;
}
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
__attribute__((target("avx512f,avx512vpopcntdq")))
inline size_t popcount_simd(const uint64_t* __restrict a, size_t n) {
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(a + i)));
    size_t total = static_cast<size_t>(_mm512_reduce_add_epi64(acc));
    for (; i < n; ++i) total += __builtin_popcountll(a[i]);
    return total;
}
#endif
inline size_t popcount(const Bitmap& bm, bool simd) {
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    if (simd) return popcount_simd(bm.words, bm.nwords);
#else
    (void)simd;
#endif
    return popcount_scalar(bm.words, bm.nwords);
}

// Fused AND-popcount over `a` and `b` without materialising the
// intersection — `popcount(a & b)` in a single streaming pass.  Mirrors
// CRoaring's `and_cardinality`, EWAH's `logicalandcount`, Concise's
// `logicalandCount`, and ibis::bitvector's `count(mask)`, so the BS / BSA
// baselines compete on equal API footing with the compressed backends.
//   simd=false → scalar __builtin_popcountll loop (BS).
//   simd=true  → AVX-512 VPOPCNTDQ-accelerated reduction (BSA).
__attribute__((optimize("no-tree-vectorize")))
inline size_t and_popcount_scalar(const uint64_t* __restrict a,
                                  const uint64_t* __restrict b, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) c += __builtin_popcountll(a[i] & b[i]);
    return c;
}
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
__attribute__((target("avx512f,avx512vpopcntdq")))
inline size_t and_popcount_simd(const uint64_t* __restrict a,
                                const uint64_t* __restrict b, size_t n) {
    __m512i acc = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
    }
    size_t total = static_cast<size_t>(_mm512_reduce_add_epi64(acc));
    for (; i < n; ++i) total += __builtin_popcountll(a[i] & b[i]);
    return total;
}
#endif
inline size_t and_popcount(const Bitmap& a, const Bitmap& b, bool simd) {
    size_t n = std::min(a.nwords, b.nwords);
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    if (simd) return and_popcount_simd(a.words, b.words, n);
#else
    (void)simd;
#endif
    return and_popcount_scalar(a.words, b.words, n);
}

// Decode set bits to row IDs (used by Q6's row-id materialisation).
inline void decode(const Bitmap& bm, std::vector<int64_t>& out) {
    out.clear();
    const uint64_t nb = bm.nbits;
    for (size_t i = 0; i < bm.nwords; ++i) {
        uint64_t w = bm.words[i];
        const uint64_t base = static_cast<uint64_t>(i) * 64;
        while (w) {
            const uint64_t pos = base + __builtin_ctzll(w);
            if (pos >= nb) return;          // safety: padding tail (always 0)
            out.push_back(static_cast<int64_t>(pos));
            w &= w - 1;
        }
    }
}

} // namespace bs
