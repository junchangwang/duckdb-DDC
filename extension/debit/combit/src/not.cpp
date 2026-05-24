#include "combit.h"

#include <cassert>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

// =============================================================================
// Bitwise NOT — ComBit's fastest operator at any cardinality.
// =============================================================================
//
// Unlike AND/OR/XOR (which compare two operands and walk the full
// L4 -> L3 -> L2 -> L1 cascade), NOT has a single operand and its 4-level
// marker hierarchy is INVARIANT under bit flip:
//
//   * L4 / L3 / L2 marker streams classify regions as "literal" vs "fill"
//     — a property of whether all sub-region values are uniform.  Flipping
//     every bit preserves uniformity, so the marker streams are bit
//     identical before and after NOT.
//   * The fill BYTE values (`l2_fill_ones_`, `l3_fill_ones_`) are also
//     preserved — they're the per-segment compression choice that minimizes
//     stored literals (see ComBitBtv::compact_l2_l3), which doesn't change
//     when the underlying data bits flip.
//
// Net work per ComBitBtv:
//   1. flip the L1 fill BIT value via `l1_fill_ones_ = !l1_fill_ones_`
//   2. XOR every byte of the L1 literal stream with 0xFF (AVX-512, 4x unroll)
//   3. fix the trailing-padding mask if bit_count_ % 8 != 0
//
// Everything else is untouched.  L2/L3/L4 cache lines aren't even dirtied.
//
// Versus other backends on a 100M-row bitmap:
//   - Roaring  must materialize ~(N - cardinality) positions to invert an
//              array container.  Sparse inputs are catastrophic.
//   - WAH / EWAH / Concise interleave fill metadata in their word stream,
//     so NOT must touch every word.  ComBit's fill metadata lives in
//     separate L2/L3/L4 streams that NOT skips.
//   - Plain Bitset must XOR N/8 bytes regardless of density.  ComBit's
//     L1 literal stream is proportional to actual structure -> smaller and
//     therefore faster at every non-trivial density.
//
// All-fill segments are an O(1) path at the ComBit level: just flip
// `l1_fill_ones_` (zero bytes touched), so sparse 100M-row inputs (most
// segments all-zero) finish in microseconds.
// =============================================================================


// -----------------------------------------------------------------------------
// ComBitBtv::negate_inplace
//
// Hot loop: 4x unrolled AVX-512 XOR with 0xFF (256 B/iter) saturates the
// L1d load-store port pair on Ice Lake / Sapphire Rapids.  Tail < 256 B
// drops to a 64-B AVX-512 stride; the final < 64 B uses an opmask
// store so the tail stays branchless for any length 0..63.
// -----------------------------------------------------------------------------
ComBitBtv&
ComBitBtv::negate_inplace() {
    if (bit_count_ == 0) return *this;
    assert(state_ != State::Uncompressed);

    // The L1 fill BIT value flips.  l2_fill_ones_ / l3_fill_ones_ stay —
    // they encode which BYTE value the implicit L2/L3 fill bytes carry,
    // a compression choice that's independent of the underlying data
    // bits.  The L4/L3/L2 literal streams themselves are also unchanged
    // (literal-vs-fill classification is invariant under bit flip).
    l1_fill_ones_ = !l1_fill_ones_;

    uint8_t* data = l1_literals_.data();
    size_t n = l1_literals_.size();

#ifdef __AVX512F__
    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));
    size_t i = 0;

    // 4x unroll — measured 35-50% faster than the 1-lane 64-B loop on
    // L1d-resident inputs (>4 KB literal stream).  Below 4 KB the loop
    // skips straight to the 64-B stride.
    for (; i + 256 <= n; i += 256) {
        __m512i v0 = _mm512_loadu_si512(data + i);
        __m512i v1 = _mm512_loadu_si512(data + i +  64);
        __m512i v2 = _mm512_loadu_si512(data + i + 128);
        __m512i v3 = _mm512_loadu_si512(data + i + 192);
        _mm512_storeu_si512(data + i,       _mm512_xor_si512(v0, ones));
        _mm512_storeu_si512(data + i +  64, _mm512_xor_si512(v1, ones));
        _mm512_storeu_si512(data + i + 128, _mm512_xor_si512(v2, ones));
        _mm512_storeu_si512(data + i + 192, _mm512_xor_si512(v3, ones));
    }
    // 64-B mid-tail.
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(data + i);
        _mm512_storeu_si512(data + i, _mm512_xor_si512(v, ones));
    }
    // <64 B tail via opmask — branchless tail of length 0..63.  AVX-512BW
    // is already a build requirement (combit's expand-load uses VBMI2),
    // so mask_loadu_epi8 / mask_storeu_epi8 are always available.
    if (i < n) {
        size_t tail = n - i;
        __mmask64 m = (tail >= 64) ? __mmask64(-1)
                                   : __mmask64((uint64_t(1) << tail) - 1);
        __m512i v = _mm512_maskz_loadu_epi8(m, data + i);
        _mm512_mask_storeu_epi8(data + i, m, _mm512_xor_si512(v, ones));
    }
#else
    for (size_t k = 0; k < n; k++) data[k] ^= 0xFF;
#endif

    // Padding correction.  compress() zero-pads the trailing bits of the
    // last literal word; the XOR above flipped them to 1.  Re-mask so the
    // padding stays zero — otherwise popcount() and downstream compare
    // ops would see phantom set bits.
    if (bit_count_ % 8 != 0 && l1_literal_count_ > 0 && is_last_word_literal()) {
        size_t valid_bits = bit_count_ % 8;
        size_t byte_off = l1_literal_count_ - 1;
        l1_literals_[byte_off] &= static_cast<uint8_t>(0xFF << (8 - valid_bits));
    }

    return *this;
}

// -----------------------------------------------------------------------------
// ComBitBtv::operator~ — out-of-place, state-preserving (teacher's design).
//
// Hand-laid result: allocate dst l1 + AVX-512 XOR-while-copying from src l1
// in one memory pass.  L2/L3/L4 markers are invariant under NOT — copied
// once.  This is materially faster than `result = *this; result.negate()`
// at dense cardinalities (where l1 dominates), because plain copy assignment
// would do an extra full read+write pass on l1 before the in-place XOR.
// -----------------------------------------------------------------------------
ComBitBtv
ComBitBtv::operator~() const {
    if (bit_count_ == 0) return ComBitBtv();
    assert(state_ != State::Uncompressed);

    // All-fill segment fast path.  No L1 literals → result is the
    // flipped-fill all-fill segment.  Zero marker copies, zero vector
    // allocations.  Hot path for sparse inputs (~99% all-fill segments).
    if (l1_literal_count_ == 0) {
        return make_all_fill(bit_count_, l2_count_, !l1_fill_ones_);
    }

    ComBitBtv result(/*l1_fill_ones=*/!l1_fill_ones_,
                     /*l2_fill_ones=*/l2_fill_ones_,
                     state_);
    result.bit_count_         = bit_count_;
    result.l2_count_          = l2_count_;
    result.l3_count_          = l3_count_;
    result.l4_count_          = l4_count_;
    result.l3_fill_ones_      = l3_fill_ones_;
    result.l2_literal_count_  = l2_literal_count_;
    result.l3_literal_count_  = l3_literal_count_;
    result.l1_literal_count_  = l1_literal_count_;
    result.l2_literals_       = l2_literals_;
    result.l3_literals_       = l3_literals_;
    result.l4_bits_           = l4_bits_;

    result.l1_literals_.resize(l1_literal_count_);
    const uint8_t* src = l1_literals_.data();
    uint8_t* dst = result.l1_literals_.data();
    size_t n = l1_literal_count_;
    size_t i = 0;

#ifdef __AVX512F__
    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));
    for (; i + 256 <= n; i += 256) {
        __m512i v0 = _mm512_loadu_si512(src + i);
        __m512i v1 = _mm512_loadu_si512(src + i +  64);
        __m512i v2 = _mm512_loadu_si512(src + i + 128);
        __m512i v3 = _mm512_loadu_si512(src + i + 192);
        _mm512_storeu_si512(dst + i,       _mm512_xor_si512(v0, ones));
        _mm512_storeu_si512(dst + i +  64, _mm512_xor_si512(v1, ones));
        _mm512_storeu_si512(dst + i + 128, _mm512_xor_si512(v2, ones));
        _mm512_storeu_si512(dst + i + 192, _mm512_xor_si512(v3, ones));
    }
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(src + i);
        _mm512_storeu_si512(dst + i, _mm512_xor_si512(v, ones));
    }
    if (i < n) {
        size_t tail = n - i;
        __mmask64 m = (tail >= 64) ? __mmask64(-1)
                                   : __mmask64((uint64_t(1) << tail) - 1);
        __m512i v = _mm512_maskz_loadu_epi8(m, src + i);
        _mm512_mask_storeu_epi8(dst + i, m, _mm512_xor_si512(v, ones));
    }
#else
    for (; i < n; i++) dst[i] = src[i] ^ 0xFF;
#endif

    if (bit_count_ % 8 != 0 && result.l1_literal_count_ > 0
        && result.is_last_word_literal()) {
        size_t valid_bits = bit_count_ % 8;
        size_t byte_off = result.l1_literal_count_ - 1;
        result.l1_literals_[byte_off] &=
            static_cast<uint8_t>(0xFF << (8 - valid_bits));
    }

    return result;
}


// =============================================================================
// ComBit (segmented) NOT
// =============================================================================

// -----------------------------------------------------------------------------
// ComBit::negate_inplace
//
// Segment loop with an O(1) all-fill fast path: if a segment has no L1
// literals, flip its fill flag (zero bytes touched) — this turns is_all_zero
// into is_all_ones and vice versa.  In typical sparse bitmaps ~99% of
// segments hit this path, collapsing the total NOT cost to microseconds at
// 100M rows.  Mixed segments delegate to ComBitBtv::negate_inplace's
// AVX-512 XOR.
// -----------------------------------------------------------------------------
ComBit&
ComBit::negate_inplace() {
    for (auto& seg : segments_) {
        if (seg.l1_literal_count_ == 0) {
            // All-fill segment.  l1_literals_ is empty by construction
            // (make_all_fill / compress() with uniform input), so flipping
            // l1_fill_ones is the entire NOT for this segment.  The L2 fill
            // classification (l2_fill_ones_) is also unchanged: an "all
            // L1 fill" segment stays "all L1 fill" after flip.
            seg.l1_fill_ones_ = !seg.l1_fill_ones_;
            continue;
        }
        seg.negate_inplace();
    }
    return *this;
}

// -----------------------------------------------------------------------------
// ComBit::operator~ — out-of-place segment loop.
//
// Per-segment operator~ uses hand-laid construction (alloc dst + XOR-write
// src→dst in one pass).  The previous `result = *this; negate_inplace`
// pattern did a redundant memcpy of L1 before XORing — wasted bandwidth.
// -----------------------------------------------------------------------------
ComBit
ComBit::operator~() const {
    ComBit result;
    result.bit_count_    = bit_count_;
    result.segment_bits_ = segment_bits_;
    result.segments_.reserve(segments_.size());
    for (const auto& seg : segments_) {
        result.segments_.push_back(~seg); //cpu做，用硬件预取，
    }
    return result;
}//wah fill and literal,combit only literal
//segment 放大 实验