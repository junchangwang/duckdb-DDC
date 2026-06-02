#include "ddc.h"

#include <cassert>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

// NOT in place
DDCBtv&
DDCBtv::negate_inplace() {
    if (bit_count_ == 0) return *this;
    assert(state_ != State::Uncompressed);

    // flip fill bit
    l1_fill_ones_ = !l1_fill_ones_;

    uint8_t* data = l1_literals_.data();
    size_t n = l1_literals_.size();

#ifdef __AVX512F__
    // SIMD xor literals
    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));
    size_t i = 0;

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

    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(data + i);
        _mm512_storeu_si512(data + i, _mm512_xor_si512(v, ones));
    }

    // masked tail
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

    // mask padding bits
    if (bit_count_ % 8 != 0 && l1_literal_count_ > 0 && is_last_word_literal()) {
        size_t valid_bits = bit_count_ % 8;
        size_t byte_off = l1_literal_count_ - 1;
        l1_literals_[byte_off] &= static_cast<uint8_t>(0xFF << (8 - valid_bits));
    }

    return *this;
}

// NOT, new btv
DDCBtv
DDCBtv::operator~() const {
    if (bit_count_ == 0) return DDCBtv();
    assert(state_ != State::Uncompressed);

    // all-fill fast path
    if (l1_literal_count_ == 0) {
        return make_all_fill(bit_count_, l2_count_, !l1_fill_ones_);
    }

    DDCBtv result( !l1_fill_ones_,
                      l2_fill_ones_,
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
    // SIMD xor copy
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

// per-segment NOT
DDC&
DDC::negate_inplace() {
    for (auto& seg : segments_) {
        if (seg.l1_literal_count_ == 0) {

            seg.l1_fill_ones_ = !seg.l1_fill_ones_;
            continue;
        }
        seg.negate_inplace();
    }
    return *this;
}

DDC
DDC::operator~() const {
    DDC result;
    result.bit_count_    = bit_count_;
    result.segment_bits_ = segment_bits_;
    result.segments_.reserve(segments_.size());
    for (const auto& seg : segments_) {
        result.segments_.push_back(~seg);
    }
    return result;
}
