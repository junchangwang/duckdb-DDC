#include "ddc.h"

// XOR kernel
DDCBtv
DDCBtv::operator^(const DDCBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    if (bit_count_ == 0) return DDCBtv();
    assert(state_ != State::Uncompressed);
    assert(other.state_ != State::Uncompressed);

    const size_t total_words = l2_count_;

    const bool compress = ddc_compress_results;
    DDCBtv result = compress ? DDCBtv(false, false, State::Compressed)
                                : DDCBtv(false, true, State::Decompressed);
    result.bit_count_ = bit_count_;
    result.l2_count_ = total_words;
    size_t l2_byte_count = (total_words + 7) / 8;

    if (compress) {
        result.l2_flat_.assign(l2_byte_count, 0x00);
    } else {

        result.l3_count_ = l2_byte_count;
        result.l2_literal_count_ = 0;
    }

    result.l1_literals_.resize(total_words);
    result.l1_literal_count_ = total_words;

    const uint8_t* a_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    uint8_t* r_l1 = result.l1_literals_.data();

    size_t r_off = 0;

#ifdef DDC_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    // SIMD path
    const size_t avx_regions = total_words / words_per_reg;

    SideCtx A = this->make_side(a_l1);
    SideCtx B = other.make_side(b_l1);

    static constexpr size_t PF_DIST = 128;

    uint8_t* result_l2 = result.l2_flat_.data();

    const bool a_zero_when_l3_zero = !A.l1_fill_ones && !A.l2_fill_ones;
    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;

    const uint8_t a_l3_fill = A.l3_fill_ones ? 0xFF : 0x00;
    const uint8_t b_l3_fill = B.l3_fill_ones ? 0xFF : 0x00;

    // per-region hot loop
    for (size_t region = 0; region < avx_regions; region++) {
        bool a_l4_lit = (A.l4_bits[region / 8] >> (region % 8)) & 1;
        bool b_l4_lit = (B.l4_bits[region / 8] >> (region % 8)) & 1;
        uint8_t l3a = a_l4_lit ? A.l3_lits[A.l3_lit_off++] : a_l3_fill;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;

        // A empty: bypass
        if (a_zero_when_l3_zero && l3a == 0) {
            if (b_zero_when_l3_zero && l3b == 0) {

                if (!compress) {
                    _mm512_storeu_si512(r_l1 + r_off, _mm512_setzero_si512());
                    r_off += 64;
                }
                continue;
            }

            __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                static_cast<__mmask64>(l3b), B.l2_lits + B.l2_lit_off);
            B.l2_lit_off += __builtin_popcount(l3b);
            __mmask64 mb = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));
            __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb,
                B.l1_lits + B.l1_lit_off);
            B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));
            if (compress) {
                __mmask64 lit_mask = _mm512_test_epi8_mask(vb, vb);
                uint64_t mask_val = static_cast<uint64_t>(lit_mask);
                std::memcpy(result_l2 + region * 8, &mask_val, 8);
                _mm512_mask_compressstoreu_epi8(r_l1 + r_off, lit_mask, vb);
                r_off += __builtin_popcountll(mask_val);
            } else {
                _mm512_storeu_si512(r_l1 + r_off, vb);
                r_off += 64;
            }
            continue;
        }
        // B empty: copy A
        if (b_zero_when_l3_zero && l3b == 0) {

            __m512i l2a_v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                static_cast<__mmask64>(l3a), A.l2_lits + A.l2_lit_off);
            A.l2_lit_off += __builtin_popcount(l3a);
            __mmask64 ma = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));
            __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec, ma,
                A.l1_lits + A.l1_lit_off);
            A.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));
            if (compress) {
                __mmask64 lit_mask = _mm512_test_epi8_mask(va, va);
                uint64_t mask_val = static_cast<uint64_t>(lit_mask);
                std::memcpy(result_l2 + region * 8, &mask_val, 8);
                _mm512_mask_compressstoreu_epi8(r_l1 + r_off, lit_mask, va);
                r_off += __builtin_popcountll(mask_val);
            } else {
                _mm512_storeu_si512(r_l1 + r_off, va);
                r_off += 64;
            }
            continue;
        }

        // prefetch
        _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_lit_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(B.l1_lits + B.l1_lit_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(r_l1 + r_off + PF_DIST), _MM_HINT_T0);

        __m512i l2a_v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
            static_cast<__mmask64>(l3a), A.l2_lits + A.l2_lit_off);
        A.l2_lit_off += __builtin_popcount(l3a);
        __mmask64 ma = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));

        __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_lit_off);
        B.l2_lit_off += __builtin_popcount(l3b);
        __mmask64 mb = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

        // expand L1
        __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec, ma, A.l1_lits + A.l1_lit_off);
        A.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));

        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb, B.l1_lits + B.l1_lit_off);
        B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i vr = _mm512_xor_si512(va, vb);
        // emit result
        if (compress) {
            __mmask64 lit_mask = _mm512_test_epi8_mask(vr, vr);
            uint64_t mask_val = static_cast<uint64_t>(lit_mask);
            std::memcpy(result_l2 + region * 8, &mask_val, 8);
            _mm512_mask_compressstoreu_epi8(r_l1 + r_off, lit_mask, vr);
            r_off += __builtin_popcountll(mask_val);
        } else {
            _mm512_storeu_si512(r_l1 + r_off, vr);
            r_off += 64;
        }
    }

    // scalar tail
    if (avx_regions * words_per_reg < total_words) {
        const uint8_t l1_fill_a = A.l1_fill_ones ? 0xFF : 0x00;
        const uint8_t l1_fill_b = B.l1_fill_ones ? 0xFF : 0x00;
        const uint8_t l2_fill_a = A.l2_fill_ones ? 0xFF : 0x00;
        const uint8_t l2_fill_b = B.l2_fill_ones ? 0xFF : 0x00;
        bool a_l4_lit = (A.l4_bits[avx_regions / 8] >> (avx_regions % 8)) & 1;
        bool b_l4_lit = (B.l4_bits[avx_regions / 8] >> (avx_regions % 8)) & 1;
        uint8_t l3a = a_l4_lit ? A.l3_lits[A.l3_lit_off++] : a_l3_fill;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2a = ((l3a >> l2i) & 1) ? A.l2_lits[A.l2_lit_off++] : l2_fill_a;
            uint8_t l2b = ((l3b >> l2i) & 1) ? B.l2_lits[B.l2_lit_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wa = ((l2a >> bit) & 1) ? A.l1_lits[A.l1_lit_off++] : l1_fill_a;
                uint8_t wb = ((l2b >> bit) & 1) ? B.l1_lits[B.l1_lit_off++] : l1_fill_b;
                uint8_t vr = wa ^ wb;
                if (compress) {
                    if (vr != 0x00) {
                        result.l2_flat_[pos / 8] |= uint8_t(1) << (pos % 8);
                        r_l1[r_off++] = vr;
                    }
                } else {
                    r_l1[r_off++] = vr;
                }
            }
        }
    }

#ifdef DDC_DEBUG
    auto t1 = clock::now();
#endif

#else

#ifdef DDC_DEBUG
    auto t1 = clock::now();
#endif

    // scalar fallback
    {
        size_t a_l1_off = 0, b_l1_off = 0;
        auto l2_a = expand_l2();
        auto l2_b = other.expand_l2();

        alignas(64) uint8_t buf_a[64], buf_b[64];
        size_t pos = 0;
        while (pos < total_words) {
            size_t chunk = std::min(size_t(64), total_words - pos);
            std::memset(buf_a, l1_fill_ones_ ? 0xFF : 0x00, 64);
            std::memset(buf_b, other.l1_fill_ones_ ? 0xFF : 0x00, 64);

            for (size_t i = 0; i < chunk; i++) {
                size_t wi = pos + i;
                if ((l2_a[wi / 8] >> (wi % 8)) & 1)
                    buf_a[i] = a_l1[a_l1_off++];
            }
            for (size_t i = 0; i < chunk; i++) {
                size_t wi = pos + i;
                if ((l2_b[wi / 8] >> (wi % 8)) & 1)
                    buf_b[i] = b_l1[b_l1_off++];
            }

            for (size_t i = 0; i < chunk; i++)
                buf_a[i] ^= buf_b[i];
            if (compress) {
                for (size_t i = 0; i < chunk; i++) {
                    if (buf_a[i] != 0x00) {
                        size_t wi = pos + i;
                        result.l2_flat_[wi / 8] |= uint8_t(1) << (wi % 8);
                        r_l1[r_off++] = buf_a[i];
                    }
                }
            } else {
                std::memcpy(r_l1 + r_off, buf_a, chunk);
                r_off += chunk;
            }
            pos += chunk;
        }
    }

#endif

#ifdef DDC_DEBUG
    auto t2 = clock::now();
    auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::cout << "  [XOR] "
              << "expand_xor: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    if (compress) result.compact_l2_l3(r_off);
    return result;
}

// per-segment XOR
DDC
DDC::operator^(const DDC& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    DDC result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& sa = segments_[i];
        const auto& sb = other.segments_[i];

        // fill shortcuts
        if (sa.is_all_zero()) { result.segments_.push_back(sb); continue; }
        if (sb.is_all_zero()) { result.segments_.push_back(sa); continue; }
        if (sa.is_all_ones()) { result.segments_.push_back(~sb); continue; }
        if (sb.is_all_ones()) { result.segments_.push_back(~sa); continue; }

        result.segments_.push_back(sa ^ sb);
    }

    return result;
}

DDCBtv&
DDCBtv::operator^=(const DDCBtv& other) {
    *this = *this ^ other;
    return *this;
}

DDC&
DDC::operator^=(const DDC& other) {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& seg = other.segments_[i];
        if (seg.is_all_zero()) continue;
        if (segments_[i].is_all_zero()) {
            segments_[i] = seg;
            continue;
        }
        if (seg.is_all_ones()) {
            segments_[i] = ~segments_[i];
            continue;
        }
        if (segments_[i].is_all_ones()) {
            segments_[i] = ~seg;
            continue;
        }
        segments_[i] = segments_[i] ^ seg;
    }
    return *this;
}
