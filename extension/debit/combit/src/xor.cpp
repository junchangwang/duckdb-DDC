#include "combit.h"

// ----------------------------------------------------------------
// Bitwise XOR operations (3-level: L3/L2/L1)
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// ComBitBtv XOR operator
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator^(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    if (bit_count_ == 0) return ComBitBtv();
    assert(state_ != State::Uncompressed);
    assert(other.state_ != State::Uncompressed);

    const size_t total_words = l2_count_;

    // Result: all words become literals (we compute every word).
    const bool compress = combit_compress_results;
    ComBitBtv result = compress ? ComBitBtv(false, false, State::Compressed)
                                : ComBitBtv(false, true, State::Decompressed);
    result.bit_count_ = bit_count_;
    result.l2_count_ = total_words;
    size_t l2_byte_count = (total_words + 7) / 8;

    if (compress) {
        result.l2_flat_.assign(l2_byte_count, 0x00);
    } else {
        // Fully expanded: all L2 bytes are 0xFF (all words literal).
        // l3_bits_ canonically all-zero in Decompressed; skip alloc.
        result.l3_count_ = l2_byte_count;
        result.l2_literal_count_ = 0;
    }

    result.l1_literals_.resize(total_words);
    result.l1_literal_count_ = total_words;

    const uint8_t* a_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    uint8_t* r_l1 = result.l1_literals_.data();

    size_t r_off = 0;

#ifdef COMBIT_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;

    // Per-side state via ComBitBtv::SideCtx (defined in combit.h, shared
    // with and.cpp / or.cpp).  Factory uses `this->` / `other.` style.
    SideCtx A = this->make_side(a_l1);
    SideCtx B = other.make_side(b_l1);

    static constexpr size_t PF_DIST = 128;

    uint8_t* result_l2 = result.l2_flat_.data();

    // --- Bypass: XOR zero-region skip ---
    // XOR: 0 ^ 0 = 0, so both-zero regions can be skipped.
    // Per-side: if that side's fills are zero, L3=0 means all-zero.
    const bool a_zero_when_l3_zero = !A.l1_fill_ones && !A.l2_fill_ones;
    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;

    const uint8_t a_l3_fill = A.l3_fill_ones ? 0xFF : 0x00;
    const uint8_t b_l3_fill = B.l3_fill_ones ? 0xFF : 0x00;

    for (size_t region = 0; region < avx_regions; region++) {
        bool a_l4_lit = (A.l4_bits[region / 8] >> (region % 8)) & 1;
        bool b_l4_lit = (B.l4_bits[region / 8] >> (region % 8)) & 1;
        uint8_t l3a = a_l4_lit ? A.l3_lits[A.l3_lit_off++] : a_l3_fill;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;

        // --- Per-side bypass: x ^ 0 = x ---
        if (a_zero_when_l3_zero && l3a == 0) {
            if (b_zero_when_l3_zero && l3b == 0) {
                // Both zero => 0 ^ 0 = 0
                if (!compress) {
                    _mm512_storeu_si512(r_l1 + r_off, _mm512_setzero_si512());
                    r_off += 64;
                }
                continue;
            }
            // a is all-zero => result = b (expand b only)
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
        if (b_zero_when_l3_zero && l3b == 0) {
            // b is all-zero => result = a (expand a only)
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

        __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec, ma, A.l1_lits + A.l1_lit_off);
        A.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));

        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb, B.l1_lits + B.l1_lit_off);
        B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i vr = _mm512_xor_si512(va, vb);
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

    // === Scalar tail: process remaining words without expand_l2() ===
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

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

#else  // !__AVX512VBMI2__

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

    // === Scalar fallback (no AVX-512) ===
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

#endif  // __AVX512VBMI2__

#ifdef COMBIT_DEBUG
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

// ----------------------------------------------------------------
// ComBit (segmented) XOR operator
// ----------------------------------------------------------------

ComBit
ComBit::operator^(const ComBit& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    ComBit result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& sa = segments_[i];
        const auto& sb = other.segments_[i];

        // Segment-level bypass (mirrors operator| / operator&):
        //   0 ^ b = b,  a ^ 0 = a,  1 ^ b = ~b,  a ^ 1 = ~a.
        // Catches 0^0=0 (via sa zero-bypass), 1^1=0 (via sa ones → ~sb=~all1=all0).
        if (sa.is_all_zero()) { result.segments_.push_back(sb); continue; }
        if (sb.is_all_zero()) { result.segments_.push_back(sa); continue; }
        if (sa.is_all_ones()) { result.segments_.push_back(~sb); continue; }
        if (sb.is_all_ones()) { result.segments_.push_back(~sa); continue; }

        result.segments_.push_back(sa ^ sb);
    }

    return result;
}

// ----------------------------------------------------------------
// ComBitBtv in-place XOR (operator^=)
// ----------------------------------------------------------------
//
// XOR has no in-place AVX-512 fast path that meaningfully beats the
// binary operator^ followed by move-assignment: every output bit
// depends on both inputs, so unlike |= (which can short-circuit on
// fill_ones RHS) and &= (fill_zero RHS), there's no per-byte case
// where the LHS byte stays unchanged.  Delegate to operator^ to
// reuse its AVX-512 core; the rvalue is move-assigned with no copy.
// ----------------------------------------------------------------

ComBitBtv&
ComBitBtv::operator^=(const ComBitBtv& other) {
    *this = *this ^ other;
    return *this;
}

// ----------------------------------------------------------------
// ComBit (segmented) in-place XOR
// ----------------------------------------------------------------

ComBit&
ComBit::operator^=(const ComBit& other) {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& seg = other.segments_[i];
        if (seg.is_all_zero()) continue;                     // a ^ 0 = a
        if (segments_[i].is_all_zero()) {                    // 0 ^ b = b
            segments_[i] = seg;
            continue;
        }
        if (seg.is_all_ones()) {                             // a ^ 1 = ~a
            segments_[i] = ~segments_[i];
            continue;
        }
        if (segments_[i].is_all_ones()) {                    // 1 ^ b = ~b
            segments_[i] = ~seg;
            continue;
        }
        segments_[i] = segments_[i] ^ seg;
    }
    return *this;
}
