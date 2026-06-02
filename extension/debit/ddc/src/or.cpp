#include "ddc.h"

// OR kernel
DDCBtv
DDCBtv::operator|(const DDCBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Compressed);
    assert(other.state_ == State::Compressed);

    if (bit_count_ == 0) return DDCBtv();

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
    const size_t avx_regions = total_words / words_per_reg;

    SideCtx A = this->make_side(a_l1);
    SideCtx B = other.make_side(b_l1);

    static constexpr size_t PF_DIST = 128;

    uint8_t* result_l2 = result.l2_flat_.data();

    const bool a_zero_when_l3_zero = !A.l1_fill_ones && !A.l2_fill_ones;
    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;

    const bool a_struct_zero = a_zero_when_l3_zero && !A.l3_fill_ones;
    const bool b_struct_zero = b_zero_when_l3_zero && !B.l3_fill_ones;

    // SIMD batch loop
    const size_t batch_count = (avx_regions + 63) / 64;
    for (size_t batch = 0; batch < batch_count; batch++) {
        const size_t batch_start = batch * 64;
        const size_t batch_end   = std::min(batch_start + 64, avx_regions);
        const size_t batch_size  = batch_end - batch_start;

        uint64_t a_l4_mask = 0, b_l4_mask = 0;
        std::memcpy(&a_l4_mask, A.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        std::memcpy(&b_l4_mask, B.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
            b_l4_mask &= valid;
        }

        // bypass empty batch
        if (a_struct_zero && b_struct_zero
            && a_l4_mask == 0 && b_l4_mask == 0) {
            if (!compress) {
                std::memset(r_l1 + r_off, 0, batch_size * 64);
                r_off += batch_size * 64;
            }

            continue;
        }

        // expand L3
        __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(A.l3_fill_vec,
            static_cast<__mmask64>(a_l4_mask), A.l3_lits + A.l3_lit_off);
        __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(B.l3_fill_vec,
            static_cast<__mmask64>(b_l4_mask), B.l3_lits + B.l3_lit_off);
        A.l3_lit_off += __builtin_popcountll(a_l4_mask);
        B.l3_lit_off += __builtin_popcountll(b_l4_mask);

        alignas(64) uint8_t l3a_buf[64], l3b_buf[64];
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);

        // per-region OR
        for (size_t r = 0; r < batch_size; r++) {
            const size_t region = batch_start + r;
            const uint8_t l3a = l3a_buf[r];
            const uint8_t l3b = l3b_buf[r];

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

            __m512i vr = _mm512_or_si512(va, vb);
            if (compress) {
                // compress-store result
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
    }

    // scalar tail
    if (avx_regions * words_per_reg < total_words) {
        const uint8_t a_l3_fill = A.l3_fill_ones ? 0xFF : 0x00;
        const uint8_t b_l3_fill = B.l3_fill_ones ? 0xFF : 0x00;
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
                uint8_t vr = wa | wb;
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
                buf_a[i] |= buf_b[i];
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
    std::cout << "  [OR] "
              << "expand_or: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    if (compress) result.compact_l2_l3(r_off);
    return result;
}

// per-segment OR
DDC
DDC::operator|(const DDC& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    DDC result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& sa = segments_[i];
        const auto& sb = other.segments_[i];

        // trivial-fill shortcuts
        if (sb.is_all_zero()) { result.segments_.push_back(sa); continue; }
        if (sa.is_all_zero()) { result.segments_.push_back(sb); continue; }
        if (sa.is_all_ones()) { result.segments_.push_back(sa); continue; }
        if (sb.is_all_ones()) { result.segments_.push_back(sb); continue; }

        if (sa.state() != DDCBtv::State::Compressed) {
            DDCBtv tmp = sa;
            tmp |= sb;
            result.segments_.push_back(std::move(tmp));
            continue;
        }
        if (sb.state() != DDCBtv::State::Compressed) {

            DDCBtv tmp = sb;
            tmp |= sa;
            result.segments_.push_back(std::move(tmp));
            continue;
        }

        result.segments_.push_back(sa | sb);
    }

    return result;
}

// in-place OR
DDCBtv&
DDCBtv::operator|=(const DDCBtv& other) {
    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Decompressed);
    assert(other.state_ != State::Uncompressed);

    if (bit_count_ == 0) return *this;

    const size_t total_words = l2_count_;
    uint8_t* r_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();

#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;

    // dense fast path
    if (other.state_ == State::Decompressed) {
        for (size_t region = 0; region < avx_regions; region++) {
            __m512i va = _mm512_loadu_si512(r_l1 + region * 64);
            __m512i vb = _mm512_loadu_si512(b_l1 + region * 64);
            _mm512_storeu_si512(r_l1 + region * 64,
                _mm512_or_si512(va, vb));
        }
        for (size_t pos = avx_regions * words_per_reg; pos < total_words; pos++)
            r_l1[pos] |= b_l1[pos];
        return *this;
    }

    SideCtx B = other.make_side(b_l1);
    static constexpr size_t PF_DIST = 128;

    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;

    const bool b_l4_skipable = b_zero_when_l3_zero && !B.l3_fill_ones;
    const uint8_t b_l3_fill = B.l3_fill_ones ? 0xFF : 0x00;

    // sparse: skip zero runs
    for (size_t region = 0; region < avx_regions; region++) {
        if (b_l4_skipable && (region & 63) == 0
            && region + 64 <= avx_regions) {
            uint64_t l4_chunk;
            std::memcpy(&l4_chunk, B.l4_bits + region / 8, 8);
            if (l4_chunk == 0) {
                region += 63;
                continue;
            }
        }

        bool b_l4_lit = (B.l4_bits[region / 8] >> (region % 8)) & 1;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;

        if (b_zero_when_l3_zero && l3b == 0) continue;

        _mm_prefetch(reinterpret_cast<const char*>(
            B.l1_lits + B.l1_lit_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(
            r_l1 + region * 64 + PF_DIST), _MM_HINT_T0);

        __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
            static_cast<__mmask64>(l3b),
            B.l2_lits + B.l2_lit_off);
        B.l2_lit_off += __builtin_popcount(l3b);
        __mmask64 mb = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb,
            B.l1_lits + B.l1_lit_off);
        B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i va = _mm512_loadu_si512(r_l1 + region * 64);
        _mm512_storeu_si512(r_l1 + region * 64,
            _mm512_or_si512(va, vb));
    }

    if (avx_regions * words_per_reg < total_words) {
        const uint8_t l1_fill_b = B.l1_fill_ones ? 0xFF : 0x00;
        const uint8_t l2_fill_b = B.l2_fill_ones ? 0xFF : 0x00;
        bool b_l4_lit = (B.l4_bits[avx_regions / 8] >> (avx_regions % 8)) & 1;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2b = ((l3b >> l2i) & 1) ? B.l2_lits[B.l2_lit_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wb = ((l2b >> bit) & 1) ? B.l1_lits[B.l1_lit_off++] : l1_fill_b;
                r_l1[pos] |= wb;
            }
        }
    }
#else
    {
        size_t b_l1_off = 0;
        auto l2_b = other.expand_l2();
        for (size_t w = 0; w < total_words; w++) {
            uint8_t wb = other.l1_fill_ones_ ? 0xFF : 0x00;
            if ((l2_b[w / 8] >> (w % 8)) & 1)
                wb = b_l1[b_l1_off++];
            r_l1[w] |= wb;
        }
    }
#endif
    return *this;
}

// in-place per-segment OR
DDC&
DDC::operator|=(const DDC& other) {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    const size_t num_segs = segments_.size();

    for (size_t i = 0; i < num_segs; i++) {
        const auto& seg = other.segments_[i];
        if (seg.is_all_zero()) continue;
        if (seg.is_all_ones()) {
            segments_[i] = DDCBtv::make_all_fill(
                seg.bit_count(), seg.l2_count(), true);
            continue;
        }

        if (segments_[i].is_all_ones()) continue;
        if (segments_[i].is_all_zero()) {
            segments_[i] = seg;
            continue;
        }
        if (segments_[i].state() != DDCBtv::State::Decompressed) {
            segments_[i] = segments_[i] | seg;
            continue;
        }

        if (i + 2 < num_segs) {
            __builtin_prefetch(
                other.segments_[i + 2].l1_literal_data(), 0, 3);
            __builtin_prefetch(
                other.segments_[i + 2].l4_data(), 0, 3);
        }

        segments_[i] |= other.segments_[i];
    }

    return *this;
}

// fast multi-way union
DDC
DDC::OR_many(size_t number, const DDC** Btvs) {
    assert(number > 0);

    if (number == 1) return *Btvs[0];

    const size_t num_segs = Btvs[0]->num_segments();

    size_t total_nz = 0;
    size_t total_slots = 0;
    for (size_t i = 0; i < number; i++) {
        const size_t nsegs = Btvs[i]->num_segments();
        for (size_t s = 0; s < nsegs; s++) {
            const DDCBtv& seg = Btvs[i]->segment(s);
            total_nz += seg.num_literals();
            total_slots += seg.l2_count();
        }
    }
    // pick scatter vs pairwise
    const bool use_scatter = (total_nz * 20 < total_slots);

    DDC result;
    result.bit_count_ = Btvs[0]->bit_count_;
    result.segment_bits_ = Btvs[0]->segment_bits_;
    result.segments_.reserve(num_segs);

    if (use_scatter) {

        for (size_t s = 0; s < num_segs; s++) {
            const DDCBtv& src = Btvs[0]->segment(s);
            DDCBtv seg( false,
                           true,
                          DDCBtv::State::Decompressed);
            seg.bit_count_ = src.bit_count_;
            seg.l2_count_ = src.l2_count_;
            const size_t l2_byte_count = (src.l2_count_ + 7) / 8;
            seg.l3_count_ = l2_byte_count;

            seg.l2_literal_count_ = 0;
            seg.l1_literals_.assign(src.l2_count_, 0x00);
            seg.l1_literal_count_ = src.l2_count_;
            result.segments_.push_back(std::move(seg));
        }

        // scatter literals into dst
        for (size_t i = 0; i < number; i++) {
            size_t cur_seg   = 0;
            size_t seg_start = 0;
            size_t seg_end   = result.segments_[0].l2_count_;
            uint8_t* dst     = result.segments_[0].l1_literals_.data();

            Btvs[i]->for_each_literal(
                [&cur_seg, &seg_start, &seg_end, &dst, &result](uint32_t word_pos, uint8_t val) {
                    while (word_pos >= seg_end) {
                        cur_seg++;
                        seg_start = seg_end;
                        seg_end  += result.segments_[cur_seg].l2_count_;
                        dst       = result.segments_[cur_seg]
                                          .l1_literals_.data();
                    }
                    dst[word_pos - seg_start] |= val;
                });
        }
    } else {
        // pairwise accumulate
        for (size_t s = 0; s < num_segs; s++) {
            result.segments_.push_back(
                Btvs[0]->segment(s) | Btvs[1]->segment(s));
            for (size_t i = 2; i < number; i++) {
                const auto& seg = Btvs[i]->segment(s);

                if (seg.is_all_zero()) continue;

                if (i + 2 < number) {
                    const auto& next = Btvs[i + 2]->segment(s);
                    __builtin_prefetch(next.l1_literal_data(), 0, 3);
                    __builtin_prefetch(next.l4_data(), 0, 3);
                }

                result.segments_[s] |= seg;
            }
        }
    }

    return result;
}
