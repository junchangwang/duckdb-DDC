#include "combit.h"

// ----------------------------------------------------------------
// Bitwise OR operations (3-level: L3/L2/L1)
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// ComBitBtv OR operator
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator|(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Compressed);
    assert(other.state_ == State::Compressed);

    if (bit_count_ == 0) return ComBitBtv();

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
        // Fully expanded: all L2 bytes are 0xFF (all words literal)
        result.l3_count_ = l2_byte_count;
        size_t l3_byte_count = (l2_byte_count + 7) / 8;
        result.l3_bits_.assign(l3_byte_count, 0);
        result.l2_literal_count_ = 0;
    }

    result.l1_literals_.resize(total_words);
    result.l1_literal_count_ = total_words;

    const uint8_t* a_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    uint8_t* r_l1 = result.l1_literals_.data();

    size_t a_l1_off = 0, b_l1_off = 0;
    size_t r_off = 0;

#ifdef COMBIT_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;
    size_t a_l2_off = 0, b_l2_off = 0;

    const __m512i fill_a_vec = l1_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    const __m512i fill_b_vec = other.l1_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();

    // L2 fill vectors for L3 expand (l2_fill_ones_ => 0xFF, else 0x00)
    const __m512i l2_fill_a_vec = l2_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    const __m512i l2_fill_b_vec = other.l2_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();

    static constexpr size_t PF_DIST = 128;

    uint8_t* result_l2 = result.l2_flat_.data();

    // --- Bypass: OR zero-region skip ---
    // OR: 0 | 0 = 0, so if BOTH regions are all-zero, result is zero.
    // Per-side: if that side's fills are zero, L3=0 means all-zero.
    const bool a_zero_fill = !l1_fill_ones_ && !l2_fill_ones_;
    const bool b_zero_fill = !other.l1_fill_ones_ && !other.l2_fill_ones_;

    for (size_t region = 0; region < avx_regions; region++) {
        uint8_t l3a = l3_bits_[region];
        uint8_t l3b = other.l3_bits_[region];

        // --- Per-side bypass: x | 0 = x ---
        if (a_zero_fill && l3a == 0) {
            if (b_zero_fill && l3b == 0) {
                // Both zero => 0 | 0 = 0
                if (!compress) {
                    _mm512_storeu_si512(r_l1 + r_off, _mm512_setzero_si512());
                    r_off += 64;
                }
                continue;
            }
            // a is all-zero => result = b (expand b only)
            __m512i l2b_v = _mm512_mask_expandloadu_epi8(l2_fill_b_vec,
                static_cast<__mmask64>(l3b), other.l2_literals_.data() + b_l2_off);
            b_l2_off += __builtin_popcount(l3b);
            __mmask64 mb = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));
            __m512i vb = _mm512_mask_expandloadu_epi8(fill_b_vec, mb,
                b_l1 + b_l1_off);
            b_l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));
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
        if (b_zero_fill && l3b == 0) {
            // b is all-zero => result = a (expand a only)
            __m512i l2a_v = _mm512_mask_expandloadu_epi8(l2_fill_a_vec,
                static_cast<__mmask64>(l3a), l2_literals_.data() + a_l2_off);
            a_l2_off += __builtin_popcount(l3a);
            __mmask64 ma = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));
            __m512i va = _mm512_mask_expandloadu_epi8(fill_a_vec, ma,
                a_l1 + a_l1_off);
            a_l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
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

        _mm_prefetch(reinterpret_cast<const char*>(a_l1 + a_l1_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b_l1 + b_l1_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(r_l1 + r_off + PF_DIST), _MM_HINT_T0);

        __m512i l2a_v = _mm512_mask_expandloadu_epi8(l2_fill_a_vec,
            static_cast<__mmask64>(l3a), l2_literals_.data() + a_l2_off);
        a_l2_off += __builtin_popcount(l3a);
        __mmask64 ma = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));

        __m512i l2b_v = _mm512_mask_expandloadu_epi8(l2_fill_b_vec,
            static_cast<__mmask64>(l3b), other.l2_literals_.data() + b_l2_off);
        b_l2_off += __builtin_popcount(l3b);
        __mmask64 mb = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

        __m512i va = _mm512_mask_expandloadu_epi8(fill_a_vec, ma, a_l1 + a_l1_off);
        a_l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));

        __m512i vb = _mm512_mask_expandloadu_epi8(fill_b_vec, mb, b_l1 + b_l1_off);
        b_l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i vr = _mm512_or_si512(va, vb);
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
        const uint8_t l1_fill_a = l1_fill_ones_ ? 0xFF : 0x00;
        const uint8_t l1_fill_b = other.l1_fill_ones_ ? 0xFF : 0x00;
        const uint8_t l2_fill_a = l2_fill_ones_ ? 0xFF : 0x00;
        const uint8_t l2_fill_b = other.l2_fill_ones_ ? 0xFF : 0x00;
        uint8_t l3a = l3_bits_[avx_regions];
        uint8_t l3b = other.l3_bits_[avx_regions];
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2a = ((l3a >> l2i) & 1) ? l2_literals_[a_l2_off++] : l2_fill_a;
            uint8_t l2b = ((l3b >> l2i) & 1) ? other.l2_literals_[b_l2_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wa = ((l2a >> bit) & 1) ? a_l1[a_l1_off++] : l1_fill_a;
                uint8_t wb = ((l2b >> bit) & 1) ? b_l1[b_l1_off++] : l1_fill_b;
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

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

#else  // !__AVX512VBMI2__

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

    // === Scalar fallback (no AVX-512) ===
    {
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

#endif  // __AVX512VBMI2__

#ifdef COMBIT_DEBUG
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

// ----------------------------------------------------------------
// ComBit (segmented) OR operator
// ----------------------------------------------------------------

ComBit
ComBit::operator|(const ComBit& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    ComBit result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (size_t i = 0; i < segments_.size(); i++)
        result.segments_.push_back(segments_[i] | other.segments_[i]);

    return result;
}

// ----------------------------------------------------------------
// ComBitBtv in-place OR (operator|=)
// ----------------------------------------------------------------

ComBitBtv&
ComBitBtv::operator|=(const ComBitBtv& other) {
    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Decompressed);
    assert(other.state_ == State::Compressed);

    if (bit_count_ == 0) return *this;

    const size_t total_words = l2_count_;
    uint8_t* r_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    size_t b_l1_off = 0;

#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;
    const __m512i fill_b_vec = other.l1_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    const __m512i l2_fill_b_vec = other.l2_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    static constexpr size_t PF_DIST = 128;

    size_t b_l2_off = 0;

    // Per-region bypass for OR:
    // b all-zero => a | 0 = a (skip entirely)
    const bool b_zero_fill = !other.l1_fill_ones_ && !other.l2_fill_ones_;

    for (size_t region = 0; region < avx_regions; region++) {
        uint8_t l3b = other.l3_bits_[region];

        if (b_zero_fill && l3b == 0) {
            // all-zeros => a | 0 = a (no-op)
            continue;
        }

        _mm_prefetch(reinterpret_cast<const char*>(
            b_l1 + b_l1_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(
            r_l1 + region * 64 + PF_DIST), _MM_HINT_T0);

        __m512i l2b_v = _mm512_mask_expandloadu_epi8(l2_fill_b_vec,
            static_cast<__mmask64>(l3b),
            other.l2_literals_.data() + b_l2_off);
        b_l2_off += __builtin_popcount(l3b);
        __mmask64 mb = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

        __m512i vb = _mm512_mask_expandloadu_epi8(fill_b_vec, mb,
            b_l1 + b_l1_off);
        b_l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i va = _mm512_loadu_si512(r_l1 + region * 64);
        _mm512_storeu_si512(r_l1 + region * 64,
            _mm512_or_si512(va, vb));
    }

    if (avx_regions * words_per_reg < total_words) {
        const uint8_t l1_fill_b = other.l1_fill_ones_ ? 0xFF : 0x00;
        const uint8_t l2_fill_b = other.l2_fill_ones_ ? 0xFF : 0x00;
        uint8_t l3b = other.l3_bits_[avx_regions];
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2b = ((l3b >> l2i) & 1) ? other.l2_literals_[b_l2_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wb = ((l2b >> bit) & 1) ? b_l1[b_l1_off++] : l1_fill_b;
                r_l1[pos] |= wb;
            }
        }
    }
#else
    {
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

// ----------------------------------------------------------------
// ComBit (segmented) in-place OR
// ----------------------------------------------------------------

ComBit&
ComBit::operator|=(const ComBit& other) {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    const size_t num_segs = segments_.size();

    for (size_t i = 0; i < num_segs; i++) {
        const auto& seg = other.segments_[i];
        bool b_all_zero = !seg.l1_fill_ones() && !seg.l2_fill_ones()
                          && seg.num_literals() == 0;
        bool b_all_ones = seg.l1_fill_ones() && !seg.l2_fill_ones()
                          && seg.num_literals() == 0;

        if (b_all_zero) {
            // other is all-zero => a | 0 = a (no-op)
            continue;
        }
        if (b_all_ones) {
            // other is all-ones => a | 1 = 1 => set this segment to all-ones
            segments_[i] = ComBitBtv(true);
            segments_[i].bit_count_ = seg.bit_count_;
            segments_[i].l2_count_ = seg.l2_count_;
            size_t l2_byte_count = (seg.l2_count_ + 7) / 8;
            segments_[i].l3_count_ = l2_byte_count;
            segments_[i].l3_bits_.assign((l2_byte_count + 7) / 8, 0x00);
            continue;
        }

        // Prefetch next segment's heap data (l3_bits_, l1_literals_).
        if (i + 2 < num_segs) {
            __builtin_prefetch(
                other.segments_[i + 2].l1_literal_data(), 0, 3);
            __builtin_prefetch(
                other.segments_[i + 2].l3_data(), 0, 3);
        }

        segments_[i] |= other.segments_[i];
    }

    return *this;
}

// ================================================================
// ComBit::OR_many  –  multi-way OR using operator| and operator|=
// ================================================================

ComBit
ComBit::OR_many(size_t number, const ComBit** Btvs) {
    assert(number > 0);

    if (number == 1) return *Btvs[0];

    const size_t num_segs = Btvs[0]->num_segments();

    // --- Density-based path selection -------------------------------
    //
    // Two OR_many paths are available:
    //
    //   (A) operator|= chain: first Btvs[0] | Btvs[1] then |= the rest
    //       on top.  Every segment pays the full 128-region AVX-512
    //       expand_load sweep inside ComBitBtv::operator|=, whose cost
    //       is proportional to the segment size regardless of density.
    //       Fast when density is non-trivial.
    //
    //   (B) scatter-OR: allocate a Decompressed zero result, then for
    //       each input bitmap walk only its non-zero L1 bytes via
    //       for_each_literal (AVX-512 L3 bitscan skips all-zero
    //       regions in bulk) and OR each byte into the result's flat
    //       per-segment buffer.  Avoids the expand_load sweep entirely
    //       but pays a per-callback cost for every non-zero byte.
    //       Fast when density is very low.
    //
    // Measured crossover on synthetic TPC-H-scale inputs
    // (N=100M, count=32):
    //
    //   density   non-zero-byte-ratio   (A)ms   (B)ms   winner
    //   0.001          0.021             34.9    26.7   scatter
    //   0.010          0.077             17.4   198.0   expand
    //   0.100          0.570             32.6  2117.0   expand
    //
    // We use a 5% non-zero-byte-ratio threshold: low enough that Q6
    // shipdate (~2.1%) clearly takes the scatter path; high enough
    // that moderate-density workloads (e.g. Q6 quantity at ~57%) stay
    // on the expand path.  The density scan itself is an O(count *
    // num_segs) sweep of inline num_literals() / l2_count() accessors
    // — microseconds for TPC-H scales, no heap traversal.
    size_t total_nz = 0;
    size_t total_slots = 0;
    for (size_t i = 0; i < number; i++) {
        const size_t nsegs = Btvs[i]->num_segments();
        for (size_t s = 0; s < nsegs; s++) {
            const ComBitBtv& seg = Btvs[i]->segment(s);
            total_nz += seg.num_literals();
            total_slots += seg.l2_count();
        }
    }
    const bool use_scatter = (total_nz * 20 < total_slots);

    ComBit result;
    result.bit_count_ = Btvs[0]->bit_count_;
    result.segment_bits_ = Btvs[0]->segment_bits_;
    result.segments_.reserve(num_segs);

    if (use_scatter) {
        // --- Path B: sparse scatter-OR ------------------------------
        //
        // Build a Decompressed zero result.  Every segment's logical
        // L2 is "all literal" (l2_fill_ones=true, l3_bits_ all zero,
        // no l2_literals_ stored) — the canonical Decompressed form
        // produced by any ComBitBtv::operator|.  Downstream
        // ComBitBtv::operator&= asserts state=Decompressed, so we
        // must leave it in this state.
        for (size_t s = 0; s < num_segs; s++) {
            const ComBitBtv& src = Btvs[0]->segment(s);
            ComBitBtv seg(/*l1_fill_ones=*/false,
                          /*l2_fill_ones=*/true,
                          ComBitBtv::State::Decompressed);
            seg.bit_count_ = src.bit_count_;
            seg.l2_count_ = src.l2_count_;
            const size_t l2_byte_count = (src.l2_count_ + 7) / 8;
            seg.l3_count_ = l2_byte_count;
            seg.l3_bits_.assign((l2_byte_count + 7) / 8, 0x00);
            seg.l2_literal_count_ = 0;
            seg.l1_literals_.assign(src.l2_count_, 0x00);
            seg.l1_literal_count_ = src.l2_count_;
            result.segments_.push_back(std::move(seg));
        }

        // for_each_literal delivers word positions in monotonically
        // increasing global order (segments in order, L3 bitscan
        // low→high).  Track (cur_seg, seg_start, seg_end) incrementally
        // so each callback costs only a bounds check and a subtraction
        // — no div/mod per non-zero byte.
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
        // --- Path A: operator|= chain with bypass + prefetch -------
        //
        // Mirror the bypass + prefetch pattern already used by
        // ComBit::operator|= so the two multi-bitmap OR paths stay in
        // lock-step:
        //   * all-zero segment bypass: a | 0 = a  (skip the |= call).
        //   * cross-bitmap prefetch for Btvs[i+2]'s heap data — the
        //     hardware prefetcher cannot follow the heap hops between
        //     different ComBit objects.
        for (size_t s = 0; s < num_segs; s++) {
            result.segments_.push_back(
                Btvs[0]->segment(s) | Btvs[1]->segment(s));
            for (size_t i = 2; i < number; i++) {
                const auto& seg = Btvs[i]->segment(s);

                if (!seg.l1_fill_ones()
                    && !seg.l2_fill_ones()
                    && seg.num_literals() == 0) {
                    continue;
                }

                if (i + 2 < number) {
                    const auto& next = Btvs[i + 2]->segment(s);
                    __builtin_prefetch(next.l1_literal_data(), 0, 3);
                    __builtin_prefetch(next.l3_data(), 0, 3);
                }

                result.segments_[s] |= seg;
            }
        }
    }

    return result;
}
