#include "combit.h"

// ----------------------------------------------------------------
// Bitwise AND operations (4-level: L4/L3/L2/L1)
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// ComBitBtv AND operator
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator&(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Compressed); // FIXME: only support compressed versions so far.
    assert(other.state_ == State::Compressed);

    if (bit_count_ == 0) return ComBitBtv();

    // Result: all words become literals (we compute every word).
    const size_t total_words = l2_count_;
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
    // -----------------------------------------------------------------
    // 4-level traversal (L4 -> L3 -> L2 -> L1) per side, then AND.
    //
    //   1 word    = 8 bits   (one L1 byte)
    //   1 L2 bit  gates  1 L1 byte           (8 words)
    //   1 L3 bit  gates  1 L2 byte           (8 L2 bits = 64 words = 1 region)
    //   1 L4 bit  gates  1 L3 byte           (8 L3 bits = 512 words)
    //   1 batch   = 8 L4 bytes               (64 regions = 4096 words)
    //
    // Each batch SIMD-expands 64 L3 bytes per side from 8 L4 bytes.
    // Per region we cascade L3 -> L2 -> L1 expand, then AND the two
    // 64-byte vectors.
    // -----------------------------------------------------------------
    const size_t avx_regions = total_words / words_per_reg;
    const size_t batch_count = (avx_regions + 63) / 64;

    // Per-side state via ComBitBtv::SideCtx (defined in combit.h, shared
    // with or.cpp / xor.cpp).  Factory call uses `this->` / `other.` style
    // so the symmetry of the two paths is visible inline; A/B identifiers
    // are kept as readable aliases.
    SideCtx A = this->make_side(a_l1);
    SideCtx B = other.make_side(b_l1);

    // Prefetch distance: 2 regions ahead (128 bytes).  L1 offsets are
    // data-dependent (via popcount), so we approximate with +PF_DIST.
    // This hides memory latency for the sequential L1 literal stream.
    static constexpr size_t PF_DIST = 256;

    uint8_t* result_l2 = result.l2_flat_.data();

    // -----------------------------------------------------------------
    // Bypass classification (AND-specific, "zero wins"):
    //
    //   A side is region-zero iff (l1_fill=0 AND l2_fill=0 AND l3_fill=0):
    //   every L2 byte in the region is 0 (literal-zero or fill-zero),
    //   and every L1 byte gated by those L2 bits is fill-zero.
    //
    //   `zero_when_l3_zero[X]` is the side-wide predicate; the L3 byte
    //   then drives the per-region decision.
    //
    // Result region is all-zero iff EITHER side is region-zero, so we
    // only need to advance the OTHER side's L2/L1 cursors.
    // -----------------------------------------------------------------
    const bool a_zero_when_l3_zero = !A.l1_fill_ones && !A.l2_fill_ones;
    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;
    // Region "implicit all-1s" when l3==0: l2_fill=0 (no L1 literals) AND
    // l1_fill=1 (all 64 words = 0xFF).  AND: a_is_ones => result = B
    // (B's L2/L1 still need full expansion since they're the output).
    const bool a_ones_when_l3_zero = !A.l2_fill_ones && A.l1_fill_ones;
    const bool b_ones_when_l3_zero = !B.l2_fill_ones && B.l1_fill_ones;

    // Structurally zero: a side's L4=0 implies 512 zero words.  Used
    // for the batch-level fast skip below (mirrors operator&='s
    // b_clear_skip but symmetric on A/B).
    const bool a_struct_zero = a_zero_when_l3_zero && !A.l3_fill_ones;
    const bool b_struct_zero = b_zero_when_l3_zero && !B.l3_fill_ones;

    for (size_t batch = 0; batch < batch_count; batch++) {
        const size_t batch_start = batch * 64;
        const size_t batch_end   = std::min(batch_start + 64, avx_regions);
        const size_t batch_size  = batch_end - batch_start;

        // ---- L4 -> L3 expand (per side, 64 L3 bytes from 8 L4 bytes) ----
        uint64_t a_l4_mask = 0, b_l4_mask = 0;
        std::memcpy(&a_l4_mask, A.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        std::memcpy(&b_l4_mask, B.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            const uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
            b_l4_mask &= valid;
        }

        // ---- Batch-level fast skip (L4 bypass) ----
        // If EITHER side is structurally zero AND its L4 batch is empty,
        // all 64 regions of that side are zero -> result batch is all
        // zero.  We skip the L3->L2->L1 cascade for this batch; the
        // non-zero side (if any) still needs its L2/L1 cursors advanced
        // past the 64 regions, which is just an L3 expand + per-region
        // advance_side (no L2->L1 expand, no AND, no store).
        //
        // Zero-side cursors don't move: popcount(l4_mask)==0 (so no L3
        // literal consumed) and every l3 byte is 0 (so no L2/L1 either).
        const bool a_batch_zero = a_struct_zero && a_l4_mask == 0;
        const bool b_batch_zero = b_struct_zero && b_l4_mask == 0;
        if (a_batch_zero || b_batch_zero) {
            if (!a_batch_zero) {
                __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(A.l3_fill_vec,
                    static_cast<__mmask64>(a_l4_mask), A.l3_lits + A.l3_lit_off);
                A.l3_lit_off += __builtin_popcountll(a_l4_mask);
                alignas(64) uint8_t l3a_buf[64];
                _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
                for (size_t r = 0; r < batch_size; r++)
                    advance_side(A, l3a_buf[r]);
            }
            if (!b_batch_zero) {
                __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(B.l3_fill_vec,
                    static_cast<__mmask64>(b_l4_mask), B.l3_lits + B.l3_lit_off);
                B.l3_lit_off += __builtin_popcountll(b_l4_mask);
                alignas(64) uint8_t l3b_buf[64];
                _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);
                for (size_t r = 0; r < batch_size; r++)
                    advance_side(B, l3b_buf[r]);
            }
            if (!compress) {
                std::memset(r_l1 + r_off, 0, batch_size * 64);
                r_off += batch_size * 64;
            }
            // compress: result.l2_flat_[batch_start..+8] already 0.
            continue;
        }

        __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(A.l3_fill_vec,
            static_cast<__mmask64>(a_l4_mask), A.l3_lits + A.l3_lit_off);
        __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(B.l3_fill_vec,
            static_cast<__mmask64>(b_l4_mask), B.l3_lits + B.l3_lit_off);
        A.l3_lit_off += __builtin_popcountll(a_l4_mask);
        B.l3_lit_off += __builtin_popcountll(b_l4_mask);

        alignas(64) uint8_t l3a_buf[64], l3b_buf[64];
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);

        // ---- Per-region loop (64 words per region) ----
        for (size_t r = 0; r < batch_size; r++) {
            const size_t  region = batch_start + r;
            const uint8_t l3a    = l3a_buf[r];
            const uint8_t l3b    = l3b_buf[r];

            // ---- Bypass: a region of zeros on either side wins ----
            const bool a_is_zero = a_zero_when_l3_zero && l3a == 0;
            const bool b_is_zero = b_zero_when_l3_zero && l3b == 0;
            if (a_is_zero || b_is_zero) {
                // Only the non-zero side has literals to skip over;
                // a region-zero side has no L2/L1 literals to advance past.
                if (!a_is_zero) advance_side(A, l3a);
                if (!b_is_zero) advance_side(B, l3b);
                if (compress) {
                    // l2_flat_[region*8 .. +7] already zero-initialized.
                } else {
                    _mm512_storeu_si512(r_l1 + r_off, _mm512_setzero_si512());
                    r_off += 64;
                }
                continue;
            }

            // ---- Bypass: 1 & x = x (ones side passes the other through) ----
            // a_is_ones / b_is_ones cannot co-occur with a_is_zero / b_is_zero
            // for the same side (mutually exclusive on l1_fill_ones flag).
            // These branches are dead in our sparse-1 benchmarks
            // (l1_fill_ones=false), so branch predictor handles trivially.
            const bool a_is_ones = a_ones_when_l3_zero && l3a == 0;
            const bool b_is_ones = b_ones_when_l3_zero && l3b == 0;
            if (a_is_ones && b_is_ones) {
                // both all-1s: result = all-1s, no cursors to advance
                if (compress) {
                    std::memset(r_l1 + r_off, 0xFF, 64);
                    std::memset(result_l2 + region * 8, 0xFF, 8);
                    r_off += 64;
                } else {
                    _mm512_storeu_si512(r_l1 + r_off,
                        _mm512_set1_epi8(static_cast<char>(-1)));
                    r_off += 64;
                }
                continue;
            }
            if (a_is_ones) {
                // 1 & B = B: expand B only and write it directly
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
            if (b_is_ones) {
                // A & 1 = A: expand A only
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

            // ---- L3 -> L2 expand (per side) ----
            __m512i l2a_v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                static_cast<__mmask64>(l3a), A.l2_lits + A.l2_lit_off);
            A.l2_lit_off += __builtin_popcount(l3a);
            const __mmask64 ma = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));

            __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                static_cast<__mmask64>(l3b), B.l2_lits + B.l2_lit_off);
            B.l2_lit_off += __builtin_popcount(l3b);
            const __mmask64 mb = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

            // ---- L2 -> L1 expand (per side), then AND ----
            __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec, ma, A.l1_lits + A.l1_lit_off);
            A.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));
            __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb, B.l1_lits + B.l1_lit_off);
            B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));
            __m512i vr = _mm512_and_si512(va, vb);

            // ---- Store ----
            if (compress) {
                const __mmask64 lit_mask = _mm512_test_epi8_mask(vr, vr);
                const uint64_t  mask_val = static_cast<uint64_t>(lit_mask);
                std::memcpy(result_l2 + region * 8, &mask_val, 8);
                _mm512_mask_compressstoreu_epi8(r_l1 + r_off, lit_mask, vr);
                r_off += __builtin_popcountll(mask_val);
            } else {
                _mm512_storeu_si512(r_l1 + r_off, vr);
                r_off += 64;
            }
        }
    }

    // === Scalar tail: process remaining words without expand_l2() ===
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
                uint8_t vr = wa & wb;
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
                buf_a[i] &= buf_b[i];
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
    std::cout << "  [AND] "
              << "expand_and: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    if (compress) result.compact_l2_l3(r_off);
    return result;
}

// ----------------------------------------------------------------
// ComBit (segmented) AND operator
// ----------------------------------------------------------------

ComBit
ComBit::operator&(const ComBit& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    ComBit result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& sa = segments_[i];
        const auto& sb = other.segments_[i];

        // Segment-level bypass:
        //   a & 0 = 0, 0 & b = 0   (zero side wins, return all-zero)
        //   1 & b = b, a & 1 = a   (ones side returns the other)
        if (sa.is_all_zero() || sb.is_all_zero()) {
            result.segments_.push_back(
                ComBitBtv::make_all_fill(sa.bit_count(), sa.l2_count(), false));
            continue;
        }
        if (sa.is_all_ones()) { result.segments_.push_back(sb); continue; }
        if (sb.is_all_ones()) { result.segments_.push_back(sa); continue; }

        // Binary ComBitBtv::operator& asserts both inputs Compressed.  If
        // either is Decompressed (chained operator& result), fall back to
        // the in-place &= path which accepts Decompressed *this.
        if (sa.state() != ComBitBtv::State::Compressed) {
            ComBitBtv tmp = sa;
            tmp &= sb;
            result.segments_.push_back(std::move(tmp));
            continue;
        }
        if (sb.state() != ComBitBtv::State::Compressed) {
            ComBitBtv tmp = sb;
            tmp &= sa;
            result.segments_.push_back(std::move(tmp));
            continue;
        }

        result.segments_.push_back(sa & sb);
    }

    return result;
}

// ----------------------------------------------------------------
// ComBitBtv in-place AND (operator&=)
// ----------------------------------------------------------------

ComBitBtv&
ComBitBtv::operator&=(const ComBitBtv& other) {
    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Decompressed);
    assert(other.state_ != State::Uncompressed);

    if (bit_count_ == 0) return *this;

    const size_t total_words = l2_count_;
    uint8_t* r_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    size_t b_l1_off = 0;

#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;

    // Decompressed RHS: l1 holds every word; degenerates to flat 64-byte AND.
    if (other.state_ == State::Decompressed) {
        for (size_t region = 0; region < avx_regions; region++) {
            __m512i va = _mm512_loadu_si512(r_l1 + region * 64);
            __m512i vb = _mm512_loadu_si512(b_l1 + region * 64);
            _mm512_storeu_si512(r_l1 + region * 64,
                _mm512_and_si512(va, vb));
        }
        for (size_t pos = avx_regions * words_per_reg; pos < total_words; pos++)
            r_l1[pos] &= b_l1[pos];
        return *this;
    }

    // Compressed RHS: stream L3 bytes from L4, per-region expand+AND.
    const __m512i fill_b_vec = other.l1_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    const __m512i l2_fill_b_vec = other.l2_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    static constexpr size_t PF_DIST = 128;

    const uint8_t* b_l4 = other.l4_bits_.data();
    const uint8_t* b_l3_lits = other.l3_literals_.data();
    const uint8_t b_l3_fill = other.l3_fill_ones_ ? 0xFF : 0x00;
    size_t b_l3_lit_off = 0;
    size_t b_l2_off = 0;

    // Per-region bypass for AND (l2_fill=0):
    //   l3b==0 && l1_fill=0 => a & 0 = 0 (write zeros)
    //   l3b==0 && l1_fill=1 => a & 1 = a (skip)
    // l2_fill=1 && l3b==0 means full-literal region; cannot bypass.

    // ComBit-native fast skip: when RHS is sparse zero-fill (l1_fill=0,
    // l2_fill=0, l3_fill=0) and 8 consecutive L4 bytes are all zero,
    // 64 regions are all-zero in RHS → result must be cleared.  One
    // memset replaces 64 SIMD stores.  Per-loop dynamic, no preprocessing.
    const bool b_clear_skip =
        !other.l2_fill_ones_ && !other.l1_fill_ones_ && b_l3_fill == 0;

    for (size_t region = 0; region < avx_regions; region++) {
        if (b_clear_skip && (region & 63) == 0
            && region + 64 <= avx_regions) {
            uint64_t l4_chunk;
            std::memcpy(&l4_chunk, b_l4 + region / 8, 8);
            if (l4_chunk == 0) {
                std::memset(r_l1 + region * 64, 0, 64 * 64);
                region += 63;
                continue;
            }
        }
        bool b_l4_lit = (b_l4[region / 8] >> (region % 8)) & 1;
        uint8_t l3b = b_l4_lit ? b_l3_lits[b_l3_lit_off++] : b_l3_fill;

        if (!other.l2_fill_ones_ && l3b == 0) {
            if (!other.l1_fill_ones_) {
                _mm512_storeu_si512(r_l1 + region * 64,
                    _mm512_setzero_si512());
            }
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
            _mm512_and_si512(va, vb));
    }

    if (avx_regions * words_per_reg < total_words) {
        const uint8_t l1_fill_b = other.l1_fill_ones_ ? 0xFF : 0x00;
        const uint8_t l2_fill_b = other.l2_fill_ones_ ? 0xFF : 0x00;
        bool b_l4_lit = (b_l4[avx_regions / 8] >> (avx_regions % 8)) & 1;
        uint8_t l3b = b_l4_lit ? b_l3_lits[b_l3_lit_off++] : b_l3_fill;
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2b = ((l3b >> l2i) & 1) ? other.l2_literals_[b_l2_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wb = ((l2b >> bit) & 1) ? b_l1[b_l1_off++] : l1_fill_b;
                r_l1[pos] &= wb;
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
            r_l1[w] &= wb;
        }
    }
#endif
    return *this;
}

// ----------------------------------------------------------------
// ComBitBtv fused AND-popcount (popcount_and)
// ----------------------------------------------------------------
//
// Mirrors operator&= region-by-region but skips the writeback: each
// AND-of-region is reduced with VPOPCNTDQ into a 64-bit lane
// accumulator and the result is materialised only as a single size_t.
// Intended for queries that group-count many AND results without
// reusing the intersection downstream (e.g. Q4's per-priority counts).
size_t
ComBitBtv::popcount_and(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Decompressed);
    assert(other.state_ != State::Uncompressed);

    if (bit_count_ == 0) return 0;

    const size_t total_words = l2_count_;
    const uint8_t* a_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    size_t b_l1_off = 0;
    size_t total = 0;

#if defined(__AVX512VBMI2__) && defined(__AVX512VPOPCNTDQ__)
    const size_t avx_regions = total_words / words_per_reg;

    // Decompressed RHS shortcut (mirrors operator&=).
    if (other.state_ == State::Decompressed) {
        __m512i acc = _mm512_setzero_si512();
        for (size_t region = 0; region < avx_regions; region++) {
            __m512i va = _mm512_loadu_si512(a_l1 + region * 64);
            __m512i vb = _mm512_loadu_si512(b_l1 + region * 64);
            acc = _mm512_add_epi64(acc,
                _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
        }
        total += static_cast<size_t>(_mm512_reduce_add_epi64(acc));
        for (size_t pos = avx_regions * words_per_reg; pos < total_words; pos++)
            total += __builtin_popcount(a_l1[pos] & b_l1[pos]);
        return total;
    }

    // Compressed RHS: stream L3 bytes from L4, per-region expand+AND+popcount.
    const __m512i fill_b_vec = other.l1_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    const __m512i l2_fill_b_vec = other.l2_fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    static constexpr size_t PF_DIST = 128;

    const uint8_t* b_l4 = other.l4_bits_.data();
    const uint8_t* b_l3_lits = other.l3_literals_.data();
    const uint8_t b_l3_fill = other.l3_fill_ones_ ? 0xFF : 0x00;
    size_t b_l3_lit_off = 0;
    size_t b_l2_off = 0;

    __m512i acc = _mm512_setzero_si512();

    // Per-region bypass (l2_fill=0):
    //   l3b==0 && l1_fill=0 => contribution is 0
    //   l3b==0 && l1_fill=1 => contribution is popcount(LHS)

    // ComBit-native fast skip: when RHS is sparse zero-fill, 8 zero L4
    // bytes mean 64 all-zero regions → contributes 0 → skip whole batch.
    const bool b_zero_skip =
        !other.l2_fill_ones_ && !other.l1_fill_ones_ && b_l3_fill == 0;

    for (size_t region = 0; region < avx_regions; region++) {
        if (b_zero_skip && (region & 63) == 0
            && region + 64 <= avx_regions) {
            uint64_t l4_chunk;
            std::memcpy(&l4_chunk, b_l4 + region / 8, 8);
            if (l4_chunk == 0) {
                region += 63;
                continue;
            }
        }
        bool b_l4_lit = (b_l4[region / 8] >> (region % 8)) & 1;
        uint8_t l3b = b_l4_lit ? b_l3_lits[b_l3_lit_off++] : b_l3_fill;

        if (!other.l2_fill_ones_ && l3b == 0) {
            if (other.l1_fill_ones_) {
                __m512i va = _mm512_loadu_si512(a_l1 + region * 64);
                acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(va));
            }
            continue;
        }

        _mm_prefetch(reinterpret_cast<const char*>(
            b_l1 + b_l1_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(
            a_l1 + region * 64 + PF_DIST), _MM_HINT_T0);

        __m512i l2b_v = _mm512_mask_expandloadu_epi8(l2_fill_b_vec,
            static_cast<__mmask64>(l3b),
            other.l2_literals_.data() + b_l2_off);
        b_l2_off += __builtin_popcount(l3b);
        __mmask64 mb = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

        __m512i vb = _mm512_mask_expandloadu_epi8(fill_b_vec, mb,
            b_l1 + b_l1_off);
        b_l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i va = _mm512_loadu_si512(a_l1 + region * 64);
        acc = _mm512_add_epi64(acc,
            _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
    }
    total += static_cast<size_t>(_mm512_reduce_add_epi64(acc));

    // Scalar tail (final partial AVX region).  bit_count_ may not be a
    // multiple of 8 — by Decompressed-result construction the padding
    // bits in l1_literals_[last] are 0, so a flat popcount over the
    // whole tail is correct.
    if (avx_regions * words_per_reg < total_words) {
        const uint8_t l1_fill_b = other.l1_fill_ones_ ? 0xFF : 0x00;
        const uint8_t l2_fill_b = other.l2_fill_ones_ ? 0xFF : 0x00;
        bool b_l4_lit = (b_l4[avx_regions / 8] >> (avx_regions % 8)) & 1;
        uint8_t l3b = b_l4_lit ? b_l3_lits[b_l3_lit_off++] : b_l3_fill;
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2b = ((l3b >> l2i) & 1) ? other.l2_literals_[b_l2_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wb = ((l2b >> bit) & 1) ? b_l1[b_l1_off++] : l1_fill_b;
                total += __builtin_popcount(a_l1[pos] & wb);
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
            total += __builtin_popcount(a_l1[w] & wb);
        }
    }
#endif
    return total;
}

// ----------------------------------------------------------------
// ComBit (segmented) fused AND-popcount
// ----------------------------------------------------------------

size_t
ComBit::popcount_and(const ComBit& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    size_t total = 0;
    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& sb = other.segments_[i];

        // Segment-level bypass: a & 0 = 0; a & 1 = a.
        if (sb.is_all_zero()) continue;
        if (sb.is_all_ones()) { total += segments_[i].popcount(); continue; }
        total += segments_[i].popcount_and(sb);
    }
    return total;
}

// ----------------------------------------------------------------
// ComBit (segmented) in-place AND
// ----------------------------------------------------------------

ComBit&
ComBit::operator&=(const ComBit& other) {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& seg = other.segments_[i];
        if (seg.is_all_ones()) continue;                     // a & 1 = a
        if (seg.is_all_zero()) {                             // a & 0 = 0
            segments_[i] = ComBitBtv::make_all_fill(
                seg.bit_count(), seg.l2_count(), false);
            continue;
        }
        // segments_[i] may be Compressed (from prior all-fill bypass or
        // ComBit::operator& binary path).  ComBitBtv::operator&= asserts
        // Decompressed; route Compressed LHS through binary & which handles
        // both Compressed inputs and produces a Decompressed result.
        if (segments_[i].is_all_zero()) continue;            // 0 & b = 0
        if (segments_[i].is_all_ones()) {                    // 1 & b = b
            segments_[i] = seg;
            continue;
        }
        if (segments_[i].state() != ComBitBtv::State::Decompressed) {
            segments_[i] = segments_[i] & seg;
            continue;
        }
        segments_[i] &= seg;
    }

    return *this;
}
