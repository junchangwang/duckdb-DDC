#ifndef DDC_H
#define DDC_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

extern bool ddc_compress_results;

class DDCBtv {
public:
    static constexpr unsigned word_size = 8;
    static constexpr size_t word_byte_size = 1;
    static constexpr size_t words_per_reg = 64;
    static constexpr size_t default_segment_bits = 1 << 16;

    enum class State { Uncompressed, Compressed, Decompressed };

    struct SizeBreakdown {
        size_t l3_bits;
        size_t l4_bits;
        size_t l3_literal_bits;
        size_t l2_literal_bits;
        size_t l1_literal_bits;
        size_t total_bits;
    };

    explicit DDCBtv(bool l1_fill_ones = false, bool l2_fill_ones = false, State state = State::Compressed);

    static DDCBtv compress(const std::vector<bool>& bits, bool l1_fill_ones = false);
    static DDCBtv compress_sparse_segment(const std::vector<uint16_t>& sorted_positions, size_t seg_bits, bool l1_fill_ones = false);
    std::vector<bool> decompress() const;

    static DDCBtv from_string(const std::string& bitstring, bool l1_fill_ones = false);
    std::string to_string() const;

    DDCBtv operator&(const DDCBtv& other) const;
    DDCBtv operator|(const DDCBtv& other) const;
    DDCBtv operator^(const DDCBtv& other) const;
    DDCBtv operator~() const;
    DDCBtv& operator|=(const DDCBtv& other);
    DDCBtv& operator&=(const DDCBtv& other);
    DDCBtv& operator^=(const DDCBtv& other);

    DDCBtv& negate_inplace();

    size_t popcount() const;
    std::vector<size_t> set_bit_positions() const;
    void set_bit_decompressed(uint32_t pos_in_segment) {
        l1_literals_[pos_in_segment >> 3] |= (uint8_t)(0x80 >> (pos_in_segment & 7));
    }
    size_t popcount_and(const DDCBtv& other) const;

    SizeBreakdown size_breakdown() const;
    size_t compressed_size_bits()  const { return size_breakdown().total_bits; }
    size_t compressed_size_bytes() const { return (compressed_size_bits() + 7) / 8; }
    size_t original_size_bits()    const { return bit_count_; }
    double compression_ratio() const;

    bool   l1_fill_ones() const { return l1_fill_ones_; }
    bool   l2_fill_ones() const { return l2_fill_ones_; }
    bool   l3_fill_ones() const { return l3_fill_ones_; }
    State  state()        const { return state_; }
    size_t l2_count()     const { return l2_count_; }
    size_t l3_count()     const { return l3_count_; }
    size_t l4_count()     const { return l4_count_; }
    size_t bit_count()    const { return bit_count_; }
    size_t num_fills()    const;
    size_t num_literals() const { return l1_literal_count_; }
    size_t l2_literal_count() const { return l2_literal_count_; }
    size_t l3_literal_count() const { return l3_literal_count_; }

    bool is_all_zero() const { return l1_literal_count_ == 0 && !l1_fill_ones_; }
    bool is_all_ones() const { return l1_literal_count_ == 0 &&  l1_fill_ones_; }

    static DDCBtv make_all_fill(size_t bit_count, size_t l2_count, bool l1_fill_ones);
    static DDCBtv make_decompressed_zero(size_t bit_count, size_t l2_count);

    const uint8_t* l1_literal_data() const { return l1_literals_.data(); }
    const uint8_t* l2_flat_data()    const { return l2_flat_.data(); }
    const uint8_t* l2_literal_data() const { return l2_literals_.data(); }
    const uint8_t* l3_literal_data() const { return l3_literals_.data(); }
    const uint8_t* l4_data()         const { return l4_bits_.data(); }

    uint64_t get_literal(size_t idx) const;

#ifdef __AVX512VBMI2__

    struct SideCtx {
        const uint8_t* l4_bits;
        const uint8_t* l3_lits;
        const uint8_t* l2_lits;
        const uint8_t* l1_lits;
        bool           l3_fill_ones;
        bool           l2_fill_ones;
        bool           l1_fill_ones;
        __m512i        l3_fill_vec;
        __m512i        l2_fill_vec;
        __m512i        l1_fill_vec;
        size_t         l3_lit_off;
        size_t         l2_lit_off;
        size_t         l1_lit_off;
    };

    inline SideCtx make_side(const uint8_t* l1_data) const {
        SideCtx c{};
        c.l4_bits      = l4_bits_.data();
        c.l3_lits      = l3_literals_.data();
        c.l2_lits      = l2_literals_.data();
        c.l1_lits      = l1_data;
        c.l3_fill_ones = l3_fill_ones_;
        c.l2_fill_ones = l2_fill_ones_;
        c.l1_fill_ones = l1_fill_ones_;
        c.l3_fill_vec  = l3_fill_ones_ ? _mm512_set1_epi8(static_cast<char>(-1))
                                       : _mm512_setzero_si512();
        c.l2_fill_vec  = l2_fill_ones_ ? _mm512_set1_epi8(static_cast<char>(-1))
                                       : _mm512_setzero_si512();
        c.l1_fill_vec  = l1_fill_ones_ ? _mm512_set1_epi8(static_cast<char>(-1))
                                       : _mm512_setzero_si512();
        return c;
    }

    static inline void advance_side(SideCtx& s, uint8_t l3) {
        __m512i l2v = _mm512_mask_expandloadu_epi8(s.l2_fill_vec,
            static_cast<__mmask64>(l3), s.l2_lits + s.l2_lit_off);
        s.l2_lit_off += __builtin_popcount(l3);
        __mmask64 m = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
        s.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(m));
    }
#endif

    void serialize(std::ostream& os) const;
    static DDCBtv deserialize(std::istream& is);

    void serialize_packed(std::ostream& os, bool is_last_seg) const;
    static DDCBtv deserialize_packed(std::istream& is, size_t segment_bits);

    void serialize_v4(std::ostream& os, bool is_last_seg) const;
    static DDCBtv deserialize_v4(std::istream& is, size_t segment_bits);

    // Cache prefetch
    inline void prefetch_layers() const {
        __builtin_prefetch(l4_bits_.data());
        __builtin_prefetch(l3_literals_.data());
        __builtin_prefetch(l2_literals_.data());
        __builtin_prefetch(l1_literals_.data());
    }

    void print(std::ostream& os = std::cout) const;

private:

    State                   state_;
    bool                    l1_fill_ones_;
    bool                    l2_fill_ones_;
    size_t                  bit_count_;
    size_t                  l2_count_;
    std::vector<uint8_t>    l2_flat_;
    size_t                  l3_count_;
    std::vector<uint8_t>    l3_bits_;
    std::vector<uint8_t>    l2_literals_;
    size_t                  l2_literal_count_;
    bool                    l3_fill_ones_;
    size_t                  l4_count_;
    std::vector<uint8_t>    l4_bits_;
    std::vector<uint8_t>    l3_literals_;
    size_t                  l3_literal_count_;
    std::vector<uint8_t>    l1_literals_;
    size_t                  l1_literal_count_;

    std::vector<uint8_t> expand_l3() const;
    std::vector<uint8_t> expand_l2() const;
    void compact_l2_l3(size_t actual_l1_count);
    void compress_l3_to_l4();
    bool is_last_word_literal() const;

    friend class DDC;
};

class DDC {
public:
    static constexpr size_t default_segment_bits = size_t(1) << 16;

    struct SizeBreakdown {
        size_t l3_bits;
        size_t l4_bits;
        size_t l3_literal_bits;
        size_t l2_literal_bits;
        size_t l1_literal_bits;
        size_t total_bits;
    };

    DDC() = default;

    static DDC compress(const std::vector<bool>& bits, bool l1_fill_ones = false, size_t segment_bits = default_segment_bits);
    static DDC from_sparse_positions(const std::vector<uint32_t>& positions, size_t num_rows, size_t segment_bits = default_segment_bits);
    // Compact builder
    static DDC from_sparse_positions_compact(const std::vector<uint32_t>& positions, size_t num_rows, size_t segment_bits = default_segment_bits);
    std::vector<bool> decompress() const;

    static DDC from_string(const std::string& bitstring, bool l1_fill_ones = false, size_t segment_bits = default_segment_bits);
    std::string to_string() const;

    DDC operator&(const DDC& other) const;
    DDC operator|(const DDC& other) const;
    DDC operator^(const DDC& other) const;
    DDC operator~() const;
    DDC& operator|=(const DDC& other);
    DDC& operator&=(const DDC& other);
    DDC& operator^=(const DDC& other);

    DDC& negate_inplace();

    static DDC OR_many(size_t number, const DDC** Btvs);

    size_t popcount() const;
    std::vector<size_t> set_bit_positions() const;
    size_t popcount_and(const DDC& other) const;

    template<typename Fn>
    void for_each_literal(Fn&& fn) const {
        size_t word_off = 0;
        for (const auto& seg : segments_) {
            const uint8_t* l1 = seg.l1_literal_data();
            const size_t l2_total = seg.l2_count();

            if (seg.is_all_zero()) {
                word_off += l2_total;
                continue;
            }

            if (seg.state() == DDCBtv::State::Decompressed) {
                size_t i = 0;
#ifdef __AVX512BW__
                for (; i + 64 <= l2_total; i += 64) {
                    __m512i chunk = _mm512_loadu_si512(l1 + i);
                    uint64_t nz = static_cast<uint64_t>(_mm512_test_epi8_mask(chunk, chunk));
                    while (nz) {
                        int b = __builtin_ctzll(nz);
                        nz &= nz - 1;
                        fn(static_cast<uint32_t>(word_off + i + b), l1[i + b]);
                    }
                }
#endif
                for (; i < l2_total; i++) {
                    if (l1[i] != 0)
                        fn(static_cast<uint32_t>(word_off + i), l1[i]);
                }
                word_off += l2_total;
                continue;
            }

            size_t l1_off = 0;
            size_t l2_lit = 0;
            const uint8_t fill = seg.l1_fill_ones() ? 0xFF : 0x00;
            const uint8_t* l2_lits = seg.l2_literal_data();
            const size_t l3_total = seg.l3_count();
            const size_t l3_bytes = (l3_total + 7) / 8;
            const bool can_skip = !seg.l2_fill_ones() && fill == 0;

            const uint8_t* l4_data = seg.l4_data();
            const uint8_t* l3_lits = seg.l3_literal_data();
            const uint8_t  l3_fill = seg.l3_fill_ones() ? 0xFF : 0x00;
            size_t l3_lit_off = 0;

#ifdef __AVX512VBMI2__

            if (can_skip) {
                for (size_t l3_base = 0; l3_base < l3_bytes; l3_base += 64) {
                    size_t chunk = l3_bytes - l3_base;
                    if (chunk > 64) chunk = 64;

                    uint64_t l4_mask = 0;
                    std::memcpy(&l4_mask, l4_data + l3_base / 8, (chunk + 7) / 8);
                    if (chunk < 64)
                        l4_mask &= (uint64_t(1) << chunk) - 1;

                    auto walk_l3_byte = [&](size_t l3_byte_idx, uint8_t l3b) {
                        uint32_t l3_tmp = l3b;
                        while (l3_tmp) {
                            int l3_bit = __builtin_ctz(l3_tmp);
                            l3_tmp &= l3_tmp - 1;
                            size_t l3_idx = l3_byte_idx * 8 + l3_bit;
                            if (l3_idx >= l3_total) break;

                            uint8_t l2_byte = l2_lits[l2_lit++];
                            if (l2_byte == 0) continue;

                            size_t base_w = l3_idx * 8;
                            size_t remaining = (base_w + 8 <= l2_total) ? 8 : (l2_total - base_w);
                            uint8_t mask = (remaining >= 8) ? l2_byte : static_cast<uint8_t>(l2_byte & ((1u << remaining) - 1));

                            uint32_t tmp = mask;
                            while (tmp) {
                                int bit = __builtin_ctz(tmp);
                                tmp &= tmp - 1;
                                uint8_t val = l1[l1_off++];
                                if (val != 0)
                                    fn(static_cast<uint32_t>(word_off + base_w + bit), val);
                            }
                        }
                    };

                    if (l3_fill == 0) {
                        while (l4_mask) {
                            size_t bidx = static_cast<size_t>(__builtin_ctzll(l4_mask));
                            l4_mask &= l4_mask - 1;
                            walk_l3_byte(l3_base + bidx, l3_lits[l3_lit_off++]);
                        }
                        continue;
                    }

                    const __m512i l3_fill_vec = _mm512_set1_epi8(static_cast<char>(l3_fill));
                    __m512i l3v = _mm512_mask_expandloadu_epi8(l3_fill_vec, static_cast<__mmask64>(l4_mask), l3_lits + l3_lit_off);
                    l3_lit_off += __builtin_popcountll(l4_mask);
                    if (chunk < 64) {
                        __mmask64 valid = static_cast<__mmask64>((uint64_t(1) << chunk) - 1);
                        l3v = _mm512_maskz_mov_epi8(valid, l3v);
                    }
                    uint64_t nz = static_cast<uint64_t>(_mm512_test_epi8_mask(l3v, l3v));
                    if (nz == 0) continue;

                    alignas(64) uint8_t l3_buf[64];
                    _mm512_store_si512(reinterpret_cast<__m512i*>(l3_buf), l3v);
                    while (nz) {
                        size_t bidx = static_cast<size_t>(__builtin_ctzll(nz));
                        nz &= nz - 1;
                        walk_l3_byte(l3_base + bidx, l3_buf[bidx]);
                    }
                }
                word_off += l2_total;
                continue;
            }
#endif

            size_t cur_l3_byte_idx = static_cast<size_t>(-1);
            uint8_t cur_l3_byte = 0;
            for (size_t l3_idx = 0; l3_idx < l3_total; l3_idx++) {
                size_t l3_byte_idx = l3_idx / 8;
                if (l3_byte_idx != cur_l3_byte_idx) {
                    cur_l3_byte_idx = l3_byte_idx;
                    bool lit = (l4_data[l3_byte_idx / 8] >> (l3_byte_idx % 8)) & 1;
                    cur_l3_byte = lit ? l3_lits[l3_lit_off++] : l3_fill;
                }
                bool l3_is_lit = (cur_l3_byte >> (l3_idx % 8)) & 1;
                uint8_t l2_byte = l3_is_lit ? l2_lits[l2_lit++] : (seg.l2_fill_ones() ? 0xFF : 0x00);
                if (l2_byte == 0 && fill == 0) continue;
                for (int bit = 0; bit < 8; bit++) {
                    size_t w = l3_idx * 8 + bit;
                    if (w >= l2_total) break;
                    bool is_lit = (l2_byte >> bit) & 1;
                    uint8_t val = is_lit ? l1[l1_off++] : fill;
                    if (val != 0)
                        fn(static_cast<uint32_t>(word_off + w), val);
                }
            }
            word_off += l2_total;
        }
    }

    SizeBreakdown size_breakdown() const;
    size_t compressed_size_bits()  const { return size_breakdown().total_bits; }
    size_t compressed_size_bytes() const { return (compressed_size_bits() + 7) / 8; }
    size_t original_size_bits()    const { return bit_count_; }
    double compression_ratio() const;

    size_t bit_count()     const { return bit_count_; }
    size_t num_segments()  const { return segments_.size(); }
    size_t segment_bits()  const { return segment_bits_; }

    const std::vector<DDCBtv>& segments() const { return segments_; }
    std::vector<DDCBtv>& segments() { return segments_; }
    const DDCBtv& segment(size_t i) const { return segments_[i]; }

    void scatter_or_decompressed(const uint32_t* positions, size_t n) {
        const uint32_t seg_bits = static_cast<uint32_t>(segment_bits_);
        if ((seg_bits & (seg_bits - 1)) == 0) {
            // Fast addressing
            const uint32_t shift = static_cast<uint32_t>(__builtin_ctz(seg_bits));
            const uint32_t mask  = seg_bits - 1;
            for (size_t i = 0; i < n; i++) {
                uint32_t p = positions[i];
                segments_[p >> shift].set_bit_decompressed(p & mask);
            }
            return;
        }
        for (size_t i = 0; i < n; i++) {
            uint32_t p = positions[i];
            uint32_t seg = p / seg_bits;
            uint32_t in  = p - seg * seg_bits;
            segments_[seg].set_bit_decompressed(in);
        }
    }

    void serialize(std::ostream& os) const;
    static DDC deserialize(std::istream& is);

    void serialize_packed(std::ostream& os) const;
    static DDC deserialize_packed(std::istream& is);

    void serialize_v4(std::ostream& os) const;
    static DDC deserialize_v4(std::istream& is);

    void print(std::ostream& os = std::cout) const;

private:
    std::vector<DDCBtv> segments_;
    size_t bit_count_    = 0;
    size_t segment_bits_ = default_segment_bits;
};

class SparseDDC {
public:
    SparseDDC() = default;

    static SparseDDC from_positions(const std::vector<uint32_t>& positions, size_t num_rows, size_t segment_bits = DDC::default_segment_bits, bool adaptive_l1_polarity = false);

    void apply_or_to(DDC& dst) const;
    static DDC or_many(size_t count, const SparseDDC** sparses, size_t num_rows, size_t segment_bits);

    size_t bit_count()    const { return bit_count_; }
    size_t segment_bits() const { return segment_bits_; }
    size_t num_set_bits() const { return num_set_bits_; }
    size_t num_non_empty_segments() const { return seg_indices_.size(); }

    const std::vector<uint32_t>&  seg_indices() const { return seg_indices_; }
    const std::vector<DDCBtv>& seg_data()    const { return seg_data_; }

    size_t storage_bytes() const;

private:
    size_t bit_count_    = 0;
    size_t segment_bits_ = DDC::default_segment_bits;
    size_t num_set_bits_ = 0;
    std::vector<uint32_t>  seg_indices_;
    std::vector<DDCBtv> seg_data_;
};

#endif
