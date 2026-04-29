#ifndef COMBIT_H
#define COMBIT_H

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

/// When true, bitwise operators (AND, OR, XOR) produce compressed
/// result bitvectors directly (region-by-region).  When false (the
/// default), results are returned fully expanded (all words as
/// literals) for maximum throughput.
extern bool combit_compress_results;

///
/// ComBitBtv: A fixed-length bitvector segment compressed with a
/// three-level structure:
///
///   L1 – literal data: 8-bit words of the original bitvector (only
///         non-fill words are stored).
///   L2 – leading bitstring for L1: one bit per 8-bit word of the
///         original bitvector (0 = fill, 1 = literal).  Stored as a
///         packed byte array; only non-zero bytes are kept when the L3
///         layer is active.
///   L3 – leading bitstring for L2: one bit per 8-bit chunk of L2
///         (i.e. per group of 64 original words = 512 bits).
///         0 = the L2 byte is all-zero (entire 64-word region is fills),
///         1 = the L2 byte is a literal and stored in l2_literals_.
///
/// Word size is fixed at 8 bits.
/// l1_fill_ones is a runtime parameter controlling the L1 fill value.
///
class ComBitBtv {
public:
    static constexpr unsigned word_size = 8;
    static constexpr size_t word_byte_size = 1;
    static constexpr size_t words_per_reg = 64;              // 512 / 8
    static constexpr size_t l2_bits_per_l3_bit = 8;          // 8 L2 bits per L3 bit
    static constexpr size_t words_per_l3_bit = 64;           // 8 * 8
    static constexpr size_t default_segment_bits = 1 << 16;  // 65536

    /// Encoding state of a ComBitBtv segment.
    ///   Uncompressed  – plain bitvector: only L1 holds raw 8-bit words;
    ///                    L2 and L3 are empty / unused.
    ///   Compressed    – full three-level encoding with meaningful L2/L3.
    ///   Decompressed  – operator result: L1 holds all words; L2 is
    ///                    logically all-ones (l2_fill_ones_=true, L3
    ///                    all-zeros, l2_literal_count_=0).
    enum class State { Uncompressed, Compressed, Decompressed };

    struct SizeBreakdown {
        size_t l3_bits;            // L3 leading bits
        size_t l2_literal_bits;    // L2 stored literal bytes * 8
        size_t l1_literal_bits;    // L1 stored literal bytes * 8
        size_t total_bits;
    };

    explicit ComBitBtv(bool l1_fill_ones = false,
                       bool l2_fill_ones = false,
                       State state = State::Compressed);

    // ----------------------------------------------------------------
    // Compression / Decompression
    // ----------------------------------------------------------------

    static ComBitBtv compress(const std::vector<bool>& bits, bool l1_fill_ones = false);
    std::vector<bool> decompress() const;

    // ----------------------------------------------------------------
    // Convenience constructors
    // ----------------------------------------------------------------

    static ComBitBtv from_string(const std::string& bitstring, bool l1_fill_ones = false);
    std::string to_string() const;

    // ----------------------------------------------------------------
    // Bitwise operations
    // ----------------------------------------------------------------

    ComBitBtv operator&(const ComBitBtv& other) const;
    ComBitBtv operator|(const ComBitBtv& other) const;
    ComBitBtv operator^(const ComBitBtv& other) const;
    ComBitBtv operator~() const;

    // In-place compound assignment (avoids allocation when accumulator
    // is already fully expanded, i.e. after the first operator| / &).
    ComBitBtv& operator|=(const ComBitBtv& other);
    ComBitBtv& operator&=(const ComBitBtv& other);

    // ----------------------------------------------------------------
    // Post-operation compression
    // ----------------------------------------------------------------

    /// Compress a fully-expanded segment in-place: compact L1 (remove
    /// zero words), rebuild L2 from scratch, apply L3 compression.
    /// No-op if the segment is already compressed.
    void compact_expanded();

    // ----------------------------------------------------------------
    // Queries
    // ----------------------------------------------------------------

    size_t popcount() const;
    std::vector<size_t> set_bit_positions() const;

    /// Fused AND + popcount: returns popcount(*this & other) without
    /// materialising the intersection.  Mirrors CRoaring's
    /// `and_cardinality`, EWAH's `logicalandcount`, ibis::bitvector's
    /// `count(mask)`, and Concise's `logicalandCount` so that ComBit
    /// competes on equal API footing in queries that group-count many
    /// AND results (e.g. TPC-H Q4's per-priority counts).
    /// Requires `*this` to be Decompressed (canonical post-operator
    /// state); `other` may be Compressed or Decompressed.
    size_t popcount_and(const ComBitBtv& other) const;

    // ----------------------------------------------------------------
    // Size / statistics
    // ----------------------------------------------------------------

    SizeBreakdown size_breakdown() const;
    size_t compressed_size_bits()  const { return size_breakdown().total_bits; }
    size_t compressed_size_bytes() const { return (compressed_size_bits() + 7) / 8; }
    size_t original_size_bits()    const { return bit_count_; }
    double compression_ratio() const;

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    bool                          l1_fill_ones()       const { return l1_fill_ones_; }
    bool                          l2_fill_ones()   const { return l2_fill_ones_; }
    State                         state()          const { return state_; }
    size_t                        l2_count()           const { return l2_count_; }
    size_t                        l3_count()           const { return l3_count_; }
    size_t                        bit_count()          const { return bit_count_; }
    size_t num_fills() const;
    size_t num_literals()  const { return l1_literal_count_; }

    // Raw data access (used by bitwise operators)
    const uint8_t* l3_data()         const { return l3_bits_.data(); }
    const uint8_t* l2_flat_data()    const { return l2_flat_.data(); }
    const uint8_t* l2_literal_data() const { return l2_literals_.data(); }
    const uint8_t* l1_literal_data() const { return l1_literals_.data(); }
    size_t         l2_literal_count() const { return l2_literal_count_; }

    uint64_t get_literal(size_t idx) const;

    // ----------------------------------------------------------------
    // Serialization
    // ----------------------------------------------------------------

    void serialize(std::ostream& os) const;
    static ComBitBtv deserialize(std::istream& is);

    // ----------------------------------------------------------------
    // Debug printing
    // ----------------------------------------------------------------

    void print(std::ostream& os = std::cout) const;

private:
    State                   state_;
    // When true, L1 fill words are all-ones (0xFF); otherwise all-zeros (0x00).
    // An L2 bit of 0 means the corresponding L1 word equals this fill value.
    bool                    l1_fill_ones_;
    // When true, L2 fill bytes are all-ones (0xFF); otherwise all-zeros (0x00).
    // An L3 bit of 0 means the corresponding L2 byte equals this fill value.
    bool                    l2_fill_ones_;
    size_t                  bit_count_;            // original bitvector length

    // --- L2: leading bitstring for L1 (1 bit per 8-bit word) ---
    // L2 is always compressed via L3/l2_literals_.
    // l2_flat_ is used only as a scratch buffer during operator computation.
    size_t                  l2_count_;             // total L2 bits (= num 8-bit words)
    std::vector<uint8_t>    l2_flat_;              // scratch buffer for operators

    // --- L3: leading bitstring for L2 (1 bit per 8-bit chunk of L2) ---
    size_t                  l3_count_;             // total L3 bits
    std::vector<uint8_t>    l3_bits_;              // packed L3 bytes
    std::vector<uint8_t>    l2_literals_;          // L2 literal bytes (non-zero L2 chunks)
    size_t                  l2_literal_count_;

    // --- L1: literal data ---
    std::vector<uint8_t>    l1_literals_;          // L1 literal word bytes
    size_t                  l1_literal_count_;

    // Rebuild flat L2 from L3 + L2 literals (for decompression / scalar paths)
    std::vector<uint8_t> expand_l2() const;

    // Finalize a compressed result: shrink L1 to actual_l1_count,
    // then optionally apply L3 compression on L2.
    void compact_l2_l3(size_t actual_l1_count);

    // Check whether the last word of the segment is a literal (L2 bit set).
    // Only meaningful when l2_count_ > 0.  Used by operator~ and popcount
    // to handle padding-bit corrections for the last partial word.
    bool is_last_word_literal() const;

    static uint64_t read_word_from_bits(const std::vector<bool>& bits,
                                        size_t word_idx);
    static void append_word_to_bits(std::vector<bool>& bits, uint64_t word);

    friend class ComBit;
};

// ====================================================================
// ComBit: Segmented bitvector composed of ComBitBtv segments
// ====================================================================

///
/// ComBit: A segmented compressed bitvector.
///
/// The bitvector is partitioned into fixed-length segments (default 2^16
/// bits each), where each segment is independently compressed as a
/// ComBitBtv (8-bit word size).
///
class ComBit {
public:
    static constexpr size_t default_segment_bits = size_t(1) << 16;

    struct SizeBreakdown {
        size_t l3_bits;
        size_t l2_literal_bits;
        size_t l1_literal_bits;
        size_t total_bits;
    };

    ComBit() = default;

    // ----------------------------------------------------------------
    // Compression / Decompression
    // ----------------------------------------------------------------

    static ComBit compress(const std::vector<bool>& bits,
                           bool l1_fill_ones = false,
                           size_t segment_bits = default_segment_bits);

    std::vector<bool> decompress() const;

    // ----------------------------------------------------------------
    // Convenience constructors
    // ----------------------------------------------------------------

    static ComBit from_string(const std::string& bitstring,
                              bool l1_fill_ones = false,
                              size_t segment_bits = default_segment_bits);

    std::string to_string() const;

    // ----------------------------------------------------------------
    // Bitwise operations (segment-wise)
    // ----------------------------------------------------------------

    ComBit operator&(const ComBit& other) const;
    ComBit operator|(const ComBit& other) const;
    ComBit operator^(const ComBit& other) const;
    ComBit operator~() const;

    ComBit& operator|=(const ComBit& other);
    ComBit& operator&=(const ComBit& other);

    static ComBit OR_many(size_t number, const ComBit** Btvs);

    // ----------------------------------------------------------------
    // Queries
    // ----------------------------------------------------------------

    size_t popcount() const;
    std::vector<size_t> set_bit_positions() const;

    /// Fused AND + popcount across all segments: returns
    /// popcount(*this & other) without materialising the intersection.
    /// See ComBitBtv::popcount_and for the per-segment contract.
    size_t popcount_and(const ComBit& other) const;

    /// Iterate over all non-zero L1 words, calling
    /// fn(uint32_t word_pos, uint8_t value) for each.  Walks L3→L2→L1
    /// and skips all-zero regions efficiently.
    template<typename Fn>
    void for_each_literal(Fn&& fn) const {
        size_t word_off = 0;
        for (const auto& seg : segments_) {
            const uint8_t* l1 = seg.l1_literal_data();
            size_t l1_off = 0;
            size_t l2_lit = 0;
            const uint8_t fill = seg.l1_fill_ones() ? 0xFF : 0x00;
            const uint8_t* l3_data = seg.l3_data();
            const uint8_t* l2_lits = seg.l2_literal_data();
            const size_t l3_total = seg.l3_count();
            const size_t l2_total = seg.l2_count();
            const size_t l3_bytes = (l3_total + 7) / 8;
            const bool can_skip = !seg.l2_fill_ones() && fill == 0;

#ifdef __AVX512BW__
            // AVX-512 fast path B: Decompressed segment (canonical result
            // of ComBit::OR_many / operator| / operator&= / operator~).
            // By the Decompressed contract, l1_literal_count_ == l2_count_
            // and every word position is materialized in l1_literals_, so
            // the actual bit pattern lives there irrespective of the
            // residual l1_fill_ones_ flag (operator~ flips that flag in
            // place but the data in l1_literals_ already encodes the
            // result).  Walk l1 directly: SIMD test 64 bytes, bitscan
            // non-zero ones.  Lets callers iterate chained-op results
            // (including post-NOT) without first having to compact them.
            if (seg.state() == ComBitBtv::State::Decompressed) {
                size_t i = 0;
                for (; i + 64 <= l2_total; i += 64) {
                    __m512i chunk = _mm512_loadu_si512(l1 + i);
                    __mmask64 nz = _mm512_test_epi8_mask(chunk, chunk);
                    if (nz == 0) continue;
                    uint64_t nz64 = static_cast<uint64_t>(nz);
                    while (nz64) {
                        int b = __builtin_ctzll(nz64);
                        nz64 &= nz64 - 1;
                        fn(static_cast<uint32_t>(word_off + i + b), l1[i + b]);
                    }
                }
                for (; i < l2_total; i++) {
                    if (l1[i] != 0)
                        fn(static_cast<uint32_t>(word_off + i), l1[i]);
                }
                word_off += l2_total;
                continue;
            }

            if (can_skip) {
                // AVX-512 fast path: batch-test 64 L3 bytes at once,
                // bitscan to skip zero regions in bulk.
                for (size_t l3_base = 0; l3_base < l3_bytes; l3_base += 64) {
                    size_t chunk = l3_bytes - l3_base;
                    if (chunk > 64) chunk = 64;
                    __mmask64 ld_mask = (chunk >= 64)
                        ? static_cast<__mmask64>(-1ULL)
                        : static_cast<__mmask64>((1ULL << chunk) - 1);
                    __m512i l3v = _mm512_maskz_loadu_epi8(ld_mask, l3_data + l3_base);
                    uint64_t nz = static_cast<uint64_t>(
                        _mm512_test_epi8_mask(l3v, l3v));
                    if (nz == 0) continue;

                    while (nz) {
                        size_t bidx = static_cast<size_t>(__builtin_ctzll(nz));
                        nz &= nz - 1;
                        size_t l3_byte_idx = l3_base + bidx;
                        uint8_t l3b = l3_data[l3_byte_idx];

                        // Each L3 set bit → one L2 literal byte
                        uint32_t l3_tmp = l3b;
                        while (l3_tmp) {
                            int l3_bit = _tzcnt_u32(l3_tmp);
                            l3_tmp &= l3_tmp - 1;
                            size_t l3_idx = l3_byte_idx * 8 + l3_bit;
                            if (l3_idx >= l3_total) break;

                            uint8_t l2_byte = l2_lits[l2_lit++];
                            if (l2_byte == 0) continue;

                            size_t base_w = l3_idx * 8;
                            size_t remaining = (base_w + 8 <= l2_total)
                                ? 8 : (l2_total - base_w);
                            uint8_t mask = (remaining >= 8)
                                ? l2_byte
                                : static_cast<uint8_t>(l2_byte & ((1u << remaining) - 1));

                            // L2 bitscan: only visit set bits (literals)
                            uint32_t tmp = mask;
                            while (tmp) {
                                int bit = _tzcnt_u32(tmp);
                                tmp &= tmp - 1;
                                uint8_t val = l1[l1_off++];
                                if (val != 0)
                                    fn(static_cast<uint32_t>(word_off + base_w + bit), val);
                            }
                        }
                    }
                }
                word_off += l2_total;
                continue;
            }
#endif
            // Scalar fallback (fill!=0 or l2_fill_literals)
            for (size_t l3_idx = 0; l3_idx < l3_total; l3_idx++) {
                bool l3_is_lit = (l3_data[l3_idx / 8] >> (l3_idx % 8)) & 1;
                uint8_t l2_byte;
                if (l3_is_lit) {
                    l2_byte = l2_lits[l2_lit++];
                } else {
                    l2_byte = seg.l2_fill_ones() ? 0xFF : 0x00;
                }
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

    // ----------------------------------------------------------------
    // Size / statistics
    // ----------------------------------------------------------------

    SizeBreakdown size_breakdown() const;
    size_t compressed_size_bits()  const { return size_breakdown().total_bits; }
    size_t compressed_size_bytes() const { return (compressed_size_bits() + 7) / 8; }
    size_t original_size_bits()    const { return bit_count_; }
    double compression_ratio() const;

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    size_t bit_count()     const { return bit_count_; }
    size_t num_segments()  const { return segments_.size(); }
    size_t segment_bits()  const { return segment_bits_; }

    const std::vector<ComBitBtv>& segments() const { return segments_; }
    std::vector<ComBitBtv>& segments() { return segments_; }
    const ComBitBtv& segment(size_t i) const { return segments_[i]; }

    // ----------------------------------------------------------------
    // Serialization
    // ----------------------------------------------------------------

    void serialize(std::ostream& os) const;
    static ComBit deserialize(std::istream& is);

    // ----------------------------------------------------------------
    // Debug printing
    // ----------------------------------------------------------------

    void print(std::ostream& os = std::cout) const;

private:
    std::vector<ComBitBtv> segments_;
    size_t bit_count_ = 0;
    size_t segment_bits_ = default_segment_bits;
};

#endif // COMBIT_H
