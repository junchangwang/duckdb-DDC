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
/// four-level structure:
///
///   L1 – literal data: word_size-bit words of the original bitvector
///         (only non-fill words are stored, packed as raw bytes).
///   L2 – leading bitstring for L1: one bit per word of the original
///         bitvector (0 = fill, 1 = literal).  Stored as a packed byte
///         array; only non-zero bytes are kept when the L3 layer is
///         active.
///   L3 – leading bitstring for L2: one bit per 8-bit chunk of L2
///         (i.e. per group of 64 original words).
///         0 = the L2 byte is all-zero (entire 64-word region is fills),
///         1 = the L2 byte is a literal and stored in l2_literals_.
///         Compressed via L4: only non-fill L3 bytes are stored in
///         l3_literals_; expand_l3() reconstructs the dense form.
///   L4 – leading bitstring for L3: one bit per L3 byte
///         (i.e. per group of 512 original words).
///         0 = the L3 byte equals the chosen L3 fill value,
///         1 = the L3 byte is a literal and stored in l3_literals_.
///
/// Word size is fixed at 8 bits (1 byte per L1 word).
///
/// l1_fill_ones / l2_fill_ones_ / l3_fill_ones_ are the per-level fill
/// polarities; chosen at compress() time to minimise stored literals.
///
class ComBitBtv {
public:
    static constexpr unsigned word_size = 8;
    static constexpr size_t word_byte_size = 1;
    static constexpr size_t words_per_reg = 64;             // 512 / 8
    static constexpr size_t l2_bits_per_l3_bit = 8;         // 8 L2 bits per L3 bit
    static constexpr size_t words_per_l3_bit = 64;          // 8 * 8
    static constexpr size_t default_segment_bits = 1 << 16; // 65536

    /// Encoding state of a ComBitBtv segment.
    ///   Uncompressed  – plain bitvector: only L1 holds raw 8-bit words;
    ///                    L2 and L3 are empty / unused.
    ///   Compressed    – full three-level encoding with meaningful L2/L3.
    ///   Decompressed  – operator result: L1 holds all words; L2 is
    ///                    logically all-ones (l2_fill_ones_=true, L3
    ///                    all-zeros, l2_literal_count_=0).
    enum class State { Uncompressed, Compressed, Decompressed };

    struct SizeBreakdown {
        size_t l3_bits;            // L3 logical bits (= l3_count_)
        size_t l4_bits;            // L4 leading bits (= l4_count_)
        size_t l3_literal_bits;    // L3 stored literal bytes * 8
        size_t l2_literal_bits;    // L2 stored literal bytes * 8
        size_t l1_literal_bits;    // L1 stored literal bytes * 8
        size_t total_bits;         // L4 + L3_lit + L2_lit + L1_lit
    };

    explicit ComBitBtv(bool l1_fill_ones = false,
                       bool l2_fill_ones = false,
                       State state = State::Compressed);

    // ----------------------------------------------------------------
    // Compression / Decompression
    // ----------------------------------------------------------------

    static ComBitBtv compress(const std::vector<bool>& bits,
                              bool l1_fill_ones = false);
    // Sparse fast path: build a segment from a sorted list of set positions.
    // O(positions.size() + seg_bits / 64) instead of O(seg_bits).
    // Pre: positions sorted ascending, all values < seg_bits, no duplicates.
    // l1_fill_ones=false only (sparse case); falls back to compress() if true.
    static ComBitBtv compress_sparse_segment(
        const std::vector<uint16_t>& sorted_positions,
        size_t seg_bits,
        bool l1_fill_ones = false);
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
    ComBitBtv& operator^=(const ComBitBtv& other);

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

    // Set a single bit at `pos_in_segment` in this segment's L1 byte
    // array.  Pre: state_ == State::Decompressed, l1_literals_ already
    // sized to l2_count_ bytes (i.e. the canonical Decompressed layout
    // produced by ComBit::from_sparse_positions({}) seed segments).
    // Used by SparseComBit's ultra-sparse fast path to scatter raw
    // positions directly into a Decompressed-zero result without going
    // through the L1/L2/L3/L4 hierarchy.
    void set_bit_decompressed(uint32_t pos_in_segment) {
        l1_literals_[pos_in_segment >> 3] |= (uint8_t)(0x80 >> (pos_in_segment & 7));
    }

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

    // Logical all-fill predicates: num_literals==0 implies !l2_fill_ones_
    // (the fill choice in compact_l2_l3 minimizes stored literals).
    bool is_all_zero() const { return l1_literal_count_ == 0 && !l1_fill_ones_; }
    bool is_all_ones() const { return l1_literal_count_ == 0 &&  l1_fill_ones_; }

    // Factory for an all-fill Compressed segment (no scratch buffers).
    static ComBitBtv make_all_fill(size_t bit_count, size_t l2_count,
                                   bool l1_fill_ones);

    // Build a Decompressed-zero segment ready for in-place scatter:
    // l1_literals_ is allocated to l2_count zero bytes, state =
    // Decompressed (l2_fill_ones canonical), l3/l4 left empty.
    // Used by ComBit::from_sparse_positions({}) for the OR-accumulator
    // seed and by SparseComBit::or_many (ultra-sparse fast path) so
    // that set_bit_decompressed can poke l1_literals_ without going
    // through the Compressed→Decompressed upgrade path.
    static ComBitBtv make_decompressed_zero(size_t bit_count, size_t l2_count);

    // Raw data access (used by bitwise operators).  In Compressed state
    // the canonical L3 bytes live in l4_bits_/l3_literals_; call
    // expand_l3() to materialise the dense l3_bits_ flat byte stream.
    const uint8_t* l4_data()         const { return l4_bits_.data(); }
    const uint8_t* l3_literal_data() const { return l3_literals_.data(); }
    const uint8_t* l2_flat_data()    const { return l2_flat_.data(); }
    const uint8_t* l2_literal_data() const { return l2_literals_.data(); }
    const uint8_t* l1_literal_data() const { return l1_literals_.data(); }
    size_t         l2_literal_count() const { return l2_literal_count_; }
    size_t         l3_literal_count() const { return l3_literal_count_; }
    size_t         l4_count()         const { return l4_count_; }
    bool           l3_fill_ones()     const { return l3_fill_ones_; }

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
    // When true, L1 fill words are all-ones (0xFF...); otherwise all-zeros.
    // An L2 bit of 0 means the corresponding L1 word equals this fill value.
    bool                    l1_fill_ones_;
    // When true, L2 fill bytes are all-ones (0xFF); otherwise all-zeros (0x00).
    // An L3 bit of 0 means the corresponding L2 byte equals this fill value.
    bool                    l2_fill_ones_;
    size_t                  bit_count_;            // original bitvector length

    // --- L2: leading bitstring for L1 (1 bit per word) ---
    // L2 is always compressed via L3/l2_literals_.
    // l2_flat_ is used only as a scratch buffer during operator computation.
    size_t                  l2_count_;             // total L2 bits (= num words = bit_count_/ws)
    std::vector<uint8_t>    l2_flat_;              // scratch buffer for operators

    // --- L3: leading bitstring for L2 (1 bit per 8-bit chunk of L2) ---
    // L3 is always compressed via L4/l3_literals_.
    // l3_bits_ is used only as a scratch buffer during operator
    // computation (mirrors how l2_flat_ relates to L2).
    size_t                  l3_count_;             // total L3 bits
    std::vector<uint8_t>    l3_bits_;              // scratch buffer for operators
    std::vector<uint8_t>    l2_literals_;          // L2 literal bytes (non-zero L2 chunks)
    size_t                  l2_literal_count_;

    // --- L4: leading bitstring for L3 (1 bit per L3 byte) ---
    // When true, L3 fill bytes are all-ones (0xFF); otherwise all-zeros (0x00).
    // An L4 bit of 0 means the corresponding L3 byte equals this fill value.
    bool                    l3_fill_ones_;
    size_t                  l4_count_;             // total L4 bits (= num L3 bytes)
    std::vector<uint8_t>    l4_bits_;              // packed L4 bytes
    std::vector<uint8_t>    l3_literals_;          // L3 literal bytes (non-fill L3 chunks)
    size_t                  l3_literal_count_;

    // --- L1: literal data ---
    // l1_literals_.size() == l1_literal_count_ (1 byte per word at ws=8).
    std::vector<uint8_t>    l1_literals_;          // L1 literal word bytes
    size_t                  l1_literal_count_;

    // Rebuild flat L3 from L4 + L3 literals (for operators / expand_l2)
    std::vector<uint8_t> expand_l3() const;

    // Rebuild flat L2 from L3 + L2 literals (for decompression / scalar paths)
    std::vector<uint8_t> expand_l2() const;

    // Finalize a compressed result: shrink L1 to actual_l1_count,
    // then apply L3 compression on L2 and L4 compression on L3.
    void compact_l2_l3(size_t actual_l1_count);

    // Apply L4 compression on the populated l3_bits_ (chooses the fill
    // value that minimises stored literals), then drops l3_bits_.  Used
    // by compress() and compact_l2_l3() at the end of compaction.
    void compress_l3_to_l4();

    // Check whether the last word of the segment is a literal (L2 bit set).
    // Only meaningful when l2_count_ > 0.  Used by operator~ and popcount
    // to handle padding-bit corrections for the last partial word.
    bool is_last_word_literal() const;

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
        size_t l3_bits;            // L3 logical bits (= sum of seg.l3_count_)
        size_t l4_bits;            // L4 leading bits (= sum of seg.l4_count_)
        size_t l3_literal_bits;    // L3 stored literal bytes * 8
        size_t l2_literal_bits;    // L2 stored literal bytes * 8
        size_t l1_literal_bits;    // L1 stored literal bytes * 8
        size_t total_bits;         // L4 + L3_lit + L2_lit + L1_lit
    };

    ComBit() = default;

    // ----------------------------------------------------------------
    // Compression / Decompression
    // ----------------------------------------------------------------

    static ComBit compress(const std::vector<bool>& bits,
                           bool l1_fill_ones = false,
                           size_t segment_bits = default_segment_bits);

    // Sparse builder: build a bitmap of `num_rows` bits where only the given
    // positions are set.  Cost is O(positions.size() + segment_bits ×
    // non-empty-segments) instead of O(num_rows) — critical for value-indexed
    // bitmaps over high-cardinality columns (e.g. lineitem.l_orderkey at SF10
    // has 15M values, each producing a bitmap with ~4 set bits in 60M-bit
    // space; the std-vector<bool> path would scan 60M*15M ≈ 9e14 bits).
    //
    // `positions` does NOT need to be sorted; values must be < num_rows.
    static ComBit from_sparse_positions(const std::vector<uint32_t>& positions,
                                        size_t num_rows,
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
    ComBit& operator^=(const ComBit& other);

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
    /// fn(uint32_t word_pos, uint8_t value) for each.  Walks L4→L3→L2→L1
    /// and skips all-zero regions efficiently.
    template<typename Fn>
    void for_each_literal(Fn&& fn) const {
        size_t word_off = 0;
        for (const auto& seg : segments_) {
            const uint8_t* l1 = seg.l1_literal_data();
            const size_t l2_total = seg.l2_count();

            // All-zero segment short-circuit: produces nothing.
            if (seg.is_all_zero()) {
                word_off += l2_total;
                continue;
            }

            // Decompressed segment: l1 holds all l2_count words, walk it directly.
            if (seg.state() == ComBitBtv::State::Decompressed) {
                size_t i = 0;
#ifdef __AVX512BW__
                for (; i + 64 <= l2_total; i += 64) {
                    __m512i chunk = _mm512_loadu_si512(l1 + i);
                    uint64_t nz = static_cast<uint64_t>(
                        _mm512_test_epi8_mask(chunk, chunk));
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

            // Compressed segment: walk L4 → L3 → L2 → L1.
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
                // AVX-512 main path.  When l3_fill==0 (sparse, the common
                // case) every non-literal L3 byte is zero, so we can walk
                // the 8-bytes-of-L4 mask directly via bitscan — no SIMD
                // expand-load, no spill buffer.  When l3_fill!=0 (rare
                // dense-fill) we fall back to expand-load + test for zero
                // regions.  Both branches consume the same L4/L3-literal
                // streams; the split is purely a per-segment performance
                // shortcut for sparse data (Path B / OR_many).
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

                            uint32_t tmp = mask;
                            while (tmp) {
                                int bit = _tzcnt_u32(tmp);
                                tmp &= tmp - 1;
                                uint8_t val = l1[l1_off++];
                                if (val != 0)
                                    fn(static_cast<uint32_t>(word_off + base_w + bit), val);
                            }
                        }
                    };

                    if (l3_fill == 0) {
                        // Sparse: only literal L3 bytes are non-zero.
                        while (l4_mask) {
                            size_t bidx = static_cast<size_t>(__builtin_ctzll(l4_mask));
                            l4_mask &= l4_mask - 1;
                            walk_l3_byte(l3_base + bidx, l3_lits[l3_lit_off++]);
                        }
                        continue;
                    }

                    // Dense fill (l3_fill != 0): expand-load + test.
                    const __m512i l3_fill_vec =
                        _mm512_set1_epi8(static_cast<char>(l3_fill));
                    __m512i l3v = _mm512_mask_expandloadu_epi8(l3_fill_vec,
                        static_cast<__mmask64>(l4_mask), l3_lits + l3_lit_off);
                    l3_lit_off += __builtin_popcountll(l4_mask);
                    if (chunk < 64) {
                        __mmask64 valid = static_cast<__mmask64>(
                            (uint64_t(1) << chunk) - 1);
                        l3v = _mm512_maskz_mov_epi8(valid, l3v);
                    }
                    uint64_t nz = static_cast<uint64_t>(
                        _mm512_test_epi8_mask(l3v, l3v));
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
            // Scalar fallback (fill!=0 or l2_fill_literals; or no AVX-512).
            // Stream-decode each L3 byte once from L4, reuse for its 8 bits.
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
                uint8_t l2_byte = l3_is_lit ? l2_lits[l2_lit++]
                                            : (seg.l2_fill_ones() ? 0xFF : 0x00);
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

    // Bulk-scatter a sorted list of set positions into Decompressed segments.
    // Pre: every segment that any position falls into is in Decompressed
    // state (this is the canonical state of `from_sparse_positions({})`
    // seeds used as Q5/Q3 OR accumulators).  Bypasses ComBitBtv's L1/L2/L3/L4
    // hierarchy entirely — for ultra-sparse per-key bitmaps (orderkey at
    // SF10: ~4 set bits in 60M rows) this is the fast path that lets
    // ComBit's per-key memory rival CRoaring's array container without
    // paying ~216 bytes/segment of struct overhead.
    void scatter_or_decompressed(const uint32_t* positions, size_t n) {
        const uint32_t seg_bits = static_cast<uint32_t>(segment_bits_);
        for (size_t i = 0; i < n; i++) {
            uint32_t p = positions[i];
            uint32_t seg = p / seg_bits;
            uint32_t in  = p - seg * seg_bits;
            segments_[seg].set_bit_decompressed(in);
        }
    }

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

// =============================================================================
// SparseComBit — sparse-storage variant of ComBit for "per-value indexed
// bitmap" workloads.  Only non-empty segments are stored; empty segments
// are implicit (l1_fill_ones=false).
//
// Use case: rabit-style per-value index over a high-cardinality column.
// For lineitem.l_orderkey at SF10 (15M unique values, ~4 set bits per value
// in a 60M-bit bitmap) the regular dense-storage ComBit would need 2.8 TB
// of segment metadata; SparseComBit fits the same workload in ~10 GB.
//
// API is intentionally narrow — built once via from_positions(), then OR'd
// into a regular dense ComBit to drive Q1/Q5/Q6 multi-OR pattern.
// =============================================================================

class SparseComBit {
public:
    SparseComBit() = default;

    // Threshold below which a key's whole bitmap is stored as a raw
    // sorted uint32_t position list (the "ultra-sparse" path) instead
    // of the L1/L2/L3/L4 ComBitBtv hierarchy.  At SF10, FK columns
    // average ~4 set bits per key (l_orderkey 60M/15M=4, l_partkey
    // 60M/2M=30) — well under this threshold — so almost all per-key
    // bitmaps avoid ComBitBtv's 216-byte-per-segment struct overhead.
    // This is what brings ComBit per-key footprint into CRoaring's
    // array-container league.  The threshold is set by the break-even
    // point where ComBitBtv hierarchy starts saving more bytes than
    // raw positions cost: ~64 set bits at seg_bits=4096.
    static constexpr size_t ULTRA_SPARSE_THRESHOLD = 64;

    // Build a SparseComBit of `num_rows` bits where only `positions` are set.
    // Cost: O(positions.size() + segment_bits × non_empty_segments).
    static SparseComBit from_positions(const std::vector<uint32_t>& positions,
                                       size_t num_rows,
                                       size_t segment_bits = ComBit::default_segment_bits);

    // True iff this SparseComBit is using the ultra-sparse storage
    // mode (raw uint32_t position list, no ComBitBtv segments).
    bool is_ultra_sparse() const { return !ultra_positions_.empty() || (num_set_bits_ <= ULTRA_SPARSE_THRESHOLD && seg_indices_.empty()); }
    const std::vector<uint32_t>& ultra_positions() const { return ultra_positions_; }

    // OR this sparse bitmap into a dense ComBit (in-place).
    // Pre: dst.bit_count() == this->bit_count() and dst.segment_bits() ==
    // this->segment_bits(); dst's segments_ is dense and at least one
    // segment per logical slot.
    void apply_or_to(ComBit& dst) const;

    // K-way OR over multiple SparseComBit's into a single ComBit result.
    // Equivalent to building an empty ComBit and calling apply_or_to for
    // each input, but does scatter-OR per segment in one pass — avoids
    // the per-pairwise ComBitBtv |= overhead that dominates Q5/Q1/Q6
    // multi-OR phases.  Counterpart of CRR's fastunion / EWAH's
    // fast_logicalor.
    static ComBit or_many(size_t count, const SparseComBit** sparses,
                          size_t num_rows, size_t segment_bits);

    size_t bit_count()    const { return bit_count_; }
    size_t segment_bits() const { return segment_bits_; }
    size_t num_set_bits() const { return num_set_bits_; }
    size_t num_non_empty_segments() const { return seg_indices_.size(); }

    // Read-only access to the per-segment storage (used by or_many).
    const std::vector<uint32_t>&  seg_indices() const { return seg_indices_; }
    const std::vector<ComBitBtv>& seg_data()    const { return seg_data_; }

    // Approximate memory footprint in bytes.
    size_t storage_bytes() const;

private:
    size_t bit_count_    = 0;
    size_t segment_bits_ = ComBit::default_segment_bits;
    size_t num_set_bits_ = 0;
    // Ultra-sparse path: when num_set_bits_ <= ULTRA_SPARSE_THRESHOLD,
    // `ultra_positions_` holds the sorted set positions and seg_indices_/
    // seg_data_ are empty.  Avoids paying ~216 bytes/segment of
    // ComBitBtv struct overhead for per-key bitmaps that have only a
    // few bits set.
    std::vector<uint32_t>  ultra_positions_;
    // Hierarchy path: per-segment ComBitBtv storage.
    std::vector<uint32_t>  seg_indices_;  // sorted ascending
    std::vector<ComBitBtv> seg_data_;     // parallel to seg_indices_
};

#endif // COMBIT_H
