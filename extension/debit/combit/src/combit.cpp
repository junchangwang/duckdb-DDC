#include "combit.h"

#include <atomic>
#include <cstdlib>
#include <map>
#include <stdexcept>

// Global option: when true, operator results are compressed directly.
bool combit_compress_results = false;

// Process-wide ComBit load accumulator.  ComBit::deserialize() adds
// each loaded bitmap's size_breakdown to g_load_stats; on process exit
// we print one summary line which bench_suite.py parses per Q.
// L3/L4/total are L4-form sizes, so total = L1 + L2 + L3 + L4.
// Set COMBIT_NO_BREAKDOWN=1 to silence.
namespace {

struct CombitLoadStats {
    std::atomic<size_t> n     {0};
    std::atomic<size_t> l1    {0};
    std::atomic<size_t> l2    {0};
    std::atomic<size_t> l3    {0};
    std::atomic<size_t> l4    {0};
    std::atomic<size_t> total {0};

    void add(const ComBit& cb) {
        auto sb = cb.size_breakdown();
        size_t b1 = (sb.l1_literal_bits + 7) / 8;
        size_t b2 = (sb.l2_literal_bits + 7) / 8;
        size_t b3 = (sb.l3_literal_bits + 7) / 8;
        size_t b4 = (sb.l4_bits         + 7) / 8;
        l1    += b1;
        l2    += b2;
        l3    += b3;
        l4    += b4;
        total += b1 + b2 + b3 + b4;
        n     += 1;
    }
};

CombitLoadStats g_load_stats;

__attribute__((destructor))
void combit_print_load_breakdown() {
    if (g_load_stats.n.load() == 0) return;
    const char* off = std::getenv("COMBIT_NO_BREAKDOWN");
    if (off && *off && off[0] != '0') return;

    const double inv_mib = 1.0 / (1024.0 * 1024.0);
    // fprintf, not std::cout: iostream may already be torn down here.
    std::fprintf(stdout,
        "  [Breakdown] ComBit   L1=%.2f MiB  L2=%.2f MiB  L3=%.2f MiB"
        "  L4=%.2f MiB  total=%.2f MiB\n",
        g_load_stats.l1.load()    * inv_mib,
        g_load_stats.l2.load()    * inv_mib,
        g_load_stats.l3.load()    * inv_mib,
        g_load_stats.l4.load()    * inv_mib,
        g_load_stats.total.load() * inv_mib);
    std::fflush(stdout);
}

} // namespace

// ====================================================================
// ComBitBtv member function definitions (3-level: L3/L2/L1)
// ====================================================================

// ----------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------

ComBitBtv::ComBitBtv(bool l1_fill_ones, bool l2_fill_ones, State state)
    : state_(state),
      l1_fill_ones_(l1_fill_ones), l2_fill_ones_(l2_fill_ones),
      bit_count_(0),
      l2_count_(0), l3_count_(0), l2_literal_count_(0),
      l3_fill_ones_(false),
      l4_count_(0), l3_literal_count_(0),
      l1_literal_count_(0) {}

ComBitBtv
ComBitBtv::make_all_fill(size_t bit_count, size_t l2_count, bool l1_fill_ones) {
    ComBitBtv s(l1_fill_ones);
    s.bit_count_ = bit_count;
    s.l2_count_  = l2_count;
    s.l3_count_  = (l2_count + 7) / 8;
    return s;
}

// ----------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------

uint64_t ComBitBtv::get_literal(size_t idx) const {
    // One byte per word at ws=8; the returned uint64 is zero-extended.
    return l1_literals_[idx];
}

/// Rebuild the flat L3 byte array from L4 + l3_literals_.  In Decompressed
/// state operators populate l3_bits_ directly as scratch, so we just
/// return that buffer.  When l3_bits_ is empty for a Decompressed segment
/// (production AVX-512 paths skip the alloc), return the canonical
/// all-zero L3 of the correct size.
std::vector<uint8_t>
ComBitBtv::expand_l3() const {
    if (state_ != State::Compressed) {
        size_t expected_bytes = (l3_count_ + 7) / 8;
        if (l3_bits_.size() == expected_bytes)
            return l3_bits_;
        return std::vector<uint8_t>(expected_bytes, 0x00);
    }

    size_t l3_byte_count = (l3_count_ + 7) / 8;
    uint8_t l3_fill_val = l3_fill_ones_ ? 0xFF : 0x00;
    std::vector<uint8_t> l3(l3_byte_count, l3_fill_val);
    size_t lit_idx = 0;
    for (size_t i = 0; i < l4_count_; i++) {
        uint8_t l4_byte = l4_bits_[i / 8];
        bool is_literal = (l4_byte >> (i % 8)) & 1;
        if (is_literal) {
            if (i < l3_byte_count)
                l3[i] = l3_literals_[lit_idx++];
        }
    }
    return l3;
}

/// Rebuild the flat L2 byte array from L3 + l2_literals_.
std::vector<uint8_t>
ComBitBtv::expand_l2() const {
    size_t l2_byte_count = (l2_count_ + 7) / 8;
    uint8_t l2_fill_val = l2_fill_ones_ ? 0xFF : 0x00;
    std::vector<uint8_t> l2(l2_byte_count, l2_fill_val);
    auto l3 = expand_l3();
    size_t lit_idx = 0;
    for (size_t i = 0; i < l3_count_; i++) {
        uint8_t l3_byte = l3[i / 8];
        bool is_literal = (l3_byte >> (i % 8)) & 1;
        if (is_literal) {
            if (i < l2_byte_count)
                l2[i] = l2_literals_[lit_idx++];
        }
    }
    return l2;
}

// ----------------------------------------------------------------
// is_last_word_literal — check L2 bit for the last word
// ----------------------------------------------------------------

bool
ComBitBtv::is_last_word_literal() const {
    size_t last_word = l2_count_ - 1;
    size_t l2_byte_idx = last_word / 8;
    size_t l3_byte_pos = l2_byte_idx / 8;
    size_t l3_bit_pos  = l2_byte_idx % 8;
    auto l3 = expand_l3();
    bool l3_lit = (l3[l3_byte_pos] >> l3_bit_pos) & 1;
    uint8_t l2_byte;
    if (l3_lit) {
        size_t l2_lit_idx = 0;
        for (size_t b = 0; b < l3_byte_pos; b++)
            l2_lit_idx += __builtin_popcount(l3[b]);
        if (l3_bit_pos > 0)
            l2_lit_idx += __builtin_popcount(
                l3[l3_byte_pos] & ((1u << l3_bit_pos) - 1));
        l2_byte = l2_literals_[l2_lit_idx];
    } else {
        l2_byte = l2_fill_ones_ ? 0xFF : 0x00;
    }
    return (l2_byte >> (last_word % 8)) & 1;
}

// ----------------------------------------------------------------
// Finalize a compressed operator result
// ----------------------------------------------------------------

void
ComBitBtv::compact_l2_l3(size_t actual_l1_count) {
    l1_literals_.resize(actual_l1_count);
    l1_literal_count_ = actual_l1_count;

    // Always build L3 compression on L2.
    size_t l2_byte_count = (l2_count_ + 7) / 8;
    size_t l2_nonzero = 0;
    size_t l2_non_ff = 0;
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat_[i] != 0x00) l2_nonzero++;
        if (l2_flat_[i] != 0xFF) l2_non_ff++;
    }

    // Choose the L2 fill value that minimizes stored literals.
    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;

    size_t l3_byte_count = (l2_byte_count + 7) / 8;

    l2_fill_ones_ = best_l2_fill_lit;
    l3_count_ = l2_byte_count;
    l3_bits_.assign(l3_byte_count, 0);
    l2_literals_.clear();
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat_[i] != l2_fill_val) {
            l3_bits_[i / 8] |= uint8_t(1) << (i % 8);
            l2_literals_.push_back(l2_flat_[i]);
        }
    }
    l2_literal_count_ = l2_literals_.size();
    l2_flat_.clear();

    // Build L4 compression on L3, drop l3_bits_.
    compress_l3_to_l4();

    state_ = State::Compressed;
}

// ----------------------------------------------------------------
// L4 compression on the populated l3_bits_
// ----------------------------------------------------------------

void
ComBitBtv::compress_l3_to_l4() {
    size_t l3_byte_count = l3_bits_.size();
    size_t l3_nonzero = 0;
    size_t l3_non_ff  = 0;
    for (size_t i = 0; i < l3_byte_count; i++) {
        if (l3_bits_[i] != 0x00) l3_nonzero++;
        if (l3_bits_[i] != 0xFF) l3_non_ff++;
    }

    // Choose the L3 fill value that minimizes stored literals.
    bool best_l3_fill_lit = (l3_non_ff < l3_nonzero);
    uint8_t l3_fill_val = best_l3_fill_lit ? 0xFF : 0x00;

    size_t l4_byte_count = (l3_byte_count + 7) / 8;

    l3_fill_ones_ = best_l3_fill_lit;
    l4_count_ = l3_byte_count;
    l4_bits_.assign(l4_byte_count, 0);
    l3_literals_.clear();
    for (size_t i = 0; i < l3_byte_count; i++) {
        if (l3_bits_[i] != l3_fill_val) {
            l4_bits_[i / 8] |= uint8_t(1) << (i % 8);
            l3_literals_.push_back(l3_bits_[i]);
        }
    }
    l3_literal_count_ = l3_literals_.size();

    // Drop l3_bits_; expand_l3() rebuilds it on demand.
    l3_bits_.clear();
    l3_bits_.shrink_to_fit();
    l3_literals_.shrink_to_fit();
}

// ----------------------------------------------------------------
// Post-operation in-place compression of expanded segments
// ----------------------------------------------------------------

void
ComBitBtv::compact_expanded() {
    // Only works on fully expanded (Decompressed) segments.
    assert(state_ == State::Decompressed);
    if (l1_literal_count_ != l2_count_) return;

    // This function assumes fill = 0x00 (l1_fill_ones_ = false),
    // since it identifies literals as non-zero words.
    assert(!l1_fill_ones_);

    const size_t total_words = l2_count_;
    size_t l2_byte_count = (total_words + 7) / 8;

    // Reset L2 to all-zero (fill = not-literal).
    l2_flat_.assign(l2_byte_count, 0x00);

    size_t r_off = 0;
    size_t w = 0;

#ifdef __AVX512BW__
    // Process 64 words (= 8 L2 bytes) at a time.
    for (; w + 64 <= total_words; w += 64) {
        __m512i chunk = _mm512_loadu_si512(l1_literals_.data() + w);
        __mmask64 nz = _mm512_test_epi8_mask(chunk, chunk);
        if (nz == 0) continue;

        // Write 8 L2 bytes from the non-zero mask (little-endian layout matches).
        uint64_t nz64 = static_cast<uint64_t>(nz);
        memcpy(l2_flat_.data() + w / 8, &nz64, 8);

        // Compress-store non-zero L1 words (writes behind reads => safe).
        _mm512_mask_compressstoreu_epi8(l1_literals_.data() + r_off, nz, chunk);
        r_off += __builtin_popcountll(nz64);
    }
#endif

    // Scalar tail.
    for (; w < total_words; w++) {
        if (l1_literals_[w] != 0x00) {
            l2_flat_[w / 8] |= uint8_t(1) << (w % 8);
            l1_literals_[r_off++] = l1_literals_[w];
        }
    }

    // Apply L3 compression on L2.
    compact_l2_l3(r_off);
}

// ----------------------------------------------------------------
// Compression
// ----------------------------------------------------------------

// Sparse fast path: build a ComBitBtv directly from a sorted list of set
// positions, avoiding the O(seg_bits) scan that compress() does.  For
// per-value index workloads (1-4 set bits per segment) this is ~70× faster.
//
// Builds the same structural result as compress(positions_to_vector_bool):
//   * l1_literals_   = one byte per word that has set bits (MSB-first packing)
//   * l2_flat        = bit set per word_idx that is literal
//   * Then builds L3 + L4 via the same density-driven fill choice as compress().
ComBitBtv
ComBitBtv::compress_sparse_segment(const std::vector<uint16_t>& sorted_positions,
                                   size_t seg_bits, bool l1_fill_ones) {
    if (l1_fill_ones) {
        // Dense-fill case is rare in our usage; fall back to compress().
        std::vector<bool> bits(seg_bits, true);
        for (uint16_t p : sorted_positions) bits[p] = false;
        return compress(bits, true);
    }

    ComBitBtv result(/*l1_fill_ones=*/false);
    result.bit_count_ = seg_bits;
    size_t num_words = (seg_bits + 7) / 8;
    if (num_words == 0) return result;
    result.l2_count_ = num_words;

    size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);

    // Group consecutive sorted positions in the same word.  Each loop
    // iteration consumes one word's worth of positions.
    size_t i = 0;
    while (i < sorted_positions.size()) {
        size_t word_idx = sorted_positions[i] / 8;
        uint8_t word = 0;
        while (i < sorted_positions.size() &&
               (size_t(sorted_positions[i] / 8)) == word_idx) {
            uint16_t bit_in_byte = sorted_positions[i] % 8;
            word |= uint8_t(1) << (7 - bit_in_byte);  // MSB-first (matches compress())
            i++;
        }
        // word != 0 since at least one position contributed; store as literal
        l2_flat[word_idx / 8] |= uint8_t(1) << (word_idx % 8);
        result.l1_literals_.push_back(word);
    }
    result.l1_literal_count_ = result.l1_literals_.size();

    // L3 from L2: same density-driven fill choice as compress().
    size_t l2_nonzero = 0, l2_non_ff = 0;
    for (size_t k = 0; k < l2_byte_count; k++) {
        if (l2_flat[k] != 0x00) l2_nonzero++;
        if (l2_flat[k] != 0xFF) l2_non_ff++;
    }
    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;
    size_t l3_byte_count = (l2_byte_count + 7) / 8;

    result.l2_fill_ones_ = best_l2_fill_lit;
    result.l3_count_     = l2_byte_count;
    result.l3_bits_.assign(l3_byte_count, 0);
    for (size_t k = 0; k < l2_byte_count; k++) {
        if (l2_flat[k] != l2_fill_val) {
            result.l3_bits_[k / 8] |= uint8_t(1) << (k % 8);
            result.l2_literals_.push_back(l2_flat[k]);
        }
    }
    result.l2_literal_count_ = result.l2_literals_.size();

    // L4 from L3 (existing helper).
    result.compress_l3_to_l4();

    result.l1_literals_.shrink_to_fit();
    result.l2_literals_.shrink_to_fit();
    return result;
}

ComBitBtv
ComBitBtv::compress(const std::vector<bool>& bits, bool l1_fill_ones) {
    ComBitBtv result(l1_fill_ones);
    result.bit_count_ = bits.size();

    size_t num_words = (bits.size() + 7) / 8;
    if (num_words == 0) return result;

    result.l2_count_ = num_words;

    // Step 1: Build flat L2 (one bit per word, packed into bytes)
    // and collect L1 literals as packed byte stream.
    size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);

    const uint8_t fill_byte = l1_fill_ones ? 0xFF : 0x00;

    // No reserve here: for sparse data, num_words/4 over-allocates by 40×
    // (only set words become literals).  push_back's geometric growth costs
    // ~log(literals) reallocs; cheaper than reserving for the worst case.
    for (size_t i = 0; i < num_words; i++) {
        // Read 8 bits (MSB-first within word) directly into a byte.
        uint8_t word = 0;
        size_t start = i * 8;
        for (size_t b = 0; b < 8 && start + b < bits.size(); b++) {
            if (bits[start + b])
                word |= uint8_t(1) << (7 - b);
        }
        if (word != fill_byte) {
            l2_flat[i / 8] |= uint8_t(1) << (i % 8);
            result.l1_literals_.push_back(word);
        }
    }
    result.l1_literal_count_ = result.l1_literals_.size();

    // Step 2: Build L3 compression on L2.
    size_t l2_nonzero = 0;
    size_t l2_non_ff  = 0;
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != 0x00) l2_nonzero++;
        if (l2_flat[i] != 0xFF) l2_non_ff++;
    }

    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;

    size_t l3_byte_count = (l2_byte_count + 7) / 8;

    result.l2_fill_ones_ = best_l2_fill_lit;
    result.l3_count_ = l2_byte_count;

    result.l3_bits_.assign(l3_byte_count, 0);
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != l2_fill_val) {
            result.l3_bits_[i / 8] |= uint8_t(1) << (i % 8);
            result.l2_literals_.push_back(l2_flat[i]);
        }
    }
    result.l2_literal_count_ = result.l2_literals_.size();

    // Step 3: Build L4 compression on L3, drop l3_bits_.
    result.compress_l3_to_l4();

    // Trim trailing capacity so 1000s of small per-day bitmaps don't
    // pollute the heap with unused buffer slack (matters for OR_many
    // walks that traverse many segments back-to-back).
    result.l1_literals_.shrink_to_fit();
    result.l2_literals_.shrink_to_fit();

    return result;
}

// ----------------------------------------------------------------
// Decompression
// ----------------------------------------------------------------

std::vector<bool>
ComBitBtv::decompress() const {
    assert(state_ != State::Uncompressed);
    std::vector<bool> result;
    result.reserve(bit_count_);

    // Reconstruct flat L2.
    auto l2 = expand_l2();

    size_t lit_idx = 0;
    for (size_t i = 0; i < l2_count_; i++) {
        uint8_t l2_byte = l2[i / 8];
        bool is_literal = (l2_byte >> (i % 8)) & 1;
        if (!is_literal) {
            for (unsigned b = 0; b < 8; b++)
                result.push_back(l1_fill_ones_);
        } else {
            uint8_t word = l1_literals_[lit_idx++];
            for (unsigned b = 0; b < 8; b++)
                result.push_back((word >> (7 - b)) & 1);
        }
    }

    result.resize(bit_count_);
    return result;
}

// ----------------------------------------------------------------
// Convenience constructors
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::from_string(const std::string& bitstring, bool l1_fill_ones) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, l1_fill_ones);
}

std::string
ComBitBtv::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size() + bits.size() / 8);
    for (size_t i = 0; i < bits.size(); i++) {
        if (i > 0 && i % 8 == 0) s += ' ';
        s += bits[i] ? '1' : '0';
    }
    return s;
}

// ----------------------------------------------------------------
// operator~
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator~() const {
    if (bit_count_ == 0) return ComBitBtv();
    assert(state_ != State::Uncompressed);

    ComBitBtv result = *this;
    result.l1_fill_ones_ = !l1_fill_ones_;

    // XOR every byte of L1 with 0xFF — flips every bit in place.
    // L2/L3 structure is unchanged: fill↔literal classification is
    // preserved because (v != old_fill) ⇔ (v^0xFF != new_fill).
    // The byte stream is invariant of word_size, so the XOR loop is too.
    uint8_t* data = result.l1_literals_.data();
    size_t n = result.l1_literals_.size();   // total bytes

#ifdef __AVX512F__
    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(data + i);
        _mm512_storeu_si512(data + i, _mm512_xor_si512(v, ones));
    }
    for (; i < n; i++)
        data[i] ^= 0xFF;
#else
    for (size_t i = 0; i < n; i++)
        data[i] ^= 0xFF;
#endif

    // Padding correction.  compress() zero-pads the trailing bits of
    // the last literal word; the XOR flipped them to 1.  Re-mask them.
    if (bit_count_ % 8 != 0 && result.l1_literal_count_ > 0
        && is_last_word_literal()) {
        size_t valid_bits = bit_count_ % 8;           // 1..7 valid bits
        size_t byte_off = result.l1_literal_count_ - 1;
        result.l1_literals_[byte_off] &=
            static_cast<uint8_t>(0xFF << (8 - valid_bits));
    }

    return result;
}

// ----------------------------------------------------------------
// Queries
// ----------------------------------------------------------------

size_t
ComBitBtv::popcount() const {
    if (l2_count_ == 0) return 0;
    assert(state_ != State::Uncompressed);

    // Fill contribution: each fill word has 8 bits of the fill value.
    size_t fill_count = l2_count_ - l1_literal_count_;
    size_t count = l1_fill_ones_ ? fill_count * 8 : 0;

    // Literal contribution: VPOPCNTDQ on packed l1_literals_ byte stream.
    // popcount over bytes is invariant of word boundaries, so this works
    // for any ws.
    const uint8_t* data = l1_literals_.data();
    size_t n = l1_literals_.size();          // total bytes
    size_t i = 0;

#ifdef __AVX512VPOPCNTDQ__
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= n; i += 64) {
        __m512i chunk = _mm512_loadu_si512(data + i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(chunk));
    }
    count += _mm512_reduce_add_epi64(acc);
#endif

    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, data + i, 8);
        count += __builtin_popcountll(w);
    }
    for (; i < n; i++)
        count += __builtin_popcount(data[i]);

    // Adjust for the last partial word when bit_count_ is not a multiple
    // of 8.  Literal padding bits are 0 by convention; only fill words
    // with fill_ones=true overcounted (8 instead of bit_count_ % 8).
    if (bit_count_ % 8 != 0) {
        size_t extra_bits = 8 - (bit_count_ % 8);
        if (l1_fill_ones_ && !is_last_word_literal())
            count -= extra_bits;
    }

    return count;
}

std::vector<size_t>
ComBitBtv::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i]) pos.push_back(i);
    }
    return pos;
}

// ----------------------------------------------------------------
// Size / statistics
// ----------------------------------------------------------------

ComBitBtv::SizeBreakdown
ComBitBtv::size_breakdown() const {
    SizeBreakdown sb{};
    sb.l3_bits         = l3_count_;
    sb.l4_bits         = l4_count_;
    sb.l3_literal_bits = l3_literal_count_ * 8;
    sb.l2_literal_bits = l2_literal_count_ * 8;
    sb.l1_literal_bits = l1_literal_count_ * 8;
    sb.total_bits      = sb.l4_bits + sb.l3_literal_bits
                       + sb.l2_literal_bits + sb.l1_literal_bits;
    return sb;
}

double
ComBitBtv::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

// ----------------------------------------------------------------
// num_fills
// ----------------------------------------------------------------

size_t
ComBitBtv::num_fills() const {
    return l2_count_ - l1_literal_count_;
}

// ----------------------------------------------------------------
// Debug printing
// ----------------------------------------------------------------

void
ComBitBtv::print(std::ostream& os) const {
    os << "ComBitBtv compressed bitvector (4-level):\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  Word size:     8 bits\n";
    os << "  L2 count: " << l2_count_ << " (words)\n";
    os << "  L3 count: " << l3_count_ << " bits\n";
    os << "  L4 count: " << l4_count_ << " bits\n";
    os << "  L3 literals: " << l3_literal_count_ << " bytes\n";
    os << "  L2 literals: " << l2_literal_count_ << " bytes\n";
    os << "  L1 literals: " << l1_literal_count_ << " words ("
       << l1_literal_count_ << " bytes)\n";

    auto sb = size_breakdown();
    os << "  Size breakdown: L4=" << sb.l4_bits
       << "  L3_lit=" << sb.l3_literal_bits
       << "  L2_lit=" << sb.l2_literal_bits
       << "  L1_lit=" << sb.l1_literal_bits
       << "  total=" << sb.total_bits << " bits\n";
    os << "  Compression ratio: " << std::fixed << std::setprecision(2)
       << compression_ratio() << "x\n";
}

// ====================================================================
// ComBit (segmented) member function definitions
// ====================================================================

// ----------------------------------------------------------------
// Compression
// ----------------------------------------------------------------

ComBit
ComBit::compress(const std::vector<bool>& bits, bool l1_fill_ones,
                 size_t segment_bits) {
    ComBit result;
    result.bit_count_ = bits.size();
    result.segment_bits_ = segment_bits;

    size_t offset = 0;
    while (offset < bits.size()) {
        size_t seg_len = std::min(segment_bits, bits.size() - offset);
        std::vector<bool> seg_bits(bits.begin() + static_cast<ptrdiff_t>(offset),
                                   bits.begin() + static_cast<ptrdiff_t>(offset + seg_len));
        result.segments_.emplace_back(
            ComBitBtv::compress(seg_bits, l1_fill_ones));
        offset += seg_len;
    }

    return result;
}

ComBit
ComBit::from_sparse_positions(const std::vector<uint32_t>& positions,
                              size_t num_rows,
                              size_t segment_bits) {
    ComBit result;
    result.bit_count_ = num_rows;
    result.segment_bits_ = segment_bits;
    const size_t num_segs = (num_rows + segment_bits - 1) / segment_bits;

    // Bucket positions by segment index.  Use a vector<vector> for hot-path
    // O(positions) bucketing; non-empty buckets get compressed individually.
    std::vector<std::vector<uint32_t>> seg_positions(num_segs);
    for (uint32_t p : positions) {
        seg_positions[p / segment_bits].push_back(p % segment_bits);
    }

    result.segments_.reserve(num_segs);
    for (size_t s = 0; s < num_segs; s++) {
        size_t seg_len = std::min(segment_bits, num_rows - s * segment_bits);
        if (seg_positions[s].empty()) {
            // Empty segment: l1_fill_ones=false, no literals, no L4 bits.
            // make_all_fill(bit_count, l2_count, fill_ones=false).
            const size_t l2_count = (seg_len + 7) / 8;
            result.segments_.push_back(
                ComBitBtv::make_all_fill(seg_len, l2_count, /*l1_fill_ones=*/false));
        } else {
            // Non-empty: build a vector<bool> JUST for this segment and
            // compress.  Cost: O(seg_len) per non-empty segment.
            std::vector<bool> seg_bits(seg_len, false);
            for (uint32_t p : seg_positions[s]) seg_bits[p] = true;
            result.segments_.emplace_back(
                ComBitBtv::compress(seg_bits, /*l1_fill_ones=*/false));
        }
    }

    return result;
}

// ----------------------------------------------------------------
// Decompression
// ----------------------------------------------------------------

std::vector<bool>
ComBit::decompress() const {
    std::vector<bool> result;
    result.reserve(bit_count_);

    for (const auto& seg : segments_) {
        auto seg_bits = seg.decompress();
        result.insert(result.end(), seg_bits.begin(), seg_bits.end());
    }

    return result;
}

// ----------------------------------------------------------------
// Convenience constructors
// ----------------------------------------------------------------

ComBit
ComBit::from_string(const std::string& bitstring, bool l1_fill_ones,
                    size_t segment_bits) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, l1_fill_ones, segment_bits);
}

std::string
ComBit::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size());
    for (size_t i = 0; i < bits.size(); i++)
        s += bits[i] ? '1' : '0';
    return s;
}

// ----------------------------------------------------------------
// operator~
// ----------------------------------------------------------------

ComBit
ComBit::operator~() const {
    ComBit result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (const auto& seg : segments_)
        result.segments_.push_back(~seg);

    return result;
}

// ----------------------------------------------------------------
// Queries
// ----------------------------------------------------------------

size_t
ComBit::popcount() const {
    size_t count = 0;
    for (const auto& seg : segments_)
        count += seg.popcount();
    return count;
}

std::vector<size_t>
ComBit::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++)
        if (bits[i]) pos.push_back(i);
    return pos;
}

// ----------------------------------------------------------------
// Size / statistics
// ----------------------------------------------------------------

ComBit::SizeBreakdown
ComBit::size_breakdown() const {
    SizeBreakdown sb{0, 0, 0, 0, 0, 0};
    for (const auto& seg : segments_) {
        auto ssb = seg.size_breakdown();
        sb.l3_bits         += ssb.l3_bits;
        sb.l2_literal_bits += ssb.l2_literal_bits;
        sb.l1_literal_bits += ssb.l1_literal_bits;
        sb.total_bits      += ssb.total_bits;
        sb.l4_bits         += ssb.l4_bits;
        sb.l3_literal_bits += ssb.l3_literal_bits;
    }
    return sb;
}

double
ComBit::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

// ----------------------------------------------------------------
// Debug printing
// ----------------------------------------------------------------

void
ComBit::print(std::ostream& os) const {
    os << "ComBit segmented bitvector:\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  Segment size:  " << segment_bits_ << " bits\n";
    os << "  Num segments:  " << segments_.size() << "\n";

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& s = segments_[i];
        os << "  Segment " << i << ": "
           << "ComBitBtv"
           << " l1_fill_ones=" << s.l1_fill_ones()
           << " bits=" << s.bit_count()
           << " fills=" << s.num_fills()
           << " literals=" << s.num_literals()
           << "\n";
    }

    auto sb = size_breakdown();
    os << "  Size breakdown: L4=" << sb.l4_bits
       << "  L3_lit=" << sb.l3_literal_bits
       << "  L2_lit=" << sb.l2_literal_bits
       << "  L1_lit=" << sb.l1_literal_bits
       << "  total=" << sb.total_bits << " bits\n";
    os << "  Compression ratio: " << std::fixed << std::setprecision(2)
       << compression_ratio() << "x\n";
}

// ====================================================================
// Serialization / Deserialization
// ====================================================================

// Helper: write POD value to stream
template<typename T>
static void write_val(std::ostream& os, T val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

// Helper: read POD value from stream
template<typename T>
static T read_val(std::istream& is) {
    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

// ----------------------------------------------------------------
// ComBitBtv::serialize / deserialize
// ----------------------------------------------------------------

// On-disk format tags.  V1 (=8) is the legacy 3-level layout from
// before the L4 work.  V2 (=9) is the current 4-level layout: L4 leading
// + L3 literal stream replaces the dense l3_bits_ buffer.
// deserialize() reads either; serialize() always writes V2.
static constexpr uint8_t COMBIT_FMT_V1 = 8;
static constexpr uint8_t COMBIT_FMT_V2 = 9;

void ComBitBtv::serialize(std::ostream& os) const {
    assert(state_ == State::Compressed);
    write_val<uint8_t>(os, COMBIT_FMT_V2);
    write_val<uint8_t>(os, l1_fill_ones_ ? 1 : 0);
    write_val<uint8_t>(os, l2_fill_ones_ ? 1 : 0);
    write_val<uint8_t>(os, l3_fill_ones_ ? 1 : 0);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, l2_count_);
    write_val<uint64_t>(os, l3_count_);

    write_val<uint64_t>(os, l4_count_);
    uint64_t l4_bytes = l4_bits_.size();
    write_val<uint64_t>(os, l4_bytes);
    if (l4_bytes > 0)
        os.write(reinterpret_cast<const char*>(l4_bits_.data()), l4_bytes);

    write_val<uint64_t>(os, l3_literal_count_);
    if (l3_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l3_literals_.data()), l3_literal_count_);

    write_val<uint64_t>(os, l2_literal_count_);
    if (l2_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l2_literals_.data()), l2_literal_count_);

    write_val<uint64_t>(os, l1_literal_count_);
    if (l1_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_literals_.data()), l1_literal_count_);
}

ComBitBtv ComBitBtv::deserialize(std::istream& is) {
    uint8_t fmt = read_val<uint8_t>(is);
    if (fmt != COMBIT_FMT_V1 && fmt != COMBIT_FMT_V2)
        throw std::runtime_error("ComBitBtv::deserialize: unknown format tag " +
                                 std::to_string(fmt));
    uint8_t fo = read_val<uint8_t>(is);
    uint8_t fl = read_val<uint8_t>(is);

    ComBitBtv btv(fo != 0, fl != 0);

    if (fmt == COMBIT_FMT_V2) {
        // 4-level layout: L4 leading + L3 literal stream.
        uint8_t fl3 = read_val<uint8_t>(is);
        btv.l3_fill_ones_ = (fl3 != 0);
        btv.bit_count_ = read_val<uint64_t>(is);
        btv.l2_count_  = read_val<uint64_t>(is);
        btv.l3_count_  = read_val<uint64_t>(is);

        btv.l4_count_ = read_val<uint64_t>(is);
        uint64_t l4_bytes = read_val<uint64_t>(is);
        btv.l4_bits_.resize(l4_bytes);
        if (l4_bytes > 0)
            is.read(reinterpret_cast<char*>(btv.l4_bits_.data()), l4_bytes);

        btv.l3_literal_count_ = read_val<uint64_t>(is);
        btv.l3_literals_.resize(btv.l3_literal_count_);
        if (btv.l3_literal_count_ > 0)
            is.read(reinterpret_cast<char*>(btv.l3_literals_.data()), btv.l3_literal_count_);
    } else {
        // V1: read the dense L3 buffer, then run L4 compression on it.
        btv.bit_count_ = read_val<uint64_t>(is);
        btv.l2_count_  = read_val<uint64_t>(is);
        btv.l3_count_  = read_val<uint64_t>(is);
        uint64_t l3_bytes = read_val<uint64_t>(is);
        btv.l3_bits_.resize(l3_bytes);
        if (l3_bytes > 0)
            is.read(reinterpret_cast<char*>(btv.l3_bits_.data()), l3_bytes);
    }

    btv.l2_literal_count_ = read_val<uint64_t>(is);
    btv.l2_literals_.resize(btv.l2_literal_count_);
    if (btv.l2_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l2_literals_.data()), btv.l2_literal_count_);

    btv.l1_literal_count_ = read_val<uint64_t>(is);
    btv.l1_literals_.resize(btv.l1_literal_count_);
    if (btv.l1_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_literals_.data()), btv.l1_literal_count_);

    if (fmt == COMBIT_FMT_V1)
        btv.compress_l3_to_l4();   // upgrades V1 → V2 in memory

    return btv;
}

// ----------------------------------------------------------------
// ComBit::serialize / deserialize
// ----------------------------------------------------------------

void ComBit::serialize(std::ostream& os) const {
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());

    for (const auto& seg : segments_)
        seg.serialize(os);
}

ComBit ComBit::deserialize(std::istream& is) {
    ComBit cb;
    cb.bit_count_ = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);

    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(ComBitBtv::deserialize(is));

    g_load_stats.add(cb);
    return cb;
}

// =============================================================================
// SparseComBit — sparse-storage variant for per-value indexing workloads.
// =============================================================================

SparseComBit
SparseComBit::from_positions(const std::vector<uint32_t>& positions,
                             size_t num_rows,
                             size_t segment_bits) {
    SparseComBit s;
    s.bit_count_    = num_rows;
    s.segment_bits_ = segment_bits;
    s.num_set_bits_ = positions.size();

    if (positions.empty()) return s;

    // Bucket positions by segment.  std::map gives sorted order for the
    // parallel arrays (operators rely on seg_indices_ being ascending).
    std::map<uint32_t, std::vector<uint32_t>> by_seg;
    for (uint32_t p : positions) {
        by_seg[static_cast<uint32_t>(p / segment_bits)]
            .push_back(static_cast<uint32_t>(p % segment_bits));
    }

    s.seg_indices_.reserve(by_seg.size());
    s.seg_data_.reserve(by_seg.size());
    std::vector<uint16_t> sorted_pos;  // reuse across segments
    for (auto& [seg_idx, local_pos] : by_seg) {
        size_t seg_off = static_cast<size_t>(seg_idx) * segment_bits;
        size_t seg_len = std::min(segment_bits, num_rows - seg_off);
        // Sort positions within the segment (uint16_t — segment_bits ≤ 64K
        // means within-segment offsets fit in 16 bits for the typical setup;
        // for larger segment_bits, we cap below).
        sorted_pos.clear();
        sorted_pos.reserve(local_pos.size());
        for (uint32_t p : local_pos) sorted_pos.push_back(static_cast<uint16_t>(p));
        std::sort(sorted_pos.begin(), sorted_pos.end());
        s.seg_indices_.push_back(seg_idx);
        s.seg_data_.emplace_back(
            ComBitBtv::compress_sparse_segment(sorted_pos, seg_len,
                                               /*l1_fill_ones=*/false));
    }
    return s;
}

void
SparseComBit::apply_or_to(ComBit& dst) const {
    assert(dst.bit_count() == bit_count_);
    assert(dst.segment_bits() == segment_bits_);
    auto& dst_segs = dst.segments();
    for (size_t i = 0; i < seg_indices_.size(); i++) {
        uint32_t idx = seg_indices_[i];
        const ComBitBtv& src = seg_data_[i];
        if (src.is_all_zero()) continue;             // 0 | dst = dst
        ComBitBtv& d = dst_segs[idx];
        if (d.is_all_zero()) {
            // Teacher's pattern equivalent: when caller seeded dst as
            // Decompressed-zero (`from_sparse_positions({})` then call
            // `apply_or_to` for each key), we want d to STAY
            // Decompressed across the whole pairwise chain.  The old
            // shortcut `d = src` adopted src's Compressed state, which
            // forced every subsequent OR through the Compressed |
            // Compressed path (allocates a fresh ComBitBtv each call).
            // Doing `d |= src` in-place when d is Decompressed-zero
            // keeps d Decompressed and lets the rest of the chain run
            // through the in-place |= path with no allocation per call.
            if (d.state() == ComBitBtv::State::Decompressed) {
                d |= src;
            } else {
                d = src;
            }
            continue;
        }
        if (d.is_all_ones()) continue;               // 1 | anything = 1
        if (src.is_all_ones()) {                     // src is fill_ones
            d = ComBitBtv::make_all_fill(src.bit_count(), src.l2_count(), true);
            continue;
        }
        // Both have data — fall back to ComBitBtv binary | (handles
        // Compressed/Decompressed combinations via cross-state shims).
        if (d.state() == ComBitBtv::State::Compressed &&
            src.state() == ComBitBtv::State::Compressed) {
            d = d | src;
        } else {
            // At least one is Decompressed — use in-place |= which accepts
            // Decompressed *this and any RHS.  Need to ensure d is Decompressed.
            if (d.state() != ComBitBtv::State::Decompressed) {
                ComBitBtv tmp = d;
                tmp |= src;
                d = std::move(tmp);
            } else {
                d |= src;
            }
        }
    }
}

size_t
SparseComBit::storage_bytes() const {
    size_t total = sizeof(*this);
    total += seg_indices_.capacity() * sizeof(uint32_t);
    total += seg_data_.capacity()    * sizeof(ComBitBtv);
    for (const auto& s : seg_data_) {
        total += s.size_breakdown().total_bits / 8;
    }
    return total;
}

ComBit
SparseComBit::or_many(size_t count, const SparseComBit** sparses,
                      size_t num_rows, size_t segment_bits) {
    ComBit dst = ComBit::from_sparse_positions({}, num_rows, segment_bits);
    if (count == 0) return dst;

    // Bucket every input segment by its output segment index.  This
    // turns 590k pairwise apply_or_to (each touching ~2 random output
    // segments) into one sequential pass per output segment — same
    // total work but vastly better cache behavior, no per-key
    // function-call overhead, and dst-segment state transition
    // (Compressed→Decompressed clone) happens at most once per output.
    size_t num_segs = (num_rows + segment_bits - 1) / segment_bits;
    std::vector<std::vector<const ComBitBtv*>> by_seg(num_segs);
    // Reserve heuristically: total input segments / num_segs * 1.5.
    size_t total_segs = 0;
    for (size_t s = 0; s < count; s++) total_segs += sparses[s]->seg_indices_.size();
    if (num_segs > 0) {
        size_t avg = total_segs / num_segs + 4;
        for (auto& v : by_seg) v.reserve(avg);
    }
    for (size_t s = 0; s < count; s++) {
        const auto& sp = *sparses[s];
        const auto& idxs = sp.seg_indices();
        const auto& data = sp.seg_data();
        for (size_t i = 0; i < idxs.size(); i++)
            by_seg[idxs[i]].push_back(&data[i]);
    }

    auto& dst_segs = dst.segments();
    for (uint32_t seg = 0; seg < num_segs; seg++) {
        auto& srcs = by_seg[seg];
        if (srcs.empty()) continue;
        ComBitBtv& d = dst_segs[seg];
        for (const ComBitBtv* src_ptr : srcs) {
            const ComBitBtv& src = *src_ptr;
            if (src.is_all_zero()) continue;
            if (d.is_all_zero())   { d = src; continue; }
            if (d.is_all_ones())   break;
            if (src.is_all_ones()) {
                d = ComBitBtv::make_all_fill(src.bit_count(), src.l2_count(), true);
                break;
            }
            if (d.state() == ComBitBtv::State::Compressed
                && src.state() == ComBitBtv::State::Compressed) {
                d = d | src;
            } else {
                if (d.state() != ComBitBtv::State::Decompressed) {
                    ComBitBtv tmp = d;
                    tmp |= src;
                    d = std::move(tmp);
                } else {
                    d |= src;
                }
            }
        }
    }
    return dst;
}
