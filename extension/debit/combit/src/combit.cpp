#include "combit.h"

#include <stdexcept>

// Global option: when true, operator results are compressed directly.
bool combit_compress_results = false;

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
      l1_literal_count_(0) {}

// ----------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------

uint64_t ComBitBtv::get_literal(size_t idx) const {
    return l1_literals_[idx];
}

uint64_t ComBitBtv::read_word_from_bits(const std::vector<bool>& bits,
                                        size_t word_idx) {
    uint64_t word = 0;
    size_t start = word_idx * 8;
    for (unsigned i = 0; i < 8 && start + i < bits.size(); i++) {
        if (bits[start + i])
            word |= uint64_t(1) << (7 - i);
    }
    return word;
}

void ComBitBtv::append_word_to_bits(std::vector<bool>& bits, uint64_t word) {
    for (int i = 7; i >= 0; i--)
        bits.push_back((word >> i) & 1);
}

/// Rebuild the flat L2 byte array from L3 + l2_literals_.
std::vector<uint8_t>
ComBitBtv::expand_l2() const {
    size_t l2_byte_count = (l2_count_ + 7) / 8;
    uint8_t l2_fill_val = l2_fill_ones_ ? 0xFF : 0x00;
    std::vector<uint8_t> l2(l2_byte_count, l2_fill_val);
    size_t lit_idx = 0;
    for (size_t i = 0; i < l3_count_; i++) {
        uint8_t l3_byte = l3_bits_[i / 8];
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
    bool l3_lit = (l3_bits_[l3_byte_pos] >> l3_bit_pos) & 1;
    uint8_t l2_byte;
    if (l3_lit) {
        size_t l2_lit_idx = 0;
        for (size_t b = 0; b < l3_byte_pos; b++)
            l2_lit_idx += __builtin_popcount(l3_bits_[b]);
        if (l3_bit_pos > 0)
            l2_lit_idx += __builtin_popcount(
                l3_bits_[l3_byte_pos] & ((1u << l3_bit_pos) - 1));
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
    state_ = State::Compressed;
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

ComBitBtv
ComBitBtv::compress(const std::vector<bool>& bits, bool l1_fill_ones) {
    ComBitBtv result(l1_fill_ones);
    result.bit_count_ = bits.size();

    size_t num_words = (bits.size() + 7) / 8;
    if (num_words == 0) return result;

    result.l2_count_ = num_words;

    // Step 1: Build flat L2 (one bit per word, packed into bytes)
    // and collect L1 literals.
    size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);

    const uint64_t fill_val = l1_fill_ones ? uint64_t(0xFF) : uint64_t(0);
    for (size_t i = 0; i < num_words; i++) {
        uint64_t word = read_word_from_bits(bits, i);
        if (word != fill_val) {
            l2_flat[i / 8] |= uint8_t(1) << (i % 8);
            result.l1_literals_.push_back(static_cast<uint8_t>(word));
        }
    }
    result.l1_literal_count_ = result.l1_literals_.size();

    // Step 2: Build L3 compression on L2.
    // Count non-zero and non-FF L2 bytes to choose best L2 fill.
    size_t l2_nonzero = 0;
    size_t l2_non_ff = 0;
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != 0x00) l2_nonzero++;
        if (l2_flat[i] != 0xFF) l2_non_ff++;
    }

    // Choose the L2 fill value that minimizes stored literals.
    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;

    size_t l3_byte_count = (l2_byte_count + 7) / 8;

    result.l2_fill_ones_ = best_l2_fill_lit;
    result.l3_count_ = l2_byte_count;  // one L3 bit per L2 byte

    result.l3_bits_.assign(l3_byte_count, 0);
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != l2_fill_val) {
            result.l3_bits_[i / 8] |= uint8_t(1) << (i % 8);
            result.l2_literals_.push_back(l2_flat[i]);
        }
    }
    result.l2_literal_count_ = result.l2_literals_.size();

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

    // Reconstruct flat L2
    auto l2 = expand_l2();

    size_t lit_idx = 0;
    for (size_t i = 0; i < l2_count_; i++) {
        uint8_t l2_byte = l2[i / 8];
        bool is_literal = (l2_byte >> (i % 8)) & 1;
        if (!is_literal) {
            for (unsigned b = 0; b < 8; b++)
                result.push_back(l1_fill_ones_);
        } else {
            append_word_to_bits(result, l1_literals_[lit_idx++]);
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

    // XOR all L1 literals with 0xFF — flips every bit in-place.
    // L2/L3 structure is unchanged: fill↔literal classification is
    // preserved because (v != old_fill) ⟺ (v^0xFF != new_fill).
    uint8_t* data = result.l1_literals_.data();
    size_t n = result.l1_literal_count_;

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

    // The compress convention zero-pads unused bits in the last word.
    // XOR flipped those padding bits to 1 — mask them back to 0.
    if (bit_count_ % 8 != 0 && n > 0 && is_last_word_literal()) {
        size_t nbits = bit_count_ % 8;
        result.l1_literals_[n - 1] &= static_cast<uint8_t>(0xFF << (8 - nbits));
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

    // Literal contribution: VPOPCNTDQ on packed l1_literals_ directly.
    const uint8_t* data = l1_literals_.data();
    size_t n = l1_literal_count_;
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

    // Adjust for the last partial word when bit_count_ is not a multiple of 8.
    // Padding bits are 0 by convention, so literal words are already correct.
    // Only fill words with fill_ones=true overcounted: 8 instead of valid bits.
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
    SizeBreakdown sb;
    sb.l3_bits = l3_count_;
    sb.l2_literal_bits = l2_literal_count_ * 8;
    sb.l1_literal_bits = l1_literal_count_ * 8;
    sb.total_bits = sb.l3_bits + sb.l2_literal_bits + sb.l1_literal_bits;
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
    os << "ComBitBtv compressed bitvector (3-level):\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  L2 count: " << l2_count_ << " (words)\n";
    os << "  L3 count: " << l3_count_ << " bits\n";
    os << "  L2 literals: " << l2_literal_count_ << " bytes\n";
    os << "  L1 literals: " << l1_literal_count_ << " bytes\n";

    auto sb = size_breakdown();
    os << "  Size breakdown: L3=" << sb.l3_bits
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
    SizeBreakdown sb{0, 0, 0, 0};
    for (const auto& seg : segments_) {
        auto ssb = seg.size_breakdown();
        sb.l3_bits += ssb.l3_bits;
        sb.l2_literal_bits += ssb.l2_literal_bits;
        sb.l1_literal_bits += ssb.l1_literal_bits;
        sb.total_bits += ssb.total_bits;
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
    os << "  Size breakdown: L3=" << sb.l3_bits
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

void ComBitBtv::serialize(std::ostream& os) const {
    assert(state_ == State::Compressed);
    write_val<uint8_t>(os, static_cast<uint8_t>(8));  // word size tag
    write_val<uint8_t>(os, l1_fill_ones_ ? 1 : 0);
    write_val<uint8_t>(os, l2_fill_ones_ ? 1 : 0);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, l2_count_);

    write_val<uint64_t>(os, l3_count_);
    uint64_t l3_bytes = l3_bits_.size();
    write_val<uint64_t>(os, l3_bytes);
    if (l3_bytes > 0)
        os.write(reinterpret_cast<const char*>(l3_bits_.data()), l3_bytes);

    write_val<uint64_t>(os, l2_literal_count_);
    if (l2_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l2_literals_.data()), l2_literal_count_);

    write_val<uint64_t>(os, l1_literal_count_);
    if (l1_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_literals_.data()), l1_literal_count_);
}

ComBitBtv ComBitBtv::deserialize(std::istream& is) {
    uint8_t ws = read_val<uint8_t>(is);
    if (ws != 8)
        throw std::runtime_error("ComBitBtv::deserialize: expected word size 8, got " +
                                 std::to_string(ws));
    uint8_t fo = read_val<uint8_t>(is);
    uint8_t fl = read_val<uint8_t>(is);

    ComBitBtv btv(fo != 0, fl != 0);
    btv.bit_count_ = read_val<uint64_t>(is);
    btv.l2_count_ = read_val<uint64_t>(is);

    btv.l3_count_ = read_val<uint64_t>(is);
    uint64_t l3_bytes = read_val<uint64_t>(is);
    btv.l3_bits_.resize(l3_bytes);
    if (l3_bytes > 0)
        is.read(reinterpret_cast<char*>(btv.l3_bits_.data()), l3_bytes);

    btv.l2_literal_count_ = read_val<uint64_t>(is);
    btv.l2_literals_.resize(btv.l2_literal_count_);
    if (btv.l2_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l2_literals_.data()), btv.l2_literal_count_);

    btv.l1_literal_count_ = read_val<uint64_t>(is);
    btv.l1_literals_.resize(btv.l1_literal_count_);
    if (btv.l1_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_literals_.data()), btv.l1_literal_count_);

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

    return cb;
}
