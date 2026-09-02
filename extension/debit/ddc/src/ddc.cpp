#include "ddc.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

bool ddc_compress_results = false;

namespace {

struct DDCLoadStats {
    std::atomic<size_t> n     {0};
    std::atomic<size_t> l1    {0};
    std::atomic<size_t> l2    {0};
    std::atomic<size_t> l3    {0};
    std::atomic<size_t> l4    {0};
    std::atomic<size_t> total {0};

    void add(const DDC& cb) {
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

DDCLoadStats g_load_stats;

__attribute__((destructor))
void ddc_print_load_breakdown() {
    if (g_load_stats.n.load() == 0) return;
    const char* off = std::getenv("DDC_NO_BREAKDOWN");
    if (off && *off && off[0] != '0') return;

    const double inv_mib = 1.0 / (1024.0 * 1024.0);

    std::fprintf(stdout,
        "  [Breakdown] DDC   L1=%.2f MiB  L2=%.2f MiB  L3=%.2f MiB"
        "  L4=%.2f MiB  total=%.2f MiB\n",
        g_load_stats.l1.load()    * inv_mib,
        g_load_stats.l2.load()    * inv_mib,
        g_load_stats.l3.load()    * inv_mib,
        g_load_stats.l4.load()    * inv_mib,
        g_load_stats.total.load() * inv_mib);
    std::fflush(stdout);
}

}

DDCBtv::DDCBtv(bool l1_fill_ones, bool l2_fill_ones, State state)
    : state_(state),
      l1_fill_ones_(l1_fill_ones), l2_fill_ones_(l2_fill_ones),
      bit_count_(0),
      l2_count_(0), l3_count_(0), l2_literal_count_(0),
      l3_fill_ones_(false),
      l4_count_(0), l3_literal_count_(0),
      l1_literal_count_(0) {}

DDCBtv
DDCBtv::make_all_fill(size_t bit_count, size_t l2_count, bool l1_fill_ones) {
    DDCBtv s(l1_fill_ones);
    s.bit_count_ = bit_count;
    s.l2_count_  = l2_count;
    s.l3_count_  = (l2_count + 7) / 8;
    return s;
}

DDCBtv
DDCBtv::make_decompressed_zero(size_t bit_count, size_t l2_count) {

    DDCBtv s( false,
                 true,
                State::Decompressed);
    s.bit_count_ = bit_count;
    s.l2_count_  = l2_count;
    s.l3_count_  = (l2_count + 7) / 8;
    s.l2_literal_count_ = 0;
    s.l1_literals_.assign(l2_count, 0x00);
    s.l1_literal_count_ = l2_count;
    return s;
}

uint64_t DDCBtv::get_literal(size_t idx) const {

    return l1_literals_[idx];
}

std::vector<uint8_t>
DDCBtv::expand_l3() const {
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

std::vector<uint8_t>
DDCBtv::expand_l2() const {
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

bool
DDCBtv::is_last_word_literal() const {
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

void
DDCBtv::compact_l2_l3(size_t actual_l1_count) {
    l1_literals_.resize(actual_l1_count);
    l1_literal_count_ = actual_l1_count;

    size_t l2_byte_count = (l2_count_ + 7) / 8;
    size_t l2_nonzero = 0;
    size_t l2_non_ff = 0;
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat_[i] != 0x00) l2_nonzero++;
        if (l2_flat_[i] != 0xFF) l2_non_ff++;
    }

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

    compress_l3_to_l4();

    state_ = State::Compressed;
}

void
DDCBtv::compress_l3_to_l4() {
    size_t l3_byte_count = l3_bits_.size();
    size_t l3_nonzero = 0;
    size_t l3_non_ff  = 0;
    for (size_t i = 0; i < l3_byte_count; i++) {
        if (l3_bits_[i] != 0x00) l3_nonzero++;
        if (l3_bits_[i] != 0xFF) l3_non_ff++;
    }

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

    l3_bits_.clear();
    l3_bits_.shrink_to_fit();
    l3_literals_.shrink_to_fit();
}

DDCBtv
DDCBtv::compress_sparse_segment(const std::vector<uint16_t>& sorted_positions,
                                   size_t seg_bits, bool l1_fill_ones) {
    if (l1_fill_ones) {

        std::vector<bool> bits(seg_bits, true);
        for (uint16_t p : sorted_positions) bits[p] = false;
        return compress(bits, true);
    }

    DDCBtv result( false);
    result.bit_count_ = seg_bits;
    size_t num_words = (seg_bits + 7) / 8;
    if (num_words == 0) return result;
    result.l2_count_ = num_words;

    // Stack layers
    size_t l2_byte_count = (num_words + 7) / 8;
    size_t l3_byte_count = (l2_byte_count + 7) / 8;
    uint8_t l2_flat[1024];
    uint8_t l3_flat[128];
    std::memset(l2_flat, 0, l2_byte_count);
    std::memset(l3_flat, 0, l3_byte_count);

    result.l1_literals_.reserve(sorted_positions.size());
    size_t i = 0;
    size_t l2_lo = l2_byte_count, l2_hi = 0;
    while (i < sorted_positions.size()) {
        size_t word_idx = sorted_positions[i] / 8;
        uint8_t word = 0;
        while (i < sorted_positions.size() &&
               (size_t(sorted_positions[i] / 8)) == word_idx) {
            uint16_t bit_in_byte = sorted_positions[i] % 8;
            word |= uint8_t(1) << (7 - bit_in_byte);
            i++;
        }

        size_t l2_idx = word_idx / 8;
        l2_flat[l2_idx] |= uint8_t(1) << (word_idx % 8);
        if (l2_idx < l2_lo) l2_lo = l2_idx;
        if (l2_idx + 1 > l2_hi) l2_hi = l2_idx + 1;
        result.l1_literals_.push_back(word);
    }
    result.l1_literal_count_ = result.l1_literals_.size();
    if (l2_lo > l2_hi) { l2_lo = 0; l2_hi = 0; }

    // Sparse range
    size_t l2_nonzero = 0, l2_non_ff = 0;
    for (size_t k = l2_lo; k < l2_hi; k++) {
        if (l2_flat[k] != 0x00) l2_nonzero++;
        if (l2_flat[k] != 0xFF) l2_non_ff++;
    }
    l2_non_ff += l2_byte_count - (l2_hi - l2_lo);
    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;

    result.l2_fill_ones_ = best_l2_fill_lit;
    result.l3_count_     = l2_byte_count;
    size_t l3_lo = l2_lo / 8, l3_hi = (l2_hi + 7) / 8;
    if (best_l2_fill_lit) { l3_lo = 0; l3_hi = l3_byte_count; }
    size_t l2_lit_count = 0;
    const size_t sc_lo = best_l2_fill_lit ? 0 : l2_lo;
    const size_t sc_hi = best_l2_fill_lit ? l2_byte_count : l2_hi;
    for (size_t k = sc_lo; k < sc_hi; k++) {
        if (l2_flat[k] != l2_fill_val) {
            l3_flat[k / 8] |= uint8_t(1) << (k % 8);
            l2_lit_count++;
        }
    }
    result.l2_literals_.reserve(l2_lit_count);
    for (size_t k = sc_lo; k < sc_hi; k++) {
        if (l2_flat[k] != l2_fill_val)
            result.l2_literals_.push_back(l2_flat[k]);
    }
    result.l2_literal_count_ = result.l2_literals_.size();

    // Inline L4
    {
        size_t l3_nonzero = 0, l3_non_ff = 0;
        for (size_t k = l3_lo; k < l3_hi; k++) {
            if (l3_flat[k] != 0x00) l3_nonzero++;
            if (l3_flat[k] != 0xFF) l3_non_ff++;
        }
        l3_non_ff += l3_byte_count - (l3_hi - l3_lo);
        bool best_l3_fill_lit = (l3_non_ff < l3_nonzero);
        uint8_t l3_fill_val = best_l3_fill_lit ? 0xFF : 0x00;
        result.l3_fill_ones_ = best_l3_fill_lit;
        result.l4_count_ = l3_byte_count;
        result.l4_bits_.assign((l3_byte_count + 7) / 8, 0);
        const size_t t_lo = best_l3_fill_lit ? 0 : l3_lo;
        const size_t t_hi = best_l3_fill_lit ? l3_byte_count : l3_hi;
        size_t l3_lits = 0;
        for (size_t k = t_lo; k < t_hi; k++)
            if (l3_flat[k] != l3_fill_val) l3_lits++;
        result.l3_literals_.reserve(l3_lits);
        for (size_t k = t_lo; k < t_hi; k++) {
            if (l3_flat[k] != l3_fill_val) {
                result.l4_bits_[k / 8] |= uint8_t(1) << (k % 8);
                result.l3_literals_.push_back(l3_flat[k]);
            }
        }
        result.l3_literal_count_ = result.l3_literals_.size();
    }
    result.state_ = State::Compressed;
    return result;
}

DDCBtv
DDCBtv::compress(const std::vector<bool>& bits, bool l1_fill_ones) {
    DDCBtv result(l1_fill_ones);
    result.bit_count_ = bits.size();

    size_t num_words = (bits.size() + 7) / 8;
    if (num_words == 0) return result;

    result.l2_count_ = num_words;

    size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);

    const uint8_t fill_byte = l1_fill_ones ? 0xFF : 0x00;

    for (size_t i = 0; i < num_words; i++) {

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

    result.compress_l3_to_l4();

    result.l1_literals_.shrink_to_fit();
    result.l2_literals_.shrink_to_fit();

    return result;
}

std::vector<bool>
DDCBtv::decompress() const {
    assert(state_ != State::Uncompressed);
    std::vector<bool> result;
    result.reserve(bit_count_);

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

DDCBtv
DDCBtv::from_string(const std::string& bitstring, bool l1_fill_ones) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, l1_fill_ones);
}

std::string
DDCBtv::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size() + bits.size() / 8);
    for (size_t i = 0; i < bits.size(); i++) {
        if (i > 0 && i % 8 == 0) s += ' ';
        s += bits[i] ? '1' : '0';
    }
    return s;
}

size_t
DDCBtv::popcount() const {
    if (l2_count_ == 0) return 0;
    assert(state_ != State::Uncompressed);

    size_t fill_count = l2_count_ - l1_literal_count_;
    size_t count = l1_fill_ones_ ? fill_count * 8 : 0;

    const uint8_t* data = l1_literals_.data();
    size_t n = l1_literals_.size();
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

    if (bit_count_ % 8 != 0) {
        size_t extra_bits = 8 - (bit_count_ % 8);
        if (l1_fill_ones_ && !is_last_word_literal())
            count -= extra_bits;
    }

    return count;
}

std::vector<size_t>
DDCBtv::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i]) pos.push_back(i);
    }
    return pos;
}

DDCBtv::SizeBreakdown
DDCBtv::size_breakdown() const {
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
DDCBtv::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

size_t
DDCBtv::num_fills() const {
    return l2_count_ - l1_literal_count_;
}

void
DDCBtv::print(std::ostream& os) const {
    os << "DDCBtv compressed bitvector (4-level):\n";
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

DDC
DDC::compress(const std::vector<bool>& bits, bool l1_fill_ones,
                 size_t segment_bits) {
    DDC result;
    result.bit_count_ = bits.size();
    result.segment_bits_ = segment_bits;

    size_t offset = 0;
    while (offset < bits.size()) {
        size_t seg_len = std::min(segment_bits, bits.size() - offset);
        std::vector<bool> seg_bits(bits.begin() + static_cast<ptrdiff_t>(offset),
                                   bits.begin() + static_cast<ptrdiff_t>(offset + seg_len));
        result.segments_.emplace_back(
            DDCBtv::compress(seg_bits, l1_fill_ones));
        offset += seg_len;
    }

    return result;
}

DDC
DDC::from_sparse_positions(const std::vector<uint32_t>& positions,
                              size_t num_rows,
                              size_t segment_bits) {
    DDC result;
    result.bit_count_ = num_rows;
    result.segment_bits_ = segment_bits;
    const size_t num_segs = (num_rows + segment_bits - 1) / segment_bits;

    std::vector<std::vector<uint32_t>> seg_positions(num_segs);
    for (uint32_t p : positions) {
        seg_positions[p / segment_bits].push_back(p % segment_bits);
    }

    result.segments_.reserve(num_segs);
    for (size_t s = 0; s < num_segs; s++) {
        size_t seg_len = std::min(segment_bits, num_rows - s * segment_bits);
        if (seg_positions[s].empty()) {

            const size_t l2_count = (seg_len + 7) / 8;
            result.segments_.push_back(
                DDCBtv::make_decompressed_zero(seg_len, l2_count));
        } else {

            std::vector<bool> seg_bits(seg_len, false);
            for (uint32_t p : seg_positions[s]) seg_bits[p] = true;
            result.segments_.emplace_back(
                DDCBtv::compress(seg_bits,  false));
        }
    }

    return result;
}

DDC
DDC::from_sparse_positions_compact(const std::vector<uint32_t>& positions,
                                   size_t num_rows, size_t segment_bits) {
    DDC result;
    result.bit_count_ = num_rows;
    result.segment_bits_ = segment_bits;
    const size_t num_segs = (num_rows + segment_bits - 1) / segment_bits;
    result.segments_.reserve(num_segs);
    size_t pi = 0;
    for (size_t s = 0; s < num_segs; s++) {
        const size_t seg_start = s * segment_bits;
        const size_t seg_len = std::min(segment_bits, num_rows - seg_start);
        const size_t l2_count = (seg_len + 7) / 8;
        const size_t pstart = pi;
        while (pi < positions.size() && positions[pi] < seg_start + seg_len) pi++;
        if (pi == pstart) {
            result.segments_.push_back(DDCBtv::make_all_fill(seg_len, l2_count, false));
        } else {
            std::vector<bool> seg_bits(seg_len, false);
            for (size_t k = pstart; k < pi; k++)
                seg_bits[positions[k] - seg_start] = true;
            result.segments_.emplace_back(DDCBtv::compress(seg_bits, false));
        }
    }
    return result;
}

std::vector<bool>
DDC::decompress() const {
    std::vector<bool> result;
    result.reserve(bit_count_);

    for (const auto& seg : segments_) {
        auto seg_bits = seg.decompress();
        result.insert(result.end(), seg_bits.begin(), seg_bits.end());
    }

    return result;
}

DDC
DDC::from_string(const std::string& bitstring, bool l1_fill_ones,
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
DDC::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size());
    for (size_t i = 0; i < bits.size(); i++)
        s += bits[i] ? '1' : '0';
    return s;
}

size_t
DDC::popcount() const {
    size_t count = 0;
    for (const auto& seg : segments_)
        count += seg.popcount();
    return count;
}

std::vector<size_t>
DDC::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++)
        if (bits[i]) pos.push_back(i);
    return pos;
}

DDC::SizeBreakdown
DDC::size_breakdown() const {
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
DDC::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

void
DDC::print(std::ostream& os) const {
    os << "DDC segmented bitvector:\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  Segment size:  " << segment_bits_ << " bits\n";
    os << "  Num segments:  " << segments_.size() << "\n";

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& s = segments_[i];
        os << "  Segment " << i << ": "
           << "DDCBtv"
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

template<typename T>
static void write_val(std::ostream& os, T val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

template<typename T>
static T read_val(std::istream& is) {
    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

static constexpr uint8_t DDC_FMT_V1 = 8;
static constexpr uint8_t DDC_FMT_V2 = 9;

void DDCBtv::serialize(std::ostream& os) const {
    assert(state_ == State::Compressed);
    write_val<uint8_t>(os, DDC_FMT_V2);
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

DDCBtv DDCBtv::deserialize(std::istream& is) {
    uint8_t fmt = read_val<uint8_t>(is);
    if (fmt != DDC_FMT_V1 && fmt != DDC_FMT_V2)
        throw std::runtime_error("DDCBtv::deserialize: unknown format tag " +
                                 std::to_string(fmt));
    uint8_t fo = read_val<uint8_t>(is);
    uint8_t fl = read_val<uint8_t>(is);

    DDCBtv btv(fo != 0, fl != 0);

    if (fmt == DDC_FMT_V2) {

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

    if (fmt == DDC_FMT_V1)
        btv.compress_l3_to_l4();

    return btv;
}

void DDC::serialize(std::ostream& os) const {
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());

    for (const auto& seg : segments_)
        seg.serialize(os);
}

DDC DDC::deserialize(std::istream& is) {
    DDC cb;
    cb.bit_count_ = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);

    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(DDCBtv::deserialize(is));

    g_load_stats.add(cb);
    return cb;
}

static constexpr uint8_t DDC_FMT_V3_MAGIC = 0xCB;

void DDCBtv::serialize_packed(std::ostream& os, bool is_last_seg) const {
    assert(state_ == State::Compressed);

    uint8_t flags = 0;
    if (l1_fill_ones_) flags |= 0x01;
    if (l2_fill_ones_) flags |= 0x02;
    if (l3_fill_ones_) flags |= 0x04;
    if (is_last_seg)   flags |= 0x08;
    write_val<uint8_t>(os, flags);

    if (is_last_seg)
        write_val<uint32_t>(os, static_cast<uint32_t>(bit_count_));

    uint32_t l4_bytes = static_cast<uint32_t>(l4_bits_.size());
    write_val<uint32_t>(os, l4_bytes);
    write_val<uint32_t>(os, static_cast<uint32_t>(l3_literal_count_));
    write_val<uint32_t>(os, static_cast<uint32_t>(l2_literal_count_));
    write_val<uint32_t>(os, static_cast<uint32_t>(l1_literal_count_));

    if (l4_bytes > 0)
        os.write(reinterpret_cast<const char*>(l4_bits_.data()), l4_bytes);
    if (l3_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l3_literals_.data()), l3_literal_count_);
    if (l2_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l2_literals_.data()), l2_literal_count_);
    if (l1_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_literals_.data()), l1_literal_count_);
}

DDCBtv DDCBtv::deserialize_packed(std::istream& is, size_t segment_bits) {
    uint8_t flags = read_val<uint8_t>(is);
    bool l1_fo  = (flags & 0x01) != 0;
    bool l2_fo  = (flags & 0x02) != 0;
    bool l3_fo  = (flags & 0x04) != 0;
    bool islast = (flags & 0x08) != 0;

    DDCBtv btv(l1_fo, l2_fo);
    btv.l3_fill_ones_ = l3_fo;

    btv.bit_count_ = islast ? read_val<uint32_t>(is) : segment_bits;
    btv.l2_count_  = (btv.bit_count_ + 7) / 8;
    btv.l3_count_  = (btv.l2_count_  + 7) / 8;
    btv.l4_count_  = (btv.l3_count_  + 7) / 8;

    uint32_t l4_bytes        = read_val<uint32_t>(is);
    btv.l3_literal_count_    = read_val<uint32_t>(is);
    btv.l2_literal_count_    = read_val<uint32_t>(is);
    btv.l1_literal_count_    = read_val<uint32_t>(is);

    btv.l4_bits_.resize(l4_bytes);
    if (l4_bytes > 0)
        is.read(reinterpret_cast<char*>(btv.l4_bits_.data()), l4_bytes);
    btv.l3_literals_.resize(btv.l3_literal_count_);
    if (btv.l3_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l3_literals_.data()), btv.l3_literal_count_);
    btv.l2_literals_.resize(btv.l2_literal_count_);
    if (btv.l2_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l2_literals_.data()), btv.l2_literal_count_);
    btv.l1_literals_.resize(btv.l1_literal_count_);
    if (btv.l1_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_literals_.data()), btv.l1_literal_count_);
    return btv;
}

void DDC::serialize_packed(std::ostream& os) const {
    write_val<uint8_t>(os, DDC_FMT_V3_MAGIC);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());
    for (size_t i = 0; i < segments_.size(); i++)
        segments_[i].serialize_packed(os,  (i + 1 == segments_.size()));
}

DDC DDC::deserialize_packed(std::istream& is) {
    uint8_t magic = read_val<uint8_t>(is);
    if (magic != DDC_FMT_V3_MAGIC)
        throw std::runtime_error("DDC::deserialize_packed: bad magic "
                                 + std::to_string(magic));
    DDC cb;
    cb.bit_count_    = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);
    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(DDCBtv::deserialize_packed(is, cb.segment_bits_));
    return cb;
}

static constexpr uint8_t DDC_FMT_V4_MAGIC = 0xC4;

void DDCBtv::serialize_v4(std::ostream& os, bool is_last_seg) const {
    assert(state_ == State::Compressed);

    uint8_t flags = 0;
    if (l1_fill_ones_)             flags |= 0x01;
    if (l2_fill_ones_)             flags |= 0x02;
    if (l3_fill_ones_)             flags |= 0x04;
    if (is_last_seg)               flags |= 0x08;
    if (l1_literal_count_ > 0)     flags |= 0x10;
    if (l2_literal_count_ > 0)     flags |= 0x20;
    if (l3_literal_count_ > 0)     flags |= 0x40;
    write_val<uint8_t>(os, flags);

    if (is_last_seg)
        write_val<uint32_t>(os, static_cast<uint32_t>(bit_count_));

    if (l3_literal_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l3_literal_count_));
    if (l2_literal_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l2_literal_count_));
    if (l1_literal_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l1_literal_count_));

    uint32_t l4_bytes = static_cast<uint32_t>(l4_bits_.size());
    if (l4_bytes > 0)
        os.write(reinterpret_cast<const char*>(l4_bits_.data()), l4_bytes);
    if (l3_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l3_literals_.data()), l3_literal_count_);
    if (l2_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l2_literals_.data()), l2_literal_count_);
    if (l1_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_literals_.data()), l1_literal_count_);
}

DDCBtv DDCBtv::deserialize_v4(std::istream& is, size_t segment_bits) {
    uint8_t flags = read_val<uint8_t>(is);
    bool l1_fo       = (flags & 0x01) != 0;
    bool l2_fo       = (flags & 0x02) != 0;
    bool l3_fo       = (flags & 0x04) != 0;
    bool islast      = (flags & 0x08) != 0;
    bool has_l1_lit  = (flags & 0x10) != 0;
    bool has_l2_lit  = (flags & 0x20) != 0;
    bool has_l3_lit  = (flags & 0x40) != 0;

    DDCBtv btv(l1_fo, l2_fo);
    btv.l3_fill_ones_ = l3_fo;

    btv.bit_count_ = islast ? read_val<uint32_t>(is) : segment_bits;
    btv.l2_count_  = (btv.bit_count_ + 7) / 8;
    btv.l3_count_  = (btv.l2_count_  + 7) / 8;
    btv.l4_count_  = (btv.l3_count_  + 7) / 8;
    size_t l4_bytes = (btv.l4_count_ + 7) / 8;

    btv.l3_literal_count_ = has_l3_lit ? read_val<uint32_t>(is) : 0;
    btv.l2_literal_count_ = has_l2_lit ? read_val<uint32_t>(is) : 0;
    btv.l1_literal_count_ = has_l1_lit ? read_val<uint32_t>(is) : 0;

    btv.l4_bits_.resize(l4_bytes);
    if (l4_bytes > 0)
        is.read(reinterpret_cast<char*>(btv.l4_bits_.data()), l4_bytes);
    btv.l3_literals_.resize(btv.l3_literal_count_);
    if (btv.l3_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l3_literals_.data()), btv.l3_literal_count_);
    btv.l2_literals_.resize(btv.l2_literal_count_);
    if (btv.l2_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l2_literals_.data()), btv.l2_literal_count_);
    btv.l1_literals_.resize(btv.l1_literal_count_);
    if (btv.l1_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_literals_.data()), btv.l1_literal_count_);
    return btv;
}

void DDC::serialize_v4(std::ostream& os) const {
    write_val<uint8_t>(os, DDC_FMT_V4_MAGIC);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());
    for (size_t i = 0; i < segments_.size(); i++)
        segments_[i].serialize_v4(os,  (i + 1 == segments_.size()));
}

DDC DDC::deserialize_v4(std::istream& is) {
    uint8_t magic = read_val<uint8_t>(is);
    if (magic != DDC_FMT_V4_MAGIC)
        throw std::runtime_error("DDC::deserialize_v4: bad magic "
                                 + std::to_string(magic));
    DDC cb;
    cb.bit_count_    = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);
    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(DDCBtv::deserialize_v4(is, cb.segment_bits_));
    return cb;
}

SparseDDC
SparseDDC::from_positions(const std::vector<uint32_t>& positions,
                             size_t num_rows,
                             size_t segment_bits, bool adaptive_l1_polarity) {
    SparseDDC s;
    s.bit_count_    = num_rows;
    s.segment_bits_ = segment_bits;
    s.num_set_bits_ = positions.size();

    if (positions.empty()) return s;

    // Sorted fastpath
    static const bool old_build = std::getenv("DDC_OLD_BUILD") != nullptr;
    if (!old_build && std::is_sorted(positions.begin(), positions.end())) {
        const size_t n = positions.size();
        size_t nseg = 0;
        for (size_t i = 0; i < n; ) {
            uint32_t seg = static_cast<uint32_t>(positions[i] / segment_bits);
            while (i < n && static_cast<uint32_t>(positions[i] / segment_bits) == seg) i++;
            nseg++;
        }
        s.seg_indices_.reserve(nseg);
        s.seg_data_.reserve(nseg);
        std::vector<uint16_t> local;
        for (size_t i = 0; i < n; ) {
            uint32_t seg_idx = static_cast<uint32_t>(positions[i] / segment_bits);
            size_t seg_off = static_cast<size_t>(seg_idx) * segment_bits;
            size_t seg_len = std::min(segment_bits, num_rows - seg_off);
            local.clear();
            while (i < n && static_cast<uint32_t>(positions[i] / segment_bits) == seg_idx)
                local.push_back(static_cast<uint16_t>(positions[i++] - seg_off));
            s.seg_indices_.push_back(seg_idx);
            if (adaptive_l1_polarity && local.size() * 2 > seg_len) {
                std::vector<uint16_t> zeros;
                zeros.reserve(seg_len - local.size());
                size_t j = 0;
                for (size_t b = 0; b < seg_len; b++) {
                    if (j < local.size() && local[j] == b) { j++; continue; }
                    zeros.push_back(static_cast<uint16_t>(b));
                }
                s.seg_data_.emplace_back(
                    DDCBtv::compress_sparse_segment(zeros, seg_len, true));
            } else {
                s.seg_data_.emplace_back(
                    DDCBtv::compress_sparse_segment(local, seg_len, false));
            }
        }
        return s;
    }

    std::map<uint32_t, std::vector<uint32_t>> by_seg;
    for (uint32_t p : positions) {
        by_seg[static_cast<uint32_t>(p / segment_bits)]
            .push_back(static_cast<uint32_t>(p % segment_bits));
    }

    s.seg_indices_.reserve(by_seg.size());
    s.seg_data_.reserve(by_seg.size());
    std::vector<uint16_t> sorted_pos;
    for (auto& [seg_idx, local_pos] : by_seg) {
        size_t seg_off = static_cast<size_t>(seg_idx) * segment_bits;
        size_t seg_len = std::min(segment_bits, num_rows - seg_off);

        sorted_pos.clear();
        sorted_pos.reserve(local_pos.size());
        for (uint32_t p : local_pos) sorted_pos.push_back(static_cast<uint16_t>(p));
        std::sort(sorted_pos.begin(), sorted_pos.end());
        s.seg_indices_.push_back(seg_idx);
        if (adaptive_l1_polarity && sorted_pos.size() * 2 > seg_len) {
            // Ones polarity
            std::vector<uint16_t> zeros;
            zeros.reserve(seg_len - sorted_pos.size());
            size_t j = 0;
            for (size_t b = 0; b < seg_len; b++) {
                if (j < sorted_pos.size() && sorted_pos[j] == b) { j++; continue; }
                zeros.push_back(static_cast<uint16_t>(b));
            }
            s.seg_data_.emplace_back(
                DDCBtv::compress_sparse_segment(zeros, seg_len, true));
        } else {
            s.seg_data_.emplace_back(
                DDCBtv::compress_sparse_segment(sorted_pos, seg_len,
                                                    false));
        }
    }
    return s;
}

void
SparseDDC::apply_or_to(DDC& dst) const {
    assert(dst.bit_count() == bit_count_);
    assert(dst.segment_bits() == segment_bits_);
    auto& dst_segs = dst.segments();
    const size_t nseg = seg_indices_.size();
    for (size_t i = 0; i < nseg; i++) {
        // Prefetch pipeline
        if (i + 2 < nseg) {
            __builtin_prefetch(reinterpret_cast<const char*>(&seg_data_[i + 2]));
            __builtin_prefetch(reinterpret_cast<const char*>(&seg_data_[i + 2]) + 128);
        }
        if (i + 1 < nseg) {
            seg_data_[i + 1].prefetch_layers();
            __builtin_prefetch(&dst_segs[seg_indices_[i + 1]]);
        }
        uint32_t idx = seg_indices_[i];
        const DDCBtv& src = seg_data_[i];
        if (src.is_all_zero()) continue;
        DDCBtv& d = dst_segs[idx];
        if (d.is_all_zero()) {

            if (d.state() == DDCBtv::State::Decompressed) {
                d |= src;
            } else {
                d = src;
            }
            continue;
        }
        if (d.is_all_ones()) continue;
        if (src.is_all_ones()) {
            // Preserve dense
            if (d.state() == DDCBtv::State::Decompressed) {
                d |= src;
            } else {
                d = DDCBtv::make_all_fill(src.bit_count(), src.l2_count(), true);
            }
            continue;
        }

        if (d.state() == DDCBtv::State::Compressed &&
            src.state() == DDCBtv::State::Compressed) {
            d = d | src;
        } else {

            if (d.state() != DDCBtv::State::Decompressed) {
                DDCBtv tmp = d;
                tmp |= src;
                d = std::move(tmp);
            } else {
                d |= src;
            }
        }
    }
}

size_t
SparseDDC::storage_bytes() const {
    size_t total = sizeof(*this);
    total += seg_indices_.capacity() * sizeof(uint32_t);
    total += seg_data_.capacity()    * sizeof(DDCBtv);
    for (const auto& s : seg_data_) {
        total += s.size_breakdown().total_bits / 8;
    }
    return total;
}

DDC
SparseDDC::or_many(size_t count, const SparseDDC** sparses,
                      size_t num_rows, size_t segment_bits) {
    DDC dst = DDC::from_sparse_positions({}, num_rows, segment_bits);
    if (count == 0) return dst;

    size_t num_segs = (num_rows + segment_bits - 1) / segment_bits;
    std::vector<std::vector<const DDCBtv*>> by_seg(num_segs);
    size_t total_segs = 0;
    for (size_t s = 0; s < count; s++) total_segs += sparses[s]->seg_indices_.size();
    if (total_segs == 0) return dst;
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
        DDCBtv& d = dst_segs[seg];
        for (const DDCBtv* src_ptr : srcs) {
            const DDCBtv& src = *src_ptr;
            if (src.is_all_zero()) continue;
            if (d.is_all_zero())   { d = src; continue; }
            if (d.is_all_ones())   break;
            if (src.is_all_ones()) {
                d = DDCBtv::make_all_fill(src.bit_count(), src.l2_count(), true);
                break;
            }
            if (d.state() == DDCBtv::State::Compressed
                && src.state() == DDCBtv::State::Compressed) {
                d = d | src;
            } else {
                if (d.state() != DDCBtv::State::Decompressed) {
                    DDCBtv tmp = d;
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
