#include <bitset_vector.hpp>
#include <bitset_simd.hpp>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace bitset {

static inline size_t padded_words(size_t n) {
    return (n + 7) & ~size_t(7);
}

static uint64_t* alloc_words(size_t n) {
    if (n == 0) return nullptr;
    size_t pw = padded_words(n);
    size_t bytes = pw * sizeof(uint64_t);
    auto* p = static_cast<uint64_t*>(aligned_alloc(64, bytes));
    memset(p, 0, bytes);
    return p;
}

static uint64_t* alloc_words_nozero(size_t n) {
    if (n == 0) return nullptr;
    size_t pw = padded_words(n);
    size_t bytes = pw * sizeof(uint64_t);
    auto* p = static_cast<uint64_t*>(aligned_alloc(64, bytes));

    if (pw > n) memset(p + n, 0, (pw - n) * sizeof(uint64_t));
    return p;
}

BitsetVector::BitsetVector(const BitsetVector& o)
    : words_(nullptr), words_cnt_(o.words_cnt_), num_bits_(o.num_bits_) {
    if (words_cnt_ > 0) {
        size_t pw = padded_words(words_cnt_);
        size_t bytes = pw * sizeof(uint64_t);
        words_ = static_cast<uint64_t*>(aligned_alloc(64, bytes));
        memcpy(words_, o.words_, words_cnt_ * sizeof(uint64_t));
        if (pw > words_cnt_)
            memset(words_ + words_cnt_, 0, (pw - words_cnt_) * sizeof(uint64_t));
    }
}

BitsetVector& BitsetVector::operator=(const BitsetVector& o) {
    if (this == &o) return *this;
    free(words_);
    words_ = nullptr;
    words_cnt_ = o.words_cnt_;
    num_bits_  = o.num_bits_;
    if (words_cnt_ > 0) {
        size_t pw = padded_words(words_cnt_);
        size_t bytes = pw * sizeof(uint64_t);
        words_ = static_cast<uint64_t*>(aligned_alloc(64, bytes));
        memcpy(words_, o.words_, words_cnt_ * sizeof(uint64_t));
        if (pw > words_cnt_)
            memset(words_ + words_cnt_, 0, (pw - words_cnt_) * sizeof(uint64_t));
    }
    return *this;
}

BitsetVector::BitsetVector(BitsetVector&& o) noexcept
    : words_(o.words_), words_cnt_(o.words_cnt_), num_bits_(o.num_bits_) {
    o.words_     = nullptr;
    o.words_cnt_ = 0;
    o.num_bits_  = 0;
}

BitsetVector& BitsetVector::operator=(BitsetVector&& o) noexcept {
    if (this == &o) return *this;
    free(words_);
    words_     = o.words_;
    words_cnt_ = o.words_cnt_;
    num_bits_  = o.num_bits_;
    o.words_     = nullptr;
    o.words_cnt_ = 0;
    o.num_bits_  = 0;
    return *this;
}

void BitsetVector::allocate(size_t n) {
    free(words_);
    words_     = alloc_words(n);
    words_cnt_ = n;
}

void BitsetVector::allocate_nozero(size_t n) {
    free(words_);
    words_     = alloc_words_nozero(n);
    words_cnt_ = n;
}

void BitsetVector::ensure_capacity(uint64_t pos) {
    size_t need = pos / 64 + 1;
    if (need <= words_cnt_) return;
    uint64_t* old = words_;
    size_t    old_cnt = words_cnt_;
    words_     = alloc_words(need);
    words_cnt_ = need;
    if (old_cnt > 0)
        memcpy(words_, old, old_cnt * sizeof(uint64_t));
    free(old);
}

void BitsetVector::set_bit(uint64_t position) {
    ensure_capacity(position);
    words_[position / 64] |= uint64_t(1) << (position % 64);
    if (position >= num_bits_)
        num_bits_ = position + 1;
}

bool BitsetVector::get_bit(uint64_t position) const {
    size_t wi = position / 64;
    if (wi >= words_cnt_) return false;
    return (words_[wi] >> (position % 64)) & 1;
}

uint64_t BitsetVector::popcount(bool use_simd) const {
    if (words_cnt_ == 0) return 0;
    if (use_simd)
        return simd::words_popcount_simd(words_, words_cnt_);
    else
        return simd::words_popcount_scalar(words_, words_cnt_);
}

BitsetVector BitsetVector::word_or(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t na = a.words_cnt_, nb = b.words_cnt_;
    size_t max_n = std::max(na, nb);
    size_t min_n = std::min(na, nb);
    result.allocate_nozero(max_n);

    if (min_n > 0) {
        if (use_simd)
            simd::words_or_simd(a.words_, b.words_, result.words_, min_n);
        else
            simd::words_or_scalar(a.words_, b.words_, result.words_, min_n);
    }
    if (max_n > min_n) {
        const uint64_t* tail = (na > nb) ? a.words_ : b.words_;
        memcpy(result.words_ + min_n, tail + min_n, (max_n - min_n) * sizeof(uint64_t));
    }
    result.num_bits_ = std::max(a.num_bits_, b.num_bits_);
    return result;
}

BitsetVector BitsetVector::word_and(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t min_n = std::min(a.words_cnt_, b.words_cnt_);
    if (min_n == 0) return result;
    result.allocate_nozero(min_n);

    if (use_simd)
        simd::words_and_simd(a.words_, b.words_, result.words_, min_n);
    else
        simd::words_and_scalar(a.words_, b.words_, result.words_, min_n);

    result.num_bits_ = std::max(a.num_bits_, b.num_bits_);
    return result;
}

void BitsetVector::word_and_inplace(BitsetVector& a, const BitsetVector& b, bool use_simd) {
    size_t min_n = std::min(a.words_cnt_, b.words_cnt_);
    if (min_n == 0) { a.words_cnt_ = 0; return; }

    if (use_simd)
        simd::words_and_inplace_simd(a.words_, b.words_, min_n);
    else
        simd::words_and_inplace_scalar(a.words_, b.words_, min_n);

    if (a.words_cnt_ > min_n) {
        memset(a.words_ + min_n, 0, (a.words_cnt_ - min_n) * sizeof(uint64_t));
        a.words_cnt_ = min_n;
    }
}

BitsetVector BitsetVector::word_xor(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t na = a.words_cnt_, nb = b.words_cnt_;
    size_t max_n = std::max(na, nb);
    size_t min_n = std::min(na, nb);
    result.allocate_nozero(max_n);

    if (min_n > 0) {
        if (use_simd)
            simd::words_xor_simd(a.words_, b.words_, result.words_, min_n);
        else
            simd::words_xor_scalar(a.words_, b.words_, result.words_, min_n);
    }
    if (max_n > min_n) {
        const uint64_t* tail = (na > nb) ? a.words_ : b.words_;
        memcpy(result.words_ + min_n, tail + min_n, (max_n - min_n) * sizeof(uint64_t));
    }
    result.num_bits_ = std::max(a.num_bits_, b.num_bits_);
    return result;
}

BitsetVector BitsetVector::word_andnot(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t na = a.words_cnt_, nb = b.words_cnt_;
    if (na == 0) return result;

    result.allocate_nozero(na);
    size_t min_n = std::min(na, nb);
    if (min_n > 0) {
        if (use_simd)
            simd::words_andnot_simd(a.words_, b.words_, result.words_, min_n);
        else
            simd::words_andnot_scalar(a.words_, b.words_, result.words_, min_n);
    }
    if (na > nb)
        memcpy(result.words_ + min_n, a.words_ + min_n, (na - min_n) * sizeof(uint64_t));
    result.num_bits_ = a.num_bits_;
    return result;
}

std::vector<uint32_t> BitsetVector::decode_positions() const {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < words_cnt_; ++i) {
        uint64_t w = words_[i];
        uint32_t base = static_cast<uint32_t>(i * 64);
        while (w) {
            int bit = __builtin_ctzll(w);
            uint32_t pos = base + bit;
            if (pos >= num_bits_) break;
            out.push_back(pos);
            w &= w - 1;
        }
    }
    return out;
}

void BitsetVector::serialize(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    size_t total_bytes = (num_bits_ + 7) / 8;
    out.write(reinterpret_cast<const char*>(words_), total_bytes);
}

bool BitsetVector::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;

    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    num_bits_ = file_size * 8;
    size_t num_words = (file_size + 7) / 8;
    allocate(num_words);
    in.read(reinterpret_cast<char*>(words_), file_size);
    return true;
}

}
