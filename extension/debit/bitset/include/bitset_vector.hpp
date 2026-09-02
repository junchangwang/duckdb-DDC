#ifndef BITSET_VECTOR_HPP
#define BITSET_VECTOR_HPP

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace bitset {

class BitsetVector {
public:
    BitsetVector() : words_(nullptr), words_cnt_(0), num_bits_(0) {}
    ~BitsetVector() { std::free(words_); }

    BitsetVector(const BitsetVector& o);
    BitsetVector& operator=(const BitsetVector& o);

    BitsetVector(BitsetVector&& o) noexcept;
    BitsetVector& operator=(BitsetVector&& o) noexcept;

    void allocate(size_t n);

    void allocate_nozero(size_t n);

    void set_bit(uint64_t position);

    bool get_bit(uint64_t position) const;

    uint64_t num_bits() const { return num_bits_; }
    void set_num_bits(uint64_t n) { num_bits_ = n; }

    uint64_t popcount(bool use_simd) const;

    const uint64_t* words() const { return words_; }
    uint64_t*       words_mut()   { return words_; }
    size_t          words_cnt() const { return words_cnt_; }

    static BitsetVector word_or(const BitsetVector& a, const BitsetVector& b, bool use_simd);
    static BitsetVector word_and(const BitsetVector& a, const BitsetVector& b, bool use_simd);
    static void word_and_inplace(BitsetVector& a, const BitsetVector& b, bool use_simd);
    static BitsetVector word_xor(const BitsetVector& a, const BitsetVector& b, bool use_simd);
    static BitsetVector word_andnot(const BitsetVector& a, const BitsetVector& b, bool use_simd);

    std::vector<uint32_t> decode_positions() const;

    void serialize(const std::string& path) const;

    bool load(const std::string& path);

private:
    void ensure_capacity(uint64_t pos);

    uint64_t* words_;
    size_t    words_cnt_;
    uint64_t  num_bits_;
};

}

#endif
