#pragma once

#include "ddc/include/ddc.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

void GetRowidsDDC(const DDC& ddc_res, std::vector<int64_t>* row_ids);

class FlatByteBuf {
    std::unique_ptr<uint8_t[]> buf_;
    size_t size_;
public:
    explicit FlatByteBuf(size_t n) : buf_(new uint8_t[n]), size_(n) {}
    uint8_t*       data()       noexcept { return buf_.get(); }
    const uint8_t* data() const noexcept { return buf_.get(); }
    size_t         size() const noexcept { return size_; }
};

// copy L1 literals
static inline FlatByteBuf ddc_or_result_to_flat(const DDC& cb) {
    size_t total = 0;
    const size_t n = cb.num_segments();
    for (size_t s = 0; s < n; s++) total += cb.segment(s).l2_count();
    FlatByteBuf out(total);
    size_t off = 0;
    // per-segment
    for (size_t s = 0; s < n; s++) {
        const auto& seg = cb.segment(s);
        const size_t wc = seg.l2_count();
        std::memcpy(out.data() + off, seg.l1_literal_data(), wc);
        off += wc;
    }
    return out;
}

// decompress + repack
static inline FlatByteBuf ddc_decompress_to_flat(const DDC& cb) {
    size_t total = 0;
    const size_t n = cb.num_segments();
    for (size_t s = 0; s < n; s++) total += cb.segment(s).l2_count();
    FlatByteBuf out(total);
    std::memset(out.data(), 0, total);

    auto bits = cb.decompress();
    const size_t segment_bits = cb.segment_bits();
    size_t flat_off = 0;
    for (size_t s = 0; s < n; s++) {
        const auto& seg = cb.segment(s);
        const size_t seg_len = seg.bit_count();
        const size_t row_off = s * segment_bits;
        // set bits
        for (size_t i = 0; i < seg_len; i++) {
            if (bits[row_off + i])
                out.data()[flat_off + i / 8] |= uint8_t(1) << (7 - (i % 8));
        }
        flat_off += seg.l2_count();
    }
    return out;
}
