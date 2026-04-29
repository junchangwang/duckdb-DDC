#pragma once

#include "combit/include/combit.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// Fast row ID extraction from a ComBit result produced by OR_many / &= /
// |= / ~ (i.e., every segment is in Decompressed state).  Two passes:
//   Pass 1 — VPOPCNTDQ-accelerated popcount to size the output buffer.
//   Pass 2 — AVX-512 nonzero-byte mask + tzcnt + byte-LUT decode.
// Used exclusively by Q6 (the one query that materialises row IDs for
// downstream DuckDB scans).  All other queries iterate via ComBit's
// seg.l1_literal_data() / for_each_literal with bespoke aggregation.
void GetRowidsComBit(const ComBit& combit_res, std::vector<int64_t>* row_ids);

// EXP4: flat buffer wrapper that skips the redundant zero-init of
// std::vector<uint8_t>(n).  The buffer is fully overwritten by memcpy
// in combit_or_result_to_flat below, so zero-init is wasted work.
class FlatByteBuf {
    std::unique_ptr<uint8_t[]> buf_;
    size_t size_;
public:
    explicit FlatByteBuf(size_t n) : buf_(new uint8_t[n]), size_(n) {}
    uint8_t*       data()       noexcept { return buf_.get(); }
    const uint8_t* data() const noexcept { return buf_.get(); }
    size_t         size() const noexcept { return size_; }
};

// Flatten a ComBit (with Decompressed segments, as returned by OR_many)
// into a single contiguous byte buffer for SIMD-friendly downstream ops.
// Each Decompressed segment contributes l2_count() flat literal bytes.
static inline FlatByteBuf combit_or_result_to_flat(const ComBit& cb) {
    size_t total = 0;
    const size_t n = cb.num_segments();
    for (size_t s = 0; s < n; s++) total += cb.segment(s).l2_count();
    FlatByteBuf out(total);  // no zero-init
    size_t off = 0;
    for (size_t s = 0; s < n; s++) {
        const auto& seg = cb.segment(s);
        const size_t wc = seg.l2_count();
        std::memcpy(out.data() + off, seg.l1_literal_data(), wc);
        off += wc;
    }
    return out;
}

// General-purpose ComBit → flat byte buffer.  Unlike
// combit_or_result_to_flat (which assumes every segment is in
// Decompressed state), this version handles arbitrary mixes of
// Compressed and Decompressed segments — useful for materialising
// flat lookup tables of bitmaps loaded from disk.
//
// Layout matches combit_or_result_to_flat: segment s starts at flat
// byte sum of l2_count() over segments [0..s), so the flat byte index
// returned by ComBit::for_each_literal can be used as a direct index
// into the buffer.
//
// Bit ordering inside each byte is MSB-first (bit 7 = first row in
// the 8-row group), matching ComBit's internal L1 layout and the
// per-byte LUT pattern used in Q3..Q10 aggregations.
//
// Cost: O(num_bits) — slower than combit_or_result_to_flat but
// independent of segment state.  Intended for one-time preparation
// outside the per-iteration timed region.
static inline FlatByteBuf combit_decompress_to_flat(const ComBit& cb) {
    size_t total = 0;
    const size_t n = cb.num_segments();
    for (size_t s = 0; s < n; s++) total += cb.segment(s).l2_count();
    FlatByteBuf out(total);
    std::memset(out.data(), 0, total);

    auto bits = cb.decompress();          // contiguous, length = bit_count_
    const size_t segment_bits = cb.segment_bits();
    size_t flat_off = 0;
    for (size_t s = 0; s < n; s++) {
        const auto& seg = cb.segment(s);
        const size_t seg_len = seg.bit_count();
        const size_t row_off = s * segment_bits;
        for (size_t i = 0; i < seg_len; i++) {
            if (bits[row_off + i])
                out.data()[flat_off + i / 8] |= uint8_t(1) << (7 - (i % 8));
        }
        flat_off += seg.l2_count();
    }
    return out;
}
