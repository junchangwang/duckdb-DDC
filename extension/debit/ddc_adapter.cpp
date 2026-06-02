#include "ddc_adapter.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// byte -> set-bit positions LUT
struct ByteEntry { uint8_t count; uint8_t pos[8]; };
static ByteEntry g_byte_lut[256];
static bool g_byte_lut_init = []() {
    for (int v = 0; v < 256; v++) {
        uint8_t c = 0;
        for (int b = 7; b >= 0; b--) {
            if (v & (1 << b))
                g_byte_lut[v].pos[c++] = 7 - b;
        }
        g_byte_lut[v].count = c;
    }
    return true;
}();

// decompress DDC to row ids
void GetRowidsDDC(const DDC& ddc_res, std::vector<int64_t>* row_ids) {

    // count set bits
    size_t total_count = 0;
    for (size_t s = 0; s < ddc_res.num_segments(); s++) {
        const auto& seg = ddc_res.segment(s);
        const uint8_t* data = seg.l1_literal_data();
        const size_t nbytes = seg.num_literals();
        size_t i = 0;
#ifdef __AVX512VPOPCNTDQ__
        __m512i acc = _mm512_setzero_si512();
        // SIMD popcnt
        for (; i + 64 <= nbytes; i += 64) {
            __m512i chunk = _mm512_loadu_si512(data + i);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(chunk));
        }
        total_count += _mm512_reduce_add_epi64(acc);
#endif
        // scalar tail
        for (; i + 8 <= nbytes; i += 8) {
            uint64_t w;
            memcpy(&w, data + i, 8);
            total_count += __builtin_popcountll(w);
        }
        for (; i < nbytes; i++)
            total_count += __builtin_popcount(data[i]);
    }

    row_ids->resize(total_count + 8);
    auto element_ptr = row_ids->data();
    uint32_t ids_count = 0;
    size_t global_offset = 0;

    // per-segment emit
    for (size_t seg_idx = 0; seg_idx < ddc_res.num_segments(); seg_idx++) {
        const auto& seg = ddc_res.segment(seg_idx);
        const size_t seg_bits = seg.bit_count();
        const uint8_t* data = seg.l1_literal_data();
        const size_t num_bytes = seg.num_literals();

        size_t i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
        for (; i + 64 <= num_bytes; i += 64) {
            __m512i chunk = _mm512_loadu_si512(data + i);
            __mmask64 nz_mask = _mm512_test_epi8_mask(chunk, chunk);
            if (nz_mask == 0) continue;  // skip zero bytes

            while (nz_mask) {
                int byte_idx = _tzcnt_u64(nz_mask);
                uint8_t byte_val = data[i + byte_idx];
                int64_t base = static_cast<int64_t>(global_offset + (i + byte_idx) * 8);
                const auto& entry = g_byte_lut[byte_val];
                for (int k = 0; k < entry.count; k++)
                    element_ptr[ids_count++] = base + entry.pos[k];
                nz_mask &= nz_mask - 1;
            }
        }
#endif

        // scalar tail
        for (; i < num_bytes; i++) {
            uint8_t byte_val = data[i];
            if (byte_val == 0) continue;
            int64_t base = static_cast<int64_t>(global_offset + i * 8);
            const auto& entry = g_byte_lut[byte_val];
            for (int k = 0; k < entry.count; k++)
                element_ptr[ids_count++] = base + entry.pos[k];
        }

        global_offset += seg_bits;
    }

    row_ids->resize(total_count);
}
