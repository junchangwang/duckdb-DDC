#include "combit_adapter.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// --- Byte-LUT for MSB-first bit extraction ---
// For each byte value (0..255): count of set bits and their MSB-first positions.
// E.g., 0xA0 = 1010_0000 → bits at positions 0 and 2, count=2.
// Total size: 256 × 9 = 2304 bytes → fits in L1d permanently.
struct ByteEntry { uint8_t count; uint8_t pos[8]; };
static ByteEntry g_byte_lut[256];
static bool g_byte_lut_init = []() {
    for (int v = 0; v < 256; v++) {
        uint8_t c = 0;
        for (int b = 7; b >= 0; b--) {
            if (v & (1 << b))
                g_byte_lut[v].pos[c++] = 7 - b;  // MSB-first: bit 7 → pos 0
        }
        g_byte_lut[v].count = c;
    }
    return true;
}();

void GetRowidsComBit(const ComBit& combit_res, std::vector<int64_t>* row_ids) {
    // --- Pass 1: Fast popcount (VPOPCNTDQ when available) ---
    size_t total_count = 0;
    for (size_t s = 0; s < combit_res.num_segments(); s++) {
        const auto& seg = combit_res.segment(s);
        const uint8_t* data = seg.l1_literal_data();
        const size_t nbytes = seg.num_literals();
        size_t i = 0;
#ifdef __AVX512VPOPCNTDQ__
        __m512i acc = _mm512_setzero_si512();
        for (; i + 64 <= nbytes; i += 64) {
            __m512i chunk = _mm512_loadu_si512(data + i);
            acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(chunk));
        }
        total_count += _mm512_reduce_add_epi64(acc);
#endif
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

    // --- Pass 2: Byte-LUT scan with AVX-512 nonzero-byte mask ---
    // For each 64-byte block, use _mm512_test_epi8_mask to get a 64-bit mask
    // of nonzero bytes. Then iterate only the nonzero bytes via tzcnt+blsr
    // and decode each byte through g_byte_lut.
    // This eliminates store-forwarding stalls from the GFNI+store+qword-load
    // approach and reduces iteration count at low density (~2%).
    for (size_t seg_idx = 0; seg_idx < combit_res.num_segments(); seg_idx++) {
        const auto& seg = combit_res.segment(seg_idx);
        const size_t seg_bits = seg.bit_count();
        const uint8_t* data = seg.l1_literal_data();
        const size_t num_bytes = seg.num_literals();

        size_t i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
        for (; i + 64 <= num_bytes; i += 64) {
            __m512i chunk = _mm512_loadu_si512(data + i);
            __mmask64 nz_mask = _mm512_test_epi8_mask(chunk, chunk);
            if (nz_mask == 0) continue;

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
