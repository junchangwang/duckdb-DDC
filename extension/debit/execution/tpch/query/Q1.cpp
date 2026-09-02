

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "execution/tpch/bitmap_table_scan.hpp"
#include "duckdb/common/enums/operator_result_type.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "ddc_adapter.h"
#include "ddc/include/ddc.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"
#include "Concise/concise.h"

#include "execution/tpch/bm_baseline_loaders.hpp"
#include "execution/tpch/bm_bench_common.hpp"
#include "execution/tpch/indexed_bitmap.hpp"
#include "execution/tpch/bsr_index.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__AVX512F__)
#  include <immintrin.h>
#endif

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q1_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

using Q1BmType = bm_bench::Backend;
static const Q1BmType Q1_BM_NEW = bm_bench::parse_backend("Q1_BM");
static const char* q1_bm_label_new() { return bm_bench::backend_label(Q1_BM_NEW); }
static std::string q1_get_sf_label() { return bm_bench::sf_label(); }

static const int Q1_ITERATIONS = bm_bench::iter_count(5);
static const int Q1_WARMUP     = bm_bench::warmup_count(1);

static std::once_flag q1_once_flag_new;

static constexpr int64_t Q1_SHIPDATE_CUTOFF = 10471;

static constexpr int64_t Q1_GE_AFTER_MONTHS[4] = {199809, 199810, 199811, 199812};

struct Q1GroupAns {
    int64_t sum_qty        = 0;
    int64_t sum_base_price = 0;
    int64_t sum_disc_price = 0;
    int64_t sum_charge     = 0;
    int64_t sum_discount   = 0;
    int64_t count_order    = 0;
};

#define Q1_AGG_ROW(g, r, qty, price, disc, tax) do { \
    (g).sum_qty        += (qty)[(r)];                 \
    (g).sum_discount   += (disc)[(r)];                \
    (g).sum_base_price += (price)[(r)];               \
    int64_t _dp = (price)[(r)] * (100 - (disc)[(r)]); \
    (g).sum_disc_price += _dp;                        \
    (g).sum_charge     += _dp * (100 + (tax)[(r)]);   \
    (g).count_order++;                                \
} while (0)

static std::array<uint8_t, 256> make_q1_reverse_table() {
    std::array<uint8_t, 256> t{};
    for (int i = 0; i < 256; i++) {
        uint8_t b = static_cast<uint8_t>(i), r = 0;
        for (int k = 0; k < 8; k++)
            if (b & (1u << k)) r |= static_cast<uint8_t>(1u << (7 - k));
        t[i] = r;
    }
    return t;
}
static const std::array<uint8_t, 256> q1_reverse_table = make_q1_reverse_table();

static inline void q1_agg_8rows(
    const int64_t* qty, const int64_t* price, const int64_t* disc, const int64_t* tax,
    size_t base, uint8_t bits_msb, Q1GroupAns& ans)
{
#if defined(__AVX512F__)
    if (!bits_msb) return;

    const uint8_t mask = q1_reverse_table[bits_msb];

    __m512i qty_v   = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qty + base));
    __m512i price_v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(price + base));
    __m512i disc_v  = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(disc + base));
    __m512i tax_v   = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tax + base));
    __m512i c100    = _mm512_set1_epi64(100);

    ans.sum_qty += _mm512_reduce_add_epi64(_mm512_maskz_compress_epi64(mask, qty_v));

    ans.sum_discount += _mm512_reduce_add_epi64(_mm512_maskz_compress_epi64(mask, disc_v));

    __m512i cprice = _mm512_maskz_compress_epi64(mask, price_v);
    ans.sum_base_price += _mm512_reduce_add_epi64(cprice);

    __m512i cdisc_inv = _mm512_maskz_compress_epi64(mask, _mm512_sub_epi64(c100, disc_v));
    __m512i dp_v = _mm512_mullo_epi64(cprice, cdisc_inv);
    ans.sum_disc_price += _mm512_reduce_add_epi64(dp_v);

    __m512i ctax_p100 = _mm512_maskz_compress_epi64(mask, _mm512_add_epi64(c100, tax_v));
    __m512i ch_v = _mm512_mullo_epi64(dp_v, ctax_p100);
    ans.sum_charge += _mm512_reduce_add_epi64(ch_v);

    ans.count_order += __builtin_popcount(static_cast<unsigned>(bits_msb));
#else
    for (int i = 0; i < 8; i++) {
        if (bits_msb & (0x80 >> i))
            Q1_AGG_ROW(ans, base + i, qty, price, disc, tax);
    }
#endif
}


// Word flatten
static inline void q1_lsb_words_to_bs(const uint64_t* w, size_t nwords,
                                      std::vector<uint8_t>& bs, size_t nbytes) {
    const size_t lim_w = std::min(nwords, (nbytes + 7) / 8);
    for (size_t i = 0; i < lim_w; i++) {
        const uint64_t x = w[i];
        if (!x) continue;
        const size_t o = i * 8;
        const size_t n = std::min<size_t>(8, nbytes - o);
        for (size_t j = 0; j < n; j++)
            bs[o + j] = q1_reverse_table[static_cast<uint8_t>(x >> (8 * j))];
    }
}

static void q1_to_byte_stream(const DDC& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    bs.assign((num_rows + 7) / 8, 0);

    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        if (word_pos < bs.size()) bs[word_pos] = val;
    });
}

static void q1_to_byte_stream(const bm_index::BsrSet& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    bs.assign((num_rows + 7) / 8, 0);
    for (int i = 0; i < b.size; i++) {
        uint32_t st = static_cast<uint32_t>(b.states[i]);
        uint64_t base_bit = static_cast<uint64_t>(static_cast<uint32_t>(b.bases[i])) * 32ULL;
        while (st) {
            int t = __builtin_ctz(st); st &= st - 1;
            uint64_t p = base_bit + static_cast<uint64_t>(t);
            if (p < num_rows) bs[p >> 3] |= static_cast<uint8_t>(1u << (7 - (p & 7)));
        }
    }
}

static void q1_to_byte_stream(const ::bitset::BitsetVector& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    const size_t nbytes = (num_rows + 7) / 8;
    bs.assign(nbytes, 0);
    q1_lsb_words_to_bs(b.words(), b.words_cnt(), bs, nbytes);
}

static void q1_to_byte_stream(const roaring::Roaring& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    const size_t nbytes = (num_rows + 7) / 8;
    bs.assign(nbytes, 0);
    roaring::api::bitset_t* dense = roaring::api::bitset_create_with_capacity(num_rows);
    if (!dense) return;
    if (roaring::api::roaring_bitmap_to_bitset(&b.roaring, dense))
        q1_lsb_words_to_bs(dense->array, dense->arraysize, bs, nbytes);
    roaring::api::bitset_free(dense);
}

static void q1_to_byte_stream(const ibis::bitvector& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    const size_t nbytes = (num_rows + 7) / 8;
    bs.assign(nbytes, 0);
    uint64_t acc = 0; int nb = 0; size_t out = 0;
    auto push31 = [&](uint32_t payload) {
        acc = (acc << 31) | payload; nb += 31;
        while (nb >= 8 && out < nbytes) {
            bs[out++] = static_cast<uint8_t>(acc >> (nb - 8)); nb -= 8;
        }
    };
    for (auto it = b.m_vec.begin(); it != b.m_vec.end() && out < nbytes; ++it) {
        const uint32_t w = *it;
        if (w > ibis::bitvector::ALLONES) {
            const uint32_t cnt = w & ibis::bitvector::MAXCNT;
            const uint32_t val = (w >= ibis::bitvector::HEADER1)
                                     ? ibis::bitvector::ALLONES : 0u;
            for (uint32_t k = 0; k < cnt && out < nbytes; k++) push31(val);
        } else {
            push31(w);
        }
    }
    if (b.active.nbits && out < nbytes) {
        acc = (acc << b.active.nbits) | b.active.val;
        nb += static_cast<int>(b.active.nbits);
        while (nb >= 8 && out < nbytes) {
            bs[out++] = static_cast<uint8_t>(acc >> (nb - 8)); nb -= 8;
        }
    }
    if (nb > 0 && out < nbytes) bs[out++] = static_cast<uint8_t>(acc << (8 - nb));
}

static void q1_to_byte_stream(const ewah::EWAHBoolArray<uint64_t>& b,
                              std::vector<uint8_t>& bs, size_t num_rows) {
    const size_t nbytes = (num_rows + 7) / 8;
    bs.assign(nbytes, 0);
    auto it = b.uncompress();
    for (size_t i = 0; it.hasNext(); i++) {
        const uint64_t x = it.next();
        const size_t o = i * 8;
        if (o >= nbytes) break;
        if (!x) continue;
        const size_t n = std::min<size_t>(8, nbytes - o);
        for (size_t j = 0; j < n; j++)
            bs[o + j] = q1_reverse_table[static_cast<uint8_t>(x >> (8 * j))];
    }
}

static void q1_to_byte_stream(const ConciseSet<false>& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    bs.assign((num_rows + 7) / 8, 0);
    for (auto it = b.begin(); it != b.end(); ++it) {
        uint32_t p = *it;
        if (p < num_rows) bs[p / 8] |= uint8_t(1) << (7 - (p % 8));
    }
}

static void q1_byte_stream_not(std::vector<uint8_t>& bs, size_t num_rows) {
    for (auto& b : bs) b = ~b;
    size_t tail = num_rows & 7;
    if (tail != 0) {

        uint8_t mask = static_cast<uint8_t>(0xFFu << (8 - tail));
        bs.back() &= mask;
    }
}

static void q1_byte_stream_and(std::vector<uint8_t>& dst,
                               const std::vector<uint8_t>& src) {
    size_t n = dst.size();
    for (size_t i = 0; i < n; i++) dst[i] &= src[i];
}

static void q1_run_aggregate(
    ClientContext& ctx,
    TableCatalogEntry& lineitem_table,
    DuckTransaction& tx,
    const std::map<std::pair<char,char>, std::vector<uint8_t>>& group_bs,
    std::map<std::pair<char,char>, Q1GroupAns>& ans)
{
    TableScanState scan_state;
    vector<StorageIndex> col_ids = {
        StorageIndex(4),
        StorageIndex(5),
        StorageIndex(6),
        StorageIndex(7),
    };
    lineitem_table.GetStorage().InitializeScan(ctx, tx, scan_state, col_ids);
    vector<LogicalType> types = {
        lineitem_table.GetColumns().GetColumnTypes()[4],
        lineitem_table.GetColumns().GetColumnTypes()[5],
        lineitem_table.GetColumns().GetColumnTypes()[6],
        lineitem_table.GetColumns().GetColumnTypes()[7],
    };

    struct Slot {
        const uint8_t* bs_base;
        Q1GroupAns*    ans_ptr;
    };
    std::vector<Slot> slots;
    slots.reserve(group_bs.size());
    for (auto& [k, bs] : group_bs) {
        slots.push_back({bs.data(), &ans[k]});
    }

    size_t row_offset = 0;
    while (true) {
        DataChunk chunk; chunk.Initialize(ctx, types);
        lineitem_table.GetStorage().Scan(tx, chunk, scan_state);
        if (chunk.size() == 0) break;
        auto qty   = FlatVector::GetData<int64_t>(chunk.data[0]);
        auto price = FlatVector::GetData<int64_t>(chunk.data[1]);
        auto disc  = FlatVector::GetData<int64_t>(chunk.data[2]);
        auto tax   = FlatVector::GetData<int64_t>(chunk.data[3]);
        idx_t n = chunk.size();

        idx_t base = 0;
        while (base + 7 < n) {
            size_t bs_off = (row_offset + base) / 8;
            for (auto& slot : slots) {
                uint8_t bits = slot.bs_base[bs_off];
                q1_agg_8rows(qty, price, disc, tax, base, bits, *slot.ans_ptr);
            }
            base += 8;
        }

        if (base < n) {
            for (auto& slot : slots) {
                for (idx_t r = base; r < n; r++) {
                    size_t pos = row_offset + r;
                    uint8_t bits = slot.bs_base[pos / 8];
                    if (bits & (0x80 >> (pos % 8)))
                        Q1_AGG_ROW(*slot.ans_ptr, r, qty, price, disc, tax);
                }
            }
        }
        row_offset += n;
    }
}

void BMTableScan::BMTPCH_Q1(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q1_once_flag_new, [&]() {

    bm_bench::warn_if_sf1();

    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q1 (BitEngine normal-path: complement OR + bit-stream + seq scan) — "
              << q1_bm_label_new() << " (" << q1_get_sf_label() << ")" << std::endl;
    std::cout << "  Pre-loaded: bitmap_shipdate + bitmap_linestatus + bitmap_returnflag" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_ge_m = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_GE_month);
    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    auto* idx_ls   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_linestatus);
    auto* idx_rf   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_returnflag);
    auto* idx_use  = idx_ge_m ? idx_ge_m : idx_ship;
    if (!idx_use || !idx_ls || !idx_rf) {
        std::cerr << "[Q1] ERROR: required bitmaps not loaded.\n";
        return;
    }
    const bool use_month_ge = (idx_ge_m != nullptr);

    if (!use_month_ge) {
        if (std::string(idx_ship->backend_name()) != idx_ls->backend_name() ||
            std::string(idx_ship->backend_name()) != idx_rf->backend_name()) {
            std::cerr << "[Q1] ERROR: backend mismatch among shipdate/linestatus/returnflag.\n";
            return;
        }
    }
    size_t num_rows = idx_use->num_rows();

    static const char RF_CHARS[] = {'A', 'N', 'R'};
    static const char LS_CHARS[] = {'F', 'O'};
    static const int  RF_COUNT   = 3;
    static const int  LS_COUNT   = 2;

    std::vector<double> tA_t, tB_t, tC_t, tot_t;
    std::map<std::pair<char,char>, Q1GroupAns> last_ans;

    for (int iter = 0; iter < Q1_ITERATIONS; iter++) {
        bool warm = iter < Q1_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q1_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        std::map<std::pair<char,char>, Q1GroupAns> ans;
        std::map<std::pair<char,char>, std::vector<uint8_t>> group_bs;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        std::vector<uint8_t> shipdate_filter_bs;
        double phaseA = 0;

        if (auto* bs_ship = dynamic_cast<bm_index::IndexedBSR*>(idx_use)) {
            auto* bs_ls_x = dynamic_cast<bm_index::IndexedBSR*>(idx_ls);
            auto* bs_rf_x = dynamic_cast<bm_index::IndexedBSR*>(idx_rf);
            if (!bs_ls_x || !bs_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            bm_index::BsrSet shipdate_gt;
            if (use_month_ge) {
                std::vector<int64_t> month_keys(Q1_GE_AFTER_MONTHS, Q1_GE_AFTER_MONTHS + 4);
                shipdate_gt = bs_ship->or_many(month_keys);
            } else {
                shipdate_gt = bs_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            }
            q1_to_byte_stream(shipdate_gt, shipdate_filter_bs, num_rows);
            q1_byte_stream_not(shipdate_filter_bs, num_rows);
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            std::map<char, std::vector<uint8_t>> ls_bs, rf_bs;
            for (int li = 0; li < LS_COUNT; li++) {
                bm_index::BsrSet b = bs_ls_x->single_key(static_cast<int64_t>(LS_CHARS[li]));
                std::vector<uint8_t> tmp;
                q1_to_byte_stream(b, tmp, num_rows);
                ls_bs[LS_CHARS[li]] = std::move(tmp);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                bm_index::BsrSet b = bs_rf_x->single_key(static_cast<int64_t>(RF_CHARS[ri]));
                std::vector<uint8_t> tmp;
                q1_to_byte_stream(b, tmp, num_rows);
                rf_bs[RF_CHARS[ri]] = std::move(tmp);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    char rf = RF_CHARS[ri], ls = LS_CHARS[li];
                    std::vector<uint8_t> bs = ls_bs[ls];
                    q1_byte_stream_and(bs, rf_bs[rf]);
                    q1_byte_stream_and(bs, shipdate_filter_bs);
                    group_bs[{rf, ls}] = std::move(bs);
                }
            }
        }
        else if (auto* cbge_ship = dynamic_cast<bm_index::IndexedDDCGE*>(idx_use)) {
            auto* cb_ls_x = dynamic_cast<bm_index::IndexedDDC*>(idx_ls);
            auto* cb_rf_x = dynamic_cast<bm_index::IndexedDDC*>(idx_rf);
            if (!cb_ls_x || !cb_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            std::vector<int64_t> month_keys(Q1_GE_AFTER_MONTHS, Q1_GE_AFTER_MONTHS + 4);
            DDC shipdate_gt = cbge_ship->or_many(month_keys);
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            DDC ship_le = ~shipdate_gt;
            DDC ls_bm[LS_COUNT], rf_bm[RF_COUNT];
            for (int li = 0; li < LS_COUNT; li++) {
                ls_bm[li] = DDC::from_sparse_positions({}, num_rows, cbge_ship->segment_bits());
                cb_ls_x->apply_or_to(ls_bm[li], static_cast<int64_t>(LS_CHARS[li]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                rf_bm[ri] = DDC::from_sparse_positions({}, num_rows, cbge_ship->segment_bits());
                cb_rf_x->apply_or_to(rf_bm[ri], static_cast<int64_t>(RF_CHARS[ri]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    DDC g = (rf_bm[ri] & ls_bm[li]) & ship_le;
                    if (g.popcount() == 0) continue;
                    std::vector<uint8_t> gbs;
                    q1_to_byte_stream(g, gbs, num_rows);
                    group_bs[{RF_CHARS[ri], LS_CHARS[li]}] = std::move(gbs);
                }
            }
        }
        else if (auto* cb_ship = dynamic_cast<bm_index::IndexedDDC*>(idx_use)) {
            auto* cb_ls_x = dynamic_cast<bm_index::IndexedDDC*>(idx_ls);
            auto* cb_rf_x = dynamic_cast<bm_index::IndexedDDC*>(idx_rf);
            if (!cb_ls_x || !cb_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();

            auto* cb_ship_bpe = dynamic_cast<bm_index::IndexedDDCBPE*>(
                static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_BPE));
            DDC shipdate_gt;
            if (cb_ship_bpe) {
                int B    = cb_ship_bpe->bucket_of(Q1_SHIPDATE_CUTOFF);
                int LAST = static_cast<int>(cb_ship_bpe->num_keys()) - 1;
                shipdate_gt = *cb_ship_bpe->prefix_at_bucket(LAST);
                shipdate_gt &= ~(*cb_ship_bpe->prefix_at_bucket(B));
                int64_t bmax = cb_ship_bpe->bucket_max(B);
                if (Q1_SHIPDATE_CUTOFF + 1 <= bmax) {
                    DDC boundary = DDC::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                    cb_ship->apply_or_range_to(boundary, Q1_SHIPDATE_CUTOFF + 1, bmax);
                    shipdate_gt |= boundary;
                }
            } else {

                std::vector<int64_t> gt_keys;
                cb_ship->for_each_key([&](int64_t k) {
                    if (k > Q1_SHIPDATE_CUTOFF) gt_keys.push_back(k);
                });
                shipdate_gt = cb_ship->or_many(gt_keys);
            }
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            DDC ship_le = ~shipdate_gt;
            DDC ls_bm[LS_COUNT], rf_bm[RF_COUNT];
            for (int li = 0; li < LS_COUNT; li++) {
                ls_bm[li] = DDC::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                cb_ls_x->apply_or_to(ls_bm[li], static_cast<int64_t>(LS_CHARS[li]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                rf_bm[ri] = DDC::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                cb_rf_x->apply_or_to(rf_bm[ri], static_cast<int64_t>(RF_CHARS[ri]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    DDC g = (rf_bm[ri] & ls_bm[li]) & ship_le;
                    if (g.popcount() == 0) continue;
                    std::vector<uint8_t> gbs;
                    q1_to_byte_stream(g, gbs, num_rows);
                    group_bs[{RF_CHARS[ri], LS_CHARS[li]}] = std::move(gbs);
                }
            }
        }
        else if (auto* cr_ship = dynamic_cast<bm_index::IndexedCRoaring*>(idx_use)) {
            auto* cr_ls_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ls);
            auto* cr_rf_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_rf);
            if (!cr_ls_x || !cr_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            roaring::Roaring shipdate_gt;
            if (use_month_ge) {

                std::vector<int64_t> month_keys(Q1_GE_AFTER_MONTHS, Q1_GE_AFTER_MONTHS + 4);
                shipdate_gt = cr_ship->or_many(month_keys);
            } else {

                auto* cr_ship_bpe = dynamic_cast<bm_index::IndexedCRoaringBPE*>(
                    static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_BPE));
                if (cr_ship_bpe) {
                    int B    = cr_ship_bpe->bucket_of(Q1_SHIPDATE_CUTOFF);
                    int LAST = static_cast<int>(cr_ship_bpe->num_keys()) - 1;
                    shipdate_gt = *cr_ship_bpe->prefix_at_bucket(LAST);
                    shipdate_gt -= *cr_ship_bpe->prefix_at_bucket(B);
                    int64_t bmax = cr_ship_bpe->bucket_max(B);
                    if (Q1_SHIPDATE_CUTOFF + 1 <= bmax) {
                        roaring::Roaring boundary;
                        if (cr_ship->run_optimized()) boundary = cr_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, bmax);
                        else cr_ship->apply_or_range_to(boundary, Q1_SHIPDATE_CUTOFF + 1, bmax);
                        shipdate_gt |= boundary;
                    }
                } else {

                    shipdate_gt = cr_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
                }
            }
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            roaring::Roaring ls_bm[LS_COUNT], rf_bm[RF_COUNT];
            for (int li = 0; li < LS_COUNT; li++) {

                cr_ls_x->apply_or_to(ls_bm[li], static_cast<int64_t>(LS_CHARS[li]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {

                cr_rf_x->apply_or_to(rf_bm[ri], static_cast<int64_t>(RF_CHARS[ri]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    roaring::Roaring g = rf_bm[ri] & ls_bm[li];
                    g -= shipdate_gt;
                    if (g.cardinality() == 0) continue;
                    std::vector<uint8_t> gbs;
                    q1_to_byte_stream(g, gbs, num_rows);
                    group_bs[{RF_CHARS[ri], LS_CHARS[li]}] = std::move(gbs);
                }
            }
        }
        else if (auto* bs_ship = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_use)) {
            auto* bs_ls_x = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_ls);
            auto* bs_rf_x = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_rf);
            if (!bs_ls_x || !bs_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ::bitset::BitsetVector shipdate_gt = bs_ship->make_empty();
            if (use_month_ge) {
                std::vector<int64_t> month_keys(Q1_GE_AFTER_MONTHS, Q1_GE_AFTER_MONTHS + 4);
                shipdate_gt = bs_ship->or_many(month_keys);
            } else {
                bs_ship->apply_or_range_to(shipdate_gt, Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            }
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            const bool simd = bs_ship->use_simd();
            ::bitset::BitsetVector ls_bm[LS_COUNT], rf_bm[RF_COUNT];
            for (int li = 0; li < LS_COUNT; li++) {
                ls_bm[li] = bs_ls_x->make_empty();
                bs_ls_x->apply_or_to(ls_bm[li], static_cast<int64_t>(LS_CHARS[li]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                rf_bm[ri] = bs_rf_x->make_empty();
                bs_rf_x->apply_or_to(rf_bm[ri], static_cast<int64_t>(RF_CHARS[ri]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    ::bitset::BitsetVector g =
                        ::bitset::BitsetVector::word_and(rf_bm[ri], ls_bm[li], simd);
                    g = ::bitset::BitsetVector::word_andnot(g, shipdate_gt, simd);
                    if (g.popcount(simd) == 0) continue;
                    std::vector<uint8_t> gbs;
                    q1_to_byte_stream(g, gbs, num_rows);
                    group_bs[{RF_CHARS[ri], LS_CHARS[li]}] = std::move(gbs);
                }
            }
        }
        else if (auto* wah_ship = dynamic_cast<bm_index::IndexedWAH*>(idx_use)) {
            auto* wah_ls_x = dynamic_cast<bm_index::IndexedWAH*>(idx_ls);
            auto* wah_rf_x = dynamic_cast<bm_index::IndexedWAH*>(idx_rf);
            if (!wah_ls_x || !wah_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ibis::bitvector shipdate_gt;
            if (use_month_ge) {
                std::vector<int64_t> month_keys(Q1_GE_AFTER_MONTHS, Q1_GE_AFTER_MONTHS + 4);
                shipdate_gt = wah_ship->or_many(month_keys);
            } else {

                shipdate_gt = wah_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            }
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            shipdate_gt.decompress();
            ibis::bitvector ls_bm[LS_COUNT], rf_bm[RF_COUNT];
            for (int li = 0; li < LS_COUNT; li++) {

                wah_ls_x->apply_or_to(ls_bm[li], static_cast<int64_t>(LS_CHARS[li]));
                ls_bm[li].decompress();
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {

                wah_rf_x->apply_or_to(rf_bm[ri], static_cast<int64_t>(RF_CHARS[ri]));
                rf_bm[ri].decompress();
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    ibis::bitvector g;
                    g.copy(rf_bm[ri]);
                    g &= ls_bm[li];
                    g -= shipdate_gt;
                    if (g.cnt() == 0) continue;
                    std::vector<uint8_t> gbs;
                    q1_to_byte_stream(g, gbs, num_rows);
                    group_bs[{RF_CHARS[ri], LS_CHARS[li]}] = std::move(gbs);
                }
            }
        }
        else if (auto* ew_ship = dynamic_cast<bm_index::IndexedEWAH*>(idx_use)) {
            auto* ew_ls_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_ls);
            auto* ew_rf_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_rf);
            if (!ew_ls_x || !ew_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ewah::EWAHBoolArray<uint64_t> shipdate_gt;
            if (use_month_ge) {
                std::vector<int64_t> month_keys(Q1_GE_AFTER_MONTHS, Q1_GE_AFTER_MONTHS + 4);
                shipdate_gt = ew_ship->or_many(month_keys);
            } else {
                shipdate_gt = ew_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            }
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            ewah::EWAHBoolArray<uint64_t> ls_bm[LS_COUNT], rf_bm[RF_COUNT];
            for (int li = 0; li < LS_COUNT; li++) {

                ew_ls_x->apply_or_to(ls_bm[li], static_cast<int64_t>(LS_CHARS[li]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {

                ew_rf_x->apply_or_to(rf_bm[ri], static_cast<int64_t>(RF_CHARS[ri]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    ewah::EWAHBoolArray<uint64_t> t1, g;
                    rf_bm[ri].logicaland(ls_bm[li], t1);
                    t1.logicalandnot(shipdate_gt, g);
                    if (g.numberOfOnes() == 0) continue;
                    std::vector<uint8_t> gbs;
                    q1_to_byte_stream(g, gbs, num_rows);
                    group_bs[{RF_CHARS[ri], LS_CHARS[li]}] = std::move(gbs);
                }
            }
        }
        else if (auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_use)) {
            auto* con_ls_x = dynamic_cast<bm_index::IndexedConcise*>(idx_ls);
            auto* con_rf_x = dynamic_cast<bm_index::IndexedConcise*>(idx_rf);
            if (!con_ls_x || !con_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ConciseSet<false> shipdate_gt;
            if (use_month_ge) {
                std::vector<int64_t> month_keys(Q1_GE_AFTER_MONTHS, Q1_GE_AFTER_MONTHS + 4);
                shipdate_gt = con_ship->or_many(month_keys);
            } else {
                shipdate_gt = con_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            }
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            ConciseSet<false> ls_bm[LS_COUNT], rf_bm[RF_COUNT];
            for (int li = 0; li < LS_COUNT; li++) {

                con_ls_x->apply_or_to(ls_bm[li], static_cast<int64_t>(LS_CHARS[li]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {

                con_rf_x->apply_or_to(rf_bm[ri], static_cast<int64_t>(RF_CHARS[ri]));
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                for (int li = 0; li < LS_COUNT; li++) {
                    ConciseSet<false> g =
                        rf_bm[ri].logicaland(ls_bm[li]).logicalandnot(shipdate_gt);
                    if (g.size() == 0) continue;
                    std::vector<uint8_t> gbs;
                    q1_to_byte_stream(g, gbs, num_rows);
                    group_bs[{RF_CHARS[ri], LS_CHARS[li]}] = std::move(gbs);
                }
            }
        }
        else {
            std::cerr << "[Q1] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }

        auto t_b1 = clk::now();

        double phaseB = q1_ms(t0, t_b1) - phaseA;

        q1_run_aggregate(context.client, lineitem_table, lineitem_tx, group_bs, ans);
        auto t_c1 = clk::now();
        double phaseC = q1_ms(t_b1, t_c1);
        double tot    = q1_ms(t0, t_c1);

        std::cout << "  " << idx_use->backend_name()
                  << ":  PhaseA=" << phaseA << "  PhaseB=" << phaseB
                  << "  PhaseC=" << phaseC << "  Total=" << tot << std::endl;
        if (!warm) {
            tA_t.push_back(phaseA);
            tB_t.push_back(phaseB);
            tC_t.push_back(phaseC);
            tot_t.push_back(tot);
        }
        last_ans = ans;
    }

    // SQL validation
    bool gt_ok = false;
    std::map<std::pair<char,char>, std::vector<double>> gt;
    {
        Connection con(*context.client.db);

        const char* sql_cutoff = use_month_ge
            ? "DATE '1998-09-01' - INTERVAL '1' DAY"
            : "DATE '1998-12-01' - INTERVAL '90' DAY";
        const std::string sql = std::string(
            "SELECT l_returnflag, l_linestatus, "
            "       sum(l_quantity), sum(l_extendedprice), "
            "       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
            "       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
            "       sum(l_discount), count(*) "
            "FROM   lineitem "
            "WHERE  l_shipdate <= ") + sql_cutoff +
            " GROUP BY l_returnflag, l_linestatus "
            " ORDER BY l_returnflag, l_linestatus";
        auto r = con.Query(sql);
        if (r && !r->HasError()) {
            for (idx_t i = 0; i < r->RowCount(); i++) {
                char rf = r->GetValue(0, i).GetValue<std::string>()[0];
                char ls = r->GetValue(1, i).GetValue<std::string>()[0];
                std::vector<double> vals(6);
                vals[0] = r->GetValue(2, i).GetValue<double>();
                vals[1] = r->GetValue(3, i).GetValue<double>();
                vals[2] = r->GetValue(4, i).GetValue<double>();
                vals[3] = r->GetValue(5, i).GetValue<double>();
                vals[4] = r->GetValue(6, i).GetValue<double>();
                vals[5] = double(r->GetValue(7, i).GetValue<int64_t>());
                gt[{rf, ls}] = vals;
            }
            gt_ok = true;
        }
    }
    if (gt_ok) {

        std::map<std::pair<char,char>, Q1GroupAns> non_empty;
        for (auto& [k, g] : last_ans)
            if (g.count_order > 0) non_empty[k] = g;
        bool ok = (non_empty.size() == gt.size());
        if (ok) {
            for (auto& [k, g] : non_empty) {
                auto it = gt.find(k);
                if (it == gt.end()) { ok = false; break; }
                double our_qty   = double(g.sum_qty)        / 100;
                double our_price = double(g.sum_base_price) / 100;
                double our_disc_price = double(g.sum_disc_price) / 10000;
                double our_charge     = double(g.sum_charge)     / 1000000;
                double our_discount   = double(g.sum_discount)   / 100;
                double our_count      = double(g.count_order);
                if (std::fabs(our_qty   - it->second[0]) > 1.0 ||
                    std::fabs(our_price - it->second[1]) > 1.0 ||
                    std::fabs(our_disc_price - it->second[2]) > 1.0 ||
                    std::fabs(our_charge     - it->second[3]) > 1.0 ||
                    std::fabs(our_discount   - it->second[4]) > 1.0 ||
                    std::fabs(our_count      - it->second[5]) > 0.5) {
                    ok = false;
                    std::cerr << "[FAIL] (" << k.first << "," << k.second << "): "
                              << "ours=(" << our_qty << "," << our_price << "," << our_disc_price
                              << "," << our_charge << "," << our_discount << "," << our_count << ")  "
                              << "gt=(" << it->second[0] << "," << it->second[1] << "," << it->second[2]
                              << "," << it->second[3] << "," << it->second[4] << "," << it->second[5] << ")\n";
                }
            }
        }
        if (ok)
            std::cout << "\n[OK] " << idx_use->backend_name()
                      << " matches DuckDB SQL ground truth ("
                      << gt.size() << " (rf,ls) groups).\n";
        else
            std::cerr << "\n[FAIL] mismatch vs SQL ground truth!\n";
    }

    std::cout << "\n  Q1 group results (rf,ls): qty / base_price / disc_price / charge / discount / count\n";
    for (auto& [k, g] : last_ans) {
        if (g.count_order == 0) continue;
        std::cout << "  (" << k.first << "," << k.second << "): "
                  << std::fixed << std::setprecision(2)
                  << double(g.sum_qty) / 100        << "  "
                  << double(g.sum_base_price) / 100 << "  "
                  << double(g.sum_disc_price) / 10000 << "  "
                  << double(g.sum_charge) / 1000000 << "  "
                  << double(g.sum_discount) / 100   << "  "
                  << g.count_order << std::endl;
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t), sT = stats(tot_t);
    int measured = std::max(0, Q1_ITERATIONS - Q1_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q1 RESULTS — " << idx_use->backend_name()
              << " only (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (shipdate ~OR + flip): " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (per-group AND→bs)   : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (seq-scan + agg)     : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  TOTAL                       : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    std::string sf = q1_get_sf_label();
    std::ofstream csv("q1_results_" + sf + ".csv");
    if (csv) {
        csv << std::fixed << std::setprecision(4);
        csv << "sf,operation,"
            << "wah_median_ms,wah_stddev_ms,wah_min_ms,wah_max_ms,"
            << "ddc_median_ms,ddc_stddev_ms,ddc_min_ms,ddc_max_ms,"
            << "croaring_median_ms,croaring_stddev_ms,croaring_min_ms,croaring_max_ms,"
            << "croaring_run_median_ms,croaring_run_stddev_ms,croaring_run_min_ms,croaring_run_max_ms,"
            << "ewah_median_ms,ewah_stddev_ms,ewah_min_ms,ewah_max_ms,"
            << "bs_median_ms,bs_stddev_ms,bs_min_ms,bs_max_ms,"
            << "bsa_median_ms,bsa_stddev_ms,bsa_min_ms,bsa_max_ms,"
            << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms,"
            << "cb_vs_wah,cr_vs_wah,crr_vs_wah,ew_vs_wah,bs_vs_wah,bsa_vs_wah,con_vs_wah\n";
        bm_bench::Stats z{0,0,0,0};
        std::string bn = idx_use->backend_name();
        auto put = [&](const std::string& op, bm_bench::Stats s) {
            csv << sf << "," << op << ",";
            auto cell = [&](bm_bench::Stats v) {
                csv << v.median << "," << v.stddev << "," << v.min_val << "," << v.max_val << ",";
            };
            cell(bn == "WAH" ? s : z);
            cell(bn == "DDC" ? s : z);
            cell(bn == "CRoaring" ? s : z);
            cell(bn == "CRoaringRun" ? s : z);
            cell(bn == "EWAH" ? s : z);
            cell(bn == "Bitset" ? s : z);
            cell(bn == "Bitset+AVX512" ? s : z);
            cell(bn == "Concise" ? s : z);
            csv << "0,0,0,0,0,0,0\n";
        };
        put("PhaseA", sA);
        put("PhaseB", sB);
        put("PhaseC", sC);
        put("TOTAL",  sT);
        std::cout << "  [CSV] q1_results_" << sf << ".csv\n";
    }

    });
}

}
