// TPC-H Q1 — Pricing Summary Report
// Verbatim port of teacher's BitEngine BMTPCH_Q1 "normal path" branch
// (Q1.cpp lines 683-836 on origin/BitEngine).
//
// Pipeline (1:1 mirror of teacher):
//
//   Phase A — shipdate complement OR + flip:
//             shipdate_gt = OR over rabit_shipdate->Btvs[i] for i > right_days
//             shipdate_filter = NOT shipdate_gt    (= shipdate <= cutoff)
//             Only ~90 bitmaps are OR'd (not 2437 full range), so the
//             multi-OR cost stays small even for backends without a
//             k-way merge (e.g. ComBit's pairwise SparseComBit).
//
//   Phase B — per-(rf, ls) group bitmap:
//             group_filter = linestatus[ls] AND returnflag[rf] AND shipdate_filter
//             (5 valid groups: A/F, N/F, N/O, R/F, A/O — A/O is empty
//              under the cutoff so usually 4 populate.)
//             Each group_filter is converted to a packed MSB-first
//             byte-stream of length (num_rows + 7) / 8.  Mirrors
//             teacher's `reduce_leadingbits` / `reduce_leadingbits_seg`.
//
//   Phase C — sequential lineitem scan + bit-stream filter + inline agg:
//             One pass over lineitem (l_quantity, l_extendedprice,
//             l_discount, l_tax).  For each result chunk, walk each
//             group's byte-stream 8 rows at a time; for each set bit
//             accumulate q1_data per teacher's Q1_AGG_ROW formula.
//
// Why this pattern (vs the previous BMFetch path):
//   Q1's predicate selects ~99% of lineitem (only the last 90 days are
//   filtered out).  BMFetching ~59M rows row-by-row defeats sequential
//   prefetching.  Teacher's path streams the whole 60M-row lineitem
//   table once with 5 cheap bit-tests per row.
//
// Pre-loaded auxiliary structures (PRAGMA load_bitmap, NOT in latency):
//   - bitmap_shipdate    (per-day IndexedX over l_shipdate)
//   - bitmap_linestatus  (per-char IndexedX over l_linestatus, key=ASCII)
//   - bitmap_returnflag  (per-char IndexedX over l_returnflag, key=ASCII)

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

#include "combit_adapter.h"
#include "combit/include/combit.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"
#include "Concise/concise.h"

#include "execution/tpch/bm_baseline_loaders.hpp"
#include "execution/tpch/bm_bench_common.hpp"
#include "execution/tpch/indexed_bitmap.hpp"

#include <algorithm>
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

// 1998-09-02 — TPC-H Q1 cutoff day (epoch days since 1970-01-01).
// Same value as teacher's `right_days = 10471`.
static constexpr int64_t Q1_SHIPDATE_CUTOFF = 10471;

struct Q1GroupAns {
    int64_t sum_qty        = 0;
    int64_t sum_base_price = 0;
    int64_t sum_disc_price = 0;
    int64_t sum_charge     = 0;
    int64_t sum_discount   = 0;
    int64_t count_order    = 0;
};

// Per-row aggregator (matches teacher's q1_exe_aggregation scalar path).
#define Q1_AGG_ROW(g, r, qty, price, disc, tax) do { \
    (g).sum_qty        += (qty)[(r)];                 \
    (g).sum_discount   += (disc)[(r)];                \
    (g).sum_base_price += (price)[(r)];               \
    int64_t _dp = (price)[(r)] * (100 - (disc)[(r)]); \
    (g).sum_disc_price += _dp;                        \
    (g).sum_charge     += _dp * (100 + (tax)[(r)]);   \
    (g).count_order++;                                \
} while (0)

// =========================================================================
// Per-backend "decompress to MSB-first packed byte-stream of length
// (num_rows + 7) / 8".  Counterpart of teacher's reduce_leadingbits.
// =========================================================================
static void q1_to_byte_stream(const ComBit& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    bs.assign((num_rows + 7) / 8, 0);
    // ComBit's L1 literals are already packed MSB-first per byte
    // (see ComBitBtv::compress: `word |= 1 << (7 - bit_in_byte)`).
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        if (word_pos < bs.size()) bs[word_pos] = val;
    });
}

static void q1_to_byte_stream(const roaring::Roaring& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    bs.assign((num_rows + 7) / 8, 0);
    for (auto it = b.begin(); it != b.end(); ++it) {
        uint32_t p = *it;
        if (p < num_rows) bs[p / 8] |= uint8_t(1) << (7 - (p % 8));
    }
}

static void q1_to_byte_stream(const ibis::bitvector& b, std::vector<uint8_t>& bs,
                              size_t num_rows) {
    bs.assign((num_rows + 7) / 8, 0);
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) {
        uint32_t p = *pit;
        if (p < num_rows) bs[p / 8] |= uint8_t(1) << (7 - (p % 8));
        pit.next();
    }
}

static void q1_to_byte_stream(const ewah::EWAHBoolArray<uint64_t>& b,
                              std::vector<uint8_t>& bs, size_t num_rows) {
    bs.assign((num_rows + 7) / 8, 0);
    for (auto it = b.begin(); it != b.end(); ++it) {
        uint32_t p = *it;
        if (p < num_rows) bs[p / 8] |= uint8_t(1) << (7 - (p % 8));
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

// In-place byte-stream complement of `bs` (NOT op).  Masks any padding
// bits past `num_rows` to 0 so they don't spuriously match in the scan.
static void q1_byte_stream_not(std::vector<uint8_t>& bs, size_t num_rows) {
    for (auto& b : bs) b = ~b;
    size_t tail = num_rows & 7;       // valid bits in last byte (MSB-first)
    if (tail != 0) {
        // Keep top `tail` bits, zero the bottom (8 - tail) bits.
        uint8_t mask = static_cast<uint8_t>(0xFFu << (8 - tail));
        bs.back() &= mask;
    }
}

// In-place byte-stream AND: dst[i] &= src[i].
static void q1_byte_stream_and(std::vector<uint8_t>& dst,
                               const std::vector<uint8_t>& src) {
    size_t n = dst.size();
    for (size_t i = 0; i < n; i++) dst[i] &= src[i];
}

// =========================================================================
// q1_run_aggregate — sequential lineitem scan + per-row bit test against
// each group's byte-stream filter + inline aggregate.  Mirrors teacher's
// inner loop (Q1.cpp lines 760-833).
// =========================================================================
static void q1_run_aggregate(
    ClientContext& ctx,
    TableCatalogEntry& lineitem_table,
    DuckTransaction& tx,
    const std::map<std::pair<char,char>, std::vector<uint8_t>>& group_bs,
    std::map<std::pair<char,char>, Q1GroupAns>& ans)
{
    TableScanState scan_state;
    vector<StorageIndex> col_ids = {
        StorageIndex(4),  // l_quantity
        StorageIndex(5),  // l_extendedprice
        StorageIndex(6),  // l_discount
        StorageIndex(7),  // l_tax
    };
    lineitem_table.GetStorage().InitializeScan(ctx, tx, scan_state, col_ids);
    vector<LogicalType> types = {
        lineitem_table.GetColumns().GetColumnTypes()[4],
        lineitem_table.GetColumns().GetColumnTypes()[5],
        lineitem_table.GetColumns().GetColumnTypes()[6],
        lineitem_table.GetColumns().GetColumnTypes()[7],
    };

    // Cache (key, byte-stream-pointer, ans-pointer) for the inner loop.
    struct Slot {
        const uint8_t* bs_base;   // pointer to start of group's byte stream
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

        // Walk 8 rows at a time using byte-aligned access into the
        // group's byte-stream.  row_offset is always a multiple of 8
        // for all but possibly the very last chunk (TPC-H lineitem
        // chunks are 2048 = 256 bytes worth of bits).
        idx_t base = 0;
        while (base + 7 < n) {
            size_t bs_off = (row_offset + base) / 8;
            for (auto& slot : slots) {
                uint8_t bits = slot.bs_base[bs_off];
                if (!bits) continue;
                // 8 rows, MSB-first scan: bit (0x80 >> i) corresponds
                // to chunk row (base + i).
                for (int i = 0; i < 8; i++) {
                    if (bits & (0x80 >> i))
                        Q1_AGG_ROW(*slot.ans_ptr, base + i, qty, price, disc, tax);
                }
            }
            base += 8;
        }
        // Tail handling for partial trailing bits.
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

// =========================================================================
// BMTPCH_Q1 — main entry
// =========================================================================
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

    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    auto* idx_ls   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_linestatus);
    auto* idx_rf   = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_returnflag);
    if (!idx_ship || !idx_ls || !idx_rf) {
        std::cerr << "[Q1] ERROR: required bitmaps not loaded.\n";
        return;
    }
    if (std::string(idx_ship->backend_name()) != idx_ls->backend_name() ||
        std::string(idx_ship->backend_name()) != idx_rf->backend_name()) {
        std::cerr << "[Q1] ERROR: backend mismatch among shipdate/linestatus/returnflag.\n";
        return;
    }
    size_t num_rows = idx_ship->num_rows();

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

        std::vector<uint8_t> shipdate_filter_bs;     // shipdate <= cutoff (after NOT)
        double phaseA = 0;

        // -----------------------------------------------------------------
        // Per-backend: build shipdate_filter_bs + per-(rf, ls) group_bs.
        // The dispatch chain mirrors Q5/Q6 (dynamic_cast on idx_ship).
        // -----------------------------------------------------------------
        if (auto* cb_ship = dynamic_cast<bm_index::IndexedComBit*>(idx_ship)) {
            auto* cb_ls_x = dynamic_cast<bm_index::IndexedComBit*>(idx_ls);
            auto* cb_rf_x = dynamic_cast<bm_index::IndexedComBit*>(idx_rf);
            if (!cb_ls_x || !cb_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            // Phase A: shipdate complement (OR over keys > cutoff, ~90 keys).
            // β scheme: BPE prefix-encoded fast path (active under
            // DEBIT_BM=cb_bpe).  shipdate>cutoff = prefix(LAST) AND_NOT
            // prefix(B) | per-day OR over (cutoff, bucket_max(B)].
            // TPC-H 1.5.7 §5 compliant — same column, alt encoding.
            auto* cb_ship_bpe = dynamic_cast<bm_index::IndexedComBitBPE*>(
                static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_BPE));
            ComBit shipdate_gt;
            if (cb_ship_bpe) {
                int B    = cb_ship_bpe->bucket_of(Q1_SHIPDATE_CUTOFF);
                int LAST = static_cast<int>(cb_ship_bpe->num_keys()) - 1;
                shipdate_gt = *cb_ship_bpe->prefix_at_bucket(LAST);
                shipdate_gt &= ~(*cb_ship_bpe->prefix_at_bucket(B));
                int64_t bmax = cb_ship_bpe->bucket_max(B);
                if (Q1_SHIPDATE_CUTOFF + 1 <= bmax) {
                    ComBit boundary = ComBit::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                    cb_ship->apply_or_range_to(boundary, Q1_SHIPDATE_CUTOFF + 1, bmax);
                    shipdate_gt |= boundary;
                }
            } else {
                // ComBit (no BPE): gather in-range keys + sca or_many.
                std::vector<int64_t> gt_keys;
                cb_ship->for_each_key([&](int64_t k) {
                    if (k > Q1_SHIPDATE_CUTOFF) gt_keys.push_back(k);
                });
                shipdate_gt = cb_ship->or_many(gt_keys);
            }
            q1_to_byte_stream(shipdate_gt, shipdate_filter_bs, num_rows);
            q1_byte_stream_not(shipdate_filter_bs, num_rows);
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            // Phase B: per (rf, ls): ls AND rf → bitmap, byte-stream, AND filter.
            std::map<char, std::vector<uint8_t>> ls_bs, rf_bs;
            for (int li = 0; li < LS_COUNT; li++) {
                ComBit b = ComBit::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                cb_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                std::vector<uint8_t> tmp;
                q1_to_byte_stream(b, tmp, num_rows);
                ls_bs[LS_CHARS[li]] = std::move(tmp);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ComBit b = ComBit::from_sparse_positions({}, num_rows, cb_ship->segment_bits());
                cb_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
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
        else if (auto* cr_ship = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ship)) {
            auto* cr_ls_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_ls);
            auto* cr_rf_x = dynamic_cast<bm_index::IndexedCRoaring*>(idx_rf);
            if (!cr_ls_x || !cr_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            // Plan A: CR pairwise per-day; CRR fastunion via or_range.
            // β scheme: BPE prefix-encoded fast path for cr_bpe.
            auto t_a0 = clk::now();
            auto* cr_ship_bpe = dynamic_cast<bm_index::IndexedCRoaringBPE*>(
                static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_BPE));
            roaring::Roaring shipdate_gt;
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
            } else if (cr_ship->run_optimized()) {
                shipdate_gt = cr_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            } else {
                cr_ship->apply_or_range_to(shipdate_gt, Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            }
            q1_to_byte_stream(shipdate_gt, shipdate_filter_bs, num_rows);
            q1_byte_stream_not(shipdate_filter_bs, num_rows);
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            std::map<char, std::vector<uint8_t>> ls_bs, rf_bs;
            for (int li = 0; li < LS_COUNT; li++) {
                roaring::Roaring b;
                cr_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                std::vector<uint8_t> tmp;
                q1_to_byte_stream(b, tmp, num_rows);
                ls_bs[LS_CHARS[li]] = std::move(tmp);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                roaring::Roaring b;
                cr_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
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
        else if (auto* wah_ship = dynamic_cast<bm_index::IndexedWAH*>(idx_ship)) {
            auto* wah_ls_x = dynamic_cast<bm_index::IndexedWAH*>(idx_ls);
            auto* wah_rf_x = dynamic_cast<bm_index::IndexedWAH*>(idx_rf);
            if (!wah_ls_x || !wah_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            // WAH: copy(first) + decompress + |= rest pairwise (teacher).
            ibis::bitvector shipdate_gt = wah_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            q1_to_byte_stream(shipdate_gt, shipdate_filter_bs, num_rows);
            q1_byte_stream_not(shipdate_filter_bs, num_rows);
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            std::map<char, std::vector<uint8_t>> ls_bs, rf_bs;
            for (int li = 0; li < LS_COUNT; li++) {
                ibis::bitvector b;
                wah_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                std::vector<uint8_t> tmp;
                q1_to_byte_stream(b, tmp, num_rows);
                ls_bs[LS_CHARS[li]] = std::move(tmp);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ibis::bitvector b;
                wah_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
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
        else if (auto* ew_ship = dynamic_cast<bm_index::IndexedEWAH*>(idx_ship)) {
            auto* ew_ls_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_ls);
            auto* ew_rf_x = dynamic_cast<bm_index::IndexedEWAH*>(idx_rf);
            if (!ew_ls_x || !ew_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ewah::EWAHBoolArray<uint64_t> shipdate_gt =
                ew_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            q1_to_byte_stream(shipdate_gt, shipdate_filter_bs, num_rows);
            q1_byte_stream_not(shipdate_filter_bs, num_rows);
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            std::map<char, std::vector<uint8_t>> ls_bs, rf_bs;
            for (int li = 0; li < LS_COUNT; li++) {
                ewah::EWAHBoolArray<uint64_t> b;
                ew_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                std::vector<uint8_t> tmp;
                q1_to_byte_stream(b, tmp, num_rows);
                ls_bs[LS_CHARS[li]] = std::move(tmp);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ewah::EWAHBoolArray<uint64_t> b;
                ew_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
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
        else if (auto* con_ship = dynamic_cast<bm_index::IndexedConcise*>(idx_ship)) {
            auto* con_ls_x = dynamic_cast<bm_index::IndexedConcise*>(idx_ls);
            auto* con_rf_x = dynamic_cast<bm_index::IndexedConcise*>(idx_rf);
            if (!con_ls_x || !con_rf_x) { std::cerr << "[Q1] type mismatch.\n"; return; }

            auto t_a0 = clk::now();
            ConciseSet<false> shipdate_gt =
                con_ship->or_range(Q1_SHIPDATE_CUTOFF + 1, INT64_MAX);
            q1_to_byte_stream(shipdate_gt, shipdate_filter_bs, num_rows);
            q1_byte_stream_not(shipdate_filter_bs, num_rows);
            auto t_a1 = clk::now();
            phaseA = q1_ms(t_a0, t_a1);

            std::map<char, std::vector<uint8_t>> ls_bs, rf_bs;
            for (int li = 0; li < LS_COUNT; li++) {
                ConciseSet<false> b;
                con_ls_x->apply_or_to(b, static_cast<int64_t>(LS_CHARS[li]));
                std::vector<uint8_t> tmp;
                q1_to_byte_stream(b, tmp, num_rows);
                ls_bs[LS_CHARS[li]] = std::move(tmp);
            }
            for (int ri = 0; ri < RF_COUNT; ri++) {
                ConciseSet<false> b;
                con_rf_x->apply_or_to(b, static_cast<int64_t>(RF_CHARS[ri]));
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
        else {
            std::cerr << "[Q1] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }

        auto t_b1 = clk::now();
        // PhaseA was captured per-branch as `phaseA`; PhaseB is everything
        // from end-of-PhaseA up to here (group_bs build).
        double phaseB = q1_ms(t0, t_b1) - phaseA;

        q1_run_aggregate(context.client, lineitem_table, lineitem_tx, group_bs, ans);
        auto t_c1 = clk::now();
        double phaseC = q1_ms(t_b1, t_c1);
        double tot    = q1_ms(t0, t_c1);

        std::cout << "  " << idx_ship->backend_name()
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

    // ----- Validate vs DuckDB SQL -----
    bool gt_ok = false;
    std::map<std::pair<char,char>, std::vector<double>> gt;
    {
        Connection con(*context.client.db);
        const std::string sql =
            "SELECT l_returnflag, l_linestatus, "
            "       sum(l_quantity), sum(l_extendedprice), "
            "       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
            "       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
            "       sum(l_discount), count(*) "
            "FROM   lineitem "
            "WHERE  l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY "
            "GROUP BY l_returnflag, l_linestatus "
            "ORDER BY l_returnflag, l_linestatus";
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
        // Drop empty (zero-count) groups from last_ans before comparing
        // — the cartesian (rf, ls) generation produces 6 entries but
        // SQL's GROUP BY only emits the 4 non-empty groups in TPC-H data.
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
            std::cout << "\n[OK] " << idx_ship->backend_name()
                      << " matches DuckDB SQL ground truth ("
                      << gt.size() << " (rf,ls) groups).\n";
        else
            std::cerr << "\n[FAIL] mismatch vs SQL ground truth!\n";
    }

    std::cout << "\n  Q1 group results (rf,ls): qty / base_price / disc_price / charge / discount / count\n";
    for (auto& [k, g] : last_ans) {
        if (g.count_order == 0) continue;  // skip empty cartesian-product placeholder groups
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
    std::cout << "  Q1 RESULTS — " << idx_ship->backend_name()
              << " only (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (shipdate ~OR + flip): " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (per-group AND→bs)   : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (seq-scan + agg)     : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  TOTAL                       : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    // CSV row
    std::string sf = q1_get_sf_label();
    std::ofstream csv("q1_results_" + sf + ".csv");
    if (csv) {
        csv << std::fixed << std::setprecision(4);
        csv << "sf,operation,"
            << "wah_median_ms,wah_stddev_ms,wah_min_ms,wah_max_ms,"
            << "combit_median_ms,combit_stddev_ms,combit_min_ms,combit_max_ms,"
            << "croaring_median_ms,croaring_stddev_ms,croaring_min_ms,croaring_max_ms,"
            << "croaring_run_median_ms,croaring_run_stddev_ms,croaring_run_min_ms,croaring_run_max_ms,"
            << "ewah_median_ms,ewah_stddev_ms,ewah_min_ms,ewah_max_ms,"
            << "bs_median_ms,bs_stddev_ms,bs_min_ms,bs_max_ms,"
            << "bsa_median_ms,bsa_stddev_ms,bsa_min_ms,bsa_max_ms,"
            << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms,"
            << "cb_vs_wah,cr_vs_wah,crr_vs_wah,ew_vs_wah,bs_vs_wah,bsa_vs_wah,con_vs_wah\n";
        bm_bench::Stats z{0,0,0,0};
        std::string bn = idx_ship->backend_name();
        auto put = [&](const std::string& op, bm_bench::Stats s) {
            csv << sf << "," << op << ",";
            auto cell = [&](bm_bench::Stats v) {
                csv << v.median << "," << v.stddev << "," << v.min_val << "," << v.max_val << ",";
            };
            cell(bn == "WAH" ? s : z);
            cell(bn == "ComBit" ? s : z);
            cell(bn == "CRoaring" ? s : z);
            cell(bn == "CRoaringRun" ? s : z);
            cell(bn == "EWAH" ? s : z);
            cell(z); cell(z);
            cell(bn == "Concise" ? s : z);
            csv << "0,0,0,0,0,0,0\n";
        };
        put("PhaseA", sA);
        put("PhaseB", sB);
        put("PhaseC", sC);
        put("TOTAL",  sT);
        std::cout << "  [CSV] q1_results_" << sf << ".csv\n";
    }

    });  // end call_once
}

}  // namespace duckdb
