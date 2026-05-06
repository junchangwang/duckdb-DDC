// =============================================================================
// IndexedBitmap — per-backend value-indexed bitmap (mirror of teacher's
// rabit::Rabit pattern).  Built once from a lineitem column scan and cached
// in the client context.  Used by Q1 / Q5 / Q6 BMTPCH paths to do bitmap
// multi-OR over qualifying value sets (replaces SQL JOIN).
//
// Storage (unified harness — Plan 1-B flat inverted index):
//   All per-value IndexedX share one flat inverted-index storage:
//     sorted_keys_     : K distinct keys, sorted ascending        (8 * K bytes)
//     key_offsets_     : (K+1)-sized offsets into all_positions_  (4 * (K+1))
//     all_positions_   : concatenated sorted row IDs per key      (4 * N bytes)
//   Lookup:  binary_search(sorted_keys_, k) → idx; positions are
//            all_positions_[key_offsets_[idx] .. key_offsets_[idx+1]).
//
// TPC-H 1.5.7 (Page 20 spec): single base table + single column (PK/FK/date);
// Comment explicitly permits "row IDs" as the auxiliary structure content.
// This flat inverted index is the textbook form of that permission.
//
// Memory accounting (layer_breakdown / storage_bytes):
//   Reports the per-backend SERIALIZED bitmap projection — compressed
//   data bytes each library would write as the auxiliary structure,
//   EXCLUDING C++ class skeletons (sizeof(SparseComBit),
//   sizeof(roaring::Roaring), sizeof(ibis::bitvector), etc.) and
//   std::vector capacity padding.  The C++ object shell is a
//   runtime-implementation choice of the "one bitmap per key" design,
//   not part of the compression scheme; 15M × sizeof(struct) would
//   otherwise dominate every FK column and hide the real compression
//   differences between backends.
//
//     ComBit / ComBitGE → ultra / L1 / L2 / L3 / L4 / header
//                         (ultra = n×4 position bytes, L* = literal
//                          bytes from size_breakdown, header = 4 B per
//                          non-empty segment manifest)
//     CRoaring / CRR    → array / bitset / run / header
//                         (containers via roaring_statistics_t + any
//                          remaining portable-serialized overhead)
//     WAH      → wah     (getSerialSize = compressed word stream bytes)
//     EWAH     → ewah    (sizeInBytes   = RLW-compressed buffer bytes)
//     Concise  → concise (sizeInBytes   = compressed word stream bytes)
//
//   Computed once per index by walking for_each_key_positions and
//   shadow-building the native bitmap per key (discarded after sizing).
//   Cached for subsequent calls.  Runtime OR is unchanged — it still
//   uses the flat scaffolding for cache-friendly merge-walk + native
//   scatter (e.g. ComBit::scatter_or_decompressed).
//
// Rationale for flat layout (vs. per-backend unordered_map<key, Bitmap>):
//   - 15M keys × ~32-96 B header overhead each dominated memory on FK
//     columns (SparseComBit header alone was 1.44 GB for l_orderkey SF10).
//   - Flat layout puts every backend on the same OR-scaffolding footing,
//     so per-Q runtime measurements are comparable; meanwhile the native
//     projection lets memory measurements remain comparable too.
//
// Per-backend or_many strategy:
//   ComBit   → dst.scatter_or_decompressed(positions, n) per key
//              (native hot path; bypasses library's dense OR walk).
//   CRoaring → dst.addMany(n, positions) per key
//              (Roaring's batch insertion with sorted input).
//   WAH      → build scratch ibis::bitvector from positions (appendFill +
//              += 1) per key, dst |= scratch.
//   EWAH     → build scratch EWBA from positions (set()) per key,
//              dst = dst.logicalor(scratch).
//   Concise  → build scratch CS from positions (append() for monotonic)
//              per key, dst = dst | scratch.
//
// Apply pattern (Q5, unchanged teacher code):
//   for (auto& [okey, _] : order_nation_map)
//       idx_orderkey.apply_or_to(btv_res, okey);
// =============================================================================

#pragma once

#include "combit/include/combit.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"
#include "Concise/concise.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace bm_index {

// [D1] Compute the SERIALIZED size of `n` sorted positions encoded as
// chunked uint16_t (Roaring's array-container layout: split high-16 +
// low-16, group by high-16 chunk).  Per chunk: 2 B chunk_high_id +
// 2 B count + count × 2 B low positions = 4 + 2n_chunk bytes.
//
// This is the shadow-projection size for ComBit's ultra path under D1
// — what the position list WOULD compress to if stored chunked instead
// of as flat uint32_t.  Aligns ComBit's ultra report with CRoaring's
// array-container report so the two bitmap-index families are compared
// at the same encoding granularity.
inline size_t chunked_uint16_size(const uint32_t* positions, size_t n) {
    if (n == 0) return 0;
    // Per-chunk header: 4 B (chunk_high_16 + count_uint16).
    // Per-position: 2 B (low_uint16).
    constexpr size_t HEADER_PER_CHUNK = 4;
    constexpr size_t BYTES_PER_POS    = 2;
    size_t bytes = 0;
    uint16_t cur_chunk = static_cast<uint16_t>(positions[0] >> 16);
    bytes += HEADER_PER_CHUNK;        // first chunk
    for (size_t i = 0; i < n; i++) {
        uint16_t chunk_id = static_cast<uint16_t>(positions[i] >> 16);
        if (chunk_id != cur_chunk) {
            bytes += HEADER_PER_CHUNK;  // new chunk header
            cur_chunk = chunk_id;
        }
        bytes += BYTES_PER_POS;
    }
    return bytes;
}


// -----------------------------------------------------------------------
// IBitmapIndex — abstract base (mirror of teacher's BaseTable from CUBIT).
// All per-backend variants inherit from this so context.client.bitmap_*
// can hold any backend's index uniformly.  BMTPCH functions dispatch via
// dynamic_cast (mirror teacher's `dynamic_cast<rabit::Rabit*>` pattern).
// -----------------------------------------------------------------------
class IBitmapIndex {
public:
    virtual ~IBitmapIndex() = default;
    virtual size_t num_rows() const = 0;
    virtual size_t num_keys() const = 0;
    virtual size_t storage_bytes() const = 0;
    virtual const char* backend_name() const = 0;
    // Per-layer storage breakdown in BYTES.  Categories:
    //   Per-value IndexedX (flat harness): positions / offsets / keys
    //   ComBitBPE: L1 / L2 / L3 / L4 / header (per-bucket dense ComBit)
    //   CRoaringBPE: array / bitset / run / header (per-bucket Roaring)
    //   Default (no breakdown): {"total": storage_bytes()}.
    // Subclasses override to expose layer split — drives Memory Detail
    // (MB) sheet rows in bench_suite.
    virtual std::vector<std::pair<std::string, size_t>> layer_breakdown() const {
        return {{"total", storage_bytes()}};
    }
};

// -----------------------------------------------------------------------
// InvertedIndex — flat-storage base shared by every per-value IndexedX.
//
// Owns the (sorted_keys, key_offsets, all_positions) triple.  Each
// IndexedX derives from this + IBitmapIndex; backend-specific code
// only needs to know how to translate a (positions, count) range into
// its native bitmap's OR operation.
//
// Build cost is one lineitem scan + one radix bucketing pass; total ~600 ms
// for 60M rows / 15M distinct keys at SF10 — matches the prior harness.
// -----------------------------------------------------------------------
class InvertedIndex {
public:
    // Build the flat inverted index from `keys` (one entry per row).
    // Caller passes the column values in row order; we bucket them into
    // per-key position lists, then emit a sorted flat layout.
    //
    // [D2] sorted_keys_ stored as uint32_t (was int64_t).  All TPC-H SF10/SF100
    // key domains fit in 32 bits (max l_orderkey=60M < 2^26, max date raw <
    // 2^14).  External API still takes int64_t for source-compat with Q files;
    // values are bounds-checked at build time and stored compactly.
    void build_inverted_index(const std::vector<int64_t>& keys, size_t num_rows) {
        num_rows_ = num_rows;
        // Pass 1: bucket row IDs by key.  unordered_map probe is one
        // hash lookup per row; for 60M rows at SF10 this is ~400 ms,
        // matching the bucket-by-key step in the prior harness.
        std::unordered_map<int64_t, std::vector<uint32_t>> by_key;
        by_key.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_key[keys[i]].push_back(static_cast<uint32_t>(i));
        // Pass 2: emit sorted_keys_, key_offsets_, all_positions_.
        sorted_keys_.reserve(by_key.size());
        for (auto& [k, _] : by_key) {
            // [D2] Bounds check: key must fit in uint32_t.  Negative keys are
            // also rejected (TPC-H doesn't have any negative-keyed columns).
            if (k < 0 || k > static_cast<int64_t>(UINT32_MAX)) {
                std::cerr << "[InvertedIndex] FATAL: key " << k
                          << " out of uint32_t range (0..2^32-1).  "
                             "Need int64 sorted_keys_ for this column.\n";
                std::abort();
            }
            sorted_keys_.push_back(static_cast<uint32_t>(k));
        }
        std::sort(sorted_keys_.begin(), sorted_keys_.end());
        key_offsets_.resize(sorted_keys_.size() + 1);
        size_t total = 0;
        for (auto& v : by_key) total += v.second.size();
        all_positions_.reserve(total);
        for (size_t i = 0; i < sorted_keys_.size(); i++) {
            key_offsets_[i] = static_cast<uint32_t>(all_positions_.size());
            auto& pos = by_key[static_cast<int64_t>(sorted_keys_[i])];
            // Positions are already monotonic (row-order scan), but sort
            // defensively — WAH/EWAH/Concise require monotonic input.
            std::sort(pos.begin(), pos.end());
            all_positions_.insert(all_positions_.end(), pos.begin(), pos.end());
        }
        key_offsets_.back() = static_cast<uint32_t>(all_positions_.size());
    }

    // Lookup positions for a single key.  Returns (nullptr, 0) if absent.
    // Binary search over sorted_keys_ is cache-friendly (~60 MB for 15M
    // uint32_t keys, fits well in L3) and avoids unordered_map's pointer-chasing.
    std::pair<const uint32_t*, size_t> positions_for(int64_t key) const {
        // [D2] sorted_keys_ is uint32_t; reject out-of-range keys gracefully.
        if (key < 0 || key > static_cast<int64_t>(UINT32_MAX)) return {nullptr, 0};
        uint32_t key32 = static_cast<uint32_t>(key);
        auto it = std::lower_bound(sorted_keys_.begin(), sorted_keys_.end(), key32);
        if (it == sorted_keys_.end() || *it != key32) return {nullptr, 0};
        size_t idx = static_cast<size_t>(it - sorted_keys_.begin());
        uint32_t lo = key_offsets_[idx];
        uint32_t hi = key_offsets_[idx + 1];
        return {all_positions_.data() + lo, static_cast<size_t>(hi - lo)};
    }

    // Same but returns the index into sorted_keys_ for range iteration.
    size_t lower_bound_idx(int64_t lo) const {
        // [D2] Clamp to uint32_t range.
        uint32_t lo32 = (lo <= 0) ? 0u
                      : (lo > static_cast<int64_t>(UINT32_MAX) ? UINT32_MAX
                                                                : static_cast<uint32_t>(lo));
        auto it = std::lower_bound(sorted_keys_.begin(), sorted_keys_.end(), lo32);
        return static_cast<size_t>(it - sorted_keys_.begin());
    }
    size_t upper_bound_idx(int64_t hi) const {
        // [D2] Clamp to uint32_t range.
        uint32_t hi32 = (hi <= 0) ? 0u
                      : (hi > static_cast<int64_t>(UINT32_MAX) ? UINT32_MAX
                                                                : static_cast<uint32_t>(hi));
        auto it = std::upper_bound(sorted_keys_.begin(), sorted_keys_.end(), hi32);
        return static_cast<size_t>(it - sorted_keys_.begin());
    }
    std::pair<const uint32_t*, size_t> positions_at(size_t idx) const {
        uint32_t lo = key_offsets_[idx];
        uint32_t hi = key_offsets_[idx + 1];
        return {all_positions_.data() + lo, static_cast<size_t>(hi - lo)};
    }

    bool has_key(int64_t key) const {
        // [D2] uint32_t storage; reject out-of-range keys.
        if (key < 0 || key > static_cast<int64_t>(UINT32_MAX)) return false;
        return std::binary_search(sorted_keys_.begin(), sorted_keys_.end(),
                                  static_cast<uint32_t>(key));
    }

    template <typename F>
    void for_each_key_base(F&& f) const {
        // [D2] Promote uint32_t back to int64_t for caller compat.
        for (uint32_t k : sorted_keys_) f(static_cast<int64_t>(k));
    }

    // Merge-walk over (sorted_input, sorted_keys_) — invokes
    // f(positions_ptr, count) for every input key that matches an
    // indexed key.  O(|sorted_input| + |sorted_keys_|) sequential scan,
    // which is much faster than |sorted_input| binary searches on a
    // large backing store (cache-miss storm on >L3 sorted_keys_).
    //
    // Pre: sorted_input is ascending and deduped.  Caller is responsible
    // for sorting the iterable into a local buffer first.
    template <typename F>
    void for_each_matched_key(const int64_t* sorted_input, size_t ni, F&& f) const {
        size_t i = 0, j = 0;
        const size_t nj = sorted_keys_.size();
        while (i < ni && j < nj) {
            int64_t a = sorted_input[i];
            // [D2] Skip out-of-range input keys — they can't match anything.
            if (a < 0 || a > static_cast<int64_t>(UINT32_MAX)) { ++i; continue; }
            uint32_t a32 = static_cast<uint32_t>(a);
            uint32_t b   = sorted_keys_[j];
            if (a32 == b) {
                uint32_t lo = key_offsets_[j];
                uint32_t hi = key_offsets_[j + 1];
                f(all_positions_.data() + lo, static_cast<size_t>(hi - lo));
                ++i; ++j;
            } else if (b < a32) {
                ++j;
            } else {
                ++i;
            }
        }
    }

    // Convenience: collect input iterable into a sorted+deduped vector,
    // then merge-walk invoking f(positions_ptr, count) for each match.
    // Returns the number of matched keys (useful for seed detection).
    template <typename Iterable, typename F>
    size_t or_many_walk(const Iterable& keys, F&& f) const {
        if (sorted_keys_.empty()) return 0;
        std::vector<int64_t> in;
        for (auto& k : keys) in.push_back(static_cast<int64_t>(k));
        if (in.empty()) return 0;
        std::sort(in.begin(), in.end());
        in.erase(std::unique(in.begin(), in.end()), in.end());
        size_t matched = 0;
        for_each_matched_key(in.data(), in.size(),
            [&](const uint32_t* p, size_t n) {
                f(p, n);
                ++matched;
            });
        return matched;
    }

    size_t num_distinct_keys() const { return sorted_keys_.size(); }
    size_t num_rows_base()     const { return num_rows_; }

    // Iterate every key's (positions_ptr, count) in ascending key order.
    // Used by per-value IndexedX::compute_layers() to drive a per-key
    // shadow build → native size accumulation → discard, without paying
    // the runtime cost (called once per index from the layer-breakdown
    // memoization path).
    template <typename F>
    void for_each_key_positions(F&& f) const {
        for (size_t i = 0, k = sorted_keys_.size(); i < k; i++) {
            uint32_t lo = key_offsets_[i];
            uint32_t hi = key_offsets_[i + 1];
            f(all_positions_.data() + lo, static_cast<size_t>(hi - lo));
        }
    }

    // Total memory held by the flat inverted index (all three vectors).
    // This is the BUILD scaffold cost — kept as the runtime auxiliary
    // structure for fast OR scatter (positions are already sorted per
    // key).  It is NOT the per-backend native bitmap footprint; that
    // is reported by layer_breakdown() via cached_native_layers().
    size_t inverted_index_bytes() const {
        return all_positions_.capacity() * sizeof(uint32_t)
             + key_offsets_.capacity()   * sizeof(uint32_t)
             + sorted_keys_.capacity()   * sizeof(uint32_t);  // [D2] was sizeof(int64_t)
    }

    // Memoize per-backend native size projection.  Each per-value
    // IndexedX defines compute_layers() that walks for_each_key_positions
    // and shadow-builds its native bitmap (e.g. SparseComBit per key for
    // ComBit, roaring::Roaring per key for CRoaring) to accumulate the
    // backend-specific layer split.  The result is cached on first call
    // (load time) and reused for both layer_breakdown() and
    // storage_bytes() — Net cost: one shadow-build pass per loaded
    // column, paid in [load_bitmap] phase, not in Q runtime.
    template <typename Compute>
    const std::vector<std::pair<std::string, size_t>>&
    cached_native_layers(Compute&& compute) const {
        if (!native_cached_) {
            native_layers_ = compute();
            size_t t = 0;
            for (auto& kv : native_layers_) t += kv.second;
            native_total_bytes_ = t;
            native_cached_ = true;
        }
        return native_layers_;
    }
    size_t cached_native_total_bytes() const { return native_total_bytes_; }

protected:
    size_t num_rows_ = 0;
    std::vector<uint32_t> all_positions_;  // concatenated, per-key sorted
    std::vector<uint32_t> key_offsets_;    // (K+1)-sized prefix-sum
    std::vector<uint32_t> sorted_keys_;    // [D2] uint32_t, K-sized, ascending
                                            // (was int64_t — saves 4 B/key,
                                            //  60 MB on l_orderkey at SF10)

    // Native projection cache.  See cached_native_layers().
    mutable bool   native_cached_      = false;
    mutable size_t native_total_bytes_ = 0;
    mutable std::vector<std::pair<std::string, size_t>> native_layers_;
};

// -----------------------------------------------------------------------
// IndexedComBit — flat-harness per-value ComBit index.
//
// Storage: inherited InvertedIndex triple (sorted_keys / offsets / positions).
// OR-ops use ComBit's native scatter_or_decompressed hot path — positions
// are poked directly into the seeded Decompressed-empty dst's L1 byte array
// (bypasses L2/L3/L4 walk, no per-key bitmap construction).
//
// segment_bits is the only ComBit-specific build knob; defaults to 4096
// (matches Q* code paths that seed btv_res with segment_bits_).
// -----------------------------------------------------------------------
class IndexedComBit : public IBitmapIndex, public InvertedIndex {
public:
    IndexedComBit() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows,
               size_t segment_bits = 4096) {
        segment_bits_ = segment_bits;
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

    // OR positions_for(key) into dst via scatter_or_decompressed — the
    // fast path from combit.h:568, which pokes raw positions into the
    // Decompressed dst's L1 byte array without L2/L3/L4 traversal.
    void apply_or_to(ComBit& dst, int64_t key) const {
        auto [p, n] = positions_for(key);
        if (n) dst.scatter_or_decompressed(p, n);
    }

    // Apply OR for all keys in [lo, hi] (inclusive).  Used by Q1 (range
    // OR over shipdate days) and Q6 (range OR over discount/quantity).
    void apply_or_range_to(ComBit& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (n) dst.scatter_or_decompressed(p, n);
        }
    }

    // K-way OR: sort+dedupe input keys, then merge-walk sorted_keys_
    // and scatter positions of matched keys into a seeded
    // Decompressed-empty dst.  Merge-walk avoids the cache-miss storm
    // that K unsorted binary searches incur on a >L3 sorted_keys_
    // (l_orderkey SF10: 15M keys / 120 MB — random probes miss L3 by
    // ~20 cache lines each, Q5's 28k unsorted-input keys cost ~100 ms
    // in pure cache-miss latency with naive binary search).
    // Used by Q3 (590k orderkey OR), Q4, Q5, Q8, Q14, Q17.
    template <typename Iterable>
    ComBit or_many(const Iterable& keys) const {
        ComBit dst = ComBit::from_sparse_positions({}, num_rows_, segment_bits_);
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            dst.scatter_or_decompressed(p, n);
        });
        return dst;
    }

    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    size_t num_keys()     const override { return num_distinct_keys(); }
    size_t num_rows()     const override { return num_rows_; }
    size_t storage_bytes() const override {
        cached_native_layers([this]{ return compute_layers(); });
        return cached_native_total_bytes();
    }
    size_t segment_bits() const { return segment_bits_; }
    const char* backend_name() const override { return "ComBit"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    size_t segment_bits_ = 4096;

    // Shadow-build per-key SparseComBit, measure its SERIALIZED form —
    // i.e. the compressed data bytes only, excluding C++ class skeleton
    // (sizeof(SparseComBit), sizeof(ComBitBtv), std::vector capacity
    // padding).  Runtime C++ overhead is an implementation detail of
    // the "one object per key" choice, not of the compression scheme;
    // for a fair TPC-H §5 auxiliary-structure comparison we strip it.
    //
    // Layer split:
    //   ultra  — chunked uint16_t encoding (Roaring array-container
    //            layout): high-16 / low-16 split, per chunk 4 B header +
    //            2 B per position.  [D1] was n × 4 B flat uint32_t.
    //   L1/L2/L3/L4 — ComBit's hierarchical literal bytes per segment
    //   header — per-segment manifest (seg_indices_, 4 B per non-empty
    //            segment); small and NOT counting the ComBitBtv struct.
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t l1 = 0, l2 = 0, l3 = 0, l4 = 0, header = 0, ultra = 0;
        std::vector<uint32_t> tmp;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            tmp.assign(p, p + n);
            SparseComBit s = SparseComBit::from_positions(tmp, num_rows_, segment_bits_);
            if (s.is_ultra_sparse()) {
                // [D1] Auto-pick per key: report min(flat uint32_t, chunked
                // uint16_t).  Chunked wins when bits cluster (avg > 2 per
                // 64K-chunk); flat wins when bits scatter sparsely (avg
                // < 2 per chunk, e.g. partkey 30 bits / 916 chunks).
                size_t flat    = n * sizeof(uint32_t);
                size_t chunked = chunked_uint16_size(p, n);
                ultra += (chunked < flat) ? chunked : flat;
            } else {
                for (const auto& seg : s.seg_data()) {
                    auto sb = seg.size_breakdown();
                    l1 += sb.l1_literal_bits / 8;
                    l2 += sb.l2_literal_bits / 8;
                    l3 += sb.l3_literal_bits / 8;
                    l4 += (sb.l4_bits + 7) / 8;
                }
                header += s.seg_indices().size() * sizeof(uint32_t);
            }
        });
        return {{"ultra", ultra}, {"L1", l1}, {"L2", l2},
                {"L3", l3}, {"L4", l4}, {"header", header}};
    }
};

// -----------------------------------------------------------------------
// IndexedComBitGE — Group Encoding variant of IndexedComBit.
//
// Same flat-harness storage as IndexedComBit; keys are GROUP IDs
// (e.g. year for shipdate, decade-bucket for quantity).  Mirrors teacher's
// rabit::Btvs_GE pattern (one bitmap per group, accessed by group index).
//
// Functionally identical to IndexedComBit at this layer; the GE-ness is
// in how it was BUILT (keys = group IDs, not raw values).  Kept distinct
// so dynamic_cast in Q6 can route to GE-aware logic.
// -----------------------------------------------------------------------
class IndexedComBitGE : public IBitmapIndex, public InvertedIndex {
public:
    IndexedComBitGE() = default;

    void build(const std::vector<int64_t>& group_ids, size_t num_rows,
               size_t segment_bits = 4096) {
        segment_bits_ = segment_bits;
        build_inverted_index(group_ids, num_rows);
    }

    bool has(int64_t group_id) const { return has_key(group_id); }

    void apply_or_to(ComBit& dst, int64_t group_id) const {
        auto [p, n] = positions_for(group_id);
        if (n) dst.scatter_or_decompressed(p, n);
    }

    void apply_or_range_to(ComBit& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (n) dst.scatter_or_decompressed(p, n);
        }
    }

    template <typename Iterable>
    ComBit or_many(const Iterable& keys) const {
        ComBit dst = ComBit::from_sparse_positions({}, num_rows_, segment_bits_);
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            dst.scatter_or_decompressed(p, n);
        });
        return dst;
    }

    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    size_t num_keys()     const override { return num_distinct_keys(); }
    size_t num_rows()     const override { return num_rows_; }
    size_t storage_bytes() const override {
        cached_native_layers([this]{ return compute_layers(); });
        return cached_native_total_bytes();
    }
    size_t segment_bits() const { return segment_bits_; }
    const char* backend_name() const override { return "ComBitGE"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    size_t segment_bits_ = 4096;

    // Identical serialized projection as IndexedComBit::compute_layers
    // — GE only affects how keys were chosen (group IDs), not per-key
    // storage.  See parent class for why C++ struct overhead is dropped.
    // [D1] ultra path uses chunked uint16_t (= Roaring array-container size).
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t l1 = 0, l2 = 0, l3 = 0, l4 = 0, header = 0, ultra = 0;
        std::vector<uint32_t> tmp;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            tmp.assign(p, p + n);
            SparseComBit s = SparseComBit::from_positions(tmp, num_rows_, segment_bits_);
            if (s.is_ultra_sparse()) {
                // [D1] Auto-pick per key: report min(flat uint32_t, chunked
                // uint16_t).  Chunked wins when bits cluster (avg > 2 per
                // 64K-chunk); flat wins when bits scatter sparsely (avg
                // < 2 per chunk, e.g. partkey 30 bits / 916 chunks).
                size_t flat    = n * sizeof(uint32_t);
                size_t chunked = chunked_uint16_size(p, n);
                ultra += (chunked < flat) ? chunked : flat;
            } else {
                for (const auto& seg : s.seg_data()) {
                    auto sb = seg.size_breakdown();
                    l1 += sb.l1_literal_bits / 8;
                    l2 += sb.l2_literal_bits / 8;
                    l3 += sb.l3_literal_bits / 8;
                    l4 += (sb.l4_bits + 7) / 8;
                }
                header += s.seg_indices().size() * sizeof(uint32_t);
            }
        });
        return {{"ultra", ultra}, {"L1", l1}, {"L2", l2},
                {"L3", l3}, {"L4", l4}, {"header", header}};
    }
};

// -----------------------------------------------------------------------
// IndexedComBitBPE — Bucketed Prefix-Encoded ComBit.
//
// Stores a cumulative bitmap per bucket: `bitmaps_pe_[b]` = rows where
// value ≤ bucket_b's upper boundary.  Range query `lo ≤ value ≤ hi`
// reduces to `bitmaps_pe_[hi_bucket] AND NOT bitmaps_pe_[lo_bucket - 1]`
// — O(1) bitmap ops regardless of range cardinality.
//
// For non-bucket-aligned ranges, the caller composes the result with
// boundary OR over the matching per-value IndexedComBit (also loaded
// over the same column).  Boundary cost ≤ bucket_size per side.
//
// TPC-H 1.5.7 §5: same column, separate auxiliary structure — allowed.
// Build is amortised over all subsequent queries; not in Q latency.
// -----------------------------------------------------------------------
class IndexedComBitBPE : public IBitmapIndex {
public:
    IndexedComBitBPE() = default;

    // Build cumulative PE bitmaps for [lo_key, hi_key] in `bucket_size`
    // increments.  Bucket b covers raw values
    //   [lo_key + b*bucket_size, lo_key + (b+1)*bucket_size - 1]
    // and `bitmaps_pe_[b]` = rows where value ≤ bucket_max(b).
    void build(const std::vector<int64_t>& values, size_t num_rows,
               int64_t lo_key, int64_t hi_key, int64_t bucket_size,
               size_t segment_bits = 4096) {
        num_rows_     = num_rows;
        lo_key_       = lo_key;
        hi_key_       = hi_key;
        bucket_size_  = bucket_size;
        segment_bits_ = segment_bits;
        size_t num_buckets = static_cast<size_t>((hi_key - lo_key) / bucket_size + 1);

        // Bucket positions (per-bucket equality position lists).
        std::vector<std::vector<uint32_t>> per_bucket_pos(num_buckets);
        for (size_t i = 0; i < num_rows; i++) {
            int64_t v = values[i];
            if (v < lo_key || v > hi_key) continue;
            size_t b = static_cast<size_t>((v - lo_key) / bucket_size);
            if (b < num_buckets) per_bucket_pos[b].push_back(static_cast<uint32_t>(i));
        }

        // Cumulative OR — bitmaps_pe_[b] = bitmaps_pe_[b-1] | bucket_eq[b].
        bitmaps_pe_.reserve(num_buckets);
        ComBit cum = ComBit::from_sparse_positions({}, num_rows, segment_bits);
        for (size_t b = 0; b < num_buckets; b++) {
            std::sort(per_bucket_pos[b].begin(), per_bucket_pos[b].end());
            ComBit bucket_eq = ComBit::from_sparse_positions(
                per_bucket_pos[b], num_rows, segment_bits);
            cum |= bucket_eq;
            bitmaps_pe_.push_back(cum);  // copy of cumulative state
        }
    }

    int bucket_of(int64_t key) const {
        if (key < lo_key_)         return -1;
        if (key > hi_key_)         return static_cast<int>(bitmaps_pe_.size()) - 1;
        return static_cast<int>((key - lo_key_) / bucket_size_);
    }
    int64_t bucket_max(int b)  const { return lo_key_ + (b + 1) * bucket_size_ - 1; }
    int64_t bucket_min(int b)  const { return lo_key_ + b * bucket_size_; }
    int64_t lo_key()           const { return lo_key_; }
    int64_t hi_key()           const { return hi_key_; }
    int64_t bucket_size_int()  const { return bucket_size_; }
    size_t  segment_bits()     const { return segment_bits_; }

    // Cumulative bitmap "rows where value ≤ bucket_max(b)".
    // `b == -1` returns an empty bitmap (no rows).
    // `b ≥ num_buckets` returns the all-rows bitmap (cap to last).
    const ComBit* prefix_at_bucket(int b) const {
        if (b < 0) return nullptr;
        size_t idx = std::min<size_t>(static_cast<size_t>(b), bitmaps_pe_.size() - 1);
        return &bitmaps_pe_[idx];
    }

    size_t num_keys() const override { return bitmaps_pe_.size(); }
    size_t num_rows() const override { return num_rows_; }
    size_t storage_bytes() const override {
        size_t t = 0;
        for (const auto& bm : bitmaps_pe_)
            for (const auto& seg : bm.segments())
                t += seg.compressed_size_bytes();
        return t;
    }
    const char* backend_name() const override { return "ComBitBPE"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        // ComBitBPE uses dense ComBit per bucket — always hierarchy mode,
        // no ultra-sparse path.  ultra=0 entry kept for cross-comparison
        // with IndexedComBit / IndexedComBitGE columns.
        size_t l1 = 0, l2 = 0, l3 = 0, l4 = 0, header = 0;
        for (const auto& bm : bitmaps_pe_) {
            header += sizeof(ComBit) + bm.segments().size() * sizeof(ComBitBtv);
            auto bd = bm.size_breakdown();
            l1 += bd.l1_literal_bits / 8;
            l2 += bd.l2_literal_bits / 8;
            l3 += bd.l3_literal_bits / 8;
            l4 += bd.l4_bits          / 8;
        }
        return {{"ultra",  size_t{0}},
                {"L1",     l1},
                {"L2",     l2},
                {"L3",     l3},
                {"L4",     l4},
                {"header", header}};
    }

private:
    size_t   num_rows_     = 0;
    int64_t  lo_key_       = 0;
    int64_t  hi_key_       = 0;
    int64_t  bucket_size_  = 1;
    size_t   segment_bits_ = 4096;
    std::vector<ComBit> bitmaps_pe_;
};

// -----------------------------------------------------------------------
// IndexedCRoaring — flat-harness per-value Roaring index.
//
// Harness: InvertedIndex triple (same as ComBit/WAH/EWAH/Concise).
// Per-key OR uses Roaring's native addMany(n, positions) batch insertion
// (sorted uint32_t run through the Roaring API's container-aware path).
// Multi-key union accumulates into one dst Roaring instead of materialising
// K Roaring objects upfront + fastunion — the flat harness means we don't
// have pre-built per-key Roarings anyway, so this is the one-touch path.
// run_optimized applies to the result bitmap only (once, after OR).
// -----------------------------------------------------------------------
class IndexedCRoaring : public IBitmapIndex, public InvertedIndex {
public:
    IndexedCRoaring() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows,
               bool run_optimize = false) {
        run_optimized_ = run_optimize;
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

    void apply_or_to(roaring::Roaring& dst, int64_t key) const {
        auto [p, n] = positions_for(key);
        if (n) dst.addMany(n, p);
    }
    void apply_or_range_to(roaring::Roaring& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (n) dst.addMany(n, p);
        }
    }
    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    template <typename Iterable>
    roaring::Roaring or_many(const Iterable& keys) const {
        roaring::Roaring dst;
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            dst.addMany(n, p);
        });
        if (run_optimized_) dst.runOptimize();
        return dst;
    }
    roaring::Roaring or_range(int64_t lo, int64_t hi) const {
        roaring::Roaring dst;
        apply_or_range_to(dst, lo, hi);
        if (run_optimized_) dst.runOptimize();
        return dst;
    }

    size_t num_keys()     const override { return num_distinct_keys(); }
    size_t num_rows()     const override { return num_rows_; }
    size_t storage_bytes() const override {
        cached_native_layers([this]{ return compute_layers(); });
        return cached_native_total_bytes();
    }
    const char* backend_name() const override { return run_optimized_ ? "CRoaringRun" : "CRoaring"; }
    bool run_optimized() const { return run_optimized_; }
    void mark_run_optimized() {
        if (run_optimized_) return;
        run_optimized_ = true;
        // Run-optimize toggle changes the projection — invalidate cache.
        native_cached_ = false;
    }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    bool run_optimized_ = false;

    // Shadow-build per-key roaring::Roaring (apply runOptimize if the
    // index is run-optimised mode); report the SERIALIZED (portable)
    // form — container bytes + portable manifest (~8 B/bitmap).  Excludes
    // the C++ roaring::Roaring wrapper and in-memory container pointer
    // arrays; these are runtime implementation details, not part of
    // Roaring's compression format.
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t arr = 0, bset = 0, run = 0, total = 0;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            roaring::Roaring r;
            r.addMany(n, p);
            if (run_optimized_) r.runOptimize();
            roaring::api::roaring_statistics_t st;
            roaring::api::roaring_bitmap_statistics(&r.roaring, &st);
            arr   += st.n_bytes_array_containers;
            bset  += st.n_bytes_bitset_containers;
            run   += st.n_bytes_run_containers;
            total += r.getSizeInBytes(/*portable=*/true);
        });
        size_t containers = arr + bset + run;
        size_t header = (total > containers) ? total - containers : 0;
        return {{"array", arr}, {"bitset", bset}, {"run", run}, {"header", header}};
    }
};

// -----------------------------------------------------------------------
// IndexedCRoaringBPE — Bucketed Prefix-Encoded CRoaring.
//
// Same data layout as IndexedComBitBPE but per-bucket bitmap is a
// roaring::Roaring.  Build always applies runOptimize on the prefix
// bitmaps (they're typically dense → run-encoded).
//
// We expect minimal speed-up over plain IndexedCRoaring + fastunion
// for Q6 (CR's container layout already amortises k-way OR via 64Ki-bit
// chunks) — but the structure is added for fair apples-to-apples
// comparison vs ComBit BPE.
// -----------------------------------------------------------------------
class IndexedCRoaringBPE : public IBitmapIndex {
public:
    IndexedCRoaringBPE() = default;

    void build(const std::vector<int64_t>& values, size_t num_rows,
               int64_t lo_key, int64_t hi_key, int64_t bucket_size,
               bool run_optimize = true) {
        num_rows_     = num_rows;
        lo_key_       = lo_key;
        hi_key_       = hi_key;
        bucket_size_  = bucket_size;
        run_optimized_ = run_optimize;
        size_t num_buckets = static_cast<size_t>((hi_key - lo_key) / bucket_size + 1);

        std::vector<std::vector<uint32_t>> per_bucket_pos(num_buckets);
        for (size_t i = 0; i < num_rows; i++) {
            int64_t v = values[i];
            if (v < lo_key || v > hi_key) continue;
            size_t b = static_cast<size_t>((v - lo_key) / bucket_size);
            if (b < num_buckets) per_bucket_pos[b].push_back(static_cast<uint32_t>(i));
        }

        bitmaps_pe_.reserve(num_buckets);
        roaring::Roaring cum;
        for (size_t b = 0; b < num_buckets; b++) {
            roaring::Roaring bucket_eq;
            if (!per_bucket_pos[b].empty())
                bucket_eq.addMany(per_bucket_pos[b].size(), per_bucket_pos[b].data());
            cum |= bucket_eq;
            roaring::Roaring snap = cum;
            if (run_optimize) snap.runOptimize();
            bitmaps_pe_.push_back(std::move(snap));
        }
    }

    int bucket_of(int64_t key) const {
        if (key < lo_key_) return -1;
        if (key > hi_key_) return static_cast<int>(bitmaps_pe_.size()) - 1;
        return static_cast<int>((key - lo_key_) / bucket_size_);
    }
    int64_t bucket_max(int b) const { return lo_key_ + (b + 1) * bucket_size_ - 1; }
    int64_t bucket_min(int b) const { return lo_key_ + b * bucket_size_; }
    int64_t lo_key()         const { return lo_key_; }
    int64_t hi_key()         const { return hi_key_; }
    int64_t bucket_size_int() const { return bucket_size_; }
    bool    run_optimized()  const { return run_optimized_; }

    const roaring::Roaring* prefix_at_bucket(int b) const {
        if (b < 0) return nullptr;
        size_t idx = std::min<size_t>(static_cast<size_t>(b), bitmaps_pe_.size() - 1);
        return &bitmaps_pe_[idx];
    }

    size_t num_keys() const override { return bitmaps_pe_.size(); }
    size_t num_rows() const override { return num_rows_; }
    size_t storage_bytes() const override {
        size_t t = 0;
        for (auto& r : bitmaps_pe_) t += r.getSizeInBytes(/*portable=*/false);
        return t;
    }
    const char* backend_name() const override {
        return run_optimized_ ? "CRoaringBPE" : "CRoaringBPE";  // same label
    }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        size_t arr = 0, bset = 0, run = 0, total = 0;
        for (auto& r : bitmaps_pe_) {
            roaring::api::roaring_statistics_t st;
            roaring::api::roaring_bitmap_statistics(&r.roaring, &st);
            arr   += st.n_bytes_array_containers;
            bset  += st.n_bytes_bitset_containers;
            run   += st.n_bytes_run_containers;
            total += r.getSizeInBytes(/*portable=*/false);
        }
        size_t header = (total > arr + bset + run) ? total - arr - bset - run : 0;
        return {{"array", arr}, {"bitset", bset}, {"run", run}, {"header", header}};
    }

private:
    size_t  num_rows_      = 0;
    int64_t lo_key_        = 0;
    int64_t hi_key_        = 0;
    int64_t bucket_size_   = 1;
    bool    run_optimized_ = true;
    std::vector<roaring::Roaring> bitmaps_pe_;
};

// -----------------------------------------------------------------------
// IndexedWAH — flat-harness per-value WAH index.
//
// Harness: InvertedIndex triple.  WAH's native bitvector is built on
// demand from a sorted position list using FastBit's appendFill + `+= 1`
// combo — same sparse-aware construction we used to use at build time,
// now invoked per-key at query time.
//
// OR strategy: for_each requested key, build scratch bv from positions,
// dst |= scratch.  Seed `dst` as Decompressed on first OR (teacher's
// BMTPCH_Q5 pattern) so subsequent |='s stay on FastBit's word-level
// dense path.
//
// This trades build-time bv construction (previously materialised once
// per key, cached in unordered_map) for query-time construction.  Build
// becomes cheaper (just position bucketing); query becomes more expensive
// for bv-heavy queries.  Fair trade for the flat-harness footprint.
// -----------------------------------------------------------------------
class IndexedWAH : public IBitmapIndex, public InvertedIndex {
public:
    IndexedWAH() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

    // Build an ibis::bitvector from a sorted positions list.  appendFill
    // collapses gaps of N zeros into one fill word in O(1); `+= 1` emits
    // a literal word at the current cursor.  O(num_set_bits) total.
    static ibis::bitvector bv_from_positions(const uint32_t* p, size_t n,
                                             size_t num_rows) {
        ibis::bitvector bv;
        size_t cursor = 0;
        for (size_t i = 0; i < n; i++) {
            uint32_t pos = p[i];
            if (pos > cursor) {
                bv.appendFill(0, static_cast<ibis::bitvector::word_t>(pos - cursor));
                cursor = pos;
            }
            bv += 1;
            cursor++;
        }
        if (cursor < num_rows)
            bv.appendFill(0, static_cast<ibis::bitvector::word_t>(num_rows - cursor));
        bv.compress();
        return bv;
    }

    void apply_or_to(ibis::bitvector& dst, int64_t key) const {
        auto [p, n] = positions_for(key);
        if (!n) return;
        ibis::bitvector bv = bv_from_positions(p, n, num_rows_);
        dst |= bv;
    }
    void apply_or_range_to(ibis::bitvector& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (!n) continue;
            ibis::bitvector bv = bv_from_positions(p, n, num_rows_);
            dst |= bv;
        }
    }
    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    // K-way OR — teacher's BMTPCH_Q5 pattern (copy+decompress+|=):
    //   first key  : dst.copy(bv); dst.decompress()
    //   rest       : dst |= bv
    // decompress() flips seed to Decompressed so subsequent |='s use
    // FastBit's word-level dense path (no per-OR allocation).
    template <typename Iterable>
    ibis::bitvector or_many(const Iterable& keys) const {
        ibis::bitvector dst;
        bool seeded = false;
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            ibis::bitvector bv = bv_from_positions(p, n, num_rows_);
            if (!seeded) {
                dst.copy(bv);
                dst.decompress();
                seeded = true;
            } else {
                dst |= bv;
            }
        });
        return dst;
    }
    ibis::bitvector or_range(int64_t lo, int64_t hi) const {
        ibis::bitvector dst;
        apply_or_range_to(dst, lo, hi);
        return dst;
    }

    size_t num_keys()     const override { return num_distinct_keys(); }
    size_t num_rows()     const override { return num_rows_; }
    size_t storage_bytes() const override {
        cached_native_layers([this]{ return compute_layers(); });
        return cached_native_total_bytes();
    }
    const char* backend_name() const override { return "WAH"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    // Shadow-build per-key ibis::bitvector via bv_from_positions, sum
    // getSerialSize() (compressed word stream bytes — fill+literal
    // words only, excludes the ibis::bitvector C++ struct).  FastBit's
    // WAH layout is a flat word stream — single "wah" line.
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t bytes = 0;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            ibis::bitvector bv = bv_from_positions(p, n, num_rows_);
            bytes += bv.getSerialSize();
        });
        return {{"wah", bytes}};
    }
};

// -----------------------------------------------------------------------
// IndexedEWAH — flat-harness per-value EWAH index.
//
// Harness: InvertedIndex triple.  EWAH's native EWBA is built on demand
// from a sorted position list using EWBA::set (which requires monotonic
// input — positions_at already returns ascending order).
//
// K-way OR seeds dst from the first scratch EWBA (copy-assign), then
// chains subsequent OR's through EWBA::logicalor(src, tmp) + swap —
// matches the library-level pairwise OR path.
// -----------------------------------------------------------------------
class IndexedEWAH : public IBitmapIndex, public InvertedIndex {
public:
    using EWBA = ewah::EWAHBoolArray<uint64_t>;
    IndexedEWAH() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

    // Build an EWBA from a sorted positions list.  EWBA::set requires
    // monotonic input; positions_at returns pre-sorted data.
    static EWBA ewba_from_positions(const uint32_t* p, size_t n, size_t num_rows) {
        EWBA e;
        for (size_t i = 0; i < n; i++) e.set(p[i]);
        if (e.sizeInBits() < num_rows) e.padWithZeroes(num_rows);
        return e;
    }

    void apply_or_to(EWBA& dst, int64_t key) const {
        auto [p, n] = positions_for(key);
        if (!n) return;
        EWBA src = ewba_from_positions(p, n, num_rows_);
        EWBA tmp;
        dst.logicalor(src, tmp);
        dst = std::move(tmp);
    }
    void apply_or_range_to(EWBA& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (!n) continue;
            EWBA src = ewba_from_positions(p, n, num_rows_);
            EWBA tmp;
            dst.logicalor(src, tmp);
            dst = std::move(tmp);
        }
    }
    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    // K-way OR via EWAH's native priority-queue fast_logicalor.  For 590k
    // keys × 4 set bits each, pairwise OR would be O(K × |dst|) =
    // ~4.4 TB of memory traffic; fast_logicalor is O(Σ|src|) with k-way
    // priority-queue merging — minutes vs hours.
    template <typename Iterable>
    EWBA or_many(const Iterable& keys) const {
        std::vector<EWBA> srcs;
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            srcs.push_back(ewba_from_positions(p, n, num_rows_));
        });
        if (srcs.empty()) return EWBA();
        if (srcs.size() == 1) return std::move(srcs[0]);
        std::vector<const EWBA*> ptrs;
        ptrs.reserve(srcs.size());
        for (auto& s : srcs) ptrs.push_back(&s);
        return ewah::fast_logicalor(ptrs.size(), ptrs.data());
    }
    EWBA or_range(int64_t lo, int64_t hi) const {
        EWBA dst;
        apply_or_range_to(dst, lo, hi);
        return dst;
    }

    size_t num_keys()     const override { return num_distinct_keys(); }
    size_t num_rows()     const override { return num_rows_; }
    size_t storage_bytes() const override {
        cached_native_layers([this]{ return compute_layers(); });
        return cached_native_total_bytes();
    }
    const char* backend_name() const override { return "EWAH"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    // Shadow-build per-key EWAHBoolArray via ewba_from_positions, sum
    // sizeInBytes() (compressed RLW buffer = words × 8 B).  Excludes
    // the ewah::EWAHBoolArray C++ struct overhead.
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t bytes = 0;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            EWBA e = ewba_from_positions(p, n, num_rows_);
            bytes += e.sizeInBytes();
        });
        return {{"ewah", bytes}};
    }
};

// -----------------------------------------------------------------------
// IndexedConcise — flat-harness per-value Concise index.
//
// Harness: InvertedIndex triple.  Concise's native CS is built on demand
// from a sorted position list using CS::append (monotonic fast path,
// preferred over CS::add when input is already sorted).
//
// K-way OR: seed dst from first scratch CS, then chain pairwise `|` OR's.
// Concise's operator| materialises a new CS each call (library-level),
// same as the logicalor path for EWAH.
// -----------------------------------------------------------------------
class IndexedConcise : public IBitmapIndex, public InvertedIndex {
public:
    using CS = ConciseSet<false>;
    IndexedConcise() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

    // Build a CS from a sorted positions list.  CS::append is the
    // monotonic-input fast path (vs. CS::add which re-validates each).
    static CS cs_from_positions(const uint32_t* p, size_t n) {
        CS c;
        for (size_t i = 0; i < n; i++) c.append(p[i]);
        return c;
    }

    void apply_or_to(CS& dst, int64_t key) const {
        auto [p, n] = positions_for(key);
        if (!n) return;
        CS src = cs_from_positions(p, n);
        dst = dst | src;
    }
    void apply_or_range_to(CS& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (!n) continue;
            CS src = cs_from_positions(p, n);
            dst = dst | src;
        }
    }
    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    // K-way OR via Concise's native priority-queue fast_logicalor.
    // Avoids the K-fold O(|dst|) cost of pairwise `dst | src` chaining
    // for high-cardinality columns (Q3/Q4 590k orderkey OR).
    template <typename Iterable>
    CS or_many(const Iterable& keys) const {
        std::vector<CS> srcs;
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            srcs.push_back(cs_from_positions(p, n));
        });
        if (srcs.empty()) return CS();
        if (srcs.size() == 1) return std::move(srcs[0]);
        std::vector<const CS*> ptrs;
        ptrs.reserve(srcs.size());
        for (auto& s : srcs) ptrs.push_back(&s);
        return CS::fast_logicalor(ptrs.size(), ptrs.data());
    }
    CS or_range(int64_t lo, int64_t hi) const {
        CS dst;
        apply_or_range_to(dst, lo, hi);
        return dst;
    }

    size_t num_keys()     const override { return num_distinct_keys(); }
    size_t num_rows()     const override { return num_rows_; }
    size_t storage_bytes() const override {
        cached_native_layers([this]{ return compute_layers(); });
        return cached_native_total_bytes();
    }
    const char* backend_name() const override { return "Concise"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    // Shadow-build per-key ConciseSet via cs_from_positions, sum
    // sizeInBytes() (compressed words × 4 B).  Excludes the ConciseSet
    // C++ struct overhead.
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t bytes = 0;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            CS c = cs_from_positions(p, n);
            bytes += c.sizeInBytes();
        });
        return {{"concise", bytes}};
    }
};

} // namespace bm_index
