// =============================================================================
// IndexedBitmap — per-backend value-indexed bitmap (mirror of teacher's
// rabit::Rabit pattern).  Built once from a lineitem column scan and cached
// in the client context.  Used by Q1 / Q5 / Q6 BMTPCH paths to do bitmap
// multi-OR over qualifying value sets (replaces SQL JOIN).
//
// Per-backend storage:
//   ComBit   → unordered_map<int64_t, SparseComBit>
//   CRoaring → unordered_map<int64_t, roaring::Roaring>
//   CRR      → CR + runOptimize at build
//   WAH      → unordered_map<int64_t, ibis::bitvector>
//   EWAH     → unordered_map<int64_t, ewah::EWAHBoolArray<uint64_t>>
//   Concise  → unordered_map<int64_t, ConciseSet<false>>
//
// Build is one lineitem scan + per-value encode.  Encode is sparse-aware
// for ComBit (SparseComBit + compress_sparse_segment) — O(set_bits) per
// value, ~22 sec for 15M values.  CR/EW/WAH/Concise have native sparse
// encoders (addMany / set / etc.).
//
// Apply pattern (Q5):
//   for (auto& [okey, _] : order_nation_map)
//       idx_orderkey.apply_or_to(btv_res, okey);
//
// Pre: btv_res is initialized to an empty (all-zero) bitmap of `num_rows`
// bits in the same backend's representation.
// =============================================================================

#pragma once

#include "combit/include/combit.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"
#include "Concise/concise.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace bm_index {

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
};

// -----------------------------------------------------------------------
// IndexedComBit — uses SparseComBit per value.
// -----------------------------------------------------------------------
class IndexedComBit : public IBitmapIndex {
public:
    IndexedComBit() = default;

    // Build from raw (key per row) column.  num_rows = length of `keys`.
    void build(const std::vector<int64_t>& keys, size_t num_rows,
               size_t segment_bits = 4096) {
        num_rows_ = num_rows;
        segment_bits_ = segment_bits;
        // Bucket rows by key.
        std::unordered_map<int64_t, std::vector<uint32_t>> by_key;
        by_key.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_key[keys[i]].push_back(static_cast<uint32_t>(i));
        // Encode each key's position list into a SparseComBit.
        index_.reserve(by_key.size());
        for (auto& [k, pos] : by_key) {
            SparseComBit s = SparseComBit::from_positions(pos, num_rows, segment_bits);
            index_.emplace(k, std::move(s));
        }
    }

    bool has(int64_t key) const { return index_.count(key) > 0; }

    // OR Btvs[key] into dst.  No-op if key not found.
    void apply_or_to(ComBit& dst, int64_t key) const {
        auto it = index_.find(key);
        if (it != index_.end()) it->second.apply_or_to(dst);
    }

    // Apply OR for all keys in [lo_inclusive, hi_inclusive].  Used by Q1
    // (range OR over shipdate days) and Q6 (range OR over discount /
    // quantity values).
    void apply_or_range_to(ComBit& dst, int64_t lo, int64_t hi) const {
        for (auto& [k, s] : index_)
            if (k >= lo && k <= hi) s.apply_or_to(dst);
    }

    // K-way OR via SparseComBit::or_many — bucket every input
    // SparseComBit segment by its output segment index, then OR all
    // matching inputs into each output segment in one sequential pass.
    // For Q3 PhaseB (K=590k orderkey OR) this avoids 590k function-call
    // overhead + random output-segment writes.  Counterpart of CRR
    // fastunion / EWAH fast_logicalor.
    template <typename Iterable>
    ComBit or_many(const Iterable& keys) const {
        std::vector<const SparseComBit*> ptrs;
        ptrs.reserve(index_.size());
        for (auto& k : keys) {
            auto it = index_.find(static_cast<int64_t>(k));
            if (it != index_.end()) ptrs.push_back(&it->second);
        }
        if (ptrs.empty())
            return ComBit::from_sparse_positions({}, num_rows_, segment_bits_);
        return SparseComBit::or_many(ptrs.size(), ptrs.data(), num_rows_, segment_bits_);
    }

    template <typename F>
    void for_each_key(F&& f) const { for (auto& [k, _] : index_) f(k); }

    size_t num_keys() const override { return index_.size(); }
    size_t storage_bytes() const override {
        size_t t = 0;
        for (auto& [_, s] : index_) t += s.storage_bytes();
        return t;
    }
    size_t num_rows() const override { return num_rows_; }
    size_t segment_bits() const { return segment_bits_; }
    const char* backend_name() const override { return "ComBit"; }

private:
    size_t num_rows_ = 0;
    size_t segment_bits_ = 4096;
    std::unordered_map<int64_t, SparseComBit> index_;
};

// -----------------------------------------------------------------------
// IndexedComBitGE — Group Encoding variant of IndexedComBit.
//
// Same per-key SparseComBit storage as IndexedComBit, but the keys are
// expected to be GROUP IDs (e.g. year for shipdate, decade-bucket for
// quantity, etc.) — i.e. the build path has already bucketed the raw
// values into a small group cardinality.  Mirrors teacher's
// rabit::Btvs_GE pattern (one bitmap per group, accessed by group index).
//
// Functionally identical to IndexedComBit at this layer; the GE-ness is
// in how it was BUILT (keys = group IDs, not raw values).  The class is
// kept distinct so dynamic_cast in Q6 can route to GE-aware logic.
// -----------------------------------------------------------------------
class IndexedComBitGE : public IBitmapIndex {
public:
    IndexedComBitGE() = default;

    // Build from already-bucketed group IDs (one per row).
    // Equivalent to IndexedComBit::build with the group IDs.
    void build(const std::vector<int64_t>& group_ids, size_t num_rows,
               size_t segment_bits = 4096) {
        num_rows_ = num_rows;
        segment_bits_ = segment_bits;
        std::unordered_map<int64_t, std::vector<uint32_t>> by_g;
        by_g.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_g[group_ids[i]].push_back(static_cast<uint32_t>(i));
        index_.reserve(by_g.size());
        for (auto& [g, pos] : by_g) {
            SparseComBit s = SparseComBit::from_positions(pos, num_rows, segment_bits);
            index_.emplace(g, std::move(s));
        }
    }

    bool has(int64_t group_id) const { return index_.count(group_id) > 0; }
    void apply_or_to(ComBit& dst, int64_t group_id) const {
        auto it = index_.find(group_id);
        if (it != index_.end()) it->second.apply_or_to(dst);
    }
    void apply_or_range_to(ComBit& dst, int64_t lo, int64_t hi) const {
        for (auto& [g, s] : index_)
            if (g >= lo && g <= hi) s.apply_or_to(dst);
    }
    template <typename F>
    void for_each_key(F&& f) const { for (auto& [g, _] : index_) f(g); }

    size_t num_keys() const override { return index_.size(); }
    size_t storage_bytes() const override {
        size_t t = 0;
        for (auto& [_, s] : index_) t += s.storage_bytes();
        return t;
    }
    size_t num_rows() const override { return num_rows_; }
    size_t segment_bits() const { return segment_bits_; }
    const char* backend_name() const override { return "ComBitGE"; }

private:
    size_t num_rows_ = 0;
    size_t segment_bits_ = 4096;
    std::unordered_map<int64_t, SparseComBit> index_;
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

private:
    size_t   num_rows_     = 0;
    int64_t  lo_key_       = 0;
    int64_t  hi_key_       = 0;
    int64_t  bucket_size_  = 1;
    size_t   segment_bits_ = 4096;
    std::vector<ComBit> bitmaps_pe_;
};

// -----------------------------------------------------------------------
// IndexedCRoaring — uses roaring::Roaring per value.
// -----------------------------------------------------------------------
class IndexedCRoaring : public IBitmapIndex {
public:
    IndexedCRoaring() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows,
               bool run_optimize = false) {
        num_rows_ = num_rows;
        run_optimized_ = run_optimize;
        std::unordered_map<int64_t, std::vector<uint32_t>> by_key;
        by_key.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_key[keys[i]].push_back(static_cast<uint32_t>(i));
        index_.reserve(by_key.size());
        for (auto& [k, pos] : by_key) {
            roaring::Roaring r;
            if (!pos.empty()) r.addMany(pos.size(), pos.data());
            if (run_optimize) r.runOptimize();
            index_.emplace(k, std::move(r));
        }
    }

    bool has(int64_t key) const { return index_.count(key) > 0; }
    void apply_or_to(roaring::Roaring& dst, int64_t key) const {
        auto it = index_.find(key);
        if (it != index_.end()) dst |= it->second;
    }
    void apply_or_range_to(roaring::Roaring& dst, int64_t lo, int64_t hi) const {
        for (auto& [k, r] : index_) if (k >= lo && k <= hi) dst |= r;
    }
    template <typename F>
    void for_each_key(F&& f) const { for (auto& [k, _] : index_) f(k); }

    // K-way OR via roaring::Roaring::fastunion (priority-queue merge,
    // O(total_cardinality * log K) instead of pairwise O(N * total_card)).
    // This is the path that lets CRoaringRun beat plain CRoaring on
    // multi-OR-heavy queries (Q5: 28k orderkey OR, Q1: 2437 day OR).
    // Returns the union; callers AND the result with sibling bitmaps.
    template <typename Iterable>
    roaring::Roaring or_many(const Iterable& keys) const {
        std::vector<const roaring::Roaring*> ptrs;
        ptrs.reserve(index_.size());
        for (auto& k : keys) {
            auto it = index_.find(static_cast<int64_t>(k));
            if (it != index_.end()) ptrs.push_back(&it->second);
        }
        if (ptrs.empty()) return roaring::Roaring();
        return roaring::Roaring::fastunion(ptrs.size(), ptrs.data());
    }
    roaring::Roaring or_range(int64_t lo, int64_t hi) const {
        std::vector<const roaring::Roaring*> ptrs;
        for (auto& [k, r] : index_)
            if (k >= lo && k <= hi) ptrs.push_back(&r);
        if (ptrs.empty()) return roaring::Roaring();
        return roaring::Roaring::fastunion(ptrs.size(), ptrs.data());
    }
    size_t num_keys() const override { return index_.size(); }
    size_t num_rows() const override { return num_rows_; }
    size_t storage_bytes() const override {
        size_t t = 0;
        for (auto& [_, r] : index_) t += r.getSizeInBytes();
        return t;
    }
    const char* backend_name() const override { return run_optimized_ ? "CRoaringRun" : "CRoaring"; }
    bool run_optimized() const { return run_optimized_; }

private:
    size_t num_rows_ = 0;
    bool   run_optimized_ = false;
    std::unordered_map<int64_t, roaring::Roaring> index_;
public:
    void mark_run_optimized() { run_optimized_ = true; }
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
        for (auto& r : bitmaps_pe_) t += r.getSizeInBytes();
        return t;
    }
    const char* backend_name() const override {
        return run_optimized_ ? "CRoaringBPE" : "CRoaringBPE";  // same label
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
// IndexedWAH — uses ibis::bitvector per value (FastBit WAH).
// -----------------------------------------------------------------------
class IndexedWAH : public IBitmapIndex {
public:
    IndexedWAH() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        num_rows_ = num_rows;
        std::unordered_map<int64_t, std::vector<uint32_t>> by_key;
        by_key.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_key[keys[i]].push_back(static_cast<uint32_t>(i));
        index_.reserve(by_key.size());
        for (auto& [k, pos] : by_key) {
            // Sparse-aware build: appendFill(0, gap) collapses a gap of N
            // zero-bits into one fill word in O(1) instead of N `+= 0`
            // appends.  This is what makes WAH per-value FK indexes
            // tractable on high-cardinality columns (l_orderkey 15M /
            // l_partkey 2M unique values at SF10) — the previous
            // single-bit append loop was O(num_rows × num_keys) which
            // exceeded a day for these columns; this is O(num_set_bits).
            ibis::bitvector bv;
            size_t cursor = 0;
            for (uint32_t p : pos) {
                if (p > cursor) {
                    bv.appendFill(0, static_cast<ibis::bitvector::word_t>(p - cursor));
                    cursor = p;
                }
                bv += 1;
                cursor++;
            }
            if (cursor < num_rows)
                bv.appendFill(0, static_cast<ibis::bitvector::word_t>(num_rows - cursor));
            bv.compress();
            index_.emplace(k, std::move(bv));
        }
    }

    bool has(int64_t key) const { return index_.count(key) > 0; }
    void apply_or_to(ibis::bitvector& dst, int64_t key) const {
        auto it = index_.find(key);
        if (it != index_.end()) dst |= it->second;
    }
    void apply_or_range_to(ibis::bitvector& dst, int64_t lo, int64_t hi) const {
        for (auto& [k, b] : index_) if (k >= lo && k <= hi) dst |= b;
    }
    template <typename F>
    void for_each_key(F&& f) const { for (auto& [k, _] : index_) f(k); }

    // K-way OR via balanced tree-merge.  FastBit doesn't ship a k-way
    // merge primitive; pairwise streaming `dst |= bv` is O(K × |dst|)
    // where |dst| grows to ~num_words by the end (catastrophic at
    // K=590k for SF10 orderkey).  Tree-merge does O(N_words × log K)
    // total work because each level only doubles the average bitvector
    // size while halving the count.  Built on FastBit's |= and copy,
    // no library changes required.
    template <typename Iterable>
    ibis::bitvector or_many(const Iterable& keys) const {
        std::vector<ibis::bitvector> level;
        level.reserve(index_.size());
        for (auto& k : keys) {
            auto it = index_.find(static_cast<int64_t>(k));
            if (it == index_.end()) continue;
            level.emplace_back();
            level.back().copy(it->second);
        }
        if (level.empty()) return ibis::bitvector();
        while (level.size() > 1) {
            std::vector<ibis::bitvector> next;
            next.reserve((level.size() + 1) / 2);
            for (size_t i = 0; i + 1 < level.size(); i += 2) {
                level[i] |= level[i+1];
                next.push_back(std::move(level[i]));
            }
            if (level.size() & 1) next.push_back(std::move(level.back()));
            level = std::move(next);
        }
        return std::move(level[0]);
    }
    ibis::bitvector or_range(int64_t lo, int64_t hi) const {
        std::vector<int64_t> ks;
        for (auto& [k, _] : index_) if (k >= lo && k <= hi) ks.push_back(k);
        return or_many(ks);
    }

    size_t num_keys() const override { return index_.size(); }
    size_t num_rows() const override { return num_rows_; }
    size_t storage_bytes() const override {
        size_t t = 0;
        for (auto& [_, b] : index_) t += b.bytes();
        return t;
    }
    const char* backend_name() const override { return "WAH"; }

private:
    size_t num_rows_ = 0;
    std::unordered_map<int64_t, ibis::bitvector> index_;
};

// -----------------------------------------------------------------------
// IndexedEWAH — uses ewah::EWAHBoolArray per value.
// -----------------------------------------------------------------------
class IndexedEWAH : public IBitmapIndex {
public:
    using EWBA = ewah::EWAHBoolArray<uint64_t>;
    IndexedEWAH() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        num_rows_ = num_rows;
        std::unordered_map<int64_t, std::vector<uint32_t>> by_key;
        by_key.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_key[keys[i]].push_back(static_cast<uint32_t>(i));
        index_.reserve(by_key.size());
        for (auto& [k, pos] : by_key) {
            EWBA e;
            // EWAH set requires monotonic increasing positions (already
            // monotonic since we scan in row order).
            for (uint32_t p : pos) e.set(p);
            if (e.sizeInBits() < num_rows) e.padWithZeroes(num_rows);
            index_.emplace(k, std::move(e));
        }
    }

    bool has(int64_t key) const { return index_.count(key) > 0; }
    void apply_or_to(EWBA& dst, int64_t key) const {
        auto it = index_.find(key);
        if (it != index_.end()) {
            EWBA tmp;
            dst.logicalor(it->second, tmp);
            dst = std::move(tmp);
        }
    }
    void apply_or_range_to(EWBA& dst, int64_t lo, int64_t hi) const {
        for (auto& [k, e] : index_) {
            if (k < lo || k > hi) continue;
            EWBA tmp;
            dst.logicalor(e, tmp);
            dst = std::move(tmp);
        }
    }
    template <typename F>
    void for_each_key(F&& f) const { for (auto& [k, _] : index_) f(k); }

    // K-way OR via balanced tree-merge.  EWAH's library-provided
    // ewah::fast_logicalor is a priority-queue merge but in practice
    // its per-merge allocation overhead dominates at K~590k inputs
    // (we observed >30 min/iter on Q3/Q4).  Plain balanced tree-merge
    // does O(N_words × log K) work with one allocation per merge and
    // strictly halving levels — measured ~5 sec/iter at K=590k.
    template <typename Iterable>
    EWBA or_many(const Iterable& keys) const {
        std::vector<EWBA> level;
        level.reserve(index_.size());
        for (auto& k : keys) {
            auto it = index_.find(static_cast<int64_t>(k));
            if (it != index_.end()) level.emplace_back(it->second);  // copy
        }
        if (level.empty()) return EWBA();
        while (level.size() > 1) {
            std::vector<EWBA> next;
            next.reserve((level.size() + 1) / 2);
            for (size_t i = 0; i + 1 < level.size(); i += 2) {
                EWBA tmp;
                level[i].logicalor(level[i+1], tmp);
                next.push_back(std::move(tmp));
            }
            if (level.size() & 1) next.push_back(std::move(level.back()));
            level = std::move(next);
        }
        return std::move(level[0]);
    }
    EWBA or_range(int64_t lo, int64_t hi) const {
        std::vector<int64_t> ks;
        for (auto& [k, _] : index_) if (k >= lo && k <= hi) ks.push_back(k);
        return or_many(ks);
    }
    size_t num_keys() const override { return index_.size(); }
    size_t num_rows() const override { return num_rows_; }
    size_t storage_bytes() const override {
        size_t t = 0;
        for (auto& [_, e] : index_) t += e.sizeInBytes();
        return t;
    }
    const char* backend_name() const override { return "EWAH"; }

private:
    size_t num_rows_ = 0;
    std::unordered_map<int64_t, EWBA> index_;
};

// -----------------------------------------------------------------------
// IndexedConcise — uses ConciseSet<false> per value.
// -----------------------------------------------------------------------
class IndexedConcise : public IBitmapIndex {
public:
    using CS = ConciseSet<false>;
    IndexedConcise() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        num_rows_ = num_rows;
        std::unordered_map<int64_t, std::vector<uint32_t>> by_key;
        by_key.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_key[keys[i]].push_back(static_cast<uint32_t>(i));
        index_.reserve(by_key.size());
        for (auto& [k, pos] : by_key) {
            CS c;
            // Concise add() requires monotonic.  Already monotonic.
            for (uint32_t p : pos) c.add(p);
            index_.emplace(k, std::move(c));
        }
    }

    bool has(int64_t key) const { return index_.count(key) > 0; }
    void apply_or_to(CS& dst, int64_t key) const {
        auto it = index_.find(key);
        if (it != index_.end()) dst = dst | it->second;
    }
    void apply_or_range_to(CS& dst, int64_t lo, int64_t hi) const {
        for (auto& [k, c] : index_) if (k >= lo && k <= hi) dst = dst | c;
    }
    template <typename F>
    void for_each_key(F&& f) const { for (auto& [k, _] : index_) f(k); }

    // K-way OR via balanced tree-merge.  Same rationale as
    // IndexedEWAH::or_many — Concise's library fast_logicalor exists
    // but allocation overhead dominates at K~590k.  Tree-merge over
    // the `|` operator gives O(N × log K) with halving levels.
    template <typename Iterable>
    CS or_many(const Iterable& keys) const {
        std::vector<CS> level;
        level.reserve(index_.size());
        for (auto& k : keys) {
            auto it = index_.find(static_cast<int64_t>(k));
            if (it != index_.end()) level.emplace_back(it->second);
        }
        if (level.empty()) return CS();
        while (level.size() > 1) {
            std::vector<CS> next;
            next.reserve((level.size() + 1) / 2);
            for (size_t i = 0; i + 1 < level.size(); i += 2)
                next.push_back(level[i] | level[i+1]);
            if (level.size() & 1) next.push_back(std::move(level.back()));
            level = std::move(next);
        }
        return std::move(level[0]);
    }
    CS or_range(int64_t lo, int64_t hi) const {
        std::vector<int64_t> ks;
        for (auto& [k, _] : index_) if (k >= lo && k <= hi) ks.push_back(k);
        return or_many(ks);
    }
    size_t num_keys() const override { return index_.size(); }
    size_t num_rows() const override { return num_rows_; }
    size_t storage_bytes() const override {
        // Concise doesn't expose a sizeof — approximate via wpc set size.
        size_t t = 0;
        for (auto& [_, c] : index_) t += c.size() * 4;  // rough
        return t;
    }
    const char* backend_name() const override { return "Concise"; }

private:
    size_t num_rows_ = 0;
    std::unordered_map<int64_t, CS> index_;
};

} // namespace bm_index
