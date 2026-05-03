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
// IndexedComBit — uses SparseComBit per value.
// -----------------------------------------------------------------------
class IndexedComBit {
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

    size_t num_keys() const { return index_.size(); }
    size_t storage_bytes() const {
        size_t t = 0;
        for (auto& [_, s] : index_) t += s.storage_bytes();
        return t;
    }
    size_t num_rows() const { return num_rows_; }
    size_t segment_bits() const { return segment_bits_; }

private:
    size_t num_rows_ = 0;
    size_t segment_bits_ = 4096;
    std::unordered_map<int64_t, SparseComBit> index_;
};

// -----------------------------------------------------------------------
// IndexedCRoaring — uses roaring::Roaring per value.
// -----------------------------------------------------------------------
class IndexedCRoaring {
public:
    IndexedCRoaring() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows,
               bool run_optimize = false) {
        num_rows_ = num_rows;
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
    size_t num_keys() const { return index_.size(); }
    size_t num_rows() const { return num_rows_; }

private:
    size_t num_rows_ = 0;
    std::unordered_map<int64_t, roaring::Roaring> index_;
};

// -----------------------------------------------------------------------
// IndexedWAH — uses ibis::bitvector per value (FastBit WAH).
// -----------------------------------------------------------------------
class IndexedWAH {
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
            // WAH compresses runs of zeros — sequentially append bits, runs
            // of zeros between set positions get RLE'd by compress().
            // Pos already monotonic (built by row-order scan).
            ibis::bitvector bv;
            size_t cursor = 0;
            for (uint32_t p : pos) {
                while (cursor < p) { bv += 0; cursor++; }
                bv += 1; cursor++;
            }
            // Pad trailing zeros to num_rows so OR/AND with sibling bitmaps
            // align.  Cheap (compress() RLE-encodes the long zero run).
            while (cursor < num_rows) { bv += 0; cursor++; }
            bv.compress();
            index_.emplace(k, std::move(bv));
        }
    }

    bool has(int64_t key) const { return index_.count(key) > 0; }
    void apply_or_to(ibis::bitvector& dst, int64_t key) const {
        auto it = index_.find(key);
        if (it != index_.end()) dst |= it->second;
    }
    size_t num_keys() const { return index_.size(); }
    size_t num_rows() const { return num_rows_; }

private:
    size_t num_rows_ = 0;
    std::unordered_map<int64_t, ibis::bitvector> index_;
};

// -----------------------------------------------------------------------
// IndexedEWAH — uses ewah::EWAHBoolArray per value.
// -----------------------------------------------------------------------
class IndexedEWAH {
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
    size_t num_keys() const { return index_.size(); }
    size_t num_rows() const { return num_rows_; }

private:
    size_t num_rows_ = 0;
    std::unordered_map<int64_t, EWBA> index_;
};

// -----------------------------------------------------------------------
// IndexedConcise — uses ConciseSet<false> per value.
// -----------------------------------------------------------------------
class IndexedConcise {
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
    size_t num_keys() const { return index_.size(); }
    size_t num_rows() const { return num_rows_; }

private:
    size_t num_rows_ = 0;
    std::unordered_map<int64_t, CS> index_;
};

} // namespace bm_index
