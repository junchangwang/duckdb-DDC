

#pragma once

#include "ddc/include/ddc.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"
#include "Concise/concise.h"
#include "bitset_vector.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace bm_index {

static constexpr size_t CB_SEGMENT_BITS = 4096;

class IBitmapIndex {
public:
    virtual ~IBitmapIndex() = default;
    virtual size_t num_rows() const = 0;
    virtual size_t num_keys() const = 0;
    virtual size_t storage_bytes() const = 0;
    virtual const char* backend_name() const = 0;

    virtual std::vector<std::pair<std::string, size_t>> layer_breakdown() const {
        return {{"total", storage_bytes()}};
    }
};

class InvertedIndex {
public:

    // Inverted index
    void build_inverted_index(const std::vector<int64_t>& keys, size_t num_rows) {
        std::unordered_map<int64_t, std::vector<uint32_t>> by_key;
        by_key.reserve(num_rows / 4);
        for (size_t i = 0; i < num_rows; i++)
            by_key[keys[i]].push_back(static_cast<uint32_t>(i));
        build_inverted_index_map(std::move(by_key), num_rows);
    }

    // Multi-key rows
    void build_inverted_index_map(std::unordered_map<int64_t, std::vector<uint32_t>>&& by_key,
                                  size_t num_rows) {
        num_rows_ = num_rows;

        // Sort keys
        sorted_keys_.reserve(by_key.size());
        for (auto& [k, _] : by_key) {

            if (k < 0 || k > static_cast<int64_t>(UINT32_MAX)) {
                std::cerr << "[InvertedIndex] FATAL: key " << k
                          << " out of uint32_t range (0..2^32-1).  "
                             "Need int64 sorted_keys_ for this column.\n";
                std::abort();
            }
            sorted_keys_.push_back(static_cast<uint32_t>(k));
        }
        std::sort(sorted_keys_.begin(), sorted_keys_.end());
        // Pack positions
        key_offsets_.resize(sorted_keys_.size() + 1);
        size_t total = 0;
        for (auto& v : by_key) total += v.second.size();
        all_positions_.reserve(total);
        for (size_t i = 0; i < sorted_keys_.size(); i++) {
            key_offsets_[i] = static_cast<uint32_t>(all_positions_.size());
            auto& pos = by_key[static_cast<int64_t>(sorted_keys_[i])];

            std::sort(pos.begin(), pos.end());
            all_positions_.insert(all_positions_.end(), pos.begin(), pos.end());
        }
        key_offsets_.back() = static_cast<uint32_t>(all_positions_.size());
    }

    // Key lookup
    std::pair<const uint32_t*, size_t> positions_for(int64_t key) const {

        if (key < 0 || key > static_cast<int64_t>(UINT32_MAX)) return {nullptr, 0};
        uint32_t key32 = static_cast<uint32_t>(key);
        auto it = std::lower_bound(sorted_keys_.begin(), sorted_keys_.end(), key32);
        if (it == sorted_keys_.end() || *it != key32) return {nullptr, 0};
        size_t idx = static_cast<size_t>(it - sorted_keys_.begin());
        uint32_t lo = key_offsets_[idx];
        uint32_t hi = key_offsets_[idx + 1];
        return {all_positions_.data() + lo, static_cast<size_t>(hi - lo)};
    }

    size_t lower_bound_idx(int64_t lo) const {

        uint32_t lo32 = (lo <= 0) ? 0u
                      : (lo > static_cast<int64_t>(UINT32_MAX) ? UINT32_MAX
                                                                : static_cast<uint32_t>(lo));
        auto it = std::lower_bound(sorted_keys_.begin(), sorted_keys_.end(), lo32);
        return static_cast<size_t>(it - sorted_keys_.begin());
    }
    size_t upper_bound_idx(int64_t hi) const {

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

        if (key < 0 || key > static_cast<int64_t>(UINT32_MAX)) return false;
        return std::binary_search(sorted_keys_.begin(), sorted_keys_.end(),
                                  static_cast<uint32_t>(key));
    }

    template <typename F>
    void for_each_key_base(F&& f) const {

        for (uint32_t k : sorted_keys_) f(static_cast<int64_t>(k));
    }

    // Merge join
    template <typename F>
    void for_each_matched_key(const int64_t* sorted_input, size_t ni, F&& f) const {
        size_t i = 0, j = 0;
        const size_t nj = sorted_keys_.size();
        while (i < ni && j < nj) {
            int64_t a = sorted_input[i];

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

    template <typename F>
    void for_each_key_positions(F&& f) const {
        for (size_t i = 0, k = sorted_keys_.size(); i < k; i++) {
            uint32_t lo = key_offsets_[i];
            uint32_t hi = key_offsets_[i + 1];
            f(all_positions_.data() + lo, static_cast<size_t>(hi - lo));
        }
    }

    size_t inverted_index_bytes() const {
        return all_positions_.capacity() * sizeof(uint32_t)
             + key_offsets_.capacity()   * sizeof(uint32_t)
             + sorted_keys_.capacity()   * sizeof(uint32_t);
    }

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
    std::vector<uint32_t> all_positions_;
    std::vector<uint32_t> key_offsets_;
    std::vector<uint32_t> sorted_keys_;

    mutable bool   native_cached_      = false;
    mutable size_t native_total_bytes_ = 0;
    mutable std::vector<std::pair<std::string, size_t>> native_layers_;
};

class IndexedDDC : public IBitmapIndex, public InvertedIndex {
public:
    IndexedDDC() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows,
               size_t segment_bits = CB_SEGMENT_BITS) {
        segment_bits_ = segment_bits;
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

    // Single-key OR
    void apply_or_to(DDC& dst, int64_t key) const {
        auto [p, n] = positions_for(key);
        if (n) dst.scatter_or_decompressed(p, n);
    }

    void apply_or_range_to(DDC& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (n) dst.scatter_or_decompressed(p, n);
        }
    }

    // Multi-key OR
    template <typename Iterable>
    DDC or_many(const Iterable& keys) const {
        DDC dst = DDC::from_sparse_positions({}, num_rows_, segment_bits_);
        if (sorted_keys_.empty()) return dst;

        std::vector<uint32_t> in;
        in.reserve(std::distance(std::begin(keys), std::end(keys)));
        for (auto& k : keys) {
            int64_t kk = static_cast<int64_t>(k);
            if (kk >= 0 && kk <= static_cast<int64_t>(UINT32_MAX))
                in.push_back(static_cast<uint32_t>(kk));
        }
        if (in.empty()) return dst;

        if (!std::is_sorted(in.begin(), in.end())) {
            std::sort(in.begin(), in.end());
            in.erase(std::unique(in.begin(), in.end()), in.end());
        }

        const size_t nj = sorted_keys_.size();
        const uint32_t* sk = sorted_keys_.data();
        const uint32_t* ko = key_offsets_.data();
        const uint32_t* ap = all_positions_.data();
        size_t j = 0;
        for (size_t i = 0, ni = in.size(); i < ni; i++) {
            uint32_t target = in[i];
            if (j >= nj) break;
            if (sk[j] > target) continue;
            // Galloping search
            size_t step = 1;
            while (j + step < nj && sk[j + step] < target) step *= 2;
            size_t lo = j + (step / 2);
            size_t hi = std::min(j + step, nj);
            while (lo < hi) {
                size_t mid = lo + ((hi - lo) >> 1);
                if (sk[mid] < target) lo = mid + 1;
                else                  hi = mid;
            }
            j = lo;
            if (j < nj && sk[j] == target) {
                uint32_t lo_off = ko[j];
                uint32_t hi_off = ko[j + 1];
                dst.scatter_or_decompressed(ap + lo_off,
                                            static_cast<size_t>(hi_off - lo_off));
                j++;
            }
        }
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
    const char* backend_name() const override { return "DDC"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    size_t segment_bits_ = 4096;

    // Size breakdown
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t l1 = 0, l2 = 0, l3 = 0, l4 = 0, header = 0;
        std::vector<uint32_t> tmp;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            tmp.assign(p, p + n);
            SparseDDC s = SparseDDC::from_positions(tmp, num_rows_, segment_bits_);
            for (const auto& seg : s.seg_data()) {
                auto sb = seg.size_breakdown();
                l1 += sb.l1_literal_bits / 8;
                l2 += sb.l2_literal_bits / 8;
                l3 += sb.l3_literal_bits / 8;
                l4 += (sb.l4_bits + 7) / 8;
            }
            header += s.seg_indices().size() * sizeof(uint16_t);
        });
        return {{"ultra", size_t{0}}, {"L1", l1}, {"L2", l2},
                {"L3", l3}, {"L4", l4}, {"header", header}};
    }
};

// Group encoding
class IndexedDDCGE : public IBitmapIndex, public InvertedIndex {
public:
    IndexedDDCGE() = default;

    void build(const std::vector<int64_t>& group_ids, size_t num_rows,
               size_t segment_bits = CB_SEGMENT_BITS) {
        segment_bits_ = segment_bits;
        build_inverted_index(group_ids, num_rows);
    }

    bool has(int64_t group_id) const { return has_key(group_id); }

    void apply_or_to(DDC& dst, int64_t group_id) const {
        auto [p, n] = positions_for(group_id);
        if (n) dst.scatter_or_decompressed(p, n);
    }

    void apply_or_range_to(DDC& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            if (n) dst.scatter_or_decompressed(p, n);
        }
    }

    template <typename Iterable>
    DDC or_many(const Iterable& keys) const {
        DDC dst = DDC::from_sparse_positions({}, num_rows_, segment_bits_);
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
    const char* backend_name() const override { return "DDCGE"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    size_t segment_bits_ = 4096;

    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t l1 = 0, l2 = 0, l3 = 0, l4 = 0, header = 0;
        std::vector<uint32_t> tmp;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            tmp.assign(p, p + n);
            SparseDDC s = SparseDDC::from_positions(tmp, num_rows_, segment_bits_);
            for (const auto& seg : s.seg_data()) {
                auto sb = seg.size_breakdown();
                l1 += sb.l1_literal_bits / 8;
                l2 += sb.l2_literal_bits / 8;
                l3 += sb.l3_literal_bits / 8;
                l4 += (sb.l4_bits + 7) / 8;
            }
            header += s.seg_indices().size() * sizeof(uint16_t);
        });
        return {{"ultra", size_t{0}}, {"L1", l1}, {"L2", l2},
                {"L3", l3}, {"L4", l4}, {"header", header}};
    }
};

class IndexedDDCBPE : public IBitmapIndex {
public:
    IndexedDDCBPE() = default;

    void build(const std::vector<int64_t>& values, size_t num_rows,
               int64_t lo_key, int64_t hi_key, int64_t bucket_size,
               size_t segment_bits = CB_SEGMENT_BITS) {
        num_rows_     = num_rows;
        lo_key_       = lo_key;
        hi_key_       = hi_key;
        bucket_size_  = bucket_size;
        segment_bits_ = segment_bits;
        size_t num_buckets = static_cast<size_t>((hi_key - lo_key) / bucket_size + 1);

        std::vector<std::vector<uint32_t>> per_bucket_pos(num_buckets);
        for (size_t i = 0; i < num_rows; i++) {
            int64_t v = values[i];
            if (v < lo_key || v > hi_key) continue;
            size_t b = static_cast<size_t>((v - lo_key) / bucket_size);
            if (b < num_buckets) per_bucket_pos[b].push_back(static_cast<uint32_t>(i));
        }

        // Prefix bitmaps
        bitmaps_pe_.reserve(num_buckets);
        DDC cum = DDC::from_sparse_positions({}, num_rows, segment_bits_);
        for (size_t b = 0; b < num_buckets; b++) {
            std::sort(per_bucket_pos[b].begin(), per_bucket_pos[b].end());
            DDC bucket_eq = DDC::from_sparse_positions(
                per_bucket_pos[b], num_rows, segment_bits_);
            cum |= bucket_eq;
            bitmaps_pe_.push_back(cum);
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

    const DDC* prefix_at_bucket(int b) const {
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
    const char* backend_name() const override { return "DDCBPE"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {

        size_t l1 = 0, l2 = 0, l3 = 0, l4 = 0, header = 0;
        for (const auto& bm : bitmaps_pe_) {
            header += sizeof(DDC) + bm.segments().size() * sizeof(DDCBtv);
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
    std::vector<DDC> bitmaps_pe_;
};

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
        if (run_optimized_) {
            // Run union
            std::vector<roaring::Roaring> bms;
            std::vector<const roaring::Roaring*> ptrs;
            or_many_walk(keys, [&](const uint32_t* p, size_t n) {
                if (!n) return;
                bms.emplace_back();
                bms.back().addMany(n, p);
                bms.back().runOptimize();
            });
            ptrs.reserve(bms.size());
            for (auto& b : bms) ptrs.push_back(&b);
            if (ptrs.empty()) return roaring::Roaring();
            return roaring::Roaring::fastunion(ptrs.size(), ptrs.data());
        }

        roaring::Roaring dst;
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            dst.addMany(n, p);
        });
        return dst;
    }
    roaring::Roaring or_range(int64_t lo, int64_t hi) const {
        if (run_optimized_) {

            std::vector<roaring::Roaring> bms;
            std::vector<const roaring::Roaring*> ptrs;
            size_t i_lo = lower_bound_idx(lo);
            size_t i_hi = upper_bound_idx(hi);
            for (size_t i = i_lo; i < i_hi; i++) {
                auto [p, n] = positions_at(i);
                if (!n) continue;
                bms.emplace_back();
                bms.back().addMany(n, p);
                bms.back().runOptimize();
            }
            ptrs.reserve(bms.size());
            for (auto& b : bms) ptrs.push_back(&b);
            if (ptrs.empty()) return roaring::Roaring();
            return roaring::Roaring::fastunion(ptrs.size(), ptrs.data());
        }
        roaring::Roaring dst;
        apply_or_range_to(dst, lo, hi);
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

        native_cached_ = false;
    }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    bool run_optimized_ = false;

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
            total += r.getSizeInBytes( true);
        });
        size_t containers = arr + bset + run;
        size_t header = (total > containers) ? total - containers : 0;
        return {{"array", arr}, {"bitset", bset}, {"run", run}, {"header", header}};
    }
};

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
        for (auto& r : bitmaps_pe_) t += r.getSizeInBytes( false);
        return t;
    }
    const char* backend_name() const override {
        return run_optimized_ ? "CRoaringBPE" : "CRoaringBPE";
    }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        size_t arr = 0, bset = 0, run = 0, total = 0;
        for (auto& r : bitmaps_pe_) {
            roaring::api::roaring_statistics_t st;
            roaring::api::roaring_bitmap_statistics(&r.roaring, &st);
            arr   += st.n_bytes_array_containers;
            bset  += st.n_bytes_bitset_containers;
            run   += st.n_bytes_run_containers;
            total += r.getSizeInBytes( false);
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

class IndexedWAH : public IBitmapIndex, public InvertedIndex {
public:
    IndexedWAH() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

    // WAH builder
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

    template <typename Iterable>
    ibis::bitvector or_many(const Iterable& keys) const {
        ibis::bitvector dst;
        bool seeded = false;
        or_many_walk(keys, [&](const uint32_t* p, size_t n) {
            ibis::bitvector bv = bv_from_positions(p, n, num_rows_);
            if (!seeded) {
                // Dense seed
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

class IndexedEWAH : public IBitmapIndex, public InvertedIndex {
public:
    using EWBA = ewah::EWAHBoolArray<uint64_t>;
    IndexedEWAH() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

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
        // Multi-way union
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

class IndexedConcise : public IBitmapIndex, public InvertedIndex {
public:
    using CS = ConciseSet<false>;
    IndexedConcise() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }

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

// Bitset baseline
class IndexedBitsetAVX512 : public IBitmapIndex, public InvertedIndex {
public:
    void build(const std::vector<int64_t>& keys, size_t num_rows, bool use_simd = true) {
        simd_ = use_simd;
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }
    bool use_simd() const { return simd_; }

    bitset::BitsetVector make_empty() const {
        bitset::BitsetVector b;
        b.allocate((num_rows_ + 63) / 64);
        b.set_num_bits(num_rows_);
        return b;
    }

    void apply_or_to(bitset::BitsetVector& dst, int64_t key) const {
        auto [p, n] = positions_for(key);
        scatter(dst, p, n);
    }

    void apply_or_range_to(bitset::BitsetVector& dst, int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        for (size_t i = i_lo; i < i_hi; i++) {
            auto [p, n] = positions_at(i);
            scatter(dst, p, n);
        }
    }

    template <typename Iterable>
    bitset::BitsetVector or_many(const Iterable& keys) const {
        bitset::BitsetVector dst = make_empty();
        for (auto& k : keys) {
            auto [p, n] = positions_for(static_cast<int64_t>(k));
            scatter(dst, p, n);
        }
        return dst;
    }

    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    size_t num_keys() const override { return num_distinct_keys(); }
    size_t num_rows() const override { return num_rows_; }

    // Dense footprint
    size_t storage_bytes() const override {
        return num_distinct_keys() * ((num_rows_ + 7) / 8);
    }
    const char* backend_name() const override {
        return simd_ ? "Bitset+AVX512" : "Bitset";
    }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return {{"dense_words", storage_bytes()}};
    }

private:
    bool simd_ = true;

    static void scatter(bitset::BitsetVector& b, const uint32_t* p, size_t n) {
        uint64_t* w = b.words_mut();
        for (size_t i = 0; i < n; i++) w[p[i] >> 6] |= uint64_t(1) << (p[i] & 63);
    }
};

}
