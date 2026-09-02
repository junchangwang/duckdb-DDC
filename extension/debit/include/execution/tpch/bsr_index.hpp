// BSR baseline

#pragma once

#include "indexed_bitmap.hpp"
#include "intersection_algos.hpp"

#include <cstdlib>
#include <cstring>

namespace bm_index {

struct BsrSet {
    static constexpr int PAD = 8;
    int*      bases  = nullptr;
    int*      states = nullptr;
    int       size   = 0;
    uint64_t  num_rows = 0;

    BsrSet() = default;
    BsrSet(const BsrSet&) = delete;
    BsrSet& operator=(const BsrSet&) = delete;
    BsrSet(BsrSet&& o) noexcept { steal(o); }
    BsrSet& operator=(BsrSet&& o) noexcept {
        if (this != &o) { release(); steal(o); }
        return *this;
    }
    ~BsrSet() { release(); }

    void alloc(int capacity) {
        release();
        size_t cap = static_cast<size_t>(capacity < 0 ? 0 : capacity);
        size_t n = cap + PAD;
        if (posix_memalign(reinterpret_cast<void**>(&bases), 64, n * sizeof(int)) != 0
            || posix_memalign(reinterpret_cast<void**>(&states), 64, n * sizeof(int)) != 0)
            std::abort();
        std::memset(bases  + cap, 0, PAD * sizeof(int));
        std::memset(states + cap, 0, PAD * sizeof(int));
    }

    uint64_t popcount() const {
        uint64_t t = 0;
        for (int i = 0; i < size; i++)
            t += static_cast<uint64_t>(__builtin_popcount(static_cast<unsigned>(states[i])));
        return t;
    }

    uint64_t bytes() const { return static_cast<uint64_t>(size) * 8; }

    template <typename RowT>
    void get_rowids(std::vector<RowT>* out) const {
        out->clear();
        out->reserve(popcount());
        for (int i = 0; i < size; i++) {
            uint32_t st = static_cast<uint32_t>(states[i]);
            uint64_t base_bit = static_cast<uint64_t>(static_cast<uint32_t>(bases[i])) * 32ULL;
            while (st) {
                int b = __builtin_ctz(st);
                st &= st - 1;
                out->push_back(static_cast<RowT>(base_bit + static_cast<uint64_t>(b)));
            }
        }
    }

private:
    void release() { free(bases); free(states); bases = states = nullptr; size = 0; }
    void steal(BsrSet& o) {
        bases = o.bases; states = o.states; size = o.size; num_rows = o.num_rows;
        o.bases = o.states = nullptr; o.size = 0;
    }
};

// Adaptive AND
inline BsrSet bsr_and(const BsrSet& a, const BsrSet& b) {
    BsrSet r;
    r.num_rows = a.num_rows ? a.num_rows : b.num_rows;
    r.alloc(std::min(a.size, b.size));
    if (a.size == 0 || b.size == 0) { r.size = 0; return r; }
    const long long big = std::max(a.size, b.size);
    const long long small = std::max(1, std::min(a.size, b.size));
    if (big / small > 32)
        r.size = intersect_simdgalloping_bsr(a.bases, a.states, a.size,
                                             b.bases, b.states, b.size,
                                             r.bases, r.states);
    else
        r.size = intersect_qfilter_bsr_b4_v2(a.bases, a.states, a.size,
                                             b.bases, b.states, b.size,
                                             r.bases, r.states);
    return r;
}

class IndexedBSR : public IBitmapIndex, public InvertedIndex {
public:
    IndexedBSR() = default;

    void build(const std::vector<int64_t>& keys, size_t num_rows) {
        build_inverted_index(keys, num_rows);
    }

    bool has(int64_t key) const { return has_key(key); }


    // Dense builder
    template <typename PosFeed>
    BsrSet build_union(size_t total_positions, PosFeed&& feed) const {
        BsrSet out;
        out.num_rows = num_rows_;
        const size_t num_chunks = (num_rows_ + 31) / 32;
        if (total_positions >= num_chunks / 2) {
            std::vector<uint32_t> dense(num_chunks, 0);
            feed([&](const uint32_t* p, size_t n) {
                for (size_t k = 0; k < n; k++)
                    dense[p[k] >> 5] |= (1u << (p[k] & 31));
            });
            size_t nz = 0;
            for (size_t i = 0; i < num_chunks; i++) nz += (dense[i] != 0);
            out.alloc(static_cast<int>(nz));
            int m = 0;
            for (size_t i = 0; i < num_chunks; i++) {
                if (dense[i]) {
                    out.bases[m] = static_cast<int>(i);
                    out.states[m] = static_cast<int>(dense[i]);
                    m++;
                }
            }
            out.size = m;
        } else {
            // Sparse builder
            std::vector<int> pos;
            pos.reserve(total_positions);
            feed([&](const uint32_t* p, size_t n) {
                pos.insert(pos.end(), p, p + n);
            });
            std::sort(pos.begin(), pos.end());
            out.alloc(static_cast<int>(pos.size()));
            out.size = pos.empty() ? 0
                     : offline_uint_trans_bsr(pos.data(), static_cast<int>(pos.size()),
                                              out.bases, out.states);
        }
        return out;
    }

    template <typename Iterable>
    BsrSet or_many(const Iterable& keys) const {
        std::vector<std::pair<const uint32_t*, size_t>> runs;
        size_t total = 0;
        {
            std::vector<int64_t> in;
            for (auto& k : keys) in.push_back(static_cast<int64_t>(k));
            std::sort(in.begin(), in.end());
            in.erase(std::unique(in.begin(), in.end()), in.end());
            for_each_matched_key(in.data(), in.size(),
                [&](const uint32_t* p, size_t n) { runs.push_back({p, n}); total += n; });
        }
        return build_union(total, [&](auto&& sink) {
            for (auto& r : runs) sink(r.first, r.second);
        });
    }

    BsrSet or_range(int64_t lo, int64_t hi) const {
        size_t i_lo = lower_bound_idx(lo);
        size_t i_hi = upper_bound_idx(hi);
        size_t total = 0;
        for (size_t i = i_lo; i < i_hi; i++) total += positions_at(i).second;
        return build_union(total, [&](auto&& sink) {
            for (size_t i = i_lo; i < i_hi; i++) {
                auto [p, n] = positions_at(i);
                sink(p, n);
            }
        });
    }

    BsrSet single_key(int64_t key) const {
        auto [p, n] = positions_for(key);
        return build_union(n, [&](auto&& sink) { if (n) sink(p, n); });
    }

    template <typename F>
    void for_each_key(F&& f) const { for_each_key_base(std::forward<F>(f)); }

    size_t num_keys()     const override { return num_distinct_keys(); }
    size_t num_rows()     const override { return num_rows_; }
    size_t storage_bytes() const override {
        cached_native_layers([this]{ return compute_layers(); });
        return cached_native_total_bytes();
    }
    const char* backend_name() const override { return "QFilter-BSR"; }
    std::vector<std::pair<std::string, size_t>> layer_breakdown() const override {
        return cached_native_layers([this]{ return compute_layers(); });
    }

private:
    std::vector<std::pair<std::string, size_t>> compute_layers() const {
        size_t base_bytes = 0, state_bytes = 0;
        std::vector<int> tmp;
        int *b = nullptr, *s = nullptr;
        int cap = 0;
        for_each_key_positions([&](const uint32_t* p, size_t n) {
            if (!n) return;
            tmp.assign(p, p + n);
            if (static_cast<int>(n) + BsrSet::PAD > cap) {
                free(b); free(s);
                cap = static_cast<int>(n) + BsrSet::PAD;
                if (posix_memalign(reinterpret_cast<void**>(&b), 64, cap * sizeof(int)) != 0
                    || posix_memalign(reinterpret_cast<void**>(&s), 64, cap * sizeof(int)) != 0)
                    std::abort();
            }
            int sz = offline_uint_trans_bsr(tmp.data(), static_cast<int>(n), b, s);
            base_bytes  += static_cast<size_t>(sz) * 4;
            state_bytes += static_cast<size_t>(sz) * 4;
        });
        free(b); free(s);
        return {{"bases", base_bytes}, {"states", state_bytes}};
    }
};

}
