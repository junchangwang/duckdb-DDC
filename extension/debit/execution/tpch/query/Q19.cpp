

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "execution/tpch/bitmap_table_scan.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "ddc_adapter.h"
#include "ddc/include/ddc.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah.h"
#include "Concise/concise.h"

#include "execution/tpch/bm_baseline_loaders.hpp"
#include <fstream>
#include <iomanip>
#include "execution/tpch/bm_bench_common.hpp"
#include "execution/tpch/indexed_bitmap.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;

static std::mutex     q19_build_mutex;
static std::atomic<bool> q19_built{false};
static double q19_fetch_ms = 0.0;
static double q19_or_ms = 0.0;
static double q19_or_sd = 0.0;
static int    q19_or_n  = 0;
static std::string q19_backend_name;

static const int Q19_ITERATIONS = bm_bench::iter_count(5);
static const int Q19_WARMUP     = bm_bench::warmup_count(1);

template <typename Btv>
static void q19_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q19_get_rowids<DDC>(const DDC& b, std::vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q19_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q19_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q19_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q19_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q19_get_rowids<::bitset::BitsetVector>(const ::bitset::BitsetVector& b, std::vector<row_t>* out) {
    out->clear();
    const uint64_t* w = b.words();
    const uint64_t nb = b.num_bits();
    for (size_t i = 0; i < b.words_cnt(); ++i) {
        uint64_t x = w[i];
        const uint64_t base = static_cast<uint64_t>(i) * 64;
        while (x) {
            const uint64_t pos = base + __builtin_ctzll(x);
            if (pos >= nb) return;
            out->push_back(static_cast<row_t>(pos));
            x &= x - 1;
        }
    }
}


void BMTableScan::TPCH_Q19_Lineitem_GetRowIds(ExecutionContext &context,
                                              vector<row_t> *row_ids)
{
    if (q19_built.load()) return;

    auto* idx_sm = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipmode);
    auto* idx_si = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipinstr);
    if (!idx_sm || !idx_si) {
        std::cerr << "[Q19] ERROR: bitmap_shipmode or bitmap_shipinstr not loaded.\n";
        row_ids->clear();
        q19_built.store(true);
        return;
    }

    size_t num_rows = idx_sm->num_rows();
    row_ids->clear();

    std::vector<double> tot_t;
    const char* q19_backend = idx_sm->backend_name();
    auto q19_run_once = [&]() {
    row_ids->clear();
    if (auto* cb_sm = dynamic_cast<bm_index::IndexedDDC*>(idx_sm)) {
        auto* cb_si = dynamic_cast<bm_index::IndexedDDC*>(idx_si);
        if (!cb_si) { std::cerr << "[Q19] CB type mismatch.\n"; return; }
        DDC bm_sm = DDC::from_sparse_positions({}, num_rows, cb_sm->segment_bits());
        DDC bm_si = DDC::from_sparse_positions({}, num_rows, cb_si->segment_bits());
        cb_sm->apply_or_to(bm_sm, 'A');
        cb_si->apply_or_to(bm_si, 'D');
        bm_sm &= bm_si;
        q19_get_rowids(bm_sm, row_ids);
    } else if (auto* cr_sm = dynamic_cast<bm_index::IndexedCRoaring*>(idx_sm)) {
        auto* cr_si = dynamic_cast<bm_index::IndexedCRoaring*>(idx_si);
        if (!cr_si) { std::cerr << "[Q19] CR type mismatch.\n"; return; }
        roaring::Roaring bm_sm, bm_si;
        cr_sm->apply_or_to(bm_sm, 'A');
        cr_si->apply_or_to(bm_si, 'D');
        bm_sm &= bm_si;
        q19_get_rowids(bm_sm, row_ids);
    } else if (auto* bs_sm = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_sm)) {
        auto* bs_si = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_si);
        if (!bs_si) { std::cerr << "[Q19] BS type mismatch.\n"; return; }
        ::bitset::BitsetVector bm_sm = bs_sm->make_empty();
        ::bitset::BitsetVector bm_si = bs_si->make_empty();
        bs_sm->apply_or_to(bm_sm, 'A');
        bs_si->apply_or_to(bm_si, 'D');
        ::bitset::BitsetVector::word_and_inplace(bm_sm, bm_si, bs_sm->use_simd());
        q19_get_rowids(bm_sm, row_ids);
    } else if (auto* wah_sm = dynamic_cast<bm_index::IndexedWAH*>(idx_sm)) {
        auto* wah_si = dynamic_cast<bm_index::IndexedWAH*>(idx_si);
        if (!wah_si) { std::cerr << "[Q19] WAH type mismatch.\n"; return; }
        ibis::bitvector bm_sm, bm_si;
        wah_sm->apply_or_to(bm_sm, 'A');
        wah_si->apply_or_to(bm_si, 'D');
        bm_sm &= bm_si;
        q19_get_rowids(bm_sm, row_ids);
    } else if (auto* ew_sm = dynamic_cast<bm_index::IndexedEWAH*>(idx_sm)) {
        auto* ew_si = dynamic_cast<bm_index::IndexedEWAH*>(idx_si);
        if (!ew_si) { std::cerr << "[Q19] EW type mismatch.\n"; return; }
        ewah::EWAHBoolArray<uint64_t> bm_sm, bm_si;
        ew_sm->apply_or_to(bm_sm, 'A');
        ew_si->apply_or_to(bm_si, 'D');
        ewah::EWAHBoolArray<uint64_t> tmp;
        bm_sm.logicaland(bm_si, tmp);
        q19_get_rowids(tmp, row_ids);
    } else if (auto* con_sm = dynamic_cast<bm_index::IndexedConcise*>(idx_sm)) {
        auto* con_si = dynamic_cast<bm_index::IndexedConcise*>(idx_si);
        if (!con_si) { std::cerr << "[Q19] CON type mismatch.\n"; return; }
        ConciseSet<false> bm_sm, bm_si;
        con_sm->apply_or_to(bm_sm, 'A');
        con_si->apply_or_to(bm_si, 'D');
        ConciseSet<false> tmp = bm_sm.logicaland(bm_si);
        q19_get_rowids(tmp, row_ids);
    } else {
        std::cerr << "[Q19] ERROR: unrecognised IBitmapIndex backend.\n";
    }
    };

    for (int it = 0; it < Q19_ITERATIONS; it++) {
        auto t0 = clk::now();
        q19_run_once();
        auto t1 = clk::now();
        if (it >= Q19_WARMUP)
            tot_t.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
    }

    auto sT = bm_bench::compute_stats(tot_t);
    q19_or_ms = sT.median;
    q19_or_sd = sT.stddev;
    q19_or_n  = (int)tot_t.size();
    q19_backend_name = q19_backend;
    q19_fetch_ms = 0.0;

    q19_built.store(true);
}

SourceResultType BMTableScan::BMTPCH_Q19(ExecutionContext &context,
                                          DataChunk &chunk,
                                          const TableScanBindData &bind_data)
{
    std::lock_guard<std::mutex> lock(q19_build_mutex);

    if (*cursor == 0 && !q19_built.load()) {
        TPCH_Q19_Lineitem_GetRowIds(context, row_ids);
        num_idlist = row_ids->size();
    }

    if (*cursor < row_ids->size()) {

        vector<StorageIndex> storage_column_ids;
        storage_column_ids.push_back(StorageIndex(1));
        storage_column_ids.push_back(StorageIndex(4));
        storage_column_ids.push_back(StorageIndex(14));
        storage_column_ids.push_back(StorageIndex(5));
        storage_column_ids.push_back(StorageIndex(6));

        TableScanState local_storage_state;
        local_storage_state.Initialize(storage_column_ids);
        ColumnFetchState column_fetch_state;

        auto &table_bind_data = bind_data;
        auto &transaction = DuckTransaction::Get(context.client, table_bind_data.table.catalog);

        data_ptr_t row_ids_data = (data_ptr_t)&((*row_ids)[*cursor]);
        Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
        idx_t fetch_count = 2048;
        if (*cursor + fetch_count > row_ids->size())
            fetch_count = row_ids->size() - *cursor;

        auto _f0 = clk::now();
        table_bind_data.table.GetStorage().Fetch(transaction, chunk,
                                                 storage_column_ids, row_ids_vec,
                                                 fetch_count, column_fetch_state);
        q19_fetch_ms +=
            std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - _f0).count() / 1000.0;
        *cursor += fetch_count;
        return SourceResultType::HAVE_MORE_OUTPUT;
    }

    {
        const double tot = q19_or_ms + q19_fetch_ms;
        const std::string sf = bm_bench::sf_label();
        std::cout << "\n================================================================\n"
                  << "  Q19 RESULTS \u2014 " << q19_backend_name << " (bitmap phase: "
                  << q19_or_n << " measured iter, median +/- stddev; fetch: single pass)\n"
                  << "================================================================\n"
                  << "  PhaseA_ship (bitmap OR+AND+rowids) : " << q19_or_ms << " +/- " << q19_or_sd << " ms\n"
                  << "  PhaseB_fetch (Fetch lineitem)   : " << q19_fetch_ms << " +/- 0 ms\n"
                  << "  TOTAL                                : " << tot << " +/- " << q19_or_sd << " ms\n"
                  << "================================================================\n\n";

        std::ofstream csv("q19_results_" + sf + ".csv");
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
                << "concise_median_ms,concise_stddev_ms,concise_min_ms,concise_max_ms\n";
            bm_bench::Stats z{0, 0, 0, 0};
            const std::string bn = q19_backend_name;
            auto put = [&](const char* op, bm_bench::Stats v) {
                csv << sf << "," << op << ",";
                auto cell = [&](bm_bench::Stats x) {
                    csv << x.median << "," << x.stddev << "," << x.min_val << "," << x.max_val << ",";
                };
                cell(bn == "WAH" ? v : z);
                cell(bn.rfind("DDC", 0) == 0 ? v : z);
                cell(bn == "CRoaring" ? v : z);
                cell(bn == "CRoaringRun" ? v : z);
                cell(bn == "EWAH" ? v : z);
                cell(bn == "Bitset" ? v : z);
                cell(bn == "Bitset+AVX512" ? v : z);
                cell(bn == "Concise" ? v : z);
                csv << "\n";
            };
            put("PhaseA_ship", bm_bench::Stats{q19_or_ms, q19_or_sd, q19_or_ms, q19_or_ms});
            put("PhaseB_fetch", bm_bench::Stats{q19_fetch_ms, 0, q19_fetch_ms, q19_fetch_ms});
            put("TOTAL", bm_bench::Stats{tot, q19_or_sd, tot, tot});
            std::cout << "  [CSV] q19_results_" << sf << ".csv\n";
        }
    }

    row_ids->clear();
    *cursor = 0;
    num_idlist = 0;
    q19_built.store(false);
    context.client.query_source = "tpch";

    return SourceResultType::FINISHED;
}

}
