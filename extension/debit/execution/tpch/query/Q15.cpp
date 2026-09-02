

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
#include <chrono>
#include <fstream>
#include <iomanip>
#include "execution/tpch/bm_bench_common.hpp"
#include "execution/tpch/indexed_bitmap.hpp"

#include <atomic>
#include <iostream>
#include <mutex>
#include <vector>

namespace duckdb {

static std::mutex     q15_build_mutex;
using clk = std::chrono::high_resolution_clock;

static std::atomic<bool> q15_built{false};
static double q15_fetch_ms = 0.0;
static double q15_or_ms = 0.0;
static double q15_or_sd = 0.0;
static int    q15_or_n  = 0;
static std::string q15_backend_name;

static const int Q15_ITERATIONS = bm_bench::iter_count(5);
static const int Q15_WARMUP     = bm_bench::warmup_count(1);

static constexpr int Q15_DATE_START = 9496;
static constexpr int Q15_DATE_END   = 9586;
static constexpr int64_t Q15_MONTH_KEYS[3] = {199601, 199602, 199603};

template <typename Btv>
static void q15_get_rowids(const Btv& b, vector<row_t>* out);
template <>
void q15_get_rowids<DDC>(const DDC& b, vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++)
            out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q15_get_rowids<roaring::Roaring>(const roaring::Roaring& b, vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q15_get_rowids<ibis::bitvector>(const ibis::bitvector& b, vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q15_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q15_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q15_get_rowids<::bitset::BitsetVector>(const ::bitset::BitsetVector& b, vector<row_t>* out) {
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


void BMTableScan::TPCH_Q15_Lineitem_GetRowIds(ExecutionContext &context,
                                              vector<row_t> *row_ids)
{
    if (q15_built.load()) return;

    auto* idx_ge_m = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate_GE_month);
    auto* idx_ship = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipdate);
    auto* idx_use  = idx_ge_m ? idx_ge_m : idx_ship;
    if (!idx_use) {
        std::cerr << "[Q15] ERROR: neither bitmap_shipdate_GE_month nor bitmap_shipdate loaded.\n";
        row_ids->clear();
        q15_built.store(true);
        return;
    }
    const bool use_month_ge = (idx_ge_m != nullptr);
    const std::vector<int64_t> q15_ge_keys(Q15_MONTH_KEYS, Q15_MONTH_KEYS + 3);

    row_ids->clear();

    std::vector<double> tot_t;
    const char* q15_backend = idx_use->backend_name();
    auto q15_run_once = [&]() {
    row_ids->clear();
    if (auto* cbge = dynamic_cast<bm_index::IndexedDDCGE*>(idx_use)) {
        DDC f = cbge->or_many(q15_ge_keys);
        q15_get_rowids(f, row_ids);
    } else if (auto* cb = dynamic_cast<bm_index::IndexedDDC*>(idx_use)) {
        DDC f = DDC::from_sparse_positions({}, cb->num_rows(), cb->segment_bits());
        cb->apply_or_range_to(f, Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* cr = dynamic_cast<bm_index::IndexedCRoaring*>(idx_use)) {
        roaring::Roaring f = use_month_ge
            ? cr->or_many(q15_ge_keys)
            : cr->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* bs = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_use)) {
        ::bitset::BitsetVector f = bs->make_empty();
        if (use_month_ge) f = bs->or_many(q15_ge_keys);
        else              bs->apply_or_range_to(f, Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* wah = dynamic_cast<bm_index::IndexedWAH*>(idx_use)) {
        ibis::bitvector f = use_month_ge
            ? wah->or_many(q15_ge_keys)
            : wah->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* ew = dynamic_cast<bm_index::IndexedEWAH*>(idx_use)) {
        ewah::EWAHBoolArray<uint64_t> f = use_month_ge
            ? ew->or_many(q15_ge_keys)
            : ew->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else if (auto* con = dynamic_cast<bm_index::IndexedConcise*>(idx_use)) {
        ConciseSet<false> f = use_month_ge
            ? con->or_many(q15_ge_keys)
            : con->or_range(Q15_DATE_START, Q15_DATE_END);
        q15_get_rowids(f, row_ids);
    } else {
        std::cerr << "[Q15] ERROR: unrecognised IBitmapIndex backend.\n";
    }
    };

    for (int it = 0; it < Q15_ITERATIONS; it++) {
        auto t0 = clk::now();
        q15_run_once();
        auto t1 = clk::now();
        if (it >= Q15_WARMUP)
            tot_t.push_back(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
    }

    auto sT = bm_bench::compute_stats(tot_t);
    q15_or_ms = sT.median;
    q15_or_sd = sT.stddev;
    q15_or_n  = (int)tot_t.size();
    q15_backend_name = q15_backend;
    q15_fetch_ms = 0.0;

    q15_built.store(true);
}

SourceResultType BMTableScan::BMTPCH_Q15(ExecutionContext &context,
                                          DataChunk &chunk,
                                          const TableScanBindData &bind_data)
{
    std::lock_guard<std::mutex> lock(q15_build_mutex);

    if (*cursor == 0 && !q15_built.load()) {
        TPCH_Q15_Lineitem_GetRowIds(context, row_ids);
        num_idlist = row_ids->size();
    }

    if (*cursor < row_ids->size()) {

        vector<StorageIndex> storage_column_ids;
        storage_column_ids.push_back(StorageIndex(2));
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
        table_bind_data.table.GetStorage().BMFetch(transaction, chunk,
                                                   storage_column_ids, row_ids_vec,
                                                   fetch_count, column_fetch_state,
                                                   num_idlist);
        q15_fetch_ms +=
            std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - _f0).count() / 1000.0;
        *cursor += fetch_count;
        return SourceResultType::HAVE_MORE_OUTPUT;
    }

    {
        const double tot = q15_or_ms + q15_fetch_ms;
        const std::string sf = bm_bench::sf_label();
        std::cout << "\n================================================================\n"
                  << "  Q15 RESULTS \u2014 " << q15_backend_name << " (bitmap phase: "
                  << q15_or_n << " measured iter, median +/- stddev; fetch: single pass)\n"
                  << "================================================================\n"
                  << "  PhaseA_OR (shipdate range OR) : " << q15_or_ms << " +/- " << q15_or_sd << " ms\n"
                  << "  PhaseB_fetch (BMFetch lineitem)   : " << q15_fetch_ms << " +/- 0 ms\n"
                  << "  TOTAL                                : " << tot << " +/- " << q15_or_sd << " ms\n"
                  << "================================================================\n\n";

        std::ofstream csv("q15_results_" + sf + ".csv");
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
            const std::string bn = q15_backend_name;
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
            put("PhaseA_OR", bm_bench::Stats{q15_or_ms, q15_or_sd, q15_or_ms, q15_or_ms});
            put("PhaseB_fetch", bm_bench::Stats{q15_fetch_ms, 0, q15_fetch_ms, q15_fetch_ms});
            put("TOTAL", bm_bench::Stats{tot, q15_or_sd, tot, tot});
            std::cout << "  [CSV] q15_results_" << sf << ".csv\n";
        }
    }

    row_ids->clear();
    *cursor = 0;
    num_idlist = 0;
    q15_built.store(false);
    context.client.query_source = "tpch";

    return SourceResultType::FINISHED;
}

}
