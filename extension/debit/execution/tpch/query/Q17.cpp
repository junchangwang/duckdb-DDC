

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
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
#include "execution/tpch/bm_bench_common.hpp"
#include "execution/tpch/indexed_bitmap.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q17_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q17_ITERATIONS = bm_bench::iter_count(5);
static const int Q17_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q17_once_flag_new;

template <typename Btv>
static void q17_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q17_get_rowids<DDC>(const DDC& b, std::vector<row_t>* out) {
    // DDC -> rowids
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++) out->push_back(static_cast<row_t>(rbase + e.pos[k]));
    });
}
template <>
void q17_get_rowids<roaring::Roaring>(const roaring::Roaring& b, std::vector<row_t>* out) {
    out->clear(); out->reserve(b.cardinality());
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q17_get_rowids<ibis::bitvector>(const ibis::bitvector& b, std::vector<row_t>* out) {
    out->clear();
    ibis::bitvector::pit pit(b);
    while (*pit != 0xFFFFFFFFU) { out->push_back(static_cast<row_t>(*pit)); pit.next(); }
}
template <>
void q17_get_rowids<ewah::EWAHBoolArray<uint64_t>>(
    const ewah::EWAHBoolArray<uint64_t>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q17_get_rowids<ConciseSet<false>>(const ConciseSet<false>& b, std::vector<row_t>* out) {
    out->clear();
    for (auto it = b.begin(); it != b.end(); ++it) out->push_back(static_cast<row_t>(*it));
}
template <>
void q17_get_rowids<::bitset::BitsetVector>(const ::bitset::BitsetVector& b, std::vector<row_t>* out) {
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


// Q17 entry
void BMTableScan::BMTPCH_Q17(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q17_once_flag_new, [&]() {
    bm_bench::warn_if_sf1();

    auto& part_table     = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "part");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q17 (BitEngine pattern, runtime part-join)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_partkey" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_pk = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_partkey);
    if (!idx_pk) { std::cerr << "[Q17] ERROR: bitmap_partkey not loaded.\n"; return; }

    std::vector<double> tA_t, tB_t, tC_t, tot_t;
    double last_avg_yearly = 0;

    for (int iter = 0; iter < Q17_ITERATIONS; iter++) {
        bool warm = iter < Q17_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q17_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        auto* cb_pk  = dynamic_cast<bm_index::IndexedDDC*>(idx_pk);
        auto* cr_pk  = dynamic_cast<bm_index::IndexedCRoaring*>(idx_pk);
        auto* wah_pk = dynamic_cast<bm_index::IndexedWAH*>(idx_pk);
        auto* bs_pk  = dynamic_cast<bm_index::IndexedBitsetAVX512*>(idx_pk);
        auto* ew_pk  = dynamic_cast<bm_index::IndexedEWAH*>(idx_pk);
        auto* con_pk = dynamic_cast<bm_index::IndexedConcise*>(idx_pk);

        std::vector<int64_t> matched_pks;
        matched_pks.reserve(2000);
        {
            // Part filter
            auto& tx = DuckTransaction::Get(context.client, part_table.catalog);
            TableScanState ss;
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(3), StorageIndex(6)};
            part_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types;
            for (int c : {0, 3, 6}) types.push_back(part_table.GetColumns().GetColumnTypes()[c]);
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                part_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[1].Flatten(chunk.size());
                chunk.data[2].Flatten(chunk.size());
                auto pk = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto pb = FlatVector::GetData<string_t>(chunk.data[1]);
                auto pc = FlatVector::GetData<string_t>(chunk.data[2]);
                auto& vb = FlatVector::Validity(chunk.data[1]);
                auto& vc = FlatVector::Validity(chunk.data[2]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (!vb.RowIsValid(i) || !vc.RowIsValid(i)) continue;
                    if (pb[i].GetSize() == 8 &&
                        std::memcmp(pb[i].GetData(), "Brand#23", 8) == 0 &&
                        pc[i].GetSize() == 7 &&
                        std::memcmp(pc[i].GetData(), "MED BOX", 7) == 0) {
                        matched_pks.push_back(pk[i]);
                    }
                }
            }
        }

        std::vector<row_t> ids;

        // Bitmap filter
        if (cb_pk) {
            auto bv = cb_pk->or_many(matched_pks); q17_get_rowids(bv, &ids);
        } else if (cr_pk) {

            roaring::Roaring bv = cr_pk->or_many(matched_pks);
            q17_get_rowids(bv, &ids);
        } else if (bs_pk) {
            ::bitset::BitsetVector bv = bs_pk->or_many(matched_pks);
            q17_get_rowids(bv, &ids);
        } else if (wah_pk) {
            auto bv = wah_pk->or_many(matched_pks); q17_get_rowids(bv, &ids);
        } else if (ew_pk) {
            auto bv = ew_pk->or_many(matched_pks); q17_get_rowids(bv, &ids);
        } else if (con_pk) {
            auto bv = con_pk->or_many(matched_pks); q17_get_rowids(bv, &ids);
        }
        else { std::cerr << "[Q17] ERROR: unrecognised backend.\n"; return; }
        auto t_a = clk::now();

        std::vector<int64_t> col_pk, col_qty, col_pr;
        col_pk.reserve(ids.size());
        col_qty.reserve(ids.size());
        col_pr.reserve(ids.size());
        std::unordered_map<int64_t, std::pair<int64_t, int32_t>> pk_qty;

        if (!ids.empty()) {
            // BMFetch lineitem cols
            vector<StorageIndex> col_ids = {StorageIndex(1), StorageIndex(4), StorageIndex(5)};
            vector<LogicalType> types = {
                lineitem_table.GetColumns().GetColumnTypes()[1],
                lineitem_table.GetColumns().GetColumnTypes()[4],
                lineitem_table.GetColumns().GetColumnTypes()[5],
            };
            idx_t cursor = 0;
            idx_t num_idlist = ids.size();
            while (cursor < ids.size()) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                ColumnFetchState column_fetch_state;
                data_ptr_t row_ids_data = (data_ptr_t)&ids[cursor];
                Vector row_ids_vec(LogicalType::ROW_TYPE, row_ids_data);
                idx_t fetch_count = 2048;
                if (cursor + fetch_count > ids.size()) fetch_count = ids.size() - cursor;
                lineitem_table.GetStorage().BMFetch(lineitem_tx, chunk, col_ids, row_ids_vec,
                                                    fetch_count, column_fetch_state, num_idlist);
                cursor += fetch_count;
                auto pk = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto qt = FlatVector::GetData<int64_t>(chunk.data[1]);
                auto pr = FlatVector::GetData<int64_t>(chunk.data[2]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    // Quantity stats
                    auto& slot = pk_qty[pk[i]];
                    slot.first  += qt[i];
                    slot.second += 1;
                    col_pk.push_back(pk[i]);
                    col_qty.push_back(qt[i]);
                    col_pr.push_back(pr[i]);
                }
            }
        }
        // Quantity threshold
        std::unordered_map<int64_t, double> pk_avg_qty;
        for (auto& [k, v] : pk_qty)
            pk_avg_qty[k] = 0.2 * double(v.first) / double(v.second);
        auto t_b = clk::now();

        // Final aggregate
        int64_t sum_extprice = 0;
        for (size_t i = 0; i < col_pk.size(); i++) {
            auto it = pk_avg_qty.find(col_pk[i]);
            if (it != pk_avg_qty.end() && double(col_qty[i]) < it->second)
                sum_extprice += col_pr[i];
        }
        auto t_c = clk::now();

        double tA  = q17_ms(t0, t_a);
        double tB  = q17_ms(t_a, t_b);
        double tC  = q17_ms(t_b, t_c);
        double tot = q17_ms(t0, t_c);

        double avg_yearly = double(sum_extprice) / 100.0 / 7.0;

        std::cout << "  " << idx_pk->backend_name()
                  << ":  PhaseA(part_scan+pk_OR)=" << tA
                  << "  PhaseB(BMFetch+avg)=" << tB
                  << "  PhaseC(filter+sum)=" << tC
                  << "  Total=" << tot
                  << "  rows=" << ids.size()
                  << "  avg_yearly=" << std::fixed << std::setprecision(6) << avg_yearly
                  << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot);
        }
        last_avg_yearly = avg_yearly;
    }

    {
        // SQL validation
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT sum(l_extendedprice) / 7.0 AS avg_yearly "
            "FROM lineitem, part "
            "WHERE p_partkey = l_partkey "
            "  AND p_brand = 'Brand#23' "
            "  AND p_container = 'MED BOX' "
            "  AND l_quantity < (SELECT 0.2 * avg(l_quantity) FROM lineitem "
            "                     WHERE l_partkey = p_partkey)");
        if (r && !r->HasError() && r->RowCount() == 1) {
            double gt = r->GetValue(0, 0).GetValue<double>();
            if (std::fabs(gt - last_avg_yearly) < 0.5)
                std::cout << "\n[OK] " << idx_pk->backend_name()
                          << " matches DuckDB SQL (avg_yearly = " << gt << ").\n";
            else
                std::cerr << "\n[FAIL] mismatch: ours=" << last_avg_yearly
                          << " gt=" << gt << "\n";
        }
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t), sT = stats(tot_t);
    int measured = std::max(0, Q17_ITERATIONS - Q17_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q17 RESULTS — " << idx_pk->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (part scan + partkey OR)    : " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (BMFetch + per-pk avg)      : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (filter qty<avg + sum_price): " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  TOTAL                              : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    // emit CSV
    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q17_results_" + sf + ".csv");
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
        std::string bn = idx_pk->backend_name();
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
        put("PhaseA_part",  sA);
        put("PhaseB_avg",   sB);
        put("PhaseC_sum",   sC);
        put("TOTAL", sT);
        std::cout << "  [CSV] q17_results_" << sf << ".csv\n";
    }

    });
}

}
