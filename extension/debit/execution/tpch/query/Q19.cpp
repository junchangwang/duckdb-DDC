// TPC-H Q19 — Discounted Revenue Query (spec v3.0.1 §2.4.19)
//
// Verbatim port of teacher's BitEngine BMTPCH_Q19 — replaces the prior
// non-compliant pre-joined join_result/0.bm (lineitem×part) with
// runtime semi-join over single-column auxiliaries.
//
// Predicate: 3 OR groups, common shipmode∈{AIR, AIR REG} AND
// shipinstruct = 'DELIVER IN PERSON':
//   group1: brand=#12, container ∈ {SM ...},  size 1..5,  qty 1..11
//   group2: brand=#23, container ∈ {MED ...}, size 1..10, qty 10..20
//   group3: brand=#34, container ∈ {LG ...},  size 1..15, qty 20..30
//
// Pipeline:
//   Phase A: bitmap_shipmode['A'] AND bitmap_shipinstruct['D']
//            ('A' first byte covers both AIR + AIR REG; 'D' is unique
//             for DELIVER IN PERSON).
//   Phase B: part scan once → unordered_map<partkey, GroupSpec> with
//            the per-group quantity range.
//   Phase C: BMFetch lineitem(l_partkey, l_quantity, l_extendedprice,
//            l_discount); per row look up partkey → group; if quantity
//            falls in range, sum extendedprice * (100 - discount).
//
// Pre-loaded auxiliary structures (all single column):
//   bitmap_shipmode      lineitem.l_shipmode
//   bitmap_shipinstruct  lineitem.l_shipinstruct

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
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace duckdb {

using clk = std::chrono::high_resolution_clock;
static inline double q19_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
}

static const int Q19_ITERATIONS = bm_bench::iter_count(5);
static const int Q19_WARMUP     = bm_bench::warmup_count(1);
static std::once_flag q19_once_flag_new;

template <typename Btv>
static void q19_get_rowids(const Btv& b, std::vector<row_t>* out);
template <>
void q19_get_rowids<ComBit>(const ComBit& b, std::vector<row_t>* out) {
    out->clear();
    b.for_each_literal([&](uint32_t word_pos, uint8_t val) {
        size_t rbase = static_cast<size_t>(word_pos) * 8;
        const auto& e = bm_bench::byte_lut[val];
        for (int k = 0; k < e.count; k++) out->push_back(static_cast<row_t>(rbase + e.pos[k]));
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

void BMTableScan::BMTPCH_Q19(ExecutionContext &context, const PhysicalTableScan &op)
{
    std::call_once(q19_once_flag_new, [&]() {
    bm_bench::warn_if_sf1();

    auto& part_table     = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "part");
    auto& lineitem_table = Catalog::GetEntry<TableCatalogEntry>(context.client, "", "", "lineitem");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  TPC-H Q19 (BitEngine pattern, runtime 3-OR part-join)" << std::endl;
    std::cout << "  Pre-loaded: bitmap_shipmode + bitmap_shipinstr" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto* idx_sm = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipmode);
    auto* idx_si = static_cast<bm_index::IBitmapIndex*>(context.client.bitmap_shipinstr);
    if (!idx_sm || !idx_si) {
        std::cerr << "[Q19] ERROR: bitmap_shipmode or bitmap_shipinstr not loaded.\n";
        return;
    }

    struct GroupSpec { int group; int64_t qty_lo_raw; int64_t qty_hi_raw; };

    std::vector<double> tA_t, tB_t, tC_t, tot_t;
    int64_t last_revenue_raw = 0;

    for (int iter = 0; iter < Q19_ITERATIONS; iter++) {
        bool warm = iter < Q19_WARMUP;
        std::cout << "\n--- Iteration " << iter+1 << "/" << Q19_ITERATIONS
                  << (warm ? " (warm-up)" : "") << " ---" << std::endl;

        auto t0 = clk::now();
        auto& lineitem_tx = DuckTransaction::Get(context.client, lineitem_table.catalog);

        // ===== Phase A: shipmode AND shipinstruct → row_ids =====
        std::vector<row_t> ids;
        size_t num_rows = idx_sm->num_rows();

        if (auto* cb_sm = dynamic_cast<bm_index::IndexedComBit*>(idx_sm)) {
            auto* cb_si = dynamic_cast<bm_index::IndexedComBit*>(idx_si);
            if (!cb_si) { std::cerr << "[Q19] CB type mismatch.\n"; return; }
            ComBit bm_sm = ComBit::from_sparse_positions({}, num_rows, cb_sm->segment_bits());
            ComBit bm_si = ComBit::from_sparse_positions({}, num_rows, cb_si->segment_bits());
            cb_sm->apply_or_to(bm_sm, 'A');     // 'AIR' or 'AIR REG'
            cb_si->apply_or_to(bm_si, 'D');     // 'DELIVER IN PERSON'
            bm_sm &= bm_si;
            q19_get_rowids(bm_sm, &ids);
        } else if (auto* cr_sm = dynamic_cast<bm_index::IndexedCRoaring*>(idx_sm)) {
            auto* cr_si = dynamic_cast<bm_index::IndexedCRoaring*>(idx_si);
            if (!cr_si) { std::cerr << "[Q19] CR type mismatch.\n"; return; }
            roaring::Roaring bm_sm, bm_si;
            cr_sm->apply_or_to(bm_sm, 'A');
            cr_si->apply_or_to(bm_si, 'D');
            bm_sm &= bm_si;
            q19_get_rowids(bm_sm, &ids);
        } else if (auto* wah_sm = dynamic_cast<bm_index::IndexedWAH*>(idx_sm)) {
            auto* wah_si = dynamic_cast<bm_index::IndexedWAH*>(idx_si);
            if (!wah_si) { std::cerr << "[Q19] WAH type mismatch.\n"; return; }
            ibis::bitvector bm_sm, bm_si;
            wah_sm->apply_or_to(bm_sm, 'A');
            wah_si->apply_or_to(bm_si, 'D');
            bm_sm &= bm_si;
            q19_get_rowids(bm_sm, &ids);
        } else if (auto* ew_sm = dynamic_cast<bm_index::IndexedEWAH*>(idx_sm)) {
            auto* ew_si = dynamic_cast<bm_index::IndexedEWAH*>(idx_si);
            if (!ew_si) { std::cerr << "[Q19] EW type mismatch.\n"; return; }
            ewah::EWAHBoolArray<uint64_t> bm_sm, bm_si;
            ew_sm->apply_or_to(bm_sm, 'A');
            ew_si->apply_or_to(bm_si, 'D');
            ewah::EWAHBoolArray<uint64_t> tmp;
            bm_sm.logicaland(bm_si, tmp);
            q19_get_rowids(tmp, &ids);
        } else if (auto* con_sm = dynamic_cast<bm_index::IndexedConcise*>(idx_sm)) {
            auto* con_si = dynamic_cast<bm_index::IndexedConcise*>(idx_si);
            if (!con_si) { std::cerr << "[Q19] CON type mismatch.\n"; return; }
            ConciseSet<false> bm_sm, bm_si;
            con_sm->apply_or_to(bm_sm, 'A');
            con_si->apply_or_to(bm_si, 'D');
            ConciseSet<false> tmp = bm_sm.logicaland(bm_si);
            q19_get_rowids(tmp, &ids);
        } else {
            std::cerr << "[Q19] ERROR: unrecognised IBitmapIndex backend.\n";
            return;
        }
        auto t_a = clk::now();

        // ===== Phase B: part scan → partkey_to_group =====
        std::unordered_map<int64_t, GroupSpec> pk_to_group;
        pk_to_group.reserve(20000);
        {
            auto& tx = DuckTransaction::Get(context.client, part_table.catalog);
            TableScanState ss;
            // p_partkey, p_brand, p_size, p_container
            vector<StorageIndex> col_ids = {StorageIndex(0), StorageIndex(3),
                                            StorageIndex(5), StorageIndex(6)};
            part_table.GetStorage().InitializeScan(context.client, tx, ss, col_ids);
            vector<LogicalType> types;
            for (int c : {0, 3, 5, 6}) types.push_back(part_table.GetColumns().GetColumnTypes()[c]);
            auto match_one = [](const string_t& s, const char* opt) {
                size_t l = std::strlen(opt);
                return s.GetSize() == l && std::memcmp(s.GetData(), opt, l) == 0;
            };
            while (true) {
                DataChunk chunk; chunk.Initialize(context.client, types);
                part_table.GetStorage().Scan(tx, chunk, ss);
                if (chunk.size() == 0) break;
                chunk.data[1].Flatten(chunk.size());
                chunk.data[3].Flatten(chunk.size());
                auto pk = FlatVector::GetData<int64_t>(chunk.data[0]);
                auto pb = FlatVector::GetData<string_t>(chunk.data[1]);
                auto sz = FlatVector::GetData<int32_t>(chunk.data[2]);
                auto pc = FlatVector::GetData<string_t>(chunk.data[3]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    if (pb[i].GetSize() < 8) continue;
                    const char* brand = pb[i].GetData();
                    int g = 0; int64_t qlo = 0, qhi = 0;
                    if (std::memcmp(brand, "Brand#12", 8) == 0 && sz[i] >= 1 && sz[i] <= 5) {
                        if (match_one(pc[i], "SM CASE") || match_one(pc[i], "SM BOX") ||
                            match_one(pc[i], "SM PACK") || match_one(pc[i], "SM PKG"))
                            { g = 1; qlo = 100;  qhi = 1100; }
                    } else if (std::memcmp(brand, "Brand#23", 8) == 0 && sz[i] >= 1 && sz[i] <= 10) {
                        if (match_one(pc[i], "MED BAG") || match_one(pc[i], "MED BOX") ||
                            match_one(pc[i], "MED PKG") || match_one(pc[i], "MED PACK"))
                            { g = 2; qlo = 1000; qhi = 2000; }
                    } else if (std::memcmp(brand, "Brand#34", 8) == 0 && sz[i] >= 1 && sz[i] <= 15) {
                        if (match_one(pc[i], "LG CASE") || match_one(pc[i], "LG BOX") ||
                            match_one(pc[i], "LG PACK") || match_one(pc[i], "LG PKG"))
                            { g = 3; qlo = 2000; qhi = 3000; }
                    }
                    if (g) pk_to_group[pk[i]] = {g, qlo, qhi};
                }
            }
        }
        auto t_b = clk::now();

        // ===== Phase C: BMFetch + per-row predicate eval + revenue agg =====
        int64_t revenue_raw = 0;
        if (!ids.empty()) {
            vector<StorageIndex> col_ids = {StorageIndex(1), StorageIndex(4),
                                            StorageIndex(5), StorageIndex(6)};
            vector<LogicalType> types;
            for (int c : {1, 4, 5, 6}) types.push_back(lineitem_table.GetColumns().GetColumnTypes()[c]);
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
                auto dc = FlatVector::GetData<int64_t>(chunk.data[3]);
                for (idx_t i = 0; i < chunk.size(); i++) {
                    auto it = pk_to_group.find(pk[i]);
                    if (it == pk_to_group.end()) continue;
                    auto& g = it->second;
                    if (qt[i] < g.qty_lo_raw || qt[i] > g.qty_hi_raw) continue;
                    revenue_raw += pr[i] * (100 - dc[i]);
                }
            }
        }
        auto t_c = clk::now();

        double tA  = q19_ms(t0, t_a);
        double tB  = q19_ms(t_a, t_b);
        double tC  = q19_ms(t_b, t_c);
        double tot = q19_ms(t0, t_c);
        std::cout << "  " << idx_sm->backend_name()
                  << ":  PhaseA(shipmode AND ship_instr)=" << tA
                  << "  PhaseB(part_scan)=" << tB
                  << "  PhaseC(BMFetch+agg)=" << tC
                  << "  Total=" << tot
                  << "  rows=" << ids.size()
                  << "  revenue=" << std::fixed << std::setprecision(4) << double(revenue_raw) / 10000
                  << std::endl;
        if (!warm) {
            tA_t.push_back(tA); tB_t.push_back(tB); tC_t.push_back(tC); tot_t.push_back(tot);
        }
        last_revenue_raw = revenue_raw;
    }

    {
        Connection con(*context.client.db);
        auto r = con.Query(
            "SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue "
            "FROM lineitem, part "
            "WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12' "
            "       AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG') "
            "       AND l_quantity >= 1 AND l_quantity <= 11 "
            "       AND p_size BETWEEN 1 AND 5 "
            "       AND l_shipmode IN ('AIR','AIR REG') "
            "       AND l_shipinstruct = 'DELIVER IN PERSON') "
            "   OR (p_partkey = l_partkey AND p_brand = 'Brand#23' "
            "       AND p_container IN ('MED BAG','MED BOX','MED PKG','MED PACK') "
            "       AND l_quantity >= 10 AND l_quantity <= 20 "
            "       AND p_size BETWEEN 1 AND 10 "
            "       AND l_shipmode IN ('AIR','AIR REG') "
            "       AND l_shipinstruct = 'DELIVER IN PERSON') "
            "   OR (p_partkey = l_partkey AND p_brand = 'Brand#34' "
            "       AND p_container IN ('LG CASE','LG BOX','LG PACK','LG PKG') "
            "       AND l_quantity >= 20 AND l_quantity <= 30 "
            "       AND p_size BETWEEN 1 AND 15 "
            "       AND l_shipmode IN ('AIR','AIR REG') "
            "       AND l_shipinstruct = 'DELIVER IN PERSON')");
        if (r && !r->HasError() && r->RowCount() == 1) {
            double gt = r->GetValue(0, 0).GetValue<double>();
            double our = double(last_revenue_raw) / 10000;
            if (std::fabs(gt - our) < 0.5)
                std::cout << "\n[OK] " << idx_sm->backend_name()
                          << " matches DuckDB SQL (revenue = " << gt << ").\n";
            else
                std::cerr << "\n[FAIL] mismatch: ours=" << our << " gt=" << gt << "\n";
        }
    }

    auto stats = [](std::vector<double>& v) {
        if (v.empty()) return bm_bench::Stats{0,0,0,0};
        return bm_bench::compute_stats(v);
    };
    auto sA = stats(tA_t), sB = stats(tB_t), sC = stats(tC_t), sT = stats(tot_t);
    int measured = std::max(0, Q19_ITERATIONS - Q19_WARMUP);

    std::cout << "\n================================================================\n";
    std::cout << "  Q19 RESULTS — " << idx_sm->backend_name()
              << " (" << measured << " measured iter, median +/- stddev)\n";
    std::cout << "================================================================\n";
    std::cout << "  PhaseA (shipmode AND shipinstruct): " << sA.median << " +/- " << sA.stddev << " ms\n";
    std::cout << "  PhaseB (part scan + group map)    : " << sB.median << " +/- " << sB.stddev << " ms\n";
    std::cout << "  PhaseC (BMFetch + per-row pred)   : " << sC.median << " +/- " << sC.stddev << " ms\n";
    std::cout << "  TOTAL                             : " << sT.median << " +/- " << sT.stddev << " ms\n";
    std::cout << "================================================================\n\n";

    std::string sf = bm_bench::sf_label();
    std::ofstream csv("q19_results_" + sf + ".csv");
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
        std::string bn = idx_sm->backend_name();
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
        put("PhaseA_ship",  sA);
        put("PhaseB_part",  sB);
        put("PhaseC_agg",   sC);
        put("TOTAL", sT);
        std::cout << "  [CSV] q19_results_" << sf << ".csv\n";
    }

    });  // end call_once
}

}  // namespace duckdb
