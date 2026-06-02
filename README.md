#### About this project

Bitmap indexing is increasingly pushed into the core of analytical DBMSs, where
a query is evaluated by combining precomputed bitvectors with bitwise
operations (AND/OR/NOT). On analytical workloads such as TPC-H, however, most
attributes have moderate cardinality, and — because range and group encoding
make their bitvectors moderate-to-dense — the bitwise operations over the
*compressed* bitvectors become the critical bottleneck: RLE-based schemes (WAH,
EWAH, CONCISE) collapse into a branch-heavy serial merge, and hybrid schemes
(Roaring) pay container-conversion overhead. A bitmap-indexed plan can end up
slower than a native columnar scan.

This repository integrates **DDC** (Decoupling Data and Control) — a bitvector
compression mechanism whose decoupled, SIMD-native format sustains
density-independent bitwise throughput — as the compression layer underneath a
suite of bitmap-oriented TPC-H operators in DuckDB. By swapping only the
underlying compression mechanism under a shared operator harness, the project
shows that DDC removes the bitwise-operation bottleneck and lets bitmap-indexed
query plans deliver end-to-end speedups across the full density range, where
prior schemes fluctuate or regress.

#### How is this project organized?

The integration is a DuckDB extension under `extension/debit`, so it tracks
DuckDB's rapid evolution. The bitmap-oriented TPC-H queries are in
`extension/debit/execution/tpch/query/Q*.cpp` — e.g. `Q5.cpp` shows how a
fact–dimension join is rewritten as a predicate on the fact table's foreign-key
column and then evaluated with bitvector merges, matching the paper. The DDC
compression library itself is vendored at `extension/debit/ddc` (and developed
as the standalone DDC-core project). The compression mechanism is selected at
run time via the `DEBIT_BM` environment variable (`cb` = DDC; `wah`, `cr`,
`crr`, `ew`, `con`, `bs` = the baselines), so every scheme runs through the same
operators on the same inputs.

#### How to run?

The build needs C++17 and a server with **AVX-512** (AVX-512F/BW/VBMI2 —
verify your CPU, especially on laptops). The DDC core compiles together with the
extension, so no separate index build step is required.

1) Build DuckDB with the extension:

```sh
make release          # -> build/release/duckdb
```

2) Generate the TPC-H dataset at SF 10:

```DuckDB
load tpch;
CALL dbgen(sf = 10);
```

3) Build the bitmap indexes and run a query. `load_bitmap` scans the column and
builds the DDC index **in memory** (no pre-generated bitmap files to download).
Run with `DEBIT_BM=cb` to select DDC:

```sh
DEBIT_BM=cb TPCH_SF=10 build/release/duckdb tpch_sf10.db
```
```DuckDB
set threads to 1;
pragma load_bitmap('shipdate', 'linestatus', 'returnflag');
pragma bm_tpch(1);
```

The bitmaps each TPC-H query needs:

- Q1: shipdate, linestatus, returnflag        Q3: orderkey, shipdate
- Q4: orderkey                                 Q5: orderkey, suppkey
- Q6: shipdate_GE, discount, quantity          Q8: orderkey, partkey
- Q10: orderkey, returnflag                    Q12: shipmode, receiptdate
- Q14: shipdate                                Q15: shipdate
- Q17: partkey                                 Q19: shipmode, shipinstruct

To compare against a baseline, set `DEBIT_BM` accordingly (e.g. `DEBIT_BM=wah`,
`DEBIT_BM=cr`). To run the query with DuckDB's native operators instead of the
bitmap-indexed plan:

```DuckDB
pragma tpch(1);
```

The synthetic micro-benchmarks for the DDC compression mechanism itself
(throughput/size vs. density, hierarchy depth, segment size, bypass) live in the
standalone DDC project.

#### Reference

DDC: Decoupling Data and Control for Bitmap Compression. *PVLDB* (under
submission).
