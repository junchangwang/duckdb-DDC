#### About this project

This repository contains the TPC-H benchmark on DuckDB using bitmap indexes compressed using different mechanisms including our DDC, WAH, EWAH, CONCISE, and CRoaring.

#### How is this project organized?

The core of this project is implemented as a standard DuckDB extension `extension/debit`, so it tracks DuckDB's rapid evolution.
The bitmap-indexing-based implementation of each TPC-H query can be found in `extension/debit/execution/tpch/query/Q*.cpp`.
For example, `Q5.cpp` shows how our system converts a fact-dimension join into a predicate on the fact table's foreign-key column and then evaluates the query by leveraging bitmaps on that column.

The compression mechanism is selected at run time via the `DEBIT_BM` environment variable (`ddc` for DDC, and `wah`, `cr`, `crr`, `ew`, `con`, `bs` for the baselines), so every scheme runs through the same operators on the same inputs.

#### How to run?

The build needs C++17 and a server with **AVX-512** (AVX-512F, BW, VBMI2). The DDC core compiles together with the extension, so there is no separate index build step.

1) Build DuckDB with the extension.

```sh
make release
```

The DuckDB binary is produced at `build/release/duckdb`.

2) Generate the TPC-H dataset at SF 10.

```DuckDB
load tpch;
CALL dbgen(sf = 10);
```

3) Build the bitmap indexes and run a query. `load_bitmap` automatically builds the bitmap for each listed column in memory every time it runs, so there are no bitmap files to generate or download in advance. Use `DEBIT_BM=ddc` to select DDC.

```sh
DEBIT_BM=ddc TPCH_SF=10 build/release/duckdb tpch_sf10.db
```
```DuckDB
set threads to 1;
pragma load_bitmap('shipdate', 'linestatus', 'returnflag');
pragma bm_tpch(1);
```

Here `load_bitmap` builds the three bitmaps that query 1 needs, and `bm_tpch(1)` runs query 1 on them. The bitmaps each TPC-H query needs are listed below.

- **Q1** shipdate, linestatus, returnflag
- **Q3** orderkey, shipdate
- **Q4** orderkey
- **Q5** orderkey, suppkey
- **Q6** shipdate_GE, discount, quantity
- **Q8** orderkey, partkey
- **Q10** orderkey, returnflag
- **Q12** shipmode, receiptdate
- **Q14** shipdate
- **Q15** shipdate
- **Q17** partkey
- **Q19** shipmode, shipinstruct

To run the query with DuckDB's native operators instead of the bitmap-indexed plan, use the following.

```DuckDB
pragma tpch(1);
```

#### Reference

DDC: Decoupling Data and Control for Bitmap Compression. (under submission).
