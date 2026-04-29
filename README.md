
#### About BitEngine project

The database community has long explored bitmap indexing in DBMSs, but most efforts failed to achieve widespread adoption. The main reason is that traditional bitmap indexes limit to scans on low-cardinality attributes on read-only workloads. A notable example is PostgreSQL, which introduced native bitmap indexes in version 8.3.23 but later removed them due to these limitations.

In recent years, researchers have proposed innovative bitmap index designs. Recent advances include update-friendly designs such as UpBit [1] and Cubit [2], with RABIT [3] extending support to high-cardinality attributes. For the first time, researchers can build bitmap indexes on attributes of any cardinality (from dozens to millions) in tables ranging from read-only to update-intensive [2,3]. These advances have sparked interest in bitmap indexing, raising an interesting question: **beyond traditionally acting as scan accelerators, to what extent can the state-of-the-art bitmap indexes be leveraged in DBMS query engines**?

To answer this, we develop BitEngine, a bitmap index-based query engine in DuckDB. At its core is a suite of bitmap-oriented operators, delivering benefits such as reduced I/O and intermediate data, improved SIMD utilization, and avoidance of costly materializations.

[3] Junchang Wang, Fu Xiao, Manos Athanassoulis. RABIT: Efficient Range Queries with Bitmap Indexing. In SIGMOD'25.
[2] Junchang Wang, Manos Athanassoulis. CUBIT: Concurrent Updatable Bitmap Indexing. In VLDB'24.
[1] Manos Athanassoulis, Zheng Yan, Stratos Idreos. UpBit: Scalable In-Memory Updatable Bitmap Indexing. In SIGMOD'16.



#### How is this project organized?

BitEngine is currently implemented as a DuckDB extension to ensure portability along with DuckDB's rapid evolution. Its source code is located in the `extension/debit` directory. We primarily evaluated BitEngine using the TPC-H benchmark, both in the project and in the paper. The streamlined implementation of TPCH queries using BitEngine can be found in the `extension/debit/execution/tpch` directory. For example, the details of how BitEngine transforms a join into a predicate on foreign key columns and how the predicate is evaluated by using bitmaps for Q5 can be found in `extension/debit/execution/tpch/query/q5.cpp`, which corresponds to the logic described in the paper. (For implementations of independent operators supporting arbitrary queries, see the BitQ branch.)



#### How to run BitEngine?

BitEngine is built on C++17, Python 3, and the liburcu development library. The testbed server must support AVX-512 instructions. Most modern processors from Intel and AMD are compatible; however, if you are using a laptop, please verify that your processor supports AVX-512.

1) Please compile the bitmap index used in the project.

```sh
cd extension/debit/CUBIT-d
./build.sh
```

2) Compile DuckDB as usual. Generate a TPCH dataset with SF 10 using the following commands.

```DuckDB
load tpch;
CALL dbgen(sf = 10);
```

3) Generate bitmap instances for the TPCH dataset on your side (see CUBIT project for details). However, generating bitmap instances for the whole TPCH dataset takes hours, such that we strongly suggest downloading the required pre-generated bitmap instances from the following github repository.

```sh
git clone --branch BITMAPS --single-branch https://github.com/junchangwang/Bitmap-dataset.git
cd Bitmap-dataset
python3 decompress_tpch.py
mv BITMAPS/* /path/to/BitEngine/
```

4) Load the corresponding bitmap instances in DuckDB before you execute the TPCH queries. 

```DuckDB
set threads to 1;
pragma load_bitmap(col_name1, col_name2);
```

Below are the required bitmap indexes for each supported TPCH query in DEBIT:

- Q1: shipdate, linestatus, returnflag
- Q5: orderkey, suppkey
- Q6: shipdate_GE, discount, quantity

For example, if you want to run Q1, please use the following command in DuckDB prompt:

```DuckDB
pragma load_bitmap(shipdate, linestatus, returnflag);
pragma bm_tpch(1);
```

You can also run TPC-H queries using the standard operators in DuckDB for comparison. For example, to run Q1 using the standard operators, you can use the following command:

```DuckDB
pragma tpch(1);
```



#### Acknowledgements

If you have any questions about the code, please feel free to contact us.