#!/bin/bash
# Run after the main bench (--only 1 3 4 5 6 14 17 19) completes.
# 1. Bench Q15 (legacy DEBIT_BM=all path; it's TPC-H 1.5.7 compliant
#    via bitmap_shipdate single-column).
# 2. Bench Q12 (legacy path; uses non-compliant commit_lt_receipt /
#    ship_lt_commit aux structures — will be flagged in the README).
# 3. Rebuild bm_resultL4.xlsx with --skip-run, including all CSVs.
# Q8 and Q10 are skipped — their pre-built multi-table aux structures
# were retired pre-port and the bitmap directories are empty.
set -e
cd /home/lichenhang/lee/thesis/check/lee/duckdb-dev

echo "=== [finalize] Q15 (legacy compliant) ==="
python3 tools/bench_suite.py --sf 10 --only 15 \
    --log-dir bm_logs_resultL4_full \
    --out-csv bm_results_long_resultL4.csv \
    --out-xlsx bm_resultL4.xlsx

echo "=== [finalize] Q12 (legacy, non-compliant aux — flagged in README) ==="
python3 tools/bench_suite.py --sf 10 --only 12 \
    --log-dir bm_logs_resultL4_full \
    --out-csv bm_results_long_resultL4.csv \
    --out-xlsx bm_resultL4.xlsx

echo "=== [finalize] Rebuild Excel with all CSVs ==="
python3 tools/bench_suite.py --sf 10 --skip-run \
    --only 1 3 4 5 6 12 14 15 17 19 \
    --log-dir bm_logs_resultL4_full \
    --out-csv bm_results_long_resultL4.csv \
    --out-xlsx bm_resultL4.xlsx

echo "=== [finalize] Done. bm_resultL4.xlsx is ready. ==="
ls -la bm_resultL4.xlsx
