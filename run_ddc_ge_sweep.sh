#!/usr/bin/env bash
# Run all 12 TPC-H Qs with DEBIT_BM=cb_ge.
# Only Q3 materially benefits (uses shipdate_GE for full years).
# Other Qs that load shipdate (Q1/Q14/Q15) pay the +26 MB but don't use it.
set -u

LOG_DIR=bm_logs_ddc_ge
mkdir -p "$LOG_DIR"

declare -A LOADS=(
  [1]="shipdate linestatus returnflag"
  [3]="orderkey shipdate"
  [4]="orderkey"
  [5]="orderkey suppkey"
  [6]="shipdate_GE discount quantity"
  [8]="orderkey partkey"
  [10]="orderkey returnflag"
  [12]="shipmode receiptdate"
  [14]="shipdate"
  [15]="shipdate"
  [17]="partkey"
  [19]="shipmode shipinstruct"
)

ITER=${DEBIT_ITER:-5}
WARM=${DEBIT_WARMUP:-1}
ORDER="1 3 4 5 6 8 10 12 14 15 17 19"

echo "============================================================"
echo " DDC+GE TPC-H sweep  (DEBIT_BM=cb_ge, iter=$ITER, warmup=$WARM)"
echo " duckdb: $(stat -c '%y' build/release/duckdb)"
echo " log dir: $LOG_DIR"
echo "============================================================"

for q in $ORDER; do
    cols="${LOADS[$q]}"
    sql=/tmp/q${q}_cbge.sql
    : > "$sql"
    for c in $cols; do echo "PRAGMA load_bitmap('$c');" >> "$sql"; done
    echo "PRAGMA bm_tpch($q);" >> "$sql"
    log="$LOG_DIR/q${q}_SF10.log"
    echo
    echo "[Q$q] cols=($cols) -> $log"
    t0=$(date +%s)
    DEBIT_BM=cb_ge DEBIT_ITER=$ITER DEBIT_WARMUP=$WARM TPCH_SF=10 \
        build/release/duckdb tpch_sf10.db < "$sql" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ $rc -ne 0 ]; then
        echo "  [ERROR] duckdb exit=$rc ($(($t1-$t0))s)"
    else
        echo "  [done $(($t1-$t0))s]  $(grep -E '^\[OK\]|^\[FAIL\]' "$log" | tail -1)"
        echo "  median: $(grep -E 'TOTAL\s+:' "$log" | tail -1 | sed -E 's/^[^:]*://')"
    fi
done

echo
echo "Done. Logs in: $LOG_DIR/"
