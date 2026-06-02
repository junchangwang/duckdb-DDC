#!/usr/bin/env bash
# Run all TPC-H Qs with the freshly-synced DDC core.
# DEBIT_BM=cb selects DDC. Each Q is a separate duckdb invocation
# (so once_flag for that Q fires fresh).
set -u

LOG_DIR=bm_logs_ddc_sync
mkdir -p "$LOG_DIR"

# col-load specs per Q
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
echo " DDC TPC-H sweep  (DEBIT_BM=cb, iter=$ITER, warmup=$WARM)"
echo " duckdb binary: $(stat -c '%y' build/release/duckdb)"
echo " log dir: $LOG_DIR"
echo "============================================================"

for q in $ORDER; do
    cols="${LOADS[$q]}"
    sql=/tmp/q${q}_cb.sql
    : > "$sql"
    for c in $cols; do echo "PRAGMA load_bitmap('$c');" >> "$sql"; done
    echo "PRAGMA bm_tpch($q);" >> "$sql"

    log="$LOG_DIR/q${q}_SF10.log"
    echo
    echo "[Q$q] cols=($cols) -> $log"
    t0=$(date +%s)
    DEBIT_BM=cb DEBIT_ITER=$ITER DEBIT_WARMUP=$WARM TPCH_SF=10 \
        build/release/duckdb tpch_sf10.db < "$sql" \
        > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    dur=$((t1-t0))

    # extract the key lines
    medline=$(grep -E "TOTAL\s+:|Total=" "$log" | tail -3)
    okline=$(grep -E "^\[OK\]|^\[FAIL\]" "$log" | tail -1)

    if [ $rc -ne 0 ]; then
        echo "  [ERROR] duckdb exit=$rc (${dur}s)"
    else
        echo "  [done ${dur}s]  $okline"
        # show median total
        med=$(grep -E "TOTAL\s+:" "$log" | tail -1 | sed -E 's/^[^:]*://')
        [ -n "$med" ] && echo "  median: $med"
    fi
done

echo
echo "============================================================"
echo " Done. Logs in: $LOG_DIR/"
echo "============================================================"
