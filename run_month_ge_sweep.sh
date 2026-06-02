#!/usr/bin/env bash
# Run all 12 TPC-H Qs with the chosen backend, using month-level GE
# (`shipdate_GE_month`, 84 keys) for Q14/Q15, year-level GE for Q6,
# and per-day shipdate for Q1/Q3 (whose filters don't align to month
# boundaries — Q3 picks up auto-built year-GE under cb_ge backend).
#
# Usage:
#   ./run_month_ge_sweep.sh cb_ge          # DDC + GE
#   ./run_month_ge_sweep.sh cr             # CRoaring (gets month-GE via load_bitmap)
#   ./run_month_ge_sweep.sh crr            # CRoaringRun
#
# Records BOTH memory (per loaded bitmap) and runtime (median TOTAL ms).
set -u

BACKEND=${1:-cb_ge}
LOG_DIR=bm_logs_month_ge_${BACKEND}
mkdir -p "$LOG_DIR"

declare -A LOADS=(
  [1]="shipdate_GE_month linestatus returnflag"
  [3]="orderkey shipdate_GE_month"
  [4]="orderkey"
  [5]="orderkey suppkey"
  [6]="shipdate_GE discount quantity"
  [8]="orderkey partkey"
  [10]="orderkey returnflag"
  [12]="shipmode receiptdate_GE"
  [14]="shipdate_GE_month"
  [15]="suppkey shipdate_GE_month"
  [17]="partkey"
  [19]="shipmode shipinstruct"
)

ITER=${DEBIT_ITER:-5}
WARM=${DEBIT_WARMUP:-1}
ORDER="1 3 4 5 6 8 10 12 14 15 17 19"

echo "============================================================"
echo " MonthGE TPC-H sweep  (DEBIT_BM=$BACKEND, iter=$ITER, warmup=$WARM)"
echo " duckdb: $(stat -c '%y' build/release/duckdb)"
echo " log dir: $LOG_DIR"
echo "============================================================"

SUMMARY="$LOG_DIR/SUMMARY.txt"
: > "$SUMMARY"
printf "%-4s %-10s %-12s %-50s\n" "Q" "status" "median_ms" "memory_per_column_MB" | tee -a "$SUMMARY"
printf -- "------------------------------------------------------------------------------------\n" | tee -a "$SUMMARY"

for q in $ORDER; do
    cols="${LOADS[$q]}"
    sql=/tmp/q${q}_${BACKEND}_mge.sql
    : > "$sql"
    for c in $cols; do echo "PRAGMA load_bitmap('$c');" >> "$sql"; done
    echo "PRAGMA bm_tpch($q);" >> "$sql"
    log="$LOG_DIR/q${q}.log"
    echo
    echo "[Q$q] cols=($cols) -> $log"
    t0=$(date +%s)
    DEBIT_BM=$BACKEND DEBIT_ITER=$ITER DEBIT_WARMUP=$WARM TPCH_SF=10 \
        build/release/duckdb tpch_sf10.db < "$sql" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)

    # Status (last [OK]/[FAIL])
    status=$(grep -E '^\[OK\]|^\[FAIL\]' "$log" | tail -1 | grep -oE '^\[OK\]|^\[FAIL\]')
    [ -z "$status" ] && status="[NORES]"

    # Median total runtime.
    #   Q1/Q3/Q4/Q5/Q8/Q10/Q12/Q14/Q17/Q19: 'TOTAL ... : X.X +/- Y.Y'
    #   Q6: no summary block — take last per-iter 'total=X.X'
    #   Q15: per-iter 'Total=X.X' per backend; take last
    median=$(grep -E 'TOTAL\s*:' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -z "$median" ]; then
        median=$(grep -oE '[tT]otal=[0-9]+\.[0-9]+' "$log" | tail -1 | sed -E 's/[tT]otal=//')
    fi
    [ -z "$median" ] && median="-"

    # Memory: parse "[load_bitmap] X: built BACKEND index (... MB)" lines
    mem_str=""
    while IFS= read -r line; do
        col=$(echo "$line" | sed -E 's/^\[load_bitmap\] ([^:]+):.*/\1/')
        mb=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+ MB\)' | head -1 | sed 's/ MB)//')
        [ -n "$mb" ] && mem_str="${mem_str}${col}=${mb}MB "
    done < <(grep -E '^\[load_bitmap\].*built.*MB\)' "$log")

    # rc != 0 but [OK] in log = shutdown-time crash after results emitted
    # (seen on Q15 + cb_ge); treat as OK for the table.
    if [ $rc -ne 0 ] && [ "$status" != "[OK]" ]; then
        printf "%-4s %-10s %-12s %-50s\n" "Q$q" "[ERR rc=$rc]" "-" "-" | tee -a "$SUMMARY"
    else
        tag="$status"
        [ $rc -ne 0 ] && tag="${status}*"   # mark shutdown crash
        printf "%-4s %-10s %-12s %s\n" "Q$q" "$tag" "$median" "$mem_str" | tee -a "$SUMMARY"
    fi
done

echo
echo "Done. Logs in: $LOG_DIR/  Summary: $SUMMARY"
