#!/usr/bin/env bash
# Streaming-Q19 sweep: PRAGMA bm_tpch(19) is now BitEngine-paper-compliant
# (returns chunks to DuckDB upstream; DuckDB does hash join + 3-OR filter
# + SUM agg).  We measure end-to-end query latency via DuckDB CLI .timer.
#
# Each backend × pass cell: 1 warmup + 5 measured iterations.  Median of
# the 5 measured runs is recorded into SUMMARY_q19_streaming.csv.
set -u

LOG_DIR=bm_logs_fix_verify
mkdir -p "$LOG_DIR"

# 1 warmup + 5 measured = 6 PRAGMA bm_tpch(19) calls per run.
ITERS=6

# Always force single-thread.  Parallel TableScan would race the
# row_ids state reset across workers; single-thread keeps the
# bench deterministic (paper Q19 numbers are also single-thread).
HARNESS_SQL_PREFIX="PRAGMA threads=1;
.timer on
"

declare -A LOADS_DAY=( [19]="shipmode shipinstruct" )
declare -A LOADS_GE=(  [19]="shipmode shipinstruct" )

PASSES=( "day cb"      "day cr"      "day crr"     "day wah"     "day ew"      "day con"
         "ge  cb_ge"   "ge  cr"      "ge  crr"     "ge  wah"     "ge  ew"      "ge  con" )

SUMMARY="$LOG_DIR/SUMMARY_q19_streaming.csv"
echo "q,pass,backend,median_ms,iters,status" > "$SUMMARY"

for spec in "${PASSES[@]}"; do
    pass=$(echo "$spec" | awk '{print $1}')
    bm=$(  echo "$spec" | awk '{print $2}')
    [ "$pass" = "day" ] && cols="${LOADS_DAY[19]}" || cols="${LOADS_GE[19]}"

    sql=/tmp/q19_streaming_${pass}_${bm}.sql
    : > "$sql"
    echo "$HARNESS_SQL_PREFIX" >> "$sql"
    for c in $cols; do echo "PRAGMA load_bitmap('$c');" >> "$sql"; done
    # 6 calls: iter 0 = warmup, iter 1..5 = measured
    for i in $(seq 1 $ITERS); do echo "PRAGMA bm_tpch(19);" >> "$sql"; done

    log="$LOG_DIR/q19_streaming_${pass}_${bm}.log"
    echo
    echo "[Q19 streaming $pass $bm] -> $log"
    t0=$(date +%s)
    DEBIT_BM=$bm TPCH_SF=10 build/release/duckdb tpch_sf10.db < "$sql" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)

    # Pull every "Run Time (s): real X.YZ" — 6 of them.  Drop first
    # (warmup) and compute median of the remaining 5.
    times=$(grep -oE "Run Time \(s\): real [0-9.]+" "$log" | awk '{print $5}')
    times_arr=($times)
    n=${#times_arr[@]}
    if [ "$n" -ge 2 ]; then
        # Drop first (warmup), sort, pick middle
        measured=("${times_arr[@]:1}")
        sorted=($(printf "%s\n" "${measured[@]}" | sort -n))
        m=${#sorted[@]}
        mid=$((m/2))
        if [ $((m % 2)) -eq 0 ]; then
            a=${sorted[$((mid-1))]}; b=${sorted[$mid]}
            median_s=$(awk -v a="$a" -v b="$b" 'BEGIN { printf "%.4f", (a+b)/2 }')
        else
            median_s=${sorted[$mid]}
        fi
        median_ms=$(awk -v s="$median_s" 'BEGIN { printf "%.4f", s*1000 }')
        echo "19,$pass,$bm,$median_ms,$n,[OK]" >> "$SUMMARY"
        echo "  median (skip warmup, $((n-1)) measured) = $median_ms ms (${times_arr[*]:1})"
    else
        echo "19,$pass,$bm,-,$n,ERR_no_timing" >> "$SUMMARY"
        echo "  [ERR] only $n timer lines"
    fi
done

echo
echo "Done. Summary: $SUMMARY"
