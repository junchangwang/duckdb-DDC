#!/usr/bin/env bash
# Streaming-Q15 sweep: BitEngine-paper-compliant (returns chunks to
# DuckDB upstream; DuckDB does GROUP BY l_suppkey + MAX + JOIN supplier).
set -u
LOG_DIR=bm_logs_fix_verify
mkdir -p "$LOG_DIR"
ITERS=6
HARNESS_SQL_PREFIX="PRAGMA threads=1;
.timer on
"
declare -A LOADS_DAY=( [15]="shipdate" )
declare -A LOADS_GE=(  [15]="suppkey shipdate_GE_month" )
PASSES=( "day cb" "day cr" "day crr" "day wah" "day ew" "day con"
         "ge cb_ge" "ge cr" "ge crr" "ge wah" "ge ew" "ge con" )
SUMMARY="$LOG_DIR/SUMMARY_q15_streaming.csv"
echo "q,pass,backend,median_ms,iters,status" > "$SUMMARY"

for spec in "${PASSES[@]}"; do
    pass=$(echo "$spec" | awk '{print $1}')
    bm=$(  echo "$spec" | awk '{print $2}')
    [ "$pass" = "day" ] && cols="${LOADS_DAY[15]}" || cols="${LOADS_GE[15]}"
    sql=/tmp/q15_streaming_${pass}_${bm}.sql
    : > "$sql"
    echo "$HARNESS_SQL_PREFIX" >> "$sql"
    for c in $cols; do echo "PRAGMA load_bitmap('$c');" >> "$sql"; done
    for i in $(seq 1 $ITERS); do echo "PRAGMA bm_tpch(15);" >> "$sql"; done
    log="$LOG_DIR/q15_streaming_${pass}_${bm}.log"
    echo
    echo "[Q15 streaming $pass $bm] -> $log"
    DEBIT_BM=$bm TPCH_SF=10 build/release/duckdb tpch_sf10.db < "$sql" > "$log" 2>&1
    rc=$?
    # Skip load_bitmap timings: take only Run Time lines AFTER the last load_bitmap line.
    # Simpler: count total Run Time lines, skip first few that come from load_bitmap PRAGMAs.
    # The load_bitmap pragmas produce a Run Time line each (since we use .timer on).
    n_loads=$(echo "$cols" | wc -w)
    times=$(grep -oE "Run Time \(s\): real [0-9.]+" "$log" | awk '{print $5}')
    times_arr=($times)
    n=${#times_arr[@]}
    # Drop first $n_loads (load_bitmap) AND first measured iter (warmup) = $((n_loads+1)) total to drop
    drop=$((n_loads + 1))
    if [ "$n" -gt "$drop" ]; then
        measured=("${times_arr[@]:$drop}")
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
        echo "15,$pass,$bm,$median_ms,$m,[OK]" >> "$SUMMARY"
        echo "  median ($m measured) = $median_ms ms ; raw times: ${measured[*]}"
    else
        echo "15,$pass,$bm,-,$n,ERR_no_timing" >> "$SUMMARY"
        echo "  [ERR] only $n timer lines (need >$drop)"
    fi
done
echo "Done. Summary: $SUMMARY"
