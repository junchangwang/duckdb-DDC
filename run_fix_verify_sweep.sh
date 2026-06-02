#!/usr/bin/env bash
# Verify the post-fix latency of Q1/10/14/15/17/19 across all 5 bitmap
# backends, including the +GE variants (cb_ge / cr+month-GE).
# Two separate runs per "Q × backend" cell:
#   - per-day bitmaps  (cb / cr / crr / wah / ew / con)
#   - month-GE bitmaps (cb_ge / cr / crr / wah / ew / con loading
#                       shipdate_GE_month so each backend's class is
#                       built with month-bucketed keys, 84 cardinality)
# Q14 / Q15 / Q1 use the month-GE column when available (the loader
# auto-builds IndexedDDCGE for cb_ge, the others build IndexedX
# with month keys).  Q10 / Q17 / Q19 do not use a shipdate column,
# so the GE rerun for them is identical to per-day — we still emit
# the bar so the table is uniform across Qs.

set -u

LOG_DIR=bm_logs_fix_verify
mkdir -p "$LOG_DIR"

# Iter + warmup defaults (env-overridable).
ITER=${DEBIT_ITER:-5}
WARM=${DEBIT_WARMUP:-1}

# Per-Q (per-day) column loads — mirrors run_ddc_sweep.sh.
declare -A LOADS_DAY=(
  [1]="shipdate linestatus returnflag"
  [10]="orderkey returnflag"
  [14]="shipdate"
  [15]="shipdate"
  [17]="partkey"
  [19]="shipmode shipinstruct"
)

# Per-Q (month-GE) column loads — mirrors run_month_ge_sweep.sh.
# Q10/Q17/Q19 don't depend on shipdate, so they reuse the per-day loads.
declare -A LOADS_GE=(
  [1]="shipdate_GE_month linestatus returnflag"
  [10]="orderkey returnflag"
  [14]="shipdate_GE_month"
  [15]="suppkey shipdate_GE_month"
  [17]="partkey"
  [19]="shipmode shipinstruct"
)

ORDER="1 10 14 15 17 19"

# Two passes: (1) per-day path with DEBIT_BM ∈ {cb, cr, crr, wah, ew, con}
#             (2) month-GE path with DEBIT_BM ∈ {cb_ge, cr, crr, wah, ew, con}
PASSES=(
  "day cb"      "day cr"      "day crr"     "day wah"     "day ew"      "day con"
  "ge  cb_ge"   "ge  cr"      "ge  crr"     "ge  wah"     "ge  ew"      "ge  con"
)

SUMMARY="$LOG_DIR/SUMMARY.csv"
echo "q,pass,backend,median_ms,status" > "$SUMMARY"

echo "============================================================"
echo " Post-fix verification sweep — Q $ORDER × 12 backend configs"
echo " ITER=$ITER WARM=$WARM"
echo " duckdb: $(stat -c '%y' build/release/duckdb)"
echo " log dir: $LOG_DIR"
echo "============================================================"

for spec in "${PASSES[@]}"; do
    pass=$(echo "$spec" | awk '{print $1}')
    bm=$(  echo "$spec" | awk '{print $2}')

    for q in $ORDER; do
        if [ "$pass" = "day" ]; then
            cols="${LOADS_DAY[$q]}"
        else
            cols="${LOADS_GE[$q]}"
        fi

        sql=/tmp/q${q}_${pass}_${bm}.sql
        : > "$sql"
        for c in $cols; do echo "PRAGMA load_bitmap('$c');" >> "$sql"; done
        echo "PRAGMA bm_tpch($q);" >> "$sql"

        log="$LOG_DIR/q${q}_${pass}_${bm}.log"
        echo
        echo "[Q$q | pass=$pass | DEBIT_BM=$bm] cols=($cols) -> $log"
        t0=$(date +%s)
        DEBIT_BM=$bm DEBIT_ITER=$ITER DEBIT_WARMUP=$WARM TPCH_SF=10 \
            build/release/duckdb tpch_sf10.db < "$sql" > "$log" 2>&1
        rc=$?
        t1=$(date +%s)
        dur=$((t1-t0))

        # status: last [OK] / [FAIL]
        status=$(grep -E '^\[OK\]|^\[FAIL\]' "$log" | tail -1 | grep -oE '^\[OK\]|^\[FAIL\]')
        [ -z "$status" ] && status="[NORES]"

        # median = first number after the last 'TOTAL : '  (Q1/Q14/Q17/Q19/Q10 style)
        median=$(grep -E 'TOTAL\s*:' "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        # Q15 uses per-backend RESULTS block: '  CB    Total= X.X +/- Y.Y'
        if [ -z "$median" ]; then
            median=$(grep -E '^  (CB|CR|CRR|WAH|EW|CON)\s' "$log" | tail -1 \
                     | grep -oE 'Total=\s*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
        fi
        [ -z "$median" ] && median="-"

        if [ $rc -ne 0 ] && [ "$status" != "[OK]" ]; then
            echo "  [ERR rc=$rc] ${dur}s"
            echo "$q,$pass,$bm,-,ERR_rc=$rc" >> "$SUMMARY"
        else
            tag="$status"
            [ $rc -ne 0 ] && tag="${status}*"
            echo "  [$tag ${dur}s]  median=$median ms"
            echo "$q,$pass,$bm,$median,$tag" >> "$SUMMARY"
        fi
    done
done

echo
echo "============================================================"
echo " Done. Summary CSV:  $SUMMARY"
echo " Per-run logs:       $LOG_DIR/q<Q>_<pass>_<bm>.log"
echo "============================================================"
