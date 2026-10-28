#!/usr/bin/env bash
# 28.10 night: the rest of queue_v17, which was stopped during the held-out of soup-v5c so
# that queue_v18 could pack the third package for the 16 GB machine first. The three soups
# of the user's 5070 Ti runs all beat arm-cluster-like-fiqa on tiny (soup-v5c = n3 + soup-v2b
# + n1 62.43, soup-v5d = n4 + soup-v2b + n1 62.42, soup-v5e = n1 + n3 + n4 + soup-v2b 62.52),
# so each gets held-out-8 and the leaderboard regime (quantized tiny10 + heldout8).
# Waits for "=== v18 done ===". Start with: setsid nohup bash scripts/queue_v19.sh > logs/queue_v19.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v19 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v19.pid"

waited=0
until grep -qE '=== v18 done ===' logs/queue_v18.out 2>/dev/null; do
    [ "$waited" -ge 21600 ] && { log "v19: v18 not finished after 6 h; exiting"; exit 1; }
    sleep 60; waited=$((waited + 60))
done
wait_gpu_free
reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
REF=$(tiny arm-cluster-like-fiqa)
log "=== v19 start: soups of the 5070 Ti runs on held-out and in the leaderboard regime (reference ${REF:-?}) ==="
for s in soup-v5e soup-v5c soup-v5d; do
    [ -f "results/$s/final/metrics.json" ] || { log "  $s: no tiny result; skipped"; continue; }
    if reaches "$s" "$REF"; then holdout "$s" "models/$s"; leaderboard "$s"
    else log "  $s: tiny $(tiny "$s") below the reference $REF; skipped"; fi
done
"$PY" scripts/queue_summary.py > results/summary_v19.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v19.md)"
log "=== v19 done ==="
