#!/usr/bin/env bash
# Completes the leaderboard-regime comparison of the three models that go into
# the final report (27.10 evening). The old-pyserini regime reproduces the
# published polish-splade entry within 0.3 on 18 tasks; base and soup-of-ft
# already have tiny10 + heldout8 + the five heldout2 tasks in it, soup-v2b only
# tiny10 + heldout8. Blocks:
#   B  old-pyserini regime, soup-v2b, the five heldout2 tasks (~2 h)
#   Q  index and query cost of base, soup-of-ft, soup-v2b on one sample
#      (results/index_cost_best.json: active doc dims, active query dims,
#      postings per query)
#   W  wandb sync of the offline runs, best effort (the network is flaky)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v10 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v10.pid"

log "=== v10 start: $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GB free on /, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="

log "=== v10 block B: old-pyserini regime, soup-v2b on the five heldout2 tasks ==="
for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
    bs=16; case "$ts" in heldout2_poquad|heldout2_treccovid) bs=8;; esac
    regime old-pyserini soup-v2b models/soup-v2b "$ts" "$bs"
done

log "=== v10 block Q: index and query cost of base, soup-of-ft, soup-v2b ==="
wait_gpu_free; wait_ram
timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 \
    --model base=models/base-splade-wrapped --model soup-of-ft=models/soup-of-ft --model soup-v2b=models/soup-v2b \
    --output-json results/index_cost_best.json > "$LOGS/index_cost_best.log" 2>&1 \
    && log "index-cost profile written (results/index_cost_best.json)" || log "index-cost profile FAILED (logs/index_cost_best.log)"

log "=== v10 block W: wandb sync (best effort, 10 min cap) ==="
timeout -k 30 600 .venv/bin/wandb sync wandb/offline-run-* > "$LOGS/wandb_sync.log" 2>&1 \
    && log "wandb sync done" || log "wandb sync incomplete (logs/wandb_sync.log)"

log "=== v10 done ==="
