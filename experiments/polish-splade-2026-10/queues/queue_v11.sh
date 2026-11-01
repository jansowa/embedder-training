#!/usr/bin/env bash
# Distillation step B of docs/PLAN-2026-10-27.md: margin-MSE on the fiqa-2073
# queries re-scored by sdadas/polish-reranker-roberta-v3 (positives and all 10
# negatives) in ~/work/speakleash/mine-hard-negatives (its JSONL export).
# Needs datasets/arms/fiqa-2073-v3.jsonl (FlagEmbedding-style rows with
# pos_scores / neg_scores from v3); refuses to start without it. Blocks:
#   T  three arms, one per teacher-score normalisation (none, minmax, zscore),
#      recipe of arm-cluster-like-fiqa otherwise (512 tokens, 10 negatives)
#   M  tiny bench + document sparsity of each arm
#   H  held-out-8 and the leaderboard regime (quantized, tiny10) for the best
#      arm if it is within 0.3 of arm-cluster-like-fiqa on tiny (62.27)
# Start with: setsid nohup bash scripts/queue_v11.sh > logs/queue_v11.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v11 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v11.pid"

DATA=datasets/arms/fiqa-2073-v3.jsonl
[ -f "$DATA" ] || { log "v11: $DATA missing (export it from ~/work/speakleash/mine-hard-negatives with the v3 reranker as the final scorer); exiting"; exit 1; }
rows=$(wc -l < "$DATA"); bad=$("$PY" -c "
import json,sys
n=0
for line in open('$DATA'):
    r=json.loads(line)
    if len(r.get('neg',[]))<10 or len(r.get('neg_scores',[]))<10 or not r.get('pos_scores'): n+=1
print(n)")
log "=== v11 start: $rows rows in $DATA, $bad rows with < 10 scored negatives, $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GB free on / ==="
[ "$bad" -eq 0 ] || { log "v11: fix the $bad short rows first (the backend needs 10 scored negatives per query); exiting"; exit 1; }

REF=$(tiny arm-cluster-like-fiqa)
log "reference: arm-cluster-like-fiqa tiny ${REF:-?} (MNRL, same rows and recipe)"

log "=== v11 block T: margin-MSE arms ==="
for tag in none minmax zscore; do
    train "mmse-fiqa-v3-$tag" "configs/next/3060/mmse_fiqa_v3_$tag.yaml" \
        && { bench "arm-mmse-fiqa-v3-$tag" "models/arm-mmse-fiqa-v3-$tag"; sparsity "arm-mmse-fiqa-v3-$tag" "models/arm-mmse-fiqa-v3-$tag"; }
done

log "=== v11 block H: held-out and leaderboard regime for the best arm ==="
best=""; best_v=""
for tag in none minmax zscore; do
    v=$(tiny "arm-mmse-fiqa-v3-$tag"); [ -n "$v" ] || continue
    if [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; then best="arm-mmse-fiqa-v3-$tag"; best_v="$v"; fi
done
if [ -n "$best" ] && { [ -z "$REF" ] || [ "$(ge "$best_v" "$(fsub "$REF" 0.3)")" = 1 ]; }; then
    holdout "$best" "models/$best"
    regime quantized "$best" "models/$best" tiny10 16
else
    log "block H skipped: best margin-MSE arm ${best:-none} (${best_v:-?}) is more than 0.3 below arm-cluster-like-fiqa ($REF)"
fi

"$PY" scripts/queue_summary.py > results/summary_v11.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v11.md)"
log "=== v11 done ==="
