#!/usr/bin/env bash
# Steps C and E of docs/PLAN-2026-10-27.md, after queue_v11 (step B, started by
# scripts/fiqa_v3_mine.sh). Waits until logs/queue_v11.out says v11 is done, then
# takes the teacher-score normalisation of the best step-B arm (tiny) and runs:
#   C  margin-MSE on fiqa-2073-v3-plus (extra synthetic positives), tiny + sparsity
#   E  docreg 0.0003 / 0.003 and the hard+lowcov half of the rows, tiny + sparsity
#   H  held-out-8, leaderboard regime (quantized: tiny10, heldout8) and the
#      256-token evaluation (tiny10) for the best margin-MSE arm overall
#   R  the rest of scripts/queue_v10.sh (soup-v2b held-out-2 in old-pyserini,
#      index cost of the best models, wandb sync); done steps are skipped there
# Configs: scripts/build_mmse_v3_configs.py -> configs/next/3060/.
# Start with: setsid nohup bash scripts/queue_v12.sh > logs/queue_v12.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v12 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v12.pid"

waited=0
until grep -qE '=== v11 done ===|exiting' logs/queue_v11.out 2>/dev/null; do
    grep -qE 'FAILED|giving up' logs/fiqa_v3_mine.out 2>/dev/null && { log "v12: the mining run failed, so v11 never starts; exiting"; exit 1; }
    [ "$waited" -ge 28800 ] && { log "v12: v11 not finished after 8 h; exiting"; exit 1; }
    sleep 120; waited=$((waited + 120))
done
wait_gpu_free

DATA=datasets/arms/fiqa-2073-v3.jsonl; PLUS=datasets/arms/fiqa-2073-v3-plus.jsonl; HALF=datasets/arms/fiqa-2073-v3-hardlowcov.jsonl
[ -f "$DATA" ] || { log "v12: $DATA missing; exiting"; exit 1; }

best=""; best_v=""; NORM=""
for tag in none minmax zscore; do
    v=$(tiny "arm-mmse-fiqa-v3-$tag"); [ -n "$v" ] || continue
    if [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; then best="arm-mmse-fiqa-v3-$tag"; best_v="$v"; NORM=$tag; fi
done
[ -n "$NORM" ] || { log "v12: no step-B arm has a tiny result; exiting"; exit 1; }
REF=$(tiny arm-cluster-like-fiqa)
log "=== v12 start: best step-B arm $best ($best_v; MNRL reference ${REF:-?}), normalisation $NORM, $(free_gb) GB free on / ==="

if [ ! -f "$HALF" ]; then
    "$PY" - <<'EOF' && log "built $HALF"
import json
ids = set()
for n in ("fiqa-hard", "fiqa-lowcov"):
    ids |= {json.loads(l)["query_id"] for l in open(f"datasets/arms/splits/{n}.jsonl", encoding="utf-8") if l.strip()}
rows = [l for l in open("datasets/arms/fiqa-2073-v3.jsonl", encoding="utf-8") if l.strip() and json.loads(l)["query_id"] in ids]
with open("datasets/arms/fiqa-2073-v3-hardlowcov.jsonl", "w", encoding="utf-8") as fh:
    fh.writelines(rows)
print("hard+lowcov rows:", len(rows))
EOF
fi

log "=== v12 block C: extra positives ($PLUS) ==="
if [ -f "$PLUS" ]; then
    name="mmse-fiqa-v3plus-$NORM"
    train "$name" "configs/next/3060/mmse_fiqa_v3plus_$NORM.yaml" \
        && { bench "arm-$name" "models/arm-$name"; sparsity "arm-$name" "models/arm-$name"; }
else
    log "block C skipped: $PLUS missing"
fi

log "=== v12 block E: docreg 0.0003 / 0.003 and the hard+lowcov half ==="
for tag in docreg3e-4 docreg3e-3 hardlowcov; do
    name="mmse-fiqa-v3-$NORM-$tag"
    train "$name" "configs/next/3060/mmse_fiqa_v3_${NORM}_${tag}.yaml" \
        && { bench "arm-$name" "models/arm-$name"; sparsity "arm-$name" "models/arm-$name"; }
done

log "=== v12 block H: held-out and leaderboard regime for the best margin-MSE arm ==="
top=""; top_v=""
for c in "$best" "arm-mmse-fiqa-v3plus-$NORM" "arm-mmse-fiqa-v3-$NORM-docreg3e-4" "arm-mmse-fiqa-v3-$NORM-docreg3e-3" "arm-mmse-fiqa-v3-$NORM-hardlowcov"; do
    v=$(tiny "$c"); [ -n "$v" ] || continue
    log "  $c tiny $v"
    if [ -z "$top_v" ] || [ "$(ge "$v" "$top_v")" = 1 ]; then top="$c"; top_v="$v"; fi
done
if [ -n "$top" ]; then
    log "best margin-MSE arm overall: $top (tiny $top_v)"
    holdout "$top" "models/$top"
    regime quantized "$top" "models/$top" tiny10 16
    regime quantized "$top" "models/$top" heldout8 16
    regime quantized "$top-len256" "models/$top" tiny10 16 256
fi

log "=== v12 block R: the rest of queue_v10 (soup-v2b held-out-2, index cost, wandb sync) ==="
bash scripts/queue_v10.sh

"$PY" scripts/queue_summary.py > results/summary_v12.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v12.md)"
log "=== v12 done ==="
