#!/usr/bin/env bash
# Night of 27/28.10, after queue_v11 (step B of docs/PLAN-2026-10-27.md, started by
# scripts/fiqa_v3_mine.sh). The first v3 export gave only "safe" negatives (median
# teacher margin 5.5 logits) and no synthetic positives, so this queue builds two
# harder teacher-data variants and trains on them, then does steps E and H and the
# rest of queue_v10. Waits until logs/queue_v11.out says v11 is done. Blocks:
#   D1 fiqa-2073-v3old.jsonl: the old rows (their own negatives) scored by roberta-v3
#      with score_flag_embedding_jsonl.py of the mining project (~5 min)
#   D2 fiqa-2073-v3hard(-plus).jsonl: scripts/fiqa_v3_mine.sh VARIANT=hard
#      (candidate u_floor 0.5, final u_floor 0.05, MINE_POSITIVES export; ~30 min)
#   T  margin-MSE arms on v3old, v3hard and v3hardplus (when it differs from v3hard)
#      with the normalisation of the best step-B arm; tiny + sparsity each
#   E  docreg 0.0003 / 0.003 and the hard+lowcov half, on the best data variant
#   H  held-out-8, leaderboard regime (quantized tiny10 + heldout8) and the
#      256-token evaluation for the best margin-MSE arm overall
#   R  the rest of scripts/queue_v10.sh (soup-v2b held-out-2, index cost, wandb sync)
# Configs: scripts/build_mmse_v3_configs.py -> configs/next/3060/mmse_fiqa_<data>_<norm>[_<tweak>].yaml
# Start with: setsid nohup bash scripts/queue_v12.sh > logs/queue_v12.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v12 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v12.pid"
MINE=/home/jsowa/work/speakleash/mine-hard-negatives
MPY="$MINE/.venv/bin/python"

waited=0
until grep -qE '=== v11 done ===|exiting' logs/queue_v11.out 2>/dev/null; do
    grep -qE 'FAILED|giving up' logs/fiqa_v3_mine.out 2>/dev/null && { log "v12: the mining run failed, so v11 never starts; exiting"; exit 1; }
    [ "$waited" -ge 28800 ] && { log "v12: v11 not finished after 8 h; exiting"; exit 1; }
    sleep 120; waited=$((waited + 120))
done
wait_gpu_free

data_file() {  # data_file VARIANT -> training file of that teacher-data variant
    case "$1" in
        v3) echo datasets/arms/fiqa-2073-v3.jsonl;; v3old) echo datasets/arms/fiqa-2073-v3old.jsonl;;
        v3hard) echo datasets/arms/fiqa-2073-v3hard.jsonl;; v3hardplus) echo datasets/arms/fiqa-2073-v3hard-plus.jsonl;;
    esac
}
arm() {  # arm NAME CONFIG: train, tiny, sparsity
    train "$1" "$2" && { bench "arm-$1" "models/arm-$1"; sparsity "arm-$1" "models/arm-$1"; }
}
best_arm() {  # best_arm NAME...: the name with the highest tiny result (empty when none has one)
    local c v best="" best_v=""
    for c in "$@"; do
        v=$(tiny "$c"); [ -n "$v" ] || continue
        if [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; then best="$c"; best_v="$v"; fi
    done
    echo "$best"
}

bestB=$(best_arm arm-mmse-fiqa-v3-none arm-mmse-fiqa-v3-minmax arm-mmse-fiqa-v3-zscore)
NORM=${bestB##*-}
[ -n "$bestB" ] || { NORM=none; log "v12: no step-B arm has a tiny result; falling back to normalisation none"; }
REF=$(tiny arm-cluster-like-fiqa)
log "=== v12 start: best step-B arm ${bestB:-none} ($(tiny "${bestB:-x}"); MNRL reference ${REF:-?}), normalisation $NORM, $(free_gb) GB free on / ==="

log "=== v12 block D1: the old rows scored by roberta-v3 (v3old) ==="
OLD=datasets/arms/fiqa-2073-v3old.jsonl; OLD_DIR=datasets/arms/fiqa-v3old-mine; mkdir -p "$OLD_DIR" results/data
if [ ! -f "$OLD" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    "$PY" - <<'EOF'
import json
# the scorer backs pos_scores/neg_scores up under original_* only when those keys are absent
with open("datasets/arms/fiqa-v3old-mine/input.jsonl", "w", encoding="utf-8") as out:
    for line in open("datasets/arms/fiqa-2073.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        r.pop("original_pos_scores", None); r.pop("original_neg_scores", None)
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF
    (set -a; source scripts/fiqa_v3_mine.env; set +a; cd "$MINE" && timeout -k 60 7200 "$MPY" app_code/curated_negatives/score_flag_embedding_jsonl.py \
        --input_path "$OLDPWD/$OLD_DIR/input.jsonl" --output_path "$OLDPWD/$OLD_DIR/scored.jsonl" \
        --reranker_model_name sdadas/polish-reranker-roberta-v3 --reranker_batch_size 8 --record_batch_size 32 \
        --report_path "$OLDPWD/$OLD_DIR/scored.jsonl.report.json") > "$LOGS/fiqa_v3old_score.log" 2>&1 \
        && "$PY" scripts/fiqa_v3_finalize.py --mined "$OLD_DIR/scored.jsonl" --rows datasets/arms/fiqa-2073.jsonl \
            --out "$OLD" --report results/data/fiqa-2073-v3old.json > "$LOGS/fiqa_v3old_finalize.log" 2>&1 \
        && log "v3old written ($OLD) in $((SECONDS - s))s" || log "v3old FAILED after $((SECONDS - s))s (logs/fiqa_v3old_*.log)"
else log "v3old already present"; fi

log "=== v12 block D2: hard re-mining (v3hard, v3hard-plus) ==="
HARD=datasets/arms/fiqa-2073-v3hard.jsonl; HARD_PLUS=datasets/arms/fiqa-2073-v3hard-plus.jsonl
if [ ! -f "$HARD" ]; then
    s=$SECONDS
    VARIANT=hard bash scripts/fiqa_v3_mine.sh > "$LOGS/fiqa_v3hard_mine.out" 2>&1 \
        && log "hard mining done in $((SECONDS - s))s" || log "hard mining FAILED after $((SECONDS - s))s (logs/fiqa_v3hard_mine.out)"
else log "v3hard already present"; fi
VARIANTS=()
[ -f "$OLD" ] && VARIANTS+=(v3old)
[ -f "$HARD" ] && VARIANTS+=(v3hard)
if [ -f "$HARD_PLUS" ] && [ -f "$HARD" ] && ! cmp -s "$HARD" "$HARD_PLUS"; then VARIANTS+=(v3hardplus); else log "v3hard-plus missing or identical to v3hard; no v3hardplus arm"; fi
log "data variants to train: ${VARIANTS[*]:-none}"

log "=== v12 block T: margin-MSE arms on the new teacher data (normalisation $NORM) ==="
for d in "${VARIANTS[@]}"; do arm "mmse-fiqa-$d-$NORM" "configs/next/3060/mmse_fiqa_${d}_${NORM}.yaml"; done

cands=("${bestB:-arm-mmse-fiqa-v3-$NORM}"); for d in "${VARIANTS[@]}"; do cands+=("arm-mmse-fiqa-$d-$NORM"); done
bestD=$(best_arm "${cands[@]}")
DATA=v3; for d in v3old v3hard v3hardplus; do [ "$bestD" = "arm-mmse-fiqa-$d-$NORM" ] && DATA=$d; done
for c in "${cands[@]}"; do log "  $c tiny $(tiny "$c")"; done
log "best data variant: $DATA (${bestD:-none}, tiny $(tiny "${bestD:-x}"))"

log "=== v12 block E: docreg 0.0003 / 0.003 and the hard+lowcov half on $DATA ==="
src=$(data_file "$DATA"); half="${src%.jsonl}-hardlowcov.jsonl"
if [ -f "$src" ] && [ ! -f "$half" ]; then
    SRC="$src" HALF="$half" "$PY" - <<'EOF' && log "built $half"
import json, os
ids = set()
for n in ("fiqa-hard", "fiqa-lowcov"):
    ids |= {json.loads(l)["query_id"] for l in open(f"datasets/arms/splits/{n}.jsonl", encoding="utf-8") if l.strip()}
rows = [l for l in open(os.environ["SRC"], encoding="utf-8") if l.strip() and json.loads(l)["query_id"] in ids]
with open(os.environ["HALF"], "w", encoding="utf-8") as fh:
    fh.writelines(rows)
print("hard+lowcov rows:", len(rows))
EOF
fi
E_ARMS=()
for tweak in docreg3e-4 docreg3e-3 hardlowcov; do
    name="mmse-fiqa-$DATA-$NORM-$tweak"; E_ARMS+=("arm-$name")
    arm "$name" "configs/next/3060/mmse_fiqa_${DATA}_${NORM}_${tweak}.yaml"
done

log "=== v12 block H: held-out and leaderboard regime for the best margin-MSE arm ==="
top=$(best_arm "${cands[@]}" "${E_ARMS[@]}")
if [ -n "$top" ]; then
    log "best margin-MSE arm overall: $top (tiny $(tiny "$top"); MNRL reference ${REF:-?})"
    holdout "$top" "models/$top"
    regime quantized "$top" "models/$top" tiny10 16
    regime quantized "$top" "models/$top" heldout8 16
    regime quantized "$top-len256" "models/$top" tiny10 16 256
else log "block H skipped: no arm has a tiny result"; fi

log "=== v12 block R: the rest of queue_v10 (soup-v2b held-out-2, index cost, wandb sync) ==="
bash scripts/queue_v10.sh

"$PY" scripts/queue_summary.py > results/summary_v12.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v12.md)"
log "=== v12 done ==="
