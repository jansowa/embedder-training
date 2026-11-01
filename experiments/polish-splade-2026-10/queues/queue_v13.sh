#!/usr/bin/env bash
# Night of 27/28.10, after queue_v11 (step B of docs/PLAN-2026-10-27.md). Replaces queue_v12,
# stopped at 22:03 before it started anything: the first step-B arm (margin-MSE, normalisation
# none) lost 4.8 points on tiny (55.36 vs base 60.14) and shrank the documents from 436 to 260
# active dims while the FLOPS terms stayed below 0.1 % of the loss. The student's margins are
# about twice the teacher's (base model: mean margin 7.5 on the old FiQA negatives, teacher
# median 5.45), so plain margin-MSE spends the training on shrinking the scores. This queue
# trains the scale-fixed variants (score_scale: auto, alpha fitted on the untrained student;
# scripts/build_mmse_v3_configs.py) and KL arms, on the safe (v3), re-scored (v3old) and hard
# (v3hard) teacher data. Waits until logs/queue_v11.out says v11 is done. Blocks:
#   D1 fiqa-2073-v3old.jsonl: the old rows scored by roberta-v3 (score_flag_embedding_jsonl.py)
#   D2 fiqa-2073-v3hard(-plus).jsonl: scripts/fiqa_v3_mine.sh VARIANT=hard
#   T1 mmse scaleauto on v3 and v3hard, KL scaleauto on v3hard, mmse scaleauto on v3old
#   H  held-out-8 and the leaderboard regime (quantized tiny10 + heldout8) for the best T1 arm
#      when it reaches the base on tiny
#   T2 KL scaleauto on v3, mmse scaleauto on v3hardplus (when it differs from v3hard)
#   H2 the same for a T2 arm when it beats the T1 best; 256-token evaluation for the best overall
#   R  the rest of scripts/queue_v10.sh (soup-v2b held-out-2, index cost, wandb sync)
# Start with: setsid nohup bash scripts/queue_v13.sh > logs/queue_v13.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v13 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v13.pid"
MINE=/home/jsowa/work/speakleash/mine-hard-negatives
MPY="$MINE/.venv/bin/python"

waited=0
until grep -qE '=== v11 done ===|exiting' logs/queue_v11.out 2>/dev/null; do
    [ "$waited" -ge 28800 ] && { log "v13: v11 not finished after 8 h; exiting"; exit 1; }
    sleep 120; waited=$((waited + 120))
done
wait_gpu_free

arm() {  # arm NAME: train configs/next/3060/<NAME with _>.yaml, tiny, sparsity, log the fitted scale
    local cfg="configs/next/3060/${1//-/_}.yaml"
    train "$1" "$cfg" && { bench "arm-$1" "models/arm-$1"; sparsity "arm-$1" "models/arm-$1"; }
    log "  arm-$1: tiny $(tiny "arm-$1"); $(scale_of "arm-$1")"
}
best_arm() {  # best_arm NAME...: the name with the highest tiny result (empty when none has one)
    local c v best="" best_v=""
    for c in "$@"; do
        v=$(tiny "$c"); [ -n "$v" ] || continue
        if [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; then best="$c"; best_v="$v"; fi
    done
    echo "$best"
}
scale_of() {  # scale_of NAME: the fitted teacher score scale of a scaleauto arm, from its run dir
    local f; f=$(ls -1dt "runs/$1/sentence-transformers/splade"/*/score_scale.json 2>/dev/null | head -1)
    [ -n "$f" ] && "$PY" -c "
import json; d = json.load(open('$f'))
print('scale alpha %.3f (student margin %.2f vs teacher %.2f, r %.2f, %d rows)' % (d['alpha'], d['student_margin_mean'], d['teacher_margin_mean'], d['pearson'], d['sample_rows']))" 2>/dev/null
}
leaderboard() {  # leaderboard NAME: held-out-8 and the leaderboard regime (quantized tiny10 + heldout8)
    holdout "$1" "models/$1"
    regime quantized "$1" "models/$1" tiny10 16
    regime quantized "$1" "models/$1" heldout8 16
}
reaches_base() { [ -n "$(tiny "$1")" ] && [ -n "$BASE" ] && [ "$(ge "$(tiny "$1")" "$BASE")" = 1 ]; }

REF=$(tiny arm-cluster-like-fiqa); BASE=$(tiny base-splade-wrapped)
for c in none minmax zscore; do log "step B: arm-mmse-fiqa-v3-$c tiny $(tiny "arm-mmse-fiqa-v3-$c")"; done
log "=== v13 start: MNRL reference ${REF:-?}, base ${BASE:-?}, $(free_gb) GB free on / ==="

log "=== v13 block D1: the old rows scored by roberta-v3 (v3old) ==="
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

log "=== v13 block D2: hard re-mining (v3hard, v3hard-plus) ==="
HARD=datasets/arms/fiqa-2073-v3hard.jsonl; HARD_PLUS=datasets/arms/fiqa-2073-v3hard-plus.jsonl
if [ ! -f "$HARD" ]; then
    s=$SECONDS
    VARIANT=hard bash scripts/fiqa_v3_mine.sh > "$LOGS/fiqa_v3hard_mine.out" 2>&1 \
        && log "hard mining done in $((SECONDS - s))s" || log "hard mining FAILED after $((SECONDS - s))s (logs/fiqa_v3hard_mine.out)"
else log "v3hard already present"; fi

log "=== v13 block T1: scale-fixed arms (margin-MSE score_scale auto, KL) ==="
T1=(mmse-fiqa-v3-none-scaleauto)
[ -f "$HARD" ] && T1+=(mmse-fiqa-v3hard-none-scaleauto kl-fiqa-v3hard-scaleauto) || log "no v3hard file; its arms skipped"
[ -f "$OLD" ] && T1+=(mmse-fiqa-v3old-none-scaleauto) || log "no v3old file; its arm skipped"
for a in "${T1[@]}"; do arm "$a"; done
names=(); for a in "${T1[@]}"; do names+=("arm-$a"); done
top=$(best_arm "${names[@]}")
log "=== v13 block H: best T1 arm ${top:-none} (tiny $(tiny "${top:-x}"); MNRL reference ${REF:-?}, base ${BASE:-?}) ==="
if [ -n "$top" ] && reaches_base "$top"; then leaderboard "$top"; else log "block H skipped: no T1 arm reaches the base on tiny"; fi

log "=== v13 block T2: KL on v3, margin-MSE on v3hard-plus ==="
T2=(kl-fiqa-v3-scaleauto)
if [ -f "$HARD_PLUS" ] && [ -f "$HARD" ] && ! cmp -s "$HARD" "$HARD_PLUS"; then T2+=(mmse-fiqa-v3hardplus-none-scaleauto); else log "v3hard-plus missing or identical to v3hard; no v3hardplus arm"; fi
for a in "${T2[@]}"; do arm "$a"; done
for a in "${T2[@]}"; do names+=("arm-$a"); done
top2=$(best_arm "${names[@]}")
for c in "${names[@]}"; do log "  $c tiny $(tiny "$c")"; done
log "=== v13 block H2: best overall ${top2:-none} (tiny $(tiny "${top2:-x}")) ==="
if [ -n "$top2" ] && reaches_base "$top2"; then
    [ "$top2" != "$top" ] && leaderboard "$top2"
    regime quantized "$top2-len256" "models/$top2" tiny10 16 256
else log "block H2 skipped: no arm reaches the base on tiny"; fi

log "=== v13 block R: the rest of queue_v10 (soup-v2b held-out-2, index cost, wandb sync) ==="
bash scripts/queue_v10.sh

"$PY" scripts/queue_summary.py > results/summary_v13.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v13.md)"
log "=== v13 done ==="
