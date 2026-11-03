#!/usr/bin/env bash
# Queue v32 (3060 Ti, after v31; the machine is free until 4.11 18:00). SPLADE-to-SPLADE distillation
# sanity test (docs/PLAN-5070-MLM-2026-11-03.md §11.1): does score distillation inside the SPLADE family
# work at all? Teacher soup-v9b (63.65 / 42.56), student sdadas/polish-splade, rows of n1 and n4.
#   1  teacher scores: kitchen bridge/splade_teacher_scores.py writes soup-v9b dot products into
#      pos_scores / neg_scores of fiqa-plus-squadv2.jsonl and mix3-6219.jsonl (suffix -tv9b).
#   2  four arms at 2 x 16: distil-v9b-{kl,mmse}-{n1,n4} (configs/next/3060/distil_v9b_*.yaml);
#      each: sparsity, leaderboard tiny10; heldout8 when tiny10 >= 62.5.
#   3  soup-v11-distil = the five v9b ingredients + the best arm, when its heldout8 >= 42.0.
#   4  summary_v32.md, commit, push.
# Controls: arm-n4-3060 62.89 / 42.01 (MNRL on the n4 rows), n1-5070 62.98 / 42.17, base 61.59 / 41.03.
# Reading: an arm >= base + 1.5 on tiny10 means same-family score KD works and waits for a bigger teacher;
# an arm <= base means the earlier cross-encoder result (KD < MNRL) was not only a scale problem.
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v32.sh > logs/queue_v32.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v32 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v32.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN=${KITCHEN:-$HOME/work/speakleash/retrieval-data-kitchen}
V32_END=${V32_END:-2026-11-04 18:00}
END=$(date -d "$V32_END" +%s)
B_SCORE=1800; B_TRAIN=6000; B_TINY=900; B_H8=2700
V9B="models/n1-5070 models/n3-5070 models/n4-5070 models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() { local need=$1; shift; fits "$need" || { log "no time before $V32_END for a ${need}s step; skipping: $*"; return 1; }; "$@"; }
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in v31 v30 v27; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }

teacher_scores() {  # teacher_scores ROWS OUT: soup-v9b dot products into pos_scores / neg_scores
    local rows="$1" out="$2"
    [ -s "$out" ] && { log "teacher scores: $out already present"; return 0; }
    [ -f "$rows" ] || { log "teacher scores: $rows missing"; record "tscore:$(basename "$out")" SKIPPED "no rows"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "teacher scores soup-v9b -> $out"; local s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$B_SCORE" "$PY" "$KITCHEN/bridge/splade_teacher_scores.py" --teacher models/soup-v9b \
        --rows "$rows" --output "$out" --max-seq-length 512 --batch-size 16 > "$LOGS/tscore_$(basename "$out" .jsonl).log" 2>&1
    local e=$((SECONDS - s))
    if [ -s "$out" ] && [ "$(wc -l < "$out")" = "$(wc -l < "$rows")" ]; then
        log "teacher scores done in ${e}s: $(tail -1 "$LOGS/tscore_$(basename "$out" .jsonl).log")"; record "tscore:$(basename "$out")" ok "-" "${e}s"
    else log "teacher scores FAILED after ${e}s"; record "tscore:$(basename "$out")" FAILED "-" "${e}s"; rm -f "$out"; return 1; fi
}

log "=== v32 start: end $V32_END; $(free_gb) GB free on / ==="
log "=== v32 step 1: teacher scores (soup-v9b) ==="
step $B_SCORE teacher_scores datasets/arms/fiqa-plus-squadv2.jsonl datasets/arms/fiqa-plus-squadv2-tv9b.jsonl
step $B_SCORE teacher_scores datasets/arms/mix3-6219.jsonl datasets/arms/mix3-6219-tv9b.jsonl

log "=== v32 step 2: distillation arms ==="
best=""; best_v=""
for tag in distil-v9b-kl-n1 distil-v9b-mmse-n1 distil-v9b-kl-n4 distil-v9b-mmse-n4; do
    cfg="$C/${tag//-/_}.yaml"
    step $B_TRAIN train "$tag" "$cfg" || continue
    step 120 sparsity "arm-$tag" "models/arm-$tag"
    step $B_TINY regime quantized "arm-$tag" "models/arm-$tag" tiny10 16
    v=$(lbv "arm-$tag" tiny10)
    log "  arm-$tag: tiny10 ${v:-?} (base 61.59, arm-n4-3060 62.89), $(sparsity_mean "$RESULTS_ROOT/sparsity/arm-$tag.json" "arm-$tag") dims/doc"
    if [ -n "$v" ] && [ "$(ge "$v" 62.5)" = 1 ]; then
        step $B_H8 regime quantized "arm-$tag" "models/arm-$tag" heldout8 16
        h=$(lbv "arm-$tag" heldout8); log "  arm-$tag: heldout8 ${h:-?} (arm-n4-3060 42.01, soup-v9b 42.56)"
        if [ -n "$h" ] && { [ -z "$best_v" ] || [ "$(ge "$h" "$best_v")" = 1 ]; }; then best="$tag"; best_v="$h"; fi
    fi
done

log "=== v32 step 3: soup with the v9b ingredients ==="
if [ -n "$best" ] && [ "$(ge "$best_v" 42.0)" = 1 ]; then
    if [ ! -d models/soup-v11-distil ]; then
        log "soup-v11-distil = v9b ingredients + arm-$best"
        timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models $V9B "models/arm-$best" --output models/soup-v11-distil > "$LOGS/soup_v11_distil.log" 2>&1 || log "soup FAILED (logs/soup_v11_distil.log)"
    fi
    if [ -d models/soup-v11-distil ]; then
        step 120 sparsity soup-v11-distil models/soup-v11-distil
        step $B_TINY regime quantized soup-v11-distil models/soup-v11-distil tiny10 16
        step $B_H8 regime quantized soup-v11-distil models/soup-v11-distil heldout8 16
        log "  soup-v11-distil: $(lbv soup-v11-distil tiny10) / $(lbv soup-v11-distil heldout8) (soup-v9b 63.65 / 42.56)"
    fi
else log "no arm reached heldout8 42.0 (best: ${best:-none} ${best_v:-})"; fi

log "=== v32 step 4: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v32.md"
{
    echo "# v32 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): SPLADE-to-SPLADE distillation, teacher soup-v9b, student polish-splade"
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-n4-3060 n1-5070 arm-distil-v9b-kl-n1 arm-distil-v9b-mmse-n1 arm-distil-v9b-kl-n4 arm-distil-v9b-mmse-n4 soup-v11-distil soup-v9b \
        --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-distil-v9b-kl-n1 --model arm-distil-v9b-mmse-n1 --model arm-distil-v9b-kl-n4 --model arm-distil-v9b-mmse-n4 --model soup-v11-distil 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v32)"; echo; echo '```'
    grep -E "^(tscore|train|sparsity|regime:quantized):(.*distil|soup-v11)" "$STATE" | tail -30
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v32.out"
log "summary written to $SUM"
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM" "$EXP_ROOT/queues/queue_v32.sh" "$C"/distil_v9b_*.yaml; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v32.out"
done
if git commit -q -m "Add the v32 results: SPLADE-to-SPLADE distillation from soup-v9b" 2>> "$LOGS/queue_v32.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v32.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v32.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v32.log)"
else log "nothing to commit or commit failed (see logs/queue_v32.out)"; fi
log "=== v32 done ($(date '+%d.%m %H:%M')) ==="
