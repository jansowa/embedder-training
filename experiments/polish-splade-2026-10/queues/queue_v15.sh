#!/usr/bin/env bash
# 28.10, after the second run of queue_v14 (logs/queue_v14b.out): the arms on the
# student-mined negatives again, at train_batch_size 1 x 32 accumulation. In v14 all four
# T1 arms died of CUDA OOM at 2 x 16: the negatives chosen by SPLADE are 2.5x longer than
# the old ones (median 1477 vs 424 characters, a third above 2000), so every batch fills
# the 512 tokens. Configs from scripts/build_splade_neg_configs.py. Blocks:
#   T1  own-b1; ctrl-b1 (the cluster-like recipe on the old rows at 1 x 32: the batch
#       control against arm-cluster-like-fiqa); mix-b1 (5 student + 5 old v3 negatives,
#       scripts/build_fiqa_mix_negatives.py); kl-own-scaleauto-b1; plus-b1; own-f02-b1;
#       soup-own-b1 unless v14 trained arm-mnrl-fiqa-splade-soup-own at 2 x 16
#   H   held-out-8 for the best new arm when it reaches the base on tiny, plus ctrl-b1 then;
#       leaderboard regime (quantized tiny10 + heldout8) when within 0.3 of the reference
#   S   soups soup-v5a (best new arm + soup-v2b) and soup-v5b (+ models/n1-5070); held-out
#       and leaderboard regime for a soup that reaches arm-cluster-like-fiqa on tiny
# Start with: setsid nohup bash scripts/queue_v15.sh > logs/queue_v15.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v15 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v15.pid"

waited=0
until grep -qE '=== v14 done ===' logs/queue_v14b.out 2>/dev/null; do
    [ "$waited" -ge 21600 ] && { log "v15: v14 (second run) not finished after 6 h; exiting"; exit 1; }
    sleep 60; waited=$((waited + 60))
done
wait_gpu_free

arm() {  # arm NAME: train configs/next/3060/<NAME with _>.yaml, tiny, sparsity, log the fitted scale
    local cfg="configs/next/3060/${1//-/_}.yaml"
    train "$1" "$cfg" && { bench "arm-$1" "models/arm-$1"; sparsity "arm-$1" "models/arm-$1"; }
    log "  arm-$1: tiny $(tiny "arm-$1") $(scale_of "arm-$1")"
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
print('(scale alpha %.3f, student margin %.2f vs teacher %.2f, r %.2f)' % (d['alpha'], d['student_margin_mean'], d['teacher_margin_mean'], d['pearson']))" 2>/dev/null
}
reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
soup() {  # soup NAME MODEL... : equal-weight soup into models/NAME, then tiny + sparsity
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    bench "$name" "models/$name"; sparsity "$name" "models/$name"
}

REF=$(tiny arm-cluster-like-fiqa); BASE=$(tiny base-splade-wrapped); NEAR=$(fsub "$REF" 0.3)
log "=== v15 start: MNRL reference ${REF:-?}, base ${BASE:-?}, $(free_gb) GB free on / ==="

log "=== v15 block T1: arms on the student-mined negatives at batch 1 x 32 ==="
T1=(mnrl-fiqa-splade-own-b1 mnrl-fiqa-ctrl-b1 mnrl-fiqa-splade-mix-b1 kl-fiqa-splade-own-scaleauto-b1 mnrl-fiqa-splade-plus-b1 mnrl-fiqa-splade-own-f02-b1)
[ -d models/arm-mnrl-fiqa-splade-soup-own ] && log "soup-own: v14 trained arm-mnrl-fiqa-splade-soup-own at 2 x 16 (tiny $(tiny arm-mnrl-fiqa-splade-soup-own)); no b1 arm" || T1+=(mnrl-fiqa-splade-soup-own-b1)
for a in "${T1[@]}"; do arm "$a"; done
names=(); for a in "${T1[@]}"; do [[ "$a" == *ctrl* ]] || names+=("arm-$a"); done
top=$(best_arm "${names[@]}")
log "=== v15 block H: best new arm ${top:-none} (tiny $(tiny "${top:-x}")); batch control arm-mnrl-fiqa-ctrl-b1 $(tiny arm-mnrl-fiqa-ctrl-b1) vs reference ${REF:-?}, base ${BASE:-?} ==="
if [ -n "$top" ] && reaches "$top" "$BASE"; then
    holdout "$top" "models/$top"
    if reaches "$top" "$NEAR"; then
        leaderboard "$top"
        [ -d models/arm-mnrl-fiqa-ctrl-b1 ] && holdout arm-mnrl-fiqa-ctrl-b1 models/arm-mnrl-fiqa-ctrl-b1
    else log "leaderboard regime skipped: $top is more than 0.3 below $REF on tiny"; fi
else log "block H skipped: no new arm reaches the base on tiny"; fi

log "=== v15 block S: soups of the best new arm ==="
for c in "${names[@]}" arm-mnrl-fiqa-ctrl-b1; do log "  $c tiny $(tiny "$c")"; done
if [ -n "$top" ]; then
    soup "soup-v5a" "models/$top" models/soup-v2b
    [ -d models/n1-5070 ] && soup "soup-v5b" "models/$top" models/soup-v2b models/n1-5070 || log "models/n1-5070 absent; no soup-v5b"
    for s in soup-v5a soup-v5b; do
        [ -f "results/$s/final/metrics.json" ] || continue
        if reaches "$s" "$REF"; then holdout "$s" "models/$s"; leaderboard "$s"; else log "  $s: tiny $(tiny "$s") below the reference $REF; no held-out"; fi
    done
else log "block S skipped: no new arm trained"; fi

"$PY" scripts/queue_summary.py > results/summary_v15.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v15.md)"
log "=== v15 done ==="
