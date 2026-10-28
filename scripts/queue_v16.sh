#!/usr/bin/env bash
# 28.10, replaces the tail of queue_v15 after its mix-b1 arm. v15 so far: ten student-mined
# negatives per query cost 2 points on tiny at the same batch regime (arm-mnrl-fiqa-splade-
# own-b1 59.53 vs the batch control arm-mnrl-fiqa-ctrl-b1 61.49, uniformly over the 10
# tasks; the final MNRL loss stays 4x higher than with the old negatives), and the batch
# regime 1 x 32 itself costs 0.8 (ctrl-b1 61.49 vs arm-cluster-like-fiqa 62.27 at 2 x 16)
# while sparsifying the model (348 active dims/doc vs 436). The remaining v15 arms (kl on
# the pure student file, pure plus, pure own-f02) would repeat the first finding.
# The main dose-response arms at 2 x 16 (mix2, plusmix, own, plus: configs
# configs/next/3060/mnrl_fiqa_splade_{mix2,plusmix,own,plus}.yaml) run on the user's
# 5070 Ti (16 GB), so this queue does the non-overlapping work on the 3060:
#   W   wait for the tiny result of arm-mnrl-fiqa-splade-mix-b1 in logs/queue_v15.out, then
#       stop queue v15 (its pid file) together with the training it has just started
#   L   leaderboard regime (quantized tiny10 + heldout8) for n1-5070, the best held-out model
#   T1  kl-fiqa-splade-mix-scaleauto (KL on the 5 + 5 mix) and mnrl-fiqa-splade-mix2-pp
#       (the mix2 file plus up to 2 promoted positives per query, the separate "missing
#       positives" experiment), each as -c2 (batch 2 x 16, splade_pooling_chunk_size 2;
#       the v14 OOM was 192 MiB short and the chunk saves ~0.9 GB) with a -b1 fallback;
#       when -c2 dies, also mix2-b1 as the local control of mix2-pp-b1
#   H   held-out-8 for the best local arm on student negatives when it reaches the base on
#       tiny; leaderboard regime when within 0.3 of the reference; ctrl-b1 on held-out then
#   S   soups soup-v5a (best + soup-v2b) and soup-v5b (+ n1-5070) when the best reaches the base
# Start with: setsid nohup bash scripts/queue_v16.sh > logs/queue_v16.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v16 MIN_RAM_GB=6
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v16.pid"

waited=0
until grep -qE 'arm-mnrl-fiqa-splade-mix-b1: tiny|=== v15 done ===' logs/queue_v15.out 2>/dev/null; do
    [ "$waited" -ge 14400 ] && { log "v16: no mix-b1 result from v15 after 4 h; exiting"; exit 1; }
    sleep 60; waited=$((waited + 60))
done

stop_v15() {  # stop the v15 queue script and the process groups of its running step
    local qp kids k n
    qp=$(cat "$LOGS/queue_v15.pid" 2>/dev/null)
    [ -n "$qp" ] && kill -0 "$qp" 2>/dev/null || { log "v15 is not running"; return 0; }
    kids=$(pgrep -P "$qp")
    kill -TERM "$qp" 2>/dev/null
    for k in $kids; do kill -TERM -- "-$k" 2>/dev/null || kill -TERM "$k" 2>/dev/null; done
    log "v15 (pid $qp) stopped after its mix-b1 arm; children: ${kids:-none}"
    for n in $(seq 1 12); do
        [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -eq 0 ] && return 0
        sleep 10
    done
    for k in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -KILL "$k" 2>/dev/null; done
    sleep 10
}
stop_v15
wait_gpu_free

arm() {  # arm NAME: train configs/next/3060/<NAME with _>.yaml, tiny, sparsity, log the fitted scale
    local cfg="configs/next/3060/${1//-/_}.yaml"
    train "$1" "$cfg" && { bench "arm-$1" "models/arm-$1"; sparsity "arm-$1" "models/arm-$1"; }
    log "  arm-$1: tiny $(tiny "arm-$1") $(scale_of "arm-$1")"
}
C2_OK=""  # set once a -c2 arm trains; "no" once one dies
arm_fit() {  # arm_fit NAME: NAME-c2 (batch 2 x 16, pooling chunk 2) unless -c2 is known to die, else NAME-b1
    if [ "$C2_OK" != no ]; then
        arm "$1-c2"
        if [ -d "models/arm-$1-c2" ]; then C2_OK=yes; return 0; fi
        C2_OK=no; log "  arm-$1-c2 did not train (logs/arm_$1-c2.log); -c2 arms off, falling back to batch 1 x 32"
    fi
    [ -d "models/arm-$1-b1" ] && { log "  arm-$1-b1 already trained (tiny $(tiny "arm-$1-b1"))"; return 0; }
    arm "$1-b1"
}
scale_of() {  # scale_of NAME: the fitted teacher score scale of a scaleauto arm, from its run dir
    local f; f=$(ls -1dt "runs/$1/sentence-transformers/splade"/*/score_scale.json 2>/dev/null | head -1)
    [ -n "$f" ] && "$PY" -c "
import json; d = json.load(open('$f'))
print('(scale alpha %.3f, student margin %.2f vs teacher %.2f, r %.2f)' % (d['alpha'], d['student_margin_mean'], d['teacher_margin_mean'], d['pearson']))" 2>/dev/null
}
best_arm() {  # best_arm NAME...: the name with the highest tiny result (empty when none has one)
    local c v best="" best_v=""
    for c in "$@"; do
        v=$(tiny "$c"); [ -n "$v" ] || continue
        if [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; then best="$c"; best_v="$v"; fi
    done
    echo "$best"
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
log "=== v16 start: reference ${REF:-?}, base ${BASE:-?}, batch control $(tiny arm-mnrl-fiqa-ctrl-b1), own-b1 $(tiny arm-mnrl-fiqa-splade-own-b1), mix-b1 $(tiny arm-mnrl-fiqa-splade-mix-b1), $(free_gb) GB free on / ==="

log "=== v16 block L: leaderboard regime for n1-5070 ==="
[ -d models/n1-5070 ] && leaderboard n1-5070 || log "models/n1-5070 absent"

log "=== v16 block T1: KL on the 5 + 5 mix, promoted positives on mix2 ==="
arm_fit kl-fiqa-splade-mix-scaleauto
arm_fit mnrl-fiqa-splade-mix2-pp
[ "$C2_OK" = no ] && arm_fit mnrl-fiqa-splade-mix2
log "dose-response at 1 x 32: 0 $(tiny arm-mnrl-fiqa-ctrl-b1) | 2 $(tiny arm-mnrl-fiqa-splade-mix2-b1) | 5 $(tiny arm-mnrl-fiqa-splade-mix-b1) | 10 $(tiny arm-mnrl-fiqa-splade-own-b1)"
log "promoted positives: mix2-pp $(tiny arm-mnrl-fiqa-splade-mix2-pp-c2)$(tiny arm-mnrl-fiqa-splade-mix2-pp-b1) vs mix2 $(tiny arm-mnrl-fiqa-splade-mix2-c2)$(tiny arm-mnrl-fiqa-splade-mix2-b1) (5070 Ti: mix2 at 2 x 16); KL mix $(tiny arm-kl-fiqa-splade-mix-scaleauto-c2)$(tiny arm-kl-fiqa-splade-mix-scaleauto-b1)"

names=()
for a in mnrl-fiqa-splade-own-b1 mnrl-fiqa-splade-mix-b1 kl-fiqa-splade-mix-scaleauto-c2 kl-fiqa-splade-mix-scaleauto-b1 mnrl-fiqa-splade-mix2-pp-c2 mnrl-fiqa-splade-mix2-pp-b1 mnrl-fiqa-splade-mix2-c2 mnrl-fiqa-splade-mix2-b1; do
    [ -d "models/arm-$a" ] && names+=("arm-$a")
done
top=$(best_arm "${names[@]}")
log "=== v16 block H: best local arm on student negatives ${top:-none} (tiny $(tiny "${top:-x}")); reference ${REF:-?}, base ${BASE:-?} ==="
if [ -n "$top" ] && reaches "$top" "$BASE"; then
    holdout "$top" "models/$top"
    if reaches "$top" "$NEAR"; then
        leaderboard "$top"
        [ -d models/arm-mnrl-fiqa-ctrl-b1 ] && holdout arm-mnrl-fiqa-ctrl-b1 models/arm-mnrl-fiqa-ctrl-b1
    else log "leaderboard regime skipped: $top is more than 0.3 below $REF on tiny"; fi
    log "=== v16 block S: soups of $top ==="
    soup "soup-v5a" "models/$top" models/soup-v2b
    [ -d models/n1-5070 ] && soup "soup-v5b" "models/$top" models/soup-v2b models/n1-5070 || log "models/n1-5070 absent; no soup-v5b"
    for s in soup-v5a soup-v5b; do
        [ -f "results/$s/final/metrics.json" ] || continue
        if reaches "$s" "$REF"; then holdout "$s" "models/$s"; leaderboard "$s"; else log "  $s: tiny $(tiny "$s") below the reference $REF; no held-out"; fi
    done
else log "blocks H and S skipped: no local arm on student negatives reaches the base on tiny"; fi

"$PY" scripts/queue_summary.py > results/summary_v16.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v16.md)"
log "=== v16 done ==="
