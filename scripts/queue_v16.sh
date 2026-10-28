#!/usr/bin/env bash
# 28.10, replaces the tail of queue_v15 after its mix-b1 arm. v15 showed that ten
# student-mined negatives per query cost 2.7 points on tiny (arm-mnrl-fiqa-splade-own-b1
# 59.53 vs arm-cluster-like-fiqa 62.27, uniformly over the 10 tasks, 369 active dims/doc
# vs 436; the final MNRL loss stays 4x higher than with the old negatives). The remaining
# v15 arms (kl on the pure student file, pure plus, pure own-f02) would repeat that, so this
# queue takes over with a dose-response on the share of student negatives per query:
#   0  arm-cluster-like-fiqa (old negatives only)  2  mix2-b1  5  mix-b1 (v15)  10  own-b1 (v15)
# and the user's combined-corpus experiment in the same 5 + 5 form (plusmix-b1), with the
# pure plus-b1 arm last. Blocks:
#   W   wait for the tiny result of arm-mnrl-fiqa-splade-mix-b1 in logs/queue_v15.out, then
#       stop queue v15 (its pid file) together with the training it has just started
#   T1  mix2-b1, plusmix-b1, plus-b1
#   H   held-out-8 for the best arm on student negatives when it reaches the base on tiny;
#       leaderboard regime when within 0.3 of the reference; ctrl-b1 on held-out then too
#   S   soups soup-v5a (best + soup-v2b) and soup-v5b (+ n1-5070) when the best reaches the base
# Start with: setsid nohup bash scripts/queue_v16.sh > logs/queue_v16.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v16 MIN_RAM_GB=6
export WANDB_MODE=offline
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

arm() {  # arm NAME: train configs/next/3060/<NAME with _>.yaml, tiny, sparsity
    local cfg="configs/next/3060/${1//-/_}.yaml"
    train "$1" "$cfg" && { bench "arm-$1" "models/arm-$1"; sparsity "arm-$1" "models/arm-$1"; }
    log "  arm-$1: tiny $(tiny "arm-$1")"
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

log "=== v16 block T1: milder doses of student negatives ==="
for a in mnrl-fiqa-splade-mix2-b1 mnrl-fiqa-splade-plusmix-b1 mnrl-fiqa-splade-plus-b1; do arm "$a"; done
log "dose-response (student negatives per query -> tiny): 0 ${REF:-?} | 2 $(tiny arm-mnrl-fiqa-splade-mix2-b1) | 5 $(tiny arm-mnrl-fiqa-splade-mix-b1) | 10 $(tiny arm-mnrl-fiqa-splade-own-b1) || plus 5+5 $(tiny arm-mnrl-fiqa-splade-plusmix-b1), plus 10 $(tiny arm-mnrl-fiqa-splade-plus-b1)"

names=(arm-mnrl-fiqa-splade-own-b1 arm-mnrl-fiqa-splade-mix-b1 arm-mnrl-fiqa-splade-mix2-b1 arm-mnrl-fiqa-splade-plusmix-b1 arm-mnrl-fiqa-splade-plus-b1)
top=$(best_arm "${names[@]}")
log "=== v16 block H: best arm on student negatives ${top:-none} (tiny $(tiny "${top:-x}")); reference ${REF:-?}, base ${BASE:-?} ==="
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
else log "blocks H and S skipped: no arm on student negatives reaches the base on tiny"; fi

"$PY" scripts/queue_summary.py > results/summary_v16.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v16.md)"
log "=== v16 done ==="
