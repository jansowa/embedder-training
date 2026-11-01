#!/usr/bin/env bash
# Queue v26b (1.11.2026 evening, 3060 Ti, after v26): the n4 arms of the three cuts that failed the v26
# single-set screen (hotpotqa-1pos 58.22, nq-nocult 59.27, fever-short 58.18, threshold 60.0), measured
# anyway because the night is free: a set that hurts alone can still help inside fiqa + squadv2.
#   1  arm-n4x-nq-nocult, then the leaderboard regime of v25's arm-n4x-nq (the same pattern with the full
#      nq-2073, truncated 61.78 / 40.37, never measured in the leaderboard regime) as its direct control,
#      then arm-n4x-hotpotqa-1pos and arm-n4x-fever-short (cb32 fallback for fever-short only):
#      train at 2 x 16, sparsity, leaderboard tiny10 + heldout8. Controls arm-n4-3060 62.89 / 42.01,
#      n1-5070 62.98 / 42.17.
#   2  soups soup-v10-<set> = the five v9b ingredients + the arm, uniform, for every arm of step 1 with
#      leaderboard tiny10 >= arm-n4-3060 (62.89); held-out-2 (5 tasks) when heldout8 >= soup-v9b (42.56).
#   3  summary_v26b.md (all v26 / v26b arms and soups), commit, push.
# Every step is skipped when it already has a result. No step starts after V26B_DEADLINE (default
# 2.11.2026 07:30; set V26B_DEADLINE=none for no deadline).
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v26b.sh > logs/queue_v26b.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v26b MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v26b.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
FALLBACK_TIMEOUT=${FALLBACK_TIMEOUT:-16000}
V26B_DEADLINE=${V26B_DEADLINE:-2026-11-02 07:30}
DEADLINE=""
[ "$V26B_DEADLINE" != none ] && DEADLINE=$(date -d "$V26B_DEADLINE" +%s)
before_deadline() { [ -z "$DEADLINE" ] || [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }
V9B="models/n1-5070 models/n3-5070 models/n4-5070 models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"

# The previous queue owns the GPU and the git index until it exits (the pid must still be that queue).
for prev in v26; do
    pf="$LOGS/queue_$prev.pid"
    if [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$prev.sh"; then
        log "waiting for queue $prev (pid $(cat "$pf")) to exit"
        while ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$prev.sh"; do sleep 60; done
        sleep 30
    fi
done

lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1"; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
train2() {  # train2 TAG CFG [FALLBACK]: train; when the 2 x 16 run fails, retry once with the cached-loss config
    local tag="$1" cfg="$2" fb="${3:-}"
    [ -d "models/arm-$tag" ] && return 0
    train "$tag" "$cfg" && return 0
    [ -d "models/arm-$tag" ] && return 0
    { [ -n "$fb" ] && [ -f "$fb" ]; } || return 1
    mv -f "$LOGS/arm_$tag.log" "$LOGS/arm_$tag.first.log" 2>/dev/null
    log "train arm-$tag: first attempt failed ($(grep -m1 -oE 'OutOfMemoryError|CUDA out of memory' "$LOGS/arm_$tag.first.log" 2>/dev/null || echo 'see logs/arm_'"$tag"'.first.log')); retry with $fb (timeout ${FALLBACK_TIMEOUT}s)"
    TRAIN_TIMEOUT=$FALLBACK_TIMEOUT train "$tag" "$fb"
}
report() {  # report ARM: one log line with the leaderboard means next to the controls
    log "  $1 leaderboard: $(lbs "$1") (arm-n4-3060 $(lbs arm-n4-3060), n1-5070 $(lbs n1-5070), arm-n4-msclean $(lbs arm-n4-msclean)), $(dims "$1") dims/doc"
}
n4_arm() {  # n4_arm TAG CFG [FALLBACK]: train, sparsity, leaderboard regime (no truncated bench / held-out)
    local tag="$1"
    step train2 "$tag" "$2" "${3:-}" || return 1
    step sparsity "arm-$tag" "models/arm-$tag"; step leaderboard "arm-$tag"
    report "arm-$tag"
}
merge_lb() {  # merge_lb NAME MODEL...: uniform soup into models/NAME, sparsity, leaderboard regime
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    step sparsity "$name" "models/$name"; step leaderboard "$name"
    log "  $name leaderboard: $(lbs "$name"), $(dims "$name") dims/doc (soup-v9b $(lbs soup-v9b), $(dims soup-v9b) dims/doc)"
}
heldout2() {  # heldout2 NAME: the five held-out-2 tasks in the leaderboard regime
    local t; for t in czywiesz gpt35wiki poquad quora treccovid; do step regime quantized "$1" "models/$1" "heldout2_$t" 16; done
    log "  $1 held-out-2: $("$PY" -c "
import json, glob, sys
v = [json.load(open(f))['mean_ndcg@10'] for f in sorted(glob.glob(sys.argv[1] + '/heldout2_*.json'))]
print(f'{sum(v)/len(v):.2f} over {len(v)} tasks' if v else 'no result')" "$REGIMES_ROOT/quantized/$1") (soup-v9b 69.42)"
}

log "=== v26b start: arm-n4-3060 $(lbs arm-n4-3060), n1-5070 $(lbs n1-5070), soup-v9b $(lbs soup-v9b), arm-n4-msclean $(lbs arm-n4-msclean), deadline $V26B_DEADLINE, $(free_gb) GB free ==="

log "=== v26b step 1: n4 arms of the cuts that failed the screen, and the leaderboard regime of arm-n4x-nq ==="
n4_arm n4x-nq-nocult "$C/n4x_nq_nocult.yaml"
[ -d models/arm-n4x-nq ] && { step leaderboard arm-n4x-nq; report arm-n4x-nq; }
n4_arm n4x-hotpotqa-1pos "$C/n4x_hotpotqa_1pos.yaml"
n4_arm n4x-fever-short "$C/n4x_fever_short.yaml" "$C/n4x_fever_short_cb32.yaml"

log "=== v26b step 2: soups v9b + arm for the arms with leaderboard tiny10 >= arm-n4-3060 ==="
CTRL=$(lbv arm-n4-3060 tiny10); CTRL=${CTRL:-62.89}
V9B_H8=$(lbv soup-v9b heldout8); V9B_H8=${V9B_H8:-42.56}
ARMS=(arm-n4x-nq-nocult arm-n4x-nq arm-n4x-hotpotqa-1pos arm-n4x-fever-short)
SOUPS=()
for m in "${ARMS[@]}"; do
    [ -d "models/$m" ] || continue
    v=$(lbv "$m" tiny10)
    if [ -n "$v" ] && [ "$(ge "$v" "$CTRL")" = 1 ]; then
        name="soup-v10-${m#arm-n4x-}"
        # shellcheck disable=SC2086
        step merge_lb "$name" $V9B "models/$m" && SOUPS+=("$name")
    else
        log "  $m: leaderboard tiny10 ${v:-missing} below arm-n4-3060 $CTRL; no soup"
    fi
done
for name in "${SOUPS[@]}"; do
    h=$(lbv "$name" heldout8)
    if [ -n "$h" ] && [ "$(ge "$h" "$V9B_H8")" = 1 ]; then
        log "  $name heldout8 $h >= soup-v9b $V9B_H8: held-out-2"; heldout2 "$name"
    else
        log "  $name heldout8 ${h:-missing} below soup-v9b $V9B_H8; no held-out-2"
    fi
done

log "=== v26b step 3: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v26b.md"
ALL=(arm-n4-msclean "${ARMS[@]}")
for s in soup-v10-msclean "${SOUPS[@]}"; do [ -d "models/$s" ] && ALL+=("$s"); done
"$PY" scripts/queue_summary.py > "$SUM" 2>> "$LOGS/queue_summary.log" && log "summary written ($SUM)"
{
    echo; echo "## Per-task, leaderboard regime (queues v26 + v26b)"; echo
    "$PY" scripts/regime_task_table.py --tasksets tiny10 heldout8 heldout2_treccovid \
        --models base-splade-wrapped n1-5070 n4-5070 soup-v9b arm-n4-3060 "${ALL[@]}" 2>> "$LOGS/queue_summary.log"
    echo; echo "## Per-task, truncated regime: v26 single-set screens vs the predecessors"; echo
    "$PY" scripts/seed_noise_table.py --models arm-mnrl-hotpotqa-1pos-2073 arm-mnrl-nq-nocult-2073 arm-mnrl-fever-short-2073 \
        arm-mnrl-msmarco-nofn arm-mnrl-hotpotqa-2073 arm-mnrl-nq-2073 arm-mnrl-fever-2073 arm-msmarco-clean-2073 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state"; echo; echo '```'; grep -E "^(train|bench|holdout|sparsity|regime):" "$STATE" | tail -40; echo '```'
} >> "$SUM"
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v26b.out"
done
if git commit -q -m "Add the v26b results" 2>> "$LOGS/queue_v26b.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v26b.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v26b.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v26b.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v26b.out)"
fi
log "=== v26b done ($(date '+%d.%m %H:%M')) ==="
