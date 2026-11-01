#!/usr/bin/env bash
# Queue v26 (planned 1.11.2026 evening, 3060 Ti): do the LLM-annotation-gated cuts turn the speakleash
# sets into components that lift n4 and soup-v9b (63.65 / 42.56 leaderboard)?
# Data (kitchen ~/work/speakleash/retrieval-data-kitchen, cut/recipes/ + cut/mix.py, analysis in
# docs/annotation-analysis-2026-11-01.md): hotpotqa-1pos-2073 (one relevance-3 positive, no bridges),
# nq-nocult-rel3-2073 (no culture_entertainment / sport), fever-short-rel3-2073 (positives <= 256
# tokens so it fits 2 x 16), msmarco-clean-nofn (1688 rows, no LLM false negatives); every cut also
# drops rows with a negative of relevance >= 2. Configs in configs/next/3060/.
#   1  single-set screen, order hotpotqa-1pos, nq-nocult, fever-short, msmarco-nofn: train at 2 x 16
#      (fever-short retried once with its _cb32 config on failure), truncated tiny, sparsity.
#      Threshold SINGLE_MIN 60.0 (base 60.14; predecessors hotpotqa-2073 57.73, nq-2073 60.37,
#      fever-2073 60.39, msmarco-clean-2073 60.49).
#   2  n4 pattern fiqa-2073 + squadv2-2073 + <cut> (arm-n4x-<set>) for hotpotqa-1pos, nq-nocult and
#      fever-short when their screen passed, and arm-n4-msclean (msmarco-clean-nofn in place of
#      msmarco-clean-2073) always: train, sparsity, leaderboard regime tiny10 + heldout8 only (no
#      truncated bench / held-out). Controls arm-n4-3060 62.89 / 42.01, n1-5070 62.98 / 42.17.
#   3  soups soup-v10-<set> = the five v9b ingredients + the arm, uniform, for every arm with
#      leaderboard tiny10 >= arm-n4-3060 (62.89): sparsity, tiny10, heldout8; held-out-2 (5 tasks)
#      for a soup whose heldout8 reaches soup-v9b (42.56).
#   4  summary_v26.md (queue summary, per-task tables), commit of the result paths, push.
# Every step is skipped when it already has a result, so a killed queue is resumed by starting it
# again. Optional deadline: V26_DEADLINE="2026-11-02 10:00" (no step starts after it); default none.
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v26.sh > logs/queue_v26.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v26 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v26.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SINGLE_MIN=${SINGLE_MIN:-60.0}
FALLBACK_TIMEOUT=${FALLBACK_TIMEOUT:-16000}   # a cb32 run needs more than the 7200 s default (fever n4x cb32: 3.7 h)
DEADLINE=""
[ -n "${V26_DEADLINE:-}" ] && DEADLINE=$(date -d "$V26_DEADLINE" +%s)
before_deadline() { [ -z "$DEADLINE" ] || [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }
V9B="models/n1-5070 models/n3-5070 models/n4-5070 models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"

# The previous queue owns the GPU and the git index until it exits (the pid must still be that queue,
# not a reused pid after a reboot).
for prev in v25; do
    pf="$LOGS/queue_$prev.pid"
    if [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$prev"; then
        log "waiting for queue $prev (pid $(cat "$pf")) to exit"
        while ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$prev"; do sleep 60; done
        sleep 30
    fi
done

lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }          # leaderboard mean only
lbs() { echo "$(regime_mean "$REGIMES_ROOT/quantized/$1/tiny10.json" | awk '{print $1}') / $(regime_mean "$REGIMES_ROOT/quantized/$1/heldout8.json" | awk '{print $1}')"; }
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

# set -> screen tag, screen config, predecessor arm, n4 arm tag, n4 config
declare -A SCREEN_TAG=( [hotpotqa-1pos]=mnrl-hotpotqa-1pos-2073 [nq-nocult]=mnrl-nq-nocult-2073
                        [fever-short]=mnrl-fever-short-2073 [msmarco-nofn]=mnrl-msmarco-nofn )
declare -A SCREEN_CFG=( [hotpotqa-1pos]=mnrl_hotpotqa_1pos_2073 [nq-nocult]=mnrl_nq_nocult_2073
                        [fever-short]=mnrl_fever_short_2073 [msmarco-nofn]=mnrl_msmarco_nofn )
declare -A PREV=( [hotpotqa-1pos]=arm-mnrl-hotpotqa-2073 [nq-nocult]=arm-mnrl-nq-2073
                  [fever-short]=arm-mnrl-fever-2073 [msmarco-nofn]=arm-msmarco-clean-2073 )
declare -A N4_TAG=( [hotpotqa-1pos]=n4x-hotpotqa-1pos [nq-nocult]=n4x-nq-nocult
                    [fever-short]=n4x-fever-short [msmarco-nofn]=n4-msclean )
declare -A N4_CFG=( [hotpotqa-1pos]=n4x_hotpotqa_1pos [nq-nocult]=n4x_nq_nocult
                    [fever-short]=n4x_fever_short [msmarco-nofn]=n4_msclean )
ORDER=(hotpotqa-1pos nq-nocult fever-short msmarco-nofn)
fallback() { [ "$1" = fever-short ] && echo "$C/$2_cb32.yaml" || echo ""; }   # cb32 only for fever-short

screen_set() {  # screen_set SET: the single-set arm at 2 x 16, truncated tiny + sparsity
    local s="$1" tag="${SCREEN_TAG[$1]}"
    step train2 "$tag" "$C/${SCREEN_CFG[$s]}.yaml" "$(fallback "$s" "${SCREEN_CFG[$s]}")" || return 1
    step bench "arm-$tag" "models/arm-$tag"; step sparsity "arm-$tag" "models/arm-$tag"
    log "  arm-$tag: tiny $(tiny "arm-$tag") (predecessor ${PREV[$s]} $(tiny "${PREV[$s]}"), base $(tiny base-splade-wrapped), threshold $SINGLE_MIN), $(dims "arm-$tag") dims/doc"
}
passed() { local v; v=$(tiny "arm-${SCREEN_TAG[$1]}"); [ -n "$v" ] && [ "$(ge "$v" "$SINGLE_MIN")" = 1 ]; }
n4_arm() {  # n4_arm SET: train, sparsity, leaderboard regime (no truncated bench / held-out)
    local s="$1" tag="${N4_TAG[$1]}"
    step train2 "$tag" "$C/${N4_CFG[$s]}.yaml" "$(fallback "$s" "${N4_CFG[$s]}")" || return 1
    step sparsity "arm-$tag" "models/arm-$tag"; step leaderboard "arm-$tag"
    log "  arm-$tag leaderboard: $(regime_mean "$REGIMES_ROOT/quantized/arm-$tag/tiny10.json") / $(regime_mean "$REGIMES_ROOT/quantized/arm-$tag/heldout8.json") (mean n_tasks; arm-n4-3060 $(lbs arm-n4-3060), n1-5070 $(lbs n1-5070)), $(dims "arm-$tag") dims/doc"
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

log "=== v26 start: base $(tiny base-splade-wrapped), arm-n4-3060 $(lbs arm-n4-3060), n1-5070 $(lbs n1-5070), soup-v9b $(lbs soup-v9b), deadline ${V26_DEADLINE:-none}, $(free_gb) GB free ==="

log "=== v26 step 1: single-set screen (${ORDER[*]}), threshold $SINGLE_MIN ==="
for s in "${ORDER[@]}"; do screen_set "$s"; done

log "=== v26 step 2: n4 pattern for the sets that passed (hotpotqa-1pos, nq-nocult, fever-short), n4-msclean always ==="
ARMS=()
for s in "${ORDER[@]}"; do
    if [ "$s" = msmarco-nofn ] || passed "$s"; then
        n4_arm "$s"
        [ -d "models/arm-${N4_TAG[$s]}" ] && ARMS+=("arm-${N4_TAG[$s]}")
    else
        log "  $s: single arm $(tiny "arm-${SCREEN_TAG[$s]}") below $SINGLE_MIN (or missing); no ${N4_TAG[$s]} arm"
    fi
done

log "=== v26 step 3: soups v9b + arm for the arms with leaderboard tiny10 >= arm-n4-3060 ==="
CTRL=$(lbv arm-n4-3060 tiny10); CTRL=${CTRL:-62.89}
V9B_H8=$(lbv soup-v9b heldout8); V9B_H8=${V9B_H8:-42.56}
SOUPS=()
for m in "${ARMS[@]}"; do
    v=$(lbv "$m" tiny10)
    if [ -n "$v" ] && [ "$(ge "$v" "$CTRL")" = 1 ]; then
        name="soup-v10-${m#arm-}"; name=${name/soup-v10-n4x-/soup-v10-}; name=${name/soup-v10-n4-/soup-v10-}
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

log "=== v26 step 4: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v26.md"
mkdir -p "$RESULTS_ROOT/data/v26-cuts"
for f in hotpotqa-1pos-2073 nq-nocult-rel3-2073 fever-short-rel3-2073 msmarco-clean-nofn \
         fiqa-squadv2-hotpotqa-1pos-6219 fiqa-squadv2-nq-nocult-6219 fiqa-squadv2-fever-short-6219 fiqa-squadv2-msmarco-nofn-5834; do
    cp -f "datasets/arms/$f.jsonl.manifest.json" "$RESULTS_ROOT/data/v26-cuts/$f.manifest.json" 2>/dev/null
done
SCREENS=(); for s in "${ORDER[@]}"; do SCREENS+=("arm-${SCREEN_TAG[$s]}"); done
"$PY" scripts/queue_summary.py > "$SUM" 2>> "$LOGS/queue_summary.log" && log "summary written ($SUM)"
{
    echo; echo "## Per-task, leaderboard regime (queue v26)"; echo
    "$PY" scripts/regime_task_table.py --tasksets tiny10 heldout8 heldout2_treccovid \
        --models base-splade-wrapped n1-5070 n4-5070 soup-v9b arm-n4-3060 "${ARMS[@]}" "${SOUPS[@]}" 2>> "$LOGS/queue_summary.log"
    echo; echo "## Per-task, truncated regime: single-set screen vs the predecessors"; echo
    "$PY" scripts/seed_noise_table.py --models "${SCREENS[@]}" arm-mnrl-hotpotqa-2073 arm-mnrl-nq-2073 arm-mnrl-fever-2073 arm-msmarco-clean-2073 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state"; echo; echo '```'; grep -E "^(train|bench|holdout|sparsity|regime):" "$STATE" | tail -40; echo '```'
} >> "$SUM"
for p in "${SCREENS[@]/#/$RESULTS_ROOT/}" "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$RESULTS_ROOT/data/v26-cuts" "$SUM"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v26.out"
done
if git commit -q -m "Add the v26 results" 2>> "$LOGS/queue_v26.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v26.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v26.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v26.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v26.out)"
fi
log "=== v26 done ($(date '+%d.%m %H:%M')) ==="
