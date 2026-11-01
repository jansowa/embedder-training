#!/usr/bin/env bash
# Queue v27 (planned 1.11.2026 23:00, reordered 23:40; 3060 Ti, after v26b; deadline 2.11 20:00).
# Order: 1 trivia screen, 2 n4-msrand, 4 SQuAD dose ladder + its soups, 1b n4x-trivia, 3 n4 seeds,
# then the gated soups. Four open questions:
#   1  trivia by content or by dose? trivia-2073 (55.15 truncated tiny) trained on ~18.9k examples
#      (~9 gated positives per row), 9 x every other single-set arm. Screen trivia-1pos-2073 (same rows,
#      best positive only) at 2 x 16: truncated tiny + sparsity, threshold SINGLE_MIN 60.0; the n4 arm
#      arm-n4x-trivia-1pos (fiqa + squadv2 + trivia-1pos) only when the screen passes.
#   2  do msmarco's LLM false negatives hurt? arm-n4-msclean (no false-negative rows, 432 steps) tied with
#      arm-n4-3060 (456 steps). arm-n4-msrand drops 385 random msmarco rows instead (432 steps, keeps 316
#      of the 385 false-negative rows): msclean > msrand means the false negatives hurt.
#   3  how noisy is the n4 control in the leaderboard regime, and does a seed soup help? Seeds 1 and 2 of
#      arm-n4-3060 (seed 42), then soup-n4seeds = the three, uniform. Every arm of v25-v27 was compared
#      with the single control 62.89 / 42.01; the seeds give the band.
#   4  does more SQuAD keep paying? arm-n5-squad16k = fiqa-2073 + 16000 fresh gated SQuAD-v2 rows (n3-5070's
#      8000 + the next 8000 of the same draw; 18073 rows, ~1200 steps, TRAIN_TIMEOUT 14400). n3 (8000) is
#      a v9b ingredient although its own leaderboard tiny10 (62.72) is below the n4 control, so n5 is
#      souped regardless of its own score: soup-v10-squad16k = v9b + n5, soup-v9b-n5for3 = v9b with n5 in
#      place of n3; held-out-2 for either when heldout8 >= soup-v9b.
# Every new arm: train, sparsity, leaderboard tiny10 + heldout8 (no truncated bench / held-out except the
# screen). Soups soup-v10-<x> = the five v9b ingredients + x for every x of steps 1-3 with leaderboard
# tiny10 >= arm-n4-3060 (62.89); held-out-2 when heldout8 >= soup-v9b (42.56). Then summary_v27.md,
# commit, push. Every step is skipped when it already has a result. No step starts after V27_DEADLINE
# (default 2.11.2026 20:00; V27_DEADLINE=none for no deadline).
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v27.sh > logs/queue_v27.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v27 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v27.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SINGLE_MIN=${SINGLE_MIN:-60.0}
V27_DEADLINE=${V27_DEADLINE:-2026-11-02 20:00}
DEADLINE=""
[ "$V27_DEADLINE" != none ] && DEADLINE=$(date -d "$V27_DEADLINE" +%s)
before_deadline() { [ -z "$DEADLINE" ] || [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }
V9B="models/n1-5070 models/n3-5070 models/n4-5070 models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"

# The previous queues own the GPU and the git index until they exit (the pid must still be that queue).
for prev in v26b v26; do
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
report() {  # report MODEL: leaderboard means next to the controls
    log "  $1 leaderboard: $(lbs "$1") (arm-n4-3060 $(lbs arm-n4-3060), arm-n4-msclean $(lbs arm-n4-msclean), n1-5070 $(lbs n1-5070)), $(dims "$1") dims/doc"
}
n4_arm() {  # n4_arm TAG CFG: train, sparsity, leaderboard regime
    local tag="$1"
    step train "$tag" "$2" || return 1
    step sparsity "arm-$tag" "models/arm-$tag"; step leaderboard "arm-$tag"
    report "arm-$tag"
}
merge_lb() {  # merge_lb NAME MODEL...: uniform soup into models/NAME, sparsity, leaderboard regime
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        before_deadline || { log "deadline reached; skipping soup $name"; return 1; }
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

log "=== v27 start: arm-n4-3060 $(lbs arm-n4-3060), arm-n4-msclean $(lbs arm-n4-msclean), soup-v9b $(lbs soup-v9b), deadline $V27_DEADLINE, $(free_gb) GB free ==="

log "=== v27 step 1a: screen trivia-1pos-2073 (dose question) ==="
if step train mnrl-trivia-1pos-2073 "$C/mnrl_trivia_1pos_2073.yaml"; then
    step bench arm-mnrl-trivia-1pos-2073 models/arm-mnrl-trivia-1pos-2073
    step sparsity arm-mnrl-trivia-1pos-2073 models/arm-mnrl-trivia-1pos-2073
fi
TRIVIA=$(tiny arm-mnrl-trivia-1pos-2073)
log "  arm-mnrl-trivia-1pos-2073: tiny ${TRIVIA:-missing} (arm-trivia-2073 $(tiny arm-trivia-2073) at ~9 x the dose, base $(tiny base-splade-wrapped), threshold $SINGLE_MIN), $(dims arm-mnrl-trivia-1pos-2073) dims/doc"

log "=== v27 step 2: arm-n4-msrand (equal-step control of n4-msclean) ==="
n4_arm n4-msrand "$C/n4_msrand.yaml"

CTRL=$(lbv arm-n4-3060 tiny10); CTRL=${CTRL:-62.89}
V9B_H8=$(lbv soup-v9b heldout8); V9B_H8=${V9B_H8:-42.56}
soup_h2() {  # soup_h2 NAME MODEL...: soup, leaderboard, held-out-2 when heldout8 >= soup-v9b
    local name="$1" h
    merge_lb "$@" || return 1
    h=$(lbv "$name" heldout8)
    if [ -n "$h" ] && [ "$(ge "$h" "$V9B_H8")" = 1 ]; then
        log "  $name heldout8 $h >= soup-v9b $V9B_H8: held-out-2"; heldout2 "$name"
    else
        log "  $name heldout8 ${h:-missing} below soup-v9b $V9B_H8; no held-out-2"
    fi
}

log "=== v27 step 4: SQuAD dose ladder, arm-n5-squad16k and its soups ==="
TRAIN_TIMEOUT=14400 n4_arm n5-squad16k "$C/n5_squad16k.yaml"
log "  ladder (leaderboard): n1-5070 $(lbs n1-5070) (squad 2073), n3-5070 $(lbs n3-5070) (8000), arm-n5-squad16k $(lbs arm-n5-squad16k) (16000)"
if [ -d models/arm-n5-squad16k ]; then
    # shellcheck disable=SC2086
    soup_h2 soup-v10-squad16k $V9B models/arm-n5-squad16k
    soup_h2 soup-v9b-n5for3 models/n1-5070 models/arm-n5-squad16k models/n4-5070 models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far
fi

log "=== v27 step 1b: n4 arm of trivia-1pos when the screen passed ==="
CANDS=(arm-n4-msrand)
if [ -n "$TRIVIA" ] && [ "$(ge "$TRIVIA" "$SINGLE_MIN")" = 1 ]; then
    n4_arm n4x-trivia-1pos "$C/n4x_trivia_1pos.yaml" && CANDS+=(arm-n4x-trivia-1pos)
else
    log "  trivia-1pos ${TRIVIA:-missing} below $SINGLE_MIN; no n4x-trivia-1pos arm"
fi

log "=== v27 step 3: seeds 1 and 2 of arm-n4-3060, then the seed soup ==="
n4_arm n4-3060-seed1 "$C/n4_3060_seed1.yaml"
n4_arm n4-3060-seed2 "$C/n4_3060_seed2.yaml"
if [ -d models/arm-n4-3060-seed1 ] && [ -d models/arm-n4-3060-seed2 ]; then
    "$PY" - "$REGIMES_ROOT/quantized" 2>/dev/null <<'EOF' | while read -r line; do log "  $line"; done
import json, statistics, sys
for ts in ("tiny10", "heldout8"):
    v = [json.load(open(f"{sys.argv[1]}/{m}/{ts}.json"))["mean_ndcg@10"] for m in ("arm-n4-3060", "arm-n4-3060-seed1", "arm-n4-3060-seed2")]
    print(f"n4-3060 seeds {ts}: {' / '.join(f'{x:.2f}' for x in v)}, mean {statistics.mean(v):.2f}, std {statistics.stdev(v):.2f}, range {max(v) - min(v):.2f}")
EOF
    merge_lb soup-n4seeds models/arm-n4-3060 models/arm-n4-3060-seed1 models/arm-n4-3060-seed2 && CANDS+=(soup-n4seeds)
fi

log "=== v27 step 5: soups v9b + x for the candidates with leaderboard tiny10 >= arm-n4-3060 ==="
SOUPS=()
for m in "${CANDS[@]}"; do
    [ -d "models/$m" ] || continue
    v=$(lbv "$m" tiny10)
    if [ -n "$v" ] && [ "$(ge "$v" "$CTRL")" = 1 ]; then
        name="soup-v10-${m#arm-}"; name=${name/soup-v10-n4x-/soup-v10-}; name=${name/soup-v10-n4-/soup-v10-}; name=${name/soup-v10-soup-/soup-v10-}
        # shellcheck disable=SC2086
        soup_h2 "$name" $V9B "models/$m"; SOUPS+=("$name")
    else
        log "  $m: leaderboard tiny10 ${v:-missing} below arm-n4-3060 $CTRL; no soup"
    fi
done

log "=== v27 step 6: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v27.md"
mkdir -p "$RESULTS_ROOT/data/v27-cuts"
for f in trivia-1pos-2073 msmarco-clean-rand1688 fiqa-squadv2-trivia-1pos-6219 fiqa-squadv2-msmarco-rand-5834 \
         squadv2-fresh-16000 fiqa-plus-squadv2-18k; do
    cp -f "datasets/arms/$f.jsonl.manifest.json" "$RESULTS_ROOT/data/v27-cuts/$f.manifest.json" 2>/dev/null
done
ALL=(arm-n4-msclean arm-n4-msrand n3-5070 arm-n5-squad16k arm-n4-3060-seed1 arm-n4-3060-seed2)
for m in soup-v10-squad16k soup-v9b-n5for3 arm-n4x-trivia-1pos soup-n4seeds "${SOUPS[@]}"; do [ -d "models/$m" ] && ALL+=("$m"); done
"$PY" scripts/queue_summary.py > "$SUM" 2>> "$LOGS/queue_summary.log" && log "summary written ($SUM)"
{
    echo; echo "## Per-task, leaderboard regime (queue v27)"; echo
    "$PY" scripts/regime_task_table.py --tasksets tiny10 heldout8 heldout2_treccovid \
        --models base-splade-wrapped n1-5070 n4-5070 soup-v9b arm-n4-3060 "${ALL[@]}" 2>> "$LOGS/queue_summary.log"
    echo; echo "## Per-task, truncated regime: trivia at one positive per row vs the full-dose trivia arm"; echo
    "$PY" scripts/seed_noise_table.py --models arm-mnrl-trivia-1pos-2073 arm-trivia-2073 arm-squadv2-2073 arm-cluster-like-fiqa 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state"; echo; echo '```'; grep -E "^(train|bench|holdout|sparsity|regime):" "$STATE" | tail -40; echo '```'
} >> "$SUM"
for p in "$RESULTS_ROOT/arm-mnrl-trivia-1pos-2073" "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$RESULTS_ROOT/data/v27-cuts" "$SUM"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v27.out"
done
if git commit -q -m "Add the v27 results" 2>> "$LOGS/queue_v27.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v27.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v27.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v27.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v27.out)"
fi
log "=== v27 done ($(date '+%d.%m %H:%M')) ==="
