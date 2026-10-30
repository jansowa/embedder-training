#!/usr/bin/env bash
# Night 30/31.10 on the 3060 Ti, unattended: waits for queue v22 to exit, no step starts after
# 31.10 07:30. Question of the night: our arms win some tasks and lose others (FiQA-style arms
# help arguana / fiqa-pl / scidocs and hurt maupqa / wiki-def / legal); can a soup or an
# anchor toward the base model keep the wins without the losses, and how much of the
# task-by-task movement is seed noise?
#   1  soups, leaderboard regime only (tiny10 + heldout8) plus sparsity:
#      soup-v9a  = n1 + n3 + n4 + long-easy + ldose2 (v7a kept, the dose arm added)
#      soup-v9b  = n1 + n3 + n4 + long-easy + lenmatch-far (v7a plus the best FiQA-only arm on tiny10)
#      soup-v9c  = v7a ingredients at 0.2125 each + base at 0.15 (anchor toward the base model)
#      soup-v9d  = v7a ingredients at 0.175 each + base at 0.30
#      Reference soup-v7a 63.54 / 42.53 (base 61.59 / 41.03).
#   2  n1-ldose2: the n1 recipe with the dose rows (FiQA ldose2 + SQuAD-v2) at 2 x 16: tiny,
#      sparsity, held-out-8, then the leaderboard regime when tiny >= n1's truncated tiny - 0.3.
#   3  seed1, seed2 of the reference recipe: tiny, sparsity, held-out-8 (noise floor per task).
#   4  held-out-2 (5 tasks) for the best new soup by leaderboard heldout8 when it is >= soup-v7a.
#   5  results/summary_v23.md with per-task tables and a per-task noise table; commit; push.
# A 2 x 16 training that fails (CUDA OOM) is retried once with its _cb32 config (same runs_dir).
# Every step is skipped when it already has a result.
# Start: setsid nohup bash scripts/queue_v23.sh > logs/queue_v23.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v23 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v23.pid"
guard_alive
DEADLINE=$(date -d "2026-10-31 07:30" +%s)
before_deadline() { [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }
C=configs/next/3060

# Queue v22 owns the GPU and the git index until it exits (its push continues in the background).
if [ -f "$LOGS/queue_v22.pid" ] && kill -0 "$(cat "$LOGS/queue_v22.pid")" 2>/dev/null; then
    log "waiting for queue v22 (pid $(cat "$LOGS/queue_v22.pid")) to exit"
    while kill -0 "$(cat "$LOGS/queue_v22.pid")" 2>/dev/null; do sleep 60; done
    sleep 30
fi

reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
lb() { "$PY" -c "import json; print(round(json.load(open('results-regimes/quantized/$1/$2.json'))['mean_ndcg@10'], 2))" 2>/dev/null; }
dims() { sparsity_mean "results/sparsity/$1.json" "$1"; }
merge_lb() {  # merge_lb NAME WEIGHTS MODEL...: weight-averaged soup into models/NAME (WEIGHTS empty = uniform), sparsity, leaderboard regime
    local name="$1" weights="$2"; shift 2
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        # shellcheck disable=SC2086
        "$PY" scripts/merge_splade_weights.py --models "$@" ${weights:+--weights $weights} --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $* ${weights:+(weights $weights)}" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    step sparsity "$name" "models/$name"; step leaderboard "$name"
    log "  $name leaderboard: $(lb "$name" tiny10) / $(lb "$name" heldout8), $(dims "$name") dims/doc (soup-v7a $(lb soup-v7a tiny10) / $(lb soup-v7a heldout8), 400; base $(lb base-splade-wrapped tiny10) / $(lb base-splade-wrapped heldout8))"
}
heldout2() {  # heldout2 NAME: the five held-out-2 tasks in the leaderboard regime
    local t; for t in czywiesz gpt35wiki poquad quora treccovid; do before_deadline || return 0; regime quantized "$1" "models/$1" "heldout2_$t" 16; done
    log "  $1 held-out-2: $("$PY" -c "
import json, glob
v = [json.load(open(f))['mean_ndcg@10'] for f in sorted(glob.glob('results-regimes/quantized/$1/heldout2_*.json'))]
print(f'{sum(v)/len(v):.2f} over {len(v)} tasks' if v else 'no result')") (soup-v7a 69.16, soup-v8a 69.29)"
}
train2() {  # train2 TAG CFG [FALLBACK]: train; when the 2 x 16 run fails, retry once with the cached-loss config
    local tag="$1" cfg="$2" fb="${3:-}"
    [ -d "models/arm-$tag" ] && return 0
    train "$tag" "$cfg" && return 0
    [ -d "models/arm-$tag" ] && return 0
    { [ -n "$fb" ] && [ -f "$fb" ]; } || return 1
    mv -f "$LOGS/arm_$tag.log" "$LOGS/arm_$tag.first.log" 2>/dev/null
    log "train arm-$tag: first attempt failed ($(grep -m1 -oE 'OutOfMemoryError|CUDA out of memory' "$LOGS/arm_$tag.first.log" 2>/dev/null || echo 'see logs/arm_'"$tag"'.first.log')); retry with $fb"
    train "$tag" "$fb"
}
arm() {  # arm TAG CFG [FALLBACK]: train, tiny, sparsity, held-out-8
    local tag="$1"
    step train2 "$tag" "$2" "${3:-}" || return 1
    step bench "arm-$tag" "models/arm-$tag"; step sparsity "arm-$tag" "models/arm-$tag"; step holdout "arm-$tag" "models/arm-$tag"
    log "  arm-$tag: tiny $(tiny "arm-$tag") held-out $(hold "arm-$tag") (cluster-like $REF / $(hold arm-cluster-like-fiqa), n1 $(tiny n1-5070) / $(hold n1-5070)), $(dims "arm-$tag") dims/doc"
}
best_by_lb() {  # best_by_lb NAME...: the name with the highest leaderboard heldout8 among those that have one
    local best="" bv="" m v
    for m in "$@"; do
        v=$(lb "$m" heldout8); [ -n "$v" ] || continue
        if [ -z "$bv" ] || [ "$(ge "$v" "$bv")" = 1 ]; then best="$m"; bv="$v"; fi
    done
    echo "$best"
}

REF=$(tiny arm-cluster-like-fiqa); BASE=$(tiny base-splade-wrapped)
log "=== v23 start: reference $REF, base $BASE, soup-v7a $(lb soup-v7a tiny10) / $(lb soup-v7a heldout8), $(free_gb) GB free ==="

log "=== v23 step 1: soups (v7a + dose arm, v7a + lenmatch-far, v7a anchored to base at 0.15 and 0.30) ==="
V7A="models/n1-5070 models/n3-5070 models/n4-5070 models/long-easy-5070"
# shellcheck disable=SC2086
step merge_lb soup-v9a "" $V7A models/arm-mnrl-fiqa-ldose2
# shellcheck disable=SC2086
step merge_lb soup-v9b "" $V7A models/arm-mnrl-fiqa-lenmatch-far
# shellcheck disable=SC2086
step merge_lb soup-v9c "0.2125 0.2125 0.2125 0.2125 0.15" $V7A models/base-splade-wrapped
# shellcheck disable=SC2086
step merge_lb soup-v9d "0.175 0.175 0.175 0.175 0.30" $V7A models/base-splade-wrapped

log "=== v23 step 2: n1-ldose2 (n1 recipe with the dose rows, 2 x 16) ==="
arm n1-ldose2 "$C/n1_ldose2.yaml" "$C/n1_ldose2_cb32.yaml"
m=arm-n1-ldose2
if [ -d "models/$m" ] && reaches "$m" "$(fsub "$(tiny n1-5070)" 0.3)"; then
    step leaderboard "$m"
    log "  $m leaderboard: $(lb "$m" tiny10) / $(lb "$m" heldout8) (n1 $(lb n1-5070 tiny10) / $(lb n1-5070 heldout8); ldose2 $(lb arm-mnrl-fiqa-ldose2 tiny10) / $(lb arm-mnrl-fiqa-ldose2 heldout8))"
elif [ -d "models/$m" ]; then
    log "  $m: tiny $(tiny "$m") more than 0.3 below n1 ($(tiny n1-5070)); no leaderboard regime"
fi

log "=== v23 step 3: seeds 1 and 2 of the reference recipe (noise floor) ==="
for s in 1 2; do arm "mnrl-fiqa-seed$s" "$C/mnrl_fiqa_seed$s.yaml" "$C/mnrl_fiqa_seed${s}_cb32.yaml"; done
log "  seeds: tiny $(tiny arm-mnrl-fiqa-seed1) / $(tiny arm-mnrl-fiqa-seed2) (seed 42: $REF), held-out $(hold arm-mnrl-fiqa-seed1) / $(hold arm-mnrl-fiqa-seed2) (seed 42: $(hold arm-cluster-like-fiqa))"

log "=== v23 step 4: held-out-2 for the best new soup ==="
BEST=$(best_by_lb soup-v9a soup-v9b soup-v9c soup-v9d)
a=$(lb "${BEST:-none}" heldout8); b=$(lb soup-v7a heldout8)
if [ -n "$BEST" ] && [ -n "$a" ] && [ -n "$b" ] && [ "$(ge "$a" "$b")" = 1 ]; then
    log "  best new soup by leaderboard heldout8: $BEST $a (soup-v7a $b)"; step heldout2 "$BEST"
else
    log "  no new soup reaches soup-v7a on the leaderboard heldout8 (best ${BEST:-none} ${a:-}); no held-out-2"
fi

log "=== v23 step 5: summary, commit, push ==="
"$PY" scripts/queue_summary.py > results/summary_v23.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v23.md)"
{
    echo; echo "## Per-task, leaderboard regime (queue v23)"; echo
    "$PY" scripts/regime_task_table.py --tasksets tiny10 heldout8 heldout2_treccovid \
        --models base-splade-wrapped arm-cluster-like-fiqa n1-5070 n4-5070 soup-v7a soup-v8a soup-v9a soup-v9b soup-v9c soup-v9d arm-n1-ldose2 arm-mnrl-fiqa-ldose2 arm-mnrl-fiqa-lenmatch-far 2>> "$LOGS/queue_summary.log"
    echo; echo "## Per-task noise of the reference recipe (truncated regime; seeds 42 = arm-cluster-like-fiqa, 1, 2)"; echo
    "$PY" scripts/seed_noise_table.py --models arm-cluster-like-fiqa arm-mnrl-fiqa-seed1 arm-mnrl-fiqa-seed2 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state"; echo; echo '```'; grep -E "^(train|bench|holdout|regime|sparsity):" "$STATE" | tail -60; echo '```'
} >> results/summary_v23.md
for p in results/arm-mnrl-fiqa-seed* results/arm-n1-ldose2 results-holdout/arm-mnrl-fiqa-seed* results-holdout/arm-n1-ldose2 results-regimes/quantized results/sparsity results/summary_v23.md; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v23.out"
done
if git commit -q -m "Add the v23 results: the anchored and extended soups, the dose inside the n1 recipe and the seed noise floor" 2>> "$LOGS/queue_v23.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v23.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v23.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v23.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v23.out)"
fi
log "=== v23 done ($(date '+%d.%m %H:%M')) ==="
