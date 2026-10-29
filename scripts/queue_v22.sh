#!/usr/bin/env bash
# Night 29/30.10 and the day of 30.10 on the 3060 Ti, unattended: no step starts after 30.10
# 19:45, everything is done by 21:00. Plan: docs/PLAN-2026-10-29.md §5 (items 1 and 4) plus the
# length question of the student-mined negatives (§9 of the 27.10 plan).
#   1  ctrl-cb32: the cluster-like rows at a TRUE batch of 32 queries via CachedSpladeLoss
#      (GradCache, mini-batch 4) instead of 2 x 16 accumulation: tiny, sparsity, diagnostic.
#      Reference arm-cluster-like-fiqa 62.27 tiny / 40.17 held-out / 396 dims per document.
#   2  ldose1/2/4: 1, 2, 4 long-easy + 9, 8, 6 old v3 negatives per query (moderate length
#      correction; long-easy-5070 alone: arguana +10, fiqa-pl -8). tiny, sparsity, length
#      diagnostic; held-out-8 for every arm at tiny >= 61.5; the best by held-out gets the
#      leaderboard regime (tiny10 + heldout8) and trec-covid.
#   3  ctrl-cb32 held-out-8 when within 0.3 of the reference on tiny.
#   4  lenmatch-hard, lenmatch-far, lenmatch-hard-mix2: student negatives drawn inside the
#      length window 0.7-1.4 x the positives (hard vs long): tiny, sparsity, diagnostic;
#      held-out-8 when within 0.3 of the reference.
#   5  soup-v8a = n1 + n3 + n4 + best dose arm (soup-v7a with long-easy replaced): tiny,
#      held-out, leaderboard; held-out-2 when its leaderboard heldout8 >= soup-v7a (42.53).
#   6  fillers: leaderboard regime for ctrl-cb32 and the best lenmatch arm; soup-v8b (v7a's
#      recipe with long-easy at weight 0.125): tiny, held-out, leaderboard.
#   7  results/summary_v22.md with per-task tables and the length diagnostics; commit; push.
# A 2 x 16 training that fails (CUDA OOM on long negatives) is retried once with its _cb32
# config (same runs_dir). Every step is skipped when it already has a result.
# Start: setsid nohup bash scripts/queue_v22.sh > logs/queue_v22.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v22 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v22.pid"
guard_alive
DEADLINE=$(date -d "2026-10-30 19:45" +%s)
before_deadline() { [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }
C=configs/next/3060

reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
lb() { "$PY" -c "import json; print(round(json.load(open('results-regimes/quantized/$1/$2.json'))['mean_ndcg@10'], 2))" 2>/dev/null; }
dims() { sparsity_mean "results/sparsity/$1.json" "$1"; }
merge() {  # merge NAME WEIGHTS MODEL...: weight-averaged soup into models/NAME (WEIGHTS empty = uniform), then tiny + sparsity
    local name="$1" weights="$2"; shift 2
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        # shellcheck disable=SC2086
        "$PY" scripts/merge_splade_weights.py --models "$@" ${weights:+--weights $weights} --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $* ${weights:+(weights $weights)}" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    bench "$name" "models/$name"; sparsity "$name" "models/$name"
}
diag() {  # diag NAME: the length-bias diagnostic of models/NAME on the own FiQA corpus
    local out="datasets/arms/diag-$1"
    [ -f "results/data/length-bias-$1.json" ] && return 0
    [ -d "models/$1" ] || return 1
    wait_gpu_free; wait_ram
    timeout -k 60 1800 "$PY" scripts/fiqa_splade_mine.py --source-dir datasets/arms/fiqa-v3-mine --out-dir "$out" --model "models/$1" --top-k 100 > "$LOGS/diag_$1.log" 2>&1 \
        && "$PY" scripts/fiqa_splade_length_bias.py --mine-dir "$out" --model models/base-splade-wrapped --report "results/data/length-bias-$1.json" >> "$LOGS/diag_$1.log" 2>&1 \
        && log "diag $1: $(diag_line "$1")" \
        || log "diag $1 FAILED (logs/diag_$1.log)"
    rm -rf "$out"
}
diag_line() {
    "$PY" -c "
import json; r = json.load(open('results/data/length-bias-$1.json')); b = r['positives_by_token_bucket']
print('positive first by length ' + ', '.join(f\"{k} {v['positive_first']:.2f}\" for k, v in b.items()) + f\"; top-1 over 512 tokens {r['student_top1']['share_over_512']:.0%}\")" 2>/dev/null
}
heldout2() {  # heldout2 NAME: the five held-out-2 tasks in the leaderboard regime
    local t; for t in czywiesz gpt35wiki poquad quora treccovid; do before_deadline || return 0; regime quantized "$1" "models/$1" "heldout2_$t" 16; done
    log "  $1 held-out-2: $("$PY" -c "
import json, glob
v = [json.load(open(f))['mean_ndcg@10'] for f in sorted(glob.glob('results-regimes/quantized/$1/heldout2_*.json'))]
print(f'{sum(v)/len(v):.2f} over {len(v)} tasks' if v else 'no result')")"
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
arm() {  # arm TAG CFG [FALLBACK]: train, tiny, sparsity, length diagnostic
    local tag="$1"
    step train2 "$tag" "$2" "${3:-}" || return 1
    step bench "arm-$tag" "models/arm-$tag"; step sparsity "arm-$tag" "models/arm-$tag"; step diag "arm-$tag"
    log "  arm-$tag: tiny $(tiny "arm-$tag") (reference $REF, base $BASE), $(dims "arm-$tag") dims/doc (cluster-like $(dims arm-cluster-like-fiqa))"
}
best_by_hold() {  # best_by_hold NAME...: the name with the highest held-out-8 among those that have one
    local best="" bv="" m v
    for m in "$@"; do
        v=$(hold "$m"); [ -n "$v" ] || continue
        if [ -z "$bv" ] || [ "$(ge "$v" "$bv")" = 1 ]; then best="$m"; bv="$v"; fi
    done
    echo "$best"
}

REF=$(tiny arm-cluster-like-fiqa); BASE=$(tiny base-splade-wrapped); NEAR=$(fsub "$REF" 0.3); DOSEMIN=61.5
log "=== v22 start: reference $REF (near $NEAR), base $BASE, dose floor $DOSEMIN, $(free_gb) GB free ==="

log "=== v22 step 1: batch control (CachedSpladeLoss, true batch 32, mini-batch 4) ==="
arm mnrl-fiqa-ctrl-cb32 "$C/mnrl_fiqa_ctrl_cb32.yaml"

log "=== v22 step 2: long-easy dose (1, 2, 4 of 10) ==="
for n in 1 2 4; do arm "mnrl-fiqa-ldose$n" "$C/mnrl_fiqa_ldose$n.yaml" "$C/mnrl_fiqa_ldose${n}_cb32.yaml"; done
DOSE=""
for n in 1 2 4; do
    m="arm-mnrl-fiqa-ldose$n"; [ -d "models/$m" ] || continue
    if reaches "$m" "$DOSEMIN"; then step holdout "$m" "models/$m" && DOSE="$DOSE $m"; else log "  $m: tiny $(tiny "$m") below $DOSEMIN; no held-out"; fi
done
# shellcheck disable=SC2086
BESTDOSE=$(best_by_hold $DOSE)
if [ -n "$BESTDOSE" ]; then
    log "  best dose arm by held-out: $BESTDOSE $(hold "$BESTDOSE") (cluster-like $(hold arm-cluster-like-fiqa), long-easy $(hold long-easy-5070), n1 $(hold n1-5070))"
    step leaderboard "$BESTDOSE"; step regime quantized "$BESTDOSE" "models/$BESTDOSE" heldout2_treccovid 16
    log "  $BESTDOSE leaderboard: $(lb "$BESTDOSE" tiny10) / $(lb "$BESTDOSE" heldout8), trec-covid $(lb "$BESTDOSE" heldout2_treccovid) (long-easy $(lb long-easy-5070 tiny10) / $(lb long-easy-5070 heldout8); n1 $(lb n1-5070 tiny10) / $(lb n1-5070 heldout8); n4 trec-covid $(lb n4-5070 heldout2_treccovid))"
else
    log "  no dose arm reached $DOSEMIN on tiny; no soup-v8a"
fi

log "=== v22 step 3: control held-out ==="
m=arm-mnrl-fiqa-ctrl-cb32
if [ -d "models/$m" ]; then
    if reaches "$m" "$NEAR"; then step holdout "$m" "models/$m"; log "  $m: tiny $(tiny "$m") held-out $(hold "$m") (cluster-like $(hold arm-cluster-like-fiqa))"; else log "  $m: tiny $(tiny "$m") below $NEAR; no held-out"; fi
fi

log "=== v22 step 4: length-matched student negatives ==="
for v in hard far hard-mix2; do arm "mnrl-fiqa-lenmatch-$v" "$C/mnrl_fiqa_lenmatch_${v//-/_}.yaml" "$C/mnrl_fiqa_lenmatch_${v//-/_}_cb32.yaml"; done
LEN=""
for v in hard far hard-mix2; do
    m="arm-mnrl-fiqa-lenmatch-$v"; [ -d "models/$m" ] || continue
    if reaches "$m" "$NEAR"; then step holdout "$m" "models/$m" && LEN="$LEN $m"; else log "  $m: tiny $(tiny "$m") below $NEAR; no held-out"; fi
done
# shellcheck disable=SC2086
BESTLEN=$(best_by_hold $LEN)
log "  lenmatch: hard $(tiny arm-mnrl-fiqa-lenmatch-hard) (full-length r1-20 60.32, own 60.13), far $(tiny arm-mnrl-fiqa-lenmatch-far) (full-length far 62.23), hard-mix2 $(tiny arm-mnrl-fiqa-lenmatch-hard-mix2) (full-length mix2 60.41); best by held-out: ${BESTLEN:-none}"

log "=== v22 step 5: soup-v8a (n1 + n3 + n4 + best dose arm) ==="
if [ -n "$BESTDOSE" ]; then
    step merge soup-v8a "" models/n1-5070 models/n3-5070 models/n4-5070 "models/$BESTDOSE"
    if [ -n "$(tiny soup-v8a)" ]; then
        step holdout soup-v8a models/soup-v8a; step leaderboard soup-v8a
        log "  soup-v8a: tiny $(tiny soup-v8a) held-out $(hold soup-v8a) leaderboard $(lb soup-v8a tiny10) / $(lb soup-v8a heldout8) vs soup-v7a $(tiny soup-v7a) / $(hold soup-v7a) / $(lb soup-v7a tiny10) / $(lb soup-v7a heldout8)"
        a=$(lb soup-v8a heldout8); b=$(lb soup-v7a heldout8)
        if [ -n "$a" ] && [ -n "$b" ] && [ "$(ge "$a" "$b")" = 1 ]; then step heldout2 soup-v8a; else log "  soup-v8a below soup-v7a on the leaderboard heldout8; no held-out-2"; fi
    fi
fi

log "=== v22 step 6: fillers ==="
m=arm-mnrl-fiqa-ctrl-cb32
[ -d "models/$m" ] && reaches "$m" "$NEAR" && { step leaderboard "$m"; log "  $m leaderboard: $(lb "$m" tiny10) / $(lb "$m" heldout8) (cluster-like 62.85 / 41.76)"; }
[ -n "$BESTLEN" ] && { step leaderboard "$BESTLEN"; log "  $BESTLEN leaderboard: $(lb "$BESTLEN" tiny10) / $(lb "$BESTLEN" heldout8)"; }
step merge soup-v8b "0.2917 0.2917 0.2916 0.125" models/n1-5070 models/n3-5070 models/n4-5070 models/long-easy-5070
if [ -n "$(tiny soup-v8b)" ]; then
    step holdout soup-v8b models/soup-v8b; step leaderboard soup-v8b
    log "  soup-v8b: tiny $(tiny soup-v8b) held-out $(hold soup-v8b) leaderboard $(lb soup-v8b tiny10) / $(lb soup-v8b heldout8) vs soup-v7a $(lb soup-v7a tiny10) / $(lb soup-v7a heldout8)"
fi

log "=== v22 step 7: summary, commit, push ==="
"$PY" scripts/queue_summary.py > results/summary_v22.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v22.md)"
{
    echo; echo "## Per-task, leaderboard regime (queue v22)"; echo
    # shellcheck disable=SC2086
    "$PY" scripts/regime_task_table.py --tasksets tiny10 heldout8 heldout2_treccovid \
        --models base-splade-wrapped arm-cluster-like-fiqa n1-5070 n4-5070 long-easy-5070 soup-v6a soup-v7a $BESTDOSE $BESTLEN arm-mnrl-fiqa-ctrl-cb32 soup-v8a soup-v8b 2>> "$LOGS/queue_summary.log"
    echo; echo "## Length diagnostics (own FiQA corpus; positive first by positive length in tokens)"; echo
    for f in results/data/length-bias-*.json; do n=$(basename "$f" .json); n=${n#length-bias-}; echo "- $n: $(diag_line "$n")"; done
    echo; echo "## Queue state"; echo; echo '```'; grep -E "^(train|bench|holdout|regime|sparsity):" "$STATE" | tail -80; echo '```'
} >> results/summary_v22.md
for p in results/arm-mnrl-fiqa-* results/soup-v8* results-holdout/arm-mnrl-fiqa-* results-holdout/soup-v8* results-regimes/quantized results/sparsity results/data results/summary_v22.md; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v22.out"
done
if git commit -q -m "Add the v22 results: the true-batch control, the long-easy doses and the length-matched student negatives" 2>> "$LOGS/queue_v22.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v22.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v22.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v22.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v22.out)"
fi
log "=== v22 done ($(date '+%d.%m %H:%M')) ==="
