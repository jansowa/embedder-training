#!/usr/bin/env bash
# Queue v30 (night 2/3.11.2026, 3060 Ti, after v29). Three things, in the order of their value:
#   1  the rest of queue v27, run synchronously with V27_DEADLINE=$V30_V27_DEADLINE: the trivia-1pos screen,
#      arm-n5-squad16k (16 000 fresh SQuAD rows; does more of a proven source keep paying?) and its two
#      soups with held-out-2 when they reach soup-v9b on heldout8. v27 skips the seeds and msrand (done in v29).
#   2  soup-v9b evaluated at 256 tokens (label soup-v9b-len256: tiny10, heldout8, the five held-out-2 sets):
#      the free +0.5 tiny10 seen on soup-of-ft; decides the max_seq_length of the published checkpoint.
#   3  arm-n4x-nq on nq-pl (4 h, cache on the data drive): v29 showed soup-v9b +1.39 over the base on nq-pl
#      (41.46 vs 40.07; the best model has 61.14); this says whether an in-domain cut moves an in-domain
#      BEIR task even though it loses on our sets. Only when it can finish before V30_END.
#   4  summary_v30.md, commit, push.
# Every step is skipped when its result exists; a step starts only when it fits before V30_END (default
# 3.11.2026 08:00). Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v30.sh > logs/queue_v30.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v30 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v30.pid"
guard_alive
V30_END=${V30_END:-2026-11-03 08:00}
V30_V27_DEADLINE=${V30_V27_DEADLINE:-2026-11-03 05:00}
V30_REGIME_TIMEOUT=${V30_REGIME_TIMEOUT:-28800}
V30_CACHE_DRIVE=${V30_CACHE_DRIVE:-/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/FlagEmbedding-regime-cache}
V30_NQ_MIN_FREE_GB=${V30_NQ_MIN_FREE_GB:-30}
END=$(date -d "$V30_END" +%s)
B_TINY=900; B_H8=2700; B_H2=1200; B_NQ=16200

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() {  # step SECONDS CMD...: run only when the step can still finish before V30_END
    local need=$1; shift
    fits "$need" || { log "no time before $V30_END for a ${need}s step; skipping: $*"; return 1; }
    "$@"
}
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in v29 v28 v27 v26b; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free

log "=== v30 start: end $V30_END; $(free_gb) GB free on / ==="

log "=== v30 step 1: queue v27 (deadline $V30_V27_DEADLINE) ==="
if grep -q "=== v27 done" "$LOGS/queue_v27.out" 2>/dev/null; then
    log "queue v27 already finished; not running it again"
else
    V27_DEADLINE="$V30_V27_DEADLINE" bash experiments/polish-splade-2026-10/queues/queue_v27.sh >> "$LOGS/queue_v27.out" 2>&1
    log "queue v27 exited ($(grep -c "FAILED" "$LOGS/queue_v27.out") FAILED lines in logs/queue_v27.out)"
    echo $$ > "$LOGS/queue_v30.pid"
fi
wait_gpu_free

log "=== v30 step 2: soup-v9b at 256 tokens ==="
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }
step $B_TINY regime quantized soup-v9b-len256 models/soup-v9b tiny10 16 256
step $B_H8 regime quantized soup-v9b-len256 models/soup-v9b heldout8 16 256
for t in czywiesz gpt35wiki poquad quora treccovid; do step $B_H2 regime quantized soup-v9b-len256 models/soup-v9b "heldout2_$t" 16 256; done
log "  soup-v9b at 256: tiny10 $(lbv soup-v9b-len256 tiny10) / heldout8 $(lbv soup-v9b-len256 heldout8) (at 512: $(lbv soup-v9b tiny10) / $(lbv soup-v9b heldout8))"
for t in czywiesz gpt35wiki poquad quora treccovid; do log "    heldout2_$t: $(lbv soup-v9b-len256 "heldout2_$t") (512: $(lbv soup-v9b "heldout2_$t"))"; done

log "=== v30 step 3: arm-n4x-nq on nq-pl ==="
CACHE_LINK="cache/regimes"
cache_free_gb() { df -BG --output=avail "$(readlink -f "$CACHE_LINK")" 2>/dev/null | tail -1 | tr -dc '0-9'; }
relocate_cache() {
    if ! mkdir -p "$V30_CACHE_DRIVE" 2>/dev/null || [ ! -w "$V30_CACHE_DRIVE" ]; then log "cache drive unavailable; cache stays on /"; return 0; fi
    if [ -L "$CACHE_LINK" ]; then log "cache/regimes is already a symlink to $(readlink "$CACHE_LINK")"; return 0; fi
    if [ -d "$CACHE_LINK" ] && [ -n "$(ls -A "$CACHE_LINK" 2>/dev/null)" ]; then log "cache/regimes is not empty; cache stays on /"; return 0; fi
    rmdir "$CACHE_LINK" 2>/dev/null
    ln -s "$V30_CACHE_DRIVE" "$CACHE_LINK" && log "cache/regimes -> $V30_CACHE_DRIVE ($(cache_free_gb) GB free)" || { mkdir -p "$CACHE_LINK"; log "could not link cache/regimes"; }
}
restore_cache() {
    if [ -L "$CACHE_LINK" ] && [ "$(readlink "$CACHE_LINK")" = "$V30_CACHE_DRIVE" ]; then
        rm -f "$CACHE_LINK"; mkdir -p "$CACHE_LINK"; log "cache/regimes restored to a plain directory on /"
    fi
}
regime_long() {  # regime_long REGIME LABEL MODEL TASKSET [BATCH]: regime() with V30_REGIME_TIMEOUT
    local reg="$1" label="$2" model="$3" ts="$4" bs="${5:-16}" out="$REGIMES_ROOT/$1/$2/$4.json"
    [ -f "$out" ] && [ "$(regime_mean "$out" | awk '{print $2}')" != "0" ] && { log "regime $reg/$label/$ts: already present ($(regime_mean "$out"))"; return 0; }
    [ -e "$model" ] || { log "regime $reg/$label/$ts: $model missing"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "regime $reg/$label/$ts (batch $bs, timeout ${V30_REGIME_TIMEOUT}s, $(cache_free_gb) GB free in the cache)"; local s=$SECONDS
    CUDA_VISIBLE_DEVICES=0 timeout -k 60 "$V30_REGIME_TIMEOUT" bash scripts/pirb_regime_eval.sh "$reg" "$label" "$model" "$ts" "$bs" > "$LOGS/regime_${reg}_${label}_${ts}.out" 2>&1
    local e=$((SECONDS - s)) m
    m=$(regime_mean "$out")
    if [ -n "$m" ] && [ "$(echo "$m" | awk '{print $2}')" != "0" ]; then
        log "regime $reg/$label/$ts -> mean=$m in ${e}s"; record "regime:$reg:$label:$ts" ok "$m" "${e}s"
    else
        log "regime $reg/$label/$ts FAILED after ${e}s (logs/regime_${reg}_${label}_${ts}.out)"
        record "regime:$reg:$label:$ts" FAILED "-" "${e}s"; rm -rf "$CACHE_LINK/${reg}_${label}_${ts}"; return 1
    fi
}
if [ -f "$REGIMES_ROOT/quantized/arm-n4x-nq/nqpl.json" ]; then
    log "arm-n4x-nq nq-pl already present: $(regime_mean "$REGIMES_ROOT/quantized/arm-n4x-nq/nqpl.json")"
elif ! fits $B_NQ; then
    log "no time before $V30_END for nq-pl (${B_NQ}s); left for the 5070"
else
    relocate_cache; trap restore_cache EXIT
    free=$(cache_free_gb)
    if [ "${free:-0}" -lt "$V30_NQ_MIN_FREE_GB" ]; then log "only ${free} GB free for the cache; skipping nq-pl"
    else regime_long quantized arm-n4x-nq models/arm-n4x-nq nqpl 8; fi
    restore_cache; trap - EXIT
fi
log "  nq-pl: base $(lbv base-splade-wrapped nqpl), soup-v9b $(lbv soup-v9b nqpl), arm-n4x-nq $(lbv arm-n4x-nq nqpl) (leaderboard polish-splade 40.42, best 61.14)"

log "=== v30 step 4: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v30.md"
{
    echo "# v30 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): soup-v9b at 256 tokens, arm-n4x-nq on nq-pl"
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped soup-v9b soup-v9b-len256 arm-n4x-nq \
        --tasksets tiny10 heldout8 heldout2_czywiesz heldout2_gpt35wiki heldout2_poquad heldout2_quora heldout2_treccovid nqpl 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v30)"; echo; echo '```'
    grep -E "^regime:quantized:(soup-v9b-len256|arm-n4x-nq):" "$STATE" | tail -20
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v30.out"
log "summary written to $SUM"
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM" "$EXP_ROOT/queues/queue_v30.sh"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v30.out"
done
if git commit -q -m "Add the v30 results: soup-v9b at 256 tokens and arm-n4x-nq on nq-pl" 2>> "$LOGS/queue_v30.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v30.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v30.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v30.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v30.out)"
fi
log "=== v30 done ($(date '+%d.%m %H:%M')) ==="
