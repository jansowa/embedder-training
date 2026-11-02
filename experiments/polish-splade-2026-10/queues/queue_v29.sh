#!/usr/bin/env bash
# Queue v29 (2.11.2026 daytime, 3060 Ti; the card must be free again by V29_END, default 20:00).
# Takes over from v28 after its base nq-pl measurement: v28 itself is stopped with SIGKILL (no EXIT trap,
# so the cache symlink and the running evaluation are left alone) and this queue first waits for that
# evaluation (V29_WAIT_PID) to finish. Then, each step only when it can still finish before V29_END:
#   1  soup-v9b on nq-pl (leaderboard regime, batch 8, ~4.2 h)                 risk question, 1.5 GB corpus
#   2  seeds 1 and 2 of arm-n4-3060 (train, sparsity, tiny10, heldout8; ~2 h each) and soup-n4seeds
#      (uniform soup of the three, ~1 h): the noise band of the n4 control and the P4 decision for the 5070
#   3  arm-n4-msrand (~2 h) when there is still time: the equal-step control of arm-n4-msclean
#   4  summary_v29.md (also the v28 numbers), commit, push
# arm-n4x-nq on nq-pl (v28 step 3c, 4 h) is dropped: the arm is not a soup ingredient, and the two seeds
# are worth more today; the 5070 can measure it later. Queue v27 is NOT chained: start it by hand when the
# card is free again; it skips every step that already has a result (the seeds and msrand among them).
# Start (after the v28 bash is killed):
#   V29_WAIT_PID=<pid of the running pirb_regime_eval.sh> setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v29.sh > logs/queue_v29.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v29 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v29.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
V29_END=${V29_END:-2026-11-02 20:00}
V29_WAIT_PID=${V29_WAIT_PID:-}
V29_REGIME_TIMEOUT=${V29_REGIME_TIMEOUT:-28800}
V29_CACHE_DRIVE=${V29_CACHE_DRIVE:-/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/FlagEmbedding-regime-cache}
V29_NQ_MIN_FREE_GB=${V29_NQ_MIN_FREE_GB:-30}
END=$(date -d "$V29_END" +%s)
# Step budgets in seconds (3060 timings: n4 train 4125 s, tiny10 600 s, heldout8 2200 s, nq-pl 15000 s).
B_NQ=16200; B_TRAIN=4800; B_SPARS=400; B_TINY=900; B_H8=2700; B_MERGE=300

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() {  # step SECONDS CMD...: run only when the step can still finish before V29_END
    local need=$1; shift
    fits "$need" || { log "no time before $V29_END for a ${need}s step; skipping: $*"; return 1; }
    "$@"
}
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in v28 v27 v26b; do
    queue_alive "$prev" && { log "queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) is still running; v29 refuses to start next to it"; exit 1; }
done

log "=== v29 start: end $V29_END; $(free_gb) GB free on / ==="
if [ -n "$V29_WAIT_PID" ] && ps -p "$V29_WAIT_PID" > /dev/null 2>&1; then
    log "waiting for the running evaluation (pid $V29_WAIT_PID: $(ps -p "$V29_WAIT_PID" -o args= | cut -c1-90)) to finish"
    while ps -p "$V29_WAIT_PID" > /dev/null 2>&1; do sleep 60; done
    sleep 30
fi
wait_gpu_free
f="$REGIMES_ROOT/quantized/base-splade-wrapped/nqpl.json"
[ -f "$f" ] && log "base-splade-wrapped nq-pl: $(regime_mean "$f")" || log "base-splade-wrapped nq-pl: no result"

# --- cache on the data drive for the big corpus (same as v28) ----------------------------------------
CACHE_LINK="cache/regimes"
cache_free_gb() { df -BG --output=avail "$(readlink -f "$CACHE_LINK")" 2>/dev/null | tail -1 | tr -dc '0-9'; }
relocate_cache() {
    if ! mkdir -p "$V29_CACHE_DRIVE" 2>/dev/null || [ ! -w "$V29_CACHE_DRIVE" ]; then log "cache drive unavailable; cache stays on /"; return 0; fi
    if [ -L "$CACHE_LINK" ]; then log "cache/regimes is already a symlink to $(readlink "$CACHE_LINK") ($(cache_free_gb) GB free)"; return 0; fi
    if [ -d "$CACHE_LINK" ] && [ -n "$(ls -A "$CACHE_LINK" 2>/dev/null)" ]; then log "cache/regimes is not empty; cache stays on /"; return 0; fi
    rmdir "$CACHE_LINK" 2>/dev/null
    ln -s "$V29_CACHE_DRIVE" "$CACHE_LINK" && log "cache/regimes -> $V29_CACHE_DRIVE ($(cache_free_gb) GB free)" || { mkdir -p "$CACHE_LINK"; log "could not link cache/regimes"; }
}
restore_cache() {
    if [ -L "$CACHE_LINK" ] && [ "$(readlink "$CACHE_LINK")" = "$V29_CACHE_DRIVE" ]; then
        rm -f "$CACHE_LINK"; mkdir -p "$CACHE_LINK"; log "cache/regimes restored to a plain directory on /"
    fi
}
relocate_cache
trap restore_cache EXIT

regime_long() {  # regime_long REGIME LABEL MODEL TASKSET [BATCH]: regime() with V29_REGIME_TIMEOUT
    local reg="$1" label="$2" model="$3" ts="$4" bs="${5:-16}" out="$REGIMES_ROOT/$1/$2/$4.json"
    [ -f "$out" ] && [ "$(regime_mean "$out" | awk '{print $2}')" != "0" ] && { log "regime $reg/$label/$ts: already present ($(regime_mean "$out"))"; return 0; }
    [ -e "$model" ] || { log "regime $reg/$label/$ts: $model missing"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "regime $reg/$label/$ts (batch $bs, timeout ${V29_REGIME_TIMEOUT}s, $(cache_free_gb) GB free in the cache)"; local s=$SECONDS
    CUDA_VISIBLE_DEVICES=0 timeout -k 60 "$V29_REGIME_TIMEOUT" bash scripts/pirb_regime_eval.sh "$reg" "$label" "$model" "$ts" "$bs" > "$LOGS/regime_${reg}_${label}_${ts}.out" 2>&1
    local e=$((SECONDS - s)) m
    m=$(regime_mean "$out")
    if [ -n "$m" ] && [ "$(echo "$m" | awk '{print $2}')" != "0" ]; then
        log "regime $reg/$label/$ts -> mean=$m in ${e}s"; record "regime:$reg:$label:$ts" ok "$m" "${e}s"
    else
        log "regime $reg/$label/$ts FAILED after ${e}s (logs/regime_${reg}_${label}_${ts}.out)"
        record "regime:$reg:$label:$ts" FAILED "-" "${e}s"; rm -rf "$CACHE_LINK/${reg}_${label}_${ts}"; return 1
    fi
}
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1"; }
n4_arm() {  # n4_arm TAG CFG: train, sparsity, leaderboard regime, each only when it fits
    local tag="$1"
    step $B_TRAIN train "$tag" "$2" || return 1
    step $B_SPARS sparsity "arm-$tag" "models/arm-$tag"
    step $B_TINY regime quantized "arm-$tag" "models/arm-$tag" tiny10 16
    step $B_H8 regime quantized "arm-$tag" "models/arm-$tag" heldout8 16
    log "  arm-$tag leaderboard: $(lbs "arm-$tag") (arm-n4-3060 $(lbs arm-n4-3060), arm-n4-msclean $(lbs arm-n4-msclean)), $(dims "arm-$tag") dims/doc"
}

log "=== v29 step 1: soup-v9b on nq-pl ==="
free=$(cache_free_gb)
if [ "${free:-0}" -lt "$V29_NQ_MIN_FREE_GB" ]; then log "only ${free} GB free for the cache; skipping nq-pl"
else step $B_NQ regime_long quantized soup-v9b models/soup-v9b nqpl 8; fi

log "=== v29 step 2: seeds 1 and 2 of arm-n4-3060, then the seed soup ==="
n4_arm n4-3060-seed1 "$C/n4_3060_seed1.yaml"
n4_arm n4-3060-seed2 "$C/n4_3060_seed2.yaml"
if [ -d models/arm-n4-3060-seed1 ] && [ -d models/arm-n4-3060-seed2 ]; then
    "$PY" - "$REGIMES_ROOT/quantized" 2>/dev/null <<'EOF' | while read -r line; do log "  $line"; done
import json, statistics, sys, os
for ts in ("tiny10", "heldout8"):
    fs = [f"{sys.argv[1]}/{m}/{ts}.json" for m in ("arm-n4-3060", "arm-n4-3060-seed1", "arm-n4-3060-seed2")]
    v = [json.load(open(f))["mean_ndcg@10"] for f in fs if os.path.exists(f)]
    if len(v) == 3:
        print(f"n4-3060 seeds {ts}: {' / '.join(f'{x:.2f}' for x in v)}, mean {statistics.mean(v):.2f}, std {statistics.stdev(v):.2f}, range {max(v) - min(v):.2f}")
    else:
        print(f"n4-3060 seeds {ts}: {len(v)} of 3 results")
EOF
    if [ ! -d models/soup-n4seeds ]; then
        if step $B_MERGE true; then
            need_space && { wait_gpu_free; wait_ram
                "$PY" scripts/merge_splade_weights.py --models models/arm-n4-3060 models/arm-n4-3060-seed1 models/arm-n4-3060-seed2 --output models/soup-n4seeds > "$LOGS/merge_soup-n4seeds.log" 2>&1 \
                    && log "soup soup-n4seeds = arm-n4-3060 + seed1 + seed2 (uniform)" || log "soup soup-n4seeds FAILED (logs/merge_soup-n4seeds.log)"; }
        fi
    fi
    if [ -d models/soup-n4seeds ]; then
        step $B_SPARS sparsity soup-n4seeds models/soup-n4seeds
        step $B_TINY regime quantized soup-n4seeds models/soup-n4seeds tiny10 16
        step $B_H8 regime quantized soup-n4seeds models/soup-n4seeds heldout8 16
        log "  soup-n4seeds leaderboard: $(lbs soup-n4seeds) (arm-n4-3060 $(lbs arm-n4-3060), soup-v9b $(lbs soup-v9b)), $(dims soup-n4seeds) dims/doc"
    fi
else
    log "  seeds incomplete; no seed soup"
fi

log "=== v29 step 3: arm-n4-msrand when there is still time ==="
if fits $((B_TRAIN + B_SPARS + B_TINY + B_H8)); then n4_arm n4-msrand "$C/n4_msrand.yaml"
else log "  no time for the whole msrand block before $V29_END; left to v27"; fi

log "=== v29 step 4: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v29.md"
MODELS=(base-splade-wrapped soup-v9b arm-n4x-nq arm-n4-3060 arm-n4-msclean)
for m in arm-n4-3060-seed1 arm-n4-3060-seed2 soup-n4seeds arm-n4-msrand; do [ -d "models/$m" ] && MODELS+=("$m"); done
{
    echo "# v28 + v29 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10)"
    echo
    echo "Leaderboard (polish-splade 2023 / best standalone): maupqa-mqa 18.78 / 26.09,"
    echo "poleval-2022-test-B-legal-questions 78.39 / 84.73, nq-pl 40.42 / 61.14."
    echo
    "$PY" scripts/regime_task_table.py --models "${MODELS[@]}" --tasksets mqa legalB nqpl tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v28, v29)"; echo; echo '```'
    grep -E "^(train|sparsity|regime):" "$STATE" | tail -30
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v29.out"
log "summary written to $SUM"
restore_cache; trap - EXIT
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM" "$EXP_ROOT/queues/queue_v29.sh"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v29.out"
done
if git commit -q -m "Add the v28 and v29 results: nq-pl, the n4 seeds and the seed soup" 2>> "$LOGS/queue_v29.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v29.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v29.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v29.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v29.out)"
fi
log "=== v29 done ($(date '+%d.%m %H:%M'); v27 not chained, start it by hand when the card is free) ==="
