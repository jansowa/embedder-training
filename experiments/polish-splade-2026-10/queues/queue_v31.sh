#!/usr/bin/env bash
# Queue v31 (3.11.2026, 3060 Ti, after v30; the machine is free until 19:00). Three things, in the order
# of their value for the MLM-path decision of 4.11 (docs/PLAN-5070-MLM-2026-11-03.md §3-§4):
#   1  data probe S2: three 1 875-step runs from sdadas/polish-distilroberta with exactly the settings of
#      mlm-path-ep3 (lr 5e-5 linear, 4 x 8, 256 tokens, 6 negatives, docreg 0.001) on (a) msmarco judged
#      38 k rows, (b) squadv2 + fiqa 20 k, (c) the probe mix 60 k; sparsity + tiny10 quantized each.
#      Says which source teaches fastest from the MLM, next to mlm-path-ep3 (the same 60 k examples as
#      3 epochs of msmarco-clean-20000) and the 5070 probe's step-1875 checkpoint.
#   2  arm-n4x-nq on nq-pl (4.5 h, cache on the data drive), the step v30 most likely skips before 08:00.
#   3  heldout8 (quantized) for the two MLM anchors mlm-path-ep3 and mlm-msmarco-ep1.
#   4  summary_v31.md, commit, push.
# Every step is skipped when its result exists and starts only when it fits before V31_END (18:30).
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v31.sh > logs/queue_v31.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v31 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v31.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
V31_END=${V31_END:-2026-11-03 18:30}
V31_REGIME_TIMEOUT=${V31_REGIME_TIMEOUT:-28800}
V31_CACHE_DRIVE=${V31_CACHE_DRIVE:-/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/FlagEmbedding-regime-cache}
V31_NQ_MIN_FREE_GB=${V31_NQ_MIN_FREE_GB:-30}
END=$(date -d "$V31_END" +%s)
B_TRAIN=6000; B_TINY=900; B_H8=2700; B_NQ=16200

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() {  # step SECONDS CMD...: run only when the step can still finish before V31_END
    local need=$1; shift
    fits "$need" || { log "no time before $V31_END for a ${need}s step; skipping: $*"; return 1; }
    "$@"
}
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in v30 v27; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }

log "=== v31 start: end $V31_END; $(free_gb) GB free on / ==="

log "=== v31 step 1: data probe S2 (three 1 875-step runs from the MLM) ==="
for tag in mlm-s2a-msmarco mlm-s2b-squadfiqa mlm-s2c-mix; do
    cfg="$C/${tag//-/_}.yaml"
    step $B_TRAIN train "$tag" "$cfg" || continue
    step 120 sparsity "arm-$tag" "models/arm-$tag"
    step $B_TINY regime quantized "arm-$tag" "models/arm-$tag" tiny10 16 512
    log "  arm-$tag: tiny10 $(lbv "arm-$tag" tiny10), $(sparsity_mean "$RESULTS_ROOT/sparsity/arm-$tag.json" "arm-$tag") dims/doc"
done
for m in mlm-path-ep3 mlm-msmarco-ep1; do
    step 120 sparsity "$m" "models/$m"
    step $B_TINY regime quantized "$m" "models/$m" tiny10 16 512
done
log "  anchors: mlm-path-ep3 $(lbv mlm-path-ep3 tiny10), mlm-msmarco-ep1 $(lbv mlm-msmarco-ep1 tiny10) (base $(lbv base-splade-wrapped tiny10))"

log "=== v31 step 2: arm-n4x-nq on nq-pl ==="
CACHE_LINK="cache/regimes"
cache_free_gb() { df -BG --output=avail "$(readlink -f "$CACHE_LINK")" 2>/dev/null | tail -1 | tr -dc '0-9'; }
relocate_cache() {
    if ! mkdir -p "$V31_CACHE_DRIVE" 2>/dev/null || [ ! -w "$V31_CACHE_DRIVE" ]; then log "cache drive unavailable; cache stays on /"; return 0; fi
    if [ -L "$CACHE_LINK" ]; then log "cache/regimes is already a symlink to $(readlink "$CACHE_LINK")"; return 0; fi
    if [ -d "$CACHE_LINK" ] && [ -n "$(ls -A "$CACHE_LINK" 2>/dev/null)" ]; then log "cache/regimes is not empty; cache stays on /"; return 0; fi
    rmdir "$CACHE_LINK" 2>/dev/null
    ln -s "$V31_CACHE_DRIVE" "$CACHE_LINK" && log "cache/regimes -> $V31_CACHE_DRIVE ($(cache_free_gb) GB free)" || { mkdir -p "$CACHE_LINK"; log "could not link cache/regimes"; }
}
restore_cache() {
    if [ -L "$CACHE_LINK" ] && [ "$(readlink "$CACHE_LINK")" = "$V31_CACHE_DRIVE" ]; then
        rm -f "$CACHE_LINK"; mkdir -p "$CACHE_LINK"; log "cache/regimes restored to a plain directory on /"
    fi
}
regime_long() {  # regime_long REGIME LABEL MODEL TASKSET [BATCH]: regime() with V31_REGIME_TIMEOUT
    local reg="$1" label="$2" model="$3" ts="$4" bs="${5:-16}" out="$REGIMES_ROOT/$1/$2/$4.json"
    [ -f "$out" ] && [ "$(regime_mean "$out" | awk '{print $2}')" != "0" ] && { log "regime $reg/$label/$ts: already present ($(regime_mean "$out"))"; return 0; }
    [ -e "$model" ] || { log "regime $reg/$label/$ts: $model missing"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "regime $reg/$label/$ts (batch $bs, timeout ${V31_REGIME_TIMEOUT}s, $(cache_free_gb) GB free in the cache)"; local s=$SECONDS
    CUDA_VISIBLE_DEVICES=0 timeout -k 60 "$V31_REGIME_TIMEOUT" bash scripts/pirb_regime_eval.sh "$reg" "$label" "$model" "$ts" "$bs" > "$LOGS/regime_${reg}_${label}_${ts}.out" 2>&1
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
    log "no time before $V31_END for nq-pl (${B_NQ}s); left for the 5070"
else
    relocate_cache; trap restore_cache EXIT
    free=$(cache_free_gb)
    if [ "${free:-0}" -lt "$V31_NQ_MIN_FREE_GB" ]; then log "only ${free} GB free for the cache; skipping nq-pl"
    else regime_long quantized arm-n4x-nq models/arm-n4x-nq nqpl 8; fi
    restore_cache; trap - EXIT
fi
log "  nq-pl: base $(lbv base-splade-wrapped nqpl), soup-v9b $(lbv soup-v9b nqpl), arm-n4x-nq $(lbv arm-n4x-nq nqpl)"

log "=== v31 step 3: heldout8 for the MLM anchors ==="
for m in mlm-path-ep3 mlm-msmarco-ep1; do step $B_H8 regime quantized "$m" "models/$m" heldout8 16 512; done

log "=== v31 step 4: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v31.md"
{
    echo "# v31 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): data probe S2 from the MLM, anchors, arm-n4x-nq on nq-pl"
    echo
    echo "S2: 1 875 steps (60 k examples) from sdadas/polish-distilroberta, settings of mlm-path-ep3; only the data differs:"
    echo "mlm-s2a-msmarco = msmarco judged 38 k rows, mlm-s2b-squadfiqa = squadv2 18 k + fiqa 2 k, mlm-s2c-mix = both (60 k)."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped mlm-path-ep3 mlm-msmarco-ep1 arm-mlm-s2a-msmarco arm-mlm-s2b-squadfiqa arm-mlm-s2c-mix arm-n4x-nq soup-v9b \
        --tasksets tiny10 heldout8 nqpl 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (300 PIRB passages per tiny task, base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model mlm-path-ep3 --model mlm-msmarco-ep1 --model arm-mlm-s2a-msmarco --model arm-mlm-s2b-squadfiqa --model arm-mlm-s2c-mix 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v31)"; echo; echo '```'
    grep -E "^(train|sparsity|regime:quantized):(arm-mlm-s2|mlm-path-ep3|mlm-msmarco-ep1|arm-n4x-nq:nqpl)" "$STATE" | tail -30
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v31.out"
log "summary written to $SUM"
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM" "$EXP_ROOT/queues/queue_v31.sh" "$C/mlm_s2a_msmarco.yaml" "$C/mlm_s2b_squadfiqa.yaml" "$C/mlm_s2c_mix.yaml"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v31.out"
done
if git commit -q -m "Add the v31 results: the S2 data probe from the MLM, its anchors and arm-n4x-nq on nq-pl" 2>> "$LOGS/queue_v31.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v31.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v31.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v31.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v31.out)"
fi
log "=== v31 done ($(date '+%d.%m %H:%M')) ==="
