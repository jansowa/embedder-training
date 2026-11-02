#!/usr/bin/env bash
# Queue v28 (night 1/2.11.2026, 3060 Ti): leaderboard-regime (quantized) measurements of the base and
# soup-v9b on three public PIRB tasks never measured here, then arm-n4x-nq on nq-pl:
#   mqa    = maupqa-mqa (50 MB corpus, 172 k queries; ~25 min per model)
#   legalB = poleval-2022-test-B-legal-questions (same 34 MB corpus as test-A; ~7 min per model)
#   nqpl   = nq-pl (1.5 GB corpus, 2.68 M passages; 3-5 h per model, batch 8)
# Then results/summary_v28.md (per-task table) and, unless V28_CHAIN_V27=0, it launches queue v27
# unchanged with V27_DEADLINE=$V28_V27_DEADLINE (v27 skips every step that already has a result).
#
# Safety: runs only after v26b (and a still-waiting v27, should it exist) has exited; every step is
# skipped when its result exists; a failed or timed-out step is logged and the queue goes on; no step
# starts after V28_DEADLINE (default 2.11.2026 13:00; none = no deadline); no commits, no training.
# nq-pl: the encoded corpus is ~20 GB (7 KB per document) and / has ~25 GB free, so cache/regimes is
# pointed at the data drive (V28_CACHE_DRIVE, ext4, ~46 GB free) for the whole queue and restored on
# exit; an nq-pl step needs >= 30 GB free there or it is skipped. The library's regime() has a 3 h
# timeout; regime_long below is the same function with V28_REGIME_TIMEOUT (default 8 h).
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v28.sh > logs/queue_v28.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v28 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v28.pid"
guard_alive
V28_DEADLINE=${V28_DEADLINE:-2026-11-02 13:00}
V28_REGIME_TIMEOUT=${V28_REGIME_TIMEOUT:-28800}
V28_CHAIN_V27=${V28_CHAIN_V27:-1}
V28_V27_DEADLINE=${V28_V27_DEADLINE:-2026-11-03 08:00}
V28_CACHE_DRIVE=${V28_CACHE_DRIVE:-/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/FlagEmbedding-regime-cache}
V28_NQ_MIN_FREE_GB=${V28_NQ_MIN_FREE_GB:-30}
DEADLINE=""
[ "$V28_DEADLINE" != none ] && DEADLINE=$(date -d "$V28_DEADLINE" +%s)
before_deadline() { [ -z "$DEADLINE" ] || [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }

queue_alive() {  # queue_alive TAG: the pid in logs/queue_TAG.pid still runs queue_TAG.sh
    local pf="$LOGS/queue_$1.pid"
    [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"
}
for prev in v26b v27 v26; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done

# --- cache on the data drive for the big corpus -------------------------------------------------
CACHE_LINK="cache/regimes"
relocate_cache() {
    if ! mkdir -p "$V28_CACHE_DRIVE" 2>/dev/null || [ ! -w "$V28_CACHE_DRIVE" ]; then
        log "cache drive $V28_CACHE_DRIVE unavailable; cache stays on /"; return 0
    fi
    if [ -L "$CACHE_LINK" ]; then log "cache/regimes is already a symlink to $(readlink "$CACHE_LINK")"; return 0; fi
    if [ -d "$CACHE_LINK" ] && [ -n "$(ls -A "$CACHE_LINK" 2>/dev/null)" ]; then
        log "cache/regimes is not empty ($(ls "$CACHE_LINK" | tr '\n' ' ')); cache stays on /"; return 0
    fi
    rmdir "$CACHE_LINK" 2>/dev/null
    if ln -s "$V28_CACHE_DRIVE" "$CACHE_LINK"; then
        log "cache/regimes -> $V28_CACHE_DRIVE ($(cache_free_gb) GB free there)"
    else
        mkdir -p "$CACHE_LINK"; log "could not link cache/regimes; cache stays on /"
    fi
}
restore_cache() {
    if [ -L "$CACHE_LINK" ] && [ "$(readlink "$CACHE_LINK")" = "$V28_CACHE_DRIVE" ]; then
        rm -f "$CACHE_LINK"; mkdir -p "$CACHE_LINK"; log "cache/regimes restored to a plain directory on /"
    fi
}
cache_free_gb() { df -BG --output=avail "$(readlink -f "$CACHE_LINK")" 2>/dev/null | tail -1 | tr -dc '0-9'; }
trap restore_cache EXIT

# regime() from queue_lib.sh with a configurable wall-clock limit (the library hardcodes 3 h).
regime_long() {  # regime_long REGIME LABEL MODEL TASKSET [BATCH]
    local reg="$1" label="$2" model="$3" ts="$4" bs="${5:-16}" out="$REGIMES_ROOT/$1/$2/$4.json"
    [ -f "$out" ] && [ "$(regime_mean "$out" | awk '{print $2}')" != "0" ] && { log "regime $reg/$label/$ts: already present ($(regime_mean "$out"))"; return 0; }
    [ -e "$model" ] || { log "regime $reg/$label/$ts: $model missing"; return 1; }
    [ -f "configs/pirb_tasks/$ts.json" ] || { log "regime $reg/$label/$ts: no task set configs/pirb_tasks/$ts.json"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "regime $reg/$label/$ts (batch $bs, timeout ${V28_REGIME_TIMEOUT}s, cache $(readlink -f "$CACHE_LINK"), $(cache_free_gb) GB free)"; local s=$SECONDS
    CUDA_VISIBLE_DEVICES=0 timeout -k 60 "$V28_REGIME_TIMEOUT" bash scripts/pirb_regime_eval.sh "$reg" "$label" "$model" "$ts" "$bs" > "$LOGS/regime_${reg}_${label}_${ts}.out" 2>&1
    local e=$((SECONDS - s)) m
    m=$(regime_mean "$out")
    if [ -n "$m" ] && [ "$(echo "$m" | awk '{print $2}')" != "0" ]; then
        log "regime $reg/$label/$ts -> mean=$m in ${e}s"; record "regime:$reg:$label:$ts" ok "$m" "${e}s"
    else
        log "regime $reg/$label/$ts FAILED after ${e}s (logs/regime_${reg}_${label}_${ts}.out, logs/regimes/${reg}_${label}_${ts}.log)"
        record "regime:$reg:$label:$ts" FAILED "-" "${e}s"; rm -rf "$CACHE_LINK/${reg}_${label}_${ts}"; return 1
    fi
}
nq_step() {  # nq_step LABEL: nq-pl only with enough room for the encoded corpus
    local free; free=$(cache_free_gb)
    if [ "${free:-0}" -lt "$V28_NQ_MIN_FREE_GB" ]; then log "only ${free} GB free for the cache; skipping nq-pl for $1"; return 1; fi
    step regime_long quantized "$1" "models/$1" nqpl 8
}

log "=== v28 start: base + soup-v9b on mqa, legalB, nqpl, then arm-n4x-nq on nqpl; deadline $V28_DEADLINE; $(free_gb) GB free on / ==="
relocate_cache

log "=== v28 step 1: maupqa-mqa ==="
step regime_long quantized base-splade-wrapped models/base-splade-wrapped mqa 16
step regime_long quantized soup-v9b models/soup-v9b mqa 16

log "=== v28 step 2: legal-questions test-B ==="
step regime_long quantized base-splade-wrapped models/base-splade-wrapped legalB 16
step regime_long quantized soup-v9b models/soup-v9b legalB 16

log "=== v28 step 3: nq-pl ==="
nq_step base-splade-wrapped
nq_step soup-v9b
nq_step arm-n4x-nq

log "=== v28 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_v28.md"
{
    echo "# v28: public PIRB tasks never measured before ($(date '+%d.%m.%Y %H:%M'), quantized regime, nDCG@10)"
    echo
    echo "Leaderboard (polish-splade entry of 2023 / best standalone model): maupqa-mqa 18.78 / 26.09,"
    echo "poleval-2022-test-B-legal-questions 78.39 / 84.73, nq-pl 40.42 / 61.14."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped soup-v9b arm-n4x-nq --tasksets mqa legalB nqpl 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state"; echo; echo '```'
    grep -E "^regime:quantized:(base-splade-wrapped|soup-v9b|arm-n4x-nq):(mqa|legalB|nqpl)" "$STATE"
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v28.out"
log "summary written to $SUM"
for ts in mqa legalB nqpl; do
    for m in base-splade-wrapped soup-v9b arm-n4x-nq; do
        f="$REGIMES_ROOT/quantized/$m/$ts.json"; [ -f "$f" ] && log "  $ts $m: $(regime_mean "$f")"
    done
done

restore_cache
trap - EXIT
if [ "$V28_CHAIN_V27" = 1 ]; then
    if queue_alive v27; then
        log "queue v27 is already running; not launching it again"
    elif grep -q "=== v27 done" "$LOGS/queue_v27.out" 2>/dev/null; then
        log "queue v27 already finished; not launching it again"
    else
        V27_DEADLINE="$V28_V27_DEADLINE" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v27.sh > "$LOGS/queue_v27.out" 2>&1 < /dev/null &
        disown 2>/dev/null
        log "queue v27 launched (deadline $V28_V27_DEADLINE, log logs/queue_v27.out)"
    fi
fi
log "=== v28 done ($(date '+%d.%m %H:%M')) ==="
