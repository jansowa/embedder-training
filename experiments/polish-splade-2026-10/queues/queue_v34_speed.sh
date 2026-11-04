#!/usr/bin/env bash
# v34-speed (4.11): does the fast training path train the same model? Pauses queue v33 with SIGSTOP (its running child
# finishes on its own, the bash does not start the next step), waits for an idle GPU, then trains arm-n4-3060 again with
# the sparse-gradient SPLADE head (training/backends/splade_fused_head.py, splade_fused_head: exact) and without
# gradient checkpointing, measures it like every arm (sparsity, tiny10, heldout8), compares with the three stock seeds
# (arm-n4-3060 62.89 / 42.01, seed1 62.77 / 42.06, seed2 62.81 / 41.93) and resumes v33 with SIGCONT on exit.
# The worst-case memory check (12 columns x 2 x 512 tokens, scripts/splade_head_memcheck.py) decides between
# n4_3060_fast.yaml (no checkpointing, 2.55x in the micro-benchmark) and n4_3060_fast_ckpt.yaml (1.85x).
# Verdict: |fast - seed mean| <= 0.30 on both tiny10 and heldout8 = within seed noise (the seed ranges are 0.12 / 0.13).
#
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v34_speed.sh > logs/queue_v34s.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v34s MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-9000}   # stock arm-n4-3060 took 4 125 s
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v34s.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
TAG=n4-3060-fast
MAX_PEAK_GB=${MAX_PEAK_GB:-7.0}
TOL=${TOL:-0.30}

PAUSED=""
resume() { if [ -n "$PAUSED" ]; then kill -CONT "$PAUSED" 2>/dev/null && log "queue v33 (pid $PAUSED) resumed with SIGCONT"; PAUSED=""; fi; }
trap resume EXIT

lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
gpu_idle_strict() { while [ "$(gpu_busy)" -ne 0 ]; do sleep 30; done; sleep 20; [ "$(gpu_busy)" -eq 0 ]; }
train_secs() { awk -F'\t' -v k="train:$1" '$1 == k && $2 == "ok" {v = $4} END {print v}' "$STATE"; }

log "=== v34-speed start: fast path (fused head exact, no checkpointing) on the arm-n4-3060 recipe; stock $(lbs arm-n4-3060), seeds $(lbs arm-n4-3060-seed1) and $(lbs arm-n4-3060-seed2) ==="

# 1. pause v33 (SIGSTOP on its bash; a running train/regime child keeps going and the bash does not continue)
V33=$(cat "$LOGS/queue_v33.pid" 2>/dev/null || true)
if [ -n "$V33" ] && ps -p "$V33" -o args= 2>/dev/null | grep -q "queue_v33.sh"; then
    if ps -o stat= -p "$V33" | grep -q '^T'; then log "queue v33 (pid $V33) is already stopped; will resume it on exit"
    else kill -STOP "$V33" && log "queue v33 (pid $V33) paused with SIGSTOP; its current step: $(ps -o args= --ppid "$V33" | grep -v '^sleep' | head -1 | cut -c1-120)"; fi
    PAUSED="$V33"
else log "queue v33 not running; nothing to pause"; fi

log "waiting for an idle GPU"
s=$SECONDS; until gpu_idle_strict; do :; done; log "GPU idle after $((SECONDS - s))s"
wait_ram

# 2. worst-case memory check of the no-checkpointing variant
MC="$LOGS/memcheck_${TAG}.log"
log "memcheck: fused head exact, no checkpointing, 12 x 2 x 512"
timeout -k 30 1200 "$PY" scripts/splade_head_memcheck.py --model sdadas/polish-splade --head exact --no-checkpointing \
    --columns 12 --micro-batch 2 --max-seq-length 512 --dtype fp16 > "$MC" 2>&1; rc=$?
PEAK=$(grep -o 'peak_gb=.*' "$MC" | tail -1 | cut -d= -f2)
CFG="$C/n4_3060_fast.yaml"; VARIANT="exact, no checkpointing"
if [ "$rc" -eq 0 ] && [ -n "$PEAK" ] && [ "$(ge "$MAX_PEAK_GB" "$PEAK")" = 1 ]; then
    log "memcheck ok: peak ${PEAK} GB <= ${MAX_PEAK_GB} GB; training without checkpointing"
else
    log "memcheck: rc=$rc peak=${PEAK:-?} GB; falling back to the fused head with checkpointing (logs/memcheck_${TAG}.log)"
    CFG="$C/n4_3060_fast_ckpt.yaml"; VARIANT="exact, with checkpointing"
fi
record "memcheck:$TAG" "$([ "$rc" -eq 0 ] && echo ok || echo FAILED)" "${PEAK:-?}" "$VARIANT"

# 3. the arm, measured like every other arm
train "$TAG" "$CFG"
if [ -d "models/arm-$TAG" ]; then
    sparsity "arm-$TAG" "models/arm-$TAG"
    regime quantized "arm-$TAG" "models/arm-$TAG" tiny10 16
    regime quantized "arm-$TAG" "models/arm-$TAG" heldout8 16
    log "  arm-$TAG: $(lbs "arm-$TAG") ($VARIANT; stock arm-n4-3060 $(lbs arm-n4-3060)), $(dims "arm-$TAG") dims/doc, train $(train_secs "arm-$TAG") vs stock 4125s"
fi

# 4. verdict and summary
VERDICT=$("$PY" - "$REGIMES_ROOT" "$TOL" "$(train_secs "arm-$TAG")" <<'EOF'
import json, sys
root, tol, secs = sys.argv[1], float(sys.argv[2]), sys.argv[3]
def mean(m, ts):
    try: return json.load(open(f"{root}/quantized/{m}/{ts}.json"))["mean_ndcg@10"]
    except Exception: return None
seeds = ["arm-n4-3060", "arm-n4-3060-seed1", "arm-n4-3060-seed2"]
out = []
ok = True
for ts in ("tiny10", "heldout8"):
    ref = [v for v in (mean(m, ts) for m in seeds) if v is not None]
    fast = mean("arm-n4-3060-fast", ts)
    if fast is None or not ref: out.append(f"{ts}: fast missing"); ok = False; continue
    mu = sum(ref) / len(ref); d = fast - mu
    inside = abs(d) <= tol
    ok = ok and inside
    out.append(f"{ts}: fast {fast:.2f} vs seed mean {mu:.2f} (range {min(ref):.2f}-{max(ref):.2f}), diff {d:+.2f} -> {'within' if inside else 'OUTSIDE'} +-{tol}")
speed = ""
if secs: speed = f"; train {secs} vs 4125s stock = {4125 / float(secs.rstrip('s')):.2f}x"
print(("PASS" if ok else "FAIL") + ": " + "; ".join(out) + speed)
EOF
)
log "verdict: $VERDICT"
record "verdict:$TAG" "${VERDICT%%:*}" "-" "${VERDICT#*: }"

SUM="$RESULTS_ROOT/summary_v34_speed.md"
{
    echo "# v34-speed ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): the fast training path on the arm-n4-3060 recipe"
    echo
    echo "arm-n4-3060-fast = the rows, recipe and seed of arm-n4-3060 trained with the sparse-gradient SPLADE head (\`splade_fused_head: exact\`, $VARIANT). Reference: three stock seeds."
    echo
    echo "Verdict (tolerance +-$TOL vs the seed mean on both task sets): **$VERDICT**"
    echo
    echo "Memory check (12 x 2 x 512, fp16, with optimizer states): peak ${PEAK:-?} GB (limit $MAX_PEAK_GB GB)."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-n4-3060 arm-n4-3060-seed1 arm-n4-3060-seed2 arm-n4-3060-fast \
        --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-n4-3060 --model arm-n4-3060-seed1 --model arm-n4-3060-seed2 --model arm-n4-3060-fast 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Queue state (v34-speed)"; echo; echo '```'
    grep -E "^(memcheck|train|sparsity|regime:quantized|verdict):arm-$TAG|^memcheck:$TAG|^verdict:$TAG" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"

git add "$SUM" "$RESULTS_ROOT/sparsity/arm-$TAG.json" "$REGIMES_ROOT/quantized/arm-$TAG" 2>/dev/null
git commit -q -m "Add the v34-speed result: the fast SPLADE head on the arm-n4-3060 recipe" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin master > "$LOGS/push_v34s.log" 2>&1 && { echo pushed >> "$LOGS/push_v34s.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown

log "=== v34-speed done ($(date '+%d.%m %H:%M')) ==="
