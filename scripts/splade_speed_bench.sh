#!/usr/bin/env bash
# Pause the running queue at its next step boundary, run the unit tests of the fused SPLADE head
# and the GPU micro-benchmark (scripts/splade_head_bench.py), then resume the queue.
#
# The queue bash gets SIGSTOP: its current child keeps running (its ancestor still exists, so the
# orphan guard leaves it alone) and the queue cannot start the next step. When the child has exited
# (zombie under the stopped bash) the GPU is free; after the benchmark SIGCONT lets the bash reap
# the child and go on as if nothing happened. The trap resumes the queue on every exit path.
#
#   setsid nohup bash scripts/splade_speed_bench.sh > logs/bench_speed.out 2>&1 < /dev/null &
#
# Environment: QUEUE (default v33), MAIN_REPO (checkout with .venv, datasets/ and logs/; default
# /home/jsowa/work/FlagEmbedding), ROWS (arm file), OUT (report dir), BENCH_ARGS (extra flags
# for the benchmark, e.g. "--variants stock8,exact"), SKIP_TESTS=1, GPU_WAIT (seconds, default 7200).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
MAIN=${MAIN_REPO:-/home/jsowa/work/FlagEmbedding}
PY="$MAIN/.venv/bin/python"
QUEUE=${QUEUE:-v33}
ROWS=${ROWS:-$MAIN/datasets/arms/mix3-6219-tv9b.jsonl}
OUT=${OUT:-$MAIN/runs/bench-speed}
BENCH_TIMEOUT=${BENCH_TIMEOUT:-1500}
GPU_WAIT=${GPU_WAIT:-7200}   # seconds to wait for a free GPU (another queue may still be on it)
MIN_RAM_GB=${MIN_RAM_GB:-6}
export ORPHAN_GUARD_EXEMPT=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] [bench] $*"; }

qpid=""
pidfile="$MAIN/logs/queue_$QUEUE.pid"
if [ -f "$pidfile" ] && ps -p "$(cat "$pidfile")" -o args= 2>/dev/null | grep -q "queue_$QUEUE.sh"; then
    qpid=$(cat "$pidfile")
else
    log "queue $QUEUE is not running; benchmarking without a pause"
fi
resume() {
    if [ -n "$qpid" ]; then
        kill -CONT "$qpid" 2>/dev/null && log "queue $QUEUE (pid $qpid) resumed" || log "could not resume pid $qpid"
    fi
}
trap resume EXIT

child_state() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f1; }
step_running() {  # a non-zombie direct child of the queue bash
    local c st
    for c in $(pgrep -P "$qpid" 2>/dev/null); do
        st=$(child_state "$c"); [ -n "$st" ] && [ "$st" != Z ] && return 0
    done
    return 1
}
if [ -n "$qpid" ]; then
    kill -STOP "$qpid" || { log "cannot stop pid $qpid"; qpid=""; exit 1; }
    log "queue $QUEUE (pid $qpid) stopped; waiting for its current step to finish"
    while step_running; do sleep 30; done
    log "queue step finished"
fi
gpu_busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
waited=0
while [ "$(gpu_busy)" != 0 ]; do
    [ "$waited" -ge "$GPU_WAIT" ] && { log "GPU still busy after ${GPU_WAIT}s; giving up"; exit 1; }
    sleep 15; waited=$((waited + 15))
done
sleep 10
avail=$(free -g | awk '/^Mem:/ {print $7}')
[ "${avail:-0}" -ge "$MIN_RAM_GB" ] || { log "only ${avail} GB RAM available; giving up"; exit 1; }

cd "$HERE" || exit 1
if [ "${SKIP_TESTS:-0}" != 1 ]; then
    log "unit tests (CPU)"
    if CUDA_VISIBLE_DEVICES="" timeout -k 30 900 "$PY" -m pytest -q tests/test_splade_fused_head.py > "$OUT/pytest.log" 2>&1; then
        log "unit tests passed: $(tail -1 "$OUT/pytest.log")"
    else
        log "unit tests FAILED ($OUT/pytest.log); no GPU benchmark"; exit 1
    fi
fi

log "GPU micro-benchmark -> $OUT/head_bench.md"
s=$SECONDS
# shellcheck disable=SC2086
timeout -k 30 "$BENCH_TIMEOUT" "$PY" scripts/splade_head_bench.py --rows "$ROWS" --output "$OUT/head_bench" ${BENCH_ARGS:-} > "$OUT/bench.log" 2>&1
rc=$?
log "benchmark finished with rc=$rc in $((SECONDS - s))s"
[ "$rc" = 0 ] && cat "$OUT/head_bench.md"
exit "$rc"
