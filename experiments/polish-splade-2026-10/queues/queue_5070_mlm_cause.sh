#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
export QUEUE_TAG=mlm-cause MIN_RAM_GB=6 WANDB_MODE=disabled OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1
source scripts/queue_lib.sh
guard_alive
set -e
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || exit 1
echo $$ > "$LOGS/queue_5070_mlm_cause.pid"
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
wait_ram
log 'read-only backbone/head transplants and initial gradient analysis'
PYTHONPATH=. "$PY" scripts/analyze_splade_collapse.py --output "$RESULTS_ROOT/data/modernbert-probe-m-cause.json" > "$LOGS/mlm_cause.log" 2>&1
log 'causal diagnostics complete'
