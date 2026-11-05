#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export TZ=Europe/Warsaw QUEUE_TAG=init-recover MIN_RAM_GB=6
export WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIRB_DTYPE=bf16 PYTHONFAULTHANDLER=1
ulimit -c 0
source scripts/queue_lib.sh
guard_alive
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || exit 1
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
echo $$ > "$LOGS/queue_5070_init_recover.pid"
log 'recovering verified duplicate optimizer files and resuming E5 from checkpoint; deadline remains 08:00'
"$PY" experiments/polish-splade-2026-10/queues/recover_init_morning.py > "$LOGS/init_morning_recovery.log" 2>&1
rc=$?
log "morning recovery ended rc=$rc"
