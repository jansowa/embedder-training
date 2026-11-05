#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=mlm3 MIN_RAM_GB=6 TZ=Europe/Warsaw WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.
ulimit -c 0
source scripts/queue_lib.sh
exec 9>logs/gpu_queue.lock
flock -n 9 || { log 'Another paired queue holds GPU lock'; exit 1; }
echo $$ > logs/queue_mlm3.pid
guard_alive
log 'Waiting for idle GPU; paired M then A, 4306 steps each, no resumes'
while [ "$(gpu_busy)" != 0 ]; do sleep 30; done
sleep 10
[ "$(gpu_busy)" = 0 ] || exit 1
wait_ram
"$PY" experiments/polish-splade-2026-10/queues/run_paired_night3.py > logs/night3-paired-driver.log 2>&1 &
runner=$!
trap 'kill -TERM "$runner" 2>/dev/null; wait "$runner"; exit 143' TERM INT
wait "$runner"
rc=$?
log "Paired queue finished rc=$rc"
exit "$rc"
