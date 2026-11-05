#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=mlm-smoke MIN_RAM_GB=6 WANDB_MODE=disabled
source scripts/queue_lib.sh
guard_alive
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || { log 'another queue holds the GPU lock'; exit 1; }
echo $$ > "$LOGS/queue_5070_mlm_smoke.pid"
need_space || exit 1
wait_gpu_free
[ "$(gpu_busy)" -eq 0 ] || exit 1
wait_ram
log 'smoke start: 3 N2 optimizer steps'
start=$SECONDS
timeout -k 60 900 "$PY" run_experiments.py \
    --grid-configuration-file "$EXP_ROOT/configs/next/5070/smoke_5070.yaml" \
    --backend sentence-transformers --training-type splade > "$LOGS/smoke_5070.log" 2>&1
rc=$?
final=$(find runs/smoke-5070 -path '*/final/model.safetensors' -print -quit)
if [ "$rc" -eq 0 ] && [ -n "$final" ]; then
    log "smoke PASS in $((SECONDS-start))s: $final"
    date -Is > "$LOGS/mlm_smoke_passed"
else
    log "smoke FAILED (exit $rc); inspect logs/smoke_5070.log"
    exit 1
fi
