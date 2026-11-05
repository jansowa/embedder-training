#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=bias-audit PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
source scripts/queue_lib.sh
guard_alive
exec 9>logs/gpu_queue.lock
flock -n 9 || exit 1
while [ "$(gpu_busy)" != 0 ]; do sleep 15; done
set -e
for step in 02150 04306; do
  label="night3-paired-a-s$step"
  log "Corrected FP32 sparsity: $label"
  "$PY" scripts/compare_splade_sparsity.py --model "$label=models/$label" \
    --scope tiny --max-docs-per-task 300 --device cuda --batch-size 8 --no-fp16 \
    --output-json "$RESULTS_ROOT/sparsity/$label.corrected.json" > "logs/bias_audit_$step.log" 2>&1
done
"$PY" "$EXP_ROOT/queues/verify_night3_bias_vectors.py" > logs/bias_vectors.log 2>&1
log 'Bias audit measurements complete'
