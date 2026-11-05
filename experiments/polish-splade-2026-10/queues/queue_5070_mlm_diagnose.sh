#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
export QUEUE_TAG=mlm-diag MIN_RAM_GB=6 WANDB_MODE=disabled
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
source scripts/queue_lib.sh
guard_alive
set -e
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || { log 'GPU queue lock is busy'; exit 1; }
echo $$ > "$LOGS/queue_5070_mlm_diagnose.pid"
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
wait_ram
log 'diagnostic CPU tests for BF16 document/query conversion'
"$PY" -m pytest -q third_party/pirb/tests/test_splade_bfloat16.py > "$LOGS/mlm_diag_tests.log" 2>&1
STATE=$(find runs/arm-modernbert-probe-m/sentence-transformers/splade -path '*/step-checkpoints/step-5000/trainer_state.json' -print -quit)
test -n "$STATE"
log 'compare original MLM, step 625 and step 5000 in FP32 and BF16'
"$PY" scripts/diagnose_splade_checkpoint.py \
    --model original=OPI-PIB/pl-ModernBERT-512-base \
    --model step625=models/modernbert-probe-m-s00625 \
    --model step5000=models/modernbert-probe-m-s05000 \
    --trainer-state "$STATE" \
    --output "$RESULTS_ROOT/data/modernbert-probe-m-diagnosis.json" > "$LOGS/mlm_checkpoint_diagnosis.log" 2>&1
log 'diagnostic complete'
