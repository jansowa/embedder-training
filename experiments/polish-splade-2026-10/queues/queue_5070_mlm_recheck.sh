#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
export QUEUE_TAG=mlm-recheck MIN_RAM_GB=6 WANDB_MODE=disabled OMP_NUM_THREADS=2
source scripts/queue_lib.sh
guard_alive
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || exit 1
echo $$ > "$LOGS/queue_5070_mlm_recheck.pid"
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
wait_ram
"$PY" -m pytest -q tests/test_queue_regime_results.py > "$LOGS/mlm_recheck_tests.log" 2>&1 || exit 1
"$PY" - "$REGIMES_ROOT/quantized/modernbert-probe-m-s05000/tiny10.json" <<'PY'
import json, sys, time
from pathlib import Path
path=Path(sys.argv[1])
if path.exists() and json.loads(path.read_text()).get('n_tasks') != 10:
    path.rename(path.with_name(path.name+'.invalid-bf16-'+str(int(time.time()))))
PY
log 'recheck s05000 with BF16-compatible document and query conversion'
if PIRB_DTYPE=bf16 regime quantized modernbert-probe-m-s05000 models/modernbert-probe-m-s05000 tiny10 16 512; then
    log 'recheck complete'
else
    log 'recheck FAILED; inspect regime log, do not treat zero completed tasks as a score'
    exit 1
fi
