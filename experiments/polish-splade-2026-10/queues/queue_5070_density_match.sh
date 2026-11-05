#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=density-match MIN_RAM_GB=6 WANDB_MODE=disabled PIRB_DTYPE=bf16
export OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=.
source scripts/queue_lib.sh
guard_alive
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || exit 1
echo $$ > "$LOGS/queue_5070_density_match.pid"
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
log 'step 0: E5 s1500/s2000; night-3 training is not part of this queue'
for n in 01500 02000; do
    label="modernbert-init-n2-e5-s$n"
    regime quantized "$label" "models/$label" tiny10 8 512
    "$PY" experiments/polish-splade-2026-10/queues/report_density_match.py > "$LOGS/density_match_report.log" 2>&1
    git add "$REGIMES_ROOT/quantized/$label/tiny10.json" "$RESULTS_ROOT/summary_5070_init.md" "$RESULTS_ROOT/data/density-match-step0.json"
    git commit --only -m "Record density-matched probe checkpoint s$n" -- "$REGIMES_ROOT/quantized/$label/tiny10.json" "$RESULTS_ROOT/summary_5070_init.md" "$RESULTS_ROOT/data/density-match-step0.json" >> "$LOGS/density_match_commits.log" 2>&1 || true
done
log 'step 0 complete; no night-3 run started'
