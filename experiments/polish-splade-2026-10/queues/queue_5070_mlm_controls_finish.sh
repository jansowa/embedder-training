#!/usr/bin/env bash
# Follow the already-running controls without editing their queue.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=mlm-controls-finish MIN_RAM_GB=6 WANDB_MODE=disabled OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_5070_mlm_controls_finish.pid"
parent=$(cat "$LOGS/queue_5070_mlm_controls.pid")
log "waiting for controls queue PID $parent; then direct model checks and report"
until ! grep -aq queue_5070_mlm_controls.sh "/proc/$parent/cmdline" 2>/dev/null; do sleep 10; done
exec 9> "$LOGS/gpu_queue.lock"
flock -w 900 9 || { log 'GPU queue lock unavailable'; "$PY" scripts/summarize_splade_controls.py; exit 1; }
until [ "$(gpu_busy)" -eq 0 ]; do sleep 10; done
wait_ram
args=()
for name in noreg-scale1 noreg-scale1e-4; do
    model=$(find "runs/modernbert-control-$name" -name final -type d 2>/dev/null | head -1)
    if [ -n "$model" ]; then args+=(--model "$name=$model"); else log "MISSING final model for $name; report will mark incomplete"; fi
done
rc=0
if [ "${#args[@]}" -gt 0 ]; then
    log 'checking both controls in FP32 and BF16, independently of retrieval'
    PYTHONPATH=. timeout -k 60 1800 "$PY" scripts/diagnose_splade_checkpoint.py "${args[@]}" --output "$RESULTS_ROOT/data/modernbert-controls-diagnosis.json" > "$LOGS/mlm_controls_diagnosis.log" 2>&1
    rc=$?
    log "direct diagnostics finished (rc=$rc)"
fi
"$PY" scripts/summarize_splade_controls.py
summary_rc=$?
log "analysis finished (diagnostic rc=$rc, summary rc=$summary_rc); report: $RESULTS_ROOT/modernbert_controls_report.md"
record 'mlm-controls:analysis' "$rc/$summary_rc" '-' 'no further training scheduled'
exit "$summary_rc"
