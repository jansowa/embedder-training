#!/usr/bin/env bash
# Bounded causal controls, never resumes the collapsed probe or starts night 3.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=mlm-controls MIN_RAM_GB=6 WANDB_MODE=disabled OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
source scripts/queue_lib.sh
guard_alive
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || { log 'another diagnostic queue owns the GPU lock'; exit 1; }
echo $$ > "$LOGS/queue_5070_mlm_controls.pid"
trap 'log "controls queue exited"' EXIT
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
wait_ram
"$PY" - <<'PY'
from copy import deepcopy
from pathlib import Path
import yaml
root = Path('experiments/polish-splade-2026-10/configs/next/5070')
base = yaml.safe_load((root / 'modernbert_probe_m.yaml').read_text())
for name, scale in [('noreg-scale1', 1.0), ('noreg-scale1e-4', 0.0001)]:
    cfg = deepcopy(base)
    cfg['runs_dir'] = f'runs/modernbert-control-{name}'
    # A native crash may leave metadata without a checkpoint. A new timestamp
    # preserves that evidence and avoids overwriting/refusing the next attempt.
    cfg['timestamp_output_dir'] = True
    cfg['hparams'][0]['document_regularizer_weight'] = 0.0
    st = cfg['sentence_transformers']
    st.update(max_steps=500, warmup_ratio=0.5, scale=scale,
              query_regularizer_weight=0.0, splade_regularizer_scheduler=False,
              save_strategy='steps', save_steps=500, keep_step_checkpoints=[], logging_steps=25,
              report_to=[], regularizer_control={'warmup_steps': 1000, 'density': deepcopy(base['sentence_transformers']['regularizer_control']['density'])})
    # Retain reference construction and RNG consumption; density control cannot
    # modify zero weights before step 1000, beyond this 500-step control.
    st['splade_activation_stats'].update(interval_steps=50)
    (root / f'modernbert_control_{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
[ "$?" = 0 ] || exit 1
for name in noreg-scale1 noreg-scale1e-4; do
    cfg="$EXP_ROOT/configs/next/5070/modernbert_control_$name.yaml"
    run=$(find "runs/modernbert-control-$name" -name final -type d 2>/dev/null | head -1)
    if [ -n "$run" ]; then log "$name already has final weights; skipping"; continue; fi
    need_space || exit 1
    while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
    wait_ram
    log "control $name: 500-step cap, LR warmup 250 steps, 8x4, no FLOPS penalty"
    s=$SECONDS
    PYTHONPATH=. timeout -k 60 3600 "$PY" scripts/run_splade_collapse_control.py --backend sentence-transformers --training-type splade --config "$cfg" > "$LOGS/mlm_control_$name.log" 2>&1
    rc=$?
    log "control $name ended in $((SECONDS-s))s (rc=$rc)"
    record "mlm-control:$name" "$rc" '-' "$((SECONDS-s))s"
    [ "$rc" = 0 ] || exit "$rc"
done
log 'bounded controls complete; no further training scheduled'
