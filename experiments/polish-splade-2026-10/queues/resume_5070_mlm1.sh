#!/usr/bin/env bash
# Run in WSL from the repository after turning the machine back on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
CONFIG=experiments/polish-splade-2026-10/configs/next/5070/mlm_probe_a_resume.yaml
# Refuse to start from scratch if this run's full checkpoint disappeared.
.venv/bin/python - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
import yaml
from training.checkpoints import find_latest_checkpoint, checkpoint_missing_files
from training.config_grid import expand_config_grid
from training.train import _apply_resume_to_configs
from training.run_metadata import verify_resume_metadata
config=yaml.safe_load(Path(sys.argv[1]).read_text())
out=Path(config['sentence_transformers']['output_dir'])
variants=expand_config_grid(config,backend='sentence-transformers',training_type='splade',resume=True)
if len(variants)!=1:
    raise SystemExit('Expected exactly one resume run')
_apply_resume_to_configs(variants,'sentence-transformers','latest')
verify_resume_metadata(out,variants[0])
checkpoint=find_latest_checkpoint(out)
if checkpoint is None or checkpoint_missing_files(checkpoint):
    raise SystemExit('No complete resume checkpoint; refusing a fresh run')
for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
    if not (checkpoint/name).is_file():
        raise SystemExit(f'Missing resume state: {checkpoint/name}')
state=json.loads((checkpoint/'trainer_state.json').read_text())
print(f"Will resume step {state['global_step']} of 18750 from {checkpoint}")
PY
export MLM_PROBE_CONFIG="$CONFIG"
# Existing 5.6 GiB of checkpoints leave about 7-8 GiB additional model storage.
export MLM_MIN_FREE_GIB="${MLM_MIN_FREE_GIB:-25}"
# A relative deadline remains valid after a shutdown of arbitrary length.
export MLM1_END="${MLM1_END:-$(date -d '+18 hours' '+%Y-%m-%d %H:%M')}"
mkdir -p logs
if [ -f logs/queue_5070_mlm1.out ]; then
    cp logs/queue_5070_mlm1.out "logs/queue_5070_mlm1_before_resume_$(date +%Y%m%d_%H%M%S).out"
fi
setsid nohup bash experiments/polish-splade-2026-10/queues/queue_5070_mlm1.sh > logs/queue_5070_mlm1.out 2>&1 < /dev/null &
launched=$!
# Keep the WSL launcher alive until bash has entered the queue and saved its PID.
for attempt in {1..30}; do
    if [ -f logs/queue_5070_mlm1.pid ] && [ "$(cat logs/queue_5070_mlm1.pid)" = "$launched" ] && kill -0 "$launched" 2>/dev/null; then
        echo "Resume queue running: PID $launched, deadline $MLM1_END; log logs/queue_5070_mlm1.out"
        exit 0
    fi
    kill -0 "$launched" 2>/dev/null || { cat logs/queue_5070_mlm1.out; echo 'Resume queue exited before startup' >&2; exit 1; }
    sleep 1
done
echo 'Queue startup was not confirmed; inspect logs/queue_5070_mlm1.out' >&2
exit 1
