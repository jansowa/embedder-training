#!/usr/bin/env bash
# Insurance for the unattended run: keeps the PIRB index cache from filling the
# disk while nobody is watching.
#
# It only ever removes cache/<task>/<model> directories for which a metrics.json
# already records that task's score, so an index a restart would still need is
# never touched. Idle unless free space drops below the threshold.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="$(pwd)/.venv/bin/python"
LOG="logs/disk_guard.log"
THRESHOLD_GB=20
INTERVAL=300

log() { echo "[$(date +%H:%M:%S)] [guard] $*" | tee -a "$LOG"; }
log "watching: prune when free space drops below ${THRESHOLD_GB} GB"

while true; do
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    if [ "${free_gb:-999}" -lt "$THRESHOLD_GB" ]; then
        log "free space ${free_gb} GB - pruning indexes whose results are saved"
        "$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, shutil
from pathlib import Path

cache = Path("third_party/pirb/cache").resolve()
freed = 0
removed = 0
for path in glob.glob("results/*/*/metrics.json") + glob.glob("results-holdout/*/*/metrics.json"):
    try:
        m = json.load(open(path))
    except Exception:
        continue
    model = m.get("model")
    if not model:
        continue
    slug = str(model).replace("/", "_").replace(".", "_")
    for key, value in m.items():
        if not key.startswith("epoch0/pirb_") or not isinstance(value, float):
            continue
        task = key.replace("epoch0/pirb_", "")
        # poleval-2022 tasks are named test-A-/test-B-<set> but share one cache dir per set
        task = task.replace("test-A-", "").replace("test-B-", "")
        if "@" in task or task in {"ndcg_tasks", "datasets", "model"}:
            continue
        d = (cache / task / slug).resolve()
        # Saved result for exactly this (task, model): the index is spent.
        if cache in d.parents and d.is_dir():
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
            freed += size
            removed += 1
print(f"pruned {removed} index directories, freed {freed / 2**30:.1f} GB")
PYEOF
        log "free space now: $(df -BG --output=avail . | tail -1 | tr -d ' ')"
    fi
    sleep "$INTERVAL"
done
