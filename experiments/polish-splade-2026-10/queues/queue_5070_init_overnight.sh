#!/usr/bin/env bash
# Second queue: observe first queue, then adaptive A/B/C until 08:00 Warsaw.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export TZ=Europe/Warsaw QUEUE_TAG=init-overnight MIN_RAM_GB=6
export WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 PIRB_DTYPE=bf16 PYTHONPATH=.
ulimit -c 0
source scripts/queue_lib.sh
guard_alive
echo $$ > "$LOGS/queue_5070_init_overnight.pid"
export INIT_OVERNIGHT_END=${INIT_OVERNIGHT_END:-$("$PY" - <<'PY'
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
now=datetime.now(ZoneInfo('Europe/Warsaw'))
end=now.replace(hour=8,minute=0,second=0,microsecond=0)
if end<=now: end+=timedelta(days=1)
print(end.isoformat())
PY
)}
log "waiting for initial queue; overnight GPU deadline $INIT_OVERNIGHT_END"
"$PY" scripts/splade_init_overnight.py --wait-primary >> "$LOGS/init_overnight_runner.log" 2>&1 || exit 1
exec 9> "$LOGS/gpu_queue.lock"
flock -w 120 9 || { log 'GPU queue lock busy; refusing to overlap'; exit 1; }
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
wait_ram
log 'first queue finished; starting adaptive overnight branch'
"$PY" scripts/splade_init_overnight.py >> "$LOGS/init_overnight_runner.log" 2>&1
rc=$?
log "overnight queue ended (rc=$rc); no night 3 launched"
exit "$rc"
