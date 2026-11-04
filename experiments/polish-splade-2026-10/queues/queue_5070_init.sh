#!/usr/bin/env bash
# D0 -> E1 -> E2 -> E3 -> conditional E4; no pull and no night-3 launch.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=init MIN_RAM_GB=6 WANDB_MODE=disabled OMP_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 PIRB_DTYPE=bf16 PYTHONPATH=.
ulimit -c 0  # preserve stack traces without a multi-GB native core dump
source scripts/queue_lib.sh
guard_alive
echo $$ > "$LOGS/queue_5070_init.pid"
for pf in queue_5070_mlm_controls.pid queue_5070_mlm_controls_finish.pid; do
    if [ -f "$LOGS/$pf" ]; then
        previous=$(cat "$LOGS/$pf")
        while grep -aq 'queue_5070_mlm_controls' "/proc/$previous/cmdline" 2>/dev/null; do sleep 10; done
    fi
done
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || { log 'GPU queue lock busy; refusing to overlap'; exit 1; }
while [ "$(gpu_busy)" -ne 0 ]; do sleep 10; done
need_space || exit 1
wait_ram
log 'starting serial D0/E1/E2/E3 and conditional E4; no long training'
"$PY" scripts/splade_init_experiments.py > "$LOGS/init_runner.log" 2>&1
rc=$?
log "initialization queue finished (rc=$rc); report $RESULTS_ROOT/summary_5070_init.md"
exit "$rc"
