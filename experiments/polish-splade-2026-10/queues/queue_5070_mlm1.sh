#!/usr/bin/env bash
# Night 1: fixed-lambda MLM probe, anchors, durable checkpoints, curve and G1.
# setsid nohup bash experiments/polish-splade-2026-10/queues/queue_5070_mlm1.sh > logs/queue_5070_mlm1.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=mlm1 MIN_RAM_GB=6 WANDB_MODE=offline TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-43200}"
source scripts/queue_lib.sh
guard_alive
exec 9> "$LOGS/gpu_queue.lock"
flock -n 9 || { log 'another queue holds the GPU lock'; exit 1; }
echo $$ > "$LOGS/queue_5070_mlm1.pid"
C="$EXP_ROOT/configs/next/5070"
MLM_PROBE_CONFIG=${MLM_PROBE_CONFIG:-$C/mlm_probe_a_fallback.yaml}
MLM1_END=${MLM1_END:-2026-11-04 08:00}
END=$(date -d "$MLM1_END" +%s)
QUEUE_STARTED=$(date -Is)
export QUEUE_STARTED
fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
# Never let the shared helpers proceed on an occupied card after their timeout.
wait_gpu_free() {
    local waited=0
    while [ "$(gpu_busy)" -ne 0 ]; do
        [ "$waited" -ge 2400 ] && { log 'FAILED GPU busy after 40 min'; return 1; }
        sleep 30; waited=$((waited+30))
    done
    sleep 10
    [ "$(gpu_busy)" -eq 0 ]
}
step() {
    local budget=$1; shift
    fits "$budget" || { log "no time before $MLM1_END; skipping: $*"; return 1; }
    need_space && wait_gpu_free && wait_ram || return 1
    "$@"
    local rc=$?
    [ "$rc" -eq 0 ] || { log "FAILED step ($rc): $*"; record "step:$*" FAILED "$rc"; }
    return "$rc"
}
regime_complete() {
    "$PY" - "$1" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    ok=d.get('n_tasks') == 10 and len(d.get('tasks',{})) == 10
except (OSError,ValueError):
    ok=False
sys.exit(0 if ok else 1)
PY
}
tiny10() {
    local name=$1 model=$2 out="$REGIMES_ROOT/quantized/$1/tiny10.json"
    regime_complete "$out" && return 0
    # Preserve a failed partial result instead of letting regime() skip it.
    [ -f "$out" ] && mv "$out" "$out.partial-$(date +%s)"
    regime quantized "$name" "$model" tiny10 16 512
    regime_complete "$out"
}
queue_alive() {
    local pf="$LOGS/queue_$1.pid"
    [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"
}
for previous in 5070_day1 5070; do
    while queue_alive "$previous"; do log "waiting for queue $previous"; sleep 60; done
done
[ -f "$LOGS/mlm_smoke_passed" ] || { log 'FAILED: no green smoke marker'; exit 1; }
log "=== mlm1 start $QUEUE_STARTED; variant fallback_fixed_lambda; deadline $MLM1_END ==="

for model in mlm-path-ep3 mlm-msmarco-ep1; do
    step 180 sparsity "$model" "models/$model"
    step 900 tiny10 "$model" "models/$model"
done

watch625() {
    exec 9>&-
    exec "$PY" - "$1" "$2" "$3" <<'PY'
import glob, json, re, sys, time
from pathlib import Path
prefix, started, logname=sys.argv[1],int(sys.argv[2]),sys.argv[3]
while time.time()-started < 32520:
    paths=[p for p in glob.glob(f'runs/{prefix}/sentence-transformers/splade/*/step-checkpoints/step-625/trainer_state.json') if Path(p).stat().st_mtime >= started]
    if paths:
        elapsed=time.time()-started
        log=Path(logname).read_text(errors='replace')
        match=re.search(r'625/18750\s+\[([\d:]+)<',log)
        loop_seconds=None
        if match:
            loop_seconds=0
            for component in match[1].split(':'): loop_seconds=loop_seconds*60+int(component)
        Path('logs/mlm_first625.json').write_text(json.dumps({
            'attempt':prefix,'wallclock_seconds_from_train_call_including_loading':elapsed,
            'wallclock_examples_per_second':20000/elapsed,
            'training_loop_seconds_from_progress_bar':loop_seconds,'checkpoint':paths[-1]},indent=2))
        break
    time.sleep(5)
PY
}
probe_train() {
    [ -f models/arm-mlm-probe-a/model.safetensors ] && return 0
    # Fresh-run budget is 35 GiB; resuming may account for existing checkpoint files.
    "$PY" - "${MLM_MIN_FREE_GIB:-35}" <<'PY'
import shutil, sys
free = shutil.disk_usage('/mnt/c').free / 2**30
minimum = float(sys.argv[1])
print(f'Physical C: free before training: {free:.2f} GiB (required {minimum:g} GiB)', flush=True)
sys.exit(0 if free >= minimum else 1)
PY
    [ "$?" -eq 0 ] || { log "FAILED: physical C: free below ${MLM_MIN_FREE_GIB:-35} GiB"; return 75; }
    local started=$(date +%s) rc
    export MLM_TRAIN_STARTED="$started"
    # A CPU-only watcher distinguishes loading wallclock from tqdm loop time.
    watch625 arm-mlm-probe-a "$started" "$LOGS/arm_mlm-probe-a.log" &
    local watcher=$!
    train mlm-probe-a "$MLM_PROBE_CONFIG"
    rc=$?
    if [ "$rc" -ne 0 ] && grep -qiE 'CUDA out of memory|torch.OutOfMemoryError' "$LOGS/arm_mlm-probe-a.log"; then
        cp "$LOGS/arm_mlm-probe-a.log" "$LOGS/arm_mlm-probe-a_oom.log"
        log 'OOM: retry once with gradient checkpointing'
        kill "$watcher" 2>/dev/null || true
        wait "$watcher" 2>/dev/null || true
        local gc_started=$(date +%s)
        watch625 arm-mlm-probe-a-gc "$gc_started" "$LOGS/arm_mlm-probe-a-gc.log" &
        watcher=$!
        train mlm-probe-a-gc "$C/mlm_probe_a_gc.yaml"
        rc=$?
        if [ -f models/arm-mlm-probe-a-gc/model.safetensors ]; then
            cp -r models/arm-mlm-probe-a-gc models/arm-mlm-probe-a
        fi
    fi
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    export MLM_TRAIN_RC="$rc" MLM_TRAIN_SECONDS="$(( $(date +%s)-started ))"
    "$PY" - <<'PY'
import json, os
from pathlib import Path
report={'queue_started':os.environ['QUEUE_STARTED'],'training_started_unix':int(os.environ['MLM_TRAIN_STARTED']),
        'training_seconds':int(os.environ['MLM_TRAIN_SECONDS']),'return_code':int(os.environ['MLM_TRAIN_RC'])}
p=Path('logs/mlm_first625.json')
report['first625']=json.loads(p.read_text()) if p.exists() else None
# train() reports a missing final for both timeouts and other errors; inspect the log too.
report['timeout_limit_seconds']=int(os.environ['TRAIN_TIMEOUT'])
report['ended_at_timeout_limit']=int(os.environ['MLM_TRAIN_SECONDS']) >= int(os.environ['TRAIN_TIMEOUT']) and int(os.environ['MLM_TRAIN_RC']) != 0
Path('logs/mlm_probe_timing.json').write_text(json.dumps(report,indent=2))
PY
    return "$rc"
}
step "$TRAIN_TIMEOUT" probe_train
training_rc=$?
[ "$training_rc" -ne 75 ] || exit 75

while IFS=$'\t' read -r number checkpoint; do
    [ -n "$checkpoint" ] || continue
    name=$(printf 'mlm-probe-a-s%05d' "$number")
    if [ ! -f "models/$name/model.safetensors" ]; then
        need_space || continue
        "$PY" - "$checkpoint" "models/$name" <<'PY'
import shutil, sys
from pathlib import Path
src, dst=map(Path,sys.argv[1:])
tmp=dst.with_name('.'+dst.name+'.tmp')
if tmp.exists():
    # This is a queue-owned partial copy only, never a source checkpoint.
    assert tmp.parent.resolve() == Path('models').resolve() and tmp.name.startswith('.mlm-probe-a-s')
    shutil.rmtree(tmp)
shutil.copytree(src,tmp,ignore=shutil.ignore_patterns('optimizer.pt','scheduler.pt','rng_state*.pth'))
if not (tmp/'model.safetensors').exists():
    raise RuntimeError(f'No weights in {tmp}')
tmp.rename(dst)
PY
    fi
    step 180 sparsity "$name" "models/$name"
    step 900 tiny10 "$name" "models/$name"
done < <("$PY" - <<'PY'
from pathlib import Path
paths=Path('runs').glob('arm-mlm-probe-a*/sentence-transformers/splade/*/step-checkpoints/step-*')
by_step={}
for path in sorted(paths,key=lambda p:p.stat().st_mtime):
    if (path/'model.safetensors').exists() and (path/'trainer_state.json').exists():
        by_step[int(path.name.split('-')[-1])]=path
for step,path in sorted(by_step.items()): print(f'{step}\t{path}')
PY
)

"$PY" "$EXP_ROOT/queues/mlm1_report.py" || log 'FAILED curve/summary generation'
models=(base-splade-wrapped mlm-msmarco-ep1 mlm-path-ep3)
for n in 00625 01250 01875 02500 05000 10000 15000 18750; do
    [ -f "$REGIMES_ROOT/quantized/mlm-probe-a-s$n/tiny10.json" ] && models+=("mlm-probe-a-s$n")
done
{
    echo; echo '## Per-task table'; echo
    "$PY" scripts/regime_task_table.py --models "${models[@]}" --tasksets tiny10
    echo; echo '## Sparsity table'; echo
    args=(--model base=base)
    for name in "${models[@]:3}"; do args+=(--model "$name=$name"); done
    "$PY" scripts/sparsity_table.py "${args[@]}"
} >> "$RESULTS_ROOT/summary_5070_mlm1.md" 2>> "$LOGS/queue_summary.log"

# Commit only this queue's named outputs; --only preserves the user's staged changes.
paths=("$C/mlm_probe_a_fallback.yaml" "$C/mlm_probe_a_gc.yaml"
       "$EXP_ROOT/queues/queue_5070_mlm1.sh" "$EXP_ROOT/queues/mlm1_report.py"
       "$RESULTS_ROOT/data/mlm-probe-a-curve.json" "$RESULTS_ROOT/summary_5070_mlm1.md")
for path in "$C/mlm_probe_a_resume.yaml" "$EXP_ROOT/queues/resume_5070_mlm1.sh" "$EXP_ROOT/docs/RESUME-MLM-PROBE-5070.md"; do
    [ ! -f "$path" ] || paths+=("$path")
done
for name in "${models[@]:1}"; do
    [ -f "$REGIMES_ROOT/quantized/$name/tiny10.json" ] && paths+=("$REGIMES_ROOT/quantized/$name/tiny10.json")
    [ -f "$RESULTS_ROOT/sparsity/$name.json" ] && paths+=("$RESULTS_ROOT/sparsity/$name.json")
done
git add -- "${paths[@]}"
if git commit --only -m 'Add fixed-lambda MLM probe results' -- "${paths[@]}" >> "$LOGS/commit_mlm1.log" 2>&1; then
    (for attempt in 1 2 3 4 5 6; do
        timeout -k 10 180 git push origin master >> "$LOGS/push_mlm1.log" 2>&1 && break
        sleep 120
    done) > /dev/null 2>&1 < /dev/null 9>&- &
fi

if [ -f datasets/arms/fiqa-plus-squadv2.jsonl ] && [ "$(date +%s)" -lt "$(date -d '2026-11-03 08:00' +%s)" ]; then
    step 7200 train n5-b16 "$C/n5_cluster_fiqa_plus_squadv2_b16.yaml"
    step 180 sparsity arm-n5-b16 models/arm-n5-b16
    step 900 regime quantized arm-n5-b16 models/arm-n5-b16 tiny10 16 512
    step 2700 regime quantized arm-n5-b16 models/arm-n5-b16 heldout8 16 512
else
    log 'P3a skipped: expected dataset unavailable at configured path'
fi
log '=== mlm1 done ==='
