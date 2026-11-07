#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=n4 MIN_RAM_GB=6 WANDB_MODE=offline TZ=Europe/Warsaw
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
source scripts/queue_lib.sh
CACHE_LINK=cache/regimes
N4_CACHE_HOST_PATH=${N4_CACHE_HOST_PATH:-/mnt/c}
disk_free_gb() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | awk '{print int($1/1073741824)}'; }
cache_free_gb() { disk_free_gb "$(readlink -f "$CACHE_LINK")"; }
host_free_gb() { disk_free_gb "$N4_CACHE_HOST_PATH"; }
p1_storage_ready() {
    local cache_free host_free
    cache_free=$(cache_free_gb); host_free=$(host_free_gb)
    if [ "${host_free:-0}" -lt 8 ]; then
        log "P1 stopped: VHDX host $N4_CACHE_HOST_PATH ${host_free:-unknown} GiB (minimum 8)"
        return 75
    fi
    if [ "${cache_free:-0}" -lt 60 ]; then
        log "P1 storage blocked: cache ${cache_free:-unknown} GiB (need 60); host ${host_free} GiB"
        return 74
    fi
    # Permissions alone do not detect a filesystem remounted read-only after I/O errors.
    "$PY" - "$CACHE_LINK" "$LOGS" <<'PY'
import os, sys, tempfile
from pathlib import Path
try:
    for directory in sys.argv[1:]:
        with tempfile.NamedTemporaryFile(dir=Path(directory), prefix='.n4-write-check-') as probe:
            probe.write(b'n4 write check\n')
            probe.flush()
            os.fsync(probe.fileno())
except OSError as exc:
    print(f'P1 storage write check failed: {exc}', file=sys.stderr)
    sys.exit(75)
PY
}
run_guarded() {
    "$PY" - "$N4_CACHE_HOST_PATH" "$LOGS" "$@" <<'DISK_GUARD_PY'
import json, os, shutil, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

host, logs, wall_limit, *command = sys.argv[1:]
logs = Path(logs)
minimum = 8 * 2**30
worker = None

def stop_group():
    if worker is None: return
    try: os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: worker.wait(timeout=10)
    except subprocess.TimeoutExpired: pass
    # Kill remaining descendants even when the process-group leader has exited.
    try: os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    worker.wait()

def storage_stop(reason, free=None):
    stop_group()
    data = {'time': datetime.now().astimezone().isoformat(timespec='seconds'),
            'reason': reason, 'host': host, 'free_bytes': free,
            'minimum_bytes': minimum, 'command': command}
    print(f'[n4] DISK STOP: {reason}; free={free}; minimum={minimum}', flush=True)
    try:
        (logs/'n4_disk_stop.json').write_text(json.dumps(data, indent=2)+'\n')
        with (logs/'queue.log').open('a') as stream:
            stream.write(f"[{data['time']}] [n4] DISK STOP: {reason}; free={free}\n")
    except OSError as exc:
        print(f'Cannot persist disk-stop marker: {exc}', file=sys.stderr, flush=True)
    return 75

def free_bytes():
    return shutil.disk_usage(host).free

def interrupted(sig, frame):
    stop_group()
    raise SystemExit(128+sig)

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    free = free_bytes()
except OSError as exc:
    sys.exit(storage_stop(f'Cannot check host disk: {exc}'))
if free < minimum:
    sys.exit(storage_stop('Host disk below 8 GiB before launch', free))
started = time.monotonic()
# Every benchmark descendant inherits this group; no timeout utility creates
# a separate process group that could escape termination.
worker = subprocess.Popen(command, start_new_session=True)
while True:
    try: free = free_bytes()
    except OSError as exc:
        sys.exit(storage_stop(f'Cannot check host disk: {exc}'))
    if free < minimum:
        sys.exit(storage_stop('Host disk below 8 GiB during benchmark', free))
    rc = worker.poll()
    if rc is not None:
        stop_group()
        sys.exit(rc if rc >= 0 else 128-rc)
    if time.monotonic()-started >= float(wall_limit):
        print('[n4] benchmark timed out; stopping process group', flush=True)
        stop_group()
        sys.exit(124)
    time.sleep(1)
DISK_GUARD_PY
}
# The shared helper proceeds after 40 minutes; these queues must never overlap.
wait_gpu_free() { while [ "$(gpu_busy)" -ne 0 ]; do sleep 30; done; sleep 10; [ "$(gpu_busy)" -eq 0 ] || wait_gpu_free; }
complete() {
    "$PY" - "$REGIMES_ROOT/quantized/$1/$2.json" "configs/pirb_tasks/$2.json" <<'PY'
import json, sys
from pathlib import Path
try:
    result = json.loads(Path(sys.argv[1]).read_text())
    config = json.loads(Path(sys.argv[2]).read_text())
    # Taskset files contain the benchmark's task definitions.
    expected = len(config)
    assert result['n_tasks'] == expected and result['mean_ndcg@10'] is not None
except (OSError, ValueError, KeyError, AssertionError, TypeError):
    sys.exit(1)
PY
}
report() {
    "$PY" - "$RESULTS_ROOT" "$REGIMES_ROOT" "$LOGS" <<'PY'
import json, subprocess, sys
from datetime import datetime
from pathlib import Path
r, g, logs = map(Path, sys.argv[1:])
def value(label, taskset):
    try:
        d = json.loads((g/'quantized'/label/f'{taskset}.json').read_text())
        return f"{d['mean_ndcg@10']:.2f}" if d['mean_ndcg@10'] is not None else '—'
    except (OSError, ValueError, KeyError): return '—'
lines = ['# Noc 4 — heldout8 nocy 3 i pełny publiczny PIRB', '',
         f"Aktualizacja: {datetime.now().astimezone().isoformat(timespec='seconds')}", '',
         '| Model s04306 | tiny10 | heldout8 | wym./dok (FP32, 512) |', '|---|---:|---:|---:|']
for label, tiny, dims in [('night3-paired-a-s04306', '56.73', '459.13'), ('night3-paired-m-s04306', '53.75', '327.85')]:
    lines.append(f'| {label} | {tiny} | {value(label, "heldout8")} | {dims} |')
lines += ['', 'A s4306: **nieporównywalne przy tej gęstości** (pasmo 327–458).',
          'Kotwice tiny10: baza 61.60; mlm-msmarco-ep1 59.29.', '',
          'P1: quantized, batch 8; wiki-trivia 14 h timeout, pozostałe 10 h. Wymagane ≥60 GiB wewnątrz cache. Zgodnie z aktualną decyzją użytkownika dopuszczono wykorzystanie wolnych bloków WSL przy mniejszym fizycznym zapasie na C:. Monitor co sekundę zatrzymuje całą grupę procesu i kolejkę przy C: <8 GiB lub błędzie odczytu miejsca.',
          'Scoring pd400: surowe cosinusy, scale=1; sonda KD czeka na S i zgodę użytkownika.', '',
          'Leaderboard polish-splade (eksport użytkownika): wiki-trivia 37.4 / 35.0 / 35.9; dbpedia 28.42; hotpotqa 55.17.', '']
p = subprocess.run([sys.executable, 'scripts/regime_task_table.py', '--models', 'base-splade-wrapped', 'soup-v9b', '--tasksets', 'wikitrivia', 'dbpediapl', 'hotpotqapl'], text=True, capture_output=True)
storage = logs/'n4_storage_status.txt'
if storage.exists():
    lines += ['## Stan dysku i wznowienia', '', storage.read_text(errors='replace'), '']
lines += [p.stdout if p.returncode == 0 else p.stderr, '', '## Scoring PolDense-400M', '']
score = logs/'score_5070_progress.txt'
lines += ['```', score.read_text(errors='replace') if score.exists() else 'Oczekuje na oba heldout8.', '```', '', '## Stan kolejki', '', '```']
state = logs/'queue_state.tsv'
lines += [state.read_text(errors='replace') if state.exists() else '', '```', '', '## FAILED', '', '```']
queue = logs/'queue.log'
lines += [*([s for s in queue.read_text(errors='replace').splitlines() if 'FAILED' in s and ('[n4]' in s or '[score]' in s)] if queue.exists() else []), '```', '']
(r/'summary_5070_n4.md').write_text('\n'.join(lines), encoding='utf-8')
PY
}
publish() {
    (
        flock -x 8
        report || return 1
        local paths=("$EXP_ROOT/queues/queue_5070_n4.sh" "$RESULTS_ROOT/summary_5070_n4.md") m ts f
        for m in night3-paired-a-s04306 night3-paired-m-s04306 base-splade-wrapped soup-v9b; do
            for ts in heldout8 wikitrivia dbpediapl hotpotqapl; do
                f="$REGIMES_ROOT/quantized/$m/$ts.json"
                [ -f "$f" ] && complete "$m" "$ts" && paths+=("$f")
            done
        done
        git add -- "${paths[@]}" || return 1
        if ! git diff --quiet HEAD -- "${paths[@]}"; then
            git commit --only -m "Update night 4 PIRB results" -- "${paths[@]}" || return 1
        fi
        # The Windows publisher watches HEAD and pushes only allowlisted paths
        # through the system credential manager (Windows interop is unavailable
        # inside this WSL session). Never export credentials to the queue.
        touch "$LOGS/n4_publish_requested"
        log 'commit ready for Windows publisher'
    ) 8> "$LOGS/n4_publish.lock"
}
if [ "${1:-}" = --publish ]; then publish; exit $?; fi
if [ "${1:-}" = --report-only ]; then report; exit $?; fi
if [ "${1:-}" = --check-storage ]; then p1_storage_ready; exit $?; fi
exec 9> "$LOGS/queue_5070_n4.lock"
flock -n 9 || { log 'n4 already running'; exit 1; }
echo $$ > "$LOGS/queue_5070_n4.pid"
guard_alive
regime_long() {
    local reg="$1" label="$2" model="$3" ts="$4" bs="${5:-8}" out="$REGIMES_ROOT/$1/$2/$4.json"
    complete "$label" "$ts" && { log "already present: $label/$ts"; return 0; }
    [ -e "$model" ] && [ -f "configs/pirb_tasks/$ts.json" ] || { log "FAILED missing model/taskset $label/$ts"; record "regime:$reg:$label:$ts" FAILED missing; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    p1_storage_ready
    local storage_rc=$?
    [ "$storage_rc" -eq 0 ] || { record "regime:$reg:$label:$ts" SKIPPED storage; return "$storage_rc"; }
    local s=$SECONDS rc
    log "regime $reg/$label/$ts batch=$bs timeout=$V28_REGIME_TIMEOUT cache=$(readlink -f "$CACHE_LINK") free=$(cache_free_gb) GiB"
    CUDA_VISIBLE_DEVICES=0 run_guarded "$V28_REGIME_TIMEOUT" bash scripts/pirb_regime_eval.sh "$reg" "$label" "$model" "$ts" "$bs" > "$LOGS/regime_${reg}_${label}_${ts}.out" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && complete "$label" "$ts"; then
        log "regime $label/$ts -> $(regime_mean "$out") in $((SECONDS-s))s"
        record "regime:$reg:$label:$ts" ok "$(regime_mean "$out")" "$((SECONDS-s))s"
    else
        log "regime $label/$ts FAILED rc=$rc after $((SECONDS-s))s"
        record "regime:$reg:$label:$ts" FAILED "$rc" "$((SECONDS-s))s"
        local target; target=$(readlink -m "$CACHE_LINK/${reg}_${label}_${ts}")
        case "$target" in "$(readlink -f "$CACHE_LINK")"/*) rm -rf -- "$target";; *) log "FAILED unsafe cache path: $target";; esac
        # Incomplete JSON must not be mistaken for a completed benchmark.
        [ "$rc" -ne 75 ] || return 75
        return 1
    fi
}
heldout_step() {
    local label="$1" dtype="$2"
    complete "$label" heldout8 && { log "already present: $label/heldout8"; return 0; }
    (
        flock -x 7
        need_space || return 1; wait_gpu_free; wait_ram
        export PIRB_DTYPE="$dtype"
        regime quantized "$label" "models/$label" heldout8 16 512
    ) 7> "$LOGS/gpu_5070.lock"
    publish || log 'FAILED updating report/publishing'
}
p1_step() {
    local label="$1" ts="$2" step_rc=0
    complete "$label" "$ts" && { log "already present: $label/$ts"; return 0; }
    while :; do
        while [ -f logs/HOLD_n4 ]; do sleep 60; done
        (
            flock -x 7
            [ -f logs/HOLD_n4 ] && exit 76
            p1_storage_ready
            local storage_rc=$?
            if [ "$storage_rc" -ne 0 ]; then
                log "SKIPPED $label/$ts: storage check failed"
                record "regime:quantized:$label:$ts" SKIPPED storage
                [ "$storage_rc" -ne 75 ] || exit 75
                exit 0
            fi
            V28_REGIME_TIMEOUT=36000
            [ "$ts" = wikitrivia ] && V28_REGIME_TIMEOUT=50400
            regime_long quantized "$label" "models/$label" "$ts" 8
        ) 7> "$LOGS/gpu_5070.lock"
        step_rc=$?
        [ "$step_rc" -eq 76 ] || break
    done
    publish || log 'FAILED updating report/publishing'
    return "$step_rc"
}
log "=== n4 start $(date -Iseconds); cache free $(cache_free_gb) GiB ==="
publish || log 'FAILED initial publishing'
heldout_step night3-paired-a-s04306 fp16
heldout_step night3-paired-m-s04306 bf16
touch "$LOGS/n4_heldout_done"
log 'both heldout8 steps finished; P1 waiting if HOLD_n4 exists'
for ts in wikitrivia dbpediapl hotpotqapl; do
    for model in base-splade-wrapped soup-v9b; do
        p1_step "$model" "$ts"
        if [ "$?" -eq 75 ]; then
            log '=== n4 stopped by disk guard; remaining tasks were not launched ==='
            exit 75
        fi
    done
done
log '=== n4 done ==='
publish || log 'FAILED final publishing'
