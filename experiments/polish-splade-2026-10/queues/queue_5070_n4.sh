#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=n4 MIN_RAM_GB=6 WANDB_MODE=offline TZ=Europe/Warsaw
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
source scripts/queue_lib.sh
CACHE_LINK=cache/regimes
cache_free_gb() { df -B1 --output=avail "$(readlink -f "$CACHE_LINK")" | tail -1 | awk '{print int($1/1073741824)}'; }
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
          'P1: quantized, batch 8; wiki-trivia 14 h timeout, pozostałe 10 h. Wymagane ≥60 GiB na cache.',
          'Scoring pd400: surowe cosinusy, scale=1; sonda KD czeka na S i zgodę użytkownika.', '',
          'Leaderboard polish-splade (eksport użytkownika): wiki-trivia 37.4 / 35.0 / 35.9; dbpedia 28.42; hotpotqa 55.17.', '']
p = subprocess.run([sys.executable, 'scripts/regime_task_table.py', '--models', 'base-splade-wrapped', 'soup-v9b', '--tasksets', 'wikitrivia', 'dbpediapl', 'hotpotqapl'], text=True, capture_output=True)
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
                [ -f "$f" ] && paths+=("$f")
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
exec 9> "$LOGS/queue_5070_n4.lock"
flock -n 9 || { log 'n4 already running'; exit 1; }
echo $$ > "$LOGS/queue_5070_n4.pid"
guard_alive
regime_long() {
    local reg="$1" label="$2" model="$3" ts="$4" bs="${5:-8}" out="$REGIMES_ROOT/$1/$2/$4.json"
    complete "$label" "$ts" && { log "already present: $label/$ts"; return 0; }
    [ -e "$model" ] && [ -f "configs/pirb_tasks/$ts.json" ] || { log "FAILED missing model/taskset $label/$ts"; record "regime:$reg:$label:$ts" FAILED missing; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    local s=$SECONDS rc
    log "regime $reg/$label/$ts batch=$bs timeout=$V28_REGIME_TIMEOUT cache=$(readlink -f "$CACHE_LINK") free=$(cache_free_gb) GiB"
    CUDA_VISIBLE_DEVICES=0 timeout -k 60 "$V28_REGIME_TIMEOUT" bash scripts/pirb_regime_eval.sh "$reg" "$label" "$model" "$ts" "$bs" > "$LOGS/regime_${reg}_${label}_${ts}.out" 2>&1
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
        return 1
    fi
}
heldout_step() {
    local label="$1" dtype="$2"
    (
        flock -x 7
        need_space || return 1; wait_gpu_free; wait_ram
        export PIRB_DTYPE="$dtype"
        regime quantized "$label" "models/$label" heldout8 16 512
    ) 7> "$LOGS/gpu_5070.lock"
    publish || log 'FAILED updating report/publishing'
}
p1_step() {
    local label="$1" ts="$2"
    complete "$label" "$ts" && { log "already present: $label/$ts"; return 0; }
    while :; do
        while [ -f logs/HOLD_n4 ]; do sleep 60; done
        (
            flock -x 7
            [ -f logs/HOLD_n4 ] && exit 75
            local free; free=$(cache_free_gb)
            if [ "${free:-0}" -lt 60 ]; then
                log "SKIPPED $label/$ts: only $free GiB on cache disk (need 60)"
                record "regime:quantized:$label:$ts" SKIPPED "disk:$free"; exit 0
            fi
            V28_REGIME_TIMEOUT=36000
            [ "$ts" = wikitrivia ] && V28_REGIME_TIMEOUT=50400
            regime_long quantized "$label" "models/$label" "$ts" 8
        ) 7> "$LOGS/gpu_5070.lock"
        [ "$?" -eq 75 ] || break
    done
    publish || log 'FAILED updating report/publishing'
}
log "=== n4 start $(date -Iseconds); cache free $(cache_free_gb) GiB ==="
publish || log 'FAILED initial publishing'
heldout_step night3-paired-a-s04306 fp16
heldout_step night3-paired-m-s04306 bf16
touch "$LOGS/n4_heldout_done"
log 'both heldout8 steps finished; P1 waiting if HOLD_n4 exists'
for ts in wikitrivia dbpediapl hotpotqapl; do
    for model in base-splade-wrapped soup-v9b; do p1_step "$model" "$ts"; done
done
log '=== n4 done ==='
publish || log 'FAILED final publishing'
