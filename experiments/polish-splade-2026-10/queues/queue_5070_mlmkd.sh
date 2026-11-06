#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=mlmkd MIN_RAM_GB=6 WANDB_MODE=offline TRAIN_TIMEOUT=43200 TZ=Europe/Warsaw
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
ulimit -c 0
source scripts/queue_lib.sh
CFG="$EXP_ROOT/configs/next/5070/mlm_kd_probe_a.yaml"
publish_probe() {
    "$PY" - <<'PY'
import json
from datetime import datetime
from pathlib import Path
e = Path('experiments/polish-splade-2026-10'); r=e/'results'; g=e/'results-regimes/quantized'
def read(p):
    try: return json.loads(p.read_text())
    except (OSError, ValueError): return {}
smoke = read(Path('logs/mlmkd_smoke_stats.json'))
lines = ['# Sonda A od MLM z destylacją PolDense-400M', '', f'Aktualizacja: {datetime.now().astimezone().isoformat(timespec="seconds")}', '',
         'Dawka docelowa: te same 137 773 przykłady po bramce i rozwinięciu pozytywów co w nocy 3. KL, T=2, bez normalizacji wyników; surowe cosinusy z siedmiu plików -tpd400.', '',
         'S=121.2 jest wartością tymczasową do smoke. Pełny trening wymaga podania S i osobnego „tak” użytkownika.', '',
         '| Model | krok | tiny10 przy 512 | wym./dok FP32 przy 512 |', '|---|---:|---:|---:|',
         '| A MNRL, noc 3 | 2150 | 55.14 | 339.06 |', '| A MNRL, noc 3 | 4306 | 56.73 | 459.13 |',
         '| ModernBERT MNRL, noc 3 | 4306 | 53.75 | 327.85 |', '| baza | — | 61.60 | — |']
final_score = final_dims = None
for step in (2150, 4306):
    label=f'mlm-kd-probe-a-s{step:05d}'
    scores=read(g/label/'tiny10.json'); sparse=read(r/'sparsity'/f'{label}.json').get(label, {})
    dims=sparse.get('ALL', {}).get('mean')
    if dims is None:
        values=[v['mean'] for k,v in sparse.items() if k != 'ALL' and v.get('mean') is not None]
        dims=sum(values)/len(values) if values else None
    score=scores.get('mean_ndcg@10') if scores.get('n_tasks') == 10 else None
    cells=[f'{x:.2f}' if x is not None else '—' for x in (score,dims)]
    lines.append(f'| A KD | {step} | {cells[0]} | {cells[1]} |')
    if step==4306: final_score, final_dims=score,dims
if final_score is None or final_dims is None: verdict='Oczekuje na zatwierdzony pełny trening i pomiary.'
elif not 327 <= final_dims <= 458: verdict='Nieporównywalne przy tej gęstości.'
elif final_score >= 59.5: verdict='≥59.5: nauczyciel zmienia nachylenie; dalszy tydzień wymaga decyzji użytkownika.'
elif final_score < 58.0: verdict='<58.0: ścieżka zamknięta na ten tydzień, karta wraca do P1.'
else: verdict='58.0–59.5: jedna noc tej samej sondy na ModernBERT przed decyzją.'
lines += ['', f'Werdykt: {verdict}', '', 'Progi pozostają stałe: pasmo 327–458; ≥59.5 / <58.0 / pomiędzy. Punkt kontrolny ewentualnego dalszego biegu: ≥61.6 przy około 500 k przykładów.', '', '## Smoke 20 kroków', '']
if smoke:
    lines += [f"S smoke: {smoke['score_scale']}; kroki: {smoke['steps']}; czas kroku: {smoke['seconds_per_step']:.2f} s; peak CUDA: {smoke['peak_allocated_mib']:.0f} MiB; peak NVIDIA sampled: {smoke['peak_nvidia_sampled_mib']:.0f} MiB.", '', 'Run smoke usunięty po zachowaniu telemetrii. save_strategy=no, report_to=[], WANDB_MODE=disabled.']
else: lines += ['Oczekuje na ukończenie smoke.']
training=read(Path('logs/mlmkd_train_stats.json'))
if training: lines += ['', f"Pełny trening: S={training['score_scale']}; ukończono {training['steps']} kroków."]
lines += ['', 'Run sondy i smoke na WSL. Limit nowego zużycia przestrzeni podczas treningu: 25 GiB (smoke: 4 GiB), w ramach odzyskanych 30 GiB. P1 pozostaje wstrzymane z braku 60 GiB fizycznego miejsca i modelu soup-v9b.', '']
(r/'summary_5070_mlmkd.md').write_text('\n'.join(lines))
PY
    [ "$?" -eq 0 ] || return 1
    (
        flock -x 8
        local paths=("$EXP_ROOT/queues/queue_5070_mlmkd.sh" "$EXP_ROOT/configs/next/5070/mlm_kd_probe_a.yaml" "$RESULTS_ROOT/summary_5070_mlmkd.md") f step
        f="$RESULTS_ROOT/data/mlm-kd-probe-a-regulator.json"; [ ! -f "$f" ] || paths+=("$f")
        for step in 02150 04306; do
            f="$RESULTS_ROOT/sparsity/mlm-kd-probe-a-s$step.json"; [ ! -f "$f" ] || paths+=("$f")
            f="$REGIMES_ROOT/quantized/mlm-kd-probe-a-s$step/tiny10.json"; [ ! -f "$f" ] || paths+=("$f")
        done
        git add -- "${paths[@]}" || return 1
        git diff --quiet HEAD -- "${paths[@]}" || git commit --only -m 'Prepare MLM distillation probe' -- "${paths[@]}" || return 1
        touch "$LOGS/n4_publish_requested"
    ) 8> "$LOGS/n4_publish.lock"
}
SMOKE=0
if [ "${1:-}" = --smoke ]; then
    SMOKE=1; CFG=logs/mlmkd_smoke.yaml
    export QUEUE_TAG=mlmkd-smoke WANDB_MODE=disabled
else
    [ "${MLMKD_APPROVED:-}" = yes ] && [ -n "${MLMKD_SCORE_SCALE:-}" ] || {
        log 'Not started: full probe requires explicit user approval and MLMKD_SCORE_SCALE=S'; exit 64;
    }
    [ -f logs/mlmkd_smoke_passed ] || { log 'Not started: smoke has not passed'; exit 64; }
fi
exec 9> "$LOGS/queue_5070_mlmkd.lock"
flock -n 9 || { log 'mlmkd already running'; exit 1; }
echo $$ > "$LOGS/queue_5070_mlmkd.pid"
guard_alive
wait_gpu_free() { while [ "$(gpu_busy)" -ne 0 ]; do sleep 30; done; sleep 10; [ "$(gpu_busy)" -eq 0 ] || wait_gpu_free; }
[ ! -e logs/HOLD_n4 ] || { log 'Not started: HOLD_n4 belongs to another job'; exit 75; }
touch logs/HOLD_n4
trap 'rm -f logs/HOLD_n4' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec 7> "$LOGS/gpu_5070.lock"
flock -x 7
need_space || exit 75
wait_gpu_free; wait_ram
if [ "$SMOKE" = 0 ]; then
    "$PY" - "$CFG" "$MLMKD_SCORE_SCALE" <<'PY'
import math, sys
from pathlib import Path
import yaml
p = Path(sys.argv[1]); scale = float(sys.argv[2])
assert math.isfinite(scale) and scale > 0
cfg = yaml.safe_load(p.read_text()); cfg['sentence_transformers']['score_scale'] = scale
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
# Only this new run is placed in WSL; existing runs on C: are preserved.
link = Path(cfg['runs_dir'])
target = Path('cache').resolve()/'mlmkd-runs'
target.mkdir(exist_ok=True)
if not link.exists() and not link.is_symlink():
    link.symlink_to(target, target_is_directory=True)
assert link.is_symlink() and link.resolve() == target, 'Unexpected run location; stop before training'
PY
    [ "$?" -eq 0 ] || exit 1
fi
log "training mode smoke=$SMOKE config=$CFG (WSL allocation budget <=25 GiB; no resume)"
# The parent owns the GPU lock; this wrapper forwards termination to all workers.
if [ "$SMOKE" = 0 ] && [ -f models/mlm-kd-probe-a-s02150/model.safetensors ] && [ -f models/mlm-kd-probe-a-s04306/model.safetensors ] && [ -f "$RESULTS_ROOT/data/mlm-kd-probe-a-regulator.json" ]; then
    log 'training already complete; using exported checkpoints'
    rc=0
else
"$PY" - "$CFG" "$SMOKE" <<'PY'
import json, os, shutil, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import yaml

cfg_path, smoke = Path(sys.argv[1]), sys.argv[2] == '1'
cfg = yaml.safe_load(cfg_path.read_text())
root = Path.cwd(); exp = root/'experiments/polish-splade-2026-10'
run_root = (root/cfg['runs_dir']).resolve()
assert run_root.is_relative_to((root/'cache').resolve()), 'Run must stay within the WSL cache budget'
existing = set(run_root.glob('**/resolved_config.yaml'))
baseline_free = shutil.disk_usage(root/'cache').free
log_name = root/'logs'/('mlmkd_smoke_train.log' if smoke else 'mlmkd_train.log')
active = None
def terminate(sig, frame):
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)
        try: active.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(active.pid, signal.SIGKILL); active.wait()
    raise SystemExit(128+sig)
signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
started = time.monotonic()
with log_name.open('w') as log:
    active = subprocess.Popen([str(root/'.venv/bin/python'), 'scripts/run_splade_collapse_control.py', '--monitor-only', '--backend', 'sentence-transformers', '--training-type', 'splade', '--config', str(cfg_path)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while active.poll() is None:
        if time.monotonic()-started > (3600 if smoke else 43200):
            terminate(signal.SIGTERM, None)
        if baseline_free-shutil.disk_usage(root/'cache').free > (4 if smoke else 25)*2**30:
            print('WSL allocation budget exceeded; stopping worker', flush=True)
            terminate(signal.SIGTERM, None)
        time.sleep(5)
rc = active.returncode
new = [p for p in run_root.glob('**/resolved_config.yaml') if p not in existing]
assert len(new) == 1, f'Expected one new run: {new}; see {log_name}'
run = new[0].parent
state = json.loads((run/'control_train_state.json').read_text()) if (run/'control_train_state.json').exists() else {}
expected = 20 if smoke else 4306
assert rc == 0 and state.get('global_step') == expected, f'Incomplete training rc={rc} step={state.get("global_step")}; see {log_name}'
telemetry = [json.loads(line) for line in (run/'collapse_control.jsonl').read_text().splitlines()]
measured = [r for r in telemetry if 'grad_norm' in r.get('metrics', {})]
pace = None
if len(measured) >= 2:
    first, last = measured[0], measured[-1]
    pace = (last['elapsed_train_seconds']-first['elapsed_train_seconds'])/(last['step']-first['step'])
sampled = [float(r['gpu'].split(',')[0]) for r in telemetry if r.get('gpu')]
stats = {'completed_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'smoke': smoke,
         'score_scale': cfg['sentence_transformers']['score_scale'], 'steps': expected,
         'seconds_per_step': pace, 'wall_seconds': time.monotonic()-started,
         'peak_allocated_mib': max(r.get('peak_allocated_mib', 0) for r in telemetry),
         'peak_nvidia_sampled_mib': max(sampled, default=0), 'run': str(run)}
(root/'logs'/('mlmkd_smoke_stats.json' if smoke else 'mlmkd_train_stats.json')).write_text(json.dumps(stats, indent=2)+'\n')
print(json.dumps(stats, indent=2), flush=True)
if smoke:
    # Remove only this invocation's temporary run after preserving telemetry/stats.
    shutil.copyfile(run/'collapse_control.jsonl', root/'logs/mlmkd_smoke_telemetry.jsonl')
    assert run.is_relative_to((root/'cache/mlmkd-smoke').resolve())
    shutil.rmtree(run)
    (root/'logs/mlmkd_smoke_passed').write_text(stats['completed_at']+'\n')
else:
    for step in (2150, 4306):
        source = run/'step-checkpoints'/f'step-{step}'
        step_state = json.loads((source/'trainer_state.json').read_text())
        assert step_state['global_step'] == step
        target = root/'models'/f'mlm-kd-probe-a-s{step:05d}'
        if target.exists():
            raise RuntimeError(f'Refuse to replace existing checkpoint: {target}')
        temporary = target.with_name(target.name+'.tmp')
        shutil.copytree(source, temporary, ignore=shutil.ignore_patterns('optimizer.pt', 'scheduler.pt', 'rng_state*.pth'))
        assert (temporary/'model.safetensors').is_file()
        temporary.rename(target)
    shutil.copyfile(run/'regularizer_control.json', exp/'results/data/mlm-kd-probe-a-regulator.json')
    # Training is finished and requested model copies are durable.
    for path in run.rglob('*'):
        if path.is_file() and path.name in ('optimizer.pt', 'scheduler.pt'):
            path.unlink()
PY
rc=$?
fi
if [ "$rc" -ne 0 ]; then log "FAILED mlmkd training/smoke rc=$rc"; record mlmkd FAILED "$rc"; exit "$rc"; fi
if [ "$SMOKE" = 1 ]; then
    log 'smoke passed; temporary run removed; full probe still awaits S and approval'
    record mlmkd-smoke ok 20
    publish_probe || log 'FAILED publishing probe smoke report'
    exit 0
fi
for step in 02150 04306; do
    label="mlm-kd-probe-a-s$step"
    need_space || continue; wait_gpu_free; wait_ram
    if [ ! -f "$RESULTS_ROOT/sparsity/$label.json" ]; then
        timeout -k 60 1800 "$PY" scripts/compare_splade_sparsity.py --model "$label=models/$label" --scope tiny --max-docs-per-task 300 --device cuda --batch-size 16 --no-fp16 --output-json "$RESULTS_ROOT/sparsity/$label.json" > "$LOGS/sparsity_$label.log" 2>&1 || { log "FAILED sparsity $label"; record "sparsity:$label" FAILED -; }
    fi
    regime quantized "$label" "models/$label" tiny10 16 512 || log "FAILED tiny10 $label"
    publish_probe || log 'FAILED publishing probe measurement report'
done
log 'mlmkd measurements finished; results ready for report and publication'
