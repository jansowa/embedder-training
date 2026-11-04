"""Serial, resumable D0/E1/E2/E3/E4 diagnostics; never launches night 3."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/polish-splade-2026-10'
RESULTS = EXP / 'results'
CONFIGS = EXP / 'configs/next/5070'
DATA = RESULTS / 'data'
STATE = DATA / 'splade-init-state.json'
PY = str(ROOT / '.venv/bin/python')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(timespec='seconds')


def execute(command, log_name, seconds, *, disk_guard=False):
    log = ROOT / 'logs' / log_name
    started = time.monotonic()
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            reason = None
            if time.monotonic() - started > seconds:
                reason = f'timeout after {seconds}s'
            if disk_guard and shutil.disk_usage(ROOT).free < 5 * 1024**3:
                reason = 'stopped evaluation/training at <5 GiB free disk'
            if reason:
                handle.write('\n' + reason + '\n'); handle.flush()
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return {'rc': 124, 'reason': reason, 'log': str(log.relative_to(ROOT))}
            time.sleep(5)
    return {'rc': process.returncode, 'seconds': round(time.monotonic() - started, 1),
            'log': str(log.relative_to(ROOT))}


def d0():
    import gc
    import torch
    import yaml
    from training.backends import sentence_transformers_backend as backend
    from training.backends.splade_initialization import microbatch, score_diagnostics
    torch.set_num_threads(2)
    source = CONFIGS / 'modernbert_probe_m.yaml'
    cfg = yaml.safe_load(source.read_text())
    st = backend._backend_config(cfg)
    rows = backend._load_training_rows(cfg, st, config_path=str(source)).rows
    queries, docs = microbatch(rows)
    report = {'written': now(), 'description': 'Identical first 8 eligible prepared rows; eval mode, 256 tokens; '
              'BF16 forward, float32 pooled score diagnostics; no training or dropout.',
              'microbatch_sha256': hashlib.sha256(json.dumps([queries, docs], ensure_ascii=False).encode()).hexdigest(),
              'training_examples': len(rows), 'models': {}}
    stack = backend._load_sparse_sentence_transformers()
    for name in ('sdadas/polish-distilroberta', 'OPI-PIB/pl-ModernBERT-512-base'):
        torch.manual_seed(42)
        model = backend._build_splade_model(stack[1], stack[-2], stack[-1], name, st).to('cuda').eval()
        report['models'][name] = score_diagnostics(model, rows, bf16=True)
        write(DATA / 'splade-init-d0.json', report)
        del model
        gc.collect(); torch.cuda.empty_cache()


def tiny_score(path):
    result = read(path, {})
    value = result.get('mean_ndcg@10')
    if result.get('n_tasks') != 10 or value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def model_ok(report):
    models = report.get('models', [])
    if {m.get('dtype') for m in models} != {'fp32', 'bf16'}:
        return False
    return all(not m['nonfinite_weights'] and all(
        o['pooled_all_finite'] and all(v > 0 for v in o['valid_token_active_dims_per_text'])
        for o in m['outputs']) for m in models)


def make_config(name, *, regularized=False, head_only=False, source=None):
    import yaml
    cfg = yaml.safe_load((CONFIGS / 'modernbert_probe_m.yaml').read_text())
    cfg['runs_dir'] = f'runs/modernbert-init-{name}'
    cfg['timestamp_output_dir'] = True
    cfg['hparams'][0].update(learning_rate=2e-5, document_regularizer_weight=0.003 if regularized else 0.0)
    if source:
        cfg['architectures'] = [str(source)]
    st = cfg['sentence_transformers']
    st.update(max_steps=500, warmup_ratio=0.5, scale=1.0, seed=42,
              train_batch_size=8, gradient_accumulation_steps=4,
              save_steps=250, save_total_limit=2, keep_step_checkpoints=[250, 500],
              report_to=[], splade_regularizer_scheduler=False,
              query_regularizer_weight=0.00015 if regularized else 0.0,
              splade_train_only_head=head_only, splade_head_learning_rate=2e-5)
    st['splade_activation_stats'].update(interval_steps=50, batch_size=8, stop_below=5)
    if regularized:
        st['regularizer_control']['warmup_steps'] = 1000
    else:
        st.pop('regularizer_control', None)
    if not source:
        st.update(splade_decoder_bias_shift='auto', splade_decoder_bias_target=400)
    else:
        st.pop('splade_decoder_bias_shift', None)
    path = CONFIGS / f'modernbert_init_{name}.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def export_model(source, target):
    if target.exists():
        return
    temporary = target.with_name(target.name + '.tmp')
    if temporary.exists():
        raise RuntimeError(f'Incomplete model export exists: {temporary}; preserving it')
    shutil.copytree(source, temporary,
                    ignore=shutil.ignore_patterns('optimizer.pt', 'scheduler.pt', 'rng_state*.pth'))
    if not (temporary / 'model.safetensors').is_file():
        raise RuntimeError(f'No model weights in {source}')
    temporary.rename(target)


def copy_trial(name, e1=False):
    directory = ROOT / ('runs/modernbert-control-noreg-scale1e-4' if e1 else f'runs/modernbert-init-{name}')
    finals = sorted(directory.glob('**/final/model.safetensors'), key=lambda p:p.stat().st_mtime)
    checkpoints = sorted(directory.glob('**/checkpoint-500/model.safetensors'), key=lambda p:p.stat().st_mtime)
    found = finals or checkpoints
    if not found:
        return None
    run = found[-1].parent.parent
    states = sorted(run.glob('checkpoint-*/trainer_state.json'), key=lambda p:int(p.parent.name.split('-')[-1]))
    saved_state = read(run/'control_train_state.json') or (read(states[-1], {}) if states else {})
    completed_steps = saved_state.get('global_step', 0)
    for step in (250, 500):
        source = run / 'step-checkpoints' / f'step-{step}'
        if not source.is_dir():
            source = run / f'checkpoint-{step}'
        if step == 500 and not source.is_dir() and completed_steps == 500 and (run / 'final').is_dir():
            source = run / 'final'
        if source.is_dir():
            export_model(source, ROOT / f'models/modernbert-init-{name}-s{step:05d}')
    return run


def cleanup_eval_cache(label):
    cache = ROOT / 'cache/regimes' / f'quantized_{label}_tiny10'
    # Only this experiment's generated evaluation workspace; never model/data dirs.
    if cache.name.startswith('quantized_modernbert-init-') and cache.parent.resolve() == (ROOT / 'cache/regimes').resolve():
        if cache.exists():
            shutil.rmtree(cache)


def measure(name, entry):
    label = f'modernbert-init-{name}'
    model = ROOT / f'models/{label}-s00500'
    output = DATA / f'{label}-diagnosis.json'
    if model.is_dir() and not output.exists():
        entry['diagnostic_process'] = execute([PY, 'scripts/diagnose_splade_checkpoint.py', '--model',
            f'{label}={model}', '--output', str(output)], f'init_{name}_diagnosis.log', 900)
    diagnosis = read(output, {})
    entry['nonzero_fp32_bf16'] = model.is_dir() and model_ok(diagnosis)
    for step in (250, 500):
        model_step = ROOT / f'models/{label}-s{step:05d}'
        if not model_step.is_dir():
            continue
        step_label = f'{label}-s{step:05d}'
        sparsity_path = RESULTS / 'sparsity' / f'{step_label}.json'
        if not sparsity_path.exists():
            entry[f'sparsity_process_{step}'] = execute([PY, 'scripts/compare_splade_sparsity.py',
                '--model', f'{step_label}={model_step}', '--scope', 'tiny', '--max-docs-per-task', '300',
                '--device', 'cuda', '--batch-size', '8', '--no-fp16', '--output-json', str(sparsity_path)],
                f'init_{name}_s{step}_sparsity.log', 1200)
        sparse = read(sparsity_path, {}).get(step_label, {})
        entry[f'dims_{step}'] = sparse.get('ALL', {}).get('mean')
        if entry[f'dims_{step}'] is None:
            values = [v['mean'] for k,v in sparse.items() if k != 'ALL' and v.get('mean') is not None]
            if values:
                entry[f'dims_{step}'] = sum(values)/len(values)
        if step == 500 or name == 'e3':
            tiny = EXP / 'results-regimes/quantized' / step_label / 'tiny10.json'
            if tiny_score(tiny) is None:
                entry[f'tiny_process_{step}'] = execute(['bash', 'scripts/pirb_regime_eval.sh', 'quantized',
                    step_label, str(model_step), 'tiny10', '8', '512'],
                    f'init_{name}_s{step}_tiny10.log', 5400, disk_guard=True)
            entry[f'tiny10_{step}'] = tiny_score(tiny)
            cleanup_eval_cache(step_label)
    entry['stable_training'] = bool(entry['nonzero_fp32_bf16'] and entry.get('trained_steps') == 500)
    entry['verdict'] = 'nonzero; quality/density assessed separately' if entry['stable_training'] else 'FAIL: incomplete/collapsed training'


def training_metadata(run, entry):
    entry['run'] = str(run.relative_to(ROOT))
    entry['initialization'] = read(run / 'splade_initialization.json')
    state = read(run / 'control_train_state.json') or read(run / 'checkpoint-500/trainer_state.json', {})
    if not state:
        states = sorted(run.glob('checkpoint-*/trainer_state.json'), key=lambda p:int(p.parent.name.split('-')[-1]))
        state = read(states[-1], {}) if states else {}
    history = state.get('log_history', [])
    entry['trained_steps'] = state.get('global_step')
    rank = [r for r in history if 'base_loss' in r]
    entry['rank_loss_500'] = rank[-1]['base_loss'] if rank and rank[-1].get('step') == 500 else None
    entry['density_stop'] = read(run / 'splade_density_stop.json')


def report(state):
    def fmt(value):
        return '—' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)
    lines = ['# SPLADE initialization: D0, E1–E4', '', f'Updated: {now()}', '',
             'All new trials: stock head, BF16, physical batch 8 × accumulation 4, 256 tokens, six negatives, '
             'same five data files. E2–E4: scale=1, backbone/head LR=2e-5, LR warmup 250 steps. '
             'E1 is diagnostic scale=0.0001, LR=5e-5, no regularization. '
             'Sparsity: FP32, 512 tokens, 300 passages per tiny task; tiny10: BF16, quantized, 512 tokens. '
             'Only complete 10-task means count as tiny10 scores.', '',
             '## D0: identical microbatch, no training', '',
             '| Model | Scores min / median / max | Median active doc dims | MNRL scale=1 |',
             '|---|---|---:|---:|']
    baseline = read(DATA / 'splade-init-d0.json', {})
    for name, d in baseline.get('models', {}).items():
        lines.append(f"| {name} | {fmt(d['score_min'])} / {fmt(d['score_median'])} / {fmt(d['score_max'])} | {fmt(d['document_dims_median'])} | {fmt(d['loss'])} |")
    lines += ['', 'D0 uses the same first eight eligible prepared rows for both models, eval mode (no dropout), '
              'BF16 forward and float32 score diagnostics. The input SHA-256 is in the raw JSON. '
              'This is a controlled initial-condition comparison, not proof of the entire causal mechanism.', '',
              '## Trials', '', '| run | c | initial scores median / max | initial loss | dims s250 / s500 | rank loss s500 | tiny10 s500 | verdict |',
              '|---|---:|---|---:|---|---:|---:|---|']
    for name, e in state.get('experiments', {}).items():
        init = e.get('initialization') or {}
        scores = f"{fmt(init.get('score_median'))} / {fmt(init.get('score_max'))}"
        lines.append(f"| {name} | {fmt(init.get('c'))} | {scores} | {fmt(init.get('loss'))} | "
                     f"{fmt(e.get('dims_250'))} / {fmt(e.get('dims_500'))} | {fmt(e.get('rank_loss_500'))} | "
                     f"{fmt(e.get('tiny10_500'))} | {e.get('verdict', e.get('phase', 'pending'))} |")
    lines += ['', f"Recipe verdict: **{state.get('recipe_verdict', 'pending')}**", '',
              'Stable E3 requires: completed 500 steps; finite nonzero FP32/BF16 diagnostic outputs; '
              'mean document dimensions in [200,600]; complete tiny10@s500 at least 1.0 nDCG percentage point '
              'above E1; and E3 tiny10@s500 > tiny10@s250. The +1.0 threshold is the operational '
              'interpretation of “clearly above E1”.', '',
              'Head-only E4 preserves tied decoder/input-embedding parameters: those shared weights remain '
              'trainable; no architecture change or untying is performed. E4-unfrozen starts a new optimizer '
              'and 250-step LR warmup from E4-head weights and does not apply the bias shift again.', '',
              'E1 initialization scores, if not explicitly measured before its original training, are not '
              'invented or substituted with D0 scale=1 loss. Full process results and exceptions are in '
              '`results/data/splade-init-state.json` and logs/init_*. No night-3 training is launched.']
    if state.get('errors'):
        lines += ['', 'Errors/incomplete steps:', *[f'- {e}' for e in state['errors']]]
    (RESULTS / 'summary_5070_init.md').write_text('\n'.join(lines) + '\n')
    state['updated'] = now()
    write(STATE, state)


def commit_results(label):
    paths = ['experiments/polish-splade-2026-10/results/summary_5070_init.md',
             'experiments/polish-splade-2026-10/results/data/splade-init-state.json']
    paths += [str(p.relative_to(ROOT)) for p in DATA.glob('modernbert-init-*-diagnosis.json')]
    paths += [str(p.relative_to(ROOT)) for p in DATA.glob('splade-init-d0.json')]
    paths += [str(p.relative_to(ROOT)) for p in (RESULTS/'sparsity').glob('modernbert-init-*.json')]
    paths += [str(p.relative_to(ROOT)) for p in (EXP/'results-regimes/quantized').glob('modernbert-init-*/tiny10.json')]
    paths += [str(p.relative_to(ROOT)) for p in CONFIGS.glob('modernbert_init_*.yaml')]
    with (ROOT/'logs/init_commits.log').open('a') as log:
        subprocess.run(['git', 'add', '--', *paths], stdout=log, stderr=log, cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '--only', '-m', f'Record SPLADE initialization {label}', '--', *paths],
                       stdout=log, stderr=log, cwd=ROOT)
    execute(['git', 'push', 'origin', 'master'], f'init_push_{label}.log', 45)


def trial(state, name, *, e1=False, regularized=False, head_only=False, source=None):
    e = state['experiments'].setdefault(name, {})
    if e.get('phase') == 'finished':
        return e
    e['phase'] = 'training' if not e1 else 'measuring existing E1'
    report(state)
    config = None if e1 else make_config(name, regularized=regularized, head_only=head_only, source=source)
    model = ROOT/f'models/modernbert-init-{name}-s00500'
    if not e1 and not model.exists():
        e['train_process'] = execute([PY, '-m', 'training.train', '--backend', 'sentence-transformers',
            '--training-type', 'splade', '--config', str(config)], f'init_{name}_train.log', 3600, disk_guard=True)
    run = copy_trial(name, e1=e1)
    if run:
        training_metadata(run, e)
    else:
        candidates = sorted((ROOT/f'runs/modernbert-init-{name}').glob('**/resolved_config.yaml'), key=lambda p:p.stat().st_mtime)
        if candidates:
            training_metadata(candidates[-1].parent, e)
    e['phase'] = 'measurement'; report(state)
    measure(name, e)
    e['phase'] = 'finished'; report(state)
    commit_results(name)
    return e


def prepare_night3():
    import yaml
    cfg = yaml.safe_load((CONFIGS/'modernbert_init_e3.yaml').read_text())
    cfg['runs_dir'] = 'runs/modernbert-night3-fresh'
    cfg['train_data'] = [{'path':'datasets/arms/squadv2-v2-all.jsonl'}]
    cfg['hparams'][0]['num_train_epochs'] = 1
    st = cfg['sentence_transformers']
    st.update(max_steps=-1, warmup_ratio=0.0, warmup_steps=250, save_steps=2500, keep_step_checkpoints=[])
    path = CONFIGS/'modernbert_night3_fresh_PREPARED_NOT_STARTED.yaml'
    path.write_text('# Prepared only; never launched by this queue. Fresh input must exist and be validated.\n'
                    + yaml.safe_dump(cfg, sort_keys=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d0', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.d0:
        d0(); return
    state = read(STATE, {'experiments': {}, 'errors': []})
    report(state)
    if not (DATA/'splade-init-d0.json').exists():
        state['d0_process'] = execute([PY, 'scripts/splade_init_experiments.py', '--d0'], 'init_d0.log', 900)
        report(state)
    commit_results('d0')
    for name, options in [('e1', {'e1':True}), ('e2', {}), ('e3', {'regularized':True})]:
        try:
            trial(state, name, **options)
        except Exception as exc:
            state['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
            state['experiments'].setdefault(name, {})['verdict'] = 'ERROR; see state/logs'
            report(state)
    if not state['experiments'].get('e2', {}).get('stable_training', False):
        try:
            head = trial(state, 'e4-head', head_only=True)
            if head.get('stable_training'):
                trial(state, 'e4-unfrozen', source=ROOT/'models/modernbert-init-e4-head-s00500')
            else:
                state['errors'].append('E4 head-only failed; no usable s500 to unfreeze. Precision test is a next option, not launched.')
        except Exception as exc:
            state['errors'].append(f'E4: {type(exc).__name__}: {exc}')
    e1, e3 = (state['experiments'].get(k, {}) for k in ('e1', 'e3'))
    stable = (e3.get('stable_training') and 200 <= (e3.get('dims_500') or 0) <= 600
              and all(e.get('tiny10_500') is not None for e in (e1,e3))
              and e3['tiny10_500'] >= e1['tiny10_500'] + 1.0
              and e3.get('tiny10_250') is not None and e3['tiny10_500'] > e3['tiny10_250'])
    state['recipe_verdict'] = 'STABLE under specified short-probe criteria' if stable else 'NOT ESTABLISHED; no night-3 launch'
    if stable:
        prepare_night3()
    report(state)
    commit_results('final')
    print(state['recipe_verdict'], flush=True)


if __name__ == '__main__':
    main()
