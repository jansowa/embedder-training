"""Collect the probe curve and write its nightly decision report without GPU imports."""
from __future__ import annotations

import json
from pathlib import Path
import re
from datetime import datetime
import yaml

ROOT = Path('experiments/polish-splade-2026-10')
STEPS = [625, 1250, 1875, 2500, 5000, 10000, 15000, 18750]
FIELDS = ['step', 'epoch', 'loss', 'base_loss', 'document_regularizer_loss',
          'query_regularizer_loss', 'learning_rate', 'document_weight', 'query_weight',
          'document_dims_mean', 'document_dims_reference',
          'splade_activation_stats_document_active_dims_mean']


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def curve():
    config = yaml.safe_load((ROOT / 'configs/next/5070/mlm_probe_a_fallback.yaml').read_text())
    backend = config['sentence_transformers']
    candidates = list(Path('runs').glob('arm-mlm-probe-a*/sentence-transformers/splade/*/step-checkpoints/step-*/trainer_state.json'))
    candidates += list(Path('runs').glob('arm-mlm-probe-a*/sentence-transformers/splade/*/checkpoint-*/trainer_state.json'))
    candidates += list(Path('runs').glob('arm-mlm-probe-a*/sentence-transformers/splade/*/trainer_state.json'))
    states = [(read(p), p) for p in candidates]
    state, path = max(states, key=lambda item: (item[0].get('global_step', 0), item[1].stat().st_mtime), default=({}, None))
    rows = []
    aliases = {
        'base_loss': ['train/base_loss', 'train_base_loss'],
        'document_regularizer_loss': ['train/document_regularizer_loss', 'train_document_regularizer_loss'],
        'query_regularizer_loss': ['train/query_regularizer_loss', 'train_query_regularizer_loss'],
        'splade_activation_stats_document_active_dims_mean': ['train/splade_activation_stats/document_active_dims_mean'],
    }
    for entry in state.get('log_history', []):
        row = {}
        for field in FIELDS:
            keys = [field, f'train/regularizer/{field}'] + aliases.get(field, [])
            row[field] = next((entry[k] for k in keys if k in entry), None)
        rows.append(row)
    timing = read('logs/mlm_probe_timing.json')
    report = {'variant': 'fallback_fixed_lambda', 'examples_per_step': 32,
              'physical_batch_size': backend['train_batch_size'],
              'gradient_accumulation_steps': backend['gradient_accumulation_steps'],
              'gradient_checkpointing': backend['gradient_checkpointing'],
              'splade_pooling_chunk_size': backend['splade_pooling_chunk_size'],
              'document_weight_config': 0.003, 'query_weight_config': 0.00015,
              'trainer_state': str(path) if path else None, 'global_step': state.get('global_step'),
              'timing': timing, 'entries': rows}
    target = ROOT / 'results/data/mlm-probe-a-curve.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def sparsity(name):
    values = read(ROOT / f'results/sparsity/{name}.json').get(name, {})
    dims = [entry['mean'] for entry in values.values() if entry.get('mean') is not None]
    return sum(dims) / len(dims) if dims else None


def fmt(value):
    return '—' if value is None else f'{value:.2f}' if isinstance(value, (int, float)) else str(value)


def main():
    report = curve()
    lines = [f'# Sonda MLM — {datetime.now().astimezone().isoformat(timespec="minutes")}', '',
             'Reżim leaderboardowy quantized, pomiary przy 512 tokenach.', '',
             '**Wariant zapasowy: stałe λ**, λ_d=0.003, λ_q=0.00015; bez regulatora i progów L0.', '',
             f"Batch fizyczny: **{report['physical_batch_size']}**, akumulacja: {report['gradient_accumulation_steps']}, efektywny batch: 32; "
             f"gradient checkpointing: {report['gradient_checkpointing']}, pooling chunk: {report['splade_pooling_chunk_size']}.", '',
             'Przebieg z batchem 16 przerwano z powodu przekroczenia VRAM i używania pamięci współdzielonej. '
             'Jego checkpointy usunięto; wyniki sondy dotyczą świeżego przebiegu z mniejszym batchem.', '',
             '| krok | przykłady | tiny10 | wym./dok (PIRB) | λ_d | gęstość próbki / baza | mkqa | pairs | templates |',
             '|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    results = {}
    for step in STEPS:
        name = f'mlm-probe-a-s{step:05d}'
        score = read(ROOT / f'results-regimes/quantized/{name}/tiny10.json')
        if score.get('n_tasks') != 10 or len(score.get('tasks', {})) != 10:
            continue
        tasks = score.get('tasks', {})
        def task(suffix):
            return next((v for k, v in tasks.items() if suffix in k), None)
        near = [row for row in report['entries'] if (row.get('step') or 0) <= step]
        def last(field):
            return next((row[field] for row in reversed(near) if row.get(field) is not None), None)
        density = last('splade_activation_stats_document_active_dims_mean')
        result = {'tiny': score.get('mean_ndcg@10'), 'dims': sparsity(name),
                  'mkqa': task('mkqa'), 'pairs': task('pairs'), 'templates': task('templates')}
        results[step] = result
        values = [step, step*32, result['tiny'], result['dims'], 0.003,
                  f'{fmt(density)} / brak pomiaru bazy', result['mkqa'], result['pairs'], result['templates']]
        lines.append('| ' + ' | '.join(fmt(v) for v in values) + ' |')
    lines += ['', '## Kotwice', '', '| model | tiny10 | wym./dok |', '|---|---:|---:|']
    for name in ['base-splade-wrapped', 'mlm-msmarco-ep1', 'mlm-path-ep3']:
        score = read(ROOT / f'results-regimes/quantized/{name}/tiny10.json')
        lines.append(f'| {name} | {fmt(score.get("mean_ndcg@10"))} | {fmt(sparsity(name))} |')
    lines += ['', '## G1', '']
    required = [5000, 10000, 15000, 18750]
    if not all(k in results for k in required):
        verdict = 'PENDING — brakuje kompletu wymaganych checkpointów.'
    else:
        final = results[18750]['tiny']
        slope = results[10000]['tiny'] - results[5000]['tiny']
        band = all(results[k]['dims'] is not None and 300 <= results[k]['dims'] <= 480 for k in [10000, 15000, 18750])
        drift = []
        for (a, old), (b, new) in zip(list(results.items()), list(results.items())[1:]):
            if new['tiny'] > old['tiny']:
                drift.extend(f'{key}: {a}→{b}' for key in ['mkqa', 'pairs', 'templates']
                             if old[key] is not None and new[key] is not None and old[key]-new[key] >= 1)
        if not band:
            verdict = 'REVIEW — stałe λ, gęstość poza 300–480; decyzja użytkownika, próg 59.5 nie obowiązuje wprost.'
        elif drift:
            verdict = 'FAIL — spadek klastra przy rosnącej średniej: ' + ', '.join(drift)
        elif final >= 59.5 and slope >= 0.8:
            verdict = 'PASS'
        elif 58 <= final < 59.5 and slope >= 0.8:
            verdict = 'MAYBE'
        else:
            verdict = 'FAIL'
        verdict += f' tiny10(s18750)={final:.2f}, s10000−s05000={slope:+.2f}; gęstości=' + ', '.join(fmt(results[k]['dims']) for k in [10000,15000,18750])
    lines += [verdict, '', '## Czas i monitoring', '', '```json',
              json.dumps(report['timing'], ensure_ascii=False, indent=2), '```', '',
              'λ_d: bez zmian (0 wzrostów, 0 spadków), końcowo 0.003. Gęstość względem bazy: niezmierzona w wariancie zapasowym.', '',
              '## FAILED', '', '```']
    try:
        lines.extend(line for line in Path('logs/queue.log').read_text().splitlines() if '[mlm1]' in line and 'FAILED' in line)
    except OSError:
        pass
    lines += ['```', '']
    validation = read('logs/mlm_restart_validation.json')
    if validation:
        lines += ['## Kontrola po restarcie z mniejszym batchem', '', '```json',
                  json.dumps(validation, ensure_ascii=False, indent=2), '```', '']
    (ROOT / 'results/summary_5070_mlm1.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
