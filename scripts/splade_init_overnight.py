"""Bounded second queue, adaptive short probes, persistent incremental report."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

from scripts import splade_init_experiments as base

ROOT, EXP, RESULTS, DATA, CONFIGS, PY = base.ROOT, base.EXP, base.RESULTS, base.DATA, base.CONFIGS, base.PY
STATE = DATA / 'splade-init-overnight.json'
SUMMARY = RESULTS / 'summary_5070_init.md'
BEGIN, END = '<!-- overnight-init-begin -->', '<!-- overnight-init-end -->'


class BudgetExpired(Exception):
    pass


def criteria(fp32_nonzero, dims, tiny, *, complete=True):
    finite = dims is not None and math.isfinite(float(dims))
    alive = bool(complete and fp32_nonzero and finite and dims >= 20)
    band = bool(finite and 150 <= dims <= 900)
    ok = bool(alive and band and tiny is not None and math.isfinite(float(tiny)) and tiny >= 50)
    return {'ALIVE': alive, 'BAND': band, 'OK': ok}


def choose_branch(e2, e3):
    if e3.get('OK'):
        return 'A'
    return 'B' if e2.get('OK') else 'C'


def should_evaluate(dims):
    return dims is not None and math.isfinite(float(dims)) and 0 < dims <= 1500


def is_oom(log):
    text = str(log).lower()
    return 'outofmemoryerror' in text or 'out of memory' in text


def get_state():
    state = base.read(STATE, {'experiments': {}, 'errors': [], 'created': base.now()})
    if 'deadline' not in state:
        state['deadline'] = os.environ['INIT_OVERNIGHT_END']
    return state


def remaining(state):
    return datetime.fromisoformat(state['deadline']).timestamp() - time.time()


def fp32_nonzero(diagnosis):
    rows = [m for m in diagnosis.get('models', []) if m.get('dtype') == 'fp32']
    return bool(rows) and all(not m['nonfinite_weights'] and all(
        o['pooled_all_finite'] and all(v > 0 for v in o['valid_token_active_dims_per_text'])
        for o in m['outputs']) for m in rows)


def sparse_mean(label):
    data = base.read(RESULTS/'sparsity'/f'{label}.json', {}).get(label, {})
    return data.get('ALL', {}).get('mean')


def summarize_point(name, step):
    label = f'modernbert-init-{name}-s{step:05d}'
    diagnostic = base.read(DATA/f'{label}-diagnosis.json')
    if diagnostic is None and step == 500:
        diagnostic = base.read(DATA/f'modernbert-init-{name}-diagnosis.json', {})
    dims = sparse_mean(label)
    tiny = base.tiny_score(EXP/'results-regimes/quantized'/label/'tiny10.json')
    return {'dims': dims, 'tiny10': tiny, 'fp32_nonzero': fp32_nonzero(diagnostic or {}),
            'evaluation': 'dense, skipped' if dims is not None and dims > 1500 else 'pending',
            **criteria(fp32_nonzero(diagnostic or {}), dims, tiny)}


def report(state):
    def f(value):
        return '—' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)
    lines = [BEGIN, '', '## Overnight adaptive probes (second queue)', '',
             f"Updated: {base.now()}; GPU deadline: **{state['deadline']}**. Branch: **{state.get('branch', 'waiting for E2/E3')}**.", '',
             'ALIVE = completed s00500, nonzero FP32 diagnostic vectors, mean document dimensions ≥20; '
             'BAND = 150–900; OK = ALIVE + BAND + complete tiny10 ≥50. '
             'Above 1500 dimensions tiny10 is skipped. Missing/partial evaluations are not numerical scores.', '',
             '| run | branch | c | dims s250 / s500 / s1000 / s2500 | tiny10 (step: score) | verdict |',
             '|---|---|---:|---|---|---|---|']
    for name, row in state['experiments'].items():
        points = row.get('points', {})
        counts = ' / '.join(f(points.get(str(step), {}).get('dims')) for step in (250,500,1000,2500))
        scores = ', '.join(f"{s}: {f(p['tiny10'])}" for s,p in points.items() if p.get('tiny10') is not None) or '—'
        dense = any(p.get('evaluation') == 'dense, skipped' for p in points.values())
        result = row.get('classification', {})
        verdict = 'OK' if result.get('OK') else '/'.join(k for k in ('ALIVE','BAND') if result.get(k)) or row.get('phase', 'pending')
        if dense:
            verdict += '; dense, skipped'
        if row.get('error'):
            verdict += '; ' + row['error']
        init = row.get('initialization') or {}
        lines.append(f"| {name} | {row.get('branch','base')} | {f(init.get('c'))} | {counts} | {scores} | {verdict} |")
    lines += ['', state.get('recommendation', 'Recommendation pending; no night-3 training will be started.'), '',
              'New overnight controllers use reference_dims=436 and band [0.75,1.05] '
              '(327–457.8 dimensions), as requested. Earlier E3 measured its reference on the training sample '
              '(about 317.6); this change is explicit and comparisons are not single-variable ablations. '
              'E3c starts density control at step 250 independently of lambda warmup. '
              'E4+lambda and its E5 start from head-only s00500, so lambda warmup starts at unfreezing. '
              'FP32 precision control uses the same numeric bias shift as its BF16 counterpart when available. '
              'OOM fallback preserves effective batch 32 but changes the in-batch negative pool; see raw state.', '',
              f"Raw state: `{STATE.relative_to(ROOT)}`. Night 3 is never launched by this queue.", '', END]
    if state['errors']:
        lines[-1:-1] = ['', 'Errors/limits:', *[f'- {e}' for e in state['errors']]]
    text = SUMMARY.read_text() if SUMMARY.exists() else '# SPLADE initialization\n'
    if BEGIN in text:
        before, tail = text.split(BEGIN, 1)
        after = tail.split(END, 1)[1] if END in tail else ''
    else:
        before, after = text + '\n\n', ''
    temporary = SUMMARY.with_suffix('.overnight.tmp')
    temporary.write_text(before + '\n'.join(lines) + after)
    temporary.replace(SUMMARY)
    state['updated'] = base.now()
    base.write(STATE, state)


def command(state, args, log_name, timeout, *, disk_guard=True):
    seconds = min(timeout, remaining(state) - 10)
    if seconds <= 0:
        raise BudgetExpired('08:00 GPU deadline reached')
    path = ROOT/'logs'/log_name
    started = time.monotonic()
    with path.open('w') as log:
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            reason = None
            if time.monotonic() - started >= seconds:
                reason = 'deadline/phase timeout'
            if disk_guard and shutil.disk_usage(ROOT).free < 5*1024**3:
                reason = 'less than 5 GiB free disk'
            if reason:
                log.write('\n' + reason + '\n'); log.flush()
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                except ProcessLookupError:
                    pass
                return {'rc':124, 'reason':reason, 'log':str(path.relative_to(ROOT))}
            time.sleep(min(3, max(0.1, remaining(state)-8)))
    return {'rc':process.returncode, 'seconds':round(time.monotonic()-started), 'log':str(path.relative_to(ROOT))}


def proc_args(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    except (OSError, UnicodeError):
        return []


def descendant(pid, ancestor):
    for _ in range(64):
        if pid == ancestor:
            return True
        if pid <= 1:
            return False
        try:
            raw = Path(f'/proc/{pid}/stat').read_text()
            pid = int(raw[raw.rindex(')')+2:].split()[1])
        except (OSError, ValueError):
            return False
    return False


def stop_dense_primary(primary_pid, state):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        args = proc_args(pid)
        if len(args) < 5 or args[0] != 'bash' or args[1] != 'scripts/pirb_regime_eval.sh' or args[2] != 'quantized':
            continue
        label = args[3]
        if not label.startswith('modernbert-init-') or not descendant(pid, primary_pid):
            continue
        dims = sparse_mean(label)
        try:
            session_leader = os.getpgid(pid) == pid
        except ProcessLookupError:
            continue
        if dims is None or dims <= 1500 or not session_leader:
            continue
        state.setdefault('dense_skips', {})[label] = {'dims':dims, 'pid':pid, 'time':base.now()}
        print(f'Stopping dense tiny10 session {pid}: {label}, {dims:.1f} dims', flush=True)
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def import_primary(state):
    source = base.read(base.STATE, {}).get('experiments', {})
    for name, original in source.items():
        row = deepcopy(original)
        row['branch'] = 'base'
        row['points'] = {str(step):summarize_point(name, step) for step in (250,500)
                         if (ROOT/f'models/modernbert-init-{name}-s{step:05d}').is_dir()}
        row['classification'] = row['points'].get('500', criteria(False,None,None))
        state['experiments'][name] = row


def wait_primary(state):
    pf = ROOT/'logs/queue_5070_init.pid'
    primary = int(pf.read_text().strip())
    while any('queue_5070_init.sh' in a for a in proc_args(primary)):
        stop_dense_primary(primary, state)
        import_primary(state); report(state)
        if remaining(state) <= 10:
            raise BudgetExpired('deadline reached while waiting for first queue')
        time.sleep(10)
    import_primary(state); report(state)


def configuration(name, recipe, *, steps=500, source=None, batch=8, seed=42):
    import yaml
    cfg = yaml.safe_load((CONFIGS/'modernbert_probe_m.yaml').read_text())
    cfg['runs_dir'] = f'runs/modernbert-init-{name}-b{batch}'
    cfg['timestamp_output_dir'] = True
    lr = recipe.get('lr',2e-5)
    cfg['hparams'][0].update(learning_rate=lr, document_regularizer_weight=recipe.get('lambda_d',0.003))
    if source:
        cfg['architectures'] = [str(source)]
    st = cfg['sentence_transformers']
    snapshots = [250,500] if steps == 500 else list(range(500,steps+1,500))
    st.update(max_steps=steps, warmup_steps=250, warmup_ratio=0.0, seed=seed,
              train_batch_size=batch, gradient_accumulation_steps=32//batch,
              bf16=not recipe.get('fp32',False), fp16=False, scale=1.0,
              save_steps=250 if steps == 500 else 500, save_total_limit=1,
              keep_step_checkpoints=snapshots, report_to=[], splade_regularizer_scheduler=False,
              splade_head_learning_rate=recipe.get('head_lr',lr),
              splade_train_only_head=recipe.get('head_only',False),
              query_regularizer_weight=recipe.get('lambda_q',0.00015))
    st['splade_activation_stats'].update(interval_steps=50,batch_size=8,stop_below=5)
    st['splade_decoder_bias_shift'] = 0.0 if source else recipe.get('c','auto')
    st['splade_decoder_bias_target'] = recipe.get('target',400)
    if recipe.get('lambda_d',0.003) == 0:
        st.pop('regularizer_control',None)
    else:
        reg = st['regularizer_control']
        reg['warmup_steps'] = recipe.get('lambda_warmup',1000)
        density = reg['density']
        density.pop('reference_model',None)
        density.update(reference_dims=436,max_document_weight=recipe.get('max_lambda',0.03),batch_size=8)
        if 'control_start' in recipe:
            density['start_step'] = recipe['control_start']
    path = CONFIGS/f'modernbert_overnight_{name}_b{batch}.yaml'
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return path


def exported(name, step):
    return ROOT/f'models/modernbert-init-{name}-s{step:05d}'


def snapshot_results(name, row, *, source_root=None):
    if source_root is None and row.get('physical_batch'):
        source_root = ROOT/f"runs/modernbert-init-{name}-b{row['physical_batch']}"
    roots = [source_root] if source_root else list((ROOT/'runs').glob(f'modernbert-init-{name}-b*'))
    states = [p for root in roots for p in root.glob('**/checkpoint-*/trainer_state.json')]
    if not states:
        return
    latest = max(states,key=lambda p:(int(p.parent.name.split('-')[-1]),p.stat().st_mtime))
    run = latest.parent.parent
    row['run'] = str(run.relative_to(ROOT))
    row['trained_steps'] = base.read(latest,{}) .get('global_step')
    row['initialization'] = base.read(run/'splade_initialization.json')
    row['density_stop'] = base.read(run/'splade_density_stop.json')
    for path in run.glob('step-checkpoints/step-*/model.safetensors'):
        step = int(path.parent.name.split('-')[-1])
        base.export_model(path.parent,exported(name,step))
    for path in run.glob('checkpoint-*/model.safetensors'):
        step = int(path.parent.name.split('-')[-1])
        if base.read(path.parent/'trainer_state.json',{}).get('global_step') != step:
            continue  # an interrupted save must never masquerade as a checkpoint
        if step in ([250,500] if row['requested_steps']==500 else range(500,row['requested_steps']+1,500)):
            base.export_model(path.parent,exported(name,step))
    # Generated duplicate optimizer snapshots only; retain the latest full trainer
    # checkpoint (optimizer/RNG) for genuine resumptions and all model exports.
    safe_root = (ROOT/'runs').resolve()
    for path in run.glob('step-checkpoints/step-*/*'):
        if path.name in ('optimizer.pt','scheduler.pt') or path.name.startswith('rng_state'):
            if path.is_file() and safe_root in path.resolve().parents:
                path.unlink()


def measure_point(state,name,row,step,tiny=False):
    model = exported(name,step)
    if not model.is_dir():
        return
    label = f'modernbert-init-{name}-s{step:05d}'
    point = row.setdefault('points',{}).setdefault(str(step),{})
    if sparse_mean(label) is None:
        point['sparsity_process'] = command(state,[PY,'scripts/compare_splade_sparsity.py','--model',f'{label}={model}',
            '--scope','tiny','--max-docs-per-task','300','--device','cuda','--batch-size','8','--no-fp16',
            '--output-json',str(RESULTS/'sparsity'/f'{label}.json')],f'overnight_{name}_s{step}_sparsity.log',900)
    point['dims'] = sparse_mean(label)
    report(state)
    diag = DATA/f'{label}-diagnosis.json'
    if not diag.exists():
        point['diagnostic_process'] = command(state,[PY,'scripts/diagnose_splade_checkpoint.py','--model',f'{label}={model}',
            '--output',str(diag)],f'overnight_{name}_s{step}_diagnosis.log',600)
    point['fp32_nonzero'] = fp32_nonzero(base.read(diag,{}))
    tiny_path = EXP/'results-regimes/quantized'/label/'tiny10.json'
    if tiny and should_evaluate(point['dims']):
        if base.tiny_score(tiny_path) is None:
            for batch in (8,4):
                result = command(state,['bash','scripts/pirb_regime_eval.sh','quantized',label,str(model),'tiny10',str(batch),'512'],
                                 f'overnight_{name}_s{step}_tiny10_b{batch}.log',1800)
                point['tiny_process'] = result
                rawlog = ROOT/'logs/regimes'/f'quantized_{label}_tiny10.log'
                text = rawlog.read_text(errors='replace') if rawlog.exists() else ''
                base.cleanup_eval_cache(label)
                if result['rc']==0 or not is_oom(text):
                    break
        point['tiny10'] = base.tiny_score(tiny_path)
        point['evaluation'] = 'complete' if point['tiny10'] is not None else 'failed/incomplete'
    elif tiny:
        point['evaluation'] = 'dense, skipped' if point['dims'] is not None and point['dims']>1500 else 'missing/zero sparsity, skipped'
    point.update(criteria(point.get('fp32_nonzero',False),point.get('dims'),point.get('tiny10')))
    if step==500:
        row['classification'] = dict((k,point[k]) for k in ('ALIVE','BAND','OK'))
    report(state)


def commit(state,label):
    report(state)
    paths = [str(SUMMARY.relative_to(ROOT)),str(STATE.relative_to(ROOT))]
    paths += [str(p.relative_to(ROOT)) for p in CONFIGS.glob('modernbert_overnight_*.yaml')]
    paths += [str(p.relative_to(ROOT)) for p in DATA.glob('modernbert-init-n2-*-diagnosis.json')]
    paths += [str(p.relative_to(ROOT)) for p in (RESULTS/'sparsity').glob('modernbert-init-n2-*.json')]
    paths += [str(p.relative_to(ROOT)) for p in (EXP/'results-regimes/quantized').glob('modernbert-init-n2-*/tiny10.json')]
    with (ROOT/'logs/overnight_commits.log').open('a') as log:
        for attempt in range(3):
            added = subprocess.run(['git','add','--',*paths],cwd=ROOT,stdout=log,stderr=log)
            if added.returncode==0:
                subprocess.run(['git','commit','--only','-m',f'Record overnight SPLADE {label}','--',*paths],cwd=ROOT,stdout=log,stderr=log)
                break
            time.sleep(2)
    if remaining(state)>45:
        command(state,['git','push','origin','master'],f'overnight_push_{label}.log',30,disk_guard=False)


def trial(state,name,recipe,branch,*,steps=500,source=None,seed=42):
    row = state['experiments'].setdefault(name,{'branch':branch,'points':{},'requested_steps':steps,'recipe':recipe})
    if row.get('phase')=='finished':
        return row
    row['phase']='training'; report(state)
    if remaining(state)<1000:
        row['phase']='skipped: time budget'; row['error']='not enough time for a checkpoint'
        report(state)
        raise BudgetExpired('insufficient budget for a meaningful new training/checkpoint')
    try:
        if not exported(name,steps).exists():
            for batch in ((4,) if recipe.get('initial_batch')==4 else (8,4)):
                cfg = configuration(name,recipe,steps=steps,source=source,batch=batch,seed=seed)
                row['physical_batch']=batch
                result = command(state,[PY,'-m','training.train','--backend','sentence-transformers','--training-type','splade',
                                       '--config',str(cfg)],f'overnight_{name}_train_b{batch}.log',14400 if steps>500 else 3600)
                row.setdefault('training_attempts',[]).append(result)
                text = (ROOT/result['log']).read_text(errors='replace')
                if result['rc']!=0 and is_oom(text) and batch==8:
                    row['oom_fallback']='batch4x8, fresh attempt; effective batch32, smaller negative pool'
                    report(state)
                    continue
                snapshot_results(name,row)
                report(state)
                break
        row['phase']='measurement'; report(state)
        measurements = (250,500) if steps==500 else range(500,steps+1,500)
        for step in measurements:
            measure_point(state,name,row,step,tiny=(step==500 if steps==500 else step in (1000,2500)))
        row['phase']='finished'
        if not exported(name,steps).exists():
            row['error']='no requested final checkpoint; early stop/training failure'
    except BudgetExpired as exc:
        snapshot_results(name,row)
        row['phase']='time budget stopped'; row['error']=str(exc)
        raise
    except Exception as exc:
        row['phase']='error'; row['error']=f'{type(exc).__name__}: {exc}'
    finally:
        commit(state,name)
    return row


def finish_recommendation(state):
    e5 = state['experiments'].get('n2-e5',{})
    points = e5.get('points',{})
    last, earlier = points.get('2500',{}), points.get('1000',{})
    controlled = [points.get(str(s),{}).get('dims') for s in (1000,1500,2000,2500)]
    stable = (last.get('fp32_nonzero') and all(v is not None and 0.75*436 <= v <= 1.05*436 for v in controlled)
              and last.get('tiny10') is not None and last['tiny10']>=50 and earlier.get('tiny10') is not None
              and last['tiny10']>earlier['tiny10'])
    if stable:
        state['recommendation']=f"Recommendation for night 3 (not launched): E5 recipe from branch {state.get('branch')}; quality rose after full lambda warmup and measured density stayed in the requested band."
    else:
        candidates = [(r.get('points',{}).get('500',{}).get('tiny10'),n) for n,r in state['experiments'].items() if r.get('classification',{}).get('OK')]
        candidates = [(score,n) for score,n in candidates if score is not None]
        if candidates:
            state['recommendation']=f"No stable recipe established by the longer E5 criterion; best short OK probe: {max(candidates)[1]}. Next hypothesis: tune regularization strength/start while retaining calibrated initialization."
        else:
            state['recommendation']='No stable recipe. Next hypothesis: backbone adaptation/precision or insufficient query activations after global bias shift; inspect E4 and the matched BF16/FP32 controls.'


def run(state):
    import_primary(state)
    regular = {'lambda_d':0.003,'lambda_q':0.00015,'lambda_warmup':1000}
    inputs={name:state['experiments'].get(name,{}) for name in ('e2','e3')}
    for name,recipe in [('e2',{'lambda_d':0.0,'lambda_q':0.0}),('e3',regular)]:
        log=ROOT/'logs'/f'init_{name}_train.log'
        if not exported(name,500).exists() and log.exists() and is_oom(log.read_text(errors='replace')):
            try:
                inputs[name]=trial(state,f'n2-{name}-oom-retry',dict(recipe,initial_batch=4),'OOM recovery')
            except BudgetExpired:
                finish_recommendation(state);commit(state,'deadline');return
    branch = choose_branch(inputs['e2'].get('classification',{}),inputs['e3'].get('classification',{}))
    state['branch']=branch
    state['branch_inputs']={name:row.get('classification',{}) for name,row in inputs.items()}
    report(state)
    selected, source = None, None
    try:
        if branch=='A':
            selected=regular
        elif branch=='B':
            recipes = [('n2-e3b',dict(regular,lambda_d=0.001,lambda_q=0.00005,lambda_warmup=2000)),
                       ('n2-e3c',dict(regular,control_start=250,max_lambda=0.01))]
            candidates=[]
            for name,recipe in recipes:
                row=trial(state,name,recipe,branch)
                score=row.get('points',{}).get('500',{}).get('tiny10')
                if score is not None and row.get('classification',{}).get('ALIVE'):
                    candidates.append((score,name,recipe))
            if candidates:
                _,state['selected_recipe'],selected=max(candidates,key=lambda c:c[0])
        else:
            no_reg={'lambda_d':0.0,'lambda_q':0.0}
            head_path=exported('e4-head',500)
            if not head_path.exists():
                trial(state,'n2-e4-head',dict(no_reg,head_only=True),branch)
                head_path=exported('n2-e4-head',500)
            unfrozen=state['experiments'].get('e4-unfrozen')
            if head_path.exists() and not (unfrozen and exported('e4-unfrozen',500).exists()):
                unfrozen=trial(state,'n2-e4-unfrozen',no_reg,branch,source=head_path)
            if unfrozen and unfrozen.get('classification',{}).get('OK'):
                row=trial(state,'n2-e4-lambda',regular,branch,source=head_path)
                if row.get('classification',{}).get('OK'):
                    selected,source=regular,head_path
            else:
                recipe=dict(no_reg,lr=5e-6,target=200)
                bf=trial(state,'n2-e2-low-bf16',recipe,branch)
                c=(bf.get('initialization') or {}).get('c')
                control=dict(recipe,fp32=True)
                if c is not None:
                    control['c']=c
                trial(state,'n2-e2-low-fp32',control,branch)
                return
        if selected and remaining(state)>1800:
            trial(state,'n2-e5',selected,branch,steps=2500,source=source)
        if branch=='A':
            for name,recipe in [('n2-e6',dict(regular,target=200)),('n2-e7',dict(regular,lr=1e-5,head_lr=2e-5))]:
                if remaining(state)>2400:
                    trial(state,name,recipe,branch)
        # A/B may finish before morning after dense evaluations are skipped.
        # Use the remaining budget for reproducibility, not a new long run.
        if branch in ('A','B') and selected:
            seed=43
            while remaining(state)>1800 and seed<=49:
                trial(state,f'n2-repeat-seed{seed}',selected,branch,source=source,seed=seed)
                seed+=1
    except BudgetExpired as exc:
        state['errors'].append(str(exc))
    finally:
        finish_recommendation(state)
        commit(state,'final')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-primary',action='store_true')
    args=parser.parse_args()
    state=get_state()
    if args.wait_primary:
        try:
            wait_primary(state)
        except BudgetExpired as exc:
            state['errors'].append(str(exc)); report(state)
        return
    if remaining(state)<=10:
        finish_recommendation(state); report(state); return
    run(state)


if __name__=='__main__':
    main()
