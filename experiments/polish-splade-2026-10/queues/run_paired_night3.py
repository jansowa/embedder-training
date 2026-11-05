"""Paired cold-start M -> A run, fixed data/seed/steps, no checkpoint resumes."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timedelta
import hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
import yaml

from scripts import splade_init_experiments as base
from training.multi_dataset import resolve_dataset_sources

ROOT,EXP,RESULTS,DATA,PY=base.ROOT,base.EXP,base.RESULTS,base.DATA,base.PY
STATE=DATA/'night3-paired-state.json'
SUMMARY=RESULTS/'summary_5070_mlm3.md'
STEPS=(2150,4306)
ACTIVE=None


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def gate(m,a):
    if not all(p.get('dims') is not None and 327<=p['dims']<=458 for p in (m,a)):
        return 'nieporównywalne przy tej gęstości'
    if any(p.get('tiny10') is None for p in (m,a)):
        return 'brak kompletnego tiny10 — GM niepoliczony'
    delta=m['tiny10']-a['tiny10']
    if delta>=2 and m['tiny10']>=57:
        return f'PASS: ModernBERT zostaje (M − A = {delta:+.2f})'
    if abs(delta)<0.8:
        return f'distilroberta: |M − A| < 0.8; więcej danych (M − A = {delta:+.2f})'
    return f'MAYBE: decyzja użytkownika (M − A = {delta:+.2f})'


def telemetry(path):
    rows=[]
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    measured=[r for r in rows if 'grad_norm' in r.get('metrics',{}) and r.get('step',0)>0]
    result={'step':measured[-1]['step'] if measured else 0}
    if len(measured)>=2:
        last=measured[-1]
        # Exclude startup and initial activation checks from steady-state pace.
        first=next((r for r in measured if r['step']>=25),measured[0])
        if last['step']>first['step']:
            result['seconds_per_step']=(last['elapsed_train_seconds']-first['elapsed_train_seconds'])/(last['step']-first['step'])
    result['peak_allocated_mib']=max((r.get('peak_allocated_mib',0) for r in rows),default=0)
    result['peak_reserved_mib']=max((r.get('peak_reserved_mib',0) for r in rows),default=0)
    sampled=[]
    for row in rows:
        try:sampled.append(float(row['gpu'].split(',')[0]))
        except (KeyError,TypeError,ValueError):pass
    result['peak_nvidia_sampled_mib']=max(sampled,default=0)
    return result


def report(state):
    def f(v):return '—' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v)
    inputs=base.read(DATA/'night3-inputs.json',{})
    lines=['# Noc 3 — parowany bieg MLM, seed 42','',f'Updated: {base.now()}', '',
           '**137 773 przykładów po bramce i rozwinięciu pozytywów; identyczna lista i kolejność dla M i A.** '
           'MS MARCO: fallback `datasets/judged/msmarco-rejected-20000.jsonl` (HF 401). '
           'Natywna bramka SQuAD-v2: 56 093 wiersze / 56 437 przykładów. '
           'Zatwierdzony plan: nominalnie jedna epoka, max_steps=4306, checkpointy 2150/4306, '
           'próg miejsca przed parą 35 GB. M: BF16, A: FP16; oba od oryginalnego MLM, '
           'bias shift auto→400, scale=1, LR=2e-5, warmup LR 250, lambda warmup 1000, regulator 436×[0.75,1.05]. '
           'Żaden retry nie używa resume. Optymalizator/scheduler są usuwane dopiero po zakończeniu procesu treningu.', '',
           '| Składnik | Wiersze po bramce | Przykłady |','|---|---:|---:|']
    for s in inputs.get('paired_preflight',{}).get('sources',[]):
        lines.append(f"| {s['original']} | {s['gated_rows']} | {s['examples']} |")
    lines+=['','| Model | Status | krok | s/krok | peak CUDA / NVIDIA sampled MiB | ETA treningu |',
            '|---|---|---:|---:|---|---|']
    for name,row in state['runs'].items():
        t=row.get('telemetry',{})
        eta=None
        if row.get('phase')=='training' and t.get('seconds_per_step'):
            eta=(datetime.now().astimezone()+timedelta(seconds=max(0,4306-t['step'])*t['seconds_per_step']+120)).isoformat(timespec='minutes')
        lines.append(f"| {name.upper()} | {row.get('phase','pending')} | {t.get('step',0)} | {f(t.get('seconds_per_step'))} | "
                     f"{f(t.get('peak_allocated_mib'))} / {f(t.get('peak_nvidia_sampled_mib'))} | {eta or '—'} |")
    lines+=['','| Model | checkpoint | dims/doc (FP32, 512) | tiny10 nDCG@10 |','|---|---:|---:|---:|']
    for name,row in state['runs'].items():
        for step in STEPS:
            point=row.get('points',{}).get(str(step),{})
            lines.append(f"| {name.upper()} | {step} | {f(point.get('dims'))} | {f(point.get('tiny10'))} |")
    m=state['runs']['m'].get('points',{}).get('4306',{})
    a=state['runs']['a'].get('points',{}).get('4306',{})
    verdict=gate(m,a) if state['runs']['m'].get('training_complete') and state['runs']['a'].get('training_complete') else 'pending'
    state['GM']=verdict
    lines+=['',f'**GM s04306: {verdict}.** Wyłącznie końcowe checkpointy obu modeli w paśmie 327–458; '
            'stara krzywa A i E5 s2500 nie są używane do GM.', '',
            '## Per task — ten sam quantized tiny10, 512 tokenów','',
            '| Task | M s2150 | A s2150 | M s4306 | A s4306 | base | mlm-msmarco-ep1 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    result_sets={}
    for name in ('m','a'):
        for step in STEPS:
            label=f'night3-paired-{name}-s{step:05d}'
            result_sets[f'{name}{step}']=base.read(EXP/'results-regimes/quantized'/label/'tiny10.json',{})
    for key,label in [('base','base-splade-wrapped'),('anchor','mlm-msmarco-ep1')]:
        result_sets[key]=base.read(EXP/'results-regimes/quantized'/label/'tiny10.json',{})
    tasks=list(result_sets['base'].get('tasks',{}))
    if not tasks:
        tasks=sorted({t for d in result_sets.values() for t in d.get('tasks',{})})
    for task in tasks:
        cells=[f(result_sets[k].get('tasks',{}).get(task)) for k in ('m2150','a2150','m4306','a4306','base','anchor')]
        lines.append('| '+task+' | '+' | '.join(cells)+' |')
    lines+=['','Referencje: base-splade-wrapped 61.60, mlm-msmarco-ep1 59.29. '
            'Stara A: s2500=56.33 przy 297 dims, s5000=55.37 przy 311 dims — tylko odniesienie, '
            'stała lambda i dryfująca gęstość.', '',
            'Krok 0: M s1500=55.63 /356 dims, M s2000=56.61 /404 dims; '
            'A s1875=55.70 /391 dims; M−A=+0.91. E5 s2500 wyłączony z werdyktu.', '',
            f"Free C: before pair: {f(state.get('start_free_gb'))} GB. Training timeouts: 12h, retry from zero 18h. "
            'CUDA peak is allocator peak; NVIDIA peak is sampled total card use.']
    if state.get('errors'):lines+=['','Errors:',*[f'- {e}' for e in state['errors']]]
    temporary=SUMMARY.with_suffix('.paired.tmp')
    temporary.write_text('\n'.join(lines)+'\n');temporary.replace(SUMMARY)
    base.write(STATE,state)


def commit(state,label):
    report(state)
    paths=[str(STATE.relative_to(ROOT)),str(SUMMARY.relative_to(ROOT))]
    paths += [str(p.relative_to(ROOT)) for p in DATA.glob('night3-*-regulator.json')]
    paths += [str(p.relative_to(ROOT)) for p in (RESULTS/'sparsity').glob('night3-paired-*.json')]
    paths += [str(p.relative_to(ROOT)) for p in (EXP/'results-regimes/quantized').glob('night3-paired-*/tiny10.json')]
    with (ROOT/'logs/night3-paired-commits.log').open('a') as log:
        subprocess.run(['git','add','--',*paths],cwd=ROOT,stdout=log,stderr=log,check=True)
        subprocess.run(['git','commit','--only','-m',f'Record paired night-three {label}','--',*paths],cwd=ROOT,stdout=log,stderr=log)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (ROOT/'logs/night3-paired-push.request').write_text(head)


def run_command(state,args,log_name,timeout,on_poll=None):
    global ACTIVE
    begin=time.monotonic()
    with (ROOT/'logs'/log_name).open('w') as log:
        ACTIVE=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while ACTIVE.poll() is None:
            if on_poll:on_poll()
            reason=None
            if time.monotonic()-begin>timeout:reason='training/measurement timeout'
            if __import__('shutil').disk_usage(ROOT).free<5*1024**3:reason='less than 5 GiB free'
            if reason:
                os.killpg(ACTIVE.pid,signal.SIGTERM)
                try:ACTIVE.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
                ACTIVE=None
                return {'rc':124,'reason':reason,'log':log_name}
            time.sleep(10)
        rc=ACTIVE.returncode;ACTIVE=None
    return {'rc':rc,'seconds':round(time.monotonic()-begin,1),'log':log_name}


def terminate(signum,frame):
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid,signal.SIGTERM)
        try:ACTIVE.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(ACTIVE.pid,signal.SIGKILL);ACTIVE.wait()
    raise SystemExit(128+signum)


def inputs_signature(config):
    cfg=yaml.safe_load(config.read_text())
    sources=resolve_dataset_sources(cfg['train_data'],cfg,config_path=str(config))
    return [{'original':s.original,'sha256':sha(s.path)} for s in sources]


def export_and_prune(run,name,process_done):
    if not process_done:raise RuntimeError('Refusing optimizer cleanup while training is active')
    for step in STEPS:
        source=run/'step-checkpoints'/f'step-{step}'
        saved=base.read(source/'trainer_state.json',{})
        if saved.get('global_step')!=step:raise RuntimeError(f'Missing complete requested checkpoint {source}')
        base.export_model(source,ROOT/f'models/night3-paired-{name}-s{step:05d}')
    regulator=base.read(run/'regularizer_control.json')
    if regulator is not None:base.write(DATA/f'night3-{name}-regulator.json',regulator)
    safe=(ROOT/'runs').resolve()
    if safe not in run.resolve().parents:raise RuntimeError('Unexpected run cleanup path')
    for file in run.rglob('*'):
        if file.is_file() and file.name in ('optimizer.pt','scheduler.pt'):
            file.unlink()


def train(state,name):
    row=state['runs'][name]
    if row.get('training_complete'):
        return
    config=EXP/f'configs/next/5070/mlm3_paired_{name}.yaml'
    signature=inputs_signature(config)
    if signature!=state['data_signature']:
        raise RuntimeError('Paired input files/order changed; stopping before training')
    for attempt,timeout in enumerate((43200,64800),1):
        row['phase']='training';row['attempt']=attempt;report(state)
        root=ROOT/f'runs/night3-paired-{name}'
        seen=set(root.glob('**/resolved_config.yaml'))
        selected=[None]
        def poll():
            new=[p for p in root.glob('**/resolved_config.yaml') if p not in seen]
            if new:selected[0]=max(new,key=lambda p:p.stat().st_mtime).parent
            if selected[0]:
                row['run']=str(selected[0].relative_to(ROOT))
                row['telemetry']=telemetry(selected[0]/'collapse_control.jsonl')
                report(state)
                if row['telemetry']['step']>=100 and not row.get('reported_100'):
                    row['reported_100']=base.now();commit(state,f'{name}-first100')
        result=run_command(state,[PY,'scripts/run_splade_collapse_control.py','--monitor-only','--backend','sentence-transformers',
            '--training-type','splade','--config',str(config)],f'night3_{name}_attempt{attempt}.log',timeout,poll)
        poll()
        row.setdefault('attempts',[]).append(result)
        run=selected[0]
        finished=base.read(run/'control_train_state.json',{}) if run else {}
        if result['rc']==0 and finished.get('global_step')==4306 and (run/'final/model.safetensors').is_file():
            export_and_prune(run,name,True)
            row['training_complete']=True;row['trainer_epoch']=finished.get('epoch');row['phase']='trained'
            commit(state,f'{name}-trained');return
        row['phase']='attempt failed';commit(state,f'{name}-attempt{attempt}-failed')
        # No resume: the next attempt creates a fresh timestamp and original MLM.
        if attempt==1:time.sleep(30)
    raise RuntimeError(f'{name}: two cold-start attempts failed; inspect training logs')


def measure(state,name):
    row=state['runs'][name]
    if not row.get('training_complete'):return
    for step in STEPS:
        row['phase']=f'evaluating {step}';report(state)
        label=f'night3-paired-{name}-s{step:05d}'
        model=ROOT/'models'/label
        point=row.setdefault('points',{}).setdefault(str(step),{})
        sparse=RESULTS/'sparsity'/f'{label}.json'
        if not sparse.exists():
            point['sparsity_process']=run_command(state,[PY,'scripts/compare_splade_sparsity.py','--model',f'{label}={model}',
                '--scope','tiny','--max-docs-per-task','300','--device','cuda','--batch-size','8','--no-fp16',
                '--output-json',str(sparse)],f'night3_{name}_{step}_sparsity.log',1800)
        point['dims']=base.read(sparse,{}).get(label,{}).get('ALL',{}).get('mean')
        result=EXP/'results-regimes/quantized'/label/'tiny10.json'
        if base.tiny_score(result) is None and point['dims'] is not None and point['dims']>0:
            os.environ['PIRB_DTYPE']='bf16' if name=='m' else 'fp16'
            point['tiny_process']=run_command(state,['bash','scripts/pirb_regime_eval.sh','quantized',label,str(model),
                                                    'tiny10','8','512'],f'night3_{name}_{step}_tiny10.log',7200)
        point['tiny10']=base.tiny_score(result)
        point['in_band']=point['dims'] is not None and 327<=point['dims']<=458
        # Own completed evaluation cache only; model weights and raw benchmark data remain.
        cache=ROOT/'cache/regimes'/f'quantized_{label}_tiny10'
        if base.tiny_score(result) is not None and cache.exists():
            if cache.parent.resolve()!=(ROOT/'cache/regimes').resolve():raise RuntimeError('Unexpected evaluation cache')
            __import__('shutil').rmtree(cache)
        commit(state,f'{name}-s{step:05d}')
    row['phase']='finished';commit(state,f'{name}-complete')


def main():
    os.chdir(ROOT)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    inputs=base.read(DATA/'night3-inputs.json')
    assert inputs['paired_preflight']['examples']==137773
    state=base.read(STATE)
    if state is None:
        free=__import__('shutil').disk_usage(ROOT).free/1e9
        if free<35:raise RuntimeError(f'Only {free:.2f} GB free; minimum 35 GB required')
        m=EXP/'configs/next/5070/mlm3_paired_m.yaml'
        a=EXP/'configs/next/5070/mlm3_paired_a.yaml'
        signature=inputs_signature(m)
        if inputs_signature(a)!=signature:raise RuntimeError('M/A source mismatch')
        state={'created':base.now(),'start_free_gb':free,'data_signature':signature,'runs':{'m':{},'a':{}},'errors':[]}
    report(state)
    for name in ('m','a'):
        try:
            train(state,name);measure(state,name)
        except Exception as exc:
            state['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
            state['runs'][name]['phase']='failed';commit(state,f'{name}-failed')
    commit(state,'final')


if __name__=='__main__':main()
