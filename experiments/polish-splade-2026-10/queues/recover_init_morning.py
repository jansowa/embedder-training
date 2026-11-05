"""Recover only redundant optimizer copies, then finish interrupted E5/seed44."""
from pathlib import Path
import hashlib
import json
import zipfile

from scripts import splade_init_experiments as base
from scripts import splade_init_overnight as night


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def clean_duplicate_states():
    root=(base.ROOT/'runs').resolve()
    receipt=base.DATA/'morning-optimizer-cleanup.json'
    previous=base.read(receipt,{})
    if previous.get('files') and all(not Path(item['path']).exists() for item in previous['files']):
        print('Verified duplicate-state cleanup already completed; preserving receipt/archive',flush=True)
        return
    candidates=[]
    archive=base.RESULTS/'morning-optimizer-cleanup-metadata.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as backup:
        for name in ('e2','e3'):
            run_root=base.ROOT/f'runs/modernbert-init-{name}'
            for path in run_root.glob('**/optimizer.pt'):
                folder=path.parent
                parent=folder.parent.name
                if parent=='step-checkpoints':
                    step=int(folder.name.removeprefix('step-'))
                elif folder.name=='checkpoint-250':
                    step=250
                else:
                    continue  # keep latest full checkpoint-500 optimizer/RNG
                if root not in path.resolve().parents:
                    raise RuntimeError(f'Unsafe cleanup path {path}')
                source=folder/'model.safetensors'
                target=base.ROOT/f'models/modernbert-init-{name}-s{step:05d}/model.safetensors'
                if not target.is_file() or sha(source)!=sha(target):
                    raise RuntimeError(f'No verified model backup for {folder}')
                for metadata in folder.glob('*.json'):
                    backup.write(metadata,str(metadata.relative_to(base.ROOT)))
                for file in folder.iterdir():
                    if file.name in ('optimizer.pt','scheduler.pt') or file.name.startswith('rng_state'):
                        if file.is_file() and root in file.resolve().parents:
                            candidates.append({'path':str(file),'bytes':file.stat().st_size})
    base.write(base.DATA/'morning-optimizer-cleanup.json',{'files':candidates,'bytes':sum(p['bytes'] for p in candidates)})
    for item in candidates:
        Path(item['path']).unlink()
    print(f"Freed redundant state files: {sum(p['bytes'] for p in candidates)/1e9:.2f} GB",flush=True)


def resume(state,name):
    row=state['experiments'][name]
    run=base.ROOT/row['run']
    states=sorted(run.glob('checkpoint-*/trainer_state.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
    checkpoint=states[-1].parent
    step=base.read(states[-1])['global_step']
    requested=row['requested_steps']
    if step<requested:
        if not all((checkpoint/f).is_file() for f in ('optimizer.pt','scheduler.pt','rng_state.pth')):
            raise RuntimeError(f'No complete resumable optimizer state at {checkpoint}')
        row['phase']='resuming saved checkpoint'; row['error']=None; night.report(state)
        result=night.command(state,[base.PY,'-m','training.train','--backend','sentence-transformers',
            '--training-type','splade','--config',str(run/'resolved_config.yaml'),
            '--resume-from-checkpoint',str(checkpoint.relative_to(base.ROOT))],f'morning_{name}_resume.log',3600)
        row.setdefault('recovery_attempts',[]).append(result)
        night.snapshot_results(name,row)
        night.report(state)
        if result['rc']!=0:
            raise RuntimeError(f"{name} resume failed rc={result['rc']}; see {result['log']}")
    for measured in ([250,500] if requested==500 else range(500,requested+1,500)):
        night.measure_point(state,name,row,measured,tiny=(measured==500 if requested==500 else measured in (1000,2500)))
    if night.exported(name,requested).is_dir():
        row['phase']='finished'; row['error']=None
    else:
        row['phase']='recovery incomplete'
    night.finish_recommendation(state)
    night.commit(state,f'morning-{name}')


state=night.get_state()
try:
    clean_duplicate_states()
    resume(state,'n2-e5')
    if night.remaining(state)>1800:
        resume(state,'n2-repeat-seed44')
except Exception as exc:
    state['errors'].append(f'Morning recovery: {type(exc).__name__}: {exc}')
finally:
    night.finish_recommendation(state)
    night.commit(state,'morning-final')
