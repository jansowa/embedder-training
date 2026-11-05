"""Daytime data import and native-filter preflight; no GPU/training."""
from pathlib import Path
import gzip,hashlib,json,os,shutil,tempfile,urllib.error

ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT)
DATA=ROOT/'experiments/polish-splade-2026-10/results/data'
REPORT=DATA/'night3-inputs.json'
EXPECTED='c6852c42a7cf310cc30a6ae8c39fcb1a4a87e221ef3bf987a78561e81d002388'
TARGET=ROOT/'datasets/arms/squadv2-v2-all.jsonl'
SETTINGS={'dataset_filter':'configs/dataset_filters/splade_positive_score_gt_23_50_no_metricx.yaml',
          'dataset_filter_cache_dir':'cache/filtered_datasets'}
def save(report):
    DATA.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def validate(path):
    with path.open('rb') as f:
        first=json.loads(f.readline())
        count=1+sum(1 for _ in f)
    required={'query','pos','neg','pos_scores_stronger_reranker','translation'}
    assert required<=first.keys(),sorted(first.keys())
    assert len(first['pos'])==1 and len(first['neg'])==10
    assert first['translation']=='squadv2-v2'
    assert count==130214,count
    return {'rows':count,'first_row_keys':sorted(first),'positives':len(first['pos']),'negatives':len(first['neg']),
            'translation':first['translation'],'bytes':path.stat().st_size,'sha256':sha(path)}

report=json.loads(REPORT.read_text()) if REPORT.exists() else {}
archive=ROOT/'squadv2-v2-all.jsonl.gz'
checksum=ROOT/'squadv2-v2-all.jsonl.gz.sha256'
if archive.exists():
    assert checksum.read_text().split()[0]==EXPECTED,'Checksum sidecar differs from requested SHA-256'
    assert sha(archive)==EXPECTED,'Gzip SHA-256 mismatch'
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    if not TARGET.exists():
        fd,name=tempfile.mkstemp(prefix='squadv2-v2-unpacking-',suffix='.tmp',dir=TARGET.parent)
        os.close(fd)
        temporary=Path(name)
        try:
            with gzip.open(archive,'rb') as inp,temporary.open('wb') as out:
                shutil.copyfileobj(inp,out,8*1024*1024)
            report['squadv2']=validate(temporary)
            temporary.rename(TARGET)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        report['squadv2']=validate(TARGET)
    manifest=ROOT/'squadv2-v2-all.jsonl.manifest.json'
    if manifest.exists():
        moved=TARGET.with_name('squadv2-v2-all.jsonl.manifest.json')
        if moved.exists() and moved.read_bytes()!=manifest.read_bytes():raise RuntimeError('Different existing manifest')
        manifest.replace(moved)
    report['gzip_sha256']=EXPECTED
    save(report)
    archive.unlink();checksum.unlink()
else:
    report['squadv2']=validate(TARGET)
report['c_free_gb_after_unpack']=shutil.disk_usage(ROOT).free/1e9
save(report)
print('UNPACK_VALIDATED',json.dumps(report),flush=True)

from training.dataset_filters.core import apply_dataset_filter_if_configured
from training.multi_dataset import resolve_dataset_sources
from training.dataset_sources import materialize_huggingface_dataset_file,DatasetSourceError
result=apply_dataset_filter_if_configured(TARGET,SETTINGS)
report['squadv2_filter']={'total':result.report['total'],'kept':result.report['kept'],'report':result.report,'output':str(result.output_path)}
save(report)
print('FRESH_GATE',json.dumps(report['squadv2_filter']),flush=True)
assert 0.40<=result.report['kept']/result.report['total']<=0.50,'Gate yield differs materially from expected ~45%; stop before night start'
url='https://huggingface.co/datasets/speakleash/msmarco_pl/resolve/main/train_pl.jsonl'
try:
    hf=materialize_huggingface_dataset_file(url)
    report['msmarco']={'choice':'speakleash/msmarco_pl train_pl','path':str(hf.output_path),'cache_hit':hf.cache_hit,'url':url}
except DatasetSourceError as exc:
    reason=str(exc)
    report['msmarco_download_error']=reason
    save(report)
    cause=exc.__cause__
    if not isinstance(cause,urllib.error.HTTPError) or cause.code not in (401,403):raise
    fallback=list((ROOT/'datasets').rglob('msmarco-rejected-20000.jsonl'))
    if len(fallback)!=1:raise RuntimeError('HF access failed; exactly one msmarco-rejected-20000.jsonl fallback is required')
    report['msmarco']={'choice':'fallback msmarco-rejected-20000','path':str(fallback[0])}
save(report)
spec={'path':report['msmarco']['path'],'max_samples':150000,'seed':42,'sample_stage':'after_filter'}
sources=resolve_dataset_sources([spec],SETTINGS)
selected=sources[0].path
with selected.open() as f:count=sum(1 for _ in f)
report['msmarco'].update(selected_path=str(selected),selected_rows=count,sampling=spec)
report['c_free_gb_after_preparation']=shutil.disk_usage(ROOT).free/1e9
save(report)
print('DAYTIME_INPUTS_READY',json.dumps(report),flush=True)
