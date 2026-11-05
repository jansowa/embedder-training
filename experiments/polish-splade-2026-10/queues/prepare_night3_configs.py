"""Produce paired configs and exact CPU-only input/run-budget preflight."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,os,shutil
import yaml

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/polish-splade-2026-10'
CONFIGS=EXP/'configs/next/5070'


def paired_configs(msmarco):
    source=Path(msmarco['path'])
    if source.is_absolute():
        try:source=source.relative_to(ROOT)
        except ValueError:pass
    hf={'path':str(source)}
    if msmarco['choice'].startswith('speakleash'):
        hf.update(max_samples=150000,seed=42,sample_stage='after_filter')
    data=[{'path':'datasets/arms/squadv2-v2-all.jsonl'},hf,
          {'path':'datasets/judged/msmarco-clean-20000.jsonl'},
          {'path':'datasets/judged/msmarco-plain-20000.jsonl'},
          {'path':'datasets/arms/squadv2-fresh-16000.jsonl'},
          {'path':'datasets/arms/squadv2-2073.jsonl'},
          {'path':'datasets/arms/fiqa-2073.jsonl'}]
    common={'backend':'sentence-transformers','training_type':'splade',
            'timestamp_output_dir':True,'save_resolved_config':True,
            'train_data':data,'dataset_mix_strategy':'concat_shuffle',
            'dataset_filter':'configs/dataset_filters/splade_positive_score_gt_23_50_no_metricx.yaml',
            'dataset_filter_cache_dir':'cache/filtered_datasets',
            'hparams':[{'learning_rate':2e-5,'num_train_epochs':1,'document_regularizer_weight':0.003,'loss':'sparse_multiple_negatives_ranking'}],
            'sentence_transformers':{
                'max_steps':4306,'max_seq_length':256,'train_batch_size':8,'gradient_accumulation_steps':4,
                'gradient_checkpointing':True,'splade_pooling_chunk_size':8,'batch_sampler':'no_duplicates',
                'seed':42,'warmup_steps':250,'warmup_ratio':0.0,'lr_scheduler_type':'constant_with_warmup',
                'scale':1.0,'negatives_per_query':6,'query_prefix':'','passage_prefix':'',
                'query_regularizer_weight':0.00015,'splade_regularizer_scheduler':False,
                'splade_decoder_bias_shift':'auto','splade_decoder_bias_target':400,'splade_train_only_head':False,
                'save_strategy':'steps','save_steps':4306,'save_total_limit':None,
                'keep_step_checkpoints':[2150,4306],'keep_epoch_checkpoints':False,'logging_steps':25,
                'splade_activation_stats':{'interval_steps':250,'sample_size':256,'batch_size':8,'include_negatives':True,'stop_below':5},
                'regularizer_control':{'warmup_steps':1000,'density':{
                    'reference_dims':436,'band':[0.75,1.05],'interval_steps':250,'factor':1.25,
                    'min_document_weight':0.0005,'max_document_weight':0.03,'query_ratio':0.05,
                    'sample_size':256,'batch_size':8,'quantization_factor':100}},'report_to':[]},
            'benchmark':{'run_pirb':False,'scope':'tiny','max_seq_length':512,'checkpoints':['final']}}
    configs={}
    for name,architecture in [('m','OPI-PIB/pl-ModernBERT-512-base'),('a','sdadas/polish-distilroberta')]:
        cfg=deepcopy(common)
        cfg['architectures']=[architecture]
        cfg['runs_dir']=f'runs/night3-paired-{name}'
        cfg['sentence_transformers'].update(bf16=name=='m',fp16=name=='a')
        configs[name]=cfg
    return configs


def budget_blockers(examples,free_bytes,steps=4306,min_free_gb=35):
    blockers=[]
    if math.ceil(examples/32)<steps:
        blockers.append(f'One epoch versus {steps} steps: {examples} examples, but {steps}x32={steps*32} presentations ({steps*32/examples:.4f} epochs).')
    if free_bytes<min_free_gb*1e9:
        blockers.append(f'Disk prerequisite: {free_bytes/1e9:.3f} GB free on C:, below required {min_free_gb} GB.')
    return blockers


def main():
    os.chdir(ROOT)
    from training.backends.sentence_transformers_backend import _backend_config,_load_training_rows
    inputs_path=EXP/'results/data/night3-inputs.json'
    report=json.loads(inputs_path.read_text())
    configs=paired_configs(report['msmarco'])
    assert configs['m']['train_data']==configs['a']['train_data']
    paths={}
    for name,cfg in configs.items():
        path=CONFIGS/f'mlm3_paired_{name}.yaml'
        path.write_text('# Prepared; preflight must pass before any queue launch. No resumes.\n'+yaml.safe_dump(cfg,sort_keys=False))
        paths[name]=str(path.relative_to(ROOT))
    cfg=configs['m']
    loaded=_load_training_rows(cfg,_backend_config(cfg),config_path=paths['m'])
    counts=Counter(loaded.dataset_ids)
    digest=hashlib.sha256()
    for row in loaded.rows:
        assert all(isinstance(row.get(k),str) and row[k] for k in ['anchor','positive']+[f'negative_{i}' for i in range(1,7)])
        digest.update(json.dumps([row['anchor'],row['positive']],ensure_ascii=False,separators=(',',':')).encode())
        digest.update(b'\n')
    sources=[]
    for source in loaded.sources:
        with source.path.open() as f:rows=sum(1 for _ in f)
        sources.append({'name':source.name,'original':source.original,'prepared_path':str(source.path),'gated_rows':rows,'examples':counts[source.name]})
    n=len(loaded.rows)
    free=shutil.disk_usage(ROOT).free
    blockers=budget_blockers(n,free)
    report['paired_preflight']={'configs':paths,'sources':sources,'examples':n,'gated_rows':sum(s['gated_rows'] for s in sources),
        'one_epoch_steps_approx':math.ceil(n/32),'epochs_at_5000':160000/n,'prepared_query_positive_order_sha256':digest.hexdigest(),
        'same_ordered_sources_and_seed':True,'free_gb':free/1e9,'blockers':blockers,'training_started':False}
    inputs_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# Night 3: paired-run preflight — not started','',
           'Both configs start from their original MLM, use the same ordered input files and seed 42, '
           'effective batch 32, bias initialization target 400, scale 1, LR 2e-5, 250 LR warmup steps, '
           'lambda warmup 1000 and density reference 436. M uses BF16; A uses FP16. '
           'Requested checkpoints: s2500/s5000. No resume is configured.','',
           '| Source | Gated rows | Training examples |','|---|---:|---:|']
    for source in sources:lines.append(f"| {source['original']} | {source['gated_rows']} | {source['examples']} |")
    lines+=['',f'**Total: {n} examples.** 5000 steps = 160000 presentations = {160000/n:.4f} epochs.',
            f'Approximate one-epoch step count: {math.ceil(n/32)}.',
            f"MS MARCO selection: {report['msmarco']['choice']}. HF request returned 401; authorized fallback selected.",
            f'Free C: space after preparation: {free/1e9:.3f} GB.','',
            '## Start blockers','',*[f'- {reason}' for reason in blockers],
            '', 'No GPU training, throughput, peak VRAM or night completion time is reported because the run has not started. '
            'Inputs and configs are ready for review; the original stop-before-start instruction is honored.', '',
            'Step 0: M s1500 = 55.63 at 356 dims; M s2000 = 56.61 at 404 dims; A s1875 = 55.70 at 391 dims. '
            'M s2000 − A s1875 = +0.91 nDCG points at similar density; E5 s2500 excluded.']
    (EXP/'results/summary_5070_mlm3.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['paired_preflight'],indent=2),flush=True)


if __name__=='__main__':main()
