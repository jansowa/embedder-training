import importlib.util
from pathlib import Path

MODULE=Path(__file__).resolve().parents[1]/'experiments/polish-splade-2026-10/queues/prepare_night3_configs.py'
spec=importlib.util.spec_from_file_location('night3_configs',MODULE)
night3=importlib.util.module_from_spec(spec)
spec.loader.exec_module(night3)


def test_paired_configs_have_identical_data_seed_and_recipe():
    configs=night3.paired_configs({'choice':'fallback msmarco-rejected-20000','path':str(night3.ROOT/'datasets/judged/msmarco-rejected-20000.jsonl')})
    m,a=configs['m'],configs['a']
    assert m['train_data']==a['train_data']
    assert len(m['train_data'])==7
    assert m['train_data'][0]['path']=='datasets/arms/squadv2-v2-all.jsonl'
    assert m['train_data'][1]['path']=='datasets/judged/msmarco-rejected-20000.jsonl'
    assert m['architectures']==['OPI-PIB/pl-ModernBERT-512-base']
    assert a['architectures']==['sdadas/polish-distilroberta']
    for cfg in (m,a):
        st=cfg['sentence_transformers']
        assert st['seed']==42 and st['scale']==1 and st['max_steps']==5000
        assert st['train_batch_size']*st['gradient_accumulation_steps']==32
        assert st['warmup_steps']==250 and st['warmup_ratio']==0
        assert st['keep_step_checkpoints']==[2500,5000]
        assert st['regularizer_control']['warmup_steps']==1000
        assert st['regularizer_control']['density']['reference_dims']==436
        assert st['splade_decoder_bias_shift']=='auto' and st['splade_decoder_bias_target']==400
        assert 'resume' not in cfg and 'resume_from_checkpoint' not in st
    assert m['sentence_transformers']['bf16'] and not m['sentence_transformers']['fp16']
    assert a['sentence_transformers']['fp16'] and not a['sentence_transformers']['bf16']
    assert {k:v for k,v in m['sentence_transformers'].items() if k not in ('bf16','fp16')}=={k:v for k,v in a['sentence_transformers'].items() if k not in ('bf16','fp16')}


def test_hf_subsampling_is_after_filter():
    cfg=night3.paired_configs({'choice':'speakleash/msmarco_pl train_pl','path':'cache/hf/dataset.jsonl'})['m']
    assert cfg['train_data'][1]=={'path':'cache/hf/dataset.jsonl','max_samples':150000,'seed':42,'sample_stage':'after_filter'}


def test_preflight_catches_repetitions_and_disk_shortfall():
    reasons=night3.budget_blockers(137773,36.38e9)
    assert len(reasons)==2
    assert '1.1613 epochs' in reasons[0]
    assert '40 GB' in reasons[1]
    assert night3.budget_blockers(160000,40e9)==[]
    assert len(night3.budget_blockers(160000,39.99e9))==1
