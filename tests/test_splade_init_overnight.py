import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import splade_init_overnight as night
from training.splade_regularizer_control import RegularizerControlError, parse_regularizer_control, RegularizerControlCallback


@pytest.mark.parametrize('dims,tiny,alive,band,ok',[
    (19,60,False,False,False),(20,60,True,False,False),(150,50,True,True,True),
    (900,50,True,True,True),(901,60,True,False,False),(400,49.9,True,True,False),
    (400,None,True,True,False),(float('nan'),60,False,False,False),
])
def test_criteria_boundaries(dims,tiny,alive,band,ok):
    assert night.criteria(True,dims,tiny)=={'ALIVE':alive,'BAND':band,'OK':ok}
    assert not night.criteria(False,dims,tiny)['OK']


def test_branch_and_dense_gate():
    assert night.choose_branch({'OK':False},{'OK':True})=='A'
    assert night.choose_branch({'OK':True},{'OK':False})=='B'
    assert night.choose_branch({'OK':False},{'OK':False})=='C'
    assert night.should_evaluate(1500)
    assert not night.should_evaluate(1500.01)
    assert not night.should_evaluate(None)
    assert night.is_oom('torch.OutOfMemoryError: CUDA')
    assert not night.is_oom('segmentation fault')


def test_configs_keep_absolute_warmup_and_effective_batch(tmp_path,monkeypatch):
    original=night.CONFIGS/'modernbert_probe_m.yaml'
    (tmp_path/'modernbert_probe_m.yaml').write_text(original.read_text())
    monkeypatch.setattr(night,'CONFIGS',tmp_path)
    cfg=yaml.safe_load(night.configuration('n2-e5',{'control_start':250,'max_lambda':0.01},steps=2500,batch=4).read_text())
    st=cfg['sentence_transformers']
    assert st['max_steps']==2500 and st['warmup_steps']==250 and st['warmup_ratio']==0
    assert st['train_batch_size']==4 and st['gradient_accumulation_steps']==8
    assert st['keep_step_checkpoints']==[500,1000,1500,2000,2500]
    density=st['regularizer_control']['density']
    assert density['start_step']==250 and density['reference_dims']==436
    assert density['max_document_weight']==0.01
    assert parse_regularizer_control(st).density.start_step==250


def test_precision_control_preserves_numeric_shift_and_no_regularization(tmp_path,monkeypatch):
    (tmp_path/'modernbert_probe_m.yaml').write_text((night.CONFIGS/'modernbert_probe_m.yaml').read_text())
    monkeypatch.setattr(night,'CONFIGS',tmp_path)
    cfg=yaml.safe_load(night.configuration('n2-control',{'lambda_d':0,'lambda_q':0,'fp32':True,'c':7.25,'lr':5e-6}).read_text())
    st=cfg['sentence_transformers']
    assert st['bf16'] is False and st['fp16'] is False
    assert st['splade_decoder_bias_shift']==7.25
    assert 'regularizer_control' not in st
    assert st['splade_head_learning_rate']==5e-6


def test_density_control_can_start_before_lambda_warmup():
    config=parse_regularizer_control({'regularizer_control':{'warmup_steps':1000,'density':{'reference_dims':436,'start_step':250}}})
    fake=SimpleNamespace(config=config,last_control_step=None)
    assert RegularizerControlCallback._should_control(fake,250)
    legacy=parse_regularizer_control({'regularizer_control':{'warmup_steps':1000,'density':{'reference_dims':436}}})
    fake.config=legacy
    assert not RegularizerControlCallback._should_control(fake,250)
    assert RegularizerControlCallback._should_control(fake,1000)
    with pytest.raises(RegularizerControlError):
        parse_regularizer_control({'regularizer_control':{'density':{'reference_dims':436,'start_step':-1}}})


def test_fp32_alive_requires_real_nonzero_finite_outputs():
    diagnosis={'models':[{'dtype':'fp32','nonfinite_weights':[],
                          'outputs':[{'pooled_all_finite':True,'valid_token_active_dims_per_text':[30,40]}]}]}
    assert night.fp32_nonzero(diagnosis)
    diagnosis['models'][0]['outputs'][0]['valid_token_active_dims_per_text']=[0,40]
    assert not night.fp32_nonzero(diagnosis)


def test_partial_checkpoint_never_becomes_s500(tmp_path,monkeypatch):
    monkeypatch.setattr(night,'ROOT',tmp_path)
    run=tmp_path/'runs/modernbert-init-test-b4/run'
    ck=run/'checkpoint-100'
    ck.mkdir(parents=True)
    (ck/'model.safetensors').write_bytes(b'partial')
    (ck/'trainer_state.json').write_text(json.dumps({'global_step':100}))
    row={'physical_batch':4,'requested_steps':500}
    night.snapshot_results('test',row)
    assert row['trained_steps']==100
    assert not (tmp_path/'models/modernbert-init-test-s00500').exists()


def test_budget_never_launches_child_after_deadline(monkeypatch):
    monkeypatch.setattr(night,'remaining',lambda state:0)
    def forbidden(*args,**kwargs):
        raise AssertionError('a child must not start after the GPU deadline')
    monkeypatch.setattr(night.subprocess,'Popen',forbidden)
    with pytest.raises(night.BudgetExpired):
        night.command({},['python','train.py'],'unused.log',100)


def test_partial_save_without_trainer_state_is_not_exported(tmp_path,monkeypatch):
    monkeypatch.setattr(night,'ROOT',tmp_path)
    run=tmp_path/'runs/modernbert-init-test-b8/run'
    complete=run/'checkpoint-250'
    complete.mkdir(parents=True)
    (complete/'model.safetensors').write_bytes(b'weights250')
    (complete/'trainer_state.json').write_text(json.dumps({'global_step':250}))
    broken=run/'checkpoint-500'
    broken.mkdir()
    (broken/'model.safetensors').write_bytes(b'partial500')
    seen=[]
    monkeypatch.setattr(night.base,'export_model',lambda src,dst:seen.append(dst.name))
    row={'physical_batch':8,'requested_steps':500}
    night.snapshot_results('test',row)
    assert seen==['modernbert-init-test-s00250']
