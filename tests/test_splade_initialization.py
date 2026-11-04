from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
from torch import nn
from training.backends.splade_initialization import (
    choose_bias_shift, configure_head_optimizer, configure_trainable,
    initialize_decoder, microbatch, quantized_counts, update_density_streak,
)

class MLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.head = nn.Linear(3, 3)
        self.decoder = nn.Linear(3, 7)
    def get_output_embeddings(self):
        return self.decoder

def model():
    wrapper = nn.Module()
    wrapper.auto_model = MLM()
    return nn.Sequential(wrapper)

def test_auto_shift_hits_target_and_is_monotonic():
    torch.manual_seed(42)
    maxima = torch.randn(256, 1000) * 2 + 5
    c = choose_bias_shift(maxima, 400)
    assert abs(float(quantized_counts(maxima, c).float().median()) - 400) <= 1
    assert choose_bias_shift(maxima, 200) > c
    assert torch.all(quantized_counts(maxima, c) <= quantized_counts(maxima))
    with pytest.raises(ValueError): choose_bias_shift(maxima, 0)

def test_head_only_and_optimizer_preserve_decay_groups():
    m = model()
    configure_trainable(m, {'splade_train_only_head': True})
    assert all(not p.requires_grad for p in m[0].auto_model.backbone.parameters())
    assert all(p.requires_grad for p in m[0].auto_model.head.parameters())
    m = model()
    params = list(m.parameters())
    trainer = SimpleNamespace(optimizer=None)
    def create():
        if trainer.optimizer is None:
            trainer.optimizer = torch.optim.AdamW([
                dict(params=[p for n,p in m.named_parameters() if not n.endswith('bias')],weight_decay=.1),
                dict(params=[p for n,p in m.named_parameters() if n.endswith('bias')],weight_decay=0),
            ],lr=2e-5)
        return trainer.optimizer
    trainer.create_optimizer = create
    configure_head_optimizer(trainer,m,{'splade_head_learning_rate': 8e-5})
    opt = trainer.create_optimizer()
    assert len(opt.param_groups) == 4
    assert len({id(p) for g in opt.param_groups for p in g['params']}) == len(params)
    for g in opt.param_groups:
        for p in g['params']:
            is_backbone = any(p is q for q in m[0].auto_model.backbone.parameters())
            assert g['lr'] == (2e-5 if is_backbone else 8e-5)
    assert trainer.create_optimizer() is opt
    assert len(opt.param_groups) == 4

def test_guard_and_microbatch_order():
    assert update_density_streak(1, 4.9) == 2
    assert update_density_streak(1, 5) == 0
    rows = [dict(anchor=f'q{n}',positive=f'p{n}',**{f'negative_{i}':f'n{n}-{i}' for i in range(1,7)}) for n in range(8)]
    q,d = microbatch(rows)
    assert len(q)==8 and len(d)==56
    assert d[:8]==[f'p{n}' for n in range(8)]
    assert d[8:16]==[f'n{n}-1' for n in range(8)]


def test_initialization_changes_only_bias_and_retries_lower_target(monkeypatch, tmp_path):
    import training.backends.splade_initialization as init
    m = model()
    original = {name: p.detach().clone() for name, p in m.named_parameters()}
    initial_bias = m[0].auto_model.decoder.bias.detach().clone()
    maxima = torch.randn(256, 1000) * 2 + 5
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(init, 'raw_maxima', lambda *a, **k: maxima - float((initial_bias - m[0].auto_model.decoder.bias).mean()))
    losses = iter([20.0, 5.0])
    monkeypatch.setattr(init, 'score_diagnostics', lambda *a, **k: {'loss': next(losses)})
    result = initialize_decoder(m, [{'positive': str(i)} for i in range(256)], {'splade_decoder_bias_shift': 'auto'}, tmp_path)
    assert [a['target'] for a in result['attempts']] == [400, 200]
    assert abs(result['median_dims_after'] - 200) <= 1
    assert result['attempts'][1]['c'] > result['attempts'][0]['c']
    assert m.training
    for name, p in m.named_parameters():
        expected = original[name] - result['c'] if name.endswith('decoder.bias') else original[name]
        assert torch.allclose(p, expected)


def test_numeric_bias_and_rollback_on_failure(monkeypatch, tmp_path):
    import training.backends.splade_initialization as init
    m = model()
    bias = m[0].auto_model.decoder.bias
    original = bias.detach().clone()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(init, 'raw_maxima', lambda *a, **k: torch.ones(256, 1000))
    monkeypatch.setattr(init, 'score_diagnostics', lambda *a, **k: {'loss': 5.0})
    initialize_decoder(m, [{'positive': 'p'}] * 256, {'splade_decoder_bias_shift': 2.0}, tmp_path)
    assert torch.allclose(bias, original - 2)
    def fail(*a, **k):
        raise RuntimeError('diagnostic failure')
    monkeypatch.setattr(init, 'score_diagnostics', fail)
    before = bias.detach().clone()
    with pytest.raises(RuntimeError):
        initialize_decoder(m, [{'positive': 'p'}] * 256, {'splade_decoder_bias_shift': 3.0}, tmp_path)
    assert torch.equal(bias, before)
    assert m.training


def test_head_only_keeps_tied_decoder_shared():
    m = model()
    auto = m[0].auto_model
    auto.embeddings = nn.Embedding(7, 3)
    auto.decoder.weight = auto.embeddings.weight
    configure_trainable(m, {'splade_train_only_head': True})
    assert auto.decoder.weight is auto.embeddings.weight
    assert auto.decoder.weight.requires_grad
    assert not auto.backbone.weight.requires_grad
    with pytest.raises(ValueError):
        configure_trainable(m, {'splade_train_only_head': 'false'})


def test_density_callback_stops_after_two_low_measurements(tmp_path, monkeypatch):
    from training.backends.sentence_transformers_backend import SpladeActivationStatsCallback
    cb = SpladeActivationStatsCallback(model=None, query_texts=[], document_texts=[], batch_size=1,
                                      interval_steps=25, quantization_factor=100, log_on_train_begin=False,
                                      log_on_train_end=False, prefix='stats')
    cb.stop_below = 5
    monkeypatch.setattr(cb, '_log_to_wandb', lambda *a: None)
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    state = SimpleNamespace(global_step=25, log_history=[])
    args = SimpleNamespace(output_dir=str(tmp_path))
    for step, value, stopped in [(25, 4.0, False), (50, 6.0, False), (75, 4.0, False), (100, 3.0, True)]:
        state.global_step = step
        monkeypatch.setattr(cb, '_compute_metrics', lambda v=value: {'stats/document_active_dims_mean': v})
        cb.on_log(args, state, control, logs={})
        assert control.should_training_stop == stopped
    assert (tmp_path / 'splade_density_stop.json').exists()
