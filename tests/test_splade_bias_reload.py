"""Regression for a shared RoBERTa bias lost by ordinary Transformers 4.48 loads."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


@pytest.fixture
def checkpoint(tmp_path):
    config = transformers.RobertaConfig(
        vocab_size=17, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16,
    )
    model = transformers.RobertaForMaskedLM(config)
    with torch.no_grad():
        model.lm_head.bias.fill_(-20)
    model.save_pretrained(tmp_path)
    return tmp_path


def test_sparsity_loader_preserves_saved_shared_bias(checkpoint, monkeypatch):
    from scripts.compare_splade_sparsity import ModelSpec, SpladeCounter
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: SimpleNamespace())
    args = SimpleNamespace(device='cpu', fp16=False, quantization_factor=100, max_seq_length=512)
    counter = SpladeCounter(ModelSpec('test', str(checkpoint)), args)
    assert counter.model.lm_head.bias is counter.model.lm_head.decoder.bias
    assert torch.equal(counter.model.lm_head.bias, torch.full((17,), -20.0))
    counter.close()


def test_pirb_loader_preserves_saved_shared_bias(checkpoint):
    # Execute the real method without importing PIRB's Java/indexing dependencies.
    source = Path(__file__).resolve().parents[1] / 'third_party/pirb/search/splade.py'
    module = ast.parse(source.read_text())
    encoder = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'SpladeEncoder')
    method = next(node for node in encoder.body if isinstance(node, ast.FunctionDef) and node.name == '_init_tokenizer_and_model')
    namespace = {'torch': torch, 'AutoModelForMaskedLM': transformers.AutoModelForMaskedLM,
                 'AutoTokenizer': SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace())}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
    encoder = SimpleNamespace(config={}, model_kwargs={}, model_name_or_path=str(checkpoint),
                              device='cpu', maxlen=512, use_bettertransformer=False)
    _, model = namespace['_init_tokenizer_and_model'](encoder)
    assert model.lm_head.bias is model.lm_head.decoder.bias
    assert torch.equal(model.lm_head.bias, torch.full((17,), -20.0))
