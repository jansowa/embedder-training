from pathlib import Path

import pytest


def test_saved_roberta_bias_survives_export_and_runtime_restore(tmp_path):
    import torch
    from transformers import RobertaConfig, RobertaForMaskedLM
    from training.backends.splade_checkpoint_compat import compatible_export, restore_and_verify_mlm_weights, same_model_source, _sha

    model = RobertaForMaskedLM(RobertaConfig(vocab_size=31, hidden_size=8, num_hidden_layers=1,
                                           num_attention_heads=2, intermediate_size=16))
    with torch.no_grad():
        model.get_output_embeddings().bias.fill_(-16.8)
    source = tmp_path / "original"
    model.save_pretrained(source)
    original_sha = _sha(source / "model.safetensors")
    target = compatible_export(source, tmp_path / "export")
    loaded = RobertaForMaskedLM.from_pretrained(target)
    assert torch.equal(loaded.get_output_embeddings().bias, model.get_output_embeddings().bias)
    assert _sha(source / "model.safetensors") == original_sha
    assert compatible_export(source, target) == target
    assert same_model_source(source, target)
    assert not same_model_source(tmp_path / "another-model", target)

    # Exercise the repair even on versions where loading the old export is already correct.
    with torch.no_grad():
        loaded.get_output_embeddings().bias.zero_()
    wrapped = [type("Transformer", (), {"auto_model": loaded})()]
    result = restore_and_verify_mlm_weights(wrapped, source)
    assert result["verified_tensors"] > 10
    assert torch.equal(loaded.get_output_embeddings().bias, model.get_output_embeddings().bias)
    with torch.no_grad():
        loaded.roberta.embeddings.word_embeddings.weight.add_(1)
    with pytest.raises(ValueError, match="differs"):
        restore_and_verify_mlm_weights(wrapped, source)
    (source / "config.json").write_text("{}")
    assert not same_model_source(source, target)


def test_conflicting_bias_aliases_are_rejected(tmp_path):
    import torch
    from safetensors.torch import save_file
    from training.backends.splade_checkpoint_compat import add_decoder_bias_alias

    save_file({"lm_head.bias": torch.zeros(3), "lm_head.decoder.bias": torch.ones(3)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Conflicting"):
        add_decoder_bias_alias(tmp_path)


def test_cpu_reload_keeps_the_existing_parameter_objects(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from training.backends import splade_checkpoint_compat as compat
    import torch

    devices = []
    class Model(torch.nn.Module):
        trust_remote_code = False

        def __init__(self, checkpoint=None, *, device=None, trust_remote_code=False):
            super().__init__()
            if checkpoint is not None:
                devices.append(device)
            self.weight = torch.nn.Parameter(torch.ones(2))

    model = Model()
    parameter = model.weight
    with torch.no_grad():
        model.weight.zero_()
    monkeypatch.setattr(compat, "restore_and_verify_mlm_weights", lambda model, checkpoint: {"verified": True})
    assert compat.reload_sparse_model_on_cpu(SimpleNamespace(model=model), tmp_path) == {"verified": True}
    assert devices == ["cpu"]
    assert model.weight is parameter
    assert torch.equal(model.weight, torch.ones(2))
