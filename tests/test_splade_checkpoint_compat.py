from pathlib import Path

import pytest


def test_saved_roberta_bias_survives_export_and_runtime_restore(tmp_path):
    import torch
    from transformers import RobertaConfig, RobertaForMaskedLM
    from training.backends.splade_checkpoint_compat import compatible_export, restore_and_verify_mlm_weights, _sha

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


def test_conflicting_bias_aliases_are_rejected(tmp_path):
    import torch
    from safetensors.torch import save_file
    from training.backends.splade_checkpoint_compat import add_decoder_bias_alias

    save_file({"lm_head.bias": torch.zeros(3), "lm_head.decoder.bias": torch.ones(3)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Conflicting"):
        add_decoder_bias_alias(tmp_path)
