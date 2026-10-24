import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import merge_splade_weights  # noqa: E402


TINY_VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "ala", "ma", "kota", "pies", "dom"]


def _write_tiny_masked_lm(path, *, seed, vocab_size=len(TINY_VOCAB), hidden_size=8):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(seed)
    config = transformers.BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=32,
    )
    model = transformers.BertForMaskedLM(config)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))

    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(TINY_VOCAB[:vocab_size]) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(str(path))
    return path


def test_resolve_weights_defaults_to_a_uniform_average():
    assert merge_splade_weights.resolve_weights(["a", "b", "c", "d"], None, None) == [0.25] * 4


def test_resolve_weights_maps_alpha_to_the_second_model():
    assert merge_splade_weights.resolve_weights(["a", "b"], None, 0.25) == [0.75, 0.25]


def test_resolve_weights_rejects_alpha_for_more_than_two_models():
    with pytest.raises(merge_splade_weights.MergeError, match="exactly two models"):
        merge_splade_weights.resolve_weights(["a", "b", "c"], None, 0.5)


def test_resolve_weights_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(merge_splade_weights.MergeError, match="must sum to 1"):
        merge_splade_weights.resolve_weights(["a", "b"], [0.5, 0.2], None)


def test_resolve_weights_rejects_a_weight_per_missing_model():
    with pytest.raises(merge_splade_weights.MergeError, match="3 values but 2 models"):
        merge_splade_weights.resolve_weights(["a", "b"], [0.2, 0.3, 0.5], None)


def test_resolve_weights_rejects_combining_weights_and_alpha():
    with pytest.raises(merge_splade_weights.MergeError, match="not both"):
        merge_splade_weights.resolve_weights(["a", "b"], [0.5, 0.5], 0.5)


def test_merged_model_averages_every_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    first = _write_tiny_masked_lm(tmp_path / "first", seed=1)
    second = _write_tiny_masked_lm(tmp_path / "second", seed=2)
    output = tmp_path / "merged"

    exit_code = merge_splade_weights.main(
        [
            "--models",
            str(first),
            str(second),
            "--weights",
            "0.5",
            "0.5",
            "--output",
            str(output),
            "--no-sanity-check",
        ]
    )
    assert exit_code == 0

    first_state = transformers.AutoModelForMaskedLM.from_pretrained(str(first)).state_dict()
    second_state = transformers.AutoModelForMaskedLM.from_pretrained(str(second)).state_dict()
    merged_state = transformers.AutoModelForMaskedLM.from_pretrained(str(output)).state_dict()

    assert set(merged_state) == set(first_state)
    assert any(tensor.is_floating_point() for tensor in merged_state.values())
    for name, merged_tensor in merged_state.items():
        expected = (first_state[name].float() + second_state[name].float()) / 2
        assert torch.allclose(merged_tensor.float(), expected, atol=1e-6), name


def test_merged_directory_is_a_sparse_encoder_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from convert_utils import is_sparse_sentence_transformer_dir

    first = _write_tiny_masked_lm(tmp_path / "first", seed=1)
    second = _write_tiny_masked_lm(tmp_path / "second", seed=2)
    output = tmp_path / "merged"

    assert merge_splade_weights.main(
        ["--models", str(first), str(second), "--alpha", "0.25", "--output", str(output), "--no-sanity-check"]
    ) == 0

    assert is_sparse_sentence_transformer_dir(str(output))
    assert (output / "1_SpladePooling" / "config.json").exists()
    assert (output / "tokenizer_config.json").exists()
    modules = json.loads((output / "modules.json").read_text(encoding="utf-8"))
    assert [module["path"] for module in modules] == ["", "1_SpladePooling"]


def test_merge_reports_mismatched_tensor_shapes(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    first = _write_tiny_masked_lm(tmp_path / "first", seed=1, hidden_size=8)
    second = _write_tiny_masked_lm(tmp_path / "second", seed=2, hidden_size=16)

    exit_code = merge_splade_weights.main(
        ["--models", str(first), str(second), "--output", str(tmp_path / "merged"), "--no-sanity-check"]
    )

    assert exit_code == 2
    error_output = capsys.readouterr().err
    assert "not compatible" in error_output
    assert "has shape" in error_output
    assert not (tmp_path / "merged").exists()


def test_merge_reports_mismatched_tokenizers(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    first = _write_tiny_masked_lm(tmp_path / "first", seed=1)
    second = _write_tiny_masked_lm(tmp_path / "second", seed=2)
    (second / "vocab.txt").write_text("\n".join(TINY_VOCAB[:-1]) + "\n", encoding="utf-8")

    exit_code = merge_splade_weights.main(
        ["--models", str(first), str(second), "--output", str(tmp_path / "merged"), "--no-sanity-check"]
    )

    assert exit_code == 2
    assert "tokenizer length" in capsys.readouterr().err


def test_sanity_check_reports_active_dimensions_and_cosines(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    first = _write_tiny_masked_lm(tmp_path / "first", seed=1)
    second = _write_tiny_masked_lm(tmp_path / "second", seed=2)
    output = tmp_path / "merged"

    assert merge_splade_weights.main(
        ["--models", str(first), str(second), "--output", str(output), "--device", "cpu", "--max-seq-length", "16"]
    ) == 0

    stdout = capsys.readouterr().out
    assert "Active dimensions:" in stdout
    assert "Cosine similarity against the merged model:" in stdout
    assert "<merged>" in stdout
