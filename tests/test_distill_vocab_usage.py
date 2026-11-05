"""distill.vocab_usage: live / dead output dimensions of a SPLADE model on a text sample."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from distill_fixtures import make_teacher, write_corpus  # noqa: E402

from distill import vocab_usage  # noqa: E402


@pytest.fixture(scope="module")
def teacher_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("teacher")
    make_teacher(path)
    return path


def test_counts_are_consistent_and_cli_writes_json(teacher_dir, tmp_path):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    corpus = tmp_path / "corpus.jsonl"
    texts = write_corpus(corpus, n=12)
    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    model = AutoModelForMaskedLM.from_pretrained(teacher_dir).eval()
    stats = vocab_usage.usage_counts(model, tokenizer, texts, 32, 5, "cpu")
    vocab = model.config.vocab_size
    assert stats["counts"].shape == (vocab,)
    assert int(stats["active_per_text"].sum()) == int(stats["counts"].sum())
    assert (stats["mass"] >= 0).all() and (stats["peak"] >= 0).all()
    summary = vocab_usage.summarise(stats, len(texts))
    assert summary["dead_dims"] + summary["active_in_ge_1_texts"] == vocab
    assert summary["dims_for_99pct_mass"] <= vocab

    out = tmp_path / "usage.json"
    assert vocab_usage.main(["--model", str(teacher_dir), "--texts", str(corpus), "--n", "12", "--device", "cpu", "--per-dim", "--output", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["n_texts"] == 12
    assert len(result["per_dim"]["counts"]) == vocab
    assert result["dead_dims"] == summary["dead_dims"]
