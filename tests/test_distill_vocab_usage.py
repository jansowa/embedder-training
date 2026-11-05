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


def test_load_texts_samples_the_whole_file(tmp_path):
    from distill.fidelity import load_texts

    corpus = tmp_path / "c.jsonl"
    with corpus.open("w") as f:
        for i in range(2000):
            f.write(json.dumps({"text": f"t{i}", "kind": "query" if i >= 1000 else "passage"}) + "\n")
    texts, kinds = load_texts(corpus, "text", 100, 1)
    assert len(texts) == 100 and len(set(texts)) == 100
    assert any(int(t[1:]) >= 1500 for t in texts)  # not just the head of the file
    q_texts, q_kinds = load_texts(corpus, "text", 10, 1, {"query"})
    assert set(q_kinds) == {"query"} and all(int(t[1:]) >= 1000 for t in q_texts)


def test_rank_agreement_is_perfect_for_the_teacher_itself(teacher_dir, tmp_path):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from distill import rank_agreement as ra
    from distill.splade import encode_pooled
    from distill_fixtures import sample_texts

    assert ra.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0 and ra.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    model = AutoModelForMaskedLM.from_pretrained(teacher_dir).eval()
    texts = sample_texts(12, seed=2)
    rows = []
    for q in range(3):
        docs = texts[q * 4 : q * 4 + 4]
        enc = tokenizer([texts[q]] + docs, padding=True, return_tensors="pt")
        with torch.no_grad():
            vec = encode_pooled(model, enc["input_ids"], enc["attention_mask"])
        scores = (vec[1:] @ vec[0]).tolist()
        rows.append({"query": texts[q], "pos": docs[:1], "neg": docs[1:], "pos_scores": scores[:1], "neg_scores": scores[1:]})
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "ra.json"
    assert ra.main(["--model", str(teacher_dir), "--rows", str(path), "--n", "3", "--device", "cpu", "--max-seq-length", "32", "--query-max-length", "32", "--output", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["rows"] == 3 and result["kendall_tau"] == pytest.approx(1.0) and result["top1_agreement"] == 1.0
