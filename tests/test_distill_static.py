"""Static sparse query encoder: table forward, initialisations, candidate pool and the training loop (CPU, tiny)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from distill_fixtures import WORDS, make_teacher, sample_texts  # noqa: E402

from distill.static_pool import load_pool, main as pool_main, quantize_rows, read_rows, to_csr  # noqa: E402
from distill.static_query import (  # noqa: E402
    StaticSparseQueryEncoder,
    credit_table,
    idf_weights,
    isolated_table,
    main as static_main,
    table_stats,
    weights_table,
)
from distill.train_static import main as train_main  # noqa: E402


def reference_forward(model: StaticSparseQueryEncoder, input_ids, attention_mask):
    out = torch.zeros(input_ids.shape[0], model.output_dim)
    for b in range(input_ids.shape[0]):
        for l in range(input_ids.shape[1]):
            if not attention_mask[b, l]:
                continue
            t = int(input_ids[b, l])
            for j in range(model.k):
                if not model.slot_mask[t, j]:
                    continue
                v, w = int(model.indices[t, j]), max(float(model.values[t, j]), 0.0)
                out[b, v] = max(out[b, v], w) if model.pooling == "max" else out[b, v] + w
    return out


@pytest.mark.parametrize("pooling", ["max", "sum"])
def test_forward_matches_the_slot_by_slot_reference(pooling):
    g = torch.Generator().manual_seed(0)
    rows, k, dim = 9, 3, 11
    indices = torch.randint(0, dim, (rows, k), generator=g)
    values = torch.randn(rows, k, generator=g)
    mask = torch.rand(rows, k, generator=g) > 0.3
    model = StaticSparseQueryEncoder(indices, values, dim, mask, pooling=pooling)
    input_ids = torch.tensor([[1, 4, 4, 7, 0], [2, 3, 8, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    assert torch.allclose(model(input_ids, attention), reference_forward(model, input_ids, attention))


def test_the_max_forward_has_a_gradient_on_the_winning_slots():
    model = StaticSparseQueryEncoder(torch.tensor([[0, 1], [1, 2]]), torch.tensor([[1.0, 0.5], [2.0, 0.3]]), 3)
    out = model(torch.tensor([[0, 1]]), torch.tensor([[1, 1]]))
    assert out.shape == (1, 3) and out[0].tolist() == pytest.approx([1.0, 2.0, 0.3])
    out.sum().backward()
    # dim 1: token 1 (2.0) wins over token 0 (0.5); token 0's slot for dim 1 gets no gradient
    assert model.values.grad.tolist() == [[1.0, 0.0], [1.0, 1.0]]


def test_weights_table_is_the_sparse_static_embedding_model():
    weights = torch.tensor([0.0, 1.5, 2.0, 0.5])
    model = weights_table(weights, 4)
    out = model(torch.tensor([[1, 3, 3]]), torch.tensor([[1, 1, 1]]))
    assert out.tolist() == [[0.0, 1.5, 0.0, 0.5]]


def test_save_load_and_prune_round_trip(tmp_path):
    model = StaticSparseQueryEncoder(torch.tensor([[0, 1], [2, 1]]), torch.tensor([[0.2, 1.0], [0.01, 3.0]]), 3, pooling="max", max_length=17, meta={"init": "x"})
    model.save(tmp_path / "m")
    loaded = StaticSparseQueryEncoder.load(tmp_path / "m")
    assert torch.equal(loaded.values, model.values) and torch.equal(loaded.indices, model.indices)
    assert loaded.max_length == 17 and loaded.meta["init"] == "x"
    pruned = loaded.pruned(min_weight=0.1, top_k=1)
    assert pruned.k == 1
    assert pruned.indices[:, 0].tolist() == [1, 1] and pruned.values[:, 0].tolist() == [1.0, 3.0]


def test_idf_ranks_rare_tokens_higher_and_zeroes_specials(tmp_path):
    make_teacher(tmp_path / "t")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tmp_path / "t")
    idf = idf_weights(tok, ["ala ma kota", "ala ma psy", "ala jest"], len(tok))
    ids = {w: tok.convert_tokens_to_ids(w) for w in ("ala", "ma", "kota", "dom")}
    assert idf[ids["ala"]] < idf[ids["ma"]] < idf[ids["kota"]] < idf[ids["dom"]]
    assert all(idf[i] == 0 for i in tok.all_special_ids)


def test_credit_table_on_one_query_reproduces_the_teacher_vector(tmp_path):
    from transformers import AutoTokenizer

    from distill.splade import encode_pooled

    teacher = make_teacher(tmp_path / "t")
    tok = AutoTokenizer.from_pretrained(tmp_path / "t")
    query = "ala ma kota i psy"
    indices, values, mask, info = credit_table(teacher, tok, [query], top_k=len(tok), min_count=0, isolated_fallback=False)
    model = StaticSparseQueryEncoder(indices, values, teacher.get_output_embeddings().weight.shape[0], mask)
    enc = tok([query], return_tensors="pt")
    with torch.no_grad():
        expected = encode_pooled(teacher, enc["input_ids"], enc["attention_mask"])
        got = model(enc["input_ids"], enc["attention_mask"])
    assert torch.allclose(got, expected, atol=1e-5)
    assert info["tokens_seen"] == len(set(enc["input_ids"][0].tolist()))


def test_isolated_table_fills_every_non_special_row(tmp_path):
    from transformers import AutoTokenizer

    teacher = make_teacher(tmp_path / "t")
    tok = AutoTokenizer.from_pretrained(tmp_path / "t")
    indices, values, mask = isolated_table(teacher, tok, top_k=4)
    live = mask.any(dim=1)
    for w in WORDS:
        assert live[tok.convert_tokens_to_ids(w)]
    assert not any(live[i] for i in tok.all_special_ids)


def test_build_cli_writes_a_loadable_table(tmp_path):
    make_teacher(tmp_path / "t")
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("".join(json.dumps({"text": t, "kind": "query"}) + "\n" for t in sample_texts(12)))
    for init in ("idf", "credit"):
        out = tmp_path / init
        assert static_main(["build", "--init", init, "--teacher", str(tmp_path / "t"), "--texts", str(corpus), "--top-k", "4", "--device", "cpu", "--dtype", "fp32", "--out", str(out)]) == 0
        model = StaticSparseQueryEncoder.load(out)
        assert table_stats(model)["live_rows"] > 0
        assert (out / "tokenizer.json").exists()


def test_quantized_csr_round_trip():
    vec = torch.tensor([[0.004, 0.006, 1.234, 0.0], [0.0, 0.0, 0.0, 2.0]])
    csr = to_csr(quantize_rows(vec, 100), 4, 100.0)
    assert csr.nnz_per_row().tolist() == [2, 1]
    assert torch.allclose(csr.dense_rows([1, 0]), torch.tensor([[0.0, 0.0, 0.0, 2.0], [0.0, 0.01, 1.23, 0.0]]))
    assert csr.document_frequency().tolist() == [0, 1, 1, 1]


def write_rows(path, n=24, seed=0):
    import random

    rng = random.Random(seed)
    texts = sample_texts(80, seed)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            q = texts[i]
            pos = [texts[(i + 1) % 80]]
            neg = [texts[(i + 2 + j) % 80] for j in range(4)]
            handle.write(json.dumps({"query": q, "pos": pos, "neg": neg, "pos_scores": [10.0 + rng.random()], "neg_scores": [rng.random() * 9 for _ in neg]}) + "\n")


def test_read_rows_deduplicates_passages(tmp_path):
    write_rows(tmp_path / "r.jsonl", n=10)
    rows, docs = read_rows(tmp_path / "r.jsonl")
    assert len(rows) == 10 and len(docs) == len(set(docs))
    assert all(r["n_pos"] == 1 and len(r["docs"]) == 5 and len(r["scores"]) == 5 for r in rows)


@pytest.mark.parametrize("target", ["rows", "splade", "mix"])
def test_pool_and_training_write_a_checkpoint(tmp_path, target):
    make_teacher(tmp_path / "t")
    write_rows(tmp_path / "r.jsonl")
    pool_dir = tmp_path / "pool"
    assert pool_main(["--doc-model", str(tmp_path / "t"), "--rows", str(tmp_path / "r.jsonl"), "--out", str(pool_dir), "--teacher-queries", "--device", "cpu", "--dtype", "fp32", "--max-seq-length", "32"]) == 0
    pool = load_pool(pool_dir)
    assert pool.meta["rows"] == 24 and pool.queries is not None and len(pool.queries) == 24
    corpus = tmp_path / "q.jsonl"
    corpus.write_text("".join(json.dumps({"query": r["query"]}) + "\n" for r in pool.rows))
    init = tmp_path / "init"
    assert static_main(["build", "--init", "credit", "--teacher", str(tmp_path / "t"), "--texts", str(corpus), "--field", "query", "--top-k", "4", "--device", "cpu", "--dtype", "fp32", "--out", str(init)]) == 0
    out = tmp_path / "run"
    extra = ["--in-batch"] if target != "rows" else []
    assert train_main(["--pool", str(pool_dir), "--init", str(init), "--out", str(out), "--target", target, "--max-steps", "4", "--batch-size", "4",
                       "--val-rows", "6", "--eval-every", "2", "--log-every", "1", "--lr", "1e-2", "--warmup-steps", "0", "--device", "cpu", "--query-l1", "1e-3"] + extra) == 0
    summary = json.loads((out / "static_summary.json").read_text())
    assert summary["steps"] == 4
    for key in ("val_splade_reference", "val_init", "val_final"):
        assert 0.0 <= summary[key]["mrr"] <= 1.0 and summary[key]["dims_per_query"] > 0
    trained = StaticSparseQueryEncoder.load(out / "final")
    assert not torch.equal(trained.values, StaticSparseQueryEncoder.load(init).values)
