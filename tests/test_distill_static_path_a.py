"""Path A of the static query table (same format, same shape): training options, slot growth, diagnostics (CPU, tiny)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from distill_fixtures import make_teacher, make_tokenizer, sample_texts  # noqa: E402

from distill.static_diag import ORACLE_MODES, combine_oracle, coverage, main as diag_main, own_token_mask, pool_oracles  # noqa: E402
from distill.static_pool import Pool, load_pool, main as pool_main, quantize_rows, save_pool, to_csr  # noqa: E402
from distill.static_query import PHRASE_SEP, StaticSparseQueryEncoder, load_static_encoder, main as static_main  # noqa: E402
from distill.static_slots import PairAccumulator, grow, main as slots_main, select_slots  # noqa: E402
from distill.train_static import StaticTrainer, main as train_main  # noqa: E402


def write_rows(path, n=40, seed=0):
    import random

    rng = random.Random(seed)
    texts = sample_texts(80, seed)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            neg = [texts[(i + 2 + j) % 80] for j in range(4)]
            handle.write(json.dumps({"query": texts[i], "pos": [texts[(i + 1) % 80]], "neg": neg, "pos_scores": [10.0 + rng.random()],
                                     "neg_scores": [rng.random() * 9 for _ in neg]}) + "\n")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny teacher, a pool of 40 rows with its query vectors and a credited table of 4 slots per token."""
    root = tmp_path_factory.mktemp("path_a")
    make_teacher(root / "t")
    write_rows(root / "r.jsonl")
    assert pool_main(["--doc-model", str(root / "t"), "--rows", str(root / "r.jsonl"), "--out", str(root / "pool"), "--teacher-queries",
                      "--device", "cpu", "--dtype", "fp32", "--max-seq-length", "32"]) == 0
    queries = root / "q.jsonl"
    queries.write_text("".join(json.dumps({"query": r["query"]}) + "\n" for r in load_pool(root / "pool").rows))
    assert static_main(["build", "--init", "credit", "--teacher", str(root / "t"), "--texts", str(queries), "--field", "query", "--top-k", "4",
                        "--device", "cpu", "--dtype", "fp32", "--out", str(root / "init")]) == 0
    return root


def train(root, out, *extra, steps=6):
    return train_main(["--pool", str(root / "pool"), "--init", str(root / "init"), "--out", str(out), "--max-steps", str(steps), "--batch-size", "4",
                       "--val-rows", "6", "--eval-every", "2", "--log-every", "1", "--lr", "1e-2", "--warmup-steps", "0", "--device", "cpu", *extra])


def test_sparse_gradients_equal_the_dense_ones():
    g = torch.Generator().manual_seed(0)
    model = StaticSparseQueryEncoder(torch.randint(0, 11, (9, 3), generator=g), torch.rand(9, 3, generator=g) + 0.1, 11)
    ids, attention = torch.tensor([[1, 4, 4, 7, 0], [2, 3, 8, 0, 0]]), torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    weights = torch.rand(2, 11, generator=g)
    (model(ids, attention) * weights).sum().backward()
    dense = model.values.grad.clone()
    model.values.grad = None
    model.sparse_grad = True
    (model(ids, attention) * weights).sum().backward()
    assert model.values.grad.is_sparse
    assert torch.allclose(model.values.grad.to_dense(), dense)


@pytest.mark.parametrize("optimizer", ["sparse_adam", "adagrad"])
def test_the_other_optimizers_train_the_table(tiny, tmp_path, optimizer):
    assert train(tiny, tmp_path / "run", "--learn-scale", "--optimizer", optimizer) == 0
    trained = StaticSparseQueryEncoder.load(tmp_path / "run" / "final")
    assert not torch.equal(trained.values, StaticSparseQueryEncoder.load(tiny / "init").values)
    summary = json.loads((tmp_path / "run" / "static_summary.json").read_text())
    assert summary["steps"] == summary["saved_step"] == 6
    assert summary["train_final"]["rows"] == 6 and 0.0 <= summary["train_final"]["mrr"] <= 1.0


def test_sparse_adam_refuses_gradient_clipping(tiny, tmp_path):
    assert train(tiny, tmp_path / "run", "--optimizer", "sparse_adam", "--max-grad-norm", "1.0") == 2


def test_the_value_floor_keeps_slots_alive_and_writes_them_as_zeros(tiny, tmp_path):
    # a learning rate this large throws many weights below zero in a few steps
    assert train(tiny, tmp_path / "plain", "--lr", "5.0") == 0
    assert bool((StaticSparseQueryEncoder.load(tmp_path / "plain" / "final").values < 0).any())
    assert train(tiny, tmp_path / "floor", "--lr", "5.0", "--value-floor", "1e-4") == 0
    values = StaticSparseQueryEncoder.load(tmp_path / "floor" / "final").values
    assert bool(((values == 0) | (values >= 0.005)).all())  # nothing negative, nothing below half a quantization step


def test_the_dimension_gain_is_folded_into_the_weights(tiny):
    from transformers import AutoTokenizer

    pool = load_pool(tiny / "pool")
    model = StaticSparseQueryEncoder.load(tiny / "init")
    args = argparse.Namespace(target="rows", learn_scale=False, teacher_scale=1.0, device="cpu", dim_gain=True, value_floor=0.0)
    trainer = StaticTrainer(pool, model, AutoTokenizer.from_pretrained(tiny / "init"), args)
    with torch.no_grad():
        trainer.log_gain.copy_(torch.randn(model.output_dim, generator=torch.Generator().manual_seed(3)) * 0.3)
        before = trainer.student_queries(pool.rows[:5])
        plain = model(**model.tokenize(trainer.tokenizer, [r["query"] for r in pool.rows[:5]]))
        trainer.finalize()
        after = trainer.student_queries(pool.rows[:5])
    assert not torch.allclose(before, plain)
    assert torch.allclose(before, after, atol=1e-6) and float(trainer.log_gain.abs().sum()) == 0.0


def test_gain_and_best_point_through_the_cli(tiny, tmp_path):
    assert train(tiny, tmp_path / "run", "--learn-scale", "--dim-gain", "--keep-best", "--best-metric", "top1_agreement", "--eval-every", "1", "--lr", "2.0") == 0
    summary = json.loads((tmp_path / "run" / "static_summary.json").read_text())
    assert summary["best_val"]["metric"] == "top1_agreement" and 1 <= summary["saved_step"] <= 6
    # the saved table is the best validation point seen, or the last step when that was at least as good
    assert summary["val_final"]["top1_agreement"] >= summary["best_val"]["value"] - 1e-9
    assert StaticSparseQueryEncoder.load(tmp_path / "run" / "final").k == 4


def test_train_extra_needs_an_encoder_with_extra_parameters(tiny, tmp_path):
    assert train(tiny, tmp_path / "run", "--train", "extra") == 2


# ---------------------------------------------------------------------- slots


def slot_fixture(tmp_path, n_rows=8):
    """Token rows that write only their own dimension; every query is "ala"; the teacher prefers the passage that
    carries "kota", the table the one with more "ala": the loss wants the row of "ala" to write "kota"."""
    tokenizer = make_tokenizer(tmp_path / "table")
    dim = len(tokenizer)
    ala, kota, ma = (tokenizer.convert_tokens_to_ids(w) for w in ("ala", "kota", "ma"))
    indices = torch.zeros(dim, 2, dtype=torch.long)
    values = torch.zeros(dim, 2)
    mask = torch.zeros(dim, 2, dtype=torch.bool)
    for token in range(4, dim):
        indices[token, 0], values[token, 0], mask[token, 0] = token, 1.0, True
    StaticSparseQueryEncoder(indices, values, dim, mask).save(tmp_path / "table", tokenizer)
    docs = torch.zeros(2, dim)
    docs[0, kota], docs[0, ala] = 2.0, 1.0
    docs[1, ala] = 1.2
    rows = [{"query": "ala", "docs": [0, 1], "n_pos": 1, "scores": [5.0, 1.0]} for _ in range(n_rows)]
    save_pool(tmp_path / "pool", {"quantization_factor": 100, "dim": dim, "rows": n_rows, "docs": 2}, rows, to_csr(quantize_rows(docs, 100), dim, 100.0), None)
    return ala, kota, ma


def test_slot_growth_adds_the_dimension_the_loss_asks_for(tmp_path):
    ala, kota, ma = slot_fixture(tmp_path)
    assert slots_main(["--pool", str(tmp_path / "pool"), "--model", str(tmp_path / "table"), "--out", str(tmp_path / "grown"), "--new-slots", "1",
                       "--val-rows", "0", "--min-support", "3", "--min-count", "3", "--batch-size", "4", "--device", "cpu"]) == 0
    before, grown = StaticSparseQueryEncoder.load(tmp_path / "table"), StaticSparseQueryEncoder.load(tmp_path / "grown")
    assert grown.indices.shape == before.indices.shape  # same format, same shape
    assert grown.indices[ala].tolist() == [ala, kota] and grown.slot_mask[ala].tolist() == [True, True]
    assert grown.values[ala].tolist() == pytest.approx([1.0, 0.05])
    assert torch.equal(grown.indices[ma], before.indices[ma]) and torch.equal(grown.slot_mask[ma], before.slot_mask[ma])
    report = json.loads((tmp_path / "grown" / "slots_report.json").read_text())
    assert report["written_into_empty_slots"] == 1 and report["rows_grown"] == 1 and report["written_over_live_slots"] == 0
    assert report["examples"][0]["row"] == "ala" and report["examples"][0]["dimension"] == "kota" and report["examples"][0]["queries_wanting"] == 8


def test_slot_growth_ignores_pairs_with_too_little_support(tmp_path):
    slot_fixture(tmp_path, n_rows=2)
    assert slots_main(["--pool", str(tmp_path / "pool"), "--model", str(tmp_path / "table"), "--out", str(tmp_path / "grown"), "--val-rows", "0",
                       "--min-support", "3", "--min-count", "3", "--batch-size", "2", "--device", "cpu"]) == 0
    assert json.loads((tmp_path / "grown" / "slots_report.json").read_text())["picks"] == 0


def test_grow_fills_dead_slots_first_and_replaces_only_below_the_bar():
    indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
    values = torch.tensor([[1.0, -0.2, 0.3], [0.9, 0.8, 0.7]])  # row 0: one dead slot, one light slot; row 1: full
    model = StaticSparseQueryEncoder(indices, values, 8)
    picks = {"rows": np.array([0, 0, 1]), "dims": np.array([6, 7, 6]), "gain": np.array([2.0, 1.0, 3.0]), "hits": np.array([5, 5, 5])}
    grown, outcome = grow(model, picks, init_value=0.05)
    assert outcome.tolist() == [1, 0, 0]  # only the dead slot of row 0 is free
    assert grown.indices[0].tolist() == [0, 6, 2] and grown.values[0].tolist() == pytest.approx([1.0, 0.05, 0.3])
    assert torch.equal(grown.indices[1], indices[1])
    grown, outcome = grow(model, picks, init_value=0.05, replace_below=0.5)
    assert outcome.tolist() == [1, 2, 0] and grown.indices[0].tolist() == [0, 6, 7]
    grown, outcome = grow(model, picks, init_value=0.05, new_k=4)
    assert grown.k == 4 and outcome.tolist() == [1, 1, 1] and grown.indices[1].tolist() == [3, 4, 5, 6]
    grown, _ = grow(model, picks, init_value=0.05, init_rel=0.5)
    assert float(grown.values[0, 1]) == pytest.approx(0.5 * (1.0 + 0.3) / 2)


def test_selection_skips_dimensions_a_row_already_writes():
    model = StaticSparseQueryEncoder(torch.tensor([[0, 1], [1, 2]]), torch.tensor([[1.0, 0.0], [1.0, 1.0]]), 4)
    acc = PairAccumulator(2, 4)
    acc.add(np.array([0, 0, 0, 1, 1]), np.array([0, 1, 3, 2, 3]), np.array([9.0, 4.0, 2.0, 9.0, 1.0]))
    acc.add(np.array([0, 0, 0, 1, 1]), np.array([0, 1, 3, 2, 3]), np.array([9.0, 4.0, 2.0, 9.0, 1.0]))
    picks = select_slots(acc, np.array([2, 2]), model, new_slots=2, min_support=2, min_count=1)
    # row 0 holds dimension 0 (live) but not 1 (its slot is dead); row 1 holds 1 and 2
    assert list(zip(picks["rows"].tolist(), picks["dims"].tolist())) == [(0, 1), (0, 3), (1, 3)]
    assert picks["gain"].tolist() == pytest.approx([4.0, 2.0, 1.0]) and picks["hits"].tolist() == [2, 2, 2]


def test_the_accumulator_drops_the_lighter_half_when_it_is_full():
    acc = PairAccumulator(4, 10, flush_every=1, max_pairs=4)
    for dim in range(6):
        acc.add(np.array([1]), np.array([dim]), np.array([float(dim + 1)]))
    acc.flush()
    assert acc.dropped > 0 and len(acc.keys) <= 4
    assert 15 in acc.keys.tolist()  # row 1, dimension 5: the heaviest pair survives


# ---------------------------------------------------------------------- diagnostics


def test_oracles_mix_the_two_vectors():
    teacher = torch.tensor([[1.0, 0.0, 2.0, 0.004]])
    student = torch.tensor([[0.5, 3.0, 0.0, 1.0]])
    own = torch.tensor([[True, False, False, True]])
    assert combine_oracle("weights", teacher, student, own).tolist() == [[1.0, 0.0, 0.0, pytest.approx(0.004)]]
    assert combine_oracle("support", teacher, student, own).tolist() == [[0.5, 0.0, 0.0, 0.0]]  # 0.004 is below half a step
    assert combine_oracle("own", teacher, student, own).tolist() == [[1.0, 0.0, 0.0, pytest.approx(0.004)]]
    assert combine_oracle("student-own", teacher, student, own).tolist() == [[0.5, 0.0, 0.0, 1.0]]
    with pytest.raises(ValueError):
        combine_oracle("nope", teacher, student, own)


def test_own_token_mask_leaves_special_tokens_out(tmp_path):
    tokenizer = make_tokenizer(tmp_path)
    mask = own_token_mask(tokenizer, ["ala ma kota", "pies"], len(tokenizer), 16)
    ids = {w: tokenizer.convert_tokens_to_ids(w) for w in ("ala", "ma", "kota", "pies")}
    assert sorted(mask[0].nonzero().flatten().tolist()) == sorted([ids["ala"], ids["ma"], ids["kota"]])
    assert mask[1].nonzero().flatten().tolist() == [ids["pies"]]


def test_pool_oracles_report_every_query_side(tiny, tmp_path):
    result = pool_oracles(tiny / "pool", tiny / "init", val_rows=10, device="cpu")
    for name in ("student", "teacher", *ORACLE_MODES):
        assert result[name]["rows"] == 10 and 0.0 <= result[name]["mrr"] <= 1.0
    assert result["weights"]["dims_per_query"] <= result["student"]["dims_per_query"]
    assert result["support"]["dims_per_query"] <= result["student"]["dims_per_query"]
    assert result["own"]["dims_per_query"] <= result["teacher"]["dims_per_query"]
    assert diag_main(["oracles", "--pool", str(tiny / "pool"), "--model", str(tiny / "init"), "--val-rows", "10", "--device", "cpu", "--out", str(tmp_path / "o.json")]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["student"] == result["student"]


def test_oracle_model_folder_names_the_table_and_the_teacher(tiny, tmp_path):
    assert diag_main(["oracle-model", "--table", str(tiny / "init"), "--teacher", str(tiny / "t"), "--mode", "support", "--out", str(tmp_path / "o")]) == 0
    config = json.loads((tmp_path / "o" / "static_oracle.json").read_text())
    assert config["mode"] == "support" and config["table"] == str((tiny / "init").resolve()) and config["teacher"] == str((tiny / "t").resolve())
    assert diag_main(["oracle-model", "--table", str(tmp_path / "missing"), "--teacher", str(tiny / "t"), "--mode", "own", "--out", str(tmp_path / "x")]) == 2


def wordpiece_tokenizer(out_dir):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(["<s>", "<pad>", "</s>", "<unk>", "ala", "ma", "kot", "##y", "##ek", "pies"])}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    fast.save_pretrained(out_dir)
    return fast


def test_coverage_counts_words_that_fall_back_to_pieces(tmp_path):
    tokenizer = wordpiece_tokenizer(tmp_path)
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in ("ala", "ma", "kot", "##y", "##ek")}
    phrases = [(ids["kot"], ids["##y"]), (ids["ala"], PHRASE_SEP, ids["ma"])]  # the word "koty", the pair "ala ma"
    base = len(tokenizer)
    indices = torch.arange(base + 2).clamp_max(base - 1).unsqueeze(1)
    values = torch.ones(base + 2, 1)
    values[ids["##ek"]] = 0.0  # a dead piece row
    model = StaticSparseQueryEncoder(indices, values, base, phrases=phrases, phrase_max_words=2, base_rows=base)
    report = coverage(model, tokenizer, ["ala ma koty", "kotek ma"])
    assert report["texts"] == 2 and report["word_rows"] == 1 and report["pair_rows"] == 1
    assert report["words_per_text"] == 2.5 and report["pieces_per_word"] == pytest.approx(7 / 5)
    assert report["multi_piece_words"] == pytest.approx(0.4) and report["multi_piece_words_with_row"] == pytest.approx(0.5)
    assert report["words_falling_back_to_pieces"] == pytest.approx(0.2)  # "kotek" of five words
    assert report["fallback_pieces_with_live_row"] == pytest.approx(0.5)  # "kot" is live, "##ek" is dead
    assert report["adjacent_pairs_with_row"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["top_uncovered_words"][0][1] == 1


def test_load_static_encoder_reads_format_1(tiny):
    assert isinstance(load_static_encoder(tiny / "init"), StaticSparseQueryEncoder)


def test_prune_can_change_the_query_length_and_nothing_else(tiny, tmp_path):
    assert static_main(["prune", "--model", str(tiny / "init"), "--out", str(tmp_path / "long"), "--max-length", "512"]) == 0
    before, after = StaticSparseQueryEncoder.load(tiny / "init"), StaticSparseQueryEncoder.load(tmp_path / "long")
    assert (before.max_length, after.max_length) == (128, 512) and after.meta["max_length_changed_to"] == 512
    assert torch.equal(torch.relu(before.values) * before.slot_mask, after.values) and torch.equal(before.slot_mask & (before.values > 0), after.slot_mask)
