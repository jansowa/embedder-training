"""Path B: the larger static query encoders of distill.static_v2 (format 2) and the vector distillation (CPU, tiny)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from distill_fixtures import make_teacher, make_tokenizer, sample_texts  # noqa: E402

from distill.static_pool import main as pool_main  # noqa: E402
from distill.static_query import (  # noqa: E402
    PHRASE_SEP,
    CreditAccumulator,
    StaticSparseQueryEncoder,
    build_phrase_vocab,
    credit_accumulate,
    credit_table,
    load_static_encoder,
    main as static_main,
    phrase_keys,
)
from distill.static_v2 import StaticQueryEncoderV2, main as v2_main, phrase_keys_v2, phrase_parts, residual_phrase_rows  # noqa: E402
from distill.train_static import main as train_main  # noqa: E402
from distill.train_static_vec import main as vec_main  # noqa: E402

IDS = torch.tensor([[1, 4, 4, 7, 0], [2, 3, 8, 0, 0]])
ATTENTION = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])


def random_table(rows=9, k=3, dim=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, dim, (rows, k), generator=g), torch.randn(rows, k, generator=g), torch.rand(rows, k, generator=g) > 0.3, dim


def randomize(model, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.trainable_parameters()["extra"]:
            if p.dim() > 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.7 + (1.0 if p is model.length_scale else 0.0))


def reference(model, input_ids, attention_mask, context_mask=None):
    """The encoder written slot by slot."""
    batch, length = input_ids.shape
    out = torch.zeros(batch, model.output_dim)
    theta = max(float(model.threshold), 0.0)
    for b in range(batch):
        rows = [int(x) for x in input_ids[b]]
        is_token = [r < model.base_rows for r in rows]
        context = [int(context_mask[b, l]) if context_mask is not None else int(attention_mask[b, l]) * is_token[l] for l in range(length)]
        ctx = None
        if model.ctx_dim:
            ctx = sum(model.ctx_emb[min(rows[l], model.base_rows - 1)] * context[l] for l in range(length)) / max(sum(context), 1)
        scale = 1.0
        if model.length_scale is not None:
            scale = max(float(model.length_scale[sum(sum(context) > e for e in model.length_edges.tolist())]), 0.0)
        vec: dict[int, float] = {}
        for l in range(length):
            if not attention_mask[b, l]:
                continue
            r = rows[l]
            gate = 1.0
            if ctx is not None:
                gate = 2.0 / (1.0 + math.exp(-float(model.gate_vec[r] @ ctx + model.gate_bias[r])))
            for j in range(model.k):
                if not model.slot_mask[r, j]:
                    continue
                w = float(model.values[r, j])
                if not model.signed:
                    w = max(w, 0.0)
                w *= gate
                if model.slot_ctx_dim:
                    w *= 2.0 / (1.0 + math.exp(-float(model.slot_gate[r, j] @ (ctx @ model.ctx_proj))))
                v = int(model.indices[r, j])
                if not (is_token[l] and v == r):
                    w *= scale
                vec[v] = max(vec.get(v, 0.0), w) if model.pooling == "max" else vec.get(v, 0.0) + w
        dense = torch.zeros(model.output_dim)
        for v, w in vec.items():
            dense[v] = w
        if model.signed or model.learn_threshold or theta != 0.0:
            dense = torch.relu(dense - theta)
        if model.top_m:
            kth = torch.topk(dense, model.top_m).values[-1]
            dense = torch.where(dense >= kth, dense, torch.zeros_like(dense))
        out[b] = dense
    return out


@pytest.mark.parametrize("pooling", ["max", "sum"])
def test_a_converted_table_computes_what_format_1_computes(pooling):
    indices, values, mask, dim = random_table()
    v1 = StaticSparseQueryEncoder(indices, values, dim, mask, pooling=pooling)
    v2 = StaticQueryEncoderV2.from_v1(v1)
    assert torch.equal(v2(IDS, ATTENTION), v1(IDS, ATTENTION))
    assert v2.beyond_v1() == [] and v2.gain_foldable
    back = v2.to_v1()
    assert torch.equal(back(IDS, ATTENTION), v1(IDS, ATTENTION)) and back.pooling == pooling


@pytest.mark.parametrize(
    "switches",
    [
        {"pooling": "sum", "signed": True},
        {"pooling": "sum", "signed": True, "threshold": 0.3},
        {"threshold": 0.2},
        {"ctx_dim": 4},
        {"ctx_dim": 4, "slot_ctx_dim": 2},
        {"length_edges": [2, 3]},
        {"top_m": 3},
        {"pooling": "sum", "signed": True, "threshold": 0.1, "learn_threshold": True, "ctx_dim": 3, "slot_ctx_dim": 2, "length_edges": [3], "top_m": 4},
    ],
)
def test_the_forward_matches_the_slot_by_slot_reference(switches):
    indices, values, mask, dim = random_table()
    model = StaticQueryEncoderV2(indices, values, dim, mask, base_rows=9, seed=2, **switches)
    randomize(model)
    with torch.no_grad():
        assert torch.allclose(model(IDS, ATTENTION), reference(model, IDS, ATTENTION), atol=1e-5)
        context = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 0]])  # the context need not be the pooled rows
        assert torch.allclose(model(IDS, ATTENTION, context), reference(model, IDS, ATTENTION, context), atol=1e-5)
    assert model.beyond_v1()


def test_gates_and_multipliers_start_as_the_identity():
    indices, values, mask, dim = random_table()
    v1 = StaticSparseQueryEncoder(indices, values, dim, mask)
    v2 = StaticQueryEncoderV2.from_v1(v1, ctx_dim=6, slot_ctx_dim=3, length_edges=[2, 4], learn_threshold=True)
    assert torch.equal(v2(IDS, ATTENTION), v1(IDS, ATTENTION))
    names = {id(p) for p in v2.trainable_parameters()["extra"]}
    assert {id(v2.ctx_emb), id(v2.gate_vec), id(v2.gate_bias), id(v2.ctx_proj), id(v2.slot_gate), id(v2.length_scale), id(v2.threshold)} == names
    # the gate has a gradient although it starts closed to the context
    v2(IDS, ATTENTION).sum().backward()
    assert float(v2.gate_vec.grad.abs().sum()) > 0 and float(v2.length_scale.grad.abs().sum()) > 0


def test_invalid_switch_combinations_are_refused():
    indices, values, mask, dim = random_table()
    with pytest.raises(ValueError):
        StaticQueryEncoderV2(indices, values, dim, mask, signed=True)  # signed needs sum pooling
    with pytest.raises(ValueError):
        StaticQueryEncoderV2(indices, values, dim, mask, slot_ctx_dim=2)  # slot gates need a context
    with pytest.raises(ValueError):
        StaticQueryEncoderV2(indices, values, dim, mask, pair_window=0)


def test_save_load_keeps_every_tensor_and_switch(tmp_path):
    indices, values, mask, dim = random_table()
    phrases = [(1, 2), (1, PHRASE_SEP, 3)]
    model = StaticQueryEncoderV2(indices, values, dim, mask, pooling="sum", signed=True, threshold=0.25, learn_threshold=True, phrase_mode="replace",
                                 pair_key="first", pair_window=2, ctx_dim=4, slot_ctx_dim=2, length_edges=[3, 6], top_m=5, max_length=17,
                                 meta={"init": "x"}, phrases=phrases, phrase_max_words=2, base_rows=7)
    randomize(model)
    model.save(tmp_path / "m")
    assert (tmp_path / "m" / "static_query_v2.json").exists() and not (tmp_path / "m" / "static_query.json").exists()
    loaded = load_static_encoder(tmp_path / "m")
    assert isinstance(loaded, StaticQueryEncoderV2)
    got, expected = loaded.options(), model.options()
    assert got.pop("threshold") == pytest.approx(expected.pop("threshold")) and got == expected
    assert loaded.phrases == phrases and loaded.meta["init"] == "x"
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    with torch.no_grad():
        assert torch.equal(loaded(IDS, ATTENTION), model(IDS, ATTENTION))
    stats = loaded.stats()
    assert stats["format"] == "static-sparse-query/2" and stats["extra_parameters"] > 0 and stats["megabytes"] >= 0
    assert stats["negative_slots"] == int((loaded.slot_mask & (loaded.values < 0)).sum()) > 0
    with pytest.raises(ValueError):
        loaded.to_v1()


def test_reconfigured_keeps_trained_gates_and_with_table_keeps_everything_else():
    indices, values, mask, dim = random_table()
    model = StaticQueryEncoderV2(indices, values, dim, mask, ctx_dim=4, length_edges=[3])
    randomize(model)
    wider = model.reconfigured(top_m=4, slot_ctx_dim=2)
    assert torch.equal(wider.gate_vec, model.gate_vec) and torch.equal(wider.length_scale, model.length_scale)
    assert wider.top_m == 4 and wider.slot_gate.shape == (9, 3, 2) and float(wider.slot_gate.abs().sum()) == 0.0
    other = model.reconfigured(ctx_dim=6)
    assert other.gate_vec.shape == (9, 6) and float(other.gate_vec.abs().sum()) == 0.0  # another size starts again
    new_indices = torch.cat([indices, torch.zeros(9, 1, dtype=indices.dtype)], dim=1)
    new_values = torch.cat([values, torch.ones(9, 1)], dim=1)
    new_mask = torch.cat([mask, torch.ones(9, 1, dtype=torch.bool)], dim=1)
    grown = model.with_table(new_indices, new_values, new_mask)
    assert grown.k == 4 and torch.equal(grown.gate_vec, model.gate_vec) and grown.ctx_dim == 4
    with pytest.raises(ValueError):
        wider.with_table(new_indices, new_values, new_mask)  # slot gates are tied to the slots


def test_pair_keys_exact_first_and_window():
    punct = np.zeros(20, dtype=bool)
    punct[9] = True
    ids = [0, 5, 6, 7, 8, 9, 2]  # <s> [5 6] [7] [8 ?] </s>
    word_ids = [None, 0, 0, 1, 2, 2, None]
    assert phrase_keys_v2(ids, word_ids, punct, 2) == phrase_keys(ids, word_ids, punct, 2)
    assert phrase_keys_v2(ids, word_ids, punct, 1) == phrase_keys(ids, word_ids, punct, 1)
    window = phrase_keys_v2(ids, word_ids, punct, 2, pair_window=2)
    assert ((5, 6, PHRASE_SEP, 8), [1, 2, 4]) in window and len(window) == 4
    first = phrase_keys_v2(ids, word_ids, punct, 2, pair_key="first")
    assert first == [((5, 6), [1, 2]), ((5, PHRASE_SEP, 7), [1, 2, 3]), ((7, PHRASE_SEP, 8), [3, 4])]


def wordpiece_tokenizer(out_dir):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(["<s>", "<pad>", "</s>", "<unk>", "ala", "ma", "kot", "##y", "##ek", "pies"])}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    fast.save_pretrained(out_dir)
    return fast


def phrase_model(tokenizer, **switches):
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in ("ala", "ma", "kot", "##y")}
    phrases = [(ids["kot"], ids["##y"]), (ids["ala"], PHRASE_SEP, ids["ma"])]  # the word "koty", the pair "ala ma"
    base = len(tokenizer)
    indices = torch.arange(base + 2).clamp_max(base - 1).unsqueeze(1)
    return StaticQueryEncoderV2(indices, torch.ones(base + 2, 1), base, phrases=phrases, phrase_max_words=2, base_rows=base, **switches), ids, base


def test_tokenize_appends_phrase_rows_and_marks_the_context(tmp_path):
    tokenizer = wordpiece_tokenizer(tmp_path)
    model, ids, base = phrase_model(tokenizer)
    enc = model.tokenize(tokenizer, ["ala ma koty", "pies"])
    assert enc["input_ids"][0].tolist() == [ids["ala"], ids["ma"], ids["kot"], ids["##y"], base + 1, base]  # rows in the order they are met
    assert enc["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]]
    assert enc["context_mask"].tolist() == [[1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0]]


def test_replace_mode_pools_the_word_row_instead_of_its_pieces(tmp_path):
    tokenizer = wordpiece_tokenizer(tmp_path)
    model, ids, base = phrase_model(tokenizer, phrase_mode="replace")
    enc = model.tokenize(tokenizer, ["koty ma koty"])
    assert enc["input_ids"][0].tolist() == [ids["kot"], ids["##y"], ids["ma"], ids["kot"], ids["##y"], base]
    assert enc["attention_mask"][0].tolist() == [0, 0, 1, 0, 0, 1]  # both occurrences of the word give way to its row
    assert enc["context_mask"][0].tolist() == [1, 1, 1, 1, 1, 0]  # the context still reads every token
    out = model(**enc)[0]
    assert float(out[ids["kot"]]) == 0.0 and float(out[ids["ma"]]) == 1.0 and float(out[base - 1]) == 1.0  # the word row writes dim base - 1


def test_sum_pooling_counts_a_repeated_token_once(tmp_path):
    tokenizer = wordpiece_tokenizer(tmp_path)
    model, ids, _ = phrase_model(tokenizer, pooling="sum")
    enc = model.tokenize(tokenizer, ["ala pies ala"])
    assert enc["attention_mask"][0].tolist()[:3] == [1, 1, 0] and enc["context_mask"][0].tolist()[:3] == [1, 1, 1]
    assert float(model(**enc)[0, ids["ala"]]) == 1.0


def test_phrase_parts_are_the_rows_pooled_with_a_phrase():
    lookup = {(1, 2): 10, (1, 2, PHRASE_SEP, 3): 11, (4, 5): 12}
    assert phrase_parts((1, 2), lookup) == [1, 2]
    assert phrase_parts((1, 2, PHRASE_SEP, 3), lookup) == [1, 2, 3, 10]  # the pieces and the row of the word "1 2"
    assert phrase_parts((1, 2, PHRASE_SEP, 4, 5), lookup) == [1, 2, 4, 5, 10, 12]
    assert phrase_parts((1, PHRASE_SEP, 1), lookup) == [1]


def test_residual_rows_are_what_a_phrase_adds_to_its_parts():
    base, dim = 4, 6
    phrases = [(1, 2), (1, 2, PHRASE_SEP, 3)]
    acc = CreditAccumulator(base + 2, dim)
    acc.add(np.array([1, 2, 3]), np.array([0, 1, 2]), np.array([1.0, 0.5, 2.0]))  # token rows
    acc.add(np.array([4, 4, 4]), np.array([0, 1, 5]), np.array([1.0, 0.1, 0.7]))  # the word "1 2"
    acc.add(np.array([5, 5, 5, 5]), np.array([0, 1, 5, 2]), np.array([1.0, 0.1, 0.7, 2.5]))  # the pair "1 2" + "3"
    acc.counts[:] = 1
    indices, values, mask = acc.table(3)
    stats = residual_phrase_rows(acc, base, phrases, indices, values, mask, top_k=3)
    word = {int(d): float(v) for d, v, m in zip(indices[4], values[4], mask[4]) if m}
    assert word == {5: pytest.approx(0.7), 1: pytest.approx(-0.4)}  # adds dim 5, takes back most of what piece 2 writes
    pair = {int(d): float(v) for d, v, m in zip(indices[5], values[5], mask[5]) if m}
    assert pair == {2: pytest.approx(0.5)}  # everything else is already written by the pieces and the word row
    assert stats == {"residual_slots_positive": 2, "residual_slots_negative": 1, "residual_rows_empty": 0}
    model = StaticQueryEncoderV2(indices, values, dim, mask, pooling="sum", signed=True, phrases=phrases, phrase_max_words=2, base_rows=base)
    out = model(torch.tensor([[1, 2, 3, 4, 5]]), torch.ones(1, 5, dtype=torch.long))[0]
    assert out.tolist() == pytest.approx([1.0, 0.1, 2.5, 0.0, 0.0, 0.7])  # the pair's credit, rebuilt from five rows


def test_residual_rows_on_one_query_reproduce_the_teacher_vector(tmp_path):
    from transformers import AutoTokenizer

    from distill.splade import encode_pooled

    teacher = make_teacher(tmp_path / "t")
    tok = AutoTokenizer.from_pretrained(tmp_path / "t")
    query = "ala ma kota i psy"
    phrases = build_phrase_vocab(tok, [query], max_words=2, min_count=1, max_phrases=100)
    assert len(phrases) == 4
    acc = credit_accumulate(teacher, tok, [query], phrases=phrases, phrase_max_words=2)
    indices, values, mask, _ = credit_table(teacher, tok, [query], top_k=len(tok), min_count=0, isolated_fallback=False, phrases=phrases,
                                            phrase_max_words=2, accumulator=acc)
    stats = residual_phrase_rows(acc, len(tok), phrases, indices, values, mask, top_k=len(tok))
    assert stats["residual_rows_empty"] == 4  # on one text the token rows already hold every credit
    model = StaticQueryEncoderV2(indices, values, teacher.get_output_embeddings().weight.shape[0], mask, pooling="sum", signed=True,
                                 phrases=phrases, phrase_max_words=2, base_rows=len(tok))
    plain = tok([query], return_tensors="pt")
    with torch.no_grad():
        expected = encode_pooled(teacher, plain["input_ids"], plain["attention_mask"])
        got = model(**model.tokenize(tok, [query]))
    assert torch.allclose(got, expected, atol=1e-5)


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
    root = tmp_path_factory.mktemp("path_b")
    make_teacher(root / "t")
    write_rows(root / "r.jsonl")
    assert pool_main(["--doc-model", str(root / "t"), "--rows", str(root / "r.jsonl"), "--out", str(root / "pool"), "--teacher-queries",
                      "--device", "cpu", "--dtype", "fp32", "--max-seq-length", "32"]) == 0
    corpus = root / "c.jsonl"
    corpus.write_text("".join(json.dumps({"text": t, "kind": "query"}) + "\n" for t in sample_texts(60, 1)))
    assert static_main(["build", "--init", "credit", "--teacher", str(root / "t"), "--texts", str(corpus), "--top-k", "4", "--phrases", "2",
                        "--phrase-min-count", "2", "--device", "cpu", "--dtype", "fp32", "--out", str(root / "v1")]) == 0
    return root


def test_convert_cli_adds_switches_and_export_v1_refuses_them(tiny, tmp_path):
    assert v2_main(["convert", "--model", str(tiny / "v1"), "--out", str(tmp_path / "same")]) == 0
    v1, same = StaticSparseQueryEncoder.load(tiny / "v1"), load_static_encoder(tmp_path / "same")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny / "v1")
    texts = sample_texts(6, 3)
    assert torch.equal(same.encode(tok, texts), v1.encode(tok, texts))
    assert v2_main(["export-v1", "--model", str(tmp_path / "same"), "--out", str(tmp_path / "back")]) == 0
    assert torch.equal(StaticSparseQueryEncoder.load(tmp_path / "back").encode(tok, texts), v1.encode(tok, texts))
    assert v2_main(["convert", "--model", str(tmp_path / "same"), "--out", str(tmp_path / "gate"), "--ctx-dim", "8", "--length-edges", "4,7", "--learn-threshold"]) == 0
    gate = load_static_encoder(tmp_path / "gate")
    assert gate.ctx_dim == 8 and gate.length_edges.tolist() == [4, 7] and gate.learn_threshold
    assert torch.equal(gate.encode(tok, texts), v1.encode(tok, texts))  # still the identity
    assert v2_main(["export-v1", "--model", str(tmp_path / "gate"), "--out", str(tmp_path / "no")]) == 2
    assert v2_main(["convert", "--model", str(tiny / "v1"), "--out", str(tmp_path / "bad"), "--signed"]) == 2  # max pooling cannot be signed
    assert v2_main(["stats", "--model", str(tmp_path / "gate")]) == 0


@pytest.mark.parametrize("rows", ["credit", "residual"])
def test_build_cli_writes_a_loadable_encoder(tiny, tmp_path, rows):
    out = tmp_path / rows
    assert v2_main(["build", "--rows", rows, "--teacher", str(tiny / "t"), "--texts", str(tiny / "c.jsonl"), "--top-k", "4", "--phrases", "2",
                    "--phrase-min-count", "2", "--pair-key", "first", "--pair-window", "2", "--device", "cpu", "--dtype", "fp32", "--out", str(out)]) == 0
    model = load_static_encoder(out)
    assert isinstance(model, StaticQueryEncoderV2) and model.pair_key == "first" and model.pair_window == 2 and model.phrases
    assert (model.pooling, model.signed) == (("sum", True) if rows == "residual" else ("max", False))
    assert model.meta["init"] == f"credit/{rows}" and model.stats()["live_rows"] > 0
    if rows == "residual":
        assert "residual_slots_negative" in model.meta


@pytest.mark.parametrize("groups", ["all", "extra"])
def test_train_static_trains_a_format_2_encoder(tiny, tmp_path, groups):
    assert v2_main(["convert", "--model", str(tiny / "v1"), "--out", str(tmp_path / "init"), "--ctx-dim", "4", "--slot-ctx-dim", "2", "--length-edges", "5"]) == 0
    assert train_main(["--pool", str(tiny / "pool"), "--init", str(tmp_path / "init"), "--out", str(tmp_path / "run"), "--max-steps", "5", "--batch-size", "4",
                       "--val-rows", "6", "--eval-every", "2", "--log-every", "1", "--lr", "1e-2", "--extra-lr", "5e-2", "--warmup-steps", "0", "--device", "cpu",
                       "--learn-scale", "--train", groups]) == 0
    before, after = load_static_encoder(tmp_path / "init"), load_static_encoder(tmp_path / "run" / "final")
    assert isinstance(after, StaticQueryEncoderV2)
    assert float(after.gate_vec.abs().sum()) > 0 and not torch.equal(after.length_scale, before.length_scale)
    assert torch.equal(after.values, before.values) == (groups == "extra")
    summary = json.loads((tmp_path / "run" / "static_summary.json").read_text())
    assert summary["table_final"]["format"] == "static-sparse-query/2" and summary["table_final"]["ctx_dim"] == 4


def test_dim_gain_is_refused_when_it_cannot_be_folded(tiny, tmp_path):
    assert v2_main(["convert", "--model", str(tiny / "v1"), "--out", str(tmp_path / "init"), "--top-m", "8"]) == 0
    with pytest.raises(ValueError):
        train_main(["--pool", str(tiny / "pool"), "--init", str(tmp_path / "init"), "--out", str(tmp_path / "run"), "--max-steps", "1", "--batch-size", "4",
                    "--val-rows", "6", "--device", "cpu", "--dim-gain"])


def test_vector_distillation_trains_the_gates_and_leaves_the_table(tiny, tmp_path):
    assert v2_main(["convert", "--model", str(tiny / "v1"), "--out", str(tmp_path / "init"), "--ctx-dim", "4"]) == 0
    assert vec_main(["--teacher", str(tiny / "t"), "--init", str(tmp_path / "init"), "--out", str(tmp_path / "run"), "--texts", str(tiny / "c.jsonl"),
                     "--val-texts", "8", "--train", "extra", "--max-steps", "6", "--batch-size", "4", "--extra-lr", "5e-2", "--warmup-steps", "0",
                     "--eval-every", "3", "--log-every", "1", "--device", "cpu", "--dtype", "fp32"]) == 0
    before, after = load_static_encoder(tmp_path / "init"), load_static_encoder(tmp_path / "run" / "final")
    assert torch.equal(after.values, before.values) and float(after.gate_vec.abs().sum()) > 0
    summary = json.loads((tmp_path / "run" / "vec_summary.json").read_text())
    assert summary["steps"] == 6 and summary["val_init"]["texts"] == 8
    for key in ("cosine", "relative_error", "precision", "recall", "dims_student", "dims_teacher"):
        assert key in summary["val_final"]
    assert after.meta["train_state"]["stage"] == "vector"


def test_vector_distillation_fits_a_format_1_table(tiny, tmp_path):
    assert vec_main(["--teacher", str(tiny / "t"), "--init", str(tiny / "v1"), "--out", str(tmp_path / "run"), "--texts", str(tiny / "c.jsonl"),
                     "--val-texts", "8", "--train", "values", "--max-steps", "30", "--batch-size", "8", "--lr", "5e-2", "--warmup-steps", "0",
                     "--eval-every", "0", "--log-every", "10", "--device", "cpu", "--dtype", "fp32"]) == 0
    after = load_static_encoder(tmp_path / "run" / "final")
    assert isinstance(after, StaticSparseQueryEncoder)
    assert not torch.equal(after.values, StaticSparseQueryEncoder.load(tiny / "v1").values)
    summary = json.loads((tmp_path / "run" / "vec_summary.json").read_text())
    assert math.isfinite(summary["val_final"]["relative_error"]) and 0.0 <= summary["val_final"]["recall"] <= 1.0
