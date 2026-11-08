"""Phrase rows of the static query table (distill.static_query): keys, encoding, credit init, save / load."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from distill_fixtures import make_teacher, sample_texts


def test_phrase_keys_strip_edge_punctuation_and_join_word_pairs():
    from distill.static_query import PHRASE_SEP, phrase_keys

    punct = np.zeros(20, dtype=bool)
    punct[9] = True  # "?"
    # <s> [5 6] [7] [8 9] </s>: a two-piece word, a one-piece word, a word with a trailing "?"
    ids = [0, 5, 6, 7, 8, 9, 2]
    word_ids = [None, 0, 0, 1, 2, 2, None]
    one = phrase_keys(ids, word_ids, punct, 1)
    assert one == [((5, 6), [1, 2])]  # only multi-piece words; "8 ?" is the one-piece word 8 after stripping
    two = phrase_keys(ids, word_ids, punct, 2)
    assert ((5, 6, PHRASE_SEP, 7), [1, 2, 3]) in two and ((7, PHRASE_SEP, 8), [3, 4]) in two and len(two) == 3


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    root = tmp_path_factory.mktemp("phrases")
    make_teacher(root / "teacher", layers=2, hidden=16, intermediate=24)
    tokenizer = AutoTokenizer.from_pretrained(root / "teacher")
    teacher = AutoModelForMaskedLM.from_pretrained(root / "teacher").eval()
    return root, tokenizer, teacher


def test_vocab_counts_texts_and_encoding_adds_the_phrase_rows(tiny):
    from distill.static_query import StaticSparseQueryEncoder, build_phrase_vocab

    _, tokenizer, _ = tiny
    texts = ["ala ma kota", "ala ma psy", "ala ma kota i psy"]
    vocab = build_phrase_vocab(tokenizer, texts, max_words=2, min_count=2, max_phrases=10)
    ala, ma, kota = (tokenizer.convert_tokens_to_ids(w) for w in ("ala", "ma", "kota"))
    assert vocab[0] == (ala, -1, ma)  # in all three texts
    assert (ma, -1, kota) in vocab and len(vocab) == 2
    base = len(tokenizer)
    rows = base + len(vocab)
    gen = torch.Generator().manual_seed(1)
    indices = torch.randint(0, base, (rows, 3), generator=gen)
    values = torch.rand(rows, 3, generator=gen)
    model = StaticSparseQueryEncoder(indices, values, base, phrases=vocab, phrase_max_words=2, base_rows=base)
    enc = model.tokenize(tokenizer, ["ala ma kota", "pies"])
    assert enc["input_ids"].shape[1] == 5 + 2  # <s> ala ma kota </s> + two phrase rows
    assert enc["input_ids"][0, -2:].tolist() == [base, base + 1] and enc["attention_mask"][1, -2:].tolist() == [0, 0]
    out = model(enc["input_ids"], enc["attention_mask"])
    expect = torch.zeros(base)
    for r in enc["input_ids"][0][enc["attention_mask"][0] > 0].tolist():
        for j in range(3):
            expect[indices[r, j]] = max(expect[indices[r, j]], values[r, j])
    assert torch.allclose(out[0], expect)


def test_credit_gives_a_phrase_row_the_dims_won_at_its_positions(tiny):
    from distill.static_query import build_phrase_vocab, credit_table, teacher_token_activations

    _, tokenizer, teacher = tiny
    text = "ala ma kota"
    vocab = build_phrase_vocab(tokenizer, [text], max_words=2, min_count=1, max_phrases=10)
    indices, values, mask, info = credit_table(teacher, tokenizer, [text], top_k=64, min_count=0, isolated_fallback=False,
                                               phrases=vocab, phrase_max_words=2)
    base = len(tokenizer)
    assert indices.shape[0] == base + len(vocab) and info["phrases_credited"] == len(vocab)
    enc = tokenizer([text], return_tensors="pt")
    pooled, where = teacher_token_activations(teacher, enc["input_ids"], enc["attention_mask"])
    row = base + vocab.index((tokenizer.convert_tokens_to_ids("ala"), -1, tokenizer.convert_tokens_to_ids("ma")))
    won = {int(v) for v in torch.nonzero(pooled[0] > 0).flatten() if int(where[0, v]) in (1, 2)}
    got = {int(indices[row, j]) for j in range(indices.shape[1]) if mask[row, j]}
    assert got == won
    for j in range(indices.shape[1]):
        if mask[row, j]:
            assert values[row, j] == pytest.approx(float(pooled[0, indices[row, j]]), rel=1e-5)


def test_save_load_and_prune_keep_the_phrases(tiny, tmp_path):
    from distill.static_query import StaticSparseQueryEncoder, build_phrase_vocab

    _, tokenizer, _ = tiny
    texts = sample_texts(40, seed=2)
    vocab = build_phrase_vocab(tokenizer, texts, max_words=2, min_count=2, max_phrases=50)
    base = len(tokenizer)
    gen = torch.Generator().manual_seed(3)
    rows = base + len(vocab)
    model = StaticSparseQueryEncoder(torch.randint(0, base, (rows, 4), generator=gen), torch.randn(rows, 4, generator=gen), base,
                                     phrases=vocab, phrase_max_words=2, base_rows=base)
    model.save(tmp_path / "t", tokenizer)
    again = StaticSparseQueryEncoder.load(tmp_path / "t")
    assert again.phrases == vocab and again.base_rows == base and again.phrase_max_words == 2
    assert torch.equal(again.encode(tokenizer, texts), model.encode(tokenizer, texts))
    pruned = model.pruned(top_k=2)
    assert pruned.phrases == vocab and pruned.k == 2
