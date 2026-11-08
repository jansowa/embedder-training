"""The Sentence Transformers packaging of a static query table (release.st_static, release.st_export)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from distill_fixtures import make_teacher, sample_texts


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    from transformers import AutoTokenizer

    from distill.static_query import StaticSparseQueryEncoder
    from release.st_export import export

    root = tmp_path_factory.mktemp("st_static")
    make_teacher(root / "doc", layers=2, hidden=16, intermediate=24)
    tokenizer = AutoTokenizer.from_pretrained(root / "doc")
    # the layout of a saved SparseEncoder (without these ST rebuilds the model from defaults)
    (root / "doc" / "config_sentence_transformers.json").write_text(json.dumps({"model_type": "SparseEncoder", "similarity_fn_name": "dot"}))
    pooling = root / "doc" / "1_SpladePooling" / "config.json"
    pooling.write_text(json.dumps(dict(json.loads(pooling.read_text()), word_embedding_dimension=len(tokenizer))))
    rows = dim = len(tokenizer)
    gen = torch.Generator().manual_seed(0)
    k = 3
    indices = torch.randint(0, dim, (rows, k), generator=gen)
    values = torch.randn(rows, k, generator=gen) * 2  # negatives exercise the relu
    mask = torch.rand(rows, k, generator=gen) > 0.2
    table = StaticSparseQueryEncoder(indices, values, dim, mask, max_length=16)
    table.save(root / "table", tokenizer)
    out = export(root / "doc", root / "table", root / "st")
    return root, table, tokenizer, out


def test_export_ships_the_classes_as_remote_code(packaged):
    _, _, _, out = packaged
    assert (out / "static_query_expansion.py").is_file()
    types = [m["type"] for m in json.loads((out / "modules.json").read_text())]
    assert types == ["static_query_expansion.StaticQueryRouter"]
    router = json.loads(next(out.rglob("router_config.json")).read_text())
    assert any(v == "static_query_expansion.SparseStaticExpansion" for v in router["types"].values())
    assert not any("release." in v for v in router["types"].values())


def test_queries_match_the_static_encoder_and_documents_match_splade(packaged):
    from sentence_transformers import SparseEncoder

    root, table, tokenizer, out = packaged
    model = SparseEncoder(str(out), trust_remote_code=True, device="cpu")
    texts = sample_texts(12, seed=3)
    q = model.encode_query(texts, convert_to_tensor=True).to_dense().numpy()
    ref = table.encode(tokenizer, texts).numpy()
    np.testing.assert_allclose(q, ref, rtol=1e-6, atol=1e-6)
    d = model.encode_document(texts, convert_to_tensor=True).to_dense().numpy()
    plain = SparseEncoder(str(root / "doc"), device="cpu").encode_document(texts, convert_to_tensor=True).to_dense().numpy()
    np.testing.assert_allclose(d, plain, rtol=1e-5, atol=1e-6)
    assert type(model[0].sub_modules["query"][0]).__name__ == "SparseStaticExpansion"


def test_saved_module_round_trips(packaged, tmp_path):
    from release.st_static import SparseStaticExpansion

    root, table, tokenizer, _ = packaged
    module = SparseStaticExpansion.from_table_dir(str(root / "table"))
    module.save(str(tmp_path / "m"))
    again = SparseStaticExpansion.from_table_dir(str(tmp_path / "m"))
    feats = module.tokenize(sample_texts(5, seed=9))
    a = module({k: v.clone() for k, v in feats.items()})["sentence_embedding"]
    b = again({k: v.clone() for k, v in feats.items()})["sentence_embedding"]
    assert torch.equal(a, b) and again.max_length == 16 and again.add_special_tokens


def test_query_only_export_and_st_options(packaged, tmp_path):
    from sentence_transformers import SparseEncoder

    from release.st_export import export_query_only

    root, table, tokenizer, out = packaged
    q_out = export_query_only(root / "table", tmp_path / "q")
    assert json.loads((q_out / "modules.json").read_text())[0]["type"] == "static_query_expansion.SparseStaticExpansion"
    texts = sample_texts(7, seed=5)
    ref = table.encode(tokenizer, texts).numpy()
    q_model = SparseEncoder(str(q_out), trust_remote_code=True, device="cpu")
    np.testing.assert_allclose(q_model.encode_query(texts, convert_to_tensor=True).to_dense().numpy(), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(q_model.encode_query(texts[0], convert_to_tensor=True).to_dense().numpy(), ref[0], rtol=1e-6, atol=1e-6)
    full = SparseEncoder(str(out), trust_remote_code=True, device="cpu")
    capped = full.encode_query(texts, convert_to_tensor=True, max_active_dims=2).to_dense()
    assert int((capped > 0).sum(dim=1).max()) <= 2
    dense = full.encode_query(texts, convert_to_tensor=True, convert_to_sparse_tensor=False)
    assert not dense.is_sparse
    np.testing.assert_allclose(dense.numpy(), ref, rtol=1e-6, atol=1e-6)


def test_phrase_rows_match_the_static_encoder(packaged, tmp_path):
    from sentence_transformers import SparseEncoder

    from distill.static_query import StaticSparseQueryEncoder, build_phrase_vocab
    from release.st_export import export_query_only

    root, _, tokenizer, _ = packaged
    texts = sample_texts(60, seed=11)
    vocab = build_phrase_vocab(tokenizer, texts, max_words=2, min_count=2, max_phrases=40)
    assert vocab
    base = len(tokenizer)
    gen = torch.Generator().manual_seed(5)
    rows = base + len(vocab)
    table = StaticSparseQueryEncoder(torch.randint(0, base, (rows, 3), generator=gen), torch.rand(rows, 3, generator=gen) * 3, base,
                                     phrases=vocab, phrase_max_words=2, base_rows=base, max_length=16)
    table.save(tmp_path / "table", tokenizer)
    q_model = SparseEncoder(str(export_query_only(tmp_path / "table", tmp_path / "q")), trust_remote_code=True, device="cpu")
    got = q_model.encode_query(texts[:20], convert_to_tensor=True).to_dense().numpy()
    np.testing.assert_allclose(got, table.encode(tokenizer, texts[:20]).numpy(), rtol=1e-6, atol=1e-6)
