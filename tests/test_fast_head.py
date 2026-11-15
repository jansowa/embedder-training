"""release.fast_head on the CPU: the patched SparseEncoder gives the stock vectors, and the patch is reversible."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from distill_fixtures import make_teacher, sample_texts  # noqa: E402
from release.fast_head import FastHeadError, disable_fast_head, enable_fast_head, masked_max_logits  # noqa: E402


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny")
    make_teacher(out)
    (out / "config_sentence_transformers.json").write_text(json.dumps({"model_type": "SparseEncoder", "similarity_fn_name": "dot"}))
    return out


def _load(model_dir, dtype=torch.float32):
    from sentence_transformers import SparseEncoder

    model = SparseEncoder(str(model_dir), device="cpu", model_kwargs={"torch_dtype": dtype})
    decoder = model[0].auto_model.get_output_embeddings()
    with torch.no_grad():  # the fixture's bias may be dropped by the fp32 loader; give the head a real bias
        g = torch.Generator().manual_seed(3)
        decoder.bias.copy_(torch.randn(decoder.bias.shape, generator=g).to(decoder.bias.dtype) * 2.0)
    return model.eval()


def _texts():
    # Different lengths: every batch carries padding; one long text so that slices split real tokens.
    return sample_texts(40, seed=5) + [" ".join(sample_texts(12, seed=9))]


def _vectors(model, texts, batch_size=7):
    return model.encode_document(texts, batch_size=batch_size, convert_to_sparse_tensor=False, show_progress_bar=False).cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_whole_sequence_is_bitwise_identical(model_dir, dtype):
    model = _load(model_dir, dtype)
    texts = _texts()
    stock = _vectors(model, texts)
    enable_fast_head(model, chunk_tokens=0)
    fast = _vectors(model, texts)
    assert (stock > 0).sum() > 0
    assert torch.equal(stock, fast)


@pytest.mark.parametrize("chunk", [1, 3, 8, 1000])
def test_slices_match_stock(model_dir, chunk):
    model = _load(model_dir)
    texts = _texts()
    stock = _vectors(model, texts)
    enable_fast_head(model, chunk_tokens=chunk)
    fast = _vectors(model, texts)
    assert torch.allclose(stock, fast, rtol=0, atol=1e-6)
    assert torch.equal((stock * 100).round() > 0, (fast * 100).round() > 0)  # the dims PIRB indexes


def test_masked_max_reference():
    g = torch.Generator().manual_seed(0)
    h, w, b = torch.randn(3, 5, 4, generator=g), torch.randn(6, 4, generator=g), torch.randn(6, generator=g)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    ref = (torch.nn.functional.linear(h, w, b) * mask.unsqueeze(-1)).amax(dim=1)
    for chunk in (0, 1, 2, 5, 9):
        assert torch.allclose(masked_max_logits(h, w, b, mask, chunk), ref, atol=1e-6)


def test_reversible_and_weights_untouched(model_dir):
    model = _load(model_dir)
    texts = _texts()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stock = _vectors(model, texts)
    enable_fast_head(model, chunk_tokens=4)
    with pytest.raises(FastHeadError):
        enable_fast_head(model)
    assert model.state_dict().keys() == before.keys()
    disable_fast_head(model)
    assert torch.equal(_vectors(model, texts), stock)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert "forward" not in model[0].__dict__


def test_grad_mode_takes_the_stock_path(model_dir):
    model = _load(model_dir)
    enable_fast_head(model, chunk_tokens=4)
    features = model.tokenize(_texts()[:3])
    with torch.enable_grad():
        out = model[0](dict(features))
    assert out["token_embeddings"].dim() == 3 and out["token_embeddings"].shape[1] == features["input_ids"].shape[1]
