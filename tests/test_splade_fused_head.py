"""The sparse-gradient SPLADE head (training/backends/splade_fused_head.py): CPU, small tensors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from training.backends.splade_fused_head import (  # noqa: E402
    SpladeFusedHeadError,
    install_fused_head,
    parse_fused_head,
    sparse_max_decoder,
    uninstall_fused_head,
)

B, L, H, V = 3, 7, 5, 11


def stock_pooling(logits: torch.Tensor, mask: torch.Tensor, chunk_size: int | None) -> torch.Tensor:
    """SpladePooling(max, relu) in training mode, as sentence-transformers 5.3.0 computes it."""
    expanded = mask.unsqueeze(-1).to(logits.dtype)
    pooled = torch.full((logits.shape[0], logits.shape[2]), float("-inf"), dtype=logits.dtype)
    chunk = logits.shape[1] if not chunk_size else chunk_size
    for i in range(0, logits.shape[1], chunk):
        piece = (logits[:, i : i + chunk] * expanded[:, i : i + chunk]).relu_().log1p()
        pooled = torch.maximum(pooled, torch.max(piece, dim=1)[0])
    return pooled


def fused_pooling(h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor, chunk: int) -> torch.Tensor:
    best = sparse_max_decoder(h, weight, bias, mask, chunk).unsqueeze(1)
    return stock_pooling(best, mask[:, :1], None)


def make_inputs(seed: int = 0, dtype=torch.float64, distinct: bool = True):
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(B, L, H, generator=gen, dtype=dtype, requires_grad=True)
    weight = torch.randn(V, H, generator=gen, dtype=dtype, requires_grad=True)
    bias = torch.randn(V, generator=gen, dtype=dtype, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[1, 4:] = 0  # padding in the second sequence
    mask[2, 2:] = 0
    return h, weight, bias, mask


def loss_of(pooled: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.5, 1.5, pooled.shape[1], dtype=pooled.dtype)
    return (pooled * weights).sum() + (pooled**2).sum()


@pytest.mark.parametrize("stock_chunk", [None, 2])
@pytest.mark.parametrize("fused_chunk", [0, 3])
def test_fused_head_matches_the_stock_pooling(stock_chunk, fused_chunk):
    h, weight, bias, mask = make_inputs()
    stock = stock_pooling(nn.functional.linear(h, weight, bias), mask, stock_chunk)
    grads_stock = torch.autograd.grad(loss_of(stock), (h, weight, bias))

    fused = fused_pooling(h, weight, bias, mask, fused_chunk)
    grads_fused = torch.autograd.grad(loss_of(fused), (h, weight, bias))

    if fused_chunk == 0:
        assert torch.equal(fused, stock)  # exact mode: the same linear kernel on the same shapes
    else:
        assert torch.allclose(fused, stock, atol=1e-14, rtol=1e-14)  # chunked: another matmul split, 1 ulp
    for name, a, b in zip(("h", "weight", "bias"), grads_stock, grads_fused):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-10), name
    # padding never receives a gradient
    assert torch.count_nonzero(grads_fused[0][1, 4:]) == 0
    assert torch.count_nonzero(grads_fused[0][2, 2:]) == 0


def test_fused_head_passes_gradcheck():
    h, weight, bias, mask = make_inputs(seed=1)
    assert torch.autograd.gradcheck(lambda a, w, b: sparse_max_decoder(a, w, b, mask, 0), (h, weight, bias))
    assert torch.autograd.gradcheck(lambda a, w, b: sparse_max_decoder(a, w, b, mask, 2), (h, weight, bias))


def test_fused_head_with_no_active_dimension_gives_zero_gradients():
    h, weight, bias, mask = make_inputs(seed=2)
    bias = (bias.detach() - 50.0).requires_grad_(True)  # every logit negative -> relu kills everything
    fused = fused_pooling(h, weight, bias, mask, 0)
    assert torch.count_nonzero(fused) == 0
    grads = torch.autograd.grad(loss_of(fused), (h, weight, bias), allow_unused=False)
    assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_fused_head_half_precision_values_are_identical():
    h, weight, bias, mask = make_inputs(seed=3, dtype=torch.float16)
    stock = stock_pooling(nn.functional.linear(h, weight, bias), mask, 2)
    fused = fused_pooling(h, weight, bias, mask, 0)
    assert fused.dtype == torch.float16
    assert torch.equal(fused, stock)


def test_ties_are_the_documented_difference():
    """Two tokens with the same maximum: SpladePooling with chunks splits the gradient, the fused head picks one token."""
    h = torch.zeros(1, 2, H, dtype=torch.float64, requires_grad=True)  # both tokens give logits == bias
    weight = torch.randn(V, H, dtype=torch.float64)
    bias = torch.full((V,), 1.0, dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.long)
    stock = stock_pooling(nn.functional.linear(h, weight, bias), mask, 1)
    (g_stock,) = torch.autograd.grad(stock.sum(), (h,))
    fused = fused_pooling(h, weight, bias, mask, 0)
    (g_fused,) = torch.autograd.grad(fused.sum(), (h,))
    assert torch.equal(fused, stock)
    assert torch.allclose(g_stock.sum(dim=1), g_fused.sum(dim=1))  # the same total gradient ...
    tokens_with_grad = lambda g: int((g.abs().sum(dim=-1) > 0).sum())  # noqa: E731
    assert tokens_with_grad(g_stock) == 2 and tokens_with_grad(g_fused) == 1  # ... on one token instead of both


def test_parse_fused_head():
    assert parse_fused_head({}) is None
    assert parse_fused_head({"splade_fused_head": False}) is None
    assert parse_fused_head({"splade_fused_head": "off"}) is None
    assert parse_fused_head({"splade_fused_head": True}) == ("exact", 0)
    assert parse_fused_head({"splade_fused_head": "exact", "splade_fused_head_chunk_size": 8}) == ("exact", 0)
    assert parse_fused_head({"splade_fused_head": "chunked"}) == ("chunked", 64)
    assert parse_fused_head({"splade_fused_head": "chunked", "splade_fused_head_chunk_size": "16"}) == ("chunked", 16)
    with pytest.raises(SpladeFusedHeadError):
        parse_fused_head({"splade_fused_head": "fast"})
    with pytest.raises(SpladeFusedHeadError):
        parse_fused_head({"splade_fused_head": "chunked", "splade_fused_head_chunk_size": 0})


class FakeMaskedLM(nn.Module):
    """Stands in for a transformers *ForMaskedLM: backbone + decoder, returns an object with .logits."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(H, H)
        self.decoder = nn.Linear(H, V)

    def get_output_embeddings(self):
        return self.decoder

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **_):
        return SimpleNamespace(logits=self.decoder(torch.tanh(self.backbone(inputs_embeds))))


class FakeMLMTransformer(nn.Module):
    """Stands in for the ST MLMTransformer module (auto_model + forward over a features dict)."""

    def __init__(self):
        super().__init__()
        self.auto_model = FakeMaskedLM()

    def forward(self, features):
        out = self.auto_model(inputs_embeds=features["inputs_embeds"], attention_mask=features["attention_mask"])
        features["token_embeddings"] = out.logits
        return features


def features_for(seed: int = 4):
    gen = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[0, 5:] = 0
    return {"inputs_embeds": torch.randn(B, L, H, generator=gen), "attention_mask": mask}


def test_install_patches_only_the_training_forward():
    module = FakeMLMTransformer()
    pooling = SimpleNamespace(pooling_strategy="max")
    reference = FakeMLMTransformer()
    reference.load_state_dict(module.state_dict())

    install_fused_head(module, pooling, mode="exact")
    assert type(module) is FakeMLMTransformer and type(module.auto_model.decoder) is nn.Linear
    assert module.state_dict().keys() == reference.state_dict().keys()

    module.train()
    feats = module(features_for())
    assert feats["token_embeddings"].shape == (B, 1, V)
    assert feats["attention_mask"].shape == (B, 1)
    ref_feats = reference.train()(features_for())
    ref_pooled = stock_pooling(ref_feats["token_embeddings"], ref_feats["attention_mask"], 2)
    fused_pooled = stock_pooling(feats["token_embeddings"], feats["attention_mask"], None)
    assert torch.equal(fused_pooled, ref_pooled)

    module.eval()
    feats = module(features_for())
    assert feats["token_embeddings"].shape == (B, L, V)
    assert feats["attention_mask"].shape == (B, L)
    assert torch.equal(feats["token_embeddings"], reference.eval()(features_for())["token_embeddings"])

    with pytest.raises(SpladeFusedHeadError):
        install_fused_head(module, pooling, mode="exact")
    uninstall_fused_head(module)
    assert "forward" not in module.__dict__ and "forward" not in module.auto_model.decoder.__dict__
    module.train()
    assert module(features_for())["token_embeddings"].shape == (B, L, V)


def test_install_rejects_sum_pooling_and_odd_models():
    with pytest.raises(SpladeFusedHeadError):
        install_fused_head(FakeMLMTransformer(), SimpleNamespace(pooling_strategy="sum"), mode="exact")
    with pytest.raises(SpladeFusedHeadError):
        install_fused_head(nn.Linear(2, 2), SimpleNamespace(pooling_strategy="max"), mode="exact")
    with pytest.raises(SpladeFusedHeadError):
        install_fused_head(FakeMLMTransformer(), SimpleNamespace(pooling_strategy="max"), mode="turbo")


def test_backend_reads_the_fused_head_option():
    from training.backends.sentence_transformers_backend import SentenceTransformersConfigError, _install_splade_fused_head

    model = [FakeMLMTransformer(), SimpleNamespace(pooling_strategy="max")]
    assert _install_splade_fused_head(model, {}) is None
    assert _install_splade_fused_head(model, {"splade_fused_head": "exact"}) == "exact"
    assert "forward" in model[0].__dict__
    with pytest.raises(SentenceTransformersConfigError):
        _install_splade_fused_head(model, {"splade_fused_head": "exact"})  # installed twice
    with pytest.raises(SentenceTransformersConfigError):
        _install_splade_fused_head([FakeMLMTransformer(), model[1]], {"splade_fused_head": "nope"})
