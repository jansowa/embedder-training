"""The combined KL + MNRL SPLADE loss: its value against its two parts, and its wiring in the backend."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")


def _embeddings(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # 3 queries, 1 positive + 2 negatives each, 8 "vocabulary" dims, non-negative like SPLADE
    return [torch.rand(3, 8, generator=g) for _ in range(4)]


def _labels():
    return torch.tensor([[10.0, 7.0, 4.0], [9.0, 8.5, 1.0], [12.0, 3.0, 2.0]])


def test_combined_loss_is_kl_plus_weighted_mnrl():
    from sentence_transformers.sparse_encoder.losses import SparseDistillKLDivLoss, SparseMultipleNegativesRankingLoss

    from training.backends.splade_kl_mnrl import SparseKLPlusMNRLLoss

    model = SimpleNamespace()
    emb, labels = _embeddings(), _labels()
    loss = SparseKLPlusMNRLLoss(model, temperature=2.0, mnrl_weight=0.5, mnrl_scale=1.0)
    kl = SparseDistillKLDivLoss(model, temperature=2.0).compute_loss_from_embeddings(emb, labels)
    mnrl = SparseMultipleNegativesRankingLoss(model, scale=1.0).compute_loss_from_embeddings(emb, labels)

    assert loss.compute_loss_from_embeddings(emb, labels).item() == pytest.approx((kl + 0.5 * mnrl).item())
    assert kl.item() > 0 and mnrl.item() > 0


def test_zero_mnrl_weight_reduces_to_the_kl_loss():
    from sentence_transformers.sparse_encoder.losses import SparseDistillKLDivLoss

    from training.backends.splade_kl_mnrl import SparseKLPlusMNRLLoss

    model = SimpleNamespace()
    emb, labels = _embeddings(1), _labels()
    combined = SparseKLPlusMNRLLoss(model, temperature=3.0, mnrl_weight=0.0).compute_loss_from_embeddings(emb, labels)
    kl = SparseDistillKLDivLoss(model, temperature=3.0).compute_loss_from_embeddings(emb, labels)

    assert combined.item() == pytest.approx(kl.item())


def test_combined_loss_backpropagates_into_the_embeddings():
    from training.backends.splade_kl_mnrl import SparseKLPlusMNRLLoss

    emb = [e.requires_grad_(True) for e in _embeddings(2)]
    loss = SparseKLPlusMNRLLoss(SimpleNamespace(), mnrl_weight=1.0).compute_loss_from_embeddings(emb, _labels())
    loss.backward()

    assert all(e.grad is not None and torch.isfinite(e.grad).all() for e in emb)


def test_combined_loss_rejects_bad_arguments():
    from training.backends.splade_kl_mnrl import SparseKLPlusMNRLLoss

    with pytest.raises(ValueError):
        SparseKLPlusMNRLLoss(SimpleNamespace(), temperature=0.0)
    with pytest.raises(ValueError):
        SparseKLPlusMNRLLoss(SimpleNamespace(), mnrl_weight=-1.0)
    with pytest.raises(ValueError):
        SparseKLPlusMNRLLoss(SimpleNamespace()).compute_loss_from_embeddings(_embeddings(), None)
    with pytest.raises(AttributeError):
        SparseKLPlusMNRLLoss(SimpleNamespace())([], _labels())


def test_loss_name_accepts_the_combined_loss_and_its_aliases():
    from training.backends.sentence_transformers_backend import (
        SPLADE_DISTILLATION_LOSSES,
        SentenceTransformersConfigError,
        _mnrl_weight,
        _splade_base_loss_name,
    )

    for name in ("sparse_kl_plus_mnrl", "kl_plus_mnrl", "kl-mnrl", "sparse_kl_plus_mnrl_loss"):
        assert _splade_base_loss_name({"loss": name}) == "sparse_kl_plus_mnrl"
    assert "sparse_kl_plus_mnrl" in SPLADE_DISTILLATION_LOSSES
    assert _mnrl_weight({}) == 1.0
    assert _mnrl_weight({"mnrl_weight": "0.25"}) == 0.25
    with pytest.raises(SentenceTransformersConfigError):
        _mnrl_weight({"mnrl_weight": -1})
    with pytest.raises(SentenceTransformersConfigError):
        _mnrl_weight({"mnrl_weight": "much"})


def test_splade_training_builds_the_combined_loss_with_scores_labels(monkeypatch, tmp_path):
    from tests.test_splade_distillation import _distillation_request, _fake_sparse_stack
    from training.backends import sentence_transformers_backend

    calls: dict = {}
    monkeypatch.setattr(sentence_transformers_backend, "_load_sparse_sentence_transformers", lambda: _fake_sparse_stack(calls))

    class FakeCombined:
        def __init__(self, model, temperature, mnrl_weight, mnrl_scale, gather_across_devices):
            calls["base_loss"] = ("kl+mnrl", temperature, mnrl_weight, mnrl_scale)
            self.temperature, self.mnrl_weight = temperature, mnrl_weight

    monkeypatch.setattr(sentence_transformers_backend, "_load_sparse_kl_plus_mnrl_loss", lambda: FakeCombined)
    request = _distillation_request(tmp_path, loss="kl_plus_mnrl", kl_temperature=2, mnrl_weight=0.5, scale=4)

    assert sentence_transformers_backend.run_training(request) == 0
    assert calls["base_loss"] == ("kl+mnrl", 2.0, 0.5, 4.0)
    # the scores layout: teacher(query, positive) first, then the negatives, untouched without score_scale
    assert calls["rows"][0]["label"] == [10.0, 7.0, 4.0]
    assert not (tmp_path / "out" / "score_scale.json").exists()
