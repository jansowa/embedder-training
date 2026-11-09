"""GradCache must preserve full-batch KL and both FLOPS gradients."""

import copy

import pytest


@pytest.mark.parametrize("combined", [False, True])
def test_cached_full_batch_loss_and_accumulated_gradients(combined):
    import torch
    from torch import nn
    from sentence_transformers.sparse_encoder.losses import (
        CachedSpladeLoss, SparseDistillKLDivLoss, SpladeLoss,
    )
    from training.backends.splade_kl_mnrl import SparseKLPlusMNRLLoss

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 7)

        def forward(self, features):
            return {"sentence_embedding": torch.nn.functional.softplus(self.linear(features["input_ids"]))}

    torch.manual_seed(42)
    plain_model = Encoder()
    cached_model = copy.deepcopy(plain_model)

    def ranking(model):
        if combined:
            return SparseKLPlusMNRLLoss(model, temperature=2.0, mnrl_weight=1.0, mnrl_scale=1.0)
        return SparseDistillKLDivLoss(model, temperature=2.0)

    kwargs = dict(document_regularizer_weight=0.003, query_regularizer_weight=0.00015)
    plain = SpladeLoss(plain_model, loss=ranking(plain_model), **kwargs)
    cached = CachedSpladeLoss(cached_model, loss=ranking(cached_model), mini_batch_size=2, **kwargs)
    # The real optimizer accumulates four full batches. Scaling must reach the replay hook.
    for _ in range(4):
        features = [{"input_ids": torch.randn(8, 3)} for _ in range(12)]
        labels = torch.randn(8, 11)
        expected = plain(features, labels)
        actual = cached(features, labels)
        assert expected.keys() == actual.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=2e-6)
        (sum(expected.values()) / 4).backward()
        (sum(actual.values()) / 4).backward()
    for expected, actual in zip(plain_model.parameters(), cached_model.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=2e-6)
