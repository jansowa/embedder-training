"""KL distillation plus in-batch contrastive ranking for SPLADE training.

``SparseKLPlusMNRLLoss`` combines, on one set of embeddings, the listwise KL divergence from
teacher scores (``SparseDistillKLDivLoss``: the student's softmax over the row's positive and
negatives matches the teacher's) with the multiple-negatives ranking loss
(``SparseMultipleNegativesRankingLoss``: the row's positive against the row's negatives and every
passage of the other rows in the batch). The KL term carries the teacher's judgement and needs no
label beyond the teacher scores; the MNRL term adds the hard label of the row's positive and the
in-batch negatives, which the KL term never sees.

    loss = kl + mnrl_weight * mnrl

The class is used inside ``SpladeLoss`` (or ``CachedSpladeLoss``) exactly like the two losses it
wraps, so the FLOPS regularisers apply unchanged. Labels are the teacher scores in the KL layout:
``[teacher(query, positive), teacher(query, negative_1), ...]`` per row.

Imports of sentence-transformers stay inside the constructor so that importing this module (and
``training.train``) does not import the backend.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor, nn


class SparseKLPlusMNRLLoss(nn.Module):
    def __init__(
        self,
        model: Any,
        *,
        temperature: float = 2.0,
        mnrl_weight: float = 1.0,
        mnrl_scale: float = 1.0,
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__()
        from sentence_transformers.sparse_encoder.losses import (
            SparseDistillKLDivLoss,
            SparseMultipleNegativesRankingLoss,
        )

        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if mnrl_weight < 0:
            raise ValueError("mnrl_weight must be non-negative.")
        self.model = model
        self.temperature = float(temperature)
        self.mnrl_weight = float(mnrl_weight)
        self.kl = SparseDistillKLDivLoss(model, temperature=self.temperature)
        self.mnrl = SparseMultipleNegativesRankingLoss(
            model, scale=float(mnrl_scale), gather_across_devices=gather_across_devices
        )

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        if labels is None:
            raise ValueError("SparseKLPlusMNRLLoss needs the teacher scores as labels (KL layout).")
        if len(embeddings) < 2:
            raise ValueError(f"Expected at least 2 embeddings (query, positive, ...), got {len(embeddings)}.")
        kl = self.kl.compute_loss_from_embeddings(embeddings, labels)
        if self.mnrl_weight == 0:
            return kl
        mnrl = self.mnrl.compute_loss_from_embeddings(embeddings, labels)
        return kl + self.mnrl_weight * mnrl

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError("SparseKLPlusMNRLLoss should not be used alone. Use it with SpladeLoss or CachedSpladeLoss.")

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "mnrl_weight": self.mnrl_weight,
            "mnrl_scale": float(self.mnrl.scale),
        }

    @property
    def citation(self) -> str:
        return ""


__all__ = ["SparseKLPlusMNRLLoss"]
