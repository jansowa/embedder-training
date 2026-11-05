"""SPLADE pooling, MLM-head access and the distillation losses shared by the student builder,
the KD trainer and the fidelity check.

Conventions (identical to sentence-transformers ``SpladePooling(max, relu)`` and to PIRB):
``vector[b, v] = max_l log1p(relu(logits[b, l, v])) * mask[b, l]``. The maximum is computed with the
sparse-gradient decoder of ``training.backends.splade_fused_head`` so that a ``[B, L, V]`` logits
tensor is never materialised for the pooled losses. The token-level losses do need per-token logits;
they are computed on position chunks under activation checkpointing (see ``token_distill_loss``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.backends.splade_fused_head import sparse_max_decoder

TOKEN_LOSSES = ("kl", "mse")
POOLED_LOSSES = ("mse", "kl")


class SpladeHeadError(ValueError):
    """Raised when a model's MLM head is not of a supported shape."""


def split_mlm_head(model: nn.Module) -> tuple[Callable[[torch.Tensor], torch.Tensor], nn.Linear]:
    """``(transform, decoder)`` such that ``model`` computes ``logits = decoder(transform(hidden))``.

    Supports RoBERTa / XLM-R (``lm_head``), BERT (``cls.predictions``), ALBERT (``predictions``) and
    ModernBERT (``head`` + ``decoder``). ``transform`` is the dense + activation + LayerNorm part.
    """
    decoder = model.get_output_embeddings()
    if not isinstance(decoder, nn.Linear):
        raise SpladeHeadError(f"the MLM decoder must be an nn.Linear, got {type(decoder).__name__}.")
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"roberta", "xlm-roberta", "camembert"}:
        head = model.lm_head

        def transform(h: torch.Tensor) -> torch.Tensor:
            return head.layer_norm(F.gelu(head.dense(h)))

    elif model_type == "albert":
        head = model.predictions
        fp32_head = bool(getattr(model.config, "head_layer", False))

        def transform(h: torch.Tensor) -> torch.Tensor:
            if fp32_head and head.dense.weight.dtype == torch.float32:
                # head-layer students keep gelu in its linear regime with a large bias: the shifted values need
                # fp32 (fp16 rounds them at 1/64-1/16, 2-6 % noise per coordinate); see prepare_head_layer
                with torch.autocast(device_type=h.device.type, enabled=False):
                    return head.LayerNorm(head.activation(head.dense(h.float())))
            return head.LayerNorm(head.activation(head.dense(h)))

    elif model_type == "bert":
        head = model.cls.predictions.transform

        def transform(h: torch.Tensor) -> torch.Tensor:
            return head.LayerNorm(head.transform_act_fn(head.dense(h)))

    elif model_type == "modernbert":
        head = model.head

        def transform(h: torch.Tensor) -> torch.Tensor:
            return head.norm(head.act(head.dense(h)))

    else:
        raise SpladeHeadError(f"unsupported MLM model type {model_type!r} for SPLADE distillation.")
    return transform, decoder


def prepare_head_layer(model: nn.Module) -> bool:
    """For a head-layer student (``config.head_layer``): run the extra layer's feed-forward in fp32 (the
    ``gelu(n + b) - n`` cancellation is destroyed by fp16 rounding of the shifted values) and switch its
    dropout off (dropout on the FFN output breaks the cancellation on the dropped coordinates). The patch
    lives on the instance only; a reloaded model needs it again. The affected modules (the layer's FFN, the
    head's dense + LayerNorm) are kept in fp32 even when the rest of the model is fp16. Without this call a
    model loaded in fp16 runs the stock fp16 path, which is what sentence-transformers / PIRB inference sees.
    Returns whether anything was patched."""
    if not getattr(model.config, "head_layer", False) or getattr(model.config, "model_type", "") != "albert":
        return False
    layer = model.base_model.encoder.albert_layer_groups[-1].albert_layers[0]
    for module in (layer.attention.attention_dropout, layer.attention.output_dropout, layer.dropout):
        module.p = 0.0
    head = model.predictions
    for module in (layer.ffn, layer.ffn_output, head.dense, head.LayerNorm):
        module.float()

    def ff_chunk_fp32(attention_output: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=attention_output.device.type, enabled=False):
            x = attention_output.float()
            return layer.ffn_output(layer.activation(layer.ffn(x))).to(attention_output.dtype)

    layer.ff_chunk = ff_chunk_fp32  # instance attribute shadows the bound method
    return True


def encode_hidden(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last hidden states ``[B, L, H]`` of the encoder body of an MLM model."""
    outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs[0]


def pooled_vector(model: nn.Module, hidden: torch.Tensor, attention_mask: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """SPLADE vector ``[B, V]`` from hidden states, through the model's own MLM head, with a sparse gradient."""
    transform, decoder = split_mlm_head(model)
    z = transform(hidden)
    best = sparse_max_decoder(z, decoder.weight, decoder.bias, attention_mask, chunk)
    return torch.log1p(torch.relu(best))


def pooled_vector_dense(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Reference pooling from full logits ``[B, L, V]`` (tests and small checks only)."""
    return torch.amax(torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1).to(logits.dtype), dim=1)


def encode_pooled(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    return pooled_vector(model, encode_hidden(model, input_ids, attention_mask), attention_mask, chunk)


def teacher_topk_targets(logits: torch.Tensor, temperature: float, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(indices [.., k], probabilities [.., k], rest probability [..])`` of the tempered teacher softmax."""
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    top_logp, indices = torch.topk(log_probs, k=min(top_k, logits.shape[-1]), dim=-1)
    probs = top_logp.exp()
    rest = (1.0 - probs.sum(dim=-1)).clamp_min(0.0)
    return indices, probs, rest


def topk_kl(student_logits: torch.Tensor, indices: torch.Tensor, probs: torch.Tensor, rest: torch.Tensor, temperature: float) -> torch.Tensor:
    """Per-position ``KL(teacher || student)`` over the teacher's top-k dimensions plus one bucket for the rest.

    Returns ``[...]`` (one value per position), multiplied by ``temperature ** 2`` as usual for soft targets.
    """
    log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    student_logp = torch.gather(log_probs, -1, indices)
    student_rest = (1.0 - student_logp.exp().sum(dim=-1)).clamp_min(1e-6)
    eps = 1e-12
    kl = (probs * (torch.log(probs + eps) - student_logp)).sum(dim=-1)
    kl = kl + rest * (torch.log(rest + eps) - torch.log(student_rest))
    return kl * (temperature**2)


def token_distill_loss(
    student: nn.Module,
    student_hidden: torch.Tensor,
    teacher: nn.Module,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    kind: str = "kl",
    temperature: float = 2.0,
    top_k: int = 128,
    chunk: int = 64,
) -> torch.Tensor:
    """Mean per-token distillation loss between the teacher's and the student's MLM logits.

    ``kind='kl'``: tempered KL on the teacher's top-k vocabulary entries plus a rest bucket (the
    ranking of the expansion terms). ``kind='mse'``: squared error of the per-token SPLADE
    activations ``log1p(relu(logits))`` summed over the vocabulary. Logits are computed on position
    chunks of ``chunk`` tokens; the student's chunks run under activation checkpointing, so at most
    one ``[B, chunk, V]`` logits tensor of the student is alive at a time.
    """
    if kind not in TOKEN_LOSSES:
        raise ValueError(f"token distillation loss must be one of {TOKEN_LOSSES}, got {kind!r}.")
    transform_s, decoder_s = split_mlm_head(student)
    transform_t, decoder_t = split_mlm_head(teacher)
    seq_len = student_hidden.shape[1]
    mask = attention_mask.to(student_hidden.dtype)
    n_tokens = mask.sum().clamp_min(1.0)
    chunk = seq_len if chunk <= 0 else chunk
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, seq_len, chunk):
        sl = slice(start, start + chunk)
        with torch.no_grad():
            teacher_logits = decoder_t(transform_t(teacher_hidden[:, sl]))
            if kind == "kl":
                targets: tuple[torch.Tensor, ...] = teacher_topk_targets(teacher_logits, temperature, top_k)
            else:
                targets = (torch.log1p(torch.relu(teacher_logits.float())),)
            del teacher_logits
        # every changing value is an argument: under checkpointing the function is re-run in backward
        def chunk_loss(h_chunk: torch.Tensor, mask_chunk: torch.Tensor, *target: torch.Tensor) -> torch.Tensor:
            logits = decoder_s(transform_s(h_chunk))
            if kind == "kl":
                per_position = topk_kl(logits, target[0], target[1], target[2], temperature)
            else:
                per_position = ((torch.log1p(torch.relu(logits.float())) - target[0]) ** 2).sum(dim=-1)
            return (per_position * mask_chunk).sum()

        if student_hidden.requires_grad:
            total = total + checkpoint(chunk_loss, student_hidden[:, sl], mask[:, sl], *targets, use_reentrant=False)
        else:
            total = total + chunk_loss(student_hidden[:, sl], mask[:, sl], *targets)
    return total / n_tokens


def pooled_distill_loss(student_vec: torch.Tensor, teacher_vec: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    """Distance between pooled SPLADE vectors ``[B, V]`` (mean over the batch).

    ``mse``: squared error summed over the vocabulary, divided by the teacher's mean squared norm over the
    batch (a relative error, so it combines with a token loss on a known scale). ``kl``: KL(teacher ||
    student) between the vectors normalised to distributions, plus the squared error of the L1 masses (so
    the student also matches the teacher's overall weight, which ``kl`` alone ignores).
    """
    if kind not in POOLED_LOSSES:
        raise ValueError(f"pooled distillation loss must be one of {POOLED_LOSSES}, got {kind!r}.")
    s = student_vec.float()
    t = teacher_vec.float()
    if kind == "mse":
        return ((s - t) ** 2).sum(dim=-1).mean() / (t**2).sum(dim=-1).mean().clamp_min(1e-6)
    eps = 1e-6
    s_mass = s.sum(dim=-1, keepdim=True) + eps
    t_mass = t.sum(dim=-1, keepdim=True) + eps
    p_t = t / t_mass
    p_s = s / s_mass
    kl = (p_t * (torch.log(p_t + eps) - torch.log(p_s + eps))).sum(dim=-1)
    mass = ((s_mass - t_mass).squeeze(-1) / t_mass.squeeze(-1)) ** 2
    return (kl + mass).mean()


def flops_loss(pooled: torch.Tensor) -> torch.Tensor:
    """FLOPS regulariser of SPLADE: ``sum_v (mean_b w[b, v]) ** 2``."""
    return (pooled.float().mean(dim=0) ** 2).sum()


def active_dims(pooled: torch.Tensor, quantization_factor: int = 100) -> torch.Tensor:
    """Number of non-zero quantised weights per vector, as PIRB indexes them (``round(w * qf) > 0``)."""
    return (torch.round(pooled.float() * quantization_factor) > 0).sum(dim=-1)


def describe_losses(config: dict[str, Any]) -> str:
    parts = []
    if config.get("token_loss"):
        parts.append(f"token {config['token_loss']} x{config.get('token_weight', 1.0)}")
    if config.get("pooled_loss"):
        parts.append(f"pooled {config['pooled_loss']} x{config.get('pooled_weight', 1.0)}")
    if config.get("flops_weight"):
        parts.append(f"flops x{config['flops_weight']}")
    return ", ".join(parts) or "none"
