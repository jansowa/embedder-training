"""Inference-only SPLADE head: max over tokens before the activation, no ``[batch, seq, vocab]`` logits.

Sentence Transformers computes SPLADE as ``MLMTransformer`` (full MLM logits ``[B, L, V]``) followed by
``SpladePooling``: ``max_l log1p(relu(z[b, l, v] * mask[b, l]))``. ``log1p`` and ``relu`` are monotone
non-decreasing (also after rounding), so that equals ``log1p(relu(max_l z[b, l, v] * mask[b, l]))``.
:func:`enable_fast_head` patches one loaded ``SparseEncoder`` so the vocabulary decoder returns only the masked
token maximum ``[B, V]``, computed over slices of ``chunk_tokens`` tokens (``0``: one ``F.linear`` over the
whole sequence, the same call as ``nn.Linear``). The unchanged ``SpladePooling`` then applies ``relu``/``log1p``
on a sequence of length one. Peak memory of the head drops from ``B x L x V`` to ``B x chunk x V``.

What is and is not changed:
  * only the vocabulary decoder (``get_output_embeddings()``) runs differently; the transformer body and the head
    transform (dense, activation, LayerNorm) run as before, and the decoder bias is added before the maximum;
  * the patch lives on the instances (``forward`` attributes), never on the classes: weights, ``state_dict`` and the
    saved files are untouched, other models in the process are unaffected, :func:`disable_fast_head` removes it;
  * inference only: in training mode the original path runs (the trainer has its own sparse-gradient head);
  * exactness: with ``chunk_tokens=0`` the decoder call has the same shapes as the stock one, and the vectors are
    identical bit for bit (soup-v20-dqf-bias015 on the CPU, 144 kd_v1 texts incl. 16 of 512 tokens, fp32 and bf16,
    15.11). Peak head memory is then one ``[B, L, V]`` tensor instead of the stock two. Slices pick other GEMM
    blockings and round differently: fp32 max |diff| 1.2e-6 with the same indexed dims, but bf16 up to 1.6e-2 (one
    bf16 step) with some indexed dims flipped. GPU fp16 is to be measured (release.gpu_bench agreement columns).

Self-contained on purpose (torch only, no imports from this repository), so the file can be copied next to a model.

    from sentence_transformers import SparseEncoder
    from fast_head import enable_fast_head
    model = SparseEncoder("<model>", device="cuda", model_kwargs={"torch_dtype": torch.float16})
    enable_fast_head(model, chunk_tokens=64)
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_CHUNK_TOKENS = 64
_STATE_ATTR = "splade_fast_head"


class FastHeadError(ValueError):
    """The model cannot take the fast head (not an MLMTransformer + SpladePooling(max) model, or already patched)."""


@torch.no_grad()
def masked_max_logits(h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, mask: torch.Tensor,
                      chunk_tokens: int = 0) -> torch.Tensor:
    """``max_l (h @ W^T + b)[b, l, v] * mask[b, l]`` as ``[B, V]``, the SpladePooling semantics (padding -> 0)."""
    seq_len = h.shape[1]
    step = seq_len if chunk_tokens <= 0 else min(int(chunk_tokens), seq_len)
    best = None
    for start in range(0, seq_len, step):
        z = F.linear(h[:, start:start + step], weight, bias)
        # In place: F.linear returns a fresh tensor; the stock SpladePooling multiplies out of place and so holds
        # two [B, L, V] tensors at its peak, this path one (bitwise the same product).
        z.mul_(mask[:, start:start + step].unsqueeze(-1).to(z.dtype))
        m = z.amax(dim=1)
        best = m if best is None else torch.maximum(best, m)
        del z
    return best


def _modules(model: Any) -> tuple[nn.Module, Any]:
    mods = list(model)
    if len(mods) < 2:
        raise FastHeadError("expected a SparseEncoder with an MLMTransformer followed by SpladePooling")
    mlm, pooling = mods[0], mods[1]
    if not hasattr(mlm, "auto_model") or not hasattr(mlm.auto_model, "get_output_embeddings"):
        raise FastHeadError(f"the first module must be an MLMTransformer, got {type(mlm).__name__}")
    # Both activations are monotone non-decreasing, so the maximum commutes with them.
    if getattr(pooling, "pooling_strategy", None) != "max" or getattr(pooling, "activation_function", "relu") not in ("relu", "log1p_relu"):
        raise FastHeadError("the fast head needs SpladePooling(pooling_strategy='max') with a relu or log1p_relu activation")
    return mlm, pooling


def enable_fast_head(model: Any, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> Any:
    """Patch a loaded ``SparseEncoder`` in place (inference only) and return it."""
    mlm, _ = _modules(model)
    auto_model = mlm.auto_model
    decoder = auto_model.get_output_embeddings()
    if not isinstance(decoder, nn.Linear):
        raise FastHeadError(f"the MLM decoder must be an nn.Linear, got {type(decoder).__name__}")
    if "forward" in mlm.__dict__ or "forward" in decoder.__dict__:
        raise FastHeadError("the model already has a patched forward (fast head or the trainer's fused head)")
    state: dict[str, Any] = {"mask": None, "chunk_tokens": int(chunk_tokens)}
    original_mlm_forward = mlm.forward
    original_decoder_forward = decoder.forward

    def decoder_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        mask = state["mask"]
        if mask is None:
            return original_decoder_forward(x)
        if x.dim() != 3 or x.shape[:2] != mask.shape:
            raise FastHeadError(f"decoder input {tuple(x.shape)} does not match the attention mask {tuple(mask.shape)}")
        return masked_max_logits(x, self.weight, self.bias, mask, state["chunk_tokens"])

    def mlm_forward(self: nn.Module, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.training or torch.is_grad_enabled():
            return original_mlm_forward(features)
        mask = features["attention_mask"]
        state["mask"] = mask
        try:
            out = original_mlm_forward(features)
        finally:
            state["mask"] = None
        pooled = out["token_embeddings"]
        if pooled.dim() != 2:
            raise FastHeadError(f"the decoder was bypassed (token_embeddings {tuple(pooled.shape)}); unsupported MLM head")
        out["token_embeddings"] = pooled.unsqueeze(1)
        # One "token" per text; SpladePooling multiplies by this mask, so it must be 1 for every row (rows whose
        # first position is padding, e.g. left padding, still get 1: their maximum already carries the mask).
        out["attention_mask"] = torch.ones_like(mask[:, :1])
        return out

    config = getattr(auto_model, "config", None)
    if getattr(config, "reference_compile", None):
        config.reference_compile = False  # ModernBERT's compiled head would bypass the decoder patch
    decoder.forward = types.MethodType(decoder_forward, decoder)
    mlm.forward = types.MethodType(mlm_forward, mlm)
    setattr(mlm, _STATE_ATTR, state)
    return model


def disable_fast_head(model: Any) -> Any:
    """Remove the patches of :func:`enable_fast_head` (no-op when absent)."""
    mlm, _ = _modules(model)
    if not hasattr(mlm, _STATE_ATTR):
        return model
    mlm.__dict__.pop("forward", None)
    mlm.__dict__.pop(_STATE_ATTR, None)
    decoder = mlm.auto_model.get_output_embeddings()
    if decoder is not None:
        decoder.__dict__.pop("forward", None)
    return model
