"""SPLADE head with a sparse gradient for training (``sentence_transformers.splade_fused_head``).

The MLM decoder (``nn.Linear`` from hidden size ``H`` to the vocabulary ``V``) followed by
``SpladePooling(max)`` computes ``max_l log1p(relu(z_l))`` over the tokens ``l`` of every
sequence. ``log1p`` and ``relu`` are monotone non-decreasing (also after rounding), so

    max_l log1p(relu(z_l · m_l)) == log1p(relu(max_l z_l · m_l))

bit for bit, where ``m`` is the attention mask (padding positions are multiplied by zero,
exactly as ``SpladePooling`` does). During training the decoder therefore only has to return
the masked token maximum ``best[b, v] = max_l z[b, l, v] · m[b, l]`` (shape ``[B, V]``) and the
unchanged ``SpladePooling`` applies ``relu``/``log1p`` on a sequence of length one. The backward
pass of the maximum is sparse: the incoming gradient is already zero outside the active
dimensions (derivative of ``relu``), and every active ``(b, v)`` touches a single token ``l*``.
That removes the dense ``[B, L, V]`` gradient, the dense ``grad_W = g^T h`` matmul and the
``[B, L, V]`` tensors kept for backward.

Differences to the stock path are limited to ties (two tokens with the same fp16 maximum:
``SpladePooling`` with chunking splits the gradient between chunks, here one token gets it)
and to the summation order (fp32 here). The forward values are identical in ``exact`` mode
(one ``F.linear`` call with the same shapes as ``nn.Linear``); ``chunked`` mode runs the
decoder on token chunks, which may pick other cuBLAS kernels and round differently.

Only the training mode is touched: in ``eval`` mode the original ``forward`` methods run, the
module classes, ``state_dict`` and ``modules.json`` never change (the patch lives on the
instances, not on the classes), so saved models load everywhere as before.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

FUSED_HEAD_MODES = ("exact", "chunked")
DEFAULT_CHUNK_SIZE = 64


class SpladeFusedHeadError(ValueError):
    """Raised when the fused head cannot be installed on a model."""


def _masked_max(z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z * mask.unsqueeze(-1).to(z.dtype)  # SpladePooling semantics: padding -> 0
    return z.max(dim=1)


class SparseMaxDecoder(torch.autograd.Function):
    """``best, l* = max_l (h @ W^T + b)[b, l, v] * mask[b, l]`` with a sparse backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx: Any, h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, mask: torch.Tensor, chunk: int):
        seq_len = h.shape[1]
        if chunk <= 0 or chunk >= seq_len:
            best, idx = _masked_max(F.linear(h, weight, bias), mask)
        else:
            best = idx = None
            for start in range(0, seq_len, chunk):
                stop = start + chunk
                m, i = _masked_max(F.linear(h[:, start:stop], weight, bias), mask[:, start:stop])
                if best is None:
                    best, idx = m, i
                else:
                    upd = m > best
                    best = torch.where(upd, m, best)
                    idx = torch.where(upd, i + start, idx)
        ctx.save_for_backward(h, weight, idx, mask)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return best

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        h, weight, idx, mask = ctx.saved_tensors
        need_h, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        # accumulate in the input precision, but never below fp32 (fp16 activations under autocast)
        acc = torch.promote_types(torch.promote_types(h.dtype, weight.dtype), torch.float32)
        bi, vi = grad.nonzero(as_tuple=True)  # active dimensions only (hundreds per sequence)
        li = idx[bi, vi]
        coef = grad[bi, vi].to(acc) * mask[bi, li].to(acc)  # padding never receives a gradient
        grad_h = grad_w = grad_b = None
        if need_h:
            grad_h = torch.zeros(h.shape, dtype=acc, device=h.device)
            grad_h.index_put_((bi, li), coef.unsqueeze(1) * weight[vi].to(acc), accumulate=True)
            grad_h = grad_h.to(h.dtype)
        if need_w:
            grad_w = torch.zeros(weight.shape, dtype=acc, device=weight.device)
            grad_w.index_add_(0, vi, coef.unsqueeze(1) * h[bi, li].to(acc))
            grad_w = grad_w.to(weight.dtype)
        if need_b and ctx.has_bias:
            grad_b = torch.zeros(weight.shape[0], dtype=acc, device=weight.device)
            grad_b.index_add_(0, vi, coef)
            grad_b = grad_b.to(ctx.bias_dtype)
        return grad_h, grad_w, grad_b, None, None


def sparse_max_decoder(
    h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, mask: torch.Tensor, chunk: int = 0
) -> torch.Tensor:
    """Masked token maximum of the decoder logits, ``[B, V]``; see :class:`SparseMaxDecoder`."""
    return SparseMaxDecoder.apply(h, weight, bias, mask, int(chunk))


def parse_fused_head(backend_config: dict[str, Any]) -> tuple[str, int] | None:
    """``(mode, chunk_size)`` from ``splade_fused_head`` / ``splade_fused_head_chunk_size``, or None when off."""
    raw = backend_config.get("splade_fused_head", False)
    if raw is None or raw is False:
        return None
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"", "false", "no", "off", "0", "none"}:
            return None
        if value in {"true", "yes", "on", "1"}:
            value = "exact"
    elif raw is True:
        value = "exact"
    else:
        raise SpladeFusedHeadError("'sentence_transformers.splade_fused_head' must be exact, chunked or false.")
    if value not in FUSED_HEAD_MODES:
        raise SpladeFusedHeadError(
            f"'sentence_transformers.splade_fused_head' must be one of {', '.join(FUSED_HEAD_MODES)} or false, got {raw!r}."
        )
    chunk = backend_config.get("splade_fused_head_chunk_size", DEFAULT_CHUNK_SIZE)
    try:
        chunk = int(chunk)
    except (TypeError, ValueError) as exc:
        raise SpladeFusedHeadError("'sentence_transformers.splade_fused_head_chunk_size' must be an integer.") from exc
    if value == "chunked" and chunk <= 0:
        raise SpladeFusedHeadError("'sentence_transformers.splade_fused_head_chunk_size' must be positive for chunked mode.")
    return value, (chunk if value == "chunked" else 0)


def install_fused_head(mlm_module: nn.Module, pooling: Any, *, mode: str = "exact", chunk_size: int = 0) -> None:
    """Patch the ``forward`` of ``mlm_module`` (an ST ``MLMTransformer``) and of its MLM decoder.

    In training mode the decoder returns the masked token maximum ``[B, V]`` and the module hands
    ``token_embeddings`` of shape ``[B, 1, V]`` with a ``[B, 1]`` attention mask to ``pooling``.
    """
    if mode not in FUSED_HEAD_MODES:
        raise SpladeFusedHeadError(f"fused head mode must be one of {', '.join(FUSED_HEAD_MODES)}, got {mode!r}.")
    if getattr(pooling, "pooling_strategy", "max") != "max":
        raise SpladeFusedHeadError("the fused SPLADE head needs SpladePooling(pooling_strategy='max').")
    auto_model = getattr(mlm_module, "auto_model", None)
    if auto_model is None or not hasattr(auto_model, "get_output_embeddings"):
        raise SpladeFusedHeadError("the fused SPLADE head needs an MLMTransformer with a transformers auto_model.")
    decoder = auto_model.get_output_embeddings()
    if not isinstance(decoder, nn.Linear):
        raise SpladeFusedHeadError(f"the MLM decoder must be an nn.Linear, got {type(decoder).__name__}.")
    if "forward" in mlm_module.__dict__ or "forward" in decoder.__dict__:
        raise SpladeFusedHeadError("the fused SPLADE head is already installed on this model.")
    chunk = int(chunk_size) if mode == "chunked" else 0
    state: dict[str, Any] = {"mask": None, "mode": mode, "chunk": chunk}
    original_module_forward = mlm_module.forward
    original_decoder_forward = decoder.forward

    def decoder_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        mask = state["mask"]
        if mask is None or not self.training:
            return original_decoder_forward(x)
        if x.dim() != 3 or x.shape[:2] != mask.shape:
            raise SpladeFusedHeadError(
                f"the fused SPLADE head expects decoder input [B, L, H] matching the mask {tuple(mask.shape)}, got {tuple(x.shape)}."
            )
        return sparse_max_decoder(x, self.weight, self.bias, mask, state["chunk"])

    def module_forward(self: nn.Module, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.training:
            return original_module_forward(features)
        attention_mask = features["attention_mask"]
        state["mask"] = attention_mask
        try:
            output = original_module_forward(features)
        finally:
            state["mask"] = None
        pooled = output["token_embeddings"]
        if pooled.dim() != 2:
            raise SpladeFusedHeadError(
                f"the fused SPLADE decoder was bypassed (token_embeddings {tuple(pooled.shape)}); the model's MLM head is not supported."
            )
        output["token_embeddings"] = pooled.unsqueeze(1)
        output["attention_mask"] = attention_mask[:, :1]  # the first token is never padding
        return output

    config = getattr(auto_model, "config", None)
    if getattr(config, "reference_compile", None):
        # ModernBERT compiles head + decoder into one function when reference_compile is set, which
        # would bypass the instance patch below; the eager path is equivalent.
        config.reference_compile = False
    decoder.forward = types.MethodType(decoder_forward, decoder)
    mlm_module.forward = types.MethodType(module_forward, mlm_module)
    mlm_module.splade_fused_head = state  # type: ignore[attr-defined]


def uninstall_fused_head(mlm_module: nn.Module) -> None:
    """Remove the instance patches installed by :func:`install_fused_head` (no-op when absent)."""
    mlm_module.__dict__.pop("forward", None)
    mlm_module.__dict__.pop("splade_fused_head", None)
    auto_model = getattr(mlm_module, "auto_model", None)
    if auto_model is not None and hasattr(auto_model, "get_output_embeddings"):
        decoder = auto_model.get_output_embeddings()
        if decoder is not None:
            decoder.__dict__.pop("forward", None)


def describe_fused_head(spec: tuple[str, int] | None) -> str:
    if spec is None:
        return "off"
    mode, chunk = spec
    return f"{mode} (chunk {chunk} tokens)" if mode == "chunked" else mode
