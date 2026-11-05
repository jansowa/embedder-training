#!/usr/bin/env python
"""Build a smaller SPLADE student from a RoBERTa SPLADE teacher, with the teacher's tokenizer.

Surgery, in this order, every step optional:

1. keep a subset of the transformer layers (``--keep-layers 0,1,2,5``);
2. prune the feed-forward width of every kept layer to ``--intermediate-size`` neurons, ranked by
   importance (``weight``: ||W_in[j]|| * ||W_out[:, j]||, or ``activation``: mean |gelu(W_in h + b)[j]|
   * ||W_out[:, j]|| over sample texts);
3. factorise the vocabulary embedding / MLM decoder to rank ``--embedding-rank`` and emit an ALBERT
   model (``AlbertForMaskedLM`` with ``embedding_size = rank``, one group per layer, i.e. no weight
   sharing): the same tokenizer, the same output dimensions, loadable by transformers,
   sentence-transformers and PIRB without custom code. The rank space is an orthonormal projection of
   the teacher's hidden space (top principal subspace of the centred embeddings), which makes the
   ALBERT embedding path a near-copy of the teacher and the conversion exact at full rank. The head
   is the approximation: ALBERT applies the gelu and LayerNorm in the rank space. ``--head-layer``
   (recommended) moves the teacher's head transform into one extra full-width layer and leaves the
   ALBERT head a linear low-rank decoder fitted to the teacher's pre-decoder vectors on the sample
   texts (``--calibration-texts``), untied from the input vocabulary; without it the head is re-fitted
   on those texts (``--calibration-steps``, ``--untie-head`` for a separate decoder) and the KD stage
   has to repair the rest.

``--fresh --hidden-size 384`` builds a randomly initialised ALBERT body of that width instead (the
"new architecture, inherited vocabulary" variant); only the embedding / decoder come from the teacher.

The output directory is a complete sentence-transformers SparseEncoder folder (tokenizer, modules.json,
1_SpladePooling copied from the teacher) plus ``student.json`` with the spec, parameter counts and an
estimate of multiply-accumulates per token.

Usage:
  .venv/bin/python -m distill.student --teacher models/soup-v9b --out models/student-b4 \
      --keep-layers 0,1,2,5 --intermediate-size 1536 [--embedding-rank 256] \
      [--calibration-texts corpus.jsonl --calibration-sample 128] [--importance activation]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "student.json",
)
AUX_DIRS = ("1_SpladePooling",)
IMPORTANCE_KINDS = ("weight", "activation")


class StudentSpecError(ValueError):
    """Raised for an inconsistent student specification."""


@dataclass
class StudentSpec:
    keep_layers: list[int]
    intermediate_size: int | None = None
    embedding_rank: int | None = None
    hidden_size: int | None = None
    fresh: bool = False
    importance: str = "weight"
    calibrate: bool = True
    calibration_steps: int = 400
    untie_head: bool = False
    head_layer: bool = False
    token_weights: torch.Tensor | None = None  # per-token weights for the input subspace (corpus counts)
    seed: int = 42

    def validate(self, teacher_layers: int, teacher_hidden: int, teacher_intermediate: int, vocab_size: int) -> None:
        if not self.keep_layers:
            raise StudentSpecError("keep_layers must name at least one layer.")
        if len(set(self.keep_layers)) != len(self.keep_layers):
            raise StudentSpecError("keep_layers must not repeat a layer.")
        if any(i < 0 or i >= teacher_layers for i in self.keep_layers):
            raise StudentSpecError(f"keep_layers must be within 0..{teacher_layers - 1}, got {self.keep_layers}.")
        if self.keep_layers != sorted(self.keep_layers):
            raise StudentSpecError("keep_layers must be increasing (the student keeps the teacher's order).")
        if self.intermediate_size is not None and not 0 < self.intermediate_size <= teacher_intermediate:
            raise StudentSpecError(f"intermediate_size must be within 1..{teacher_intermediate}.")
        if self.embedding_rank is not None and not 0 < self.embedding_rank <= min(vocab_size, teacher_hidden):
            raise StudentSpecError(f"embedding_rank must be within 1..{min(vocab_size, teacher_hidden)}.")
        if self.hidden_size is not None and self.hidden_size != teacher_hidden and not self.fresh:
            raise StudentSpecError("a hidden_size different from the teacher's needs --fresh (layers cannot be copied).")
        if self.fresh and self.embedding_rank is None:
            raise StudentSpecError("--fresh builds an ALBERT body and needs --embedding-rank.")
        if (self.untie_head or self.head_layer) and self.embedding_rank is None:
            raise StudentSpecError("--untie-head / --head-layer apply to the ALBERT output and need --embedding-rank.")
        if self.head_layer and self.fresh:
            raise StudentSpecError("--head-layer copies the teacher's head transform and cannot be combined with --fresh.")
        if self.importance not in IMPORTANCE_KINDS:
            raise StudentSpecError(f"importance must be one of {IMPORTANCE_KINDS}.")
        if self.token_weights is not None and int(self.token_weights.numel()) != vocab_size:
            raise StudentSpecError(f"token_weights must have one entry per token ({vocab_size}), got {int(self.token_weights.numel())}.")


@dataclass
class BuildReport:
    teacher: str
    spec: dict[str, Any]
    architecture: str = ""
    params: dict[str, int] = field(default_factory=dict)
    teacher_params: dict[str, int] = field(default_factory=dict)
    macs_per_token: dict[str, float] = field(default_factory=dict)
    ffn_importance_kept: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


# --------------------------------------------------------------------------------------------- helpers


def parse_layers(raw: str) -> list[int]:
    try:
        return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]
    except ValueError as exc:
        raise StudentSpecError(f"keep-layers must be a comma-separated list of integers, got {raw!r}.") from exc


def count_params(model: nn.Module) -> dict[str, int]:
    """Parameter counts: total, vocabulary embedding, body (encoder layers), head (without the tied decoder)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    embeddings = model.get_input_embeddings().weight.numel()
    body = sum(p.numel() for name, p in model.named_parameters() if ".layer." in name or "albert_layer_groups" in name)
    return {"total": total, "vocabulary": embeddings, "body": body, "other": total - embeddings - body}


def macs_per_token(config: Any) -> dict[str, float]:
    """Multiply-accumulates per token, without the sequence-dependent attention scores."""
    hidden = int(config.hidden_size)
    inter = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    vocab = int(config.vocab_size)
    rank = int(getattr(config, "embedding_size", hidden))
    layer = 4 * hidden * hidden + 2 * hidden * inter
    mapping = rank * hidden if rank != hidden else 0
    head_dense = hidden * rank
    decoder = rank * vocab
    return {
        "body": float(layers * layer),
        "embedding_mapping": float(mapping),
        "head_transform": float(head_dense),
        "decoder": float(decoder),
        "total": float(layers * layer + mapping + head_dense + decoder),
    }


def ffn_importance_weight(layer: nn.Module) -> torch.Tensor:
    w_in = layer.intermediate.dense.weight  # [I, H]
    w_out = layer.output.dense.weight  # [H, I]
    return w_in.norm(dim=1) * w_out.norm(dim=0)


@torch.no_grad()
def ffn_importance_activation(model: nn.Module, batches: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
    """Mean |activation| of every FFN neuron (after GELU) times ||W_out[:, j]||, per layer, over the batches."""
    layers = list(model.base_model.encoder.layer)
    sums = [torch.zeros(layer.intermediate.dense.out_features, dtype=torch.float64) for layer in layers]
    counts = [0.0]
    hooks = []
    current_mask: dict[str, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            mask = current_mask["mask"].unsqueeze(-1).to(output.dtype)
            sums[index] += (output.abs() * mask).sum(dim=(0, 1)).double().cpu()

        return hook

    for index, layer in enumerate(layers):
        hooks.append(layer.intermediate.register_forward_hook(make_hook(index)))
    try:
        for batch in batches:
            current_mask["mask"] = batch["attention_mask"]
            counts[0] += float(batch["attention_mask"].sum())
            model.base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    finally:
        for hook in hooks:
            hook.remove()
    result = []
    for layer, total in zip(layers, sums):
        mean_act = (total / max(counts[0], 1.0)).float()
        result.append(mean_act * layer.output.dense.weight.norm(dim=0).cpu())
    return result


# ---------------------------------------------------------------------------------------- surgery


def prune_roberta(teacher: nn.Module, spec: StudentSpec, importance: list[torch.Tensor] | None = None) -> tuple[nn.Module, dict[str, float]]:
    """A ``RobertaForMaskedLM`` with the kept layers and (optionally) pruned FFN widths; copies of the teacher's weights."""
    from transformers import RobertaForMaskedLM

    config = teacher.config.__class__.from_dict(teacher.config.to_dict())
    config.num_hidden_layers = len(spec.keep_layers)
    if spec.intermediate_size is not None:
        config.intermediate_size = spec.intermediate_size
    student = RobertaForMaskedLM(config)
    student.roberta.embeddings.load_state_dict(teacher.roberta.embeddings.state_dict())
    kept_fraction: dict[str, float] = {}
    for new_index, old_index in enumerate(spec.keep_layers):
        src = teacher.roberta.encoder.layer[old_index]
        dst = student.roberta.encoder.layer[new_index]
        dst.attention.load_state_dict(src.attention.state_dict())
        scores = importance[old_index] if importance is not None else ffn_importance_weight(src)
        keep = scores.shape[0] if spec.intermediate_size is None else spec.intermediate_size
        idx = torch.topk(scores, k=keep).indices.sort().values
        kept_fraction[str(old_index)] = float(scores[idx].sum() / scores.sum().clamp_min(1e-12))
        with torch.no_grad():
            dst.intermediate.dense.weight.copy_(src.intermediate.dense.weight[idx])
            dst.intermediate.dense.bias.copy_(src.intermediate.dense.bias[idx])
            dst.output.dense.weight.copy_(src.output.dense.weight[:, idx])
            dst.output.dense.bias.copy_(src.output.dense.bias)
            dst.output.LayerNorm.load_state_dict(src.output.LayerNorm.state_dict())
    student.lm_head.load_state_dict(teacher.lm_head.state_dict())
    student.tie_weights()
    student.eval()  # the calibration passes below must not see dropout
    return student, kept_fraction


def embedding_subspace(
    word: torch.Tensor, positions: torch.Tensor | None, rank: int, row_weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, float]:
    """Orthonormal basis ``Q [H, r]`` of the top-``r`` principal subspace of the *centred* token (and position)
    embeddings. LayerNorm centres every vector, so the all-ones direction never matters; position rows are
    weighted so that their total energy counts like the tokens' (every token gets one of each). ``row_weights``
    (one per token, e.g. corpus frequencies) weights the token rows by their square roots, so that the subspace
    serves the tokens that actually occur. Returns ``Q`` and the fraction of the (weighted) centred energy it
    keeps. ``rank == H`` gives the identity, i.e. an exact conversion.

    Why a projection and not a balanced SVD ``E ~ A B``: the ALBERT LayerNorm acts in the rank space, and it
    equals the teacher's LayerNorm (up to the constant ``sqrt(H / r)`` and a ``1 / sqrt(r)`` mean term) only when
    the rank-space coordinates are an orthonormal projection of the hidden-space vector.
    """
    hidden = int(word.shape[1])
    if rank >= hidden:
        return torch.eye(hidden), 1.0
    w = word.float() - word.float().mean(dim=-1, keepdim=True)
    total = float(word.shape[0])
    if row_weights is not None:
        weights = row_weights.float().clamp_min(0.0)
        weights = weights * (total / weights.sum())  # mean weight 1: same total energy as unweighted
        w = w * weights.sqrt()[:, None]
    rows = [w]
    if positions is not None and positions.shape[0] > 0:
        p = positions.float() - positions.float().mean(dim=-1, keepdim=True)
        rows.append(p * (total / positions.shape[0]) ** 0.5)
    _, s, vh = torch.linalg.svd(torch.cat(rows), full_matrices=False)
    energy = float((s[:rank] ** 2).sum() / (s**2).sum())
    return vh[:rank].T.contiguous(), energy


LINEAR_MARGIN = 6.0  # gelu(x) == x to 1e-8 for x > 6: a bias of (margin - min x) keeps gelu in its linear regime
HEAD_RIDGE = 0.1  # ridge on the second moment of z before whitening (fraction of the mean eigenvalue)


def _linear_bias(values: torch.Tensor) -> float:
    return max(10.0, LINEAR_MARGIN - float(values.min()))


def to_albert(
    pruned: nn.Module, spec: StudentSpec, calibration_batches: list[dict[str, torch.Tensor]] | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    """An ``AlbertForMaskedLM`` with the pruned RoBERTa's layers and a rank-``r`` vocabulary; see the module docstring."""
    from transformers import AlbertConfig, AlbertForMaskedLM

    src_config = pruned.config
    rank = int(spec.embedding_rank)
    hidden = int(spec.hidden_size or src_config.hidden_size)
    heads = int(src_config.num_attention_heads) if hidden == src_config.hidden_size else max(1, hidden // 64)
    offset = int(src_config.pad_token_id) + 1  # RoBERTa position ids start after the padding index
    n_positions = int(src_config.max_position_embeddings) - offset
    head_layer = bool(spec.head_layer and not spec.fresh)
    if head_layer and not calibration_batches:
        raise StudentSpecError("--head-layer needs calibration texts (the decoder subspace is fitted to real tokens).")
    layers = int(src_config.num_hidden_layers) + (1 if head_layer else 0)
    intermediate = max(int(src_config.intermediate_size), 2 * hidden) if head_layer else int(src_config.intermediate_size)
    config = AlbertConfig(
        vocab_size=src_config.vocab_size,
        embedding_size=rank,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_hidden_groups=layers,
        inner_group_num=1,
        num_attention_heads=heads,
        intermediate_size=intermediate,
        hidden_act="gelu",
        hidden_dropout_prob=float(src_config.hidden_dropout_prob),
        attention_probs_dropout_prob=float(src_config.attention_probs_dropout_prob),
        classifier_dropout_prob=0.0,
        max_position_embeddings=n_positions,
        type_vocab_size=1,
        layer_norm_eps=float(src_config.layer_norm_eps),
        pad_token_id=src_config.pad_token_id,
        bos_token_id=src_config.bos_token_id,
        eos_token_id=src_config.eos_token_id,
        tie_word_embeddings=not (spec.untie_head or head_layer),
        head_layer=head_layer,  # custom marker saved in config.json; distill.splade.prepare_head_layer reads it
    )
    torch.manual_seed(spec.seed)
    student = AlbertForMaskedLM(config)
    report: dict[str, Any] = {"rank": rank, "hidden_size": hidden, "fresh": spec.fresh, "untie_head": not config.tie_word_embeddings, "head_layer": head_layer}

    emb_t = pruned.roberta.embeddings
    emb_s = student.albert.embeddings
    with torch.no_grad():
        e = emb_t.word_embeddings.weight.float()
        positions = emb_t.position_embeddings.weight[offset : offset + n_positions].float()
        q, energy = embedding_subspace(e, positions, rank, spec.token_weights)
        report["embedding_subspace_energy"] = energy
        report["embedding_subspace_weighted"] = spec.token_weights is not None
        emb_s.word_embeddings.weight.copy_(e @ q)
        emb_s.position_embeddings.weight.copy_(positions @ q)
        emb_s.token_type_embeddings.weight.copy_(emb_t.token_type_embeddings.weight.float() @ q)
        student.predictions.bias.copy_(pruned.lm_head.bias)
        if not config.tie_word_embeddings:
            student.predictions.decoder.weight.copy_(emb_s.word_embeddings.weight)
    student.tie_weights()

    if not spec.fresh:
        _copy_layers_to_albert(pruned, student)
        _init_albert_embedding_mapping(pruned, student, q)
        if head_layer:
            report.update(_build_head_layer(pruned, student, calibration_batches))
            report.update(_init_linear_head(pruned, student, calibration_batches, rank))
        else:
            report["head_subspace_energy"] = _init_albert_head(pruned, student, q, not config.tie_word_embeddings)
            if spec.calibrate and calibration_batches:
                report["joint_calibration"] = _calibrate_albert_joint(pruned, student, calibration_batches, spec.calibration_steps)
    return student, report


def _copy_linear(dst: nn.Linear, src: nn.Linear) -> None:
    """Copy ``src`` into the top-left block of ``dst`` (``dst`` may be wider); the rest is zero."""
    with torch.no_grad():
        dst.weight.zero_()
        dst.bias.zero_()
        o, i = src.weight.shape
        dst.weight[:o, :i].copy_(src.weight)
        dst.bias[:o].copy_(src.bias)


def _copy_layers_to_albert(pruned: nn.Module, student: nn.Module) -> None:
    for index, src in enumerate(pruned.roberta.encoder.layer):
        dst = student.albert.encoder.albert_layer_groups[index].albert_layers[0]
        with torch.no_grad():
            dst.attention.query.load_state_dict(src.attention.self.query.state_dict())
            dst.attention.key.load_state_dict(src.attention.self.key.state_dict())
            dst.attention.value.load_state_dict(src.attention.self.value.state_dict())
            dst.attention.dense.load_state_dict(src.attention.output.dense.state_dict())
            dst.attention.LayerNorm.load_state_dict(src.attention.output.LayerNorm.state_dict())
            _copy_linear(dst.ffn, src.intermediate.dense)
            _copy_linear(dst.ffn_output, src.output.dense)
            dst.full_layer_layer_norm.load_state_dict(src.output.LayerNorm.state_dict())


@torch.no_grad()
def _build_head_layer(pruned: nn.Module, student: nn.Module, batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    """The teacher's head transform ``z = LN_h(gelu(W h + b))`` as one extra ALBERT layer at full width.

    The last copied layer emits its normalised output ``n`` (its LayerNorm affine ``gamma_p, beta_p`` is folded
    into ``W``). The extra layer: attention contributes nothing (value and output projections zero) and its
    LayerNorm leaves ``n`` as it is; the FFN computes ``gelu(W' n + b') - n`` with ``2H`` neurons (the second
    half is the identity pushed through gelu's linear regime by a bias above ``-min n`` measured on the
    calibration texts), so the residual sum is ``gelu(W' n + b')`` and the final LayerNorm with the head's
    affine yields ``z`` exactly in fp32 (fp16 rounds the shifted values at ~1e-2, acceptable for an init).
    """
    hidden = int(pruned.config.hidden_size)
    last = student.albert.encoder.albert_layer_groups[len(pruned.roberta.encoder.layer) - 1].albert_layers[0]
    gamma_p, beta_p = last.full_layer_layer_norm.weight.clone(), last.full_layer_layer_norm.bias.clone()
    ns = []
    for batch in batches:
        h = pruned.roberta(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        ns.append(((h - beta_p) / gamma_p)[batch["attention_mask"].bool()])
    n = torch.cat(ns)
    big = _linear_bias(n)
    last.full_layer_layer_norm.weight.fill_(1.0)
    last.full_layer_layer_norm.bias.zero_()

    head_t = pruned.lm_head
    layer = student.albert.encoder.albert_layer_groups[-1].albert_layers[0]
    for lin in (layer.attention.query, layer.attention.key, layer.attention.value, layer.attention.dense):
        lin.weight.zero_()
        lin.bias.zero_()
    layer.attention.LayerNorm.weight.fill_(1.0)
    layer.attention.LayerNorm.bias.zero_()
    w = head_t.dense.weight.float() * gamma_p[None, :]
    b = head_t.dense.weight.float() @ beta_p + head_t.dense.bias.float()
    layer.ffn.weight.zero_()
    layer.ffn.bias.zero_()
    layer.ffn.weight[:hidden].copy_(w)
    layer.ffn.bias[:hidden].copy_(b)
    layer.ffn.weight[hidden : 2 * hidden].copy_(torch.eye(hidden))
    layer.ffn.bias[hidden : 2 * hidden].fill_(big)
    layer.ffn_output.weight.zero_()
    layer.ffn_output.weight[:, :hidden].copy_(torch.eye(hidden))
    layer.ffn_output.weight[:, hidden : 2 * hidden].copy_(-torch.eye(hidden))
    layer.ffn_output.bias.fill_(big)
    layer.full_layer_layer_norm.weight.copy_(head_t.layer_norm.weight)
    layer.full_layer_layer_norm.bias.copy_(head_t.layer_norm.bias)
    return {"head_layer_bias": big, "head_layer_min_n": float(n.min()), "head_layer_max_n": float(n.max())}


@torch.no_grad()
def _init_linear_head(pruned: nn.Module, student: nn.Module, batches: list[dict[str, torch.Tensor]], rank: int) -> dict[str, float]:
    """With the head transform in the body, the ALBERT head only has to be ``E z ~ A_out (P z)``: the best rank-``r``
    decoder for the *distribution* of the teacher's pre-decoder vectors ``z`` on the calibration texts
    (``E C^1/2 = U S V^T`` with ``C`` the second moment of ``z``; ``P = V_r^T C^-1/2``, ``A_out = E C^1/2 V_r``).
    The whitening is ridge-regularised (``HEAD_RIDGE`` x mean eigenvalue), otherwise near-null directions of
    ``C`` blow up the rank coordinates. The head's gelu is pushed into its linear regime by a bias above
    ``-min(P z)`` and its LayerNorm scaled to the mean per-token norm, which leaves a per-token normalisation
    as the only approximation besides the rank."""
    from distill.splade import split_mlm_head

    transform_t, _ = split_mlm_head(pruned)
    zs = []
    for batch in batches:
        h = pruned.roberta(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        zs.append(transform_t(h)[batch["attention_mask"].bool()])
    z = torch.cat(zs).float()
    e = pruned.roberta.embeddings.word_embeddings.weight.float()
    cov = z.T @ z / z.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(0.0) + HEAD_RIDGE * evals.mean()
    sqrt_cov = evecs @ torch.diag(evals.sqrt()) @ evecs.T
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    _, s, vh = torch.linalg.svd(e @ sqrt_cov, full_matrices=False)
    v = vh[:rank].T  # [H, r]
    projector = v.T @ inv_sqrt  # [r, H]
    w = z @ projector.T
    sigma = w.std(dim=-1, unbiased=False)
    head_s = student.predictions
    head_s.dense.weight.copy_(projector)
    big = _linear_bias(w)
    head_s.dense.bias.fill_(big)
    head_s.LayerNorm.weight.fill_(float(sigma.mean()))
    head_s.LayerNorm.bias.zero_()
    head_s.decoder.weight.copy_((e @ sqrt_cov) @ v)
    head_s.bias.copy_(pruned.lm_head.bias)
    return {
        "head_subspace_energy": float((s[:rank] ** 2).sum() / (s**2).sum()),
        "head_token_scale_cv": float(sigma.std() / sigma.mean()),
        "head_bias": big,
        "head_min_w": float(w.min()),
        "head_max_w": float(w.max()),
        "head_tokens": int(z.shape[0]),
    }


@torch.no_grad()
def _init_albert_embedding_mapping(pruned: nn.Module, student: nn.Module, q: torch.Tensor) -> None:
    """Rank-space LayerNorm without affine, ``mapping_in = diag(gamma) Q sqrt(H / r)`` and bias ``beta``: for a
    token vector ``x`` the body then sees ``gamma * x_S sqrt(H) / ||x_S|| + beta`` with ``x_S`` the projection
    of ``x`` onto the subspace, which is the teacher's ``LN(x)`` when ``x`` lies in the subspace."""
    emb_t = pruned.roberta.embeddings
    emb_s = student.albert.embeddings
    hidden, rank = q.shape
    emb_s.LayerNorm.weight.fill_(1.0)
    emb_s.LayerNorm.bias.zero_()
    mapping = student.albert.encoder.embedding_hidden_mapping_in
    mapping.weight.copy_(emb_t.LayerNorm.weight[:, None] * q * (hidden / rank) ** 0.5)
    mapping.bias.copy_(emb_t.LayerNorm.bias)


@torch.no_grad()
def _init_albert_head(pruned: nn.Module, student: nn.Module, q: torch.Tensor, untied: bool) -> float:
    """Head ``decoder(LN_r(gelu(dense(h))))``: ``dense = Q^T W``, the rank-space LayerNorm scaled by ``sqrt(H / r)``.
    Tied: the decoder is the input vocabulary ``E Q``. Untied: the decoder gets its own subspace ``Q_h`` of
    ``E diag(gamma_h)`` (the head LayerNorm gain folded in) and the bias absorbs ``E beta_h``. The gelu is applied
    in the rank space either way, which the teacher does not do; exact only at full rank. Returns the energy the
    decoder subspace keeps."""
    head_t = pruned.lm_head
    head_s = student.predictions
    hidden, rank = q.shape
    identity = rank == hidden
    gamma, beta = head_t.layer_norm.weight.float(), head_t.layer_norm.bias.float()
    e = pruned.roberta.embeddings.word_embeddings.weight.float()
    energy = 1.0
    if untied and not identity:
        eg = e * gamma[None, :]
        q_h, energy = embedding_subspace(eg, None, rank)
        head_s.decoder.weight.copy_(eg @ q_h)
        head_s.bias.copy_(head_t.bias.float() + e @ beta)
        head_s.dense.weight.copy_(q_h.T @ head_t.dense.weight.float())
        head_s.dense.bias.copy_(q_h.T @ head_t.dense.bias.float())
        head_s.LayerNorm.weight.fill_((hidden / rank) ** 0.5)
        head_s.LayerNorm.bias.zero_()
        return energy
    head_s.dense.weight.copy_(q.T @ head_t.dense.weight.float())
    head_s.dense.bias.copy_(q.T @ head_t.dense.bias.float())
    if identity:
        head_s.LayerNorm.weight.copy_(gamma)
    else:
        head_s.LayerNorm.weight.fill_(float(gamma.mean()) * (hidden / rank) ** 0.5)
    head_s.LayerNorm.bias.copy_(q.T @ beta)
    return energy


def _splade(logits: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.relu(logits))


def _calibrate_albert_joint(
    pruned: nn.Module, student: nn.Module, batches: list[dict[str, torch.Tensor]], steps: int, lr: float = 1e-3
) -> dict[str, float]:
    """Fit the ALBERT embedding path and MLM head (the vocabulary matrix included) so that

    * the body input ``mapping(LN_r(A[x] + P_r + T_r))`` matches the teacher's ``LN_H(E[x] + P + T)`` and
    * the SPLADE token weights ``log1p(relu(logits))`` match the teacher's (plus a small term on the raw logits)

    on real token hidden states ``h`` of the pruned student's body. The embedding path starts close to the
    optimum (orthonormal projection), so it moves at ``lr / 5``; the head starts with the gelu in the wrong
    space and moves at ``lr``. Returns relative errors before and after.
    """
    from distill.splade import split_mlm_head

    emb_t = pruned.roberta.embeddings
    emb_s = student.albert.embeddings
    mapping = student.albert.encoder.embedding_hidden_mapping_in
    head_s = student.predictions
    transform_t, decoder_t = split_mlm_head(pruned)
    transform_s, decoder_s = split_mlm_head(student)
    vocab = emb_s.word_embeddings.weight.shape[0]
    n_positions = emb_s.position_embeddings.weight.shape[0]
    offset = int(pruned.config.pad_token_id) + 1
    generator = torch.Generator().manual_seed(2)

    with torch.no_grad():
        hiddens = []
        for batch in batches:
            h = pruned.roberta(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
            hiddens.append(h[batch["attention_mask"].bool()])
        hidden = torch.cat(hiddens)
        n_tokens = min(65536, vocab * 4)
        tokens = torch.randint(0, vocab, (n_tokens,), generator=generator)
        positions = torch.randint(0, n_positions, (n_tokens,), generator=generator)
        emb_target = emb_t.LayerNorm(
            emb_t.word_embeddings.weight[tokens] + emb_t.position_embeddings.weight[positions + offset] + emb_t.token_type_embeddings.weight[0]
        )

    def emb_forward(idx: torch.Tensor) -> torch.Tensor:
        v = emb_s.word_embeddings.weight[tokens[idx]] + emb_s.position_embeddings.weight[positions[idx]] + emb_s.token_type_embeddings.weight[0]
        return mapping(emb_s.LayerNorm(v))

    def logits_t(h: torch.Tensor) -> torch.Tensor:
        return decoder_t(transform_t(h))

    def logits_s(h: torch.Tensor) -> torch.Tensor:
        return decoder_s(transform_s(h))

    def relative(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).sum() / (target**2).sum().clamp_min(1e-12)

    def errors() -> tuple[float, float, float]:
        with torch.no_grad():
            idx = torch.arange(min(8192, n_tokens))
            e_err = float(relative(emb_forward(idx), emb_target[idx]))
            sample = torch.arange(min(1024, hidden.shape[0]))
            target = logits_t(hidden[sample])
            pred = logits_s(hidden[sample])
            return e_err, float(relative(pred, target)), float(relative(_splade(pred), _splade(target)))

    before = errors()
    params_emb = [
        emb_s.word_embeddings.weight, emb_s.position_embeddings.weight, emb_s.token_type_embeddings.weight,
        emb_s.LayerNorm.weight, emb_s.LayerNorm.bias, mapping.weight, mapping.bias,
    ]
    params_head = [head_s.dense.weight, head_s.dense.bias, head_s.LayerNorm.weight, head_s.LayerNorm.bias, head_s.bias]
    if head_s.decoder.weight.data_ptr() != emb_s.word_embeddings.weight.data_ptr():
        params_head.append(head_s.decoder.weight)
    opt = torch.optim.Adam([{"params": params_emb, "lr": lr / 5}, {"params": params_head, "lr": lr}])
    for _ in range(max(steps, 0)):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, n_tokens, (4096,), generator=generator)
        e_loss = relative(emb_forward(idx), emb_target[idx])
        sample = torch.randint(0, hidden.shape[0], (256,), generator=generator)
        with torch.no_grad():
            target = logits_t(hidden[sample])
        pred = logits_s(hidden[sample])
        h_loss = relative(_splade(pred), _splade(target)) + 0.01 * relative(pred, target)
        (e_loss + h_loss).backward()
        opt.step()
    after = errors()
    return {
        "embedding_relative_error_before": before[0], "embedding_relative_error_after": after[0],
        "logits_relative_error_before": before[1], "logits_relative_error_after": after[1],
        "splade_relative_error_before": before[2], "splade_relative_error_after": after[2],
        "steps": steps, "tokens": int(hidden.shape[0]),
    }


# -------------------------------------------------------------------------------------------- I/O


def copy_aux_files(src: Path, dst: Path) -> list[str]:
    """Copy tokenizer and sentence-transformers files (not the weights) from ``src`` to ``dst``."""
    copied = []
    for name in AUX_FILES:
        if (src / name).is_file() and not (dst / name).is_file():
            shutil.copy2(src / name, dst / name)
            copied.append(name)
    for name in AUX_DIRS:
        if (src / name).is_dir() and not (dst / name).exists():
            shutil.copytree(src / name, dst / name)
            copied.append(name)
    return copied


def save_student(model: nn.Module, teacher_dir: Path, out_dir: Path, report: BuildReport) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    copy_aux_files(teacher_dir, out_dir)
    (out_dir / "student.json").write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_token_weights(path: Path, smoothing: float = 0.01) -> torch.Tensor:
    """Corpus token counts plus ``smoothing`` x their mean, so that unseen tokens keep a small weight."""
    counts = torch.tensor(json.loads(path.read_text(encoding="utf-8"))["counts"], dtype=torch.float32)
    return counts + smoothing * counts.mean()


def read_texts(path: Path, field_name: str, limit: int, seed: int) -> list[str]:
    """``limit`` texts of a JSON-lines file, evenly spaced after a seeded shuffle of the first 20 x limit lines."""
    import random

    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(texts) >= 20 * limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get(field_name)
            if isinstance(value, str) and value.strip():
                texts.append(value)
            elif isinstance(value, list):
                texts.extend(v for v in value if isinstance(v, str) and v.strip())
    random.Random(seed).shuffle(texts)
    return texts[:limit]


def tokenize_batches(tokenizer: Any, texts: list[str], max_seq_length: int, batch_size: int) -> list[dict[str, torch.Tensor]]:
    batches = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start : start + batch_size], truncation=True, max_length=max_seq_length, padding=True, return_tensors="pt")
        batches.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return batches


# ------------------------------------------------------------------------------------------- build


def build_student(teacher_dir: Path, spec: StudentSpec, sample_texts: list[str] | None = None, max_seq_length: int = 256) -> tuple[nn.Module, BuildReport]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    started = time.time()
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_dir, torch_dtype=torch.float32)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if teacher.config.model_type not in {"roberta", "xlm-roberta", "camembert"}:
        raise StudentSpecError(f"the teacher must be a RoBERTa-family MLM, got {teacher.config.model_type!r}.")
    spec.validate(teacher.config.num_hidden_layers, teacher.config.hidden_size, teacher.config.intermediate_size, teacher.config.vocab_size)
    spec_dict = {k: v for k, v in asdict(spec).items() if k != "token_weights"}
    spec_dict["token_weights"] = spec.token_weights is not None
    report = BuildReport(teacher=str(teacher_dir), spec=spec_dict, teacher_params=count_params(teacher))
    report.macs_per_token["teacher"] = macs_per_token(teacher.config)["total"]

    batches: list[dict[str, torch.Tensor]] = []
    if sample_texts:
        tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
        batches = tokenize_batches(tokenizer, sample_texts, max_seq_length, 16)
    importance = None
    if spec.importance == "activation":
        if not batches:
            raise StudentSpecError("importance 'activation' needs sample texts (--calibration-texts).")
        importance = ffn_importance_activation(teacher, batches)

    student, kept = prune_roberta(teacher, spec, importance)
    report.ffn_importance_kept = kept
    report.architecture = "roberta"
    if spec.embedding_rank is not None:
        student, calibration = to_albert(student, spec, batches)
        report.calibration = calibration
        report.architecture = "albert"
    student.eval()
    report.params = count_params(student)
    report.macs_per_token["student"] = macs_per_token(student.config)["total"]
    report.macs_per_token["ratio"] = report.macs_per_token["student"] / max(report.macs_per_token["teacher"], 1.0)
    report.seconds = round(time.time() - started, 1)
    return student, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--keep-layers", required=True, help="comma-separated teacher layer indices, increasing")
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--embedding-rank", type=int, default=None, help="factorise the vocabulary to this rank (ALBERT output)")
    parser.add_argument("--hidden-size", type=int, default=None, help="with --fresh: width of the new random body")
    parser.add_argument("--fresh", action="store_true", help="random ALBERT body; only vocabulary / decoder from the teacher")
    parser.add_argument("--importance", choices=IMPORTANCE_KINDS, default="weight")
    parser.add_argument("--calibration-texts", type=Path, default=None, help="JSON lines with a text field (also used for activation importance)")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--calibration-sample", type=int, default=128)
    parser.add_argument("--calibration-steps", type=int, default=400)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--untie-head", action="store_true", help="ALBERT output: separate decoder matrix (its own subspace) instead of the tied vocabulary")
    parser.add_argument("--token-counts", type=Path, default=None, help="JSON with a 'counts' list (one per token): frequency-weighted input subspace")
    parser.add_argument("--head-layer", action="store_true", help="ALBERT output: the teacher's head transform as an extra full-width layer and a linear low-rank decoder fitted to real tokens (implies an untied decoder)")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        print(f"[ERROR] {args.out} exists and is not empty (use --force).", file=sys.stderr)
        return 2
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    spec = StudentSpec(
        keep_layers=parse_layers(args.keep_layers),
        intermediate_size=args.intermediate_size,
        embedding_rank=args.embedding_rank,
        hidden_size=args.hidden_size,
        fresh=args.fresh,
        importance=args.importance,
        calibrate=not args.no_calibrate,
        calibration_steps=args.calibration_steps,
        untie_head=args.untie_head,
        head_layer=args.head_layer,
        token_weights=load_token_weights(args.token_counts) if args.token_counts else None,
        seed=args.seed,
    )
    texts = read_texts(args.calibration_texts, args.text_field, args.calibration_sample, args.seed) if args.calibration_texts else None
    student, report = build_student(args.teacher, spec, texts, args.max_seq_length)
    save_student(student, args.teacher, args.out, report)
    print(json.dumps({"out": str(args.out), "architecture": report.architecture, "params": report.params, "macs_per_token": report.macs_per_token, "calibration": report.calibration}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
