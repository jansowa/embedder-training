#!/usr/bin/env python
"""Larger static query encoders (format ``static-sparse-query/2``): path B of docs/distillation.md.

The released static table (``distill.static_query``, format 1) is a lookup: every token or phrase row writes k
non-negative (dimension, weight) slots and a query vector is the maximum over its rows. This module keeps the rows
and the slots and adds what a lookup cannot express. Every addition is off by default, so a table converted without
options computes exactly what format 1 computes; each one is a separate switch, to be measured separately:

  pooling sum + signed   q = relu(sum of the rows - threshold), weights of either sign: rows vote, and a phrase row
                         can take back what its pieces write. ``build --rows residual`` initialises the phrase rows
                         as the difference between the phrase's credit and what its parts already write;
  threshold              a scalar (learned with ``learn_threshold``) subtracted from every pooled dimension;
  phrase_mode replace    a word that owns a row is encoded by that row alone, not by the row plus its pieces;
  pair_key first,        word-pair rows keyed by the first piece of each word (a crude stem: one row for the
  pair_window n          inflected forms of a pair), matched for words up to n positions apart;
  ctx_dim d              a context gate: the weights of every row are multiplied by 2 * sigmoid(g_row . c + b_row),
                         c = the mean of d-dimensional token embeddings over the query. No attention and no layers:
                         one more lookup and one dot product per row. ``slot_ctx_dim`` adds a gate per slot on a
                         projection of c (which of a row's expansions fit this query);
  length_edges           one learned multiplier of the expansion slots (every slot that is not a token's own
                         dimension) per query-length bucket;
  top_m                  only the m heaviest dimensions of the pooled vector are kept.

Gates start at exactly 1 (zero gate vectors), the threshold at 0 and the length multipliers at 1: a converted table
is the starting point of every experiment and training can only move away from it.

Files: ``static_query_v2.json`` + ``static_query_v2.safetensors`` + the tokenizer. A folder carries one format;
``distill.static_query.load_static_encoder`` picks the class. ``distill.train_static`` trains both formats with the
same loop, ``distill.train_static_vec`` fits an encoder to the teacher's query vectors without a pool, the PIRB fork
loads a folder of this format as a query model (``search/splade_static_ext.py``). The released Sentence Transformers
module (``release/st_static.py``) reads format 1 only: ``export-v1`` writes a format-1 table back when the encoder
uses nothing beyond it (more rows or more slots, the same arithmetic).

Usage:
  # the released table plus a 32-dimensional context gate (identity until trained)
  .venv/bin/python -m distill.static_v2 convert --model models/static-X/phr2-k8-kd-r2 --ctx-dim 32 \\
      --out models/static2-X/phr2-k8-gate32
  # signed sum with residual phrase rows, credited from the document model
  .venv/bin/python -m distill.static_v2 build --rows residual --teacher models/X --texts datasets/kd/kd_v1.jsonl \\
      --kinds query --top-k 8 --phrases 2 --out models/static2-X/res2-k8
  .venv/bin/python -m distill.static_v2 stats --model models/static2-X/res2-k8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from distill.static_query import (
    CONFIG_V2_NAME,
    PHRASE_SEP,
    POOLINGS,
    CreditAccumulator,
    StaticSparseQueryEncoder,
    build_phrase_vocab,
    credit_accumulate,
    credit_table,
    load_static_encoder,
    load_texts,
    output_dim_of,
    pack_phrases,
    punct_token_mask,
    split_words,
    table_stats,
    unpack_phrases,
)

FORMAT = "static-sparse-query/2"
WEIGHTS_V2_NAME = "static_query_v2.safetensors"
PHRASE_MODES = ("add", "replace")
PAIR_KEYS = ("exact", "first")
EXTRA_TENSORS = ("ctx_emb", "gate_vec", "gate_bias", "ctx_proj", "slot_gate", "length_scale")


def phrase_keys_v2(ids: list[int], word_ids: list[int | None], punct: np.ndarray, max_words: int, pair_key: str = "exact", pair_window: int = 1) -> list[tuple[tuple[int, ...], list[int]]]:
    """``distill.static_query.phrase_keys`` with two more key schemes for word pairs: ``pair_key = "first"`` keys a pair
    by the first piece of each word, ``pair_window = n`` pairs a word with each of the next n words (one row whatever
    the gap). ``exact`` with window 1 is format 1."""
    words = split_words(ids, word_ids, punct)
    out = []
    for j, (key, pos) in enumerate(words):
        if max_words >= 1 and len(key) > 1:
            out.append((key, pos))
        if max_words >= 2:
            for gap in range(1, pair_window + 1):
                if j + gap >= len(words):
                    break
                other, other_pos = words[j + gap]
                pair = (key[0], PHRASE_SEP, other[0]) if pair_key == "first" else key + (PHRASE_SEP,) + other
                out.append((pair, pos + other_pos))
    return out


class StaticQueryEncoderV2(nn.Module):
    """``[B, L]`` row ids -> ``[B, V]`` sparse query vectors; see the module docstring for the switches."""

    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        output_dim: int,
        slot_mask: torch.Tensor | None = None,
        *,
        pooling: str = "max",
        signed: bool = False,
        threshold: float = 0.0,
        learn_threshold: bool = False,
        phrase_mode: str = "add",
        pair_key: str = "exact",
        pair_window: int = 1,
        ctx_dim: int = 0,
        slot_ctx_dim: int = 0,
        length_edges: tuple[int, ...] | list[int] = (),
        top_m: int = 0,
        add_special_tokens: bool = True,
        max_length: int = 128,
        meta: dict[str, Any] | None = None,
        phrases: list[tuple[int, ...]] | None = None,
        phrase_max_words: int = 0,
        base_rows: int | None = None,
        extra: dict[str, torch.Tensor] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if indices.shape != values.shape or indices.dim() != 2:
            raise ValueError(f"indices {tuple(indices.shape)} and values {tuple(values.shape)} must both be [rows, k].")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}.")
        if signed and pooling != "sum":
            raise ValueError("signed weights need pooling 'sum' (a maximum cannot take anything back).")
        if phrase_mode not in PHRASE_MODES or pair_key not in PAIR_KEYS or pair_window < 1:
            raise ValueError(f"phrase_mode in {PHRASE_MODES}, pair_key in {PAIR_KEYS}, pair_window >= 1.")
        if slot_ctx_dim and not ctx_dim:
            raise ValueError("slot_ctx_dim needs ctx_dim (the slot gates read a projection of the context).")
        if slot_mask is None:
            slot_mask = torch.ones_like(indices, dtype=torch.bool)
        if indices.numel() and int(indices.max()) >= output_dim:
            raise ValueError(f"an index {int(indices.max())} is outside the output dimension {output_dim}.")
        extra = dict(extra or {})
        rows, k = indices.shape
        self.register_buffer("indices", indices.long().clone())
        self.register_buffer("slot_mask", slot_mask.bool().clone())
        self.values = nn.Parameter(values.float().clone())
        self.output_dim = int(output_dim)
        self.pooling, self.signed = pooling, bool(signed)
        self.learn_threshold = bool(learn_threshold)
        self.threshold = nn.Parameter(torch.tensor(float(threshold)), requires_grad=self.learn_threshold)
        self.phrase_mode, self.pair_key, self.pair_window = phrase_mode, pair_key, int(pair_window)
        self.ctx_dim, self.slot_ctx_dim, self.top_m = int(ctx_dim), int(slot_ctx_dim), int(top_m)
        self.add_special_tokens = add_special_tokens
        self.max_length = int(max_length)
        self.meta = dict(meta or {})
        self.phrases = list(phrases or [])
        self.phrase_max_words = int(phrase_max_words) if self.phrases else 0
        self.base_rows = int(base_rows if base_rows is not None else rows - len(self.phrases))
        if self.base_rows + len(self.phrases) != rows:
            raise ValueError(f"{rows} rows != {self.base_rows} token rows + {len(self.phrases)} phrase rows.")
        self._lookup = {key: self.base_rows + i for i, key in enumerate(self.phrases)}
        self._word_rows = {self.base_rows + i for i, key in enumerate(self.phrases) if PHRASE_SEP not in key}
        self._punct: tuple[int, np.ndarray] | None = None
        self.sparse_grad = False
        generator = torch.Generator().manual_seed(seed)

        def tensor(name: str, shape: tuple[int, ...], init: str) -> nn.Parameter:
            given = extra.get(name)
            if given is not None:
                if tuple(given.shape) != shape:
                    raise ValueError(f"{name} has shape {tuple(given.shape)}, this encoder needs {shape}.")
                return nn.Parameter(given.float().clone())
            if init == "zeros":
                return nn.Parameter(torch.zeros(shape))
            if init == "ones":
                return nn.Parameter(torch.ones(shape))
            return nn.Parameter(torch.randn(shape, generator=generator) / shape[-2 if name == "ctx_proj" else -1] ** 0.5)

        for name in EXTRA_TENSORS:
            self.register_parameter(name, None)
        if self.ctx_dim:
            self.ctx_emb = tensor("ctx_emb", (self.base_rows, self.ctx_dim), "normal")
            self.gate_vec = tensor("gate_vec", (rows, self.ctx_dim), "zeros")
            self.gate_bias = tensor("gate_bias", (rows,), "zeros")
            if self.slot_ctx_dim:
                self.ctx_proj = tensor("ctx_proj", (self.ctx_dim, self.slot_ctx_dim), "normal")
                self.slot_gate = tensor("slot_gate", (rows, k, self.slot_ctx_dim), "zeros")
        edges = sorted(int(e) for e in length_edges)
        self.register_buffer("length_edges", torch.tensor(edges, dtype=torch.long) if edges else None)
        if edges:
            self.length_scale = tensor("length_scale", (len(edges) + 1,), "ones")

    # ------------------------------------------------------------------ shape and bookkeeping

    @property
    def rows(self) -> int:
        return self.indices.shape[0]

    @property
    def k(self) -> int:
        return self.indices.shape[1]

    @property
    def dedupe(self) -> bool:
        """Under sum pooling a row counts once per query (the credit init counts it once per text)."""
        return self.pooling == "sum"

    @property
    def gain_foldable(self) -> bool:
        """A per-dimension multiplier of the output can be folded into the weights (``train_static --dim-gain``)."""
        return float(self.threshold) == 0.0 and not self.learn_threshold and self.top_m == 0

    def active_slot_mask(self) -> torch.Tensor:
        values = self.values.detach()
        return self.slot_mask & ((values != 0) if self.signed else (values > 0))

    def trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        extra = [getattr(self, name) for name in EXTRA_TENSORS if getattr(self, name) is not None]
        if self.learn_threshold:
            extra.append(self.threshold)
        return {"values": [self.values], "extra": extra}

    def options(self) -> dict[str, Any]:
        return {
            "pooling": self.pooling, "signed": self.signed, "threshold": float(self.threshold), "learn_threshold": self.learn_threshold,
            "phrase_mode": self.phrase_mode, "pair_key": self.pair_key, "pair_window": self.pair_window, "ctx_dim": self.ctx_dim,
            "slot_ctx_dim": self.slot_ctx_dim, "length_edges": self.length_edges.tolist() if self.length_edges is not None else [],
            "top_m": self.top_m, "add_special_tokens": self.add_special_tokens, "max_length": self.max_length,
            "phrase_max_words": self.phrase_max_words, "base_rows": self.base_rows,
        }

    def extra_tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().cpu() for name in EXTRA_TENSORS if getattr(self, name) is not None}

    def stats(self) -> dict[str, Any]:
        out = table_stats(self)
        tensors = self.saved_tensors()
        out.update(
            format=FORMAT, pooling=self.pooling, signed=self.signed, threshold=round(float(self.threshold), 5), phrase_mode=self.phrase_mode,
            pair_key=self.pair_key, pair_window=self.pair_window, ctx_dim=self.ctx_dim, slot_ctx_dim=self.slot_ctx_dim, top_m=self.top_m,
            length_scale=[round(float(x), 4) for x in self.length_scale.detach()] if self.length_scale is not None else None,
            negative_slots=int((self.slot_mask & (self.values.detach() < 0)).sum()) if self.signed else 0,
            extra_parameters=sum(p.numel() for p in self.trainable_parameters()["extra"]),
            megabytes=round(sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6, 2),
        )
        return out

    # ------------------------------------------------------------------ encoding

    def slot_values(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.sparse_grad:
            return torch.nn.functional.embedding(input_ids, self.values, sparse=True)
        return self.values[input_ids]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``attention_mask`` marks the rows that are pooled, ``context_mask`` the query's tokens (the context and
        the length are read from them; by default every pooled token row)."""
        batch = input_ids.shape[0]
        idx = self.indices[input_ids]  # [B, L, k]
        val = self.slot_values(input_ids)
        if not self.signed:
            val = torch.relu(val)
        val = val * self.slot_mask[input_ids].to(val.dtype)
        is_token = input_ids < self.base_rows
        if context_mask is None:
            context_mask = attention_mask * is_token.to(attention_mask.dtype)
        if self.ctx_dim:
            weight = context_mask.to(val.dtype).unsqueeze(-1)
            ctx = (self.ctx_emb[input_ids.clamp_max(self.base_rows - 1)] * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)  # [B, d]
            gate = 2.0 * torch.sigmoid((self.gate_vec[input_ids] * ctx.unsqueeze(1)).sum(dim=-1) + self.gate_bias[input_ids])  # [B, L]
            val = val * gate.unsqueeze(-1)
            if self.slot_ctx_dim:
                val = val * 2.0 * torch.sigmoid(torch.einsum("blkd,bd->blk", self.slot_gate[input_ids], ctx @ self.ctx_proj))
        if self.length_scale is not None:
            scale = self.length_scale.clamp_min(0.0)[torch.bucketize(context_mask.sum(dim=1), self.length_edges)]  # [B]
            own = (idx == input_ids.unsqueeze(-1)) & is_token.unsqueeze(-1)  # a token row's slot for the token itself
            val = torch.where(own, val, val * scale.view(batch, 1, 1))
        val = val * attention_mask.unsqueeze(-1).to(val.dtype)
        out = val.new_zeros(batch, self.output_dim)
        if self.pooling == "max":  # weights are >= 0 here, so the zero start is the empty max
            out = out.scatter_reduce(1, idx.reshape(batch, -1), val.reshape(batch, -1), reduce="amax", include_self=True)
        else:
            out = out.scatter_add(1, idx.reshape(batch, -1), val.reshape(batch, -1))
        if self.signed or self.learn_threshold or float(self.threshold) != 0.0:
            out = torch.relu(out - self.threshold.clamp_min(0.0))
        if 0 < self.top_m < self.output_dim:
            kth = torch.topk(out, self.top_m, dim=1).values[:, -1:]
            out = torch.where(out >= kth, out, torch.zeros_like(out))
        return out

    def key_fn(self):
        return partial(phrase_keys_v2, pair_key=self.pair_key, pair_window=self.pair_window)

    def tokenize(self, tokenizer, texts: list[str], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """``input_ids`` (token ids, then the phrase rows of each text), ``attention_mask`` (rows that are pooled) and
        ``context_mask`` (the text's tokens)."""
        enc = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=self.add_special_tokens, return_tensors="pt")
        input_ids, context = enc["input_ids"], enc["attention_mask"]
        pooled = context.clone()
        extra: list[list[int]] = [[] for _ in texts]
        if self.phrases:
            if self._punct is None or self._punct[0] != id(tokenizer):
                self._punct = (id(tokenizer), punct_token_mask(tokenizer))
            for i, ids in enumerate(input_ids.tolist()):
                seen: set[int] = set()
                for key, positions in phrase_keys_v2(ids, enc.word_ids(i), self._punct[1], self.phrase_max_words, self.pair_key, self.pair_window):
                    row = self._lookup.get(key)
                    if row is None:
                        continue
                    if self.phrase_mode == "replace" and row in self._word_rows:
                        pooled[i, positions] = 0  # the word row stands for its pieces, at every occurrence
                    if row not in seen:
                        seen.add(row)
                        extra[i].append(row)
        if self.dedupe:
            for i, ids in enumerate(input_ids.tolist()):
                seen = set()
                for pos, tid in enumerate(ids):
                    if pooled[i, pos]:
                        if tid in seen:
                            pooled[i, pos] = 0
                        seen.add(tid)
        width = max((len(e) for e in extra), default=0)
        if width:
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            ext_ids = torch.full((len(texts), width), pad, dtype=input_ids.dtype)
            ext_mask = torch.zeros((len(texts), width), dtype=pooled.dtype)
            for i, rows in enumerate(extra):
                if rows:
                    ext_ids[i, : len(rows)] = torch.tensor(rows, dtype=input_ids.dtype)
                    ext_mask[i, : len(rows)] = 1
            input_ids = torch.cat([input_ids, ext_ids], dim=1)
            pooled = torch.cat([pooled, ext_mask], dim=1)
            context = torch.cat([context, torch.zeros_like(ext_mask)], dim=1)
        return {"input_ids": input_ids.to(device), "attention_mask": pooled.to(device), "context_mask": context.to(device)}

    def encode(self, tokenizer, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        device = self.values.device
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                out.append(self(**self.tokenize(tokenizer, texts[start : start + batch_size], device)).cpu())
        return torch.cat(out) if out else torch.zeros(0, self.output_dim)

    # ------------------------------------------------------------------ files and conversions

    def saved_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "indices": self.indices.to(torch.int32).cpu().contiguous(),
            "values": self.values.detach().float().cpu().contiguous(),
            "slot_mask": self.slot_mask.cpu().contiguous(),
            "threshold": self.threshold.detach().float().cpu().reshape(1),
        }
        tensors.update({name: t.contiguous() for name, t in self.extra_tensors().items()})
        if self.phrases:
            tensors["phrase_pieces"], tensors["phrase_offsets"] = pack_phrases(self.phrases)
        return tensors

    def config_dict(self) -> dict[str, Any]:
        options = self.options()
        options.pop("threshold")  # the trained value lives in the tensors
        return {"format": FORMAT, "rows": self.rows, "k": self.k, "output_dim": self.output_dim, "phrases": len(self.phrases),
                "active_slots": int(self.active_slot_mask().sum()), **options, "meta": self.meta}

    def save(self, out_dir: Path, tokenizer=None) -> None:
        from safetensors.torch import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.saved_tensors(), str(out_dir / WEIGHTS_V2_NAME))
        (out_dir / CONFIG_V2_NAME).write_text(json.dumps(self.config_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if tokenizer is not None:
            tokenizer.save_pretrained(str(out_dir))

    @classmethod
    def load(cls, model_dir: Path, device: torch.device | str = "cpu") -> "StaticQueryEncoderV2":
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        config = json.loads((model_dir / CONFIG_V2_NAME).read_text(encoding="utf-8"))
        if config.get("format") != FORMAT:
            raise ValueError(f"{model_dir / CONFIG_V2_NAME}: format {config.get('format')!r}, this code reads {FORMAT!r}.")
        tensors = load_file(str(model_dir / WEIGHTS_V2_NAME))
        options = {key: config[key] for key in ("pooling", "signed", "learn_threshold", "phrase_mode", "pair_key", "pair_window", "ctx_dim", "slot_ctx_dim",
                                                "length_edges", "top_m", "add_special_tokens", "max_length", "phrase_max_words", "base_rows")}
        model = cls(
            tensors["indices"], tensors["values"], config["output_dim"], tensors.get("slot_mask"), threshold=float(tensors["threshold"][0]),
            meta=config.get("meta"), phrases=unpack_phrases(tensors["phrase_pieces"], tensors["phrase_offsets"]) if "phrase_pieces" in tensors else None,
            extra={name: tensors[name] for name in EXTRA_TENSORS if name in tensors}, **options,
        )
        return model.to(device)

    @classmethod
    def from_v1(cls, table: StaticSparseQueryEncoder, **options: Any) -> "StaticQueryEncoderV2":
        """A format-1 table as an encoder of this format; without ``options`` it computes the same vectors."""
        base = {"pooling": table.pooling, "add_special_tokens": table.add_special_tokens, "max_length": table.max_length, "meta": dict(table.meta),
                "phrases": table.phrases, "phrase_max_words": table.phrase_max_words, "base_rows": table.base_rows}
        base.update(options)
        return cls(table.indices.cpu(), table.values.detach().cpu(), table.output_dim, table.slot_mask.cpu(), **base)

    def reconfigured(self, **options: Any) -> "StaticQueryEncoderV2":
        """A copy with some switches changed; trained gates and multipliers are kept where their shapes still fit."""
        merged = dict(self.options(), meta=dict(self.meta), phrases=self.phrases)
        merged.update(options)
        probe = {"ctx_dim": merged["ctx_dim"], "slot_ctx_dim": merged["slot_ctx_dim"], "edges": sorted(merged["length_edges"])}
        same_gate = probe["ctx_dim"] == self.ctx_dim
        extra = {}
        for name, t in self.extra_tensors().items():
            if name in ("ctx_emb", "gate_vec", "gate_bias") and same_gate:
                extra[name] = t
            elif name in ("ctx_proj", "slot_gate") and same_gate and probe["slot_ctx_dim"] == self.slot_ctx_dim:
                extra[name] = t
            elif name == "length_scale" and probe["edges"] == (self.length_edges.tolist() if self.length_edges is not None else []):
                extra[name] = t
        return StaticQueryEncoderV2(self.indices.cpu(), self.values.detach().cpu(), self.output_dim, self.slot_mask.cpu(), extra=extra, **merged)

    def with_table(self, indices: torch.Tensor, values: torch.Tensor, slot_mask: torch.Tensor, meta: dict[str, Any] | None = None) -> "StaticQueryEncoderV2":
        """A copy with another table for the same rows (``distill.static_slots``); gates and multipliers are kept."""
        if self.slot_ctx_dim and indices.shape[1] != self.k:
            raise ValueError("an encoder with slot gates cannot change its number of slots per row.")
        merged = dict(self.options(), meta=dict(self.meta if meta is None else meta), phrases=self.phrases)
        return StaticQueryEncoderV2(indices, values, self.output_dim, slot_mask, extra=self.extra_tensors(), **merged)

    def beyond_v1(self) -> list[str]:
        """The switches in use that format 1 cannot express (empty = ``to_v1`` is exact)."""
        used = []
        if self.signed:
            used.append("signed")
        if self.learn_threshold or float(self.threshold) != 0.0:
            used.append("threshold")
        if self.phrases and self.phrase_mode != "add":
            used.append("phrase_mode")
        if self.phrase_max_words >= 2 and (self.pair_key != "exact" or self.pair_window != 1):
            used.append("pair keys")
        if self.ctx_dim:
            used.append("context gate")
        if self.length_scale is not None:
            used.append("length multipliers")
        if self.top_m:
            used.append("top_m")
        return used

    def to_v1(self) -> StaticSparseQueryEncoder:
        used = self.beyond_v1()
        if used:
            raise ValueError(f"this encoder uses {', '.join(used)}: format 1 cannot hold it.")
        return StaticSparseQueryEncoder(
            self.indices.cpu(), self.values.detach().cpu(), self.output_dim, self.slot_mask.cpu(), self.pooling, self.add_special_tokens, self.max_length,
            dict(self.meta), phrases=self.phrases, phrase_max_words=self.phrase_max_words, base_rows=self.base_rows,
        )


def phrase_parts(key: tuple[int, ...], lookup: dict[tuple[int, ...], int]) -> list[int]:
    """The rows that are pooled together with the row of phrase ``key``: the token rows of its pieces and, for a word
    pair, the rows of its words when they own one. Each row once."""
    words: list[list[int]] = [[]]
    for piece in key:
        if piece == PHRASE_SEP:
            words.append([])
        else:
            words[-1].append(piece)
    parts = [piece for word in words for piece in word]
    if len(words) > 1:
        parts += [lookup[tuple(word)] for word in words if len(word) > 1 and tuple(word) in lookup]
    return list(dict.fromkeys(parts))


def residual_phrase_rows(acc: CreditAccumulator, base: int, phrases: list[tuple[int, ...]], indices: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
                         top_k: int, min_abs: float = 1e-3) -> dict[str, int]:
    """Rewrite the phrase rows of a credited table (in place) for signed sum pooling: a phrase row becomes its mean
    credit minus what the rows pooled with it already write, cut to the ``top_k`` largest magnitudes. Word rows are
    settled first, word pairs after them (a pair is pooled with the rows of its words)."""
    acc.flush()
    rows, dims = acc.keys // acc.output_dim, acc.keys % acc.output_dim
    lookup = {key: base + i for i, key in enumerate(phrases)}
    idx, val, live = indices.numpy(), values.numpy(), mask.numpy()  # views: the tensors are rewritten
    negative = positive = empty = 0
    for i in sorted(range(len(phrases)), key=lambda i: PHRASE_SEP in phrases[i]):
        row = base + i
        lo, hi = np.searchsorted(rows, row, side="left"), np.searchsorted(rows, row, side="right")
        count = max(int(acc.counts[row]), 1)
        residual = dict(zip(dims[lo:hi].tolist(), (acc.sums[lo:hi] / count).tolist()))
        for part in phrase_parts(phrases[i], lookup):
            for j in np.flatnonzero(live[part] & (val[part] != 0)):
                d = int(idx[part, j])
                residual[d] = residual.get(d, 0.0) - float(val[part, j])
        top = [(d, v) for d, v in sorted(residual.items(), key=lambda kv: -abs(kv[1]))[:top_k] if abs(v) >= min_abs]
        idx[row], val[row], live[row] = 0, 0.0, False
        for j, (d, v) in enumerate(top):
            idx[row, j], val[row, j], live[row, j] = d, v, True
        negative += sum(v < 0 for _, v in top)
        positive += sum(v > 0 for _, v in top)
        empty += not top
    return {"residual_slots_positive": positive, "residual_slots_negative": negative, "residual_rows_empty": empty}


# ---------------------------------------------------------------------- command line

SWITCHES = ("pooling", "signed", "threshold", "learn_threshold", "phrase_mode", "ctx_dim", "slot_ctx_dim", "length_edges", "top_m")


def add_switches(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pooling", choices=POOLINGS, default=None)
    parser.add_argument("--signed", action="store_true", default=None, help="weights of either sign under sum pooling")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--learn-threshold", action="store_true", default=None)
    parser.add_argument("--phrase-mode", choices=PHRASE_MODES, default=None)
    parser.add_argument("--ctx-dim", type=int, default=None, help="size of the context embeddings (0 = no gate)")
    parser.add_argument("--slot-ctx-dim", type=int, default=None, help="size of the projected context of the per-slot gates (0 = none)")
    parser.add_argument("--length-edges", default=None, help="comma list of token counts that split queries into length buckets, e.g. 6,10,16,32")
    parser.add_argument("--top-m", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="initialisation of the context embeddings")


def switches_of(args: argparse.Namespace) -> dict[str, Any]:
    chosen = {name: getattr(args, name) for name in SWITCHES if getattr(args, name) is not None}
    if "length_edges" in chosen:
        chosen["length_edges"] = [int(x) for x in str(chosen["length_edges"]).split(",") if x.strip()]
    return chosen


def convert_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    model = load_static_encoder(args.model)
    switches = switches_of(args)
    try:
        if isinstance(model, StaticQueryEncoderV2):
            model = model.reconfigured(seed=args.seed, **switches)
        else:
            model = StaticQueryEncoderV2.from_v1(model, seed=args.seed, **switches)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    model.meta = dict(model.meta, converted_from=str(args.model))
    model.save(args.out, AutoTokenizer.from_pretrained(args.model))
    print(json.dumps({"out": str(args.out), **model.stats()}, indent=2))
    return 0


def build_main(args: argparse.Namespace) -> int:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    switches = switches_of(args)
    if args.rows == "residual":
        switches.update(pooling="sum", signed=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.teacher)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    teacher = AutoModelForMaskedLM.from_pretrained(args.teacher, torch_dtype=dtype).to(args.device).eval()
    kinds = set(args.kinds.split(",")) if args.kinds else None
    texts = load_texts(args.texts, args.n, kinds, args.field, args.text_seed)
    key_fn = partial(phrase_keys_v2, pair_key=args.pair_key, pair_window=args.pair_window)
    started = time.time()
    phrases: list[tuple[int, ...]] = []
    if args.phrases:
        phrases = build_phrase_vocab(tokenizer, texts, args.phrases, args.phrase_min_count, args.max_phrases, args.max_length, key_fn=key_fn)
        print(f"[phrases] {len(phrases)} phrase rows (max {args.phrases} words, min count {args.phrase_min_count}, pair key {args.pair_key}, window {args.pair_window})", flush=True)
    common = {"batch_size": args.batch_size, "max_length": args.max_length, "device": args.device, "log_every": args.log_every,
              "phrases": phrases, "phrase_max_words": args.phrases, "key_fn": key_fn}
    acc = credit_accumulate(teacher, tokenizer, texts, **common)
    indices, values, mask, info = credit_table(teacher, tokenizer, texts, args.top_k, min_count=args.min_count, isolated_fallback=not args.no_isolated_fallback,
                                               accumulator=acc, **common)
    base = len(tokenizer)
    if args.rows == "residual" and phrases:
        info.update(residual_phrase_rows(acc, base, phrases, indices, values, mask, args.top_k))
    meta = {"init": f"credit/{args.rows}", "teacher": str(args.teacher), "texts": str(args.texts), "n": args.n, **info,
            "phrase_min_count": args.phrase_min_count, "max_phrases": args.max_phrases, "build_seconds": round(time.time() - started, 1)}
    try:
        model = StaticQueryEncoderV2(indices, values * args.scale, output_dim_of(tokenizer, teacher), mask, pair_key=args.pair_key, pair_window=args.pair_window,
                                     max_length=args.max_length, meta=meta, phrases=phrases, phrase_max_words=args.phrases, base_rows=base, seed=args.seed, **switches)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    model.save(args.out, tokenizer)
    print(json.dumps({"out": str(args.out), **model.stats()}, indent=2))
    return 0


def export_v1_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    try:
        table = StaticQueryEncoderV2.load(args.model).to_v1()
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    table.save(args.out, AutoTokenizer.from_pretrained(args.model))
    print(json.dumps({"out": str(args.out), **table_stats(table)}, indent=2))
    return 0


def stats_main(args: argparse.Namespace) -> int:
    model = StaticQueryEncoderV2.load(args.model)
    print(json.dumps({"model": str(args.model), **model.stats(), "beyond_format_1": model.beyond_v1()}, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="a format-1 table (or an encoder of this format) with switches changed")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    add_switches(c)
    b = sub.add_parser("build", help="credit a table from a SPLADE teacher with this format's row schemes")
    b.add_argument("--rows", choices=("credit", "residual"), default="credit",
                   help="credit: every row is its mean credit (as format 1); residual: signed sum, phrase rows are what the phrase adds to its parts")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--teacher", type=Path, required=True)
    b.add_argument("--tokenizer", type=Path, default=None)
    b.add_argument("--texts", type=Path, required=True)
    b.add_argument("--field", default="text")
    b.add_argument("--kinds", default=None)
    b.add_argument("--n", type=int, default=0)
    b.add_argument("--text-seed", type=int, default=42)
    b.add_argument("--top-k", type=int, default=8)
    b.add_argument("--min-count", type=int, default=3)
    b.add_argument("--no-isolated-fallback", action="store_true")
    b.add_argument("--phrases", type=int, choices=(0, 1, 2), default=0)
    b.add_argument("--phrase-min-count", type=int, default=5)
    b.add_argument("--max-phrases", type=int, default=200000)
    b.add_argument("--pair-key", choices=PAIR_KEYS, default="exact")
    b.add_argument("--pair-window", type=int, default=1)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--max-length", type=int, default=128)
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    b.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16" if torch.cuda.is_available() else "fp32")
    b.add_argument("--log-every", type=int, default=200)
    add_switches(b)
    e = sub.add_parser("export-v1", help="write a format-1 table when the encoder uses nothing beyond it")
    e.add_argument("--model", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("stats")
    s.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"convert": convert_main, "build": build_main, "export-v1": export_v1_main, "stats": stats_main}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
