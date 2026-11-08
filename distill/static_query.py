#!/usr/bin/env python
"""Static sparse query encoder: a SPLADE-shaped query vector from a lookup table, no transformer.

Every input token ``t`` owns one row of a sparse table: ``k`` (vocabulary dimension, weight) pairs. A query is
tokenized and its vector over the vocabulary is the maximum (or the sum) of its tokens' rows:

    q[v] = max_{l : mask[l] = 1} relu(values[ids[l], j])   over the slots j with indices[ids[l], j] = v

Two shapes cover the useful variants:

* ``weights``    k = 1 and ``indices[t, 0] = t``: one weight per token, only the query's own tokens are active.
                 The same model as sentence-transformers' ``SparseStaticEmbedding`` (an "inference-free" query side).
* ``expansion``  k > 1: each token also activates up to k - 1 other vocabulary dimensions (inflected forms,
                 synonyms), learned from a SPLADE teacher; a context-free SPLADE.

The document side is not touched: queries encoded here are scored against an index built by a full SPLADE model
(``q . d`` over the shared vocabulary, the same tokenizer). Training lives in ``distill.train_static``; the candidate
pool with frozen document vectors in ``distill.static_pool``.

Initialisations (``python -m distill.static_query build``):

* ``idf``        weights, BM25 idf of each token over a passage sample (special tokens 0);
* ``ones``       weights, all 1;
* ``credit``     expansion table from a SPLADE teacher on a query sample: for every query, every active dimension v of
                 the teacher's pooled vector is credited to the token at the position where the max was reached;
                 row t = mean credit over the queries that contain t, top-k kept. On a single query with k large
                 the table reproduces the teacher's vector exactly (a test checks it). Tokens seen in fewer than
                 ``--min-count`` queries get their row from the teacher run on ``<s> t </s>`` alone (``isolated``).
* ``isolated``   the expansion table from ``<s> t </s>`` for every token (Model2Vec style, no corpus).

Saved folder: ``static_query.json`` (config), ``static_query.safetensors`` (``indices`` int32 [R, k], ``values``
float32 [R, k], ``slot_mask`` bool [R, k]) and the tokenizer files. ``third_party/pirb/search/splade.py`` loads such
a folder as ``query_model`` (``PIRB_QUERY_MODEL`` in ``scripts/pirb_regime_eval.sh``).

Usage:
  .venv/bin/python -m distill.static_query build --init idf --tokenizer models/arm-dense-kl-n5kj-ens400 \\
      --texts datasets/kd/kd_v1.jsonl --kinds passage --n 300000 --out models/static/idf
  .venv/bin/python -m distill.static_query build --init credit --teacher models/arm-dense-kl-n5kj-ens400 \\
      --texts datasets/kd/kd_v1.jsonl --kinds query --n 400000 --top-k 32 --out models/static/credit-k32
  .venv/bin/python -m distill.static_query stats --model models/static/credit-k32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CONFIG_NAME = "static_query.json"
WEIGHTS_NAME = "static_query.safetensors"
POOLINGS = ("max", "sum")
INITS = ("idf", "ones", "credit", "isolated")


class StaticSparseQueryEncoder(nn.Module):
    """``[B, L]`` token ids -> ``[B, V]`` sparse query vectors through a per-token table of ``k`` (dim, weight) slots."""

    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        output_dim: int,
        slot_mask: torch.Tensor | None = None,
        pooling: str = "max",
        add_special_tokens: bool = True,
        max_length: int = 128,
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if indices.shape != values.shape or indices.dim() != 2:
            raise ValueError(f"indices {tuple(indices.shape)} and values {tuple(values.shape)} must both be [rows, k].")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}.")
        if slot_mask is None:
            slot_mask = torch.ones_like(indices, dtype=torch.bool)
        if indices.numel() and int(indices.max()) >= output_dim:
            raise ValueError(f"an index {int(indices.max())} is outside the output dimension {output_dim}.")
        self.register_buffer("indices", indices.long().clone())
        self.register_buffer("slot_mask", slot_mask.bool().clone())
        self.values = nn.Parameter(values.float().clone())
        self.output_dim = int(output_dim)
        self.pooling = pooling
        self.add_special_tokens = add_special_tokens
        self.max_length = int(max_length)
        self.meta = dict(meta or {})

    @property
    def rows(self) -> int:
        return self.indices.shape[0]

    @property
    def k(self) -> int:
        return self.indices.shape[1]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch = input_ids.shape[0]
        idx = self.indices[input_ids]  # [B, L, k]
        val = torch.relu(self.values[input_ids]) * self.slot_mask[input_ids].to(self.values.dtype)
        val = val * attention_mask.unsqueeze(-1).to(val.dtype)
        out = val.new_zeros(batch, self.output_dim)
        if self.pooling == "max":  # weights are >= 0, so the zero start is the empty max
            return out.scatter_reduce(1, idx.reshape(batch, -1), val.reshape(batch, -1), reduce="amax", include_self=True)
        return out.scatter_add(1, idx.reshape(batch, -1), val.reshape(batch, -1))

    def tokenize(self, tokenizer, texts: list[str], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=self.add_special_tokens, return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}

    def encode(self, tokenizer, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        device = self.values.device
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                enc = self.tokenize(tokenizer, texts[start : start + batch_size], device)
                out.append(self(enc["input_ids"], enc["attention_mask"]).cpu())
        return torch.cat(out) if out else torch.zeros(0, self.output_dim)

    def config_dict(self) -> dict[str, Any]:
        return {
            "format": "static-sparse-query/1",
            "rows": self.rows,
            "k": self.k,
            "output_dim": self.output_dim,
            "pooling": self.pooling,
            "add_special_tokens": self.add_special_tokens,
            "max_length": self.max_length,
            "active_slots": int(self.active_slot_mask().sum()),
            "meta": self.meta,
        }

    def active_slot_mask(self) -> torch.Tensor:
        return self.slot_mask & (self.values.detach() > 0)

    def save(self, out_dir: Path, tokenizer=None) -> None:
        from safetensors.torch import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "indices": self.indices.to(torch.int32).cpu().contiguous(),
                "values": self.values.detach().float().cpu().contiguous(),
                "slot_mask": self.slot_mask.cpu().contiguous(),
            },
            str(out_dir / WEIGHTS_NAME),
        )
        (out_dir / CONFIG_NAME).write_text(json.dumps(self.config_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if tokenizer is not None:
            tokenizer.save_pretrained(str(out_dir))

    @classmethod
    def load(cls, model_dir: Path, device: torch.device | str = "cpu") -> "StaticSparseQueryEncoder":
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        config = json.loads((model_dir / CONFIG_NAME).read_text(encoding="utf-8"))
        tensors = load_file(str(model_dir / WEIGHTS_NAME))
        model = cls(
            tensors["indices"],
            tensors["values"],
            config["output_dim"],
            slot_mask=tensors.get("slot_mask"),
            pooling=config.get("pooling", "max"),
            add_special_tokens=config.get("add_special_tokens", True),
            max_length=config.get("max_length", 128),
            meta=config.get("meta"),
        )
        return model.to(device)

    @torch.no_grad()
    def pruned(self, min_weight: float = 0.0, top_k: int = 0) -> "StaticSparseQueryEncoder":
        """A copy with slots below ``min_weight`` (after relu) dropped and at most ``top_k`` slots per row."""
        values = torch.relu(self.values.detach()) * self.slot_mask
        keep = values > max(min_weight, 0.0)
        values = torch.where(keep, values, torch.zeros_like(values))
        k = self.k if top_k <= 0 else min(top_k, self.k)
        top_val, order = torch.topk(values, k=k, dim=1)
        top_idx = torch.gather(self.indices, 1, order)
        mask = top_val > 0
        meta = dict(self.meta, pruned={"min_weight": min_weight, "top_k": top_k})
        return StaticSparseQueryEncoder(
            torch.where(mask, top_idx, torch.zeros_like(top_idx)), top_val, self.output_dim, mask, self.pooling, self.add_special_tokens, self.max_length, meta
        )


def output_dim_of(tokenizer, teacher: nn.Module | None = None) -> int:
    """Width of the query vectors: the teacher's MLM head when given (it may be padded past the tokenizer), else the tokenizer."""
    if teacher is not None:
        return int(teacher.get_output_embeddings().weight.shape[0])
    return len(tokenizer)


def special_token_ids(tokenizer) -> list[int]:
    return sorted({int(i) for i in tokenizer.all_special_ids})


def weights_table(weights: torch.Tensor, output_dim: int, **kwargs: Any) -> StaticSparseQueryEncoder:
    """k = 1, ``indices[t] = t``: one weight per token (the ``SparseStaticEmbedding`` model)."""
    rows = weights.shape[0]
    indices = torch.arange(rows).clamp_max(output_dim - 1).unsqueeze(1)
    mask = (torch.arange(rows) < output_dim).unsqueeze(1)
    return StaticSparseQueryEncoder(indices, weights.float().unsqueeze(1) * mask, output_dim, mask, **kwargs)


def idf_weights(tokenizer, texts: list[str], rows: int, batch_size: int = 1024, max_length: int = 512) -> torch.Tensor:
    """BM25 idf ``log(1 + (N - df + 0.5) / (df + 0.5))`` of each token id over ``texts``; special tokens get 0."""
    df = np.zeros(rows, dtype=np.int64)
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start : start + batch_size], add_special_tokens=False, truncation=True, max_length=max_length)
        for ids in enc["input_ids"]:
            uniq = np.unique(np.asarray(ids, dtype=np.int64))
            df[uniq[uniq < rows]] += 1
    n = max(len(texts), 1)
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
    weights = torch.from_numpy(idf).float()
    for tid in special_token_ids(tokenizer):
        if tid < rows:
            weights[tid] = 0.0
    return weights


class CreditAccumulator:
    """Sparse running sums of (token row, vocabulary dim) -> credited weight, coalesced in chunks."""

    def __init__(self, rows: int, output_dim: int, flush_every: int = 4_000_000) -> None:
        self.rows = rows
        self.output_dim = output_dim
        self.flush_every = flush_every
        self.keys = np.zeros(0, dtype=np.int64)
        self.sums = np.zeros(0, dtype=np.float64)
        self._pending_keys: list[np.ndarray] = []
        self._pending_vals: list[np.ndarray] = []
        self._pending = 0
        self.counts = np.zeros(rows, dtype=np.int64)

    def add(self, rows: np.ndarray, dims: np.ndarray, values: np.ndarray) -> None:
        self._pending_keys.append(rows.astype(np.int64) * self.output_dim + dims.astype(np.int64))
        self._pending_vals.append(values.astype(np.float64))
        self._pending += len(rows)
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending_keys:
            return
        keys = np.concatenate([self.keys, *self._pending_keys])
        vals = np.concatenate([self.sums, *self._pending_vals])
        self.keys, inverse = np.unique(keys, return_inverse=True)
        self.sums = np.bincount(inverse, weights=vals, minlength=len(self.keys))
        self._pending_keys, self._pending_vals, self._pending = [], [], 0

    def table(self, top_k: int, divide_by_count: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(indices, values, mask)`` [rows, top_k]: per row the top-k mean credits."""
        self.flush()
        rows = self.keys // self.output_dim
        dims = self.keys % self.output_dim
        vals = self.sums / np.maximum(self.counts[rows], 1) if divide_by_count else self.sums.copy()
        order = np.lexsort((-vals, rows))  # by row, then by value descending
        rows, dims, vals = rows[order], dims[order], vals[order]
        starts = np.searchsorted(rows, rows, side="left")
        rank = np.arange(len(rows)) - starts
        keep = rank < top_k
        indices = torch.zeros(self.rows, top_k, dtype=torch.long)
        values = torch.zeros(self.rows, top_k, dtype=torch.float32)
        mask = torch.zeros(self.rows, top_k, dtype=torch.bool)
        r, c = torch.from_numpy(rows[keep]), torch.from_numpy(rank[keep])
        indices[r, c] = torch.from_numpy(dims[keep])
        values[r, c] = torch.from_numpy(vals[keep]).float()
        mask[r, c] = True
        return indices, values, mask


@torch.no_grad()
def teacher_token_activations(teacher: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """``(pooled [B, V], argmax position [B, V])`` of ``max_l log1p(relu(logits[b, l, v])) * mask``, on position chunks."""
    from distill.splade import encode_hidden, split_mlm_head

    transform, decoder = split_mlm_head(teacher)
    hidden = encode_hidden(teacher, input_ids, attention_mask)
    batch, seq_len = input_ids.shape
    best = torch.zeros(batch, decoder.weight.shape[0], device=input_ids.device)
    where = torch.zeros(batch, decoder.weight.shape[0], dtype=torch.long, device=input_ids.device)
    for start in range(0, seq_len, chunk):
        act = torch.log1p(torch.relu(decoder(transform(hidden[:, start : start + chunk])).float()))
        act = act * attention_mask[:, start : start + chunk].unsqueeze(-1).to(act.dtype)
        value, pos = act.max(dim=1)
        better = value > best
        best = torch.where(better, value, best)
        where = torch.where(better, pos + start, where)
    return best, where


def credit_batch(acc: CreditAccumulator, teacher: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, only_positions: torch.Tensor | None = None, threshold: float = 0.0) -> None:
    """Credit every active teacher dimension of each text to the token at its argmax position and count token presence."""
    pooled, where = teacher_token_activations(teacher, input_ids, attention_mask)
    token_at = torch.gather(input_ids, 1, where)
    active = pooled > threshold
    if only_positions is not None:  # isolated mode: only the credits that land on the probed position
        active = active & (where == only_positions.unsqueeze(1))
    b, v = active.nonzero(as_tuple=True)
    rows = token_at[b, v]
    keep = rows < acc.rows
    acc.add(rows[keep].cpu().numpy(), v[keep].cpu().numpy(), pooled[b, v][keep].cpu().numpy())
    for ids, mask in zip(input_ids.cpu().numpy(), attention_mask.cpu().numpy()):
        if only_positions is None:
            uniq = np.unique(ids[mask > 0])
            acc.counts[uniq[uniq < acc.rows]] += 1


def credit_table(
    teacher: nn.Module,
    tokenizer,
    texts: list[str],
    top_k: int,
    *,
    batch_size: int = 64,
    max_length: int = 128,
    min_count: int = 3,
    isolated_fallback: bool = True,
    device: str = "cpu",
    log_every: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Expansion table from the teacher's pooled vectors on ``texts`` (queries); rare tokens from ``<s> t </s>``."""
    rows = len(tokenizer)
    out_dim = output_dim_of(tokenizer, teacher)
    acc = CreditAccumulator(rows, out_dim)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    started = time.time()
    for n_batch, start in enumerate(range(0, len(order), batch_size)):
        enc = tokenizer([texts[i] for i in order[start : start + batch_size]], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        credit_batch(acc, teacher, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        if log_every and n_batch % log_every == 0:
            print(f"[credit] {start + batch_size}/{len(texts)} texts, {time.time() - started:.0f}s", flush=True)
    indices, values, mask = acc.table(top_k)
    specials = set(special_token_ids(tokenizer))
    rare = [t for t in range(rows) if acc.counts[t] < min_count and t not in specials] if isolated_fallback else []
    if rare:
        iso_idx, iso_val, iso_mask = isolated_table(teacher, tokenizer, top_k, token_ids=rare, batch_size=max(batch_size, 256), device=device)
        rare_t = torch.tensor(rare)
        indices[rare_t], values[rare_t], mask[rare_t] = iso_idx[rare_t], iso_val[rare_t], iso_mask[rare_t]
    info = {"texts": len(texts), "tokens_seen": int((acc.counts > 0).sum()), "tokens_isolated": len(rare), "min_count": min_count, "seconds": round(time.time() - started, 1)}
    return indices, values, mask, info


def isolated_table(teacher: nn.Module, tokenizer, top_k: int, token_ids: list[int] | None = None, batch_size: int = 256, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row t = the teacher's pooled weights credited to t in ``<s> t </s>`` (special positions excluded), top-k."""
    rows = len(tokenizer)
    out_dim = output_dim_of(tokenizer, teacher)
    acc = CreditAccumulator(rows, out_dim)
    specials = set(special_token_ids(tokenizer))
    ids = [t for t in (token_ids if token_ids is not None else range(rows)) if t not in specials]
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        seqs = [tokenizer.build_inputs_with_special_tokens([t]) for t in chunk]
        width = max(len(s) for s in seqs)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=device)
        attention_mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)
        positions = torch.tensor([s.index(t) for s, t in zip(seqs, chunk)], device=device)
        credit_batch(acc, teacher, input_ids, attention_mask, only_positions=positions)
    acc.counts[:] = 1
    return acc.table(top_k)


def table_stats(model: StaticSparseQueryEncoder) -> dict[str, Any]:
    active = model.active_slot_mask()
    per_row = active.sum(dim=1)
    live = per_row > 0
    self_slot = ((model.indices == torch.arange(model.rows, device=model.indices.device).unsqueeze(1)) & active).any(dim=1)
    vals = torch.relu(model.values.detach())[active]
    return {
        "rows": model.rows,
        "k": model.k,
        "live_rows": int(live.sum()),
        "active_slots": int(active.sum()),
        "slots_per_live_row": round(float(per_row[live].float().mean()), 2) if live.any() else 0.0,
        "rows_with_self_slot": int(self_slot.sum()),
        "weight_mean": round(float(vals.mean()), 4) if vals.numel() else 0.0,
        "weight_max": round(float(vals.max()), 4) if vals.numel() else 0.0,
    }


def load_texts(path: Path, n: int, kinds: set[str] | None, field: str = "text", seed: int = 42) -> list[str]:
    """Texts of a JSON-lines file (a kd corpus with ``text`` / ``kind``, or training rows with ``--field query``)."""
    from distill.data import TextCorpus

    return TextCorpus([path], field=field, kinds=kinds, max_items=n, seed=seed).texts


def build_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok_dir = args.tokenizer or args.teacher
    if tok_dir is None:
        print("[ERROR] --tokenizer or --teacher is required", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    common = {"pooling": args.pooling, "add_special_tokens": not args.no_special_tokens, "max_length": args.max_length}
    meta: dict[str, Any] = {"init": args.init, "tokenizer": str(tok_dir), "texts": str(args.texts) if args.texts else None, "n": args.n}
    started = time.time()
    if args.init in ("idf", "ones"):
        rows = len(tokenizer)
        out_dim = rows
        if args.teacher:
            from transformers import AutoModelForMaskedLM

            out_dim = output_dim_of(tokenizer, AutoModelForMaskedLM.from_pretrained(args.teacher))
        if args.init == "idf":
            if not args.texts:
                print("[ERROR] --init idf needs --texts (passages)", file=sys.stderr)
                return 2
            texts = load_texts(args.texts, args.n, kinds, args.field, args.seed)
            weights = idf_weights(tokenizer, texts, rows) * args.scale
            meta["idf_texts"] = len(texts)
        else:
            weights = torch.full((rows,), args.scale)
            for tid in special_token_ids(tokenizer):
                weights[tid] = 0.0
        model = weights_table(weights, out_dim, meta=meta, **common)
    else:
        from transformers import AutoModelForMaskedLM

        if not args.teacher:
            print(f"[ERROR] --init {args.init} needs --teacher", file=sys.stderr)
            return 2
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
        teacher = AutoModelForMaskedLM.from_pretrained(args.teacher, torch_dtype=dtype).to(args.device).eval()
        out_dim = output_dim_of(tokenizer, teacher)
        if args.init == "credit":
            if not args.texts:
                print("[ERROR] --init credit needs --texts (queries)", file=sys.stderr)
                return 2
            texts = load_texts(args.texts, args.n, kinds, args.field, args.seed)
            indices, values, mask, info = credit_table(
                teacher, tokenizer, texts, args.top_k, batch_size=args.batch_size, max_length=args.max_length,
                min_count=args.min_count, isolated_fallback=not args.no_isolated_fallback, device=args.device, log_every=args.log_every,
            )
            meta.update(info)
        else:
            indices, values, mask = isolated_table(teacher, tokenizer, args.top_k, batch_size=args.batch_size, device=args.device)
        meta["teacher"] = str(args.teacher)
        model = StaticSparseQueryEncoder(indices, values * args.scale, out_dim, mask, meta=meta, **common)
    model.meta["build_seconds"] = round(time.time() - started, 1)
    model.save(args.out, tokenizer)
    print(json.dumps({"out": str(args.out), **table_stats(model)}, indent=2))
    return 0


def stats_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    model = StaticSparseQueryEncoder.load(args.model)
    result: dict[str, Any] = {"model": str(args.model), **table_stats(model)}
    if args.show:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        vocab = {v: k for k, v in tokenizer.get_vocab().items()}
        shown = {}
        for word in args.show.split(","):
            for tid in tokenizer(word, add_special_tokens=False)["input_ids"]:
                active = model.active_slot_mask()[tid]
                pairs = sorted(((float(model.values[tid, j]), vocab.get(int(model.indices[tid, j]), "?")) for j in range(model.k) if active[j]), reverse=True)
                shown[vocab.get(tid, str(tid))] = [(t, round(w, 3)) for w, t in pairs[:16]]
        result["rows_shown"] = shown
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def prune_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    model = StaticSparseQueryEncoder.load(args.model).pruned(args.min_weight, args.top_k)
    model.save(args.out, AutoTokenizer.from_pretrained(args.model))
    print(json.dumps({"out": str(args.out), **table_stats(model)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="initialise a static query table")
    b.add_argument("--init", choices=INITS, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--teacher", type=Path, default=None, help="SPLADE model (credit / isolated; also fixes the output width)")
    b.add_argument("--tokenizer", type=Path, default=None, help="tokenizer folder (defaults to --teacher)")
    b.add_argument("--texts", type=Path, default=None, help="JSON lines: corpus with a text field, or training rows (--field query)")
    b.add_argument("--field", default="text")
    b.add_argument("--kinds", default=None, help="comma list of corpus kinds to keep (kd_v1: query, passage)")
    b.add_argument("--n", type=int, default=0, help="sample this many texts (0 = all)")
    b.add_argument("--top-k", type=int, default=32)
    b.add_argument("--min-count", type=int, default=3)
    b.add_argument("--no-isolated-fallback", action="store_true")
    b.add_argument("--scale", type=float, default=1.0, help="multiply the initial weights")
    b.add_argument("--pooling", choices=POOLINGS, default="max")
    b.add_argument("--no-special-tokens", action="store_true", help="encode queries without <s> / </s>")
    b.add_argument("--max-length", type=int, default=128)
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    b.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16" if torch.cuda.is_available() else "fp32")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--log-every", type=int, default=200)
    s = sub.add_parser("stats", help="table statistics, optionally the rows of a few words")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--show", default="", help="comma list of words whose token rows to print")
    p = sub.add_parser("prune", help="drop small slots / cap slots per row")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args(argv)
    return {"build": build_main, "stats": stats_main, "prune": prune_main}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
