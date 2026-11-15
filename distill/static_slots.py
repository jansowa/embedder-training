#!/usr/bin/env python
"""Learn which dimensions a static query table writes, not only how strongly: new slots from the training loss.

``distill.train_static`` fits the weights of the slots a table was initialised with; the slots themselves come from
the credit init (the teacher's mean activation per token) and never change. This module proposes slots from the loss
instead. On a candidate pool it computes, for every query, the direction in which the whole query vector should move
to lower the KL against the teacher,

    want = scale * (p_teacher - p_student) @ D        over the row's candidates, one value per vocabulary dimension,

and keeps the dimensions the query does not write yet where a positive weight would help most (``--candidates`` per
query). Each is credited to every row the query activates (its tokens, words, word pairs). Summed over the pool and
divided by the number of queries a row occurs in, a (row, dimension) pair stands out only when the dimension is
wanted consistently in the presence of that row. The best pairs of each row (``--new-slots``, wanted in at least
``--min-support`` queries) are written into the row's unused slots with a small weight (``--init-value``);
``distill.train_static`` then decides what they are worth, and a next round can repeat the step.

The table keeps its format and, by default, its shape (path A of docs/distillation.md): new slots go only into
slots that are empty or dead (weight 0). ``--replace-below W`` also gives up live slots lighter than W; ``--new-k K``
widens every row to K slots first, which is a larger table (path B).

The rows ``distill.train_static`` holds out for validation (same ``--val-rows`` / ``--seed``) are never read.

Usage:
  .venv/bin/python -m distill.static_slots --pool cache/static_pools/kd-X-r2 --model models/static-X/phr2-k8-kd-r2 \\
      --out models/static-X/phr2-k8-kd-r2-g1 --new-slots 2 --device cuda
  .venv/bin/python -m distill.train_static --pool cache/static_pools/kd-X-r2 --init models/static-X/phr2-k8-kd-r2-g1 ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPORT_NAME = "slots_report.json"


class PairAccumulator:
    """Running sum and hit count per (row, dimension), coalesced in chunks. When the distinct pairs pass ``max_pairs``
    the lighter half is dropped: pairs that far down could not be selected, and the 16 GB box has no room for them."""

    def __init__(self, rows: int, dim: int, flush_every: int = 2_000_000, max_pairs: int = 20_000_000) -> None:
        self.rows, self.dim = rows, dim
        self.flush_every, self.max_pairs = flush_every, max_pairs
        self.keys = np.zeros(0, dtype=np.int64)
        self.sums = np.zeros(0, dtype=np.float64)
        self.hits = np.zeros(0, dtype=np.int64)
        self.dropped = 0
        self._keys: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []
        self._pending = 0

    def add(self, rows: np.ndarray, dims: np.ndarray, values: np.ndarray) -> None:
        self._keys.append(rows.astype(np.int64) * self.dim + dims.astype(np.int64))
        self._vals.append(values.astype(np.float64))
        self._pending += len(rows)
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._keys:
            return
        new_keys = np.concatenate(self._keys)
        keys = np.concatenate([self.keys, new_keys])
        vals = np.concatenate([self.sums, *self._vals])
        hits = np.concatenate([self.hits, np.ones(len(new_keys), dtype=np.int64)])
        self.keys, inverse = np.unique(keys, return_inverse=True)
        self.sums = np.bincount(inverse, weights=vals, minlength=len(self.keys))
        self.hits = np.bincount(inverse, weights=hits, minlength=len(self.keys)).astype(np.int64)
        self._keys, self._vals, self._pending = [], [], 0
        if len(self.keys) > self.max_pairs:
            keep = np.sort(np.argpartition(-self.sums, self.max_pairs // 2)[: self.max_pairs // 2])
            self.dropped += len(self.keys) - len(keep)
            self.keys, self.sums, self.hits = self.keys[keep], self.sums[keep], self.hits[keep]


@torch.no_grad()
def collect_gains(trainer, row_ids: list[int], candidates: int, batch_size: int, scale: float, log_every: int = 0) -> tuple[PairAccumulator, np.ndarray]:
    """``(pair gains, queries per row)`` over the pool rows ``row_ids``; see the module docstring for ``want``."""
    from distill.static_query import special_token_ids

    model = trainer.model
    specials = np.array(special_token_ids(trainer.tokenizer), dtype=np.int64)
    acc = PairAccumulator(model.rows, model.output_dim)
    counts = np.zeros(model.rows, dtype=np.int64)
    started = time.time()
    for n_batch, start in enumerate(range(0, len(row_ids), batch_size)):
        batch = trainer.batch_tensors(row_ids[start : start + batch_size])
        enc = model.tokenize(trainer.tokenizer, [r["query"] for r in batch["rows"]], trainer.device)
        q = model(**enc)
        scores = (q @ batch["docs"].T) * scale
        mask = batch["own"] & ~torch.isnan(batch["row_scores"])
        low = torch.finfo(scores.dtype).min
        p_student = torch.softmax(torch.where(mask, scores, torch.full_like(scores, low)), dim=-1)
        p_teacher = torch.softmax(torch.where(mask, torch.nan_to_num(batch["row_scores"]), torch.full_like(scores, low)), dim=-1)
        want = ((p_teacher - p_student).masked_fill(~mask, 0.0) @ batch["docs"]) * scale  # [B, V] = -dKL/dq
        want = want.masked_fill(q > 0, 0.0)  # a dimension the query writes already is the weights' business
        top_val, top_dim = torch.topk(want, min(candidates, want.shape[1]), dim=1)
        ids, active = enc["input_ids"].cpu().numpy(), enc["attention_mask"].cpu().numpy() > 0
        top_val, top_dim = top_val.cpu().numpy(), top_dim.cpu().numpy()
        for b in range(len(ids)):
            rows = np.unique(ids[b][active[b]])
            rows = rows[~np.isin(rows, specials)]
            if not len(rows):
                continue
            counts[rows] += 1
            keep = top_val[b] > 0
            if keep.any():
                dims, vals = top_dim[b][keep], top_val[b][keep]
                acc.add(np.repeat(rows, len(dims)), np.tile(dims, len(rows)), np.tile(vals, len(rows)))
        if log_every and n_batch % log_every == 0:
            print(f"[slots] {start + batch_size}/{len(row_ids)} rows, {time.time() - started:.0f}s", flush=True)
    acc.flush()
    return acc, counts


def select_slots(acc: PairAccumulator, counts: np.ndarray, model, new_slots: int, min_support: int = 3, min_count: int = 5,
                 min_gain: float = 0.0, vocab_size: int | None = None) -> dict[str, np.ndarray]:
    """Per row the ``new_slots`` best dimensions it does not hold yet: mean gain per query of the row, wanted in at
    least ``min_support`` queries of a row seen in at least ``min_count``. Sorted by row, best first."""
    acc.flush()
    rows, dims = acc.keys // acc.dim, acc.keys % acc.dim
    gain = acc.sums / np.maximum(counts[rows], 1)
    ok = (acc.hits >= min_support) & (counts[rows] >= min_count) & (gain > min_gain)
    if vocab_size is not None:
        ok &= dims < vocab_size  # dimensions past the tokenizer have no term in the index
    held = model.active_slot_mask().cpu().numpy()
    held_keys = (np.arange(model.rows, dtype=np.int64)[:, None] * acc.dim + model.indices.cpu().numpy().astype(np.int64))[held]
    ok &= ~np.isin(acc.keys, held_keys)
    rows, dims, gain, hits = rows[ok], dims[ok], gain[ok], acc.hits[ok]
    order = np.lexsort((-gain, rows))
    rows, dims, gain, hits = rows[order], dims[order], gain[order], hits[order]
    rank = np.arange(len(rows)) - np.searchsorted(rows, rows, side="left")
    keep = rank < new_slots
    return {"rows": rows[keep], "dims": dims[keep], "gain": gain[keep], "hits": hits[keep]}


@torch.no_grad()
def grow(model, picks: dict[str, np.ndarray], init_value: float = 0.05, init_rel: float = 0.0, replace_below: float = 0.0, new_k: int = 0):
    """``(model with the picked slots written, per-pick outcome)``: 1 = an empty or dead slot was used, 2 = a live slot
    lighter than ``replace_below`` was given up, 0 = the row had no room."""
    signed = bool(getattr(model, "signed", False))
    indices, values, mask = model.indices.cpu().clone(), model.values.detach().cpu().clone(), model.slot_mask.cpu().clone()
    if new_k > indices.shape[1]:
        pad = new_k - indices.shape[1]
        indices = torch.cat([indices, torch.zeros(indices.shape[0], pad, dtype=indices.dtype)], dim=1)
        values = torch.cat([values, torch.zeros(values.shape[0], pad)], dim=1)
        mask = torch.cat([mask, torch.zeros(mask.shape[0], pad, dtype=torch.bool)], dim=1)
    weight = (values.abs() if signed else torch.relu(values)) * mask
    outcome = np.zeros(len(picks["rows"]), dtype=np.int64)
    bounds = np.flatnonzero(np.diff(picks["rows"], prepend=-1))
    for lo, hi in zip(bounds, list(bounds[1:]) + [len(picks["rows"])]):
        row = int(picks["rows"][lo])
        w = weight[row]
        live = w[w > 0]
        value = max(init_value, init_rel * float(live.mean())) if len(live) else init_value
        slots = torch.argsort(w, stable=True).tolist()  # empty and dead slots first, then the lightest live ones
        for n, pick in enumerate(range(lo, hi)):
            if n >= len(slots):
                break
            slot = slots[n]
            if float(w[slot]) > 0 and float(w[slot]) >= replace_below:
                break
            outcome[pick] = 2 if float(w[slot]) > 0 else 1
            indices[row, slot], values[row, slot], mask[row, slot] = int(picks["dims"][pick]), value, True
    grown = model.with_table(indices, values, mask)
    return grown.to(model.values.device), outcome


def row_text(model, tokenizer, row: int) -> str:
    from distill.static_query import PHRASE_SEP

    if row < model.base_rows:
        return str(tokenizer.convert_ids_to_tokens(int(row)))
    words, current = [], []
    for piece in model.phrases[row - model.base_rows]:
        if piece == PHRASE_SEP:
            words.append(current)
            current = []
        else:
            current.append(piece)
    words.append(current)
    return " + ".join(tokenizer.decode(w).strip() for w in words if w)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="static query encoder folder (usually a trained table)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--new-slots", type=int, default=2, help="at most this many new slots per row")
    parser.add_argument("--candidates", type=int, default=16, help="dimensions kept per query")
    parser.add_argument("--min-support", type=int, default=3, help="queries in which a (row, dimension) pair must be wanted")
    parser.add_argument("--min-count", type=int, default=5, help="queries a row must occur in")
    parser.add_argument("--min-gain", type=float, default=0.0, help="mean gain per query of the row a pair must exceed")
    parser.add_argument("--init-value", type=float, default=0.05, help="weight of a new slot")
    parser.add_argument("--init-rel", type=float, default=0.0, help="or this share of the row's mean live weight, when larger")
    parser.add_argument("--replace-below", type=float, default=0.0, help="also give up live slots lighter than this (0 = only empty / dead slots)")
    parser.add_argument("--new-k", type=int, default=0, help="widen every row to this many slots first (a larger table)")
    parser.add_argument("--max-rows", type=int, default=60000, help="pool rows read (0 = all training rows)")
    parser.add_argument("--val-rows", type=int, default=2000, help="held out exactly as distill.train_static does")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scale", type=float, default=0.0, help="multiplier of the student scores (0 = train_static's initial estimate)")
    parser.add_argument("--teacher-scale", type=float, default=1.0)
    parser.add_argument("--show", type=int, default=40, help="examples in the report")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from distill.static_pool import load_pool
    from distill.static_query import load_static_encoder
    from distill.train_static import StaticTrainer, encoder_stats

    started = time.time()
    pool = load_pool(args.pool)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_static_encoder(args.model, args.device).eval()
    trainer_args = argparse.Namespace(target="rows", learn_scale=True, teacher_scale=args.teacher_scale, device=args.device)
    try:
        trainer = StaticTrainer(pool, model, tokenizer, trainer_args)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    ids = [i for i in range(len(pool.rows)) if trainer.usable(i)]
    random.Random(args.seed).shuffle(ids)
    train_ids = ids[args.val_rows :]
    if args.max_rows:
        train_ids = train_ids[: args.max_rows]
    if not train_ids:
        print("[ERROR] no pool rows left after the validation split", file=sys.stderr)
        return 2
    scale = args.scale or trainer.init_scale(trainer.batch_tensors(train_ids[: args.batch_size]))
    before = encoder_stats(model)
    acc, counts = collect_gains(trainer, train_ids, args.candidates, args.batch_size, scale, args.log_every)
    picks = select_slots(acc, counts, model, args.new_slots, args.min_support, args.min_count, args.min_gain, vocab_size=len(tokenizer))
    grown, outcome = grow(model, picks, args.init_value, args.init_rel, args.replace_below, args.new_k)
    written = outcome > 0
    df = trainer.df_fraction.cpu().numpy()
    total = picks["gain"] * counts[picks["rows"]]
    examples = [
        {"row": row_text(model, tokenizer, int(picks["rows"][i])), "dimension": str(tokenizer.convert_ids_to_tokens(int(picks["dims"][i]))),
         "gain_per_query": round(float(picks["gain"][i]), 5), "queries_wanting": int(picks["hits"][i]), "queries_of_row": int(counts[picks["rows"][i]])}
        for i in np.argsort(-np.where(written, total, -1.0))[: args.show] if written[i]
    ]
    report = {
        "pool": str(args.pool), "model": str(args.model), "rows_read": len(train_ids), "scale": round(float(scale), 5),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pairs_collected": int(len(acc.keys)), "pairs_dropped_for_memory": int(acc.dropped),
        "picks": int(len(outcome)), "written_into_empty_slots": int((outcome == 1).sum()), "written_over_live_slots": int((outcome == 2).sum()),
        "rows_without_room": int(len(np.unique(picks["rows"][~written]))), "rows_grown": int(len(np.unique(picks["rows"][written]))),
        # every new slot counted as active in each query of its row: the most the postings can grow before training
        "postings_fraction_added_at_most": round(float((counts[picks["rows"][written]] * df[picks["dims"][written]]).sum() / max(len(train_ids), 1)), 4),
        "table_before": before, "table_after": encoder_stats(grown), "examples": examples, "seconds": round(time.time() - started, 1),
    }
    grown.meta = dict(grown.meta, slots={k: report[k] for k in ("pool", "rows_read", "picks", "written_into_empty_slots", "written_over_live_slots")}, grown_from=str(args.model))
    grown.save(args.out, tokenizer)
    (args.out / REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("args", "table_before", "table_after")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
