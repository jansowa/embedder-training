#!/usr/bin/env python
"""Ranking agreement of a student with the teacher scores stored in training rows.

For each row (``query``, ``pos``, ``neg``, ``pos_scores``, ``neg_scores`` = the teacher's dot products, as in
the ``-tv9b`` arm files) the student's dot products are compared with the teacher's over the row's
documents: Kendall tau (concordant minus discordant pairs over all pairs), whether the top-1 document
agrees, and whether the student puts a positive first. The score the queue reads is the mean tau; it
measures what the ranking stage (``sparse_distill_kl_div``) will have to change, which the pooled cosine
on passages does not show.

Usage:
  .venv/bin/python -m distill.rank_agreement --model models/kd-b4-s2 --rows datasets/arms/mix3v2-6219-tv9b.jsonl \
      --n 500 --device cuda --dtype fp16 --output results/rank_agreement/kd-b4-s2.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from distill.splade import encode_pooled, prepare_head_layer


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Tau-a over all pairs (ties count as neither concordant nor discordant)."""
    n = len(a)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def load_rows(path: Path, n: int, seed: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("query") and row.get("pos") and row.get("pos_scores") and row.get("neg_scores"):
                rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:n]


@torch.no_grad()
def encode(model, tokenizer, texts: list[str], max_seq_length: int, batch_size: int, device: str) -> torch.Tensor:
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        vec = encode_pooled(model, enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu()
        out.append((idx, vec))
    result = torch.zeros(len(texts), out[0][1].shape[1]) if out else torch.zeros(0, 0)
    for idx, vec in out:
        result[idx] = vec
    return result


def agreement(model, tokenizer, rows: list[dict], max_seq_length: int, query_max_length: int, batch_size: int, device: str) -> dict:
    queries = [r["query"] for r in rows]
    docs: list[str] = []
    spans: list[tuple[int, int, int]] = []  # start, n_pos, n_docs
    teacher: list[list[float]] = []
    for r in rows:
        pos, neg = list(r["pos"]), list(r["neg"])
        spans.append((len(docs), len(pos), len(pos) + len(neg)))
        docs.extend(pos + neg)
        teacher.append([float(x) for x in r["pos_scores"]] + [float(x) for x in r["neg_scores"]])
    q_vec = encode(model, tokenizer, queries, query_max_length, batch_size, device)
    d_vec = encode(model, tokenizer, docs, max_seq_length, batch_size, device)
    taus, top1, pos_first, margins = [], [], [], []
    for i, (start, n_pos, n_docs) in enumerate(spans):
        s = (d_vec[start : start + n_docs] @ q_vec[i]).tolist()
        t = teacher[i][:n_docs]
        if len(s) != len(t) or n_docs < 2:
            continue
        taus.append(kendall_tau(s, t))
        top1.append(float(max(range(n_docs), key=s.__getitem__) == max(range(n_docs), key=t.__getitem__)))
        pos_first.append(float(max(range(n_docs), key=s.__getitem__) < n_pos))
        margins.append(max(s[:n_pos]) - max(s[n_pos:]) if n_docs > n_pos else 0.0)
    n = len(taus)
    return {
        "rows": n,
        "kendall_tau": sum(taus) / max(n, 1),
        "top1_agreement": sum(top1) / max(n, 1),
        "positive_first": sum(pos_first) / max(n, 1),
        "mean_margin": sum(margins) / max(n, 1),
        "tau_p10": sorted(taus)[int(0.1 * n)] if n else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--fp32-head", action="store_true", help="head-layer students: extra layer FFN + head transform in fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    rows = load_rows(args.rows, args.n, args.seed)
    if not rows:
        print("[ERROR] no usable rows", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    if args.fp32_head:
        prepare_head_layer(model)
    started = time.time()
    result = agreement(model, tokenizer, rows, args.max_seq_length, args.query_max_length, args.batch_size, args.device)
    result.update({"model": str(args.model), "rows_file": str(args.rows), "dtype": args.dtype, "fp32_head": bool(args.fp32_head), "seconds": round(time.time() - started, 1)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
