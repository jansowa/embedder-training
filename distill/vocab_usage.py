#!/usr/bin/env python
"""Which output dimensions a SPLADE model ever uses.

Encodes a text sample with the model, quantises the pooled vectors the way the index does
(``round(w * quantization_factor) > 0`` = active) and reports, per vocabulary dimension, in how many
texts it is active and how much weight it carries. The point: dimensions that never fire on any text
are dead weight in the MLM head (one row of the 50 001 x H decoder each); with an untied head they can
be dropped, and the count of live dimensions bounds how small the head can get without any loss.

Output JSON: live dimension counts at several thresholds, the mass covered by the top-K dimensions,
and the full per-dimension counts / mass (``--per-dim``) for a later pruning step.

Usage:
  .venv/bin/python -m distill.vocab_usage --model models/soup-v9b --texts datasets/kd/kd_v1.jsonl \
      --n 20000 --device cpu --output results/vocab_usage/soup-v9b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def usage_counts(
    model: torch.nn.Module, tokenizer, texts: list[str], max_seq_length: int, batch_size: int, device: str,
    quantization_factor: float = 100.0, head_chunk: int = 64,
) -> dict[str, torch.Tensor]:
    from distill.splade import encode_pooled

    vocab = int(model.config.vocab_size)
    counts = torch.zeros(vocab, dtype=torch.long)
    mass = torch.zeros(vocab, dtype=torch.float64)
    peak = torch.zeros(vocab, dtype=torch.float32)
    active_per_text: list[int] = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), batch_size):
        batch = [texts[i] for i in order[start : start + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt")
        with torch.no_grad():
            vec = encode_pooled(model, enc["input_ids"].to(device), enc["attention_mask"].to(device), head_chunk).float().cpu()
        active = torch.round(vec * quantization_factor) > 0
        counts += active.sum(dim=0)
        mass += (vec * active).sum(dim=0).double()
        peak = torch.maximum(peak, vec.max(dim=0).values)
        active_per_text.extend(int(x) for x in active.sum(dim=1))
    return {"counts": counts, "mass": mass, "peak": peak, "active_per_text": torch.tensor(active_per_text)}


def summarise(stats: dict[str, torch.Tensor], n_texts: int) -> dict:
    counts, mass = stats["counts"], stats["mass"]
    vocab = int(counts.numel())
    live = {f"active_in_ge_{k}_texts": int((counts >= k).sum()) for k in (1, 2, 5, 10, 50, 100)}
    sorted_mass = torch.sort(mass, descending=True).values
    total = float(sorted_mass.sum())
    cumulative = torch.cumsum(sorted_mass, dim=0) / max(total, 1e-12)
    coverage = {f"top_{k}_dims_mass": float(cumulative[min(k, vocab) - 1]) for k in (5000, 10000, 15000, 20000, 25000, 30000, 40000)}
    dims_for = {f"dims_for_{int(f * 100)}pct_mass": int((cumulative < f).sum()) + 1 for f in (0.9, 0.95, 0.99, 0.999)}
    per_text = stats["active_per_text"].float()
    return {
        "vocab_size": vocab,
        "n_texts": n_texts,
        "dead_dims": int((counts == 0).sum()),
        **live,
        **coverage,
        **dims_for,
        "active_dims_per_text": {"mean": float(per_text.mean()), "p50": float(per_text.median()), "p90": float(per_text.quantile(0.9))},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--texts", required=True, type=Path, help="JSON lines with a text field (and optionally kind)")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--kinds", default=None, help="comma-separated kinds to keep (default: all)")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--quantization-factor", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--per-dim", action="store_true", help="also write the per-dimension counts, mass and peak")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from distill.fidelity import load_texts

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    kinds = set(args.kinds.split(",")) if args.kinds else None
    texts, text_kinds = load_texts(args.texts, args.text_field, args.n, args.seed, kinds)
    if not texts:
        print("[ERROR] no texts loaded", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    started = time.time()
    stats = usage_counts(model, tokenizer, texts, args.max_seq_length, args.batch_size, args.device, args.quantization_factor)
    result = {
        "model": str(args.model),
        "texts": str(args.texts),
        "kinds": sorted(set(text_kinds)),
        "max_seq_length": args.max_seq_length,
        "quantization_factor": args.quantization_factor,
        "seconds": round(time.time() - started, 1),
        **summarise(stats, len(texts)),
    }
    if args.per_dim:
        result["per_dim"] = {
            "counts": stats["counts"].tolist(),
            "mass": [round(x, 4) for x in stats["mass"].tolist()],
            "peak": [round(x, 4) for x in stats["peak"].tolist()],
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_dim"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
