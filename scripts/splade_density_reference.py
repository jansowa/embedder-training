#!/usr/bin/env python
"""Calibrate SPLADE density measures on a sample of training texts.

For a model and a few FlagEmbedding-style JSONL files ({"query", "pos": [...], "neg": [...]}) the script
samples documents (positives + negatives, evenly across files) and as many queries (seed 42), encodes
them with ``SparseEncoder`` and reports, for documents and queries, the mean / median / p10 / p90 of

* the raw L0, ``(emb != 0).sum()``: what ``SpladeLoss(document_regularizer_threshold=...)`` counts, and
* the quantized L0, ``round(w * qf) > 0``: what PIRB, ``compare_splade_sparsity.py`` and the
  ``regularizer_control`` density controller count,

plus a least-squares fit raw ~ a * quantized + b over the documents and the raw threshold matching
``--threshold-at`` quantized dimensions. Example:

    python scripts/splade_density_reference.py --model models/base-splade-wrapped \
        --jsonl datasets/judged/msmarco-clean-20000.jsonl datasets/arms/fiqa-2073.jsonl \
        --sample 256 --max-seq-length 256 --threshold-at 200 \
        --output-json experiments/polish-splade-2026-10/results/data/density-reference-base.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Local SparseEncoder directory or Hugging Face model id.")
    parser.add_argument("--jsonl", nargs="+", required=True, type=Path, help="Training JSONL files.")
    parser.add_argument("--sample", type=int, default=256, help="Documents (and queries) to encode. Default: 256.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Default: the model's own length.")
    parser.add_argument("--quantization-factor", type=int, default=100)
    parser.add_argument("--threshold-at", type=float, default=200, help="Quantized dims to translate into a raw L0.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fp32", action="store_true", help="Encode in fp32 on CUDA (default: fp16 on CUDA).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.sample <= 0 or args.batch_size <= 0 or args.quantization_factor <= 0:
        parser.error("--sample, --batch-size and --quantization-factor must be positive.")
    return args


def _quotas(total: int, parts: int) -> list[int]:
    return [total // parts + (1 if idx < total % parts else 0) for idx in range(parts)]


def sample_texts(paths: list[Path], sample: int, seed: int) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    """Up to ``sample`` documents and ``sample`` queries, split evenly over the files."""
    rng = random.Random(seed)
    documents: list[str] = []
    queries: list[str] = []
    per_file: dict[str, dict[str, int]] = {}
    for path, quota in zip(paths, _quotas(sample, len(paths))):
        file_documents: set[str] = set()
        file_queries: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("query"):
                    file_queries.add(row["query"])
                for key in ("pos", "neg"):
                    values = row.get(key) or []
                    file_documents.update(value for value in ([values] if isinstance(values, str) else values) if value)
        picked_documents = rng.sample(sorted(file_documents), min(quota, len(file_documents)))
        picked_queries = rng.sample(sorted(file_queries), min(quota, len(file_queries)))
        documents.extend(picked_documents)
        queries.extend(picked_queries)
        per_file[str(path)] = {
            "documents_available": len(file_documents),
            "queries_available": len(file_queries),
            "documents": len(picked_documents),
            "queries": len(picked_queries),
        }
    return documents, queries, per_file


def _percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": _percentile(ordered, 0.5),
        "p10": _percentile(ordered, 0.1),
        "p90": _percentile(ordered, 0.9),
    }


def linear_fit(x: list[int], y: list[int]) -> tuple[float, float]:
    """Least squares y ~ a * x + b."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((value - mean_x) ** 2 for value in x)
    if sxx == 0:
        return 0.0, mean_y
    slope = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x, y)) / sxx
    return slope, mean_y - slope * mean_x


def count_dims(model, texts: list[str], encoder_name: str, batch_size: int, quantization_factor: int):
    import torch

    raw: list[int] = []
    quantized: list[int] = []
    encoder = getattr(model, encoder_name)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = encoder(
            batch, batch_size=len(batch), show_progress_bar=False, convert_to_tensor=True, convert_to_sparse_tensor=False
        )
        if isinstance(embeddings, list):
            embeddings = torch.stack([value.to_dense() if value.is_sparse else value for value in embeddings])
        elif embeddings.is_sparse:
            embeddings = embeddings.to_dense()
        raw.extend((embeddings != 0).sum(dim=-1).cpu().tolist())
        quantized.extend((torch.round(embeddings.float() * quantization_factor) > 0).sum(dim=-1).cpu().tolist())
        del embeddings
    return [int(v) for v in raw], [int(v) for v in quantized]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    import torch
    from sentence_transformers.sparse_encoder import SparseEncoder

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" and not args.fp32 else {}
    documents, queries, per_file = sample_texts(args.jsonl, args.sample, args.seed)
    if not documents:
        print("No documents found in the JSONL files.", file=sys.stderr)
        return 1
    model = SparseEncoder(args.model, device=device, model_kwargs=model_kwargs)
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length
    model.eval()
    with torch.inference_mode():
        doc_raw, doc_quantized = count_dims(model, documents, "encode_document", args.batch_size, args.quantization_factor)
        query_raw, query_quantized = count_dims(model, queries, "encode_query", args.batch_size, args.quantization_factor)
    slope, intercept = linear_fit(doc_quantized, doc_raw)
    result = {
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": {key: [str(p) for p in value] if key == "jsonl" else str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "device": device,
        "dtype": "float16" if model_kwargs else "float32",
        "max_seq_length": model.max_seq_length,
        "files": per_file,
        "documents": {"raw_l0": summarize(doc_raw), "quantized_l0": summarize(doc_quantized)},
        "queries": {"raw_l0": summarize(query_raw), "quantized_l0": summarize(query_quantized)},
        "fit_documents_raw_vs_quantized": {"a": slope, "b": intercept},
        "threshold_at_quantized": args.threshold_at,
        "recommended_document_regularizer_threshold": int(round(slope * args.threshold_at + intercept)),
    }
    print(f"model {args.model} on {device} ({result['dtype']}), max_seq_length {model.max_seq_length}")
    print(f"{'texts':<10} {'measure':<10} {'n':>5} {'mean':>9} {'median':>9} {'p10':>9} {'p90':>9}")
    for kind in ("documents", "queries"):
        for measure in ("raw_l0", "quantized_l0"):
            stats = result[kind][measure]
            if not stats.get("n"):
                continue
            print(
                f"{kind:<10} {measure.split('_')[0]:<10} {stats['n']:>5} {stats['mean']:>9.1f} "
                f"{stats['median']:>9.1f} {stats['p10']:>9.1f} {stats['p90']:>9.1f}"
            )
    print(
        f"fit (documents): raw = {slope:.3f} * quantized + {intercept:.1f}; "
        f"recommended document_regularizer_threshold for {args.threshold_at:g} dims: "
        f"{result['recommended_document_regularizer_threshold']}"
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
