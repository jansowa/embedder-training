#!/usr/bin/env python
"""Candidate pool for training a static query encoder against a frozen SPLADE document side.

Reads training rows (``query``, ``pos``, ``neg``, optional ``pos_scores`` / ``neg_scores`` = teacher scores, as in the
``-tens400`` arm files), deduplicates the passages, encodes them once with the document model and stores the vectors
quantized exactly as the PIRB index stores them (``round(w * 100)``, zeros dropped). With ``--teacher-queries`` the
same model also encodes the queries (the symmetric SPLADE score ``q_splade . d`` is then available as a target and as
the reference the static encoder is measured against). Everything the static trainer needs then fits in memory and
no transformer runs during its training.

Output folder:
  pool.json            meta (models, files, counts, quantization)
  rows.jsonl           {"query", "docs": [doc ids, positives first], "n_pos", "scores": [row teacher scores] | null}
  docs.safetensors     CSR of the document vectors: indptr int64 [N + 1], indices int32, values int16 (= round(w * qf))
  queries.safetensors  the same for the teacher's query vectors, one per row (only with --teacher-queries)

Usage:
  .venv/bin/python -m distill.static_pool --doc-model models/arm-dense-kl-n5kj-ens400 \\
      --rows datasets/arms/n5-plus-kj-tens400.jsonl --out cache/static_pools/n5kj --teacher-queries \\
      --max-seq-length 512 --batch-size 16 --device cuda --dtype fp16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

POOL_META = "pool.json"
ROWS_NAME = "rows.jsonl"
DOCS_NAME = "docs.safetensors"
QUERIES_NAME = "queries.safetensors"


@dataclass
class SparseMatrix:
    """Rows of quantized sparse vectors in CSR form; ``scale`` turns the stored ints back into weights."""

    indptr: torch.Tensor  # int64 [N + 1]
    indices: torch.Tensor  # int32 [nnz]
    values: torch.Tensor  # int16 [nnz]
    dim: int
    scale: float

    def __len__(self) -> int:
        return self.indptr.shape[0] - 1

    def nnz_per_row(self) -> torch.Tensor:
        return self.indptr[1:] - self.indptr[:-1]

    def dense_rows(self, rows: torch.Tensor | list[int], device: torch.device | str = "cpu") -> torch.Tensor:
        """``[len(rows), dim]`` float weights of the given rows."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        starts, ends = self.indptr[rows], self.indptr[rows + 1]
        lengths = ends - starts
        out = torch.zeros(len(rows), self.dim, device=device)
        if int(lengths.sum()) == 0:
            return out
        row_of = torch.repeat_interleave(torch.arange(len(rows)), lengths)
        offsets = torch.arange(int(lengths.sum())) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        flat = torch.repeat_interleave(starts, lengths) + offsets
        out[row_of.to(device), self.indices[flat].long().to(device)] = self.values[flat].float().to(device) / self.scale
        return out

    def document_frequency(self) -> torch.Tensor:
        return torch.bincount(self.indices.long(), minlength=self.dim)

    def tensors(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}indptr": self.indptr, f"{prefix}indices": self.indices, f"{prefix}values": self.values}


def quantize_rows(vectors: torch.Tensor, quantization_factor: int, min_weight: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per row ``(indices int32, values int16)`` of ``round(w * qf)`` > 0, as PIRB's ``encode_document_vector``."""
    data = vectors.float().cpu().numpy()
    if min_weight > 0:
        data = np.where(data >= min_weight, data, 0.0)
    q = np.rint(data * quantization_factor)
    out = []
    for row in q:
        idx = np.nonzero(row > 0)[0]
        out.append((idx.astype(np.int32), np.minimum(row[idx], np.iinfo(np.int16).max).astype(np.int16)))
    return out


def to_csr(rows: list[tuple[np.ndarray, np.ndarray]], dim: int, scale: float) -> SparseMatrix:
    lengths = np.array([len(i) for i, _ in rows], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    indices = np.concatenate([i for i, _ in rows]) if rows else np.zeros(0, np.int32)
    values = np.concatenate([v for _, v in rows]) if rows else np.zeros(0, np.int16)
    return SparseMatrix(torch.from_numpy(indptr), torch.from_numpy(indices.astype(np.int32)), torch.from_numpy(values.astype(np.int16)), dim, scale)


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], max_length: int, batch_size: int, device: str, quantization_factor: int, min_weight: float, log_every: int = 0) -> SparseMatrix:
    from distill.splade import encode_pooled

    dim = int(model.get_output_embeddings().weight.shape[0])
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    rows: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(texts)
    started = time.time()
    for n_batch, start in enumerate(range(0, len(order), batch_size)):
        idx = order[start : start + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        vec = encode_pooled(model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for i, row in zip(idx, quantize_rows(vec, quantization_factor, min_weight)):
            rows[i] = row
        if log_every and n_batch % log_every == 0:
            done = start + len(idx)
            print(f"[pool] {done}/{len(texts)} texts, {done / max(time.time() - started, 1e-6):.1f}/s", flush=True)
    return to_csr([r for r in rows if r is not None], dim, float(quantization_factor))


def read_rows(path: Path, max_rows: int = 0, max_neg: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows with doc ids into a deduplicated passage list (positives first, then negatives)."""
    doc_id: dict[str, int] = {}
    docs: list[str] = []
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query, pos, neg = row.get("query"), list(row.get("pos") or []), list(row.get("neg") or [])
            if not isinstance(query, str) or not query or not pos:
                continue
            pos_scores, neg_scores = row.get("pos_scores"), row.get("neg_scores")
            scores = None
            if isinstance(pos_scores, list) and isinstance(neg_scores, list) and len(pos_scores) == len(pos) and len(neg_scores) == len(neg):
                scores = [float(x) for x in pos_scores] + [float(x) for x in neg_scores]
            if max_neg and len(neg) > max_neg:
                neg = neg[:max_neg]
                if scores is not None:
                    scores = scores[: len(pos) + max_neg]
            ids = []
            for text in pos + neg:
                if text not in doc_id:
                    doc_id[text] = len(docs)
                    docs.append(text)
                ids.append(doc_id[text])
            rows.append({"query": query, "docs": ids, "n_pos": len(pos), "scores": scores})
            if max_rows and len(rows) >= max_rows:
                break
    return rows, docs


@dataclass
class Pool:
    meta: dict[str, Any]
    rows: list[dict[str, Any]]
    docs: SparseMatrix
    queries: SparseMatrix | None


def save_pool(out: Path, meta: dict[str, Any], rows: list[dict[str, Any]], docs: SparseMatrix, queries: SparseMatrix | None) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    save_file(docs.tensors(), str(out / DOCS_NAME))
    if queries is not None:
        save_file(queries.tensors(), str(out / QUERIES_NAME))
    with (out / ROWS_NAME).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / POOL_META).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_pool(path: Path) -> Pool:
    from safetensors.torch import load_file

    path = Path(path)
    meta = json.loads((path / POOL_META).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / ROWS_NAME).open(encoding="utf-8")]
    t = load_file(str(path / DOCS_NAME))
    scale = float(meta["quantization_factor"])
    docs = SparseMatrix(t["indptr"], t["indices"], t["values"], int(meta["dim"]), scale)
    queries = None
    if (path / QUERIES_NAME).exists():
        q = load_file(str(path / QUERIES_NAME))
        queries = SparseMatrix(q["indptr"], q["indices"], q["values"], int(meta["dim"]), scale)
    return Pool(meta, rows, docs, queries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc-model", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--teacher-queries", action="store_true", help="also encode the queries with the document model")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-neg", type=int, default=0, help="keep at most this many negatives per row (0 = all)")
    parser.add_argument("--max-seq-length", type=int, default=512, help="document length (the PIRB fork indexes at 512)")
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--quantization-factor", type=int, default=100)
    parser.add_argument("--min-weight", type=float, default=0.0, help="PIRB_MIN_WEIGHT of the index this pool imitates")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args(argv)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    started = time.time()
    rows, docs = read_rows(args.rows, args.max_rows, args.max_neg)
    if not rows:
        print("[ERROR] no usable rows", file=sys.stderr)
        return 2
    print(f"[pool] {len(rows)} rows, {len(docs)} unique passages", flush=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.doc_model)
    model = AutoModelForMaskedLM.from_pretrained(args.doc_model, torch_dtype=dtype).to(args.device).eval()
    doc_mat = encode_texts(model, tokenizer, docs, args.max_seq_length, args.batch_size, args.device, args.quantization_factor, args.min_weight, args.log_every)
    query_mat = None
    if args.teacher_queries:
        query_mat = encode_texts(model, tokenizer, [r["query"] for r in rows], args.query_max_length, max(args.batch_size, 64), args.device, args.quantization_factor, args.min_weight, args.log_every)
    meta = {
        "doc_model": str(args.doc_model),
        "rows_file": str(args.rows),
        "rows": len(rows),
        "rows_with_scores": sum(r["scores"] is not None for r in rows),
        "docs": len(docs),
        "dim": doc_mat.dim,
        "quantization_factor": args.quantization_factor,
        "min_weight": args.min_weight,
        "max_seq_length": args.max_seq_length,
        "query_max_length": args.query_max_length,
        "teacher_queries": bool(args.teacher_queries),
        "doc_dims_mean": round(float(doc_mat.nnz_per_row().float().mean()), 1),
        "query_dims_mean": round(float(query_mat.nnz_per_row().float().mean()), 1) if query_mat is not None else None,
        "dtype": args.dtype,
        "seconds": round(time.time() - started, 1),
    }
    save_pool(args.out, meta, rows, doc_mat, query_mat)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
