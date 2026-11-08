#!/usr/bin/env python
"""Mined candidate pool for training a static query encoder by distillation from the symmetric SPLADE.

``distill.static_pool`` builds a pool from labelled training rows: a few candidates per query, mostly easy, so the
static table starts near the ceiling of every metric and its training has nothing to learn. This module builds the
pool the way a retriever is distilled instead:

1. a seeded sample of passages and of queries from an unlabelled corpus (``kd_v1``: ``{"text", "kind"}`` lines),
   read in two streaming passes so the corpus is never held in memory;
2. the frozen document model encodes the passages (the index side) and the queries (the teacher's query side), both
   quantized as the PIRB index stores them (``round(w * qf)``);
3. every query is searched on the GPU over the passage sample with the teacher's query vector and with the query
   vector of each ``--student`` table (exact sparse x dense product, no index): the candidates are the teacher's
   top ``--teacher-top`` plus each student's top ``--student-top`` (the student's own false positives are the
   negatives it has to learn to push down), sorted by the teacher score and capped at ``--max-candidates``;
4. the target of each candidate is the teacher score ``q_splade . d``; no labels are used.

The output is a ``distill.static_pool`` folder (``rows.jsonl`` with ``n_pos = 1`` = the teacher's top passage,
``scores`` = teacher scores, ``docs.safetensors``, ``queries.safetensors``), so ``distill.train_static --target rows``
trains on it unchanged. ``pool.json`` also records, per student, the recall of the teacher's top 10 inside the
student's top ``--student-top`` on the sample (the gap training has to close).

Usage:
  .venv/bin/python -m distill.static_mine --doc-model models/arm-dense-kl-n5kj-ens400 --corpus datasets/kd/kd_v1.jsonl \\
      --n-docs 300000 --n-queries 120000 --student models/static/idf --student models/static/credit-k32 \\
      --out cache/static_pools/kd-n5kj --device cuda --dtype fp16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

from distill.static_pool import DOCS_NAME, POOL_META, QUERIES_NAME, ROWS_NAME, SparseMatrix, encode_texts

QUERY_MARK = '"kind": "query"'
PASSAGE_MARK = '"kind": "passage"'


def kind_of_line(line: str) -> str | None:
    """The corpus kind of one JSON line without parsing it (quotes inside a JSON string are escaped, so the marks
    can only match the ``kind`` field itself)."""
    if QUERY_MARK in line:
        return "query"
    if PASSAGE_MARK in line:
        return "passage"
    return None


def sample_corpus(path: Path, n_docs: int, n_queries: int, seed: int = 42, min_chars: int = 1) -> tuple[list[str], list[str]]:
    """Two streaming passes: count the passage and query lines, then read the seeded sample of each kind.
    ``n = 0`` keeps every line of that kind."""
    counts = {"passage": 0, "query": 0}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            kind = kind_of_line(line)
            if kind:
                counts[kind] += 1
    rng = random.Random(seed)
    want = {"passage": n_docs, "query": n_queries}
    chosen = {
        kind: set(range(counts[kind])) if not want[kind] or want[kind] >= counts[kind] else set(rng.sample(range(counts[kind]), want[kind]))
        for kind in counts
    }
    seen = {"passage": 0, "query": 0}
    out: dict[str, list[str]] = {"passage": [], "query": []}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            kind = kind_of_line(line)
            if not kind:
                continue
            i = seen[kind]
            seen[kind] += 1
            if i in chosen[kind]:
                text = json.loads(line).get("text")
                if isinstance(text, str) and len(text) >= min_chars:
                    out[kind].append(text)
    return out["passage"], out["query"]


def csr_on_device(mat: SparseMatrix, device: str) -> torch.Tensor:
    """``[N, dim]`` float32 sparse CSR tensor of a quantized pool matrix (weights = stored ints / scale)."""
    return torch.sparse_csr_tensor(
        mat.indptr.to(device), mat.indices.to(device).long(), mat.values.to(device).float() / mat.scale, size=(len(mat), mat.dim)
    )


def merge_candidates(teacher_scores: torch.Tensor, teacher_ids: torch.Tensor, student_ids: list[torch.Tensor], max_candidates: int) -> tuple[list[int], list[float]]:
    """Union of the teacher's and the students' top ids of one query, sorted by the teacher score, capped.
    ``teacher_scores`` is the query's full ``[N]`` score column."""
    ids = torch.unique(torch.cat([teacher_ids] + student_ids))
    scores = teacher_scores[ids]
    order = torch.argsort(scores, descending=True)[:max_candidates]
    return ids[order].tolist(), [round(float(x), 4) for x in scores[order]]


@torch.no_grad()
def mine(
    docs: SparseMatrix,
    teacher_queries: SparseMatrix,
    queries: list[str],
    students: list[tuple[str, Any]],
    tokenizer,
    teacher_top: int,
    student_top: int,
    max_candidates: int,
    batch_size: int,
    device: str,
    rows_out: Path,
    log_every: int = 50,
) -> dict[str, Any]:
    """Writes one row ``{"query", "docs", "n_pos": 1, "scores"}`` per query to ``rows_out`` (streamed, the rows are
    never held in memory) and returns the counts and recall diagnostics."""
    d = csr_on_device(docs, device)
    qf = docs.scale
    n = len(docs)
    t_top, s_top = min(teacher_top, n), min(student_top, n)
    n_rows = n_cands = student_only = recall_total = 0
    recall_hits = {name: 0 for name, _ in students}
    started = time.time()
    with Path(rows_out).open("w", encoding="utf-8") as handle:
        for n_batch, start in enumerate(range(0, len(queries), batch_size)):
            ids = list(range(start, min(start + batch_size, len(queries))))
            texts = [queries[i] for i in ids]
            cols = [teacher_queries.dense_rows(ids, device)]
            for _, model in students:
                enc = model.tokenize(tokenizer, texts, device)
                q = model(enc["input_ids"], enc["attention_mask"])
                cols.append(torch.round(q * qf) / qf)  # the quantized regime quantizes the query side too
            # a contiguous [dim, C] right-hand side: the transposed view costs cuSPARSE ~20x (4.7 s vs 0.21 s for
            # 300 k passages x 384 columns on the 3060 Ti)
            q_all = torch.cat(cols, dim=0).T.contiguous()
            scores = d @ q_all  # [N, (1 + S) * B]
            b = len(ids)
            t_ids = torch.topk(scores[:, :b], t_top, dim=0).indices.T  # [B, t_top]
            s_ids = [torch.topk(scores[:, (k + 1) * b : (k + 2) * b], s_top, dim=0).indices.T for k in range(len(students))]
            t_cols = scores[:, :b].T.contiguous()  # [B, N] teacher scores
            for j, qi in enumerate(ids):
                cand, cand_scores = merge_candidates(t_cols[j], t_ids[j], [s[j] for s in s_ids], max_candidates)
                teacher_set = set(t_ids[j].tolist())
                top10 = set(t_ids[j, : min(10, t_top)].tolist())
                recall_total += len(top10)
                for (name, _), s in zip(students, s_ids):
                    recall_hits[name] += len(top10 & set(s[j].tolist()))
                student_only += sum(c not in teacher_set for c in cand)
                n_rows += 1
                n_cands += len(cand)
                handle.write(json.dumps({"query": queries[qi], "docs": cand, "n_pos": 1, "scores": cand_scores}, ensure_ascii=False) + "\n")
            del scores, t_cols, q_all
            if log_every and n_batch % log_every == 0:
                done = ids[-1] + 1
                print(f"[mine] {done}/{len(queries)} queries, {done / max(time.time() - started, 1e-6):.1f}/s", flush=True)
    return {
        "rows": n_rows,
        "teacher_top": t_top,
        "student_top": s_top,
        "max_candidates": max_candidates,
        "candidates_mean": round(n_cands / max(n_rows, 1), 2),
        "student_only_candidates_mean": round(student_only / max(n_rows, 1), 2),
        "student_recall_of_teacher_top10": {name: round(hits / max(recall_total, 1), 4) for name, hits in recall_hits.items()},
        "search_seconds": round(time.time() - started, 1),
    }


def release_host_memory() -> None:
    """Return freed heap pages to the OS (the 16 GB box hangs instead of OOM-killing; glibc keeps freed arenas)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc-model", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path, help="JSON lines with text + kind (query / passage)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--student", action="append", type=Path, default=[], help="static table whose top candidates join the pool (repeatable)")
    parser.add_argument("--n-docs", type=int, default=300000)
    parser.add_argument("--n-queries", type=int, default=120000)
    parser.add_argument("--teacher-top", type=int, default=32)
    parser.add_argument("--student-top", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=512, help="document length (the PIRB fork indexes at 512)")
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--quantization-factor", type=int, default=100)
    parser.add_argument("--min-weight", type=float, default=0.0, help="PIRB_MIN_WEIGHT of the index this pool imitates")
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16, help="document encoding batch")
    parser.add_argument("--search-batch", type=int, default=128, help="queries per search step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args(argv)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from distill.static_query import StaticSparseQueryEncoder

    started = time.time()
    passages, queries = sample_corpus(args.corpus, args.n_docs, args.n_queries, args.seed, args.min_chars)
    passages = list(dict.fromkeys(passages))
    queries = list(dict.fromkeys(queries))
    if not passages or not queries:
        print("[ERROR] the corpus sample has no passages or no queries", file=sys.stderr)
        return 2
    print(f"[mine] {len(passages)} passages, {len(queries)} queries sampled in {time.time() - started:.0f}s", flush=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.doc_model)
    model = AutoModelForMaskedLM.from_pretrained(args.doc_model, torch_dtype=dtype).to(args.device).eval()
    t = time.time()
    doc_mat = encode_texts(model, tokenizer, passages, args.max_seq_length, args.batch_size, args.device, args.quantization_factor, args.min_weight, args.log_every)
    encode_docs_s = time.time() - t
    t = time.time()
    query_mat = encode_texts(model, tokenizer, queries, args.query_max_length, max(args.batch_size, 64), args.device, args.quantization_factor, args.min_weight, args.log_every)
    encode_queries_s = time.time() - t
    del model, passages
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    release_host_memory()
    students = []
    for path in args.student:
        table = StaticSparseQueryEncoder.load(path, args.device)
        if table.output_dim != doc_mat.dim:
            print(f"[ERROR] {path} writes {table.output_dim} dims, the documents have {doc_mat.dim}", file=sys.stderr)
            return 2
        students.append((path.name, table))
    from safetensors.torch import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / POOL_META).unlink(missing_ok=True)  # pool.json is written last: its presence marks a complete pool
    save_file(doc_mat.tensors(), str(args.out / DOCS_NAME))
    save_file(query_mat.tensors(), str(args.out / QUERIES_NAME))
    info = mine(doc_mat, query_mat, queries, students, tokenizer, args.teacher_top, args.student_top, args.max_candidates, args.search_batch,
                args.device, args.out / ROWS_NAME, max(args.log_every // 10, 1))
    meta = {
        "kind": "mined",
        "doc_model": str(args.doc_model),
        "corpus": str(args.corpus),
        "students": [str(p) for p in args.student],
        "rows_with_scores": info["rows"],
        "docs": len(doc_mat),
        "dim": doc_mat.dim,
        "quantization_factor": args.quantization_factor,
        "min_weight": args.min_weight,
        "max_seq_length": args.max_seq_length,
        "query_max_length": args.query_max_length,
        "teacher_queries": True,
        "doc_dims_mean": round(float(doc_mat.nnz_per_row().float().mean()), 1),
        "query_dims_mean": round(float(query_mat.nnz_per_row().float().mean()), 1),
        "dtype": args.dtype,
        "seed": args.seed,
        "encode_docs_seconds": round(encode_docs_s, 1),
        "encode_queries_seconds": round(encode_queries_s, 1),
        **info,
        "seconds": round(time.time() - started, 1),
    }
    (args.out / POOL_META).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
