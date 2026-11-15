#!/usr/bin/env python
"""Index cost of an asymmetric pair on the cost sample of splade_index_cost.py: the passages are encoded by one SPLADE
model (the index), the queries by another query side: a static query table (a directory with static_query.json, see
distill.static_query) or a second SPLADE model. Same sample, same activity rule (round(w * 100) > 0) and same numbers as
splade_index_cost.py, so postings/query is comparable with its output. The symmetric numbers of the document model are
written next to each pair as a check against earlier reports.

  python scripts/splade_index_cost_asym.py --doc-model full=models/soup-v20-dqf-bias015 \
      --query static=models/static-soup-v20-dqf-bias015/phr2-k8-kd-r2 --output-json OUT.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from splade_index_cost import POLEVAL, TINY, read_jsonl  # noqa: E402


def sample(data_dir: Path, per_task: int, seed: int, tasks: list[str] | None = None) -> tuple[list[str], list[str], list[str]]:
    """The passages and queries of splade_index_cost.py (same order of the draws from one generator) and the task of
    every query. With `tasks` (PIRB data directories; poleval-2022/NAME reads test-A and test-B) another task set is
    sampled the same way; its postings are not comparable with the standard sample (another index)."""
    rng = random.Random(seed)
    docs, queries, qtask = [], [], []

    def plain(t: str) -> None:
        nonlocal docs, queries, qtask
        d = data_dir / t
        docs += read_jsonl(d / "passages/passages.jsonl", per_task, rng)
        q = read_jsonl(d / "queries/queries.jsonl", per_task, rng)
        queries += q
        qtask += [t] * len(q)

    def poleval(t: str) -> None:
        nonlocal docs, queries, qtask
        d = data_dir / "poleval-2022" / t
        docs += read_jsonl(d / "passages/passages.jsonl", per_task, rng)
        for split in ("test-A", "test-B"):
            q = read_jsonl(d / split / "queries.jsonl", per_task // 2, rng)
            queries += q
            qtask += [f"poleval-2022-{split}-{t}"] * len(q)

    if tasks is None:
        for t in TINY:
            plain(t)
        for t in POLEVAL:
            poleval(t)
    else:
        for t in tasks:
            poleval(t.split("/", 1)[1]) if t.startswith("poleval-2022/") else plain(t)
    return docs, queries, qtask


def doc_stats(D: np.ndarray, qf: int) -> tuple[np.ndarray, float]:
    """Document frequency of every term and the mean active dims per passage."""
    Da = np.rint(D * qf) > 0
    return Da.sum(axis=0), float(Da.sum(axis=1).mean())


def profile(df: np.ndarray, doc_active: float, Q: np.ndarray, qf: int, qtask: list[str] | None = None) -> dict:
    Qa = np.rint(Q * qf) > 0
    postings = np.array([int(df[row].sum()) for row in Qa])  # per query: sum over its active terms of df
    active = Qa.sum(axis=1)
    rep = {
        "doc_active_mean": doc_active,
        "query_active_mean": float(active.mean()),
        "query_active_median": float(np.median(active)),
        "postings_per_query": float(postings.mean()),
        "postings_per_query_median": float(np.median(postings)),
        "postings_per_query_p95": float(np.percentile(postings, 95)),
        "query_weight_sum_mean": float(Q.sum(axis=1).mean()),
    }
    if qtask:
        tasks = np.array(qtask)
        rep["per_task"] = {
            t: {"n": int((tasks == t).sum()), "query_active_mean": float(active[tasks == t].mean()), "postings_per_query": float(postings[tasks == t].mean())}
            for t in dict.fromkeys(qtask)
        }
        rest = tasks != "arguana-pl"
        if rest.any() and not rest.all():
            rep["without_arguana"] = {"n": int(rest.sum()), "query_active_mean": float(active[rest].mean()), "postings_per_query": float(postings[rest].mean())}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-model", required=True, help="LABEL=PATH of the SPLADE model that builds the index")
    ap.add_argument("--query", action="append", required=True, help="LABEL=PATH of a query side (static table dir or SPLADE model), repeatable")
    ap.add_argument("--data-dir", type=Path, default=Path("third_party/pirb/data"))
    ap.add_argument("--per-task", type=int, default=150)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--quantization-factor", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tasks", default="", help="comma list of PIRB data directories instead of the standard cost sample (poleval-2022/NAME for a PolEval set)")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    tasks = [t for t in args.tasks.split(",") if t] or None
    docs, queries, qtask = sample(args.data_dir, args.per_task, args.seed, tasks)
    print(f"sample ({'standard cost sample' if tasks is None else ', '.join(tasks)}): {len(docs)} passages, {len(queries)} queries", flush=True)
    qf = args.quantization_factor

    from sentence_transformers.sparse_encoder import SparseEncoder

    def encode_splade(path: str, texts: list[str], batch: int) -> np.ndarray:
        model = SparseEncoder(path, device=args.device)
        model.max_seq_length = args.max_seq_length
        with torch.no_grad():
            out = model.encode(texts, batch_size=batch, convert_to_sparse_tensor=False, convert_to_tensor=True).float().cpu().numpy()
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return out

    doc_label, doc_path = args.doc_model.split("=", 1)
    model = SparseEncoder(doc_path, device=args.device)
    model.max_seq_length = args.max_seq_length
    with torch.no_grad():
        D = model.encode(docs, batch_size=args.batch_size, convert_to_sparse_tensor=False, convert_to_tensor=True).float().cpu().numpy()
        Q_sym = model.encode(queries, batch_size=args.batch_size * 4, convert_to_sparse_tensor=False, convert_to_tensor=True).float().cpu().numpy()
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    df, doc_active = doc_stats(D, qf)
    dim = D.shape[1]
    del D
    report = {"index": doc_label, "index_path": doc_path, "n_docs": len(docs), "n_queries": len(queries), "tasks": tasks or "standard cost sample",
              "symmetric": profile(df, doc_active, Q_sym, qf, qtask), "pairs": {}}
    s = report["symmetric"]
    print(f"{doc_label} (symmetric): dims/doc {s['doc_active_mean']:.1f}, dims/query {s['query_active_mean']:.1f}, postings/query {s['postings_per_query']:.0f}", flush=True)
    del Q_sym

    for spec in args.query:
        label, path = spec.split("=", 1)
        if (Path(path) / "static_query.json").exists():
            from transformers import AutoTokenizer

            from distill.static_query import StaticSparseQueryEncoder

            table = StaticSparseQueryEncoder.load(Path(path))
            Q = table.encode(AutoTokenizer.from_pretrained(path), queries).float().numpy()
            kind = "static table"
            del table
        else:
            Q = encode_splade(path, queries, args.batch_size * 4)
            kind = "SPLADE model"
        if Q.shape[1] != dim:
            raise SystemExit(f"{label}: query dimension {Q.shape[1]} != index dimension {dim}")
        rep = profile(df, doc_active, Q, qf, qtask)
        rep.update({"query_path": path, "query_kind": kind})
        report["pairs"][label] = rep
        print(f"{label} ({kind}) over {doc_label}: dims/query {rep['query_active_mean']:.1f} (median {rep['query_active_median']:.0f}), "
              f"postings/query {rep['postings_per_query']:.0f} (median {rep['postings_per_query_median']:.0f}, p95 {rep['postings_per_query_p95']:.0f})", flush=True)
        if "without_arguana" in rep:
            a, w = rep["per_task"]["arguana-pl"], rep["without_arguana"]
            print(f"   arguana-pl: dims/query {a['query_active_mean']:.1f}, postings/query {a['postings_per_query']:.0f}; "
                  f"the other tasks: dims/query {w['query_active_mean']:.1f}, postings/query {w['postings_per_query']:.0f}", flush=True)
        del Q
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
