#!/usr/bin/env python
"""Index-size and query-cost profile of SPLADE models on the same PIRB sample.

For every model, encodes the same passages and queries (sampled from the tiny
tasks) and reports, with the usual quantisation (round(w*100) > 0 = active):

  doc_active     mean active dims per passage      -> index size (postings)
  query_active   mean active dims per query        -> posting lists visited
  postings/query mean over queries of sum_j 1[q_j active] * df_j, where df_j is
                 the number of sampled passages with term j active
                 -> how many (term, passage) pairs a query scores; the cost
                    that actually grows when both sides expand
  flops          sum_j mean_d(w_dj) * mean_q(w_qj), the SPLADE FLOPS regulariser

All numbers are relative to the same sample, so they are comparable across
models but not with numbers from other scripts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

TINY = ["maupqa-gpt3-cc", "maupqa-mkqa", "maupqa-poleval2021-pairs", "maupqa-templates",
        "arguana-pl", "nfcorpus-pl", "scifact-pl", "gpt-exams"]
POLEVAL = ["allegro-faq"]  # test-A and test-B share one passages/queries dir


def read_jsonl(path: Path, n: int, rng: random.Random, fields=("contents", "text", "query")) -> list[str]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                for f in fields:
                    if r.get(f):
                        rows.append(str(r[f])); break
    rng.shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="LABEL=PATH, repeatable")
    ap.add_argument("--data-dir", type=Path, default=Path("third_party/pirb/data"))
    ap.add_argument("--per-task", type=int, default=150)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--quantization-factor", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-json", type=Path, default=Path("results/index_cost.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs, queries = [], []
    for t in TINY:
        d = args.data_dir / t
        docs += read_jsonl(d / "passages/passages.jsonl", args.per_task, rng)
        queries += read_jsonl(d / "queries/queries.jsonl", args.per_task, rng)
    for t in POLEVAL:
        d = args.data_dir / "poleval-2022" / t
        docs += read_jsonl(d / "passages/passages.jsonl", args.per_task, rng)
        for split in ("test-A", "test-B"):
            queries += read_jsonl(d / split / "queries.jsonl", args.per_task // 2, rng)
    print(f"sample: {len(docs)} passages, {len(queries)} queries")

    from sentence_transformers.sparse_encoder import SparseEncoder
    qf = args.quantization_factor
    report = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        device = args.device
        for attempt in range(2):
            try:
                model = SparseEncoder(path, device=device)
                model.max_seq_length = args.max_seq_length
                with torch.no_grad():
                    D = model.encode(docs, batch_size=args.batch_size, convert_to_sparse_tensor=False, convert_to_tensor=True).float().cpu().numpy()
                    Q = model.encode(queries, batch_size=args.batch_size * 4, convert_to_sparse_tensor=False, convert_to_tensor=True).float().cpu().numpy()
                break
            except torch.OutOfMemoryError:
                print(f"{label}: CUDA OOM on the shared GPU, retrying on CPU", flush=True)
                del model; torch.cuda.empty_cache(); device = "cpu"
        Da = np.rint(D * qf) > 0
        Qa = np.rint(Q * qf) > 0
        df = Da.sum(axis=0)                       # docs per term
        postings = (Qa * df).sum(axis=1)          # per query: sum over active terms of df
        flops = float(((D.mean(axis=0)) * (Q.mean(axis=0))).sum())
        rep = {
            "doc_active_mean": float(Da.sum(axis=1).mean()),
            "doc_active_median": float(np.median(Da.sum(axis=1))),
            "query_active_mean": float(Qa.sum(axis=1).mean()),
            "query_active_median": float(np.median(Qa.sum(axis=1))),
            "postings_per_query": float(postings.mean()),
            "flops": flops,
            "doc_weight_sum_mean": float(D.sum(axis=1).mean()),
            "query_weight_sum_mean": float(Q.sum(axis=1).mean()),
        }
        report[label] = rep
        print(f"{label:22s} doc_active {rep['doc_active_mean']:7.1f}  query_active {rep['query_active_mean']:6.1f}  "
              f"postings/query {rep['postings_per_query']:9.0f}  flops {flops:8.3f}", flush=True)
        del model, D, Q
        torch.cuda.empty_cache()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"n_docs": len(docs), "n_queries": len(queries), "models": report}, indent=2))
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
