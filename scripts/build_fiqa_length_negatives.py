#!/usr/bin/env python
"""Length-decorrelated, student-easy negatives for the gated FiQA rows (28.10).

Base polish-splade rewards passage length (positives over 512 tokens rank first in 57 % of
the pairs, positives up to 128 tokens in 33 %; the student-mined negatives are 2.5x longer
than the old ones). The student-negative arms could not separate "long" from "hard". These
candidates separate them: passages outside the query's SPLADE top-100 (easy for the
student), chosen by length only:
  long-easy    longer (in SPLADE tokens) than the query's longest positive
  short-easy   shorter than the query's shortest positive (the control: same difficulty,
               the opposite length signal)
Per query --per-query candidates are drawn at random (seeded); when the corpus has too few
passages beyond the bound, the closest ones in length fill up (reported). The output is a
mine directory for scripts/fiqa_splade_mine.sh (VARIANT=long-easy / short-easy): the
parquet inputs copied from --mine-dir, negative_candidates.parquet in the student miner's
schema (candidate_ranking 0, all selected) and a splade_mining_report.json with the
sampling statistics; roberta-v3 then vetoes false negatives and the export keeps 10.

    python scripts/build_fiqa_length_negatives.py --mine-dir datasets/arms/fiqa-splade-own-full-mine \
        --variant long-easy --out-dir datasets/arms/fiqa-splade-long-easy-mine
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

COPY = ("queries", "corpus", "relevant", "relevant_with_candidate_score", "relevant_with_score")


def quantiles(values) -> dict:
    s = pd.Series(list(values), dtype=float)
    if s.empty:
        return {}
    return {"p10": float(s.quantile(0.1)), "median": float(s.median()), "p90": float(s.quantile(0.9)), "mean": float(s.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mine-dir", type=Path, required=True, help="a student mining directory (its top-100 candidates mark what is easy)")
    parser.add_argument("--variant", choices=["long-easy", "short-easy"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/base-splade-wrapped"), help="tokenizer source")
    parser.add_argument("--per-query", type=int, default=20, help="candidates per query (the teacher pass keeps the safe ones, the export 10)")
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    d = args.mine_dir
    corpus = pd.read_parquet(d / "corpus.parquet")
    corpus["id"] = corpus["id"].astype(str)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    corpus["tok"] = np.array(tokenizer(corpus["text"].tolist(), add_special_tokens=True, truncation=False, return_length=True)["length"])
    lengths = corpus.set_index("id")["tok"]
    order = corpus.sort_values("tok")[["id", "tok"]].reset_index(drop=True)
    tok_sorted = order["tok"].to_numpy()
    id_sorted = order["id"].to_numpy()

    relevant = pd.read_parquet(d / "relevant.parquet")
    relevant["query_id"] = relevant["query_id"].astype(str); relevant["document_id"] = relevant["document_id"].astype(str)
    top = pd.read_parquet(d / "negative_candidates.parquet", columns=["query_id", "document_id"])
    top["query_id"] = top["query_id"].astype(str); top["document_id"] = top["document_id"].astype(str)
    positives = relevant.groupby("query_id")["document_id"].apply(set)
    top100 = top.groupby("query_id")["document_id"].apply(set)

    rng = random.Random(args.seed)
    rows = []
    stats = {"candidate_tokens": [], "positive_bound_tokens": [], "filled_by_closest": 0, "queries": 0, "candidates_over_512": 0}
    for qid, pos_ids in positives.items():
        excluded = pos_ids | top100.get(qid, set())
        pos_tok = lengths.reindex(list(pos_ids)).dropna()
        if pos_tok.empty:
            continue
        if args.variant == "long-easy":
            bound = float(pos_tok.max()); lo = int(np.searchsorted(tok_sorted, bound, side="right")); pool = id_sorted[lo:]
            closest = id_sorted[:lo][::-1]  # just below the bound, longest first
        else:
            bound = float(pos_tok.min()); hi = int(np.searchsorted(tok_sorted, bound, side="left")); pool = id_sorted[:hi]
            closest = id_sorted[hi:]  # just above the bound, shortest first
        pool = [i for i in pool if i not in excluded]
        chosen = rng.sample(pool, min(args.per_query, len(pool))) if pool else []
        if len(chosen) < args.per_query:
            for i in closest:
                if len(chosen) >= args.per_query:
                    break
                if i not in excluded and i not in chosen:
                    chosen.append(i); stats["filled_by_closest"] += 1
        stats["queries"] += 1
        stats["positive_bound_tokens"].append(bound)
        for rank, doc_id in enumerate(chosen, start=1):
            t = float(lengths[doc_id])
            stats["candidate_tokens"].append(t)
            stats["candidates_over_512"] += int(t > 512)
            rows.append({"query_id": qid, "document_id": doc_id, "candidate_ranking": 0.0, "candidate_percentile": 0.0,
                         "candidate_selected": True, "retrieval_rank": rank, "retrieval_offset": 0.0, "retrieval_score": t,
                         "retrieval_source": f"length:{args.variant}"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copy(d / f"{name}.parquet", args.out_dir / f"{name}.parquet")
    pd.DataFrame(rows).to_parquet(args.out_dir / "negative_candidates.parquet", index=False)
    report = {
        "variant": args.variant, "source_mine_dir": str(d), "queries": stats["queries"], "candidate_rows": len(rows),
        "per_query": args.per_query, "filled_by_closest_length": stats["filled_by_closest"],
        "candidate_tokens": quantiles(stats["candidate_tokens"]), "positive_bound_tokens": quantiles(stats["positive_bound_tokens"]),
        "candidates_over_512_share": round(stats["candidates_over_512"] / max(1, len(rows)), 4),
        "corpus_tokens": quantiles(corpus["tok"]),
    }
    (args.out_dir / "splade_mining_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
