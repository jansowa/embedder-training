#!/usr/bin/env python
"""Passage length against the base SPLADE scores of the student mining (28.10).

Reads a scripts/fiqa_splade_mine.sh output directory (corpus, candidates, positives with
their SPLADE scores, the teacher's verdicts) and reports, in tokens of the SPLADE
tokenizer: the corpus length distribution, the length of the student's top candidates and
of the teacher-confirmed negatives that outscore the positive, the rank correlation of
score and length, and the positives' ranks by length bucket. Answers "does the model
struggle near 512 tokens?" (no: long positives rank first more often; the model rewards
length, so short positives are the weak link).

    python scripts/fiqa_splade_length_bias.py --mine-dir datasets/arms/fiqa-splade-own-mine \
        --model models/base-splade-wrapped --report results/data/fiqa-2073-splade-own-length.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BUCKETS = [0, 128, 256, 512, 10**6]
LABELS = ["<=128", "129-256", "257-512", ">512"]


def quantiles(series: pd.Series) -> dict[str, int]:
    return {f"p{int(q * 100)}": int(v) for q, v in series.quantile([0.1, 0.5, 0.9]).items()}


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mine-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/base-splade-wrapped"), help="tokenizer source")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from scipy.stats import spearmanr
    from transformers import AutoTokenizer

    d = args.mine_dir
    corpus = pd.read_parquet(d / "corpus.parquet")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    lengths = tokenizer(corpus["text"].tolist(), add_special_tokens=True, truncation=False, return_length=True)["length"]
    tokens = pd.Series(np.array(lengths), index=corpus["id"].astype(str))

    cand = pd.read_parquet(d / "negative_candidates.parquet", columns=["query_id", "document_id", "candidate_ranking", "retrieval_rank"])
    cand["tok"] = cand["document_id"].astype(str).map(tokens)
    neg = pd.read_parquet(d / "negatives.parquet", columns=["query_id", "document_id", "final_selected"])
    pos = pd.read_parquet(d / "relevant_with_score.parquet")
    pos["tok"] = pos["document_id"].astype(str).map(tokens)

    best = pos.groupby("query_id")["positive_candidate_ranking"].max()
    merged = cand.merge(neg, on=["query_id", "document_id"], how="left").merge(best.rename("best_pos"), left_on="query_id", right_index=True)
    real_above = merged[(merged["candidate_ranking"] > merged["best_pos"]) & merged["final_selected"].fillna(False).astype(bool)]

    def length_block(series: pd.Series) -> dict:
        return {"tokens": quantiles(series), "share_over_512": round(float((series > 512).mean()), 4), "share_over_256": round(float((series > 256).mean()), 4), "n": int(series.size)}

    report = {
        "mine_dir": str(d), "tokenizer": str(args.model),
        "corpus": length_block(tokens),
        "student_top1": length_block(cand[cand["retrieval_rank"] <= 1]["tok"]),
        "student_top10": length_block(cand[cand["retrieval_rank"] <= 10]["tok"]),
        "student_top100": length_block(cand["tok"]),
        "positives": length_block(pos["tok"]),
        "real_negatives_above_positive": length_block(real_above["tok"]),
        "spearman_score_vs_tokens_all_candidates": round(float(spearmanr(cand["candidate_ranking"], cand["tok"]).correlation), 4),
        "spearman_score_vs_tokens_within_query_median": round(float(cand.groupby("query_id").apply(lambda g: spearmanr(g["candidate_ranking"], g["tok"]).correlation).median()), 4),
    }

    top1 = cand.groupby("query_id")["candidate_ranking"].max().rename("top1")
    pos = pos.merge(top1, left_on="query_id", right_index=True)
    pairs = cand.merge(pos[["query_id", "document_id", "positive_candidate_ranking"]].rename(columns={"document_id": "pid"}), on="query_id")
    n_above = pairs[pairs["candidate_ranking"] > pairs["positive_candidate_ranking"]].groupby(["query_id", "pid"]).size().rename("n_above")
    pos = pos.merge(n_above, left_on=["query_id", "document_id"], right_index=True, how="left").fillna({"n_above": 0})
    pos["bucket"] = pd.cut(pos["tok"], BUCKETS, labels=LABELS)
    by_bucket = {}
    for label, g in pos.groupby("bucket", observed=True):
        by_bucket[str(label)] = {
            "pairs": int(len(g)),
            "positive_first": round(float((g["n_above"] == 0).mean()), 3),
            "positive_in_top10": round(float((g["n_above"] < 10).mean()), 3),
            "positive_beyond_top100": round(float((g["n_above"] >= 100).mean()), 3),
            "splade_score_median": round(float(g["positive_candidate_ranking"].median()), 2),
            "score_minus_top1_median": round(float((g["positive_candidate_ranking"] - g["top1"]).median()), 2),
        }
    report["positives_by_token_bucket"] = by_bucket
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
