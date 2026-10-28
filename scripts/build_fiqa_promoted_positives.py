#!/usr/bin/env python
"""Promote probable missing positives of the gated FiQA rows (28.10, a separate experiment).

In the student mining (scripts/fiqa_splade_mine.sh, own variant) roberta-v3 rejected 791
SPLADE candidates in 288 queries because they score at or above the query's best positive
(a FiQA question usually has several good answers, only some of them annotated). This
script adds up to --max-promoted of them per query as extra positives of a base file
(default the mix2 file, so the arm compares with arm-mnrl-fiqa-splade-mix2): the same
query, pos_scores = the roberta-v3 score, pos_is_synthetic = True, neg_source untouched.
A promoted passage that sits among the row's negatives is removed from them. Nothing is
assumed about whether this helps: it is one arm against its base.

    python scripts/build_fiqa_promoted_positives.py --base datasets/arms/fiqa-2073-splade-mix2.jsonl \
        --mine-dir datasets/arms/fiqa-splade-own-mine --max-promoted 2 \
        --out datasets/arms/fiqa-2073-splade-mix2-pp.jsonl --report results/data/fiqa-2073-splade-mix2-pp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NEG_KEYS = ("neg", "neg_scores", "neg_id", "neg_selection_tier", "original_neg_scores", "neg_source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mine-dir", type=Path, required=True)
    parser.add_argument("--max-promoted", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    d = args.mine_dir
    neg = pd.read_parquet(d / "negatives.parquet", columns=["query_id", "document_id", "final_ranking", "final_selected", "candidate_ranking"])
    pos = pd.read_parquet(d / "relevant_with_score.parquet")
    corpus = pd.read_parquet(d / "corpus.parquet")
    text_of = dict(zip(corpus["id"].astype(str), corpus["text"]))
    best_v3 = pos.groupby("query_id")["positive_ranking"].max()
    m = neg.merge(best_v3.rename("best_v3"), left_on="query_id", right_index=True)
    promote = m[(~m["final_selected"].astype(bool)) & (m["final_ranking"] >= m["best_v3"])].sort_values(["query_id", "final_ranking"], ascending=[True, False])
    promote["query_id"] = promote["query_id"].astype(str)
    promote["document_id"] = promote["document_id"].astype(str)
    by_query = {q: g.head(args.max_promoted) for q, g in promote.groupby("query_id")}

    rows_out = 0
    promoted = 0
    queries_touched = 0
    negatives_removed = 0
    pos_count = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.base.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["query_id"])
            g = by_query.get(qid)
            if g is not None and len(g):
                queries_touched += 1
                existing = set(row["pos"]) | {str(p) for p in row.get("pos_id", [])}
                for doc_id, score in zip(g["document_id"], g["final_ranking"]):
                    text = text_of.get(doc_id)
                    if text is None or text in existing or doc_id in existing:
                        continue
                    if text in row["neg"]:
                        i = row["neg"].index(text)
                        for key in NEG_KEYS:
                            if key in row and i < len(row[key]):
                                del row[key][i]
                        negatives_removed += 1
                    row["pos"].append(text)
                    row.setdefault("pos_scores", []).append(round(float(score), 4))
                    row.setdefault("pos_id", []).append(doc_id)
                    row.setdefault("pos_is_synthetic", []).append(True)
                    if "original_pos_scores" in row:
                        row["original_pos_scores"].append(round(float(score), 4))
                    existing.add(text)
                    promoted += 1
            pos_count[len(row["pos"])] = pos_count.get(len(row["pos"]), 0) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_out += 1

    report = {
        "base": str(args.base), "mine_dir": str(d), "written": str(args.out), "rows_written": rows_out,
        "max_promoted_per_query": args.max_promoted, "eligible_pairs": int(len(promote)), "eligible_queries": int(promote["query_id"].nunique()),
        "promoted_pairs": promoted, "queries_touched": queries_touched, "negatives_removed": negatives_removed,
        "positives_per_row": {str(k): v for k, v in sorted(pos_count.items())},
        "promoted_v3_scores": {k: round(float(v), 3) for k, v in promote.groupby("query_id").head(args.max_promoted)["final_ranking"].quantile([0.1, 0.5, 0.9]).items()},
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
