#!/usr/bin/env python
"""Negatives by difficulty for the student, for the gated FiQA rows (28.10).

Difficulty is what base polish-splade thinks of a passage: its rank among the top-100
non-positive candidates of the query (scripts/fiqa_splade_mine.py) or its score margin to
the query's best positive. Every band keeps the teacher filter of the student mining
(final_selected from the roberta-v3 pass with the two percentile thresholds), so a band
file differs from the others only in where its negatives sit for the student. Within a
band the negatives are drawn at random (seeded), so the file shows the band's typical
difficulty rather than its hardest edge; a query short of candidates in its band is
backfilled from the next deeper band (reported). Needs a mine directory whose top-100
candidates are all scored by the teacher (VARIANT=own-full of scripts/fiqa_splade_mine.sh).

Bands (datasets/arms/fiqa-2073-splade-band-<name>.jsonl; the teacher keeps only about half
of the top-10, so the first band is 20 wide to hold 10 strict negatives per query):
  r1-20    SPLADE ranks 1-20 (the hardest; roughly the own file)
  r21-50   ranks 21-50
  r51-100  ranks 51-100
  mixed    3 from ranks 1-20, 4 from 21-50, 3 from 51-100
  above    score above the best positive (the student is wrong; only 49 % of the queries
           have such passages, the rest is backfilled from the deepest strict ones)
  near     score within 3 below the best positive (the student is right by little)
  far      score more than 3 below the best positive, still in the top-100

    python scripts/build_fiqa_band_negatives.py --mine-dir datasets/arms/fiqa-splade-own-full-mine \
        --base datasets/arms/fiqa-2073-splade-own.jsonl --band all --out-dir datasets/arms --report-dir results/data
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import pandas as pd

NEG_KEYS = ("neg", "neg_scores", "neg_id", "neg_selection_tier", "original_neg_scores", "neg_source")
# band -> list of (kind, low, high, count); kind rank: low <= rank <= high; kind margin: low < margin <= high
BANDS: dict[str, list[tuple[str, float, float, int]]] = {
    "r1-20": [("rank", 1, 20, 10)],
    "r21-50": [("rank", 21, 50, 10)],
    "r51-100": [("rank", 51, 100, 10)],
    "mixed": [("rank", 1, 20, 3), ("rank", 21, 50, 4), ("rank", 51, 100, 3)],
    "above": [("margin", 0.0, math.inf, 10)],
    "near": [("margin", -3.0, 0.0, 10)],
    "far": [("margin", -math.inf, -3.0, 10)],
}


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]  # noqa: E731
    return {"p10": round(pick(0.1), 3), "median": round(pick(0.5), 3), "p90": round(pick(0.9), 3), "mean": round(statistics.fmean(values), 3)}


def in_band(frame: pd.DataFrame, kind: str, low: float, high: float) -> pd.DataFrame:
    if kind == "rank":
        return frame[(frame["retrieval_rank"] >= low) & (frame["retrieval_rank"] <= high)]
    return frame[(frame["margin"] > low) & (frame["margin"] <= high)]


def build(band: str, strict: pd.DataFrame, rows: list[dict], text_of: dict[str, str], seed: int, out: Path, report: Path) -> None:
    rng = random.Random(seed)
    spec = BANDS[band]
    total = sum(c for _, _, _, c in spec)
    by_query = {q: g for q, g in strict.groupby("query_id")}
    stats = {"rank": [], "margin": [], "v3": [], "chars": []}
    backfilled = 0
    short_rows = 0
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            qid = str(row["query_id"])
            g = by_query.get(qid)
            chosen: list[pd.Series] = []
            taken: set[str] = set()
            if g is not None:
                for kind, low, high, count in spec:
                    pool = in_band(g, kind, low, high)
                    pool = pool[~pool["document_id"].isin(taken)]
                    pick = pool.sample(n=min(count, len(pool)), random_state=rng.randrange(1 << 30)) if len(pool) else pool
                    for _, r in pick.iterrows():
                        chosen.append(r)
                        taken.add(r["document_id"])
                if len(chosen) < total:  # backfill: deeper first, then anything strict
                    rest = g[~g["document_id"].isin(taken)].sort_values("retrieval_rank", ascending=False)
                    for _, r in rest.head(total - len(chosen)).iterrows():
                        chosen.append(r)
                        taken.add(r["document_id"])
                        backfilled += 1
            if len(chosen) < total:
                short_rows += 1
            if not chosen:
                continue
            new_row = {k: v for k, v in row.items() if k not in NEG_KEYS}
            new_row["neg"] = [text_of[r["document_id"]] for r in chosen]
            new_row["neg_scores"] = [round(float(r["final_ranking"]), 4) for r in chosen]
            new_row["neg_id"] = [r["document_id"] for r in chosen]
            new_row["neg_selection_tier"] = ["strict"] * len(chosen)
            new_row["original_neg_scores"] = [round(float(r["candidate_ranking"]), 4) for r in chosen]
            new_row["neg_source"] = [f"splade-band:{band}"] * len(chosen)
            new_row["neg_splade_rank"] = [int(r["retrieval_rank"]) for r in chosen]
            new_row["neg_splade_margin"] = [round(float(r["margin"]), 3) for r in chosen]
            for r in chosen:
                stats["rank"].append(float(r["retrieval_rank"])); stats["margin"].append(float(r["margin"]))
                stats["v3"].append(float(r["final_ranking"])); stats["chars"].append(float(len(text_of[r["document_id"]])))
            handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            written += 1
    rep = {"band": band, "spec": [list(s) for s in spec], "rows_written": written, "rows_short_of_negatives": short_rows,
           "backfilled_negatives": backfilled, "negatives": len(stats["rank"]),
           "splade_rank": quantiles(stats["rank"]), "splade_margin_to_best_positive": quantiles(stats["margin"]),
           "v3_neg_scores": quantiles(stats["v3"]), "neg_chars": quantiles(stats["chars"])}
    report.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{band}: rows {written} (short {short_rows}, backfilled {backfilled}) rank {rep['splade_rank']} margin {rep['splade_margin_to_best_positive']} v3 {rep['v3_neg_scores']} chars {rep['neg_chars']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mine-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="rows whose query/positive fields are kept (negatives replaced)")
    parser.add_argument("--band", default="all", help="one of %s or all" % ", ".join(BANDS))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/arms"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/data"))
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    d = args.mine_dir
    neg = pd.read_parquet(d / "negatives.parquet", columns=["query_id", "document_id", "candidate_ranking", "retrieval_rank", "final_ranking", "final_selected"])
    cand = pd.read_parquet(d / "negative_candidates.parquet", columns=["query_id", "document_id"])
    pos = pd.read_parquet(d / "relevant_with_candidate_score.parquet")
    corpus = pd.read_parquet(d / "corpus.parquet")
    text_of = dict(zip(corpus["id"].astype(str), corpus["text"]))
    neg["query_id"] = neg["query_id"].astype(str); neg["document_id"] = neg["document_id"].astype(str)
    best = pos.groupby(pos["query_id"].astype(str))["positive_candidate_ranking"].max()
    neg = neg.merge(best.rename("best_pos"), left_on="query_id", right_index=True)
    neg["margin"] = neg["candidate_ranking"] - neg["best_pos"]
    coverage = len(neg) / max(1, len(cand))
    print(f"teacher coverage of the top-100 candidates: {coverage:.1%} ({len(neg)} of {len(cand)})")
    if coverage < 0.95:
        print("WARNING: the deep bands need a fully scored mine directory (VARIANT=own-full); results will be backfilled heavily")
    strict = neg[neg["final_selected"].astype(bool)]
    rows = [json.loads(line) for line in args.base.open(encoding="utf-8") if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for band in (BANDS if args.band == "all" else [args.band]):
        build(band, strict, rows, text_of, args.seed, args.out_dir / f"fiqa-2073-splade-band-{band}.jsonl", args.report_dir / f"fiqa-2073-splade-band-{band}.json")


if __name__ == "__main__":
    main()
