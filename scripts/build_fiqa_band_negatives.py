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
  mid      score 2 to 5 below the best positive (an absolute-margin middle band)
  far      score more than 3 below the best positive, still in the top-100
  cap3     the 10 hardest strict negatives (highest score) among those more than 3 below
           the best positive: hard in absolute terms, never next to the positive
Only far and cap3 fill 10 per query for most queries (80 %); above, near and mid are
mostly backfilled from the deepest strict candidates and are kept for the record.

    python scripts/build_fiqa_band_negatives.py --mine-dir datasets/arms/fiqa-splade-own-full-mine \
        --base datasets/arms/fiqa-2073-splade-own.jsonl --band all --out-dir datasets/arms --report-dir results/data

Length-matched variant (30.10): every band so far was drawn at the student's length (median
1400-1600 characters against 424 for the old negatives), so difficulty and length were never
separated. With --len-window LO HI a negative must have between LO and HI times the tokens of
the query's positives (their median); the band is drawn inside that window, backfilled from
deeper ranks inside the window, and only then from outside it (reported as
backfilled_outside_window; above 30 % the file does not answer the length question).
--suffix names the output, e.g. fiqa-2073-splade-band-r1-20-lenmatch.jsonl.
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
    "mid": [("margin", -5.0, -2.0, 10)],
    "far": [("margin", -math.inf, -3.0, 10)],
    "cap3": [("margin-top", -math.inf, -3.0, 10)],  # the 10 hardest strict negatives at least 3 below the best positive
    "hard": [("margin-top", -math.inf, math.inf, 10)],  # the 10 hardest strict negatives, no cap (with --len-window: the hardest inside the window)
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


def pick(pool: pd.DataFrame, kind: str, count: int, rng: random.Random) -> pd.DataFrame:
    """Random within the band, or the hardest (highest margin) first for a *-top band."""
    if not len(pool):
        return pool
    if kind.endswith("-top"):
        return pool.sort_values("margin", ascending=False).head(count)
    return pool.sample(n=min(count, len(pool)), random_state=rng.randrange(1 << 30))


def build(band: str, strict: pd.DataFrame, rows: list[dict], text_of: dict[str, str], seed: int, out: Path, report: Path,
          window: tuple[float, float] | None = None, pos_tokens: dict[str, float] | None = None, suffix: str = "") -> None:
    rng = random.Random(seed)
    spec = BANDS[band]
    total = sum(c for _, _, _, c in spec)
    by_query = {q: g for q, g in strict.groupby("query_id")}
    stats = {"rank": [], "margin": [], "v3": [], "chars": [], "tokens": [], "pos_tokens": [], "ratio": []}
    backfilled = 0
    backfilled_outside = 0
    short_rows = 0
    from_base = 0
    dropped = 0
    written = 0
    label = f"{band}{suffix}"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            qid = str(row["query_id"])
            g = by_query.get(qid)
            chosen: list[pd.Series] = []
            taken: set[str] = set()
            target = (pos_tokens or {}).get(qid)
            if g is not None:
                if window is not None:  # draw inside the length window around the query's positives
                    gw = g[(g["tokens"] >= window[0] * target) & (g["tokens"] <= window[1] * target)] if target else g.iloc[0:0]
                else:
                    gw = g
                for kind, low, high, count in spec:
                    pool = in_band(gw, kind, low, high)
                    pool = pool[~pool["document_id"].isin(taken)]
                    for _, r in pick(pool, kind, count, rng).iterrows():
                        chosen.append(r)
                        taken.add(r["document_id"])
                if len(chosen) < total:  # backfill: deeper first (inside the window when there is one)
                    rest = gw[~gw["document_id"].isin(taken)].sort_values("retrieval_rank", ascending=False)
                    for _, r in rest.head(total - len(chosen)).iterrows():
                        chosen.append(r)
                        taken.add(r["document_id"])
                        backfilled += 1
                if window is not None and len(chosen) < total:  # last resort: outside the window, deeper first
                    rest = g[~g["document_id"].isin(taken)].sort_values("retrieval_rank", ascending=False)
                    for _, r in rest.head(total - len(chosen)).iterrows():
                        chosen.append(r)
                        taken.add(r["document_id"])
                        backfilled_outside += 1
            new_row = {k: v for k, v in row.items() if k not in NEG_KEYS}
            new_row["neg"] = [text_of[r["document_id"]] for r in chosen]
            new_row["neg_scores"] = [round(float(r["final_ranking"]), 4) for r in chosen]
            new_row["neg_id"] = [r["document_id"] for r in chosen]
            new_row["neg_selection_tier"] = ["strict"] * len(chosen)
            new_row["original_neg_scores"] = [round(float(r["candidate_ranking"]), 4) for r in chosen]
            new_row["neg_source"] = [f"splade-band:{label}"] * len(chosen)
            new_row["neg_splade_rank"] = [int(r["retrieval_rank"]) for r in chosen]
            new_row["neg_splade_margin"] = [round(float(r["margin"]), 3) for r in chosen]
            for r in chosen:
                stats["rank"].append(float(r["retrieval_rank"])); stats["margin"].append(float(r["margin"]))
                stats["v3"].append(float(r["final_ranking"])); stats["chars"].append(float(len(text_of[r["document_id"]])))
                if "tokens" in r and target:
                    stats["tokens"].append(float(r["tokens"])); stats["pos_tokens"].append(float(target))
                    stats["ratio"].append(float(r["tokens"]) / float(target))
            if len(chosen) < total:  # the strict pool ran out: complete from the base row's own negatives
                short_rows += 1
                for i, text in enumerate(row["neg"]):
                    if len(new_row["neg"]) >= total:
                        break
                    if text in new_row["neg"] or str(row["neg_id"][i]) in taken:
                        continue
                    new_row["neg"].append(text)
                    new_row["neg_scores"].append(row["neg_scores"][i])
                    new_row["neg_id"].append(row["neg_id"][i])
                    new_row["neg_selection_tier"].append(row.get("neg_selection_tier", ["strict"] * len(row["neg"]))[i])
                    new_row["original_neg_scores"].append(row.get("original_neg_scores", [None] * len(row["neg"]))[i])
                    new_row["neg_source"].append("base-row")
                    new_row["neg_splade_rank"].append(-1)
                    new_row["neg_splade_margin"].append(None)
                    from_base += 1
            if len(new_row["neg"]) < total:
                dropped += 1
                continue
            handle.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            written += 1
    n_neg = max(1, len(stats["rank"]))
    rep = {"band": band, "suffix": suffix, "spec": [list(s) for s in spec], "rows_written": written, "rows_dropped": dropped,
           "rows_short_in_band": short_rows, "negatives_from_base_row": from_base,
           "backfilled_negatives": backfilled, "negatives": len(stats["rank"]),
           "splade_rank": quantiles(stats["rank"]), "splade_margin_to_best_positive": quantiles(stats["margin"]),
           "v3_neg_scores": quantiles(stats["v3"]), "neg_chars": quantiles(stats["chars"])}
    if window is not None:
        in_window = sum(1 for x in stats["ratio"] if window[0] <= x <= window[1])
        rep.update({"len_window": list(window), "backfilled_outside_window": backfilled_outside,
                    "share_outside_window": round(1 - in_window / n_neg, 4),
                    "share_backfilled_outside_or_base": round((backfilled_outside + from_base) / n_neg, 4),
                    "neg_tokens": quantiles(stats["tokens"]), "pos_tokens": quantiles(stats["pos_tokens"]),
                    "neg_to_pos_token_ratio": quantiles(stats["ratio"])})
    report.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    extra = f" window {rep['len_window']} outside {rep['share_outside_window']:.1%} (backfilled outside {backfilled_outside}) neg tokens {rep['neg_tokens']} pos tokens {rep['pos_tokens']}" if window is not None else ""
    print(f"{label}: rows {written} (dropped {dropped}, short in band {short_rows}, from base row {from_base}, backfilled {backfilled}) rank {rep['splade_rank']} margin {rep['splade_margin_to_best_positive']} v3 {rep['v3_neg_scores']} chars {rep['neg_chars']}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mine-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="rows whose query/positive fields are kept (negatives replaced)")
    parser.add_argument("--band", default="all", help="one of %s or all" % ", ".join(BANDS))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/arms"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/data"))
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--len-window", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                        help="keep negatives whose token count is LO..HI times the median tokens of the query's positives")
    parser.add_argument("--suffix", default="", help="output name suffix, e.g. -lenmatch")
    parser.add_argument("--tokenizer", type=Path, default=Path("models/base-splade-wrapped"), help="tokenizer for --len-window")
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
    window = None
    pos_tokens: dict[str, float] | None = None
    if args.len_window is not None:
        window = (float(args.len_window[0]), float(args.len_window[1]))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
        ids = corpus["id"].astype(str).tolist()
        lengths = tokenizer(corpus["text"].tolist(), add_special_tokens=True, truncation=False, return_length=True)["length"]
        tok_of = dict(zip(ids, (int(x) for x in lengths)))
        neg["tokens"] = neg["document_id"].map(tok_of)
        pos_len = pos.assign(query_id=pos["query_id"].astype(str), tokens=pos["document_id"].astype(str).map(tok_of))
        pos_tokens = pos_len.groupby("query_id")["tokens"].median().to_dict()
        print(f"length window {window[0]:.2f}-{window[1]:.2f} x the median positive tokens; corpus tokens median {statistics.median(tok_of.values()):.0f}")
    coverage = len(neg) / max(1, len(cand))
    print(f"teacher coverage of the top-100 candidates: {coverage:.1%} ({len(neg)} of {len(cand)})")
    if coverage < 0.95:
        print("WARNING: the deep bands need a fully scored mine directory (VARIANT=own-full); results will be backfilled heavily")
    strict = neg[neg["final_selected"].astype(bool)]
    rows = [json.loads(line) for line in args.base.open(encoding="utf-8") if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for band in (BANDS if args.band == "all" else [args.band]):
        build(band, strict, rows, text_of, args.seed, args.out_dir / f"fiqa-2073-splade-band-{band}{args.suffix}.jsonl",
              args.report_dir / f"fiqa-2073-splade-band-{band}{args.suffix}.json", window=window, pos_tokens=pos_tokens, suffix=args.suffix)


if __name__ == "__main__":
    main()
