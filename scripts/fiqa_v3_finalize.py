#!/usr/bin/env python
"""Turn one export of the mine-hard-negatives pipeline (output.jsonl with v3
scores) into a training arm for the margin-MSE configs, and write a small JSON
report next to the results.

Checks: every query of the source rows is present, every row has at least
NUM_NEGATIVES scored negatives and a score per positive (queue_v11 refuses
anything else). Rows that fail are dropped and counted in the report, not
silently kept. Rows are written in the order of the source file.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"p10": round(pick(0.1), 3), "median": round(statistics.median(s), 3), "p90": round(pick(0.9), 3),
            "mean": round(sum(s) / len(s), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=Path, required=True, help="output.jsonl of create_flag_embedding_jsonl.py")
    ap.add_argument("--rows", type=Path, default=Path("datasets/arms/fiqa-2073.jsonl"), help="source rows")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--num-negatives", type=int, default=10)
    args = ap.parse_args()

    source = [json.loads(l) for l in args.rows.open(encoding="utf-8") if l.strip()]
    src_by_qid = {str(r["query_id"]): r for r in source}
    mined = {}
    for line in args.mined.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            mined[str(r["query_id"])] = r

    out_rows, dropped = [], collections.Counter()
    stats = collections.Counter()
    neg_hist = collections.Counter()
    tiers = collections.Counter()
    pos_scores, neg_scores, margins = [], [], []
    old_neg_kept = old_neg_total = 0
    old_neg_promoted = 0
    for qid, src in src_by_qid.items():
        r = mined.get(qid)
        if r is None:
            dropped["missing_in_export"] += 1
            continue
        pos, neg = r.get("pos", []), r.get("neg", [])
        ps, ns = r.get("pos_scores", []), r.get("neg_scores", [])
        neg_hist[len(neg)] += 1
        if not pos or len(ps) != len(pos) or any(s is None for s in ps):
            dropped["positive_scores_incomplete"] += 1
            continue
        if len(neg) < args.num_negatives or len(ns) < len(neg) or any(s is None for s in ns[: len(neg)]):
            dropped["fewer_than_num_negatives"] += 1
            continue
        row = {
            "query_id": qid,
            "query": r["query"],
            "pos": pos,
            "neg": neg,
            "pos_scores": ps,
            "neg_scores": ns[: len(neg)],
            "prompt": "",
            "type": "retrieval",
        }
        for k in ("pos_id", "neg_id", "pos_is_synthetic", "neg_selection_tier", "original_pos_scores",
                  "original_neg_scores"):
            if k in r:
                row[k] = r[k]
        out_rows.append(row)

        stats["rows"] += 1
        stats["positives"] += len(pos)
        stats["synthetic_positives"] += sum(bool(x) for x in r.get("pos_is_synthetic", []))
        stats["queries_with_synthetic_positive"] += int(any(r.get("pos_is_synthetic", [])))
        tiers.update(r.get("neg_selection_tier", []))
        pos_scores.extend(float(s) for s in ps)
        neg_scores.extend(float(s) for s in ns[: len(neg)])
        margins.append(min(float(s) for s in ps) - max(float(s) for s in ns[: len(neg)]))
        if min(float(s) for s in ps) < max(float(s) for s in ns[: len(neg)]):
            stats["queries_where_a_negative_outscores_a_positive"] += 1
        old_negs = {str(i) for i in src.get("neg_id", [])}
        new_negs = {str(i) for i in r.get("neg_id", [])}
        new_pos = {str(i) for i, syn in zip(r.get("pos_id", []), r.get("pos_is_synthetic", [])) if syn}
        old_neg_total += len(old_negs)
        old_neg_kept += len(old_negs & new_negs)
        old_neg_promoted += len(old_negs & new_pos)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source_rows": len(source),
        "mined_rows": len(mined),
        "written": args.out.as_posix(),
        "rows_written": len(out_rows),
        "dropped": dict(dropped),
        "negatives_per_query_in_export": dict(sorted(neg_hist.items())),
        "negative_tiers": dict(tiers),
        "positives": stats["positives"],
        "synthetic_positives": stats["synthetic_positives"],
        "queries_with_synthetic_positive": stats["queries_with_synthetic_positive"],
        "old_negatives_kept_share": round(old_neg_kept / old_neg_total, 3) if old_neg_total else None,
        "old_negatives_promoted_to_positive": old_neg_promoted,
        "queries_where_a_negative_outscores_a_positive": stats["queries_where_a_negative_outscores_a_positive"],
        "v3_pos_scores": quantiles(pos_scores),
        "v3_neg_scores": quantiles(neg_scores),
        "v3_min_pos_minus_max_neg": quantiles(margins),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if dropped:
        raise SystemExit(f"{sum(dropped.values())} rows dropped: {dict(dropped)}")


if __name__ == "__main__":
    main()
