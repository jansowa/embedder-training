#!/usr/bin/env python
"""Mix student-mined and old negatives for the gated FiQA rows (28.10).

The negatives chosen by SPLADE (scripts/fiqa_splade_mine.sh) are about 2.5x longer than
the old, dense-retriever-mined ones (median 1477 vs 424 characters), while the positives
sit in between (934). A model trained on the student file alone could learn "long means
negative", the mirror image of the old shortcut, and queue v15 showed that ten student
negatives per query cost 2.7 points on tiny (arm-mnrl-fiqa-splade-own-b1 59.53 vs 62.27).
This file gives every query --student-count negatives from the student file (the highest
SPLADE score first, the hardest for the student, strict tier first) and --old-count from
the old file (its export order: the hardest for roberta-v3 first). Both files carry
roberta-v3 scores in neg_scores, so the mix also works for the KL arms.

    python scripts/build_fiqa_mix_negatives.py \
        --student datasets/arms/fiqa-2073-splade-own.jsonl \
        --student-mine-dir datasets/arms/fiqa-splade-own-mine \
        --old datasets/arms/fiqa-2073-v3.jsonl --student-count 5 --old-count 5 \
        --out datasets/arms/fiqa-2073-splade-mix.jsonl --report results/data/fiqa-2073-splade-mix.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

LIST_KEYS = ("neg", "neg_scores", "neg_id", "neg_selection_tier", "original_neg_scores")


def read_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["query_id"])] = row
    return rows


def negatives_of(row: dict) -> list[dict]:
    n = len(row["neg"])
    return [
        {key: (row[key][i] if key in row and i < len(row[key]) else None) for key in LIST_KEYS}
        for i in range(n)
    ]


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]  # noqa: E731
    return {"p10": round(pick(0.1), 3), "median": round(pick(0.5), 3), "p90": round(pick(0.9), 3), "mean": round(statistics.fmean(values), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--student-mine-dir", type=Path, required=True, help="directory with negatives.parquet (SPLADE score per candidate)")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--student-count", type=int, default=5)
    parser.add_argument("--old-count", type=int, default=5)
    args = parser.parse_args()

    student = read_rows(args.student)
    old = read_rows(args.old)
    scores = pd.read_parquet(args.student_mine_dir / "negatives.parquet", columns=["query_id", "document_id", "candidate_ranking"])
    splade_score = {(str(q), str(d)): float(s) for q, d, s in scores.itertuples(index=False)}

    counts = {"student": 0, "old": 0, "student_backfill": 0, "old_backfill": 0}
    dropped_duplicates = 0
    lengths: dict[str, list[int]] = {"student": [], "old": []}
    teacher: dict[str, list[float]] = {"student": [], "old": []}
    missing_old = 0
    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for qid, row in student.items():
            total = args.student_count + args.old_count
            st = negatives_of(row)
            # hardest for the student first; strict before relaxed backfill
            st.sort(key=lambda n: (n["neg_selection_tier"] != "strict", -splade_score.get((qid, str(n["neg_id"])), float("-inf"))))
            ol = negatives_of(old[qid]) if qid in old else []
            if not ol:
                missing_old += 1
            seen_ids = {str(p) for p in row.get("pos_id", [])}
            seen_text = set(row["pos"])
            chosen: list[tuple[str, dict]] = []

            def take(source: str, pool: list[dict], limit: int, backfill: bool = False) -> None:
                nonlocal dropped_duplicates
                taken = 0
                for n in pool:
                    if taken >= limit:
                        break
                    key = str(n["neg_id"])
                    if key in seen_ids or n["neg"] in seen_text:
                        dropped_duplicates += 1
                        continue
                    seen_ids.add(key)
                    seen_text.add(n["neg"])
                    chosen.append((source, n))
                    counts[f"{source}_backfill" if backfill else source] += 1
                    taken += 1

            take("student", st, args.student_count)
            take("old", ol, args.old_count)
            if len(chosen) < total:  # one source ran short: fill from the other
                take("old", ol, total - len(chosen), backfill=True)
                take("student", st, total - len(chosen), backfill=True)
            for source, n in chosen:
                lengths[source].append(len(n["neg"]))
                if n["neg_scores"] is not None:
                    teacher[source].append(float(n["neg_scores"]))
            new_row = {k: v for k, v in row.items() if k not in LIST_KEYS}
            for key in LIST_KEYS:
                new_row[key] = [n[key] for _, n in chosen]
            new_row["neg_source"] = [source for source, _ in chosen]
            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            written += 1

    report = {
        "student": str(args.student), "old": str(args.old), "written": str(args.out), "rows_written": written,
        "student_count": args.student_count, "old_count": args.old_count, "negatives_by_source": counts,
        "dropped_duplicates": dropped_duplicates, "queries_missing_in_old": missing_old,
        "neg_chars": {k: quantiles([float(x) for x in v]) for k, v in lengths.items()},
        "v3_neg_scores": {k: quantiles(v) for k, v in teacher.items()},
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
