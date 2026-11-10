#!/usr/bin/env python
"""Cross-encoder teacher scores for distillation rows, in three steps.

  pairs     rows JSONL -> parquet of (row, query, passage) for the data kitchen's bridge/reranker_pair_scores.py
  merge     the reranker's score parquet back into the rows: within-row centred least squares against a
            reference scoring of the same rows (e.g. the -tv9b twin by the SPLADE teacher) gives the slope that
            puts the logits on the student's scale (the KL loss is shift-invariant per row); pos_scores /
            neg_scores are replaced, the previous ones move to *_source on the first call, a sidecar
            <output>.teacher.json records the scale and the within-row agreement
  ensemble  per-pair mean of the scores of several row files with the same rows in the same order

  python scripts/reranker_rows.py pairs --rows R.jsonl --out R.pairs.parquet
  python scripts/reranker_rows.py merge --rows R.jsonl --scores R.scores.parquet --reference R-tv9b.jsonl --out R-trr.jsonl
  python scripts/reranker_rows.py ensemble --inputs R-tv9b.jsonl R-tpd400.jsonl R-trr.jsonl --out R-tens3.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def pairs(args: argparse.Namespace) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids, queries, passages = [], [], []
    for i, r in enumerate(read(args.rows)):
        for side in ("pos", "neg"):
            for k, text in enumerate(r.get(side) or []):
                ids.append(f"{i}|{side}|{k}")
                queries.append(r["query"])
                passages.append(text)
    pq.write_table(pa.table({"row": ids, "query": queries, "passage": passages}), args.out)
    print(f"{len(ids)} pairs from {i + 1} rows -> {args.out}")
    return 0


def merge(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    rows = read(args.rows)
    ref = read(args.reference) if args.reference else None
    table = pq.read_table(args.scores).to_pydict()
    scores = {rid: float(s) for rid, s in zip(table["row"], table["score"])}
    raw = []
    for i, r in enumerate(rows):
        raw.append({side: [scores[f"{i}|{side}|{k}"] for k in range(len(r.get(side) or []))] for side in ("pos", "neg")})
    scale = args.scale
    stats: dict = {}
    if ref is not None:
        if len(ref) != len(rows):
            print(f"[ERROR] reference has {len(ref)} rows, rows {len(rows)}", file=sys.stderr)
            return 2
        xs, ys, taus, pears = [], [], [], []
        for r, rr, x in zip(rows, ref, raw):
            assert str(r.get("query_id")) == str(rr.get("query_id")), "reference rows out of order"
            a = x["pos"] + x["neg"]
            b = list(rr["pos_scores"])[: len(x["pos"])] + list(rr["neg_scores"])[: len(x["neg"])]
            if len(a) != len(b) or len(a) < 2:
                continue
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            ca, cb = [v - ma for v in a], [v - mb for v in b]
            xs.extend(ca)
            ys.extend(cb)
            va, vb = sum(v * v for v in ca), sum(v * v for v in cb)
            if va > 0 and vb > 0:
                pears.append(sum(p * q for p, q in zip(ca, cb)) / math.sqrt(va * vb))
            conc = sum((1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1 if (a[i] - a[j]) * (b[i] - b[j]) < 0 else 0) for i in range(len(a)) for j in range(i + 1, len(a)))
            taus.append(conc / (len(a) * (len(a) - 1) / 2))
        sxx = sum(v * v for v in xs)
        scale = sum(p * q for p, q in zip(xs, ys)) / sxx if sxx > 0 else 1.0
        stats = {"within_row_pearson_vs_reference": sum(pears) / max(len(pears), 1), "within_row_kendall_vs_reference": sum(taus) / max(len(taus), 1), "scale_from": str(args.reference)}
    margins, flipped = [], 0
    for r, x in zip(rows, raw):
        if "pos_scores_source" not in r:
            r["pos_scores_source"], r["neg_scores_source"] = r.get("pos_scores"), r.get("neg_scores")
        r["pos_scores"] = [v * scale for v in x["pos"]]
        r["neg_scores"] = [v * scale for v in x["neg"]]
        r["teacher_model"] = f"{args.model} (scale {scale:.4f})"
        if r["pos_scores"] and r["neg_scores"]:
            m = min(r["pos_scores"]) - max(r["neg_scores"])
            margins.append(m)
            flipped += m < 0
    write(args.out, rows)
    margins.sort()
    side = {"teacher": args.model, "scale": scale, "rows": len(rows), "median_margin": margins[len(margins) // 2] if margins else None, "rows_with_negative_above_positive": flipped, **stats}
    Path(str(args.out) + ".teacher.json").write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(side, indent=2))
    return 0


def ensemble(args: argparse.Namespace) -> int:
    files = [read(p) for p in args.inputs]
    n = len(files[0])
    assert all(len(f) == n for f in files), "row counts differ"
    out = []
    for i in range(n):
        base = dict(files[0][i])
        for f in files[1:]:
            r = f[i]
            assert str(r.get("query_id")) == str(base.get("query_id")) and len(r["pos"]) == len(base["pos"]) and len(r["neg"]) == len(base["neg"]), f"row {i} differs"
        base["pos_scores"] = [sum(f[i]["pos_scores"][k] for f in files) / len(files) for k in range(len(base["pos"]))]
        base["neg_scores"] = [sum(f[i]["neg_scores"][k] for f in files) / len(files) for k in range(len(base["neg"]))]
        base["teacher_model"] = "ensemble: mean(" + ", ".join(str(f[i].get("teacher_model")) for f in files) + ")"
        out.append(base)
    write(args.out, out)
    print(f"{n} rows -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pairs"); a.add_argument("--rows", type=Path, required=True); a.add_argument("--out", type=Path, required=True); a.set_defaults(fn=pairs)
    b = sub.add_parser("merge"); b.add_argument("--rows", type=Path, required=True); b.add_argument("--scores", type=Path, required=True)
    b.add_argument("--reference", type=Path, default=None); b.add_argument("--scale", type=float, default=1.0); b.add_argument("--model", default="sdadas/polish-reranker-roberta-v3")
    b.add_argument("--out", type=Path, required=True); b.set_defaults(fn=merge)
    c = sub.add_parser("ensemble"); c.add_argument("--inputs", type=Path, nargs="+", required=True); c.add_argument("--out", type=Path, required=True); c.set_defaults(fn=ensemble)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
