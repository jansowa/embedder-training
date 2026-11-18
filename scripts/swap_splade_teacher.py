#!/usr/bin/env python
"""Replace the SPLADE member of a two-teacher ensemble in a row file.

The rows of an ensemble teacher (`scripts/reranker_rows.py ensemble`, or the kitchen's mine-and-score tools) keep only
the per-pair mean of a SPLADE teacher and a scaled dense teacher. Given the same rows scored by the old SPLADE teacher
and by a new one (kitchen `bridge/splade_teacher_scores.py`, same rows in the same order), the dense part of a pair is
`2 x ensemble - old`, and the new ensemble is `(new + dense) / 2 = ensemble + (new - old) / 2`: the way of combining,
the dense scale and the weights stay what they were.

The sidecar `<out>.teacher.json` (and the last line printed) reports, for the old and the new ensemble and for the
members: the median margin min(pos) - max(neg), the rows with a negative above a positive, the median probability of
the positive at temperature 2, and the within-row agreement (Pearson, Kendall) of the new ensemble with the old one and
of the members with each other.

    python scripts/swap_splade_teacher.py --ens rows.jsonl --old-splade rows-tv9b.jsonl --new-splade rows-tnew.jsonl \\
        --out rows-tensnew.jsonl --new-name "models/new (raw)"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Iterator


def read(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def kendall(x: list[float], y: list[float]) -> float | None:
    conc = disc = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            conc += s > 0
            disc += s < 0
    return (conc - disc) / (conc + disc) if conc + disc else None


def swap(ens: dict, old: dict, new: dict) -> tuple[list[float], list[float], dict[str, list[float]]]:
    """New pos / neg scores of one row and the score vectors (positives first) of every member."""
    for key in ("pos", "neg"):
        if not (len(ens[key]) == len(ens[f"{key}_scores"]) == len(old[f"{key}_scores"]) == len(new[f"{key}_scores"])):
            raise ValueError(f"row {ens.get('query_id')}: the {key} lists differ in length between the files")
    if ens["query"] != old["query"] or ens["query"] != new["query"]:
        raise ValueError(f"row {ens.get('query_id')}: the files are not the same rows in the same order")
    e = list(ens["pos_scores"]) + list(ens["neg_scores"])
    o = list(old["pos_scores"]) + list(old["neg_scores"])
    n = list(new["pos_scores"]) + list(new["neg_scores"])
    out = [ev + (nv - ov) / 2.0 for ev, ov, nv in zip(e, o, n)]
    dense = [2.0 * ev - ov for ev, ov in zip(e, o)]
    k = len(ens["pos_scores"])
    return out[:k], out[k:], {"old_ensemble": e, "new_ensemble": out, "old_splade": o, "new_splade": n, "dense": dense}


class Acc:
    def __init__(self) -> None:
        self.margin: dict[str, list[float]] = {}
        self.flip: dict[str, int] = {}
        self.mass: dict[str, list[float]] = {}
        self.agree: dict[str, list[float]] = {}
        self.std_ratio: list[float] = []
        self.rows = 0

    def add(self, n_pos: int, vectors: dict[str, list[float]]) -> None:
        self.rows += 1
        for name, v in vectors.items():
            pos, neg = v[:n_pos], v[n_pos:]
            self.margin.setdefault(name, []).append(min(pos) - max(neg))
            self.flip[name] = self.flip.get(name, 0) + (max(neg) > min(pos))
            s = [max(pos)] + neg
            m = max(s)
            e = [math.exp((x - m) / 2.0) for x in s]
            self.mass.setdefault(name, []).append(e[0] / sum(e))
        for a, b in (("new_ensemble", "old_ensemble"), ("new_splade", "old_splade"), ("new_splade", "dense"), ("old_splade", "dense")):
            for fn_name, fn in (("pearson", pearson), ("kendall", kendall)):
                value = fn(vectors[a], vectors[b])
                if value is not None:
                    self.agree.setdefault(f"{fn_name}_{a}_vs_{b}", []).append(value)
        so, sn = st.pstdev(vectors["old_splade"]), st.pstdev(vectors["new_splade"])
        if so > 0:
            self.std_ratio.append(sn / so)

    def report(self) -> dict:
        out: dict = {"rows": self.rows, "members": {}}
        for name in self.margin:
            out["members"][name] = {"median_margin": st.median(self.margin[name]), "rows_with_negative_above_positive": self.flip[name],
                                    "share_with_negative_above_positive": self.flip[name] / max(self.rows, 1),
                                    "positive_mass_median": st.median(self.mass[name]), "positive_mass_above_095": sum(x > 0.95 for x in self.mass[name]) / max(self.rows, 1)}
        out["within_row"] = {k: sum(v) / len(v) for k, v in self.agree.items()}
        out["within_row_std_ratio_new_to_old_splade_median"] = st.median(self.std_ratio) if self.std_ratio else None
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ens", type=Path, required=True, help="rows with the scores of the old ensemble")
    ap.add_argument("--old-splade", type=Path, required=True, help="the same rows scored by the SPLADE teacher that is in the ensemble")
    ap.add_argument("--new-splade", type=Path, required=True, help="the same rows scored by the SPLADE teacher that takes its place")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--new-name", default="the new SPLADE teacher")
    ap.add_argument("--old-name", default="models/soup-v9b")
    ap.add_argument("--check-old", type=Path, default=None, help="an earlier scoring by the old SPLADE teacher (any subset of the rows): its agreement with --old-splade, matched by query text")
    ap.add_argument("--stats-only", action="store_true", help="no output file; the row files may be a prefix of --ens (smoke)")
    args = ap.parse_args(argv)
    if not args.stats_only and args.out is None:
        ap.error("--out is required without --stats-only")

    earlier: dict[str, list[float]] = {}
    if args.check_old is not None:
        for r in read(args.check_old):
            earlier[r["query"]] = list(r["pos_scores"]) + list(r["neg_scores"])
    acc, diffs = Acc(), []
    olds, news = read(args.old_splade), read(args.new_splade)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp") if args.out else None
    fh = tmp.open("w", encoding="utf-8") if tmp else None
    for ens in read(args.ens):
        old, new = next(olds, None), next(news, None)
        if old is None or new is None:
            if args.stats_only:
                break
            raise ValueError("a scored file has fewer rows than --ens")
        pos, neg, vectors = swap(ens, old, new)
        acc.add(len(pos), vectors)
        was = earlier.get(ens["query"])
        if was is not None and len(was) == len(vectors["old_splade"]):
            diffs.extend(abs(a - b) for a, b in zip(was, vectors["old_splade"]))
        if fh:
            row = dict(ens)
            row["pos_scores"], row["neg_scores"] = pos, neg
            row["teacher_model"] = str(ens.get("teacher_model", "")).replace(args.old_name, args.new_name) if args.old_name in str(ens.get("teacher_model", "")) else f"ensemble: mean({args.new_name}, the dense member of: {ens.get('teacher_model')})"
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if fh:
        fh.close()
        if next(olds, None) is not None or next(news, None) is not None:
            raise ValueError("a scored file has more rows than --ens")
        tmp.replace(args.out)
    rep = acc.report()
    rep.update({"ens": str(args.ens), "old_splade": str(args.old_splade), "new_splade": str(args.new_splade), "new_name": args.new_name, "old_name": args.old_name,
                "formula": "new ensemble = old ensemble + (new SPLADE - old SPLADE) / 2 per pair; dense member = 2 x old ensemble - old SPLADE"})
    if diffs:
        diffs.sort()
        rep["old_splade_against_its_earlier_scoring"] = {"pairs": len(diffs), "median_abs_diff": diffs[len(diffs) // 2], "p99_abs_diff": diffs[int(0.99 * (len(diffs) - 1))], "max_abs_diff": diffs[-1]}
    if args.out:
        Path(str(args.out) + ".teacher.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    m, w = rep["members"], rep["within_row"]
    print(json.dumps(rep, indent=2))
    print("rows=%d margin old/new ensemble %.3f/%.3f, negative above positive %.4f/%.4f, P(pos) median %.3f/%.3f; old/new SPLADE alone: flips %.4f/%.4f; "
          "pearson_new_vs_old_ensemble=%.4f kendall_new_vs_old_ensemble=%.4f pearson_new_vs_old_splade=%.4f std_ratio_new_to_old_splade=%.3f" % (
              rep["rows"], m["old_ensemble"]["median_margin"], m["new_ensemble"]["median_margin"], m["old_ensemble"]["share_with_negative_above_positive"],
              m["new_ensemble"]["share_with_negative_above_positive"], m["old_ensemble"]["positive_mass_median"], m["new_ensemble"]["positive_mass_median"],
              m["old_splade"]["share_with_negative_above_positive"], m["new_splade"]["share_with_negative_above_positive"],
              w.get("pearson_new_ensemble_vs_old_ensemble", float("nan")), w.get("kendall_new_ensemble_vs_old_ensemble", float("nan")),
              w.get("pearson_new_splade_vs_old_splade", float("nan")), rep["within_row_std_ratio_new_to_old_splade_median"] or float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
