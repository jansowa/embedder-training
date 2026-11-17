#!/usr/bin/env python
"""Bookkeeping of the candidates of a queue that looks for a replacement of the frozen full model.

The candidates are lines of a TSV file (stage, name, label, raw model, description); every number is read from the
result files of the quantized regime, the sparsity files and the index-cost files. The score of the replacement rule is
the mean over the 19 tasks of tiny10, heldout8 and trec-covid, every task with the same weight, on the unrounded
numbers; `m11` is the same mean over tiny10 and trec-covid alone (for candidates without a heldout8 yet), `m23` adds
the four other held-out-2 tasks.

    python scripts/b2_candidates.py mean --regimes R LABEL m19
    python scripts/b2_candidates.py table --cands C.tsv --regimes R --sparsity S --cost-dir FORMS --cost COST.json --ref LABEL
    python scripts/b2_candidates.py best --cands C.tsv --regimes R --sparsity S --cost-dir FORMS --metric m19 --stages p1,p2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TREC = "heldout2_treccovid"
SETS = {"tiny10": 10, "heldout8": 8, TREC: 1, "heldout2_czywiesz": 1, "heldout2_gpt35wiki": 1, "heldout2_quora": 1, "heldout2_poquad": 1}
METRICS = {
    "m11": ["tiny10", TREC],
    "m19": ["tiny10", "heldout8", TREC],
    "m23": ["tiny10", "heldout8", "heldout2_czywiesz", "heldout2_gpt35wiki", "heldout2_quora", "heldout2_poquad", TREC],
}


def task_values(regimes: list[Path], label: str, taskset: str) -> list[float] | None:
    """nDCG@10 of every task of a task set, or None when the set is missing or incomplete."""
    for root in regimes:
        path = root / label / f"{taskset}.json"
        if path.exists():
            try:
                values = [v for v in json.loads(path.read_text())["tasks"].values() if v is not None]
            except (ValueError, KeyError, AttributeError):
                return None
            return values if len(values) == SETS.get(taskset, len(values)) else None
    return None


def mean(regimes: list[Path], label: str, sets: list[str]) -> float | None:
    values: list[float] = []
    for taskset in sets:
        got = task_values(regimes, label, taskset)
        if got is None:
            return None
        values += got
    return sum(values) / len(values) if values else None


def set_mean(regimes: list[Path], label: str, taskset: str) -> float | None:
    return mean(regimes, label, [taskset])


def dims_per_doc(sparsity: Path, label: str) -> float | None:
    path = sparsity / f"{label}.json"
    try:
        values = [t["mean"] for t in json.loads(path.read_text())[label].values() if t.get("mean") is not None]
    except (OSError, ValueError, KeyError, AttributeError):
        return None
    return sum(values) / len(values) if values else None


def query_cost(cost_dir: Path | None, cost_files: list[Path], label: str) -> tuple[float | None, float | None]:
    """(dims/query, postings/query) of a label from its own cost file or from one of the shared ones."""
    files = ([cost_dir / f"{label}.json"] if cost_dir else []) + list(cost_files)
    for path in files:
        try:
            m = json.loads(path.read_text())["models"].get(label)
        except (OSError, ValueError, KeyError, AttributeError):
            continue
        if m:
            return m.get("query_active_mean"), m.get("postings_per_query")
    return None, None


def read_cands(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if len(f) >= 5 and f[2]:
                rows.append({"stage": f[0], "name": f[1], "label": f[2], "raw": f[3], "desc": f[4]})
    return rows


def measure(args: argparse.Namespace, row: dict) -> dict:
    label = row["label"]
    dq, pq = query_cost(args.cost_dir, args.cost, label)
    out = dict(row, tiny10=set_mean(args.regimes, label, "tiny10"), heldout8=set_mean(args.regimes, label, "heldout8"), trec=set_mean(args.regimes, label, TREC),
               dims=dims_per_doc(args.sparsity, label), dims_query=dq, postings=pq)
    for name, sets in METRICS.items():
        out[name] = mean(args.regimes, label, sets)
    out["within"] = bool(out["dims"] is not None and out["dims"] <= args.max_dims and pq is not None and pq <= args.max_postings)
    return out


def fmt(value: float | None, pattern: str = "%.2f") -> str:
    return pattern % value if value is not None else "-"


def cmd_mean(args: argparse.Namespace) -> int:
    sets = METRICS.get(args.sets[0], args.sets) if len(args.sets) == 1 else args.sets
    value = mean(args.regimes, args.label, sets)
    if value is not None:
        print("%.3f" % value)
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    rows = [measure(args, r) for r in read_cands(args.cands)]
    if args.stages:
        keep = set(args.stages.split(","))
        rows = [r for r in rows if r["stage"] in keep or r["label"] == args.ref]
    ref = next((r for r in rows if r["label"] == args.ref), None)
    ref19 = ref["m19"] if ref else None
    ref11 = ref["m11"] if ref else None
    print("| candidate | measured form | tiny10 | heldout8 | trec-covid | mean of 19 tasks | vs the frozen model | mean of 11 tasks (tiny10 + trec-covid) | vs the frozen model | dims/doc | dims/query | postings/query | within the limits |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    order = sorted(rows, key=lambda r: (r["label"] != args.ref, -(r["m19"] if r["m19"] is not None else -1.0), -(r["m11"] if r["m11"] is not None else -1.0)))
    for r in order:
        d19 = fmt(r["m19"] - ref19, "%+.3f") if r["m19"] is not None and ref19 is not None and r is not ref else ("reference" if r is ref else "-")
        d11 = fmt(r["m11"] - ref11, "%+.3f") if r["m11"] is not None and ref11 is not None and r is not ref else ("reference" if r is ref else "-")
        bold = "**" if r["m19"] is not None and ref19 is not None and r is not ref and r["within"] and r["m19"] - ref19 >= args.margin else ""
        print("| %s%s%s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            bold, r["name"], bold, r["label"], fmt(r["tiny10"]), fmt(r["heldout8"]), fmt(r["trec"]), fmt(r["m19"], "%.3f"), d19, fmt(r["m11"], "%.3f"), d11,
            fmt(r["dims"], "%.1f"), fmt(r["dims_query"], "%.1f"), fmt(r["postings"], "%.0f"), "yes" if r["within"] else "no"))
    return 0


def cmd_best(args: argparse.Namespace) -> int:
    rows = [measure(args, r) for r in read_cands(args.cands)]
    if args.stages:
        keep = set(args.stages.split(","))
        rows = [r for r in rows if r["stage"] in keep]
    if args.name_prefix:
        rows = [r for r in rows if r["label"].startswith(tuple(args.name_prefix.split(",")))]
    rows = [r for r in rows if r[args.metric] is not None and (r["within"] or args.any_cost) and r["label"] not in set(args.exclude)]
    if rows:
        b = max(rows, key=lambda r: r[args.metric])
        print("%s\t%s\t%s\t%.3f" % (b["label"], b["raw"], b["name"], b[args.metric]))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--regimes", type=Path, action="append", required=True, help="results-regimes/<regime> directory (repeatable)")

    def lookup(p: argparse.ArgumentParser) -> None:
        common(p)
        p.add_argument("--cands", type=Path, required=True)
        p.add_argument("--sparsity", type=Path, required=True)
        p.add_argument("--cost-dir", type=Path, default=None, help="directory with one index-cost JSON per label")
        p.add_argument("--cost", type=Path, action="append", default=[], help="index-cost JSON with several models (repeatable)")
        p.add_argument("--max-dims", type=float, default=410.0)
        p.add_argument("--max-postings", type=float, default=16807.0)
        p.add_argument("--stages", default=None, help="comma-separated stages to keep")

    p = sub.add_parser("mean")
    common(p)
    p.add_argument("label")
    p.add_argument("sets", nargs="+", help="m11, m19, m23 or task-set names")
    p.set_defaults(func=cmd_mean)
    p = sub.add_parser("table")
    lookup(p)
    p.add_argument("--ref", required=True, help="label of the frozen model")
    p.add_argument("--margin", type=float, default=0.2)
    p.set_defaults(func=cmd_table)
    p = sub.add_parser("best")
    lookup(p)
    p.add_argument("--metric", choices=sorted(METRICS), default="m19")
    p.add_argument("--name-prefix", default=None, help="comma-separated label prefixes to keep")
    p.add_argument("--exclude", action="append", default=[], help="label to leave out (repeatable)")
    p.add_argument("--any-cost", action="store_true", help="do not require the cost limits")
    p.set_defaults(func=cmd_best)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
