#!/usr/bin/env python
"""Markdown reading of one arm against a reference arm: result table, verdict, task groups, per-task tables.

The queues that add rows to an arm all end the same way: the new arm is compared with a reference at a comparable
index cost by a fixed threshold on tiny10 and heldout8, and the per-task tables show where the difference sits.
This script prints that part of a summary and writes the verdict as JSON, so a queue can branch on it.

    python scripts/arm_reading.py --regimes experiments/<series>/results-regimes/quantized \\
        --sparsity experiments/<series>/results/sparsity --cost experiments/<series>/results/index_cost_v60.json \\
        --column "20 k arm=arm-a-bias013" --column "dose3=arm-b-bias015" --column "dose3 + FAQ=arm-c-bias015" \\
        --pick arm-c-bias015 --ref arm-b-bias015 --group "FAQ tasks=mfaq,poleval-2022-test-A-allegro-faq" \\
        --m23 arm-a-bias013 --m23 arm-c-bias015 --json verdict.json

A group is `NAME=task,task`; a task ending in `*` is a prefix. The deltas of the group table are against the first
column. The 23-task mean is (10 x tiny10 + 8 x heldout8 + the five held-out-2 tasks) / 23.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

H2 = ["czywiesz", "gpt35wiki", "quora", "poquad", "treccovid"]
SETS = [("tiny10", 10), ("heldout8", 8)]


def load(path: Path) -> dict:
    try:
        return json.load(open(path))
    except Exception:
        return {}


def fmt(x, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def sign(x) -> str:
    return "-" if x is None else f"{x:+.2f}"


class Results:
    def __init__(self, regimes: Path, sparsity: Path, cost: Path | None):
        self.regimes, self.sparsity = regimes, sparsity
        self.cost = load(cost).get("models", {}) if cost else {}

    def res(self, label: str, taskset: str) -> dict:
        return load(self.regimes / label / f"{taskset}.json")

    def val(self, label: str, taskset: str, n: int | None = None):
        d = self.res(label, taskset)
        return d.get("mean_ndcg@10") if d and (n is None or d.get("n_tasks") == n) else None

    def tasks(self, label: str, tasksets: list[str]) -> dict:
        out: dict = {}
        for ts in tasksets:
            out.update({k: v for k, v in (self.res(label, ts).get("tasks") or {}).items() if v is not None})
        return out

    def dims(self, label: str):
        try:
            d = load(self.sparsity / f"{label}.json")[label]
            v = [t["mean"] for t in d.values() if t.get("mean") is not None]
            return sum(v) / len(v)
        except Exception:
            return None

    def c(self, label: str, key: str, nd: int = 0) -> str:
        s = self.cost.get(label)
        return fmt(s[key], nd) if s else "-"


def in_group(task: str, members: list[str]) -> bool:
    return any(task.startswith(m[:-1]) if m.endswith("*") else task == m for m in members)


def mean(values: list[float]):
    return sum(values) / len(values) if values else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regimes", type=Path, required=True, help="results-regimes/<regime> directory")
    ap.add_argument("--sparsity", type=Path, required=True)
    ap.add_argument("--cost", type=Path, default=None, help="index cost JSON of scripts/splade_index_cost.py")
    ap.add_argument("--column", action="append", default=[], metavar="NAME=LABEL", help="models of the tables, in order")
    ap.add_argument("--pick", required=True, help="label of the new arm (its measured form)")
    ap.add_argument("--ref", required=True, help="label of the reference the verdict is against")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=task,task")
    ap.add_argument("--extra-set", action="append", default=[], help="more task sets for the per-task data, e.g. heldout2_czywiesz")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--win", default="THE NEW ARM WINS")
    ap.add_argument("--noise", default="within the seed noise")
    ap.add_argument("--loss", default="a LOSS on both sets")
    ap.add_argument("--m23", action="append", default=[], metavar="LABEL", help="rows of the 23-task table")
    ap.add_argument("--json", type=Path, default=None, help="write the verdict here")
    args = ap.parse_args(argv)

    r = Results(args.regimes, args.sparsity, args.cost)
    columns = [(c.split("=", 1)[0], c.split("=", 1)[1]) for c in args.column]
    for name, label in (("reference", args.ref), ("new arm", args.pick)):
        if label not in [lab for _, lab in columns]:
            columns.append((name, label))
    names = {label: name for name, label in columns}
    pick, ref = args.pick, args.ref

    print("## Result\n")
    print("| model | rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, label in columns:
        if r.val(label, "tiny10") is None:
            continue
        shown = f"**{label}**" if label == pick else label
        print(f"| {shown} | {name} | {fmt(r.val(label, 'tiny10'))} | {fmt(r.val(label, 'heldout8'))} | {fmt(r.val(label, 'heldout2_treccovid'))} | "
              f"{fmt(r.dims(label), 1)} | {r.c(label, 'doc_active_mean', 1)} | {r.c(label, 'query_active_mean', 1)} | {r.c(label, 'postings_per_query')} |")

    t, h, d = r.val(pick, "tiny10", 10), r.val(pick, "heldout8", 8), r.dims(pick)
    rt, rh, rd = r.val(ref, "tiny10", 10), r.val(ref, "heldout8", 8), r.dims(ref)
    verdict = {"pick": pick, "ref": ref, "threshold": args.threshold, "verdict": "incomplete"}
    if None in (t, h, rt, rh):
        text = f"{pick} has no complete result on tiny10 and heldout8 (or the reference {ref} has none)"
    else:
        dt, dh = t - rt, h - rh
        kind = "win" if dt >= args.threshold and dh >= args.threshold else "loss" if dt <= -args.threshold and dh <= -args.threshold else "noise"
        word = {"win": args.win, "loss": args.loss, "noise": args.noise}[kind]
        verdict.update({"verdict": kind, "tiny10": t, "heldout8": h, "dims_per_doc": d, "ref_tiny10": rt, "ref_heldout8": rh, "ref_dims_per_doc": rd,
                        "delta_tiny10": dt, "delta_heldout8": dh, "trec_covid": r.val(pick, "heldout2_treccovid"), "ref_trec_covid": r.val(ref, "heldout2_treccovid")})
        text = ("%s %.2f / %.2f at %s dims/doc against %s %s %.2f / %.2f at %s: delta %+.2f / %+.2f, trec-covid %s against %s -> %s"
                % (pick, t, h, fmt(d, 1), names[ref], ref, rt, rh, fmt(rd, 1), dt, dh, fmt(r.val(pick, "heldout2_treccovid")), fmt(r.val(ref, "heldout2_treccovid")), word))
    print("\n## Reading\n\n" + text + ".")
    print("\nSeed noise of this recipe is 0.13-0.2 (one seed here).")

    tasksets = [ts for ts, _ in SETS] + list(args.extra_set)
    per = {label: r.tasks(label, tasksets) for _, label in columns}
    base_name, base = columns[0]
    others = [(n, lab) for n, lab in columns[1:] if per[lab]]
    if args.group and per[base] and others:
        print(f"\n## Task groups (mean delta against {base_name}; tasks both models have)\n")
        print("| group | tasks | " + " | ".join(n for n, _ in others) + " |\n|---|---:|" + "---:|" * len(others))
        for g in args.group:
            gname, members = g.split("=", 1)
            members = [m.strip() for m in members.split(",") if m.strip()]
            cells, n_tasks = [], 0
            for _, lab in others:
                deltas = [per[lab][k] - per[base][k] for k in per[lab] if k in per[base] and in_group(k, members)]
                n_tasks = max(n_tasks, len(deltas))
                cells.append(sign(mean(deltas)) + (f" ({len(deltas)})" if deltas and len(deltas) != n_tasks else ""))
            print(f"| {gname} | {n_tasks} | " + " | ".join(cells) + " |")
        verdict["groups_vs_first_column"] = {
            g.split("=", 1)[0]: {n: mean([per[lab][k] - per[base][k] for k in per[lab] if k in per[base] and in_group(k, [m.strip() for m in g.split("=", 1)[1].split(",")])])
                                 for n, lab in others} for g in args.group}

    for ts in tasksets:
        rows = r.res(pick, ts).get("tasks") or r.res(ref, ts).get("tasks") or {}
        if not rows:
            continue
        shown = [(n, lab) for n, lab in columns if r.res(lab, ts).get("tasks")]
        print(f"\n### {ts} per task\n\n| task | " + " | ".join(n for n, _ in shown) + f" | {names[pick]} - {names[ref]} |\n|---|" + "---:|" * (len(shown) + 1))
        for k in rows:
            vals = {lab: (r.res(lab, ts).get("tasks") or {}).get(k) for _, lab in shown}
            delta = vals.get(pick) - vals.get(ref) if vals.get(pick) is not None and vals.get(ref) is not None else None
            print(f"| {k} | " + " | ".join(fmt(vals[lab]) for _, lab in shown) + f" | {sign(delta)} |")

    if args.m23:
        print("\n## 23 tasks\n\n23-task mean = (10 x tiny10 + 8 x heldout8 + the five held-out-2 tasks) / 23.\n")
        print("| model | tiny10 | heldout8 | " + " | ".join(H2) + " | held-out-2 | 23 tasks | dims/doc | cost: dims/query | cost: postings/query |\n|---|---:|---:|" + "---:|" * (len(H2) + 5))
        verdict["m23"] = {}
        for label in dict.fromkeys(args.m23):
            a, b = r.val(label, "tiny10", 10), r.val(label, "heldout8", 8)
            h2 = [r.val(label, f"heldout2_{x}", 1) for x in H2]
            full = None not in (a, b) and all(x is not None for x in h2)
            m23 = (10 * a + 8 * b + sum(h2)) / 23 if full else None
            verdict["m23"][label] = m23
            shown = f"**{label}**" if label == pick else label
            print(f"| {shown} | {fmt(a)} | {fmt(b)} | " + " | ".join(fmt(x) for x in h2) + f" | {fmt(sum(h2) / 5 if all(x is not None for x in h2) else None)} | {fmt(m23)} | "
                  f"{fmt(r.dims(label), 1)} | {r.c(label, 'query_active_mean', 1)} | {r.c(label, 'postings_per_query')} |")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
