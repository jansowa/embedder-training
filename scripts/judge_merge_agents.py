#!/usr/bin/env python
"""Merge the Opus-subagent verdicts on the FiQA pairs and relate them to the split arms.

Inputs : datasets/judge_fiqa/in/batch_*.jsonl   (id, query, passage)
         datasets/judge_fiqa/out/batch_*.jsonl  (id + four axes + glowny_problem)
         datasets/arms/fiqa-2073.jsonl, datasets/arms/splits/fiqa-*.jsonl,
         results/base_loss/fiqa-2073.jsonl, results/lexical_difficulty/full/fiqa-2073.jsonl
Outputs: results/judge/fiqa-2073.agent-opus.judged.jsonl  (rows + judge.pos[0] + clean, same layout as judge_pairs.py)
         results/judge/fiqa-2073.agent-opus.report.json   (axis rates overall, per split arm, by base-loss and coverage deciles)
Prints the report as a table. Missing batches are reported, not fatal.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

AXES = ("odpowiadalne", "polszczyzna", "sensowne_zapytanie", "zbyt_ogolne")


def read_jsonl(p: Path) -> list[dict]:
    out = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def to_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return None


def main() -> None:
    inp = {p.name: read_jsonl(p) for p in sorted(Path("datasets/judge_fiqa/in").glob("batch_*.jsonl"))}
    verdicts: dict[str, dict] = {}
    missing, bad = [], 0
    for name, rows in inp.items():
        outp = Path("datasets/judge_fiqa/out") / name
        if not outp.exists():
            missing.append(name); continue
        for v in read_jsonl(outp):
            rec = {a: to_bool(v.get(a)) for a in AXES}
            if any(x is None for x in rec.values()) or "id" not in v:
                bad += 1; continue
            rec["glowny_problem"] = str(v.get("glowny_problem", ""))
            verdicts[str(v["id"])] = rec
    print(f"batches: {len(inp)} in, {len(inp) - len(missing)} judged, {len(verdicts)} verdicts, {bad} malformed lines, missing: {missing or 'none'}")

    rows = read_jsonl(Path("datasets/arms/fiqa-2073.jsonl"))
    loss = {str(r["query_id"]): r for r in read_jsonl(Path("results/base_loss/fiqa-2073.jsonl"))}
    lex = {str(r["query_id"]): r for r in read_jsonl(Path("results/lexical_difficulty/full/fiqa-2073.jsonl"))}
    Path("results/judge").mkdir(parents=True, exist_ok=True)
    with Path("results/judge/fiqa-2073.agent-opus.judged.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            v = verdicts.get(str(r["query_id"]))
            clean = None if v is None else (v["odpowiadalne"] and v["polszczyzna"] and v["sensowne_zapytanie"] and not v["zbyt_ogolne"])
            fh.write(json.dumps({**r, "judge": {"pos": [v], "neg": []}, "clean": clean}, ensure_ascii=False) + "\n")

    def rates(ids):
        vs = [verdicts[i] for i in ids if i in verdicts]
        if not vs:
            return {"n": 0}
        out = {"n": len(vs), "judged_frac": len(vs) / max(1, len(ids))}
        for a in AXES:
            out[a] = sum(v[a] for v in vs) / len(vs)
        out["clean"] = sum(v["odpowiadalne"] and v["polszczyzna"] and v["sensowne_zapytanie"] and not v["zbyt_ogolne"] for v in vs) / len(vs)
        return out

    report = {"overall": rates([str(r["query_id"]) for r in rows]), "splits": {}, "by_loss_decile": {}, "by_coverage_decile": {}}
    for p in sorted(Path("datasets/arms/splits").glob("fiqa-*.jsonl")):
        report["splits"][p.stem] = rates([str(r["query_id"]) for r in read_jsonl(p)])
    ids = [str(r["query_id"]) for r in rows if str(r["query_id"]) in loss]
    ids.sort(key=lambda i: loss[i]["loss_own"])
    for d in range(10):
        chunk = ids[d * len(ids) // 10:(d + 1) * len(ids) // 10]
        lo, hi = loss[chunk[0]]["loss_own"], loss[chunk[-1]]["loss_own"]
        report["by_loss_decile"][f"d{d}"] = {"loss_own_range": [lo, hi], **rates(chunk)}
    ids = [str(r["query_id"]) for r in rows if str(r["query_id"]) in lex and lex[str(r["query_id"])].get("idf_coverage") is not None]
    ids.sort(key=lambda i: lex[i]["idf_coverage"])
    for d in range(10):
        chunk = ids[d * len(ids) // 10:(d + 1) * len(ids) // 10]
        report["by_coverage_decile"][f"d{d}"] = {"idf_coverage_range": [lex[chunk[0]]["idf_coverage"], lex[chunk[-1]]["idf_coverage"]], **rates(chunk)}
    # answerable vs not: base-loss and coverage medians
    ans = [i for i in verdicts if verdicts[i]["odpowiadalne"] and i in loss]
    nans = [i for i in verdicts if not verdicts[i]["odpowiadalne"] and i in loss]
    report["answerable_vs_not"] = {
        "n_answerable": len(ans), "n_not": len(nans),
        "loss_own_mean": {"answerable": statistics.mean(loss[i]["loss_own"] for i in ans) if ans else None, "not": statistics.mean(loss[i]["loss_own"] for i in nans) if nans else None},
        "margin_own_median": {"answerable": statistics.median(loss[i]["margin_own"] for i in ans) if ans else None, "not": statistics.median(loss[i]["margin_own"] for i in nans) if nans else None},
        "idf_coverage_median": {"answerable": statistics.median(lex[i]["idf_coverage"] for i in ans if i in lex and lex[i].get("idf_coverage") is not None) if ans else None,
                                "not": statistics.median(lex[i]["idf_coverage"] for i in nans if i in lex and lex[i].get("idf_coverage") is not None) if nans else None},
    }
    Path("results/judge/fiqa-2073.agent-opus.report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def fmt(r):
        return "  ".join(f"{a[:6]}={100 * r[a]:5.1f}%" for a in (*AXES, "clean")) if r.get("n") else "no verdicts"
    print(f"overall (n={report['overall'].get('n', 0)}): {fmt(report['overall'])}")
    for k, r in report["splits"].items():
        print(f"{k:16s} (n={r.get('n', 0):4d}): {fmt(r)}")
    print("by base-loss decile (low -> high):")
    for k, r in report["by_loss_decile"].items():
        print(f"  {k} loss {r['loss_own_range'][0]:.3f}..{r['loss_own_range'][1]:.3f}: {fmt(r)}")
    print("by idf_coverage decile (low -> high):")
    for k, r in report["by_coverage_decile"].items():
        print(f"  {k} cov {r['idf_coverage_range'][0]:.2f}..{r['idf_coverage_range'][1]:.2f}: {fmt(r)}")
    print("answerable vs not:", json.dumps(report["answerable_vs_not"]))


if __name__ == "__main__":
    main()
