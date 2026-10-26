#!/usr/bin/env python
"""Validate generated synthetic queries against their input batches and merge them.

Checks per row: JSON shape, chunk_id order, form cycle, length limits per form,
and the "no more than 3 consecutive words copied from the passage" rule
(token level, case-insensitive). Rows with a violation are kept but flagged so
the verification step can see them; the summary prints how many each batch has.

Writes datasets/synth/generated.jsonl with chunk_id, title, text, form, query, flags.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)
FORMS = ("pytanie", "pytanie", "fraza", "problem")
LIMITS = {"pytanie": (5, 18), "fraza": (2, 6), "problem": (6, 20)}


def longest_copied_run(query: str, passage: str) -> int:
    q = [w.lower() for w in WORD.findall(query)]
    p = [w.lower() for w in WORD.findall(passage)]
    pset = {}
    for i, w in enumerate(p):
        pset.setdefault(w, []).append(i)
    best = 0
    for i, w in enumerate(q):
        for j in pset.get(w, []):
            k = 0
            while i + k < len(q) and j + k < len(p) and q[i + k] == p[j + k]:
                k += 1
            best = max(best, k)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("datasets/synth/gen_in"))
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/synth/gen_out"))
    ap.add_argument("--merged", type=Path, default=Path("datasets/synth/generated.jsonl"))
    ap.add_argument("--max-copy", type=int, default=3)
    args = ap.parse_args()

    merged = []
    missing = []
    for inp in sorted(args.in_dir.glob("batch_*.jsonl")):
        outp = args.out_dir / inp.name
        if not outp.exists():
            missing.append(inp.name)
            continue
        rows_in = [json.loads(l) for l in inp.open(encoding="utf-8") if l.strip()]
        rows_out = []
        bad_json = 0
        for l in outp.open(encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                rows_out.append(json.loads(l))
            except json.JSONDecodeError:
                bad_json += 1
        by_id = {r.get("chunk_id"): r for r in rows_out}
        flags_count = {}
        for pos, rin in enumerate(rows_in):
            rout = by_id.get(rin["chunk_id"])
            flags = []
            if rout is None:
                flags.append("missing")
                query, form = "", ""
            else:
                query = str(rout.get("query", "")).strip()
                form = str(rout.get("form", "")).strip()
                expected = FORMS[pos % 4]
                if form != expected:
                    flags.append(f"form:{form or '?'}!={expected}")
                nw = len(WORD.findall(query))
                lo, hi = LIMITS.get(expected, (1, 30))
                if not lo <= nw <= hi:
                    flags.append(f"len:{nw}")
                if expected == "pytanie" and not query.endswith("?"):
                    flags.append("no_qmark")
                run = longest_copied_run(query, rin["text"])
                if run > args.max_copy:
                    flags.append(f"copy:{run}")
                if re.search(r"\b(tekst|tekście|fragment|fragmencie|artykuł|artykule|autor)\b", query, re.IGNORECASE):
                    flags.append("meta_ref")
                if rout.get("flag"):
                    flags.append(str(rout["flag"]))
            for f in flags:
                flags_count[f.split(":")[0]] = flags_count.get(f.split(":")[0], 0) + 1
            merged.append({"chunk_id": rin["chunk_id"], "title": rin["title"], "text": rin["text"],
                           "form": form, "query": query, "flags": flags, "batch": inp.stem})
        print(f"{inp.name}: in={len(rows_in)} out={len(rows_out)} bad_json={bad_json} flags={flags_count}")
    if missing:
        print(f"missing outputs: {len(missing)} ({missing[:5]}...)")
    args.merged.parent.mkdir(parents=True, exist_ok=True)
    with args.merged.open("w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    clean = sum(1 for r in merged if not r["flags"])
    print(f"merged {len(merged)} rows -> {args.merged}; without flags: {clean}")


if __name__ == "__main__":
    main()
