#!/usr/bin/env python
"""Split generated synthetic pairs into verification batches, and merge verdicts back.

  split : datasets/synth/generated.jsonl -> datasets/synth/ver_in/batch_NNN.jsonl
          (chunk_id, text, query); rows already flagged by the validator are
          excluded, they are not worth a judge call.
  merge : datasets/synth/ver_out/batch_NNN.jsonl (chunk_id + verdict fields)
          -> datasets/synth/verified.jsonl with the verdict attached and a
          `clean` boolean: odpowiadalne & polszczyzna & sensowne_zapytanie &
          not zbyt_ogolne & not kopiuje_pasaz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AXES = ("odpowiadalne", "polszczyzna", "sensowne_zapytanie", "zbyt_ogolne", "kopiuje_pasaz")


def split(args: argparse.Namespace) -> None:
    """One verification file per generation batch (same name), unflagged rows only.

    Files that already exist are left alone, so the split can be re-run as more
    generation batches land without disturbing verification runs in flight.
    """
    rows = [json.loads(l) for l in args.generated.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("query") and not r.get("flags")]
    args.ver_in.mkdir(parents=True, exist_ok=True)
    by_batch: dict[str, list[dict]] = {}
    for r in rows:
        by_batch.setdefault(r.get("batch", "batch_unk"), []).append(r)
    new = 0
    for name, batch_rows in sorted(by_batch.items()):
        p = args.ver_in / f"{name}.jsonl"
        if p.exists():
            continue
        with p.open("w", encoding="utf-8") as fh:
            for r in batch_rows:
                fh.write(json.dumps({"chunk_id": r["chunk_id"], "text": r["text"], "query": r["query"]}, ensure_ascii=False) + "\n")
        new += 1
    print(f"{len(rows)} unflagged rows in {len(by_batch)} batches; wrote {new} new files to {args.ver_in}")


def merge(args: argparse.Namespace) -> None:
    verdicts = {}
    bad = 0
    for p in sorted(args.ver_out.glob("batch_*.jsonl")):
        for l in p.open(encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                bad += 1
                continue
            if "chunk_id" in r:
                verdicts[r["chunk_id"]] = r
    rows = [json.loads(l) for l in args.generated.open(encoding="utf-8") if l.strip()]
    out, n_clean, n_judged = [], 0, 0
    counts = {a: 0 for a in AXES}
    for r in rows:
        v = verdicts.get(r["chunk_id"])
        if v is None:
            continue
        n_judged += 1
        verdict = {a: bool(v.get(a, False)) for a in AXES}
        for a in AXES:
            counts[a] += int(verdict[a])
        clean = verdict["odpowiadalne"] and verdict["polszczyzna"] and verdict["sensowne_zapytanie"] \
            and not verdict["zbyt_ogolne"] and not verdict["kopiuje_pasaz"]
        n_clean += int(clean)
        r2 = dict(r)
        r2["verdict"] = verdict
        r2["glowny_problem"] = v.get("glowny_problem", "")
        r2["clean"] = clean
        out.append(r2)
    with args.verified.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"judged {n_judged} of {len(rows)} generated rows (bad json lines: {bad}); clean: {n_clean} "
          f"({100 * n_clean / max(1, n_judged):.1f}%)")
    print("axis true-counts:", counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("split", "merge"))
    ap.add_argument("--generated", type=Path, default=Path("datasets/synth/generated.jsonl"))
    ap.add_argument("--ver-in", type=Path, default=Path("datasets/synth/ver_in"))
    ap.add_argument("--ver-out", type=Path, default=Path("datasets/synth/ver_out"))
    ap.add_argument("--verified", type=Path, default=Path("datasets/synth/verified.jsonl"))
    ap.add_argument("--batch-size", type=int, default=40)
    args = ap.parse_args()
    split(args) if args.cmd == "split" else merge(args)


if __name__ == "__main__":
    main()
