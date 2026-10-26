#!/usr/bin/env python
"""Pick the chunks worth writing synthetic queries for, and split them into batches.

Random Wikipedia skews towards biographies, sports results and places, which is
exactly the entity-heavy material that made trivia-style fine-tunes collapse
(queries end up copying rare names). This keeps the more conceptual half:
chunks with a low density of mid-sentence capitalised tokens and of digits,
whose title does not look like a person's name, one chunk per article.

Writes datasets/synth/gen_in/batch_NNN.jsonl with {chunk_id, title, text}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)
SENT_START = re.compile(r"(?:^|[.!?]\s+)(\w+)")


def person_like_title(title: str) -> bool:
    parts = title.split()
    if not 2 <= len(parts) <= 3:
        return False
    return all(p[:1].isupper() and p.isalpha() for p in parts)


def entity_density(text: str) -> tuple[float, float]:
    tokens = WORD.findall(text)
    if not tokens:
        return 1.0, 1.0
    starts = set(SENT_START.findall(text))
    caps = sum(1 for t in tokens if t[:1].isupper() and t not in starts and not t.isdigit())
    digits = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    return caps / len(tokens), digits / len(tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=Path("datasets/synth/chunks_pool.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("datasets/synth/gen_in"))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--max-cap-ratio", type=float, default=0.10)
    ap.add_argument("--max-digit-ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.chunks.open(encoding="utf-8")]
    kept, seen = [], set()
    stats = {"person_title": 0, "caps": 0, "digits": 0, "dup_article": 0}
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for r in rows:
        if person_like_title(r["title"]):
            stats["person_title"] += 1
            continue
        cap, dig = entity_density(r["text"])
        if cap > args.max_cap_ratio:
            stats["caps"] += 1
            continue
        if dig > args.max_digit_ratio:
            stats["digits"] += 1
            continue
        if r["article_id"] in seen:
            stats["dup_article"] += 1
            continue
        seen.add(r["article_id"])
        r["cap_ratio"], r["digit_ratio"] = round(cap, 3), round(dig, 3)
        kept.append(r)
        if len(kept) >= args.n:
            break
    print(f"candidates {len(rows)}, kept {len(kept)}, dropped {stats}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("batch_*.jsonl"):
        old.unlink()
    for b in range(0, len(kept), args.batch_size):
        batch = kept[b : b + args.batch_size]
        with (args.out_dir / f"batch_{b // args.batch_size:03d}.jsonl").open("w", encoding="utf-8") as fh:
            for r in batch:
                fh.write(json.dumps({"chunk_id": r["chunk_id"], "title": r["title"], "text": r["text"]}, ensure_ascii=False) + "\n")
    with (args.out_dir.parent / "chunks_selected.jsonl").open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {(len(kept) + args.batch_size - 1) // args.batch_size} batches to {args.out_dir}")
    print("sample titles:", [r["title"] for r in kept[:15]])


if __name__ == "__main__":
    main()
