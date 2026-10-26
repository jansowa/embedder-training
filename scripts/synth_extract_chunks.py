#!/usr/bin/env python
"""Sample native-Polish Wikipedia chunks from PolQA for synthetic-query generation.

PolQA passages are ~50 words. Consecutive passages of one article are glued into
chunks of TARGET_MIN..TARGET_MAX words, so the chunks are about as long as the
FiQA positives (~170 words) the fine-tune that works was trained on.

Any chunk sharing a 12-word shingle with a passage of a tiny or held-out-8 PIRB
task is dropped, so the synthetic set cannot leak evaluation documents.

Output: one JSON per line with {chunk_id, article_id, title, text, n_words}.
A separate, larger pool (for negative mining) is written next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)
SHINGLE = 12
SHINGLE_STEP = 4

EVAL_TASKS = [
    "maupqa-gpt3-cc", "maupqa-mkqa", "maupqa-poleval2021-pairs", "maupqa-templates",
    "arguana-pl", "nfcorpus-pl", "scifact-pl", "gpt-exams",
    "maupqa-gpt3.5-cc", "maupqa-multilingual-NLI", "mfaq", "maupqa-wiki-def",
    "scidocs-pl", "maupqa-1z10", "fiqa-pl",
]
EVAL_EXTRA_PATHS = [
    "third_party/pirb/data/poleval-2022/allegro-faq/passages/passages.jsonl",
    "third_party/pirb/data/poleval-2022/legal-questions/passages/passages.jsonl",
]


def norm_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return WORD.findall(text)


def shingles(words: list[str], step: int = SHINGLE_STEP):
    for i in range(0, max(1, len(words) - SHINGLE + 1), step):
        yield hashlib.blake2b(" ".join(words[i : i + SHINGLE]).encode(), digest_size=8).digest()


def all_shingles(words: list[str]):
    for i in range(0, max(1, len(words) - SHINGLE + 1)):
        yield hashlib.blake2b(" ".join(words[i : i + SHINGLE]).encode(), digest_size=8).digest()


def load_eval_shingles(repo: Path) -> set[bytes]:
    paths = [repo / "third_party/pirb/data" / t / "passages/passages.jsonl" for t in EVAL_TASKS]
    paths += [repo / p for p in EVAL_EXTRA_PATHS]
    seen: set[bytes] = set()
    n = 0
    for p in paths:
        if not p.exists():
            print(f"[warn] missing eval corpus {p}")
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                text = row.get("contents") or row.get("text") or ""
                for h in shingles(norm_words(text)):
                    seen.add(h)
                n += 1
    print(f"eval passages: {n}, shingles: {len(seen)}")
    return seen


def article_id(pid: str) -> str:
    return pid.rsplit("-", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-generate", type=int, default=4000)
    ap.add_argument("--n-pool", type=int, default=60000)
    ap.add_argument("--min-words", type=int, default=120)
    ap.add_argument("--max-words", type=int, default=250)
    ap.add_argument("--article-sample-rate", type=float, default=0.02,
                    help="Fraction of articles considered at all (speed).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo", type=Path, default=Path("."))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    eval_sh = load_eval_shingles(args.repo)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    dropped_leak = 0
    cur_art = None
    cur_title = ""
    buf: list[str] = []
    buf_words = 0
    keep_article = False

    def flush():
        nonlocal buf, buf_words, dropped_leak
        if buf and buf_words >= args.min_words:
            text = " ".join(buf)
            words = norm_words(text)
            if any(h in eval_sh for h in all_shingles(words)):
                dropped_leak += 1
            else:
                chunks.append({"article_id": cur_art, "title": cur_title, "text": text, "n_words": len(words)})
        buf, buf_words = [], 0

    with args.passages.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pid = str(row["id"])
            art = article_id(pid)
            if art != cur_art:
                if keep_article:
                    flush()
                cur_art, cur_title = art, row.get("title", "")
                keep_article = rng.random() < args.article_sample_rate
                buf, buf_words = [], 0
            if not keep_article:
                continue
            text = row["text"].strip()
            nw = len(norm_words(text))
            if nw < 15:
                continue
            if buf_words + nw > args.max_words and buf_words >= args.min_words:
                flush()
            buf.append(text)
            buf_words += nw
            if buf_words >= args.max_words:
                flush()
        if keep_article:
            flush()

    print(f"chunks built: {len(chunks)}, dropped for eval overlap: {dropped_leak}")
    rng.shuffle(chunks)
    # one chunk per article for the generation set, so topics are diverse
    seen_art: set[str] = set()
    gen: list[dict] = []
    rest: list[dict] = []
    for c in chunks:
        if c["article_id"] not in seen_art and len(gen) < args.n_generate:
            seen_art.add(c["article_id"])
            gen.append(c)
        else:
            rest.append(c)
    pool = gen + rest[: max(0, args.n_pool - len(gen))]
    for i, c in enumerate(gen):
        c["chunk_id"] = f"pq-{i:05d}"
    for j, c in enumerate(pool[len(gen):], start=len(gen)):
        c["chunk_id"] = f"pq-{j:05d}"
    with (args.out_dir / "chunks_generate.jsonl").open("w", encoding="utf-8") as fh:
        for c in gen:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (args.out_dir / "chunks_pool.jsonl").open("w", encoding="utf-8") as fh:
        for c in pool:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    import statistics as st
    print(f"generate set: {len(gen)} (mean words {st.mean(c['n_words'] for c in gen):.0f}), pool: {len(pool)}")


if __name__ == "__main__":
    main()
