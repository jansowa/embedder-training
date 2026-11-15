#!/usr/bin/env python
"""Diagnostics of a static query encoder: upper bounds (oracles) and vocabulary coverage. Nothing here trains.

The oracles say where the distance between a static table and the full SPLADE query side comes from, before any
experiment is built. Each one mixes the teacher's query vector ``t`` (the document model run on the query) with the
table's vector ``s``:

  weights      ``t`` on the dimensions the table activates: the ceiling of anything that only re-weights the
               table's slots in context (gates). What is still missing to the teacher is the value of dimensions
               the table cannot reach: the ceiling of new slots and new rows.
  support      ``s`` on the dimensions the teacher activates: the ceiling of pure noise suppression (residual
               rows, a threshold, a word row replacing the rows of its pieces).
  own          ``t`` on the query's own tokens: the full model without query-side expansion.
  student-own  ``s`` on the query's own tokens: the table without its expansions.

``oracles`` measures them on the held-out rows of a mined pool (the validation numbers of ``distill.train_static``:
agreement with the teacher's ranking of each row's candidates); ``oracle-model`` writes a folder that the PIRB fork
loads as a query model (``PIRB_QUERY_MODEL``), so the same four are measured as nDCG@10 on a task set.

``coverage`` reads only the tokenizer and the table (CPU, no model): how many words of a text sample fall back to
subword pieces because they own no row, how many adjacent word pairs own a row, how many token occurrences meet a
row that does not write the token's own dimension. Run it on query-like texts that the table was not built from.

Usage:
  .venv/bin/python -m distill.static_diag oracles --pool cache/static_pools/kd-X-r2 \\
      --model models/static-X/phr2-k8-kd-r2 --out results/static-X/oracles_phr2-k8-kd-r2.json
  .venv/bin/python -m distill.static_diag oracle-model --table models/static-X/phr2-k8-kd-r2 --teacher models/X \\
      --mode weights --out models/static-X/oracle-weights
  .venv/bin/python -m distill.static_diag coverage --model models/static-X/phr2-k8-kd-r2 \\
      --texts datasets/kd/kd_v1.jsonl --kinds query --n 20000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ORACLE_MODES = ("weights", "support", "own", "student-own")
ORACLE_CONFIG = "static_oracle.json"


def combine_oracle(mode: str, teacher: torch.Tensor, student: torch.Tensor, own: torch.Tensor, min_weight: float = 0.005) -> torch.Tensor:
    """``[B, V]`` query vectors of one oracle. ``own`` [B, V] bool marks each query's own token dimensions; a
    dimension counts as active from ``min_weight`` on (half a quantization step: what survives ``round(w * 100)``)."""
    if mode == "weights":
        return teacher * (student >= min_weight)
    if mode == "support":
        return student * (teacher >= min_weight)
    if mode == "own":
        return teacher * own
    if mode == "student-own":
        return student * own
    raise ValueError(f"unknown oracle mode {mode!r}; one of {ORACLE_MODES}.")


def own_token_mask(tokenizer, texts: list[str], dim: int, max_length: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """``[B, dim]`` bool: the vocabulary dimensions of each text's own tokens (special tokens are not the text's)."""
    enc = tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=False)
    mask = torch.zeros(len(texts), dim, dtype=torch.bool)
    for i, ids in enumerate(enc["input_ids"]):
        ids = [t for t in ids if t < dim]
        if ids:
            mask[i, torch.tensor(ids)] = True
    return mask.to(device)


def pool_oracles(pool_dir: Path, model_dir: Path, val_rows: int = 2000, seed: int = 42, device: str = "cpu", batch_size: int = 128) -> dict[str, Any]:
    """The student, the teacher and the four oracles on the rows ``distill.train_static`` holds out with the same
    ``--val-rows`` and ``--seed`` (so the ``student`` line is that run's ``val_final``)."""
    from transformers import AutoTokenizer

    from distill.static_pool import load_pool
    from distill.static_query import load_static_encoder
    from distill.train_static import StaticTrainer

    pool = load_pool(pool_dir)
    if pool.queries is None:
        raise ValueError(f"{pool_dir} has no teacher query vectors (static_pool --teacher-queries, or a mined pool).")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_static_encoder(model_dir, device).eval()
    target = "rows" if any(r["scores"] is not None for r in pool.rows) else "splade"
    args = argparse.Namespace(target=target, learn_scale=False, teacher_scale=1.0, device=device)
    trainer = StaticTrainer(pool, model, tokenizer, args)
    ids = [i for i in range(len(pool.rows)) if trainer.usable(i)]
    random.Random(seed).shuffle(ids)
    val_ids = ids[:val_rows]
    half_step = 0.5 / trainer.qf

    def variant(mode: str):
        def query_fn(batch: dict[str, Any]) -> torch.Tensor:
            student = trainer.student_queries(batch["rows"])
            own = own_token_mask(tokenizer, [r["query"] for r in batch["rows"]], model.output_dim, model.max_length, device)
            return combine_oracle(mode, batch["teacher_q"], student, own, half_step)

        return query_fn

    result: dict[str, Any] = {
        "pool": str(pool_dir), "model": str(model_dir), "rows": len(val_ids), "seed": seed, "agreement_with": target,
        "student": trainer.evaluate(val_ids, batch_size=batch_size),
        "teacher": trainer.evaluate(val_ids, use_teacher_queries=True, batch_size=batch_size),
    }
    for mode in ORACLE_MODES:
        result[mode] = trainer.evaluate(val_ids, batch_size=batch_size, query_fn=variant(mode))
    return result


def write_oracle_model(out: Path, table: Path, teacher: Path, mode: str, min_weight: float = 0.005) -> dict[str, Any]:
    """A folder the PIRB fork recognises as a query model (``search/splade_static_ext.py``): the teacher and the table
    are loaded from their own folders, this one only names them and the oracle."""
    if mode not in ORACLE_MODES:
        raise ValueError(f"unknown oracle mode {mode!r}; one of {ORACLE_MODES}.")
    config = {"format": "static-oracle/1", "mode": mode, "table": str(Path(table).resolve()), "teacher": str(Path(teacher).resolve()), "min_weight": min_weight}
    out.mkdir(parents=True, exist_ok=True)
    (out / ORACLE_CONFIG).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return config


def coverage(model, tokenizer, texts: list[str], batch_size: int = 1024, show: int = 30) -> dict[str, Any]:
    """Word and pair coverage of a static encoder's rows on ``texts`` (tokenized as the encoder tokenizes queries)."""
    from distill.static_query import PHRASE_SEP, punct_token_mask, split_words

    punct = punct_token_mask(tokenizer)
    active = model.active_slot_mask().cpu().numpy()
    indices = model.indices.cpu().numpy()
    base = model.base_rows
    live = active.any(axis=1)
    self_slot = np.zeros(model.rows, dtype=bool)
    span = min(base, model.output_dim)
    self_slot[:span] = ((indices[:span] == np.arange(span)[:, None]) & active[:span]).any(axis=1)
    lookup = model._lookup
    pair_key = getattr(model, "pair_key", "exact")
    n = Counter()
    uncovered: Counter = Counter()
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start : start + batch_size], truncation=True, max_length=model.max_length, add_special_tokens=model.add_special_tokens)
        for i, ids in enumerate(enc["input_ids"]):
            words = split_words(ids, enc.word_ids(i), punct)
            n["texts"] += 1
            n["truncated"] += len(ids) >= model.max_length
            n["words"] += len(words)
            for key, _ in words:
                n["pieces"] += len(key)
                if len(key) > 1:
                    n["multi"] += 1
                    if key in lookup:
                        n["multi_with_row"] += 1
                    else:
                        uncovered[key] += 1
                        n["fallback_pieces"] += len(key)
                        n["fallback_pieces_live"] += int(sum(live[t] for t in key if t < base))
                        n["fallback_pieces_self"] += int(sum(self_slot[t] for t in key if t < base))
                else:
                    t = key[0]
                    n["single"] += 1
                    n["single_live"] += int(t < base and live[t])
                    n["single_self"] += int(t < base and self_slot[t])
            for j in range(len(words) - 1):
                left, right = words[j][0], words[j + 1][0]
                key = (left[0], PHRASE_SEP, right[0]) if pair_key == "first" else left + (PHRASE_SEP,) + right
                n["pairs"] += 1
                n["pairs_with_row"] += key in lookup

    def share(part: str, whole: str) -> float:
        return round(n[part] / n[whole], 4) if n[whole] else 0.0

    return {
        "texts": n["texts"],
        "rows": model.rows,
        "word_rows": sum(PHRASE_SEP not in key for key in model.phrases),
        "pair_rows": sum(PHRASE_SEP in key for key in model.phrases),
        "words_per_text": round(n["words"] / max(n["texts"], 1), 2),
        "pieces_per_word": round(n["pieces"] / max(n["words"], 1), 3),
        "texts_truncated": share("truncated", "texts"),
        "multi_piece_words": share("multi", "words"),
        "multi_piece_words_with_row": share("multi_with_row", "multi"),
        # the headline: words whose meaning the table can only read from context-free pieces
        "words_falling_back_to_pieces": round((n["multi"] - n["multi_with_row"]) / n["words"], 4) if n["words"] else 0.0,
        "fallback_pieces_with_live_row": share("fallback_pieces_live", "fallback_pieces"),
        "fallback_pieces_with_self_slot": share("fallback_pieces_self", "fallback_pieces"),
        "single_piece_words_with_live_row": share("single_live", "single"),
        "single_piece_words_with_self_slot": share("single_self", "single"),
        "adjacent_pairs_with_row": share("pairs_with_row", "pairs"),
        "top_uncovered_words": [[tokenizer.decode(list(key)).strip(), count] for key, count in uncovered.most_common(show)],
    }


def oracles_main(args: argparse.Namespace) -> int:
    try:
        result = pool_oracles(args.pool, args.model, args.val_rows, args.seed, args.device, args.batch_size)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    header = f"{'query side':14s} {'top1':>7s} {'tau':>7s} {'mrr':>7s} {'dims/q':>8s} {'post.frac':>10s}"
    print(header)
    for name in ("student", *ORACLE_MODES, "teacher"):
        r = result[name]
        print(f"{name:14s} {r['top1_agreement']:7.4f} {r['kendall_tau']:7.4f} {r['mrr']:7.4f} {r['dims_per_query']:8.1f} {r['postings_fraction']:10.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def oracle_model_main(args: argparse.Namespace) -> int:
    for path in (args.table, args.teacher):
        if not Path(path).is_dir():
            print(f"[ERROR] {path} is not a folder", file=sys.stderr)
            return 2
    print(json.dumps(write_oracle_model(args.out, args.table, args.teacher, args.mode, args.min_weight), indent=2, ensure_ascii=False))
    return 0


def coverage_main(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from distill.static_query import load_static_encoder, load_texts

    kinds = set(args.kinds.split(",")) if args.kinds else None
    texts = load_texts(args.texts, args.n, kinds, args.field, args.seed)
    if not texts:
        print("[ERROR] no texts", file=sys.stderr)
        return 2
    result = {"model": str(args.model), "texts_file": str(args.texts), "kinds": args.kinds, "seed": args.seed,
              **coverage(load_static_encoder(args.model), AutoTokenizer.from_pretrained(args.model), texts, show=args.show)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("oracles", help="student, teacher and the oracles on the held-out rows of a pool")
    o.add_argument("--pool", type=Path, required=True)
    o.add_argument("--model", type=Path, required=True, help="static query encoder folder")
    o.add_argument("--val-rows", type=int, default=2000)
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--batch-size", type=int, default=128)
    o.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    o.add_argument("--out", type=Path, default=None)
    m = sub.add_parser("oracle-model", help="write a query-model folder for the PIRB fork")
    m.add_argument("--table", type=Path, required=True)
    m.add_argument("--teacher", type=Path, required=True, help="the document model (its query side is the teacher)")
    m.add_argument("--mode", choices=ORACLE_MODES, required=True)
    m.add_argument("--min-weight", type=float, default=0.005)
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("coverage", help="word / pair coverage of the rows on a text sample (CPU, no model)")
    c.add_argument("--model", type=Path, required=True)
    c.add_argument("--texts", type=Path, required=True, help="JSON lines with a text field (kd corpus) or rows with --field query")
    c.add_argument("--field", default="text")
    c.add_argument("--kinds", default=None)
    c.add_argument("--n", type=int, default=20000)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--show", type=int, default=30, help="most frequent uncovered words to list")
    c.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    return {"oracles": oracles_main, "oracle-model": oracle_model_main, "coverage": coverage_main}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
