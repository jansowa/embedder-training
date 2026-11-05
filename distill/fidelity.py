#!/usr/bin/env python
"""How closely a student reproduces a teacher's SPLADE vectors, and how fast it is.

For a sample of texts both models are encoded (eval mode, no gradient, the same tokenizer) and the
pooled vectors compared: cosine, overlap of the active dimensions as PIRB indexes them
(``round(w * 100) > 0``), relative L1 error, active dimensions per text, and the wall time per text
of each model on the same batches (encoder + head). Reported per kind (query / passage) when the
texts carry one.

Usage:
  .venv/bin/python -m distill.fidelity --teacher models/soup-v9b --student models/student-b4 \
      --texts corpus.jsonl --n 512 --max-seq-length 256 --device cuda --output results/fidelity.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from distill.splade import active_dims, encode_pooled, prepare_head_layer


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    ordered = sorted(values)

    def pct(q: float) -> float:
        pos = q * (len(ordered) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {"mean": sum(ordered) / len(ordered), "p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9)}


@torch.no_grad()
def compare_models(
    teacher: nn.Module,
    student: nn.Module,
    tokenizer: Any,
    texts: list[str],
    kinds: list[str] | None = None,
    *,
    max_seq_length: int = 256,
    batch_size: int = 32,
    device: str = "cpu",
    quantization_factor: int = 100,
    head_chunk: int = 64,
) -> dict[str, Any]:
    """Agreement statistics of the student against the teacher on ``texts`` (see the module docstring)."""
    teacher.eval()
    student.eval()
    kinds = kinds or ["all"] * len(texts)
    per_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seconds = {"teacher": 0.0, "student": 0.0}
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        chunk_kinds = kinds[start : start + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_seq_length, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        vectors = {}
        for name, model in (("teacher", teacher), ("student", student)):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            vectors[name] = encode_pooled(model, input_ids, attention_mask, head_chunk).float()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds[name] += time.perf_counter() - started
        t, s = vectors["teacher"], vectors["student"]
        cosine = torch.nn.functional.cosine_similarity(s, t, dim=-1)
        t_active = torch.round(t * quantization_factor) > 0
        s_active = torch.round(s * quantization_factor) > 0
        inter = (t_active & s_active).sum(dim=-1).float()
        union = (t_active | s_active).sum(dim=-1).float().clamp_min(1.0)
        recall = inter / t_active.sum(dim=-1).float().clamp_min(1.0)
        l1 = (s - t).abs().sum(dim=-1) / t.abs().sum(dim=-1).clamp_min(1e-6)
        t_dims = active_dims(t, quantization_factor).float()
        s_dims = active_dims(s, quantization_factor).float()
        tokens = attention_mask.sum(dim=-1).float()
        for i, kind in enumerate(chunk_kinds):
            stats = per_kind[kind]
            stats["cosine"].append(float(cosine[i]))
            stats["jaccard_active"].append(float(inter[i] / union[i]))
            stats["teacher_active_recall"].append(float(recall[i]))
            stats["l1_relative_error"].append(float(l1[i]))
            stats["teacher_active_dims"].append(float(t_dims[i]))
            stats["student_active_dims"].append(float(s_dims[i]))
            stats["tokens"].append(float(tokens[i]))
    result: dict[str, Any] = {
        "n": len(texts),
        "max_seq_length": max_seq_length,
        "device": device,
        "seconds_per_text": {name: value / max(len(texts), 1) for name, value in seconds.items()},
        "student_speedup": seconds["teacher"] / max(seconds["student"], 1e-9),
        "kinds": {},
    }
    for kind, stats in per_kind.items():
        result["kinds"][kind] = {"n": len(stats["cosine"]), **{key: _summary(values) for key, values in stats.items()}}
    if len(per_kind) > 1:
        merged: dict[str, list[float]] = defaultdict(list)
        for stats in per_kind.values():
            for key, values in stats.items():
                merged[key].extend(values)
        result["kinds"]["all"] = {"n": len(texts), **{key: _summary(values) for key, values in merged.items()}}
    return result


def load_texts(path: Path, field: str, n: int, seed: int, kinds: set[str] | None = None) -> tuple[list[str], list[str]]:
    """``n`` texts with their kinds, a seeded reservoir sample over the *whole* file (the corpus is ordered by
    source, so a sample of its head would be one source)."""
    import random

    rng = random.Random(seed)
    pool: list[tuple[str, str]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(row.get("kind", "all"))
            if kinds is not None and kind not in kinds:
                continue
            text = row.get(field)
            if not (isinstance(text, str) and text.strip()):
                continue
            seen += 1
            if len(pool) < n:
                pool.append((text, kind))
            else:
                j = rng.randrange(seen)
                if j < n:
                    pool[j] = (text, kind)
    rng.shuffle(pool)
    return [t for t, _ in pool], [k for _, k in pool]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--texts", required=True, type=Path, help="JSON lines with a text field (and optionally kind)")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--kinds", default=None, help="comma-separated kinds to keep (default: all)")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fp32-head", action="store_true", help="head-layer students: the extra layer's FFN and the head transform in fp32, as the KD trainer runs them (default: the stock path, as inference sees it)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForMaskedLM.from_pretrained(args.teacher, torch_dtype=dtype).to(args.device)
    student = AutoModelForMaskedLM.from_pretrained(args.student, torch_dtype=dtype).to(args.device)
    if args.fp32_head and not prepare_head_layer(student):
        print("[WARN] --fp32-head: the student is not a head-layer ALBERT; nothing to patch", file=sys.stderr)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    texts, text_kinds = load_texts(args.texts, args.text_field, args.n, args.seed, kinds)
    if not texts:
        print("[ERROR] no texts loaded.", file=sys.stderr)
        return 2
    result = compare_models(
        teacher, student, tokenizer, texts, text_kinds, max_seq_length=args.max_seq_length, batch_size=args.batch_size, device=args.device
    )
    result.update({"teacher": str(args.teacher), "student": str(args.student), "texts": str(args.texts), "dtype": args.dtype, "fp32_head": bool(args.fp32_head)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for kind, stats in result["kinds"].items():
        print(
            f"{kind}: n={stats['n']} cosine={stats['cosine']['mean']:.4f} jaccard={stats['jaccard_active']['mean']:.3f} "
            f"recall={stats['teacher_active_recall']['mean']:.3f} dims teacher/student={stats['teacher_active_dims']['mean']:.0f}/"
            f"{stats['student_active_dims']['mean']:.0f}"
        )
    print(f"seconds/text teacher={result['seconds_per_text']['teacher']:.4f} student={result['seconds_per_text']['student']:.4f} speedup={result['student_speedup']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
