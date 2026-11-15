#!/usr/bin/env python
"""Fit a static query encoder to the teacher's query vectors, text by text (no pool, no candidates).

For every text the frozen document model gives its query vector ``t`` (the pooled SPLADE activation) and the static
encoder its own ``s``; the loss is the squared distance ``sum_v w_v (s_v - t_v)^2`` (``--zero-weight`` = w on the
dimensions the teacher leaves at zero, 1 elsewhere). The ranking KL of ``distill.train_static`` supervises one
distribution over ~40 candidates per query; this supervises every dimension of every query, on any text, so it is
the stage in which parameters that depend on the context can see far more data than a mined pool holds: the gates
of a ``distill.static_v2`` encoder (``--train extra``), the weights of a table of either format (``--train values``)
or both. It knows nothing about ranking: follow it with ``distill.train_static`` on a pool.

Validation on ``--val-texts`` held-out texts: cosine to the teacher vector, relative error, active dimensions, and
precision / recall of the student's dimensions against the teacher's (weights quantized as the index stores them).

Usage:
  .venv/bin/python -m distill.train_static_vec --teacher models/X --init models/static2-X/phr2-k8-gate32 \\
      --texts datasets/kd/kd_v1.jsonl --kinds query --n 400000 --train extra --epochs 2 --extra-lr 1e-2 \\
      --out runs/static2-X-phr2-k8-gate32-vec
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

from distill.static_query import load_static_encoder, load_texts
from distill.train_kd import lr_at
from distill.train_static import OPTIMIZERS, TRAIN_GROUPS, build_optimizers, encoder_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, type=Path, help="the document model (its query vectors are the target)")
    parser.add_argument("--init", required=True, type=Path, help="static query encoder folder (either format)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--texts", required=True, type=Path)
    parser.add_argument("--field", default="text")
    parser.add_argument("--kinds", default=None, help="comma list of corpus kinds (kd_v1: query, passage)")
    parser.add_argument("--n", type=int, default=0, help="texts sampled (0 = all)")
    parser.add_argument("--val-texts", type=int, default=2000)
    parser.add_argument("--train", choices=TRAIN_GROUPS, default="all")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--zero-weight", type=float, default=1.0, help="loss weight of the dimensions the teacher does not activate")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--extra-lr", type=float, default=0.0, help="learning rate of the non-table parameters (0 = --lr)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--schedule", choices=("constant", "linear", "cosine"), default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--quantization-factor", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16" if torch.cuda.is_available() else "fp32", help="of the teacher")
    return parser.parse_args(argv)


class VectorDistiller:
    def __init__(self, teacher, tokenizer, model, args: argparse.Namespace) -> None:
        self.teacher, self.tokenizer, self.model, self.args = teacher, tokenizer, model, args
        self.device = args.device

    @torch.no_grad()
    def teacher_vectors(self, texts: list[str]) -> torch.Tensor:
        from distill.splade import encode_pooled

        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.model.max_length, return_tensors="pt")
        vectors = encode_pooled(self.teacher, enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)).float()
        return vectors[:, : self.model.output_dim]

    def student_vectors(self, texts: list[str]) -> torch.Tensor:
        return self.model(**self.model.tokenize(self.tokenizer, texts, self.device))

    def loss(self, texts: list[str]) -> tuple[torch.Tensor, dict[str, float]]:
        target = self.teacher_vectors(texts)
        student = self.student_vectors(texts)
        error = (student - target) ** 2
        if self.args.zero_weight != 1.0:
            error = torch.where(target > 0, error, error * self.args.zero_weight)
        loss = error.sum(dim=-1).mean()
        return loss, {"dims_per_query": float((student.detach() > 0).sum(dim=-1).float().mean())}

    @torch.no_grad()
    def evaluate(self, texts: list[str], batch_size: int = 256) -> dict[str, Any]:
        qf = float(self.args.quantization_factor)
        sums = {"cosine": 0.0, "relative_error": 0.0, "dims_student": 0.0, "dims_teacher": 0.0, "precision": 0.0, "recall": 0.0}
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            target = torch.round(self.teacher_vectors(chunk) * qf) / qf
            student = torch.round(self.student_vectors(chunk) * qf) / qf
            s_on, t_on = student > 0, target > 0
            both = (s_on & t_on).sum(dim=-1).float()
            sums["cosine"] += float(torch.nn.functional.cosine_similarity(student, target, dim=-1).sum())
            sums["relative_error"] += float(((student - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-6)).sum())
            sums["dims_student"] += float(s_on.sum())
            sums["dims_teacher"] += float(t_on.sum())
            sums["precision"] += float((both / s_on.sum(dim=-1).clamp_min(1)).sum())
            sums["recall"] += float((both / t_on.sum(dim=-1).clamp_min(1)).sum())
        n = max(len(texts), 1)
        return {"texts": len(texts), **{key: round(value / n, 4) for key, value in sums.items()}}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    texts = load_texts(args.texts, args.n, kinds, args.field, args.seed)
    random.Random(args.seed).shuffle(texts)
    val_texts, train_texts = texts[: args.val_texts], texts[args.val_texts :]
    if not train_texts:
        print("[ERROR] no training texts left after the validation split", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(args.init)
    model = load_static_encoder(args.init, args.device)
    model.meta = dict(model.meta, init_dir=str(args.init), vector_teacher=str(args.teacher))
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    teacher = AutoModelForMaskedLM.from_pretrained(args.teacher, torch_dtype=dtype).to(args.device).eval()
    distiller = VectorDistiller(teacher, tokenizer, model, args)
    try:
        optimizers = build_optimizers(model, argparse.Namespace(log_scale=None, log_gain=None), args)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    trained = [p for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
    args.batch_size = min(args.batch_size, len(train_texts))
    steps_per_epoch = max(len(train_texts) // args.batch_size, 1)
    total_steps = args.max_steps or max(int(steps_per_epoch * args.epochs), 1)
    log_file = (args.out / "log.jsonl").open("w", encoding="utf-8")

    def log(record: dict[str, Any]) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(json.dumps(record), flush=True)

    summary: dict[str, Any] = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "train_texts": len(train_texts),
                               "val_texts": len(val_texts), "total_steps": total_steps, "table": encoder_stats(model)}
    if val_texts:
        summary["val_init"] = distiller.evaluate(val_texts)
        log({"step": 0, "kind": "val", **summary["val_init"]})
    started = time.time()
    step, epoch = 0, 0
    while step < total_steps:
        order = list(range(len(train_texts)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            if step >= total_steps:
                break
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total_steps, args.warmup_steps, args.schedule, group["base_lr"])
            loss, parts = distiller.loss([train_texts[i] for i in order[start : start + args.batch_size]])
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trained, args.max_grad_norm)
            for optimizer in optimizers:
                optimizer.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                log({"step": step, "kind": "train", "loss": round(float(loss.detach()), 5), "lr": optimizers[0].param_groups[0]["lr"],
                     **{k: round(v, 2) for k, v in parts.items()}, "seconds": round(time.time() - started, 1)})
            if val_texts and args.eval_every and step % args.eval_every == 0:
                log({"step": step, "kind": "val", **distiller.evaluate(val_texts)})
        epoch += 1

    final_dir = args.out / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    model.meta["train_state"] = {"step": step, "epochs": round(step / steps_per_epoch, 2), "stage": "vector"}
    model.save(final_dir, tokenizer)
    if val_texts:
        summary["val_final"] = distiller.evaluate(val_texts)
        log({"step": step, "kind": "val_final", **summary["val_final"]})
    summary.update({"steps": step, "table_final": encoder_stats(model), "seconds": round(time.time() - started, 1)})
    (args.out / "vec_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
