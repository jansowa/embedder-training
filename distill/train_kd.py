#!/usr/bin/env python
"""Representation distillation of a SPLADE teacher into a student on unlabeled text.

Per batch of texts (queries and passages alike, one tokenizer for both models):
  teacher  -> hidden states (no gradient)             student -> hidden states
  token loss   KL on the teacher's top-k tempered logits per token, or MSE of the per-token SPLADE
               activations (``--token-loss kl|mse``), computed on position chunks under checkpointing;
  pooled loss  MSE or KL between the pooled SPLADE vectors (``--pooled-loss mse|kl``), with the
               sparse-gradient decoder, so no [B, L, V] tensor is kept;
  FLOPS        optional regulariser on the student's pooled vectors (``--flops-weight``).
Any combination of weights; ``--token-loss none --pooled-loss mse`` is the pooled stage.

Checkpoints are complete SparseEncoder folders (weights + the tokenizer / modules files copied from
the student directory), so every one can be measured by ``scripts/pirb_regime_eval.sh``. A fidelity
sample (``--fidelity-texts``) is compared at every checkpoint and logged, which gives a learning curve
without PIRB.

Usage:
  .venv/bin/python -m distill.train_kd --teacher models/soup-v9b --student models/student-b4 \
      --corpus datasets/kd/kd_v1.jsonl --out runs/kd-b4-token --token-loss kl --pooled-loss none \
      --max-seq-length 256 --batch-size 16 --lr 5e-5 --epochs 1 --save-every 2000
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from distill.data import TextCorpus
from distill.fidelity import compare_models, load_texts
from distill.splade import describe_losses, encode_hidden, flops_loss, pooled_distill_loss, pooled_vector, prepare_head_layer, token_distill_loss
from distill.student import copy_aux_files

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def lr_at(step: int, total_steps: int, warmup: int, schedule: str, lr: float) -> float:
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    if schedule == "linear":
        remaining = max(total_steps - warmup, 1)
        return lr * max(0.0, (total_steps - step) / remaining)
    if schedule == "cosine":
        remaining = max(total_steps - warmup, 1)
        progress = min(max(step - warmup, 0) / remaining, 1.0)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {schedule!r}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--student", required=True, type=Path, help="student folder from distill.student (or a checkpoint)")
    parser.add_argument("--corpus", required=True, type=Path, nargs="+", help="JSON lines with a text field")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--kinds", default=None, help="comma-separated kinds to train on (default: all)")
    parser.add_argument("--max-items", type=int, default=0, help="seeded random subset of the corpus (0 = all)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimiser steps (0 = by epochs)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--schedule", choices=("constant", "linear", "cosine"), default="constant")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--teacher-precision", choices=tuple(DTYPES), default=None, help="default: same as --precision")
    parser.add_argument("--token-loss", choices=("none", "kl", "mse"), default="kl")
    parser.add_argument("--token-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--pooled-loss", choices=("none", "mse", "kl"), default="none")
    parser.add_argument("--pooled-weight", type=float, default=1.0)
    parser.add_argument("--flops-weight", type=float, default=0.0)
    parser.add_argument("--head-chunk", type=int, default=64, help="position chunk of the decoder (token loss and sparse max)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="checkpoint the student's encoder layers too")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=0, help="checkpoint every N optimiser steps (0 = only final)")
    parser.add_argument("--keep-steps", default="", help="comma-separated steps whose checkpoints are kept; others are rolling")
    parser.add_argument("--keep-last", type=int, default=1, help="rolling checkpoints to keep besides --keep-steps")
    parser.add_argument("--fidelity-texts", type=Path, default=None, help="JSON lines sample compared at every checkpoint")
    parser.add_argument("--fidelity-n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--wandb", default=None, help="W&B run name (project from WANDB_PROJECT); off when unset")
    return parser.parse_args(argv)


class Checkpoints:
    def __init__(self, out: Path, student_dir: Path, keep_steps: set[int], keep_last: int) -> None:
        self.out = out
        self.student_dir = student_dir
        self.keep_steps = keep_steps
        self.keep_last = keep_last
        self.rolling: list[Path] = []

    def save(self, model: nn.Module, name: str, step: int, extra: dict[str, Any]) -> Path:
        path = self.out / name
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        copy_aux_files(self.student_dir, path)
        (path / "kd_state.json").write_text(json.dumps({"step": step, **extra}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if name.startswith("step-") and step not in self.keep_steps:
            self.rolling.append(path)
            while len(self.rolling) > self.keep_last:
                shutil.rmtree(self.rolling.pop(0), ignore_errors=True)
        return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device
    dtype = DTYPES[args.precision]
    teacher_dtype = DTYPES[args.teacher_precision or args.precision]
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.bfloat16  # fp16 autocast is not supported on CPU
        teacher_dtype = torch.float32 if teacher_dtype == torch.float16 else teacher_dtype
    autocast_enabled = dtype != torch.float32
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    teacher = AutoModelForMaskedLM.from_pretrained(args.teacher, torch_dtype=teacher_dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForMaskedLM.from_pretrained(args.student, torch_dtype=torch.float32).to(device)
    student.train()
    if prepare_head_layer(student):
        print("[INFO] head-layer student: the extra layer's feed-forward and the head transform run in fp32, its dropout is off", flush=True)
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    if teacher.config.vocab_size != student.config.vocab_size:
        print("[ERROR] teacher and student must share the vocabulary (same tokenizer).", file=sys.stderr)
        return 2

    kinds = set(args.kinds.split(",")) if args.kinds else None
    corpus = TextCorpus(args.corpus, args.text_field, kinds, args.max_items, seed=args.seed)
    if len(corpus) == 0:
        print("[ERROR] the corpus is empty.", file=sys.stderr)
        return 2
    batches_per_epoch = math.ceil(len(corpus) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    fidelity_texts: list[str] = []
    fidelity_kinds: list[str] = []
    if args.fidelity_texts:
        fidelity_texts, fidelity_kinds = load_texts(args.fidelity_texts, args.text_field, args.fidelity_n, args.seed)

    loss_config = {
        "token_loss": None if args.token_loss == "none" else args.token_loss,
        "token_weight": args.token_weight,
        "pooled_loss": None if args.pooled_loss == "none" else args.pooled_loss,
        "pooled_weight": args.pooled_weight,
        "flops_weight": args.flops_weight,
    }
    if not loss_config["token_loss"] and not loss_config["pooled_loss"]:
        print("[ERROR] choose at least one of --token-loss / --pooled-loss.", file=sys.stderr)
        return 2
    run_config = {**vars(args), "corpus": [str(p) for p in args.corpus], "teacher": str(args.teacher), "student": str(args.student), "out": str(args.out)}
    run_config.update({"fidelity_texts": str(args.fidelity_texts) if args.fidelity_texts else None, "corpus_stats": vars(corpus.stats)})
    run_config.update({"total_steps": total_steps, "steps_per_epoch": steps_per_epoch, "losses": describe_losses(loss_config)})
    (args.out / "kd_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"[INFO] corpus {len(corpus)} texts, {steps_per_epoch} steps/epoch, {total_steps} steps, losses: {describe_losses(loss_config)}", flush=True)

    decay, no_decay = [], []
    for name, p in student.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name else decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(name=args.wandb, config=run_config)

    checkpoints = Checkpoints(args.out, args.student, {int(s) for s in args.keep_steps.split(",") if s.strip()}, args.keep_last)
    log_path = args.out / "log.jsonl"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(record: dict[str, Any]) -> None:
        log_handle.write(json.dumps(record, sort_keys=True) + "\n")
        log_handle.flush()
        if wandb_run is not None:
            wandb_run.log(record, step=record.get("step"))

    def fidelity(step: int) -> dict[str, Any] | None:
        if not fidelity_texts:
            return None
        student.eval()
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=autocast_enabled):
            result = compare_models(teacher, student, tokenizer, fidelity_texts, fidelity_kinds, max_seq_length=args.max_seq_length, device=device)
        student.train()
        summary: dict[str, Any] = {"step": step}
        summary.update({"fidelity/" + kind + "/cosine": stats["cosine"]["mean"] for kind, stats in result["kinds"].items()})
        summary.update({"fidelity/" + kind + "/jaccard": stats["jaccard_active"]["mean"] for kind, stats in result["kinds"].items()})
        summary.update({"fidelity/" + kind + "/l1": stats["l1_relative_error"]["mean"] for kind, stats in result["kinds"].items()})
        summary.update({"fidelity/" + kind + "/student_dims": stats["student_active_dims"]["mean"] for kind, stats in result["kinds"].items()})
        summary["fidelity/teacher_dims"] = next(iter(result["kinds"].values()))["teacher_active_dims"]["mean"]
        log(summary)
        print(f"[FIDELITY] step {step}: " + ", ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in summary.items() if k != "step"), flush=True)
        return result

    step = 0
    micro = 0
    examples = 0
    tokens = 0
    started = time.time()
    window: dict[str, float] = {"loss": 0.0, "token": 0.0, "pooled": 0.0, "flops": 0.0, "n": 0.0}
    fidelity(0)
    done = False
    epoch = 0
    while not done:
        for texts in corpus.iter_batches(args.batch_size, args.seed, epoch):
            enc = tokenizer(texts, truncation=True, max_length=args.max_seq_length, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=autocast_enabled):
                with torch.no_grad():
                    teacher_hidden = encode_hidden(teacher, input_ids, attention_mask)
                student_hidden = encode_hidden(student, input_ids, attention_mask)
                loss = student_hidden.new_zeros((), dtype=torch.float32)
                parts: dict[str, float] = {}
                if loss_config["token_loss"]:
                    token = token_distill_loss(
                        student, student_hidden, teacher, teacher_hidden, attention_mask,
                        kind=loss_config["token_loss"], temperature=args.temperature, top_k=args.top_k, chunk=args.head_chunk,
                    )
                    loss = loss + args.token_weight * token
                    parts["token"] = float(token)
                if loss_config["pooled_loss"] or args.flops_weight:
                    student_vec = pooled_vector(student, student_hidden, attention_mask, args.head_chunk)
                    if loss_config["pooled_loss"]:
                        with torch.no_grad():
                            teacher_vec = pooled_vector(teacher, teacher_hidden, attention_mask, args.head_chunk)
                        pooled = pooled_distill_loss(student_vec, teacher_vec, loss_config["pooled_loss"])
                        loss = loss + args.pooled_weight * pooled
                        parts["pooled"] = float(pooled)
                    if args.flops_weight:
                        flops = flops_loss(student_vec)
                        loss = loss + args.flops_weight * flops
                        parts["flops"] = float(flops)
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            examples += len(texts)
            tokens += int(attention_mask.sum())
            window["loss"] += float(loss)
            for key, value in parts.items():
                window[key] += value
            window["n"] += 1
            if micro % args.grad_accum != 0:
                continue
            lr = lr_at(step, total_steps, args.warmup_steps, args.schedule, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                n = max(window["n"], 1.0)
                elapsed = time.time() - started
                record = {
                    "step": step, "epoch": epoch, "lr": lr, "grad_norm": grad_norm, "loss": window["loss"] / n,
                    "examples": examples, "tokens": tokens, "elapsed_s": round(elapsed, 1),
                    "examples_per_s": round(examples / max(elapsed, 1e-9), 2), "tokens_per_s": round(tokens / max(elapsed, 1e-9), 1),
                }
                for key in ("token", "pooled", "flops"):
                    if window[key]:
                        record["loss_" + key] = window[key] / n
                log(record)
                print(f"[STEP {step}/{total_steps}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items() if k != "step"), flush=True)
                window = {"loss": 0.0, "token": 0.0, "pooled": 0.0, "flops": 0.0, "n": 0.0}
            if args.save_every and step % args.save_every == 0 and step < total_steps:
                checkpoints.save(student, f"step-{step}", step, {"examples": examples, "tokens": tokens})
                fidelity(step)
            if step >= total_steps:
                done = True
                break
        epoch += 1
        if not done and args.max_steps == 0 and epoch >= math.ceil(args.epochs):
            done = True
    final = checkpoints.save(student, "final", step, {"examples": examples, "tokens": tokens})
    result = fidelity(step)
    summary = {
        "steps": step, "examples": examples, "tokens": tokens, "seconds": round(time.time() - started, 1), "final": str(final),
        "fidelity": {kind: {"cosine": s["cosine"]["mean"], "jaccard_active": s["jaccard_active"]["mean"]} for kind, s in result["kinds"].items()} if result else None,
    }
    (args.out / "kd_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_handle.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[DONE] {json.dumps(summary, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
