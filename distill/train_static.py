#!/usr/bin/env python
"""Train a static sparse query encoder (``distill.static_query``) against a frozen SPLADE document side.

Inputs: a candidate pool from ``distill.static_pool`` (rows + quantized document vectors of the document model, and
optionally that model's own query vectors) and an initial table from ``distill.static_query build``. Only the table's
weights are trained; the slots (which vocabulary dimension each token row may activate) stay as initialised, so
``prune`` / ``top_k`` decide the cost and training decides the weights. No transformer runs, a step is a lookup,
one scatter-max and a ``[B, V] x [V, docs]`` product.

Loss per row: ``KL(softmax(teacher / T) || softmax(student / T)) * T^2`` over the row's candidates, where
  --target rows    teacher = the row's stored scores (``pos_scores`` + ``neg_scores``, e.g. the ensemble teacher of the
                   ``-tens400`` files; same dot-product scale as SPLADE, ``--teacher-scale`` otherwise);
  --target splade  teacher = ``q_splade . d`` of the document model on its own queries (pool with --teacher-queries):
                   the static encoder learns to rank like the symmetric model it replaces. With ``--in-batch`` the
                   candidates are every passage of the batch, so the student also learns not to match other rows'
                   passages (only this target has teacher scores for them).
  --target mix     ``--mix-weight`` x splade + (1 - w) x rows, both on the row's own candidates (+ in-batch for splade).
Regularisers on the student's query vectors: ``--query-l1`` (mean L1 mass) and ``--query-flops`` (SPLADE FLOPS).
``--learn-scale``: the student scores enter the KL multiplied by ``exp(s)``, one learned scalar initialised to the
ratio of the teacher's and the student's score spread on the first batch. A global scale does not change any ranking,
so it is not part of the table; it only stops the KL from spending the table's weights on matching the teacher's
scale (an idf table scores ~8x higher than the SPLADE query it imitates). ``--scale-lr`` is its learning rate.
Pools mined by ``distill.static_mine`` (teacher top + student top passages of an unlabelled corpus, teacher scores as
row scores, ``n_pos = 1`` = the teacher's top passage) train with ``--target rows``.

Options that change how the weights are fitted, never the format of the table (path A of docs/distillation.md):
  --optimizer      ``adam`` (default) treats the zero gradient of an absent row as a gradient, so a rare row keeps
                   moving on momentum after its one occurrence; ``sparse_adam`` updates only the rows of the batch,
                   ``adagrad`` gives every row a step that shrinks with its own gradient history;
  --value-floor    weights are clamped to this small positive value after every step, so no slot dies behind the
                   relu; slots that stay below half a quantization step are written as zeros (the quantized regime
                   would drop them anyway);
  --dim-gain       one learned multiplier per vocabulary dimension, shared by every row that writes the dimension
                   (rows too rare to be trained on their own still move); folded into the weights when saving;
  --keep-best      the saved table is the best validation point, not the last step.
A folder of ``distill.static_v2`` (a larger encoder, path B) trains with the same loop: ``--train`` picks the table
weights, the encoder's other parameters (gates, threshold, length scales; ``--extra-lr``) or both.

Validation (``--val-rows`` held out by seed, every ``--eval-every`` steps and at the end), on each row's own candidates,
with the query weights quantized as PIRB's quantized regime does (round(w * qf) / qf): Kendall tau and top-1 agreement
with the target, the share of rows whose top candidate is a positive, MRR of the first positive, active dims per query
and ``postings_fraction`` = sum over the query's dims of df(v) / N_docs on the pool (multiply by a corpus size to get
postings per query). The symmetric SPLADE (pool queries) is measured the same way once, as the reference.

Usage:
  .venv/bin/python -m distill.train_static --pool cache/static_pools/n5kj --init models/static/credit-k32 \\
      --out runs/static-credit-k32 --target rows --epochs 3 --batch-size 64 --lr 5e-3 --device cuda
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
import torch.nn.functional as F

from distill.rank_agreement import kendall_tau
from distill.static_pool import Pool, load_pool
from distill.static_query import load_static_encoder, table_stats
from distill.train_kd import lr_at

TARGETS = ("rows", "splade", "mix")
OPTIMIZERS = ("adam", "sparse_adam", "adagrad")
TRAIN_GROUPS = ("all", "values", "extra")
BEST_METRICS = ("mrr", "top1_agreement", "kendall_tau")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--init", required=True, type=Path, help="static query folder (distill.static_query build / a checkpoint)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target", choices=TARGETS, default="rows")
    parser.add_argument("--mix-weight", type=float, default=0.5, help="--target mix: weight of the splade term")
    parser.add_argument("--in-batch", action="store_true", help="splade target over every passage of the batch")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--teacher-scale", type=float, default=1.0, help="multiply the stored row scores")
    parser.add_argument("--query-l1", type=float, default=0.0)
    parser.add_argument("--query-flops", type=float, default=0.0)
    parser.add_argument("--learn-scale", action="store_true", help="learn a global multiplier of the student scores inside the KL")
    parser.add_argument("--scale-lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--schedule", choices=("constant", "linear", "cosine"), default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="clip (0 = off; not with --optimizer sparse_adam)")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam", help="optimizer of the table weights")
    parser.add_argument("--train", choices=TRAIN_GROUPS, default="all", help="table weights, the other parameters of a static_v2 encoder, or both")
    parser.add_argument("--extra-lr", type=float, default=0.0, help="learning rate of the non-table parameters (0 = --lr)")
    parser.add_argument("--value-floor", type=float, default=0.0, help="clamp the weights to at least this after every step (0 = off)")
    parser.add_argument("--dim-gain", action="store_true", help="learn one multiplier per vocabulary dimension, folded into the weights when saving")
    parser.add_argument("--gain-lr", type=float, default=1e-2)
    parser.add_argument("--keep-best", action="store_true", help="save the best validation point instead of the last step")
    parser.add_argument("--best-metric", choices=BEST_METRICS, default="mrr")
    parser.add_argument("--val-rows", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args(argv)


class StaticTrainer:
    def __init__(self, pool: Pool, model, tokenizer, args: argparse.Namespace) -> None:
        self.pool = pool
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = args.device
        if model.output_dim != pool.docs.dim:
            raise ValueError(f"the static table writes {model.output_dim} dims but the pool's documents have {pool.docs.dim}.")
        if args.target in ("splade", "mix") and pool.queries is None:
            raise ValueError(f"--target {args.target} needs a pool built with --teacher-queries.")
        if args.target in ("rows", "mix") and not any(r["scores"] is not None for r in pool.rows):
            raise ValueError(f"--target {args.target} needs rows with pos_scores / neg_scores.")
        self.df_fraction = (pool.docs.document_frequency().float() / max(len(pool.docs), 1)).to(self.device)
        self.qf = float(pool.meta["quantization_factor"])
        self.log_scale = torch.nn.Parameter(torch.zeros((), device=self.device)) if args.learn_scale else None
        self.log_gain = torch.nn.Parameter(torch.zeros(model.output_dim, device=self.device)) if getattr(args, "dim_gain", False) else None
        if self.log_gain is not None and not getattr(model, "gain_foldable", True):
            raise ValueError("--dim-gain cannot be folded into this encoder (a threshold or a top-m cut follows the pooling).")

    @torch.no_grad()
    def init_scale(self, batch: dict[str, Any]) -> float:
        """``log_scale`` = log(std of the teacher scores / std of the student scores) over the batch's own candidates."""
        if self.log_scale is None:
            return 1.0
        q = self.student_queries(batch["rows"])
        student = q @ batch["docs"].T
        if self.args.target in ("rows", "mix"):
            mask = batch["own"] & ~torch.isnan(batch["row_scores"])
            teacher = batch["row_scores"]
        else:
            mask = batch["own"]
            teacher = batch["teacher_q"] @ batch["docs"].T
        s_std, t_std = float(student[mask].std()), float(teacher[mask].std())
        if s_std > 0 and t_std > 0:
            self.log_scale.fill_(float(torch.log(torch.tensor(t_std / s_std))))
        return float(self.log_scale.exp())

    def usable(self, row_id: int) -> bool:
        if self.args.target == "splade":
            return True
        return self.pool.rows[row_id]["scores"] is not None

    def batch_tensors(self, row_ids: list[int]) -> dict[str, Any]:
        rows = [self.pool.rows[i] for i in row_ids]
        doc_ids = sorted({d for r in rows for d in r["docs"]})
        col = {d: j for j, d in enumerate(doc_ids)}
        own = torch.zeros(len(rows), len(doc_ids), dtype=torch.bool)
        row_scores = torch.full((len(rows), len(doc_ids)), float("nan"))
        for b, r in enumerate(rows):
            for j, d in enumerate(r["docs"]):
                own[b, col[d]] = True
                if r["scores"] is not None:
                    row_scores[b, col[d]] = r["scores"][j] * self.args.teacher_scale
        docs = self.pool.docs.dense_rows(doc_ids, self.device)
        out = {"rows": rows, "docs": docs, "own": own.to(self.device), "row_scores": row_scores.to(self.device), "doc_ids": doc_ids, "col": col}
        if self.pool.queries is not None:
            out["teacher_q"] = self.pool.queries.dense_rows(row_ids, self.device)
        return out

    def student_queries(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        enc = self.model.tokenize(self.tokenizer, [r["query"] for r in rows], self.device)
        q = self.model(**enc)
        return q if self.log_gain is None else q * self.log_gain.exp()

    @torch.no_grad()
    def exported_values(self) -> torch.Tensor:
        """The weights as they are saved: the per-dimension gain folded in, floor-level slots written as zeros."""
        values = self.model.values.detach().clone()
        if self.log_gain is not None:
            values = values * self.log_gain.exp()[self.model.indices]
        floor = float(getattr(self.args, "value_floor", 0.0) or 0.0)
        if floor > 0 and not getattr(self.model, "signed", False):
            values = torch.where(values < 0.5 / self.qf, torch.zeros_like(values), values)
        return values

    @torch.no_grad()
    def finalize(self) -> None:
        """Make the model in memory the table that is saved (after this the gain is spent)."""
        self.model.values.copy_(self.exported_values())
        if self.log_gain is not None:
            self.log_gain.zero_()

    @staticmethod
    def kl(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
        neg_inf = torch.finfo(student.dtype).min
        s = torch.where(mask, student / temperature, torch.full_like(student, neg_inf))
        t = torch.where(mask, teacher / temperature, torch.full_like(teacher, neg_inf))
        p_t = F.softmax(t, dim=-1)
        loss = (p_t * (F.log_softmax(t, dim=-1) - F.log_softmax(s, dim=-1))).masked_fill(~mask, 0.0).sum(dim=-1)
        return loss.mean() * temperature**2

    def loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        args = self.args
        q = self.student_queries(batch["rows"])
        scores = q @ batch["docs"].T
        if self.log_scale is not None:
            scores = scores * self.log_scale.exp()
        parts: dict[str, torch.Tensor] = {}
        if args.target in ("rows", "mix"):
            mask = batch["own"] & ~torch.isnan(batch["row_scores"])
            parts["rows"] = self.kl(scores, torch.nan_to_num(batch["row_scores"]), mask, args.temperature)
        if args.target in ("splade", "mix"):
            teacher = batch["teacher_q"] @ batch["docs"].T
            mask = torch.ones_like(batch["own"]) if args.in_batch else batch["own"]
            parts["splade"] = self.kl(scores, teacher, mask, args.temperature)
        if args.target == "mix":
            total = args.mix_weight * parts["splade"] + (1.0 - args.mix_weight) * parts["rows"]
        else:
            total = parts[args.target]
        if args.query_l1:
            parts["l1"] = q.sum(dim=-1).mean()
            total = total + args.query_l1 * parts["l1"]
        if args.query_flops:
            parts["flops"] = (q.mean(dim=0) ** 2).sum()
            total = total + args.query_flops * parts["flops"]
        logs = {k: float(v.detach()) for k, v in parts.items()}
        logs["dims_per_query"] = float((q.detach() > 0).sum(dim=-1).float().mean())
        return total, logs

    @torch.no_grad()
    def evaluate(self, row_ids: list[int], use_teacher_queries: bool = False, batch_size: int = 128, query_fn=None) -> dict[str, Any]:
        """``query_fn(batch) -> [B, V]`` replaces the student's query vectors (the oracles of distill.static_diag)."""
        taus, top1, pos_first, mrr, dims, postings = [], [], [], [], [], []
        for start in range(0, len(row_ids), batch_size):
            ids = row_ids[start : start + batch_size]
            batch = self.batch_tensors(ids)
            if query_fn is not None:
                q = query_fn(batch)
            else:
                q = batch["teacher_q"] if use_teacher_queries else self.student_queries(batch["rows"])
            q = torch.round(q * self.qf) / self.qf
            scores = (q @ batch["docs"].T).cpu()
            # agreement is read against the stored row scores when the target uses them, else against the symmetric SPLADE
            if self.args.target in ("rows", "mix"):
                target = batch["row_scores"].cpu()
            else:
                target = (batch["teacher_q"] @ batch["docs"].T).cpu()
            active = q > 0
            dims.extend(active.sum(dim=-1).tolist())
            postings.extend((active.float() * self.df_fraction).sum(dim=-1).tolist())
            for b, r in enumerate(batch["rows"]):
                cols = [batch["col"][d] for d in r["docs"]]
                s = [float(scores[b, c]) for c in cols]
                n_pos = r["n_pos"]
                order = sorted(range(len(s)), key=lambda i: -s[i])
                pos_first.append(float(order[0] < n_pos))
                rank = next(k for k, i in enumerate(order) if i < n_pos)
                mrr.append(1.0 / (rank + 1))
                t = [float(target[b, c]) for c in cols]
                if not any(x != x for x in t):  # rows without stored scores give NaN targets
                    taus.append(kendall_tau(s, t))
                    top1.append(float(order[0] == max(range(len(t)), key=t.__getitem__)))
        n = max(len(pos_first), 1)
        return {
            "rows": len(pos_first),
            "kendall_tau": round(sum(taus) / max(len(taus), 1), 4),
            "top1_agreement": round(sum(top1) / max(len(top1), 1), 4),
            "positive_first": round(sum(pos_first) / n, 4),
            "mrr": round(sum(mrr) / n, 4),
            "dims_per_query": round(sum(dims) / max(len(dims), 1), 2),
            "postings_fraction": round(sum(postings) / max(len(postings), 1), 4),
        }


def save_checkpoint(model, tokenizer, path: Path, state: dict[str, Any], trainer: "StaticTrainer | None" = None) -> None:
    """Save the encoder; with ``trainer`` the table is written as ``trainer.exported_values()`` (gain folded, floor
    slots zeroed) while the weights in memory stay as the optimizer left them."""
    model.meta["train_state"] = state
    if trainer is None:
        model.save(path, tokenizer)
        return
    kept = model.values.detach().clone()
    with torch.no_grad():
        model.values.copy_(trainer.exported_values())
    try:
        model.save(path, tokenizer)
    finally:
        with torch.no_grad():
            model.values.copy_(kept)


def build_optimizers(model, trainer: StaticTrainer, args: argparse.Namespace) -> list[torch.optim.Optimizer]:
    """One optimizer for the table weights (``--optimizer``) and one Adam for every dense parameter next to them
    (the other parameters of a static_v2 encoder, the score scale, the per-dimension gain). Each group carries its
    ``base_lr`` for the schedule."""
    params = model.trainable_parameters()
    optimizers: list[torch.optim.Optimizer] = []
    if args.train in ("all", "values"):
        group = [{"params": params["values"], "lr": args.lr, "base_lr": args.lr}]
        if args.optimizer == "sparse_adam":
            if args.max_grad_norm > 0:
                raise ValueError("--max-grad-norm does not work on the sparse gradients of --optimizer sparse_adam.")
            model.sparse_grad = True
            optimizers.append(torch.optim.SparseAdam(group))
        elif args.optimizer == "adagrad":
            optimizers.append(torch.optim.Adagrad(group))
        else:
            optimizers.append(torch.optim.Adam(group))
    else:
        for p in params["values"]:
            p.requires_grad_(False)
    dense: list[dict[str, Any]] = []
    extra_lr = args.extra_lr or args.lr
    if args.train in ("all", "extra") and params["extra"]:
        dense.append({"params": params["extra"], "lr": extra_lr, "base_lr": extra_lr})
    else:
        for p in params["extra"]:
            p.requires_grad_(False)
    if not optimizers and not dense:
        raise ValueError(f"--train {args.train}: this encoder has no such parameters.")
    if trainer.log_scale is not None:
        dense.append({"params": [trainer.log_scale], "lr": args.scale_lr, "base_lr": args.scale_lr})
    if trainer.log_gain is not None:
        dense.append({"params": [trainer.log_gain], "lr": args.gain_lr, "base_lr": args.gain_lr})
    if dense:
        optimizers.append(torch.optim.Adam(dense))
    return optimizers


def snapshot(model, trainer: StaticTrainer) -> dict[str, Any]:
    state = {"model": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    for name in ("log_scale", "log_gain"):
        value = getattr(trainer, name)
        if value is not None:
            state[name] = value.detach().clone()
    return state


@torch.no_grad()
def restore(model, trainer: StaticTrainer, state: dict[str, Any]) -> None:
    model.load_state_dict(state["model"])
    for name in ("log_scale", "log_gain"):
        if name in state:
            getattr(trainer, name).copy_(state[name])


def encoder_stats(model) -> dict[str, Any]:
    return model.stats() if hasattr(model, "stats") else table_stats(model)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    pool = load_pool(args.pool)
    tokenizer = AutoTokenizer.from_pretrained(args.init)
    model = load_static_encoder(args.init, args.device)
    model.meta = dict(model.meta, init_dir=str(args.init), pool=str(args.pool))
    trainer = StaticTrainer(pool, model, tokenizer, args)

    ids = [i for i in range(len(pool.rows)) if trainer.usable(i)]
    random.Random(args.seed).shuffle(ids)
    val_ids, train_ids = ids[: args.val_rows], ids[args.val_rows :]
    if not train_ids:
        print("[ERROR] no training rows left after the validation split", file=sys.stderr)
        return 2
    args.batch_size = min(args.batch_size, len(train_ids))  # otherwise an epoch has no full batch and the loop never ends
    steps_per_epoch = max(len(train_ids) // args.batch_size, 1)
    total_steps = args.max_steps or max(int(steps_per_epoch * args.epochs), 1)
    if trainer.log_scale is not None:
        summary_scale_init = trainer.init_scale(trainer.batch_tensors(train_ids[: args.batch_size]))
    try:
        optimizers = build_optimizers(model, trainer, args)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    trained = [p for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
    floor = args.value_floor if args.value_floor > 0 and not getattr(model, "signed", False) and model.values.requires_grad else 0.0
    log_path = args.out / "log.jsonl"
    log_file = log_path.open("w", encoding="utf-8")

    def log(record: dict[str, Any]) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(json.dumps(record), flush=True)

    summary: dict[str, Any] = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "pool": pool.meta,
                               "train_rows": len(train_ids), "val_rows": len(val_ids), "total_steps": total_steps, "table": encoder_stats(model)}
    if trainer.log_scale is not None:
        summary["scale_init"] = round(summary_scale_init, 5)
        log({"step": 0, "kind": "scale_init", "scale": summary["scale_init"]})
    if val_ids:
        if pool.queries is not None:
            summary["val_splade_reference"] = trainer.evaluate(val_ids, use_teacher_queries=True)
            log({"step": 0, "kind": "val_splade_reference", **summary["val_splade_reference"]})
        summary["val_init"] = trainer.evaluate(val_ids)
        log({"step": 0, "kind": "val", **summary["val_init"]})

    started = time.time()
    step, epoch = 0, 0
    best: tuple[float, int] | None = None
    best_state: dict[str, Any] | None = None
    while step < total_steps:
        order = list(train_ids)
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            if step >= total_steps:
                break
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total_steps, args.warmup_steps, args.schedule, group["base_lr"])
            loss, parts = trainer.loss(trainer.batch_tensors(order[start : start + args.batch_size]))
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trained, args.max_grad_norm)
            for optimizer in optimizers:
                optimizer.step()
            if floor > 0:
                with torch.no_grad():
                    model.values.clamp_(min=floor)
            step += 1
            if step % args.log_every == 0 or step == 1:
                scale = {"scale": round(float(trainer.log_scale.detach().exp()), 5)} if trainer.log_scale is not None else {}
                log({"step": step, "kind": "train", "loss": round(float(loss.detach()), 5), "lr": optimizers[0].param_groups[0]["lr"],
                     **{k: round(v, 4) for k, v in parts.items()}, **scale, "seconds": round(time.time() - started, 1)})
            if val_ids and args.eval_every and step % args.eval_every == 0:
                val = trainer.evaluate(val_ids)
                log({"step": step, "kind": "val", **val})
                if best is None or val[args.best_metric] > best[0]:
                    best = (val[args.best_metric], step)
                    if args.keep_best:
                        best_state = snapshot(model, trainer)
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.out / f"step-{step}", {"step": step}, trainer)
        epoch += 1

    saved_step = step
    if args.keep_best and val_ids and best_state is not None:
        last = trainer.evaluate(val_ids)
        if best is not None and best[0] > last[args.best_metric]:
            restore(model, trainer, best_state)
            saved_step = best[1]
            log({"step": step, "kind": "keep_best", "restored_step": saved_step, args.best_metric: best[0], "last": last[args.best_metric]})
    trainer.finalize()
    final_dir = args.out / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    save_checkpoint(model, tokenizer, final_dir, {"step": saved_step, "epochs": round(saved_step / steps_per_epoch, 2)})
    if val_ids:
        summary["val_final"] = trainer.evaluate(val_ids)
        log({"step": step, "kind": "val_final", **summary["val_final"]})
        # the same measurement on rows the table was fitted on: a gap to val_final is over-fitting, none is a table
        # that cannot hold more (capacity), whatever the amount of data
        summary["train_final"] = trainer.evaluate(train_ids[: len(val_ids)])
        log({"step": step, "kind": "train_final", **summary["train_final"]})
    summary.update({"steps": step, "saved_step": saved_step, "best_val": {"metric": args.best_metric, "value": best[0], "step": best[1]} if best else None,
                    "best_val_mrr": best if args.best_metric == "mrr" else None, "table_final": encoder_stats(model), "seconds": round(time.time() - started, 1)})
    if trainer.log_scale is not None:
        summary["scale_final"] = round(float(trainer.log_scale.detach().exp()), 5)
    (args.out / "static_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
