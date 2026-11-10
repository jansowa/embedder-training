"""Joint training of the document encoder with a static query table ("inference-free SPLADE" on our terms).

So far the static query line kept the document side frozen (the index of the full model) and learned only the
lookup table; that line stalled 4 points below the symmetric model (60.2 / 38.6 against 64.2 / 43.1). Here the
document encoder is trained too, so it can take over the expansion work the table cannot do (a document learns to
emit the surface forms a static query will carry). Training rows are the usual (query, pos, neg, pos_scores,
neg_scores) files with teacher scores; the loss is KL(teacher || student) over each row's candidates, where
``student = scale * q_table . d_splade``, plus the FLOPS regularisers of SPLADE on both sides (ramped over the
first third of the steps, as sentence-transformers does).

  --freeze-table   keep the table fixed (the existing index consumers keep their query side; only the documents move)

Outputs ``<out>/final/doc`` (a SparseEncoder folder, measurable by scripts/pirb_regime_eval.sh) and
``<out>/final/query`` (a static table folder for PIRB_QUERY_MODEL), plus ``joint_summary.json`` and ``log.jsonl``.

  .venv/bin/python -m distill.train_joint_static --rows datasets/arms/n5-plus-kj-tens400.jsonl \\
      --doc-model models/arm-dense-kl-n5kj-ens400-r2v12-bias005 --table models/static-r2v12b/phr2-k8-kd-r2 \\
      --out runs/joint-phr2-r2v12b --epochs 2 --rows-per-step 2 --grad-accum 16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from distill.splade import active_dims, encode_hidden, flops_loss, pooled_vector
from distill.static_query import StaticSparseQueryEncoder
from distill.student import copy_aux_files
from distill.train_kd import lr_at

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, type=Path, nargs="+")
    p.add_argument("--doc-model", required=True, type=Path)
    p.add_argument("--table", required=True, type=Path, help="static query folder (distill.static_query)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--freeze-table", action="store_true")
    p.add_argument("--max-neg", type=int, default=10)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--rows-per-step", type=int, default=2, help="rows per micro-batch")
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-5, help="document encoder")
    p.add_argument("--table-lr", type=float, default=5e-3)
    p.add_argument("--scale-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--schedule", choices=("constant", "linear", "cosine"), default="linear")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--doc-flops", type=float, default=1e-3, help="FLOPS weight on the documents (ramped over the first third)")
    p.add_argument("--query-flops", type=float, default=5e-5, help="FLOPS weight on the table's query vectors")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--precision", choices=tuple(DTYPES), default="fp16")
    p.add_argument("--head-chunk", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--val-rows", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def read_rows(paths: list[Path], max_neg: int) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                pos, neg = list(r.get("pos") or []), list(r.get("neg") or [])[:max_neg]
                ps, ns = r.get("pos_scores"), r.get("neg_scores")
                if not pos or ps is None or ns is None or len(ps) < len(pos) or len(ns) < len(neg):
                    continue
                docs = pos + neg
                scores = [float(x) for x in list(ps)[: len(pos)] + list(ns)[: len(neg)]]
                if any(s != s for s in scores):
                    continue
                rows.append({"query": r["query"], "docs": docs, "scores": scores, "n_pos": len(pos)})
    return rows


class Joint:
    def __init__(self, doc_model, table: StaticSparseQueryEncoder, tokenizer, args: argparse.Namespace) -> None:
        self.doc_model, self.table, self.tokenizer, self.args = doc_model, table, tokenizer, args
        self.device = args.device
        self.dtype = DTYPES[args.precision]
        self.autocast = self.dtype != torch.float32 and self.device != "cpu"
        self.log_scale = torch.nn.Parameter(torch.zeros((), device=self.device))

    def encode_docs(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(texts, truncation=True, max_length=self.args.max_seq_length, padding=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=self.autocast):
            hidden = encode_hidden(self.doc_model, ids, mask)
            vec = pooled_vector(self.doc_model, hidden, mask, self.args.head_chunk)
        return vec.float()

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        enc = self.table.tokenize(self.tokenizer, texts, self.device)
        return self.table(enc["input_ids"], enc["attention_mask"])

    def row_scores(self, rows: list[dict[str, Any]], q: torch.Tensor, d: torch.Tensor) -> list[torch.Tensor]:
        out, start = [], 0
        for b, r in enumerate(rows):
            n = len(r["docs"])
            out.append(q[b] @ d[start : start + n].T)
            start += n
        return out

    @torch.no_grad()
    def init_scale(self, rows: list[dict[str, Any]]) -> float:
        q = self.encode_queries([r["query"] for r in rows])
        d = self.encode_docs([t for r in rows for t in r["docs"]])
        s = torch.cat(self.row_scores(rows, q, d))
        t = torch.tensor([x for r in rows for x in r["scores"]], device=self.device)
        s_std, t_std = float(s.std()), float(t.std())
        if s_std > 0 and t_std > 0:
            self.log_scale.fill_(math.log(t_std / s_std))
        return float(self.log_scale.exp())

    def loss(self, rows: list[dict[str, Any]], ramp: float) -> tuple[torch.Tensor, dict[str, float]]:
        args = self.args
        q = self.encode_queries([r["query"] for r in rows])
        d = self.encode_docs([t for r in rows for t in r["docs"]])
        scale = self.log_scale.exp()
        kl = q.new_zeros(())
        for b, s in enumerate(self.row_scores(rows, q, d)):
            t = torch.tensor(rows[b]["scores"], device=self.device)
            log_p_t = F.log_softmax(t / args.temperature, dim=-1)
            log_p_s = F.log_softmax(s * scale / args.temperature, dim=-1)
            kl = kl + (log_p_t.exp() * (log_p_t - log_p_s)).sum() * args.temperature**2
        kl = kl / len(rows)
        parts = {"kl": float(kl)}
        total = kl
        if args.doc_flops:
            fd = flops_loss(d)
            total = total + ramp * args.doc_flops * fd
            parts["doc_flops"] = float(fd)
        if args.query_flops and not args.freeze_table:
            fq = flops_loss(q)
            total = total + ramp * args.query_flops * fq
            parts["query_flops"] = float(fq)
        parts["dims_doc"] = float(active_dims(d.detach()).float().mean())
        parts["dims_query"] = float(active_dims(q.detach()).float().mean())
        parts["scale"] = float(scale)
        return total, parts

    @torch.no_grad()
    def evaluate(self, rows: list[dict[str, Any]], batch: int = 4) -> dict[str, float]:
        self.doc_model.eval()
        top1, pos_first, mrr, dd, dq = [], [], [], [], []
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            q = torch.round(self.encode_queries([r["query"] for r in chunk]) * 100) / 100
            d = torch.round(self.encode_docs([t for r in chunk for t in r["docs"]]) * 100) / 100
            dd.extend(active_dims(d).tolist())
            dq.extend(active_dims(q).tolist())
            for b, s in enumerate(self.row_scores(chunk, q, d)):
                s = s.tolist()
                t = chunk[b]["scores"]
                order = sorted(range(len(s)), key=lambda i: -s[i])
                pos_first.append(float(order[0] < chunk[b]["n_pos"]))
                rank = next(k for k, i in enumerate(order) if i < chunk[b]["n_pos"])
                mrr.append(1.0 / (rank + 1))
                top1.append(float(order[0] == max(range(len(t)), key=t.__getitem__)))
        self.doc_model.train()
        n = max(len(top1), 1)
        return {
            "rows": len(top1), "top1_agreement": round(sum(top1) / n, 4), "positive_first": round(sum(pos_first) / n, 4),
            "mrr": round(sum(mrr) / n, 4), "dims_per_doc": round(sum(dd) / max(len(dd), 1), 1), "dims_per_query": round(sum(dq) / max(len(dq), 1), 1),
        }


def save_final(joint: Joint, out: Path, state: dict[str, Any]) -> Path:
    final = out / "final"
    (final / "doc").mkdir(parents=True, exist_ok=True)
    joint.doc_model.save_pretrained(final / "doc", safe_serialization=True)
    copy_aux_files(joint.args.doc_model, final / "doc")
    joint.table.meta["train_state"] = state
    joint.table.save(final / "query", joint.tokenizer)
    (final / "joint_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.doc_model)
    doc_model = AutoModelForMaskedLM.from_pretrained(args.doc_model, torch_dtype=torch.float32).to(args.device)
    doc_model.train()
    if args.gradient_checkpointing:
        doc_model.gradient_checkpointing_enable()
    table = StaticSparseQueryEncoder.load(args.table, args.device)
    if table.output_dim != doc_model.config.vocab_size:
        print(f"[ERROR] table dim {table.output_dim} != vocabulary {doc_model.config.vocab_size}", file=sys.stderr)
        return 2
    table.values.requires_grad_(not args.freeze_table)

    rows = read_rows(args.rows, args.max_neg)
    if len(rows) < 2 * args.val_rows + 10:
        print(f"[ERROR] only {len(rows)} usable rows", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_rows, train_rows = rows[: args.val_rows], rows[args.val_rows :]
    micro_per_epoch = math.ceil(len(train_rows) / args.rows_per_step)
    steps_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    warmup = int(args.warmup_ratio * total_steps)
    ramp_steps = max(total_steps // 3, 1)
    joint = Joint(doc_model, table, tokenizer, args)
    scale0 = joint.init_scale(train_rows[: 4 * args.rows_per_step])

    decay, no_decay = [], []
    for name, p in doc_model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay, "base_lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "base_lr": args.lr},
        {"params": [joint.log_scale], "weight_decay": 0.0, "base_lr": args.scale_lr},
    ]
    if not args.freeze_table:
        groups.append({"params": [table.values], "weight_decay": 0.0, "base_lr": args.table_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    scaler = torch.amp.GradScaler(args.device.split(":")[0], enabled=(joint.dtype == torch.float16 and joint.autocast))
    config = {**vars(args), "rows": [str(p) for p in args.rows], "doc_model": str(args.doc_model), "table": str(args.table), "out": str(args.out),
              "train_rows": len(train_rows), "val_rows": len(val_rows), "total_steps": total_steps, "steps_per_epoch": steps_per_epoch, "scale_init": scale0}
    (args.out / "joint_config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    log_handle = (args.out / "log.jsonl").open("a", encoding="utf-8")

    def log(record: dict[str, Any]) -> None:
        log_handle.write(json.dumps(record, sort_keys=True) + "\n")
        log_handle.flush()

    val0 = joint.evaluate(val_rows)
    log({"step": 0, "val": val0})
    print(f"[INFO] {len(train_rows)} train rows, {len(val_rows)} val rows, {total_steps} steps, scale {scale0:.3f}; val at start {val0}", flush=True)

    step = micro = 0
    window: dict[str, float] = {}
    started = time.time()
    done = False
    epoch = 0
    while not done:
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.rows_per_step):
            batch = [train_rows[i] for i in order[start : start + args.rows_per_step]]
            ramp = min(1.0, (step + 1) / ramp_steps)
            loss, parts = joint.loss(batch, ramp)
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            for k, v in parts.items():
                window[k] = window.get(k, 0.0) + v
            window["loss"] = window.get("loss", 0.0) + float(loss)
            window["n"] = window.get("n", 0.0) + 1
            if micro % args.grad_accum != 0:
                continue
            lr = lr_at(step, total_steps, warmup, args.schedule, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["base_lr"]
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                n = max(window.get("n", 1.0), 1.0)
                record = {"step": step, "epoch": epoch, "lr_factor": lr, "grad_norm": grad_norm, "elapsed_s": round(time.time() - started, 1)}
                record.update({k: round(v / n, 5) for k, v in window.items() if k != "n"})
                log(record)
                print(f"[STEP {step}/{total_steps}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items() if k != "step"), flush=True)
                window = {}
            if args.eval_every and step % args.eval_every == 0 and step < total_steps:
                val = joint.evaluate(val_rows)
                log({"step": step, "val": val})
                print(f"[VAL {step}] {val}", flush=True)
            if step >= total_steps:
                done = True
                break
        epoch += 1
        if not done and args.max_steps == 0 and epoch >= math.ceil(args.epochs):
            done = True
    val = joint.evaluate(val_rows)
    state = {"steps": step, "seconds": round(time.time() - started, 1), "scale": float(joint.log_scale.exp()), "val_final": val, "val_init": val0, "freeze_table": args.freeze_table}
    final = save_final(joint, args.out, state)
    (args.out / "joint_summary.json").write_text(json.dumps({**state, "final": str(final)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_handle.close()
    print(f"[DONE] {json.dumps(state, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
