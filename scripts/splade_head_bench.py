#!/usr/bin/env python
"""Micro-benchmark of the SPLADE training step: stock head vs the sparse-gradient head.

One process loads the model once, takes the rows of one optimizer step from an arm file
(``--rows-per-step`` rows, ``--micro-batch`` rows per forward, 1 query + 1 positive +
``--negatives`` negatives per row, tokenized per column exactly like the ST data collator)
and runs forward + backward of ``SpladeLoss(MNRL)`` on the same micro-batches for every
variant. Dropout masks are identical across variants (the RNG is re-seeded before every
micro-batch), so the gradients of two variants can be compared directly; the stock head is
run twice to show the noise floor of the non-deterministic kernels (embedding backward,
``index_add_``).

Variants (``--variants``):

    stock8        SpladePooling(chunk_size=8) with gradient checkpointing  (today's config)
    stock8-again  the same once more: noise floor
    stock0        SpladePooling(chunk_size=None): no slice gradients
    exact         fused head, exact mode, gradient checkpointing on
    exact-nockpt  fused head, exact mode, gradient checkpointing off
    chunked       fused head, chunked mode (--fused-chunk tokens)
    stock8-fp32   the stock head without autocast (fp32): the precision reference; the stock fp16
                  backward rounds the head gradients to fp16, the fused head accumulates in fp32

Writes ``<output>.json`` and ``<output>.md``. Run it alone on the GPU (scripts/splade_speed_bench.sh
pauses the queue for it).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VARIANTS: dict[str, dict[str, Any]] = {
    "stock8": {"chunk": 8, "fused": None, "ckpt": True},
    "stock8-again": {"chunk": 8, "fused": None, "ckpt": True},
    "stock0": {"chunk": None, "fused": None, "ckpt": True},
    "exact": {"chunk": None, "fused": "exact", "ckpt": True},
    "exact-nockpt": {"chunk": None, "fused": "exact", "ckpt": False},
    "chunked": {"chunk": None, "fused": "chunked", "ckpt": True},
    "stock8-fp32": {"chunk": 8, "fused": None, "ckpt": True, "autocast": False},  # precision reference
}
DEFAULT_VARIANTS = "stock8,stock8-again,stock0,exact,exact-nockpt,stock8-fp32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sdadas/polish-splade")
    parser.add_argument("--rows", required=True, help="FlagEmbedding jsonl with query / pos / neg")
    parser.add_argument("--rows-per-step", type=int, default=32, help="rows of one optimizer step (2 x 16 today)")
    parser.add_argument("--micro-batch", type=int, default=2, help="rows per forward (train_batch_size)")
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--fused-chunk", type=int, default=64, help="token chunk of the 'chunked' variant")
    parser.add_argument("--warmup-batches", type=int, default=2, help="untimed micro-batches before the measured ones")
    parser.add_argument("--repeats", type=int, default=1, help="timed passes over the step's micro-batches per variant")
    parser.add_argument("--document-regularizer-weight", type=float, default=0.001)
    parser.add_argument("--query-regularizer-weight", type=float, default=0.00005)
    parser.add_argument("--loss-scale", type=float, default=1024.0, help="constant stand-in for the GradScaler")
    parser.add_argument("--output", required=True, help="path prefix of the .json / .md report")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="autocast dtype (bf16 for ModernBERT)")
    return parser.parse_args(argv)


def load_rows(path: Path, count: int, negatives: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("pos") and len(row.get("neg") or []) >= negatives:
                rows.append({"query": row["query"], "pos": row["pos"][0], "neg": row["neg"][:negatives]})
    if len(rows) < count:
        raise SystemExit(f"{path}: only {len(rows)} usable rows, need {count}")
    return random.Random(seed).sample(rows, count)


def build_model(model_name: str, max_seq_length: int, device: str):
    from sentence_transformers.sparse_encoder import SparseEncoder
    from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling

    mlm = MLMTransformer(model_name, max_seq_length=max_seq_length)
    model = SparseEncoder(modules=[mlm, SpladePooling(pooling_strategy="max", chunk_size=8)])
    model.max_seq_length = max_seq_length
    return model.to(device)


def tokenize_step(model, rows: list[dict[str, Any]], micro_batch: int, negatives: int, device: str):
    """One optimizer step as a list of micro-batches; each is a list of per-column feature dicts."""
    batches = []
    for start in range(0, len(rows), micro_batch):
        chunk = rows[start : start + micro_batch]
        columns = [[r["query"] for r in chunk], [r["pos"] for r in chunk]]
        columns += [[r["neg"][k] for r in chunk] for k in range(negatives)]
        batches.append([{k: v.to(device) for k, v in model.tokenize(texts).items()} for texts in columns])
    return batches


def set_checkpointing(model, enabled: bool) -> None:
    auto_model = model[0].auto_model
    if enabled:
        auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    else:
        auto_model.gradient_checkpointing_disable()


def run_variant(name: str, spec: dict[str, Any], model, loss_module, batches, args, initial_state) -> dict[str, Any]:
    import torch

    from training.backends.splade_fused_head import install_fused_head, uninstall_fused_head

    uninstall_fused_head(model[0])
    model.load_state_dict(initial_state)  # copies the CPU tensors into the existing GPU parameters
    model[1].chunk_size = spec["chunk"]
    set_checkpointing(model, spec["ckpt"])
    if spec["fused"]:
        install_fused_head(model[0], model[1], mode=spec["fused"], chunk_size=args.fused_chunk)
    model.train()
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    def step(batch_index: int, features) -> float:
        torch.manual_seed(args.seed * 1000 + batch_index)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed * 1000 + batch_index)
        amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda and spec.get("autocast", True)):
            # the ST modules write token_embeddings / sentence_embedding (with their autograd graphs) into
            # the feature dicts; the collator makes fresh dicts every step, so hand over copies here too
            losses = loss_module([dict(column) for column in features], labels=None)
            loss = sum(losses.values()) if isinstance(losses, dict) else losses
        (loss * args.loss_scale).backward()
        return float(loss.detach())

    encoder = getattr(getattr(model[0].auto_model, "base_model", None), "encoder", None)
    print(
        f"[bench] {name}: gradient_checkpointing={getattr(encoder, 'gradient_checkpointing', '?')} "
        f"pooling_chunk={model[1].chunk_size} fused={'splade_fused_head' in model[0].__dict__} "
        f"allocated before warm-up {torch.cuda.memory_allocated() / 2**30:.2f} GB" if use_cuda else "",
        flush=True,
    )
    for i in range(min(args.warmup_batches, len(batches))):
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        step(i, batches[i])
        if use_cuda:
            lengths = [int(f["attention_mask"].shape[1]) for f in batches[i]]
            print(
                f"[bench] {name}: warm-up batch {i} lengths {lengths}: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GB, "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
                flush=True,
            )
    model.zero_grad(set_to_none=True)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times: list[float] = []
    losses: list[float] = []
    for repeat in range(args.repeats):
        for i, features in enumerate(batches):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            value = step(i, features)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            if repeat == 0:
                losses.append(value)
    peak = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else float("nan")
    grads = {
        n: (p.grad.detach().float() / (args.loss_scale * args.repeats)).cpu()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    uninstall_fused_head(model[0])
    return {
        "name": name,
        "spec": {**spec, "fused_chunk": args.fused_chunk if spec["fused"] == "chunked" else None},
        "micro_batch_seconds": times,
        "step_seconds": sum(times) / args.repeats,
        "peak_memory_gb": peak,
        "losses": losses,
        "grads": grads,
    }


def compare(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    import torch

    ref, oth = reference["grads"], other["grads"]
    total_ref = total_diff = 0.0
    worst = ("", 0.0)
    for name, g in ref.items():
        h = oth.get(name)
        if h is None:
            continue
        diff = float(torch.linalg.vector_norm(g - h))
        norm = float(torch.linalg.vector_norm(g))
        total_ref += norm**2
        total_diff += diff**2
        rel = diff / norm if norm > 0 else (0.0 if diff == 0 else float("inf"))
        if rel > worst[1]:
            worst = (name, rel)
    loss_diffs = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(reference["losses"], other["losses"])]
    return {
        "grad_diff_norm": total_diff**0.5,
        "grad_ref_norm": total_ref**0.5,
        "grad_rel_diff": (total_diff**0.5 / total_ref**0.5) if total_ref > 0 else float("nan"),
        "worst_param": worst[0],
        "worst_param_rel_diff": worst[1],
        "max_loss_rel_diff": max(loss_diffs) if loss_diffs else float("nan"),
    }


def fmt(x: float, digits: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{digits}g}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; known: {', '.join(VARIANTS)}")

    rows = load_rows(Path(args.rows), args.rows_per_step, args.negatives, args.seed)
    model = build_model(args.model, args.max_seq_length, args.device)
    from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss

    loss_module = SpladeLoss(
        model,
        loss=SparseMultipleNegativesRankingLoss(model),
        document_regularizer_weight=args.document_regularizer_weight,
        query_regularizer_weight=args.query_regularizer_weight,
    )
    batches = tokenize_step(model, rows, args.micro_batch, args.negatives, args.device)
    token_counts = [int(f["attention_mask"].shape[1]) for b in batches for f in b]
    initial_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}  # off the GPU: the stock path fills it
    print(
        f"[bench] {args.model}: {len(batches)} micro-batches x {len(batches[0])} columns, padded lengths "
        f"min {min(token_counts)} / mean {sum(token_counts) / len(token_counts):.0f} / max {max(token_counts)}; "
        f"attention {getattr(model[0].auto_model.config, '_attn_implementation', '?')}",
        flush=True,
    )

    results = []
    for name in names:
        print(f"[bench] {name} ...", flush=True)
        try:
            result = run_variant(name, VARIANTS[name], model, loss_module, batches, args, initial_state)
        except torch.OutOfMemoryError:
            print(f"[bench] {name}: out of memory; skipped", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        print(
            f"[bench] {name}: {result['step_seconds']:.2f} s/step, peak {result['peak_memory_gb']:.2f} GB, "
            f"loss[0] {result['losses'][0]:.5f}",
            flush=True,
        )
        results.append(result)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    reference = results[0]
    for r in results:
        r["vs_reference"] = compare(reference, r)
    noise = next((r["vs_reference"] for r in results[1:] if r["spec"] == reference["spec"]), None)

    lines = [
        f"# SPLADE head micro-benchmark ({time.strftime('%Y-%m-%d %H:%M')})",
        "",
        f"Model `{args.model}`, rows `{args.rows}`, {len(batches)} micro-batches of {args.micro_batch} rows "
        f"(1 + 1 + {args.negatives} columns), seed {args.seed}, {args.dtype} autocast, loss scale {args.loss_scale:g}; "
        f"padded lengths min {min(token_counts)} / mean {sum(token_counts) / len(token_counts):.0f} / max {max(token_counts)}. "
        f"torch {torch.__version__}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()}.",
        "",
        "Gradient columns compare with the first variant on identical micro-batches and dropout masks: "
        "`rel` = ‖g − g_ref‖ / ‖g_ref‖ over all parameters, `worst` = the parameter with the largest relative difference. "
        f"Noise floor (stock head twice): {fmt(noise['grad_rel_diff']) if noise else 'not measured'}.",
        "",
        "| variant | pooling chunk | fused | ckpt | s/step | speed-up | peak GB | grad rel | worst param (rel) | max loss rel |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in results:
        s, c = r["spec"], r["vs_reference"]
        lines.append(
            f"| {r['name']} | {s['chunk'] if s['chunk'] else 'none'} | {s['fused'] or '-'} | {'on' if s['ckpt'] else 'off'} "
            f"| {r['step_seconds']:.2f} | {reference['step_seconds'] / r['step_seconds']:.2f}x | {r['peak_memory_gb']:.2f} "
            f"| {fmt(c['grad_rel_diff'])} | {c['worst_param']} ({fmt(c['worst_param_rel_diff'])}) | {fmt(c['max_loss_rel_diff'])} |"
        )
    if noise:
        lines += ["", f"Acceptance for a category-B change: grad rel ≤ 2 × noise floor = {fmt(2 * noise['grad_rel_diff'])}."]
    if len(results) > 1:
        lines += ["", "Pairwise ‖g_row − g_col‖ / ‖g_col‖ (the fp32 column, when present, is the precision reference):", ""]
        lines.append("| | " + " | ".join(r["name"] for r in results) + " |")
        lines.append("| --- |" + " ---: |" * len(results))
        for a in results:
            cells = [fmt(compare(b, a)["grad_rel_diff"]) if a is not b else "-" for b in results]
            lines.append(f"| {a['name']} | " + " | ".join(cells) + " |")
    report = "\n".join(lines) + "\n"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(report, encoding="utf-8")
    payload = {
        "args": vars(args),
        "token_counts": token_counts,
        "results": [{k: v for k, v in r.items() if k != "grads"} for r in results],
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
