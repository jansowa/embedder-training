"""Worst-case GPU memory of one SPLADE training step (all columns padded to max_seq_length).

Loads the model, optionally installs the sparse-gradient head (training/backends/splade_fused_head.py),
runs one fp16/bf16 forward + backward on ``--columns`` micro-batches of ``--micro-batch`` texts padded to
``--max-seq-length`` tokens, then one AdamW step so the optimizer states are allocated, and prints the peak.

Exit code 0 with ``peak_gb=<x>`` on the last line, 3 on CUDA OOM (``peak_gb=OOM``), 1 on other errors.
Example (the 2 x 16 recipe of the 3060 arms, 1 + 1 + 10 columns):

    python scripts/splade_head_memcheck.py --model sdadas/polish-splade --head exact --no-checkpointing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sdadas/polish-splade")
    parser.add_argument("--head", choices=["stock", "exact", "chunked"], default="stock")
    parser.add_argument("--fused-chunk", type=int, default=64)
    parser.add_argument("--pooling-chunk-size", type=int, default=8, help="SpladePooling chunk_size of the stock head")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--document-regularizer-weight", type=float, default=0.001)
    parser.add_argument("--query-regularizer-weight", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda", help="cuda (the measurement) or cpu (a dry run of the code path)")
    args = parser.parse_args()

    import torch
    from sentence_transformers.sparse_encoder import SparseEncoder
    from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
    from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling

    mlm = MLMTransformer(args.model, max_seq_length=args.max_seq_length)
    pooling = SpladePooling(pooling_strategy="max", chunk_size=args.pooling_chunk_size or None)
    model = SparseEncoder(modules=[mlm, pooling]).to(args.device)
    if args.head != "stock":
        from training.backends.splade_fused_head import install_fused_head

        install_fused_head(mlm, pooling, mode=args.head, chunk_size=args.fused_chunk)
    if args.no_checkpointing:
        mlm.auto_model.gradient_checkpointing_disable()
    else:
        mlm.auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    loss = SpladeLoss(
        model,
        loss=SparseMultipleNegativesRankingLoss(model),
        document_regularizer_weight=args.document_regularizer_weight,
        query_regularizer_weight=args.query_regularizer_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    texts = ["słowo " * (2 * args.max_seq_length)] * args.micro_batch
    tokenized = model.tokenize(texts)
    length = int(tokenized["input_ids"].shape[1])
    features = [{k: v.to(args.device) for k, v in tokenized.items()} for _ in range(args.columns)]
    print(f"model {args.model} head {args.head} checkpointing {not args.no_checkpointing} "
          f"columns {args.columns} x {args.micro_batch} x {length} tokens {args.dtype}", flush=True)
    amp_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    cuda = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and args.dtype == "fp16")
    peak = (lambda: torch.cuda.max_memory_allocated() / 2**30) if cuda else (lambda: 0.0)
    model.train()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(2):  # the second step runs with the optimizer states allocated
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(args.device.split(":")[0], dtype=amp_dtype, enabled=cuda):
                value = sum(loss([dict(column) for column in features], labels=None).values())
            scaler.scale(value).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"step {step} loss {float(value):.4f} peak_gb={peak():.2f}", flush=True)
    except torch.OutOfMemoryError:
        print(f"OOM at peak {peak():.2f} GB", flush=True)
        print("peak_gb=OOM")
        return 3
    print(f"peak_gb={peak():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
