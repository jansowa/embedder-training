# 8-bit GPU inference (torchao)

`release/gpu_quant.py` quantizes a SPLADE `SparseEncoder` in memory with torchao, so the plain Sentence
Transformers API (`encode_query` / `encode_document`) runs on int8 or fp8 tensor cores. `release/gpu_bench.py`
measures every variant against an fp16 / fp32 reference: speed, memory and agreement of the vectors. Nothing is
exported: the schemes are dynamic (no calibration), so a quantized model is a pure function of the published
weights. What a release could ship is the recipe and the measured numbers, not another checkpoint.

Status: written and smoke-tested on the CPU (tests/test_gpu_quant.py, tests/test_fast_head.py, the worker plumbing
with `--device cpu` including torch.compile through inductor). **Never run on a GPU yet**; torchao is not installed in
`.venv` (the fp16 / fast-head variants do not need it).

## The fast head (release/fast_head.py)

`enable_fast_head(model, chunk_tokens)` patches a loaded `SparseEncoder` so the vocabulary decoder returns the masked
token maximum `[B, V]` and `relu`/`log1p` run after it (both are monotone, so the values are the same). `chunk_tokens=0`
is one decoder call over the whole sequence: bitwise identical to the stock vectors (frozen model on the CPU, fp32 and
bf16, 144 kd_v1 texts incl. 512-token ones), with one `[B, L, V]` tensor at the head's peak instead of the stock two
(the stock pooling multiplies by the mask out of place). `chunk_tokens=N` slices the decoder: less memory again, but other
GEMM blockings round differently (fp32: 1e-6, same indexed dims; bf16: one bf16 step, some indexed dims flip).
Variants: `fasthead` (= 0), `fasthead64`; with compile only the transformer body is compiled (`body`), the head runs
eager. `poolchunk<N>` is the zero-code alternative (`SpladePooling.chunk_size`, the logits stay).

The report's `graphs (+timed)` column counts compiled graphs after the warm-up and those added during the timed passes;
anything after the `+` means the timings include compilation.

## Variants

`scheme[:option,option]`, the first `--variants` entry is the reference.

| scheme | weights | activations | GPU | expectation |
|---|---|---|---|---|
| `fp32`, `fp16`, `bf16` | as named | as named | any | baselines (PIRB runs fp16) |
| `int8-dyn` | int8 per channel | int8 per token, dynamic | Ampere+ (cc ≥ 8.0: RTX 30xx, A100, 3060 Ti) | INT8 tensor cores, ~2× fp16 peak on the matmuls |
| `fp8-dyn` | fp8 e4m3 | fp8, dynamic, per row (`tensor`: one scale) | cc ≥ 8.9: Ada, Hopper, Blackwell (5070 Ti) | FP8 tensor cores, ~2× fp16 peak on the matmuls |
| `int8-wo` | int8 | base dtype | any | memory only, a control |

Options: `nohead` (MLM head stays in the base dtype), `eager` / `compile` / `max-autotune` / `reduce-overhead`
(torch.compile mode, default from `--compile`), `fp16` / `bf16` (base dtype of a quantized variant, default bf16),
`row` / `tensor` (fp8 scale granularity).

Why these expectations are modest (estimates, not measurements):

* the 6-layer RoBERTa has ~42.5 M Linear MACs per token in the encoder and ~38.4 M in the vocabulary decoder
  (768 → 50 001), so the decoder is ~47 % of the quantizable work: `nohead` gives away about half of the gain;
* attention, LayerNorm and the SPLADE pooling over `[batch, seq, vocab]` stay in the base dtype;
* dynamic quantization adds a scale/cast kernel in front of every matmul. Without `torch.compile` to fuse them
  the 8-bit variants are usually *slower* than fp16 (hence compile is the default, and `:eager` is the control);
* gains appear at large batch × length (corpus encoding). Single queries are latency-bound: expect ~1.0× or worse.

A rough prior for corpus encoding was 1.2–1.5× over compiled fp16; published evidence (15.11 search) points lower,
roughly 0.9–1.2×: torchao's own W8A8 example reports 1.03× and recommends models above 8B; a TTS study measured W8A8
at 1.11× the latency of compiled fp16 on an RTX PRO 6000 (Blackwell); dynamic int8 on small encoders (DistilBERT,
XLM-R) was below 1× on CPUs because the quantization overhead exceeds the GEMM gain. The larger lever for SPLADE
is the head: Sparton (arXiv 2603.25011, github.com/thongnt99/sparton) fuses decoder matmul + relu + log1p + max and
reports up to 4.8× on the head and an order of magnitude less peak memory, i.e. larger batches on 8 GB cards. The decoder is also the precision-sensitive
part: its logits decide which terms pass `relu` and the 0.005 index cut, so dims per document and postings per
query are as important as the cosine.

## Install (separate venv on the card that runs it)

torchao pins against a torch version; pick the release that matches the venv's torch from the torchao README
compatibility table (`.venv` has torch 2.7.1+cu128). Keep it out of `requirements.lock`.

```bash
uv venv --python 3.10.4 .venv-gpuq && uv pip sync --python .venv-gpuq/bin/python requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --python .venv-gpuq/bin/python "torchao==<matching release>"
```

Open questions to settle on the first run: whether the row-wise fp8 `_scaled_mm` kernel exists for sm_120
(5070 Ti) in this torch (if `fp8-dyn` fails, `fp8-dyn:tensor` uses the per-tensor cuBLASLt path), and whether
torchao's tensor subclasses compile with `dynamic=True` without falling back to eager.

## Running

The parent refuses to start a worker while another process holds the GPU (`--allow-busy-gpu` overrides), so it
cannot slip next to a queue step; on the 3060 box run it as a queue step or by hand on an idle card. Every
variant is a subprocess; one that fails is reported with its error and the run continues.

```bash
# 5070 Ti: the full matrix
.venv-gpuq/bin/python -m release.gpu_bench --model models/soup-v20-dqf-bias015 \
    --variants fp16:eager fp16 fp16:fasthead bf16 int8-dyn int8-dyn:nohead int8-dyn:nohead,fasthead fp8-dyn fp8-dyn:nohead fp8-dyn:nohead,fasthead fp8-dyn:tensor \
    --passage-batch 32,64 --compile-threads 8 \
    --out experiments/polish-splade-2026-10/results/gpu-quant/soup-v20-dqf-bias015_5070ti.json

# 3060 Ti (no fp8): the default variant list = baselines, poolchunk, fast head, int8 (plan of 15.11)
.venv-gpuq/bin/python -m release.gpu_bench --model models/soup-v20-dqf-bias015 \
    --out experiments/polish-splade-2026-10/results/gpu-quant/soup-v20-dqf-bias015_3060ti.json
```

Defaults: 1024 queries and 4096 passages from `datasets/kd/kd_v1.jsonl` (seed 7), agreement on the first 512 /
1024, passage batches 16 and 32 (median of 3 passes; the kd_v1 sample has mean 108 tokens, 7.5 % of the passages ≥ 256,
so the first, longest batches pad to 512 and the `[batch, 512, 50 001]` logits of SpladePooling decide what fits: 8 GB
cards stop at 32), query batch 64, 300 single-query latencies,
`TORCHINDUCTOR_COMPILE_THREADS=1` (RAM on the 16 GB box; `--compile-threads 8` on the 5070). Outputs are moved
to the CPU inside the timing (`--keep-on-gpu` to leave them), because a user needs them there.

Report columns: warm-up seconds (compile cost, paid once per process), query p50 ms at batch 1, queries/s,
passages/s per batch, speed-up (best passages/s over the reference's best), peak CUDA MB, then cosine, Jaccard of
the indexed dims, dims per query / passage, in-sample postings per query (`release.fidelity.postings_per_query`),
and the in-sample top-10 retrieval agreement. `head_bias_mean` in the JSON catches loaders that drop the
vocabulary bias (a checkpoint with only `lm_head.bias` loads without it in fp32 under transformers 4.48; the
released models carry both keys).

## Reading (proposal)

1. A variant matters only if it beats **compiled** fp16 on passages/s at batch 128, not just eager fp16.
2. Agreement against fp16: dims per passage and postings per query within a few per cent, cosine and top-10
   agreement close to what `bf16` (pure rounding, no quantization) gets; `bf16` itself is the noise floor of a
   lower-precision base.
3. The survivors go to PIRB (both sides quantized, tiny10 + heldout8) before anything is written in a model card.

## What a model card could show (after the measurements, versions pinned)

```python
import torch
from sentence_transformers import SparseEncoder
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow

model = SparseEncoder("<repo>", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
am = model[0].auto_model
quantize_(am, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
          filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear))
am.forward = torch.compile(am.forward, dynamic=True)
emb = model.encode_document(docs, batch_size=128)
```

Do not save a quantized model: quantizing `lm_head.decoder` unties it from the word embeddings.
