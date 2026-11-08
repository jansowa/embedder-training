# ONNX exports for release

`release/` turns a finished SPLADE checkpoint (anything `AutoModelForMaskedLM` loads, plus its tokenizer) into
ONNX graphs for CPU inference and measures what each form costs in speed and in retrieval quality. It never
trains and does not import from `training/` or `distill/`.

| file | role |
|---|---|
| `release/onnx_splade.py` | export: fp32 graph + `dynamic-int8`, `static-int8`, `mixed-int8` |
| `release/onnx_encoder.py` | `OnnxSplade`: onnxruntime + `tokenizers` encoder, no torch (what a release user needs) |
| `release/onnx_bench.py` | CPU benchmark: latency, throughput, RSS, agreement with fp32 |
| `release/fidelity.py`, `release/texts.py` | numpy agreement statistics, seeded text samples |
| `third_party/pirb/search/splade.py` | `OnnxSpladeEncoder`: PIRB picks it for any directory with `onnx_splade.json` |
| `experiments/polish-splade-2026-10/queues/queue_onnx.sh` | the whole check for one model (export, CPU bench, PIRB) |

Dependencies: `onnx` and `onnxruntime` (both in `requirements.in`; the CPU build of onnxruntime is enough).

## Variants

Every variant is a self-contained directory: `model.onnx`, `tokenizer.json` and friends, `config.json`, and
`onnx_splade.json` (graph kind, output dimension, max length, quantization settings, agreement with fp32 on
held-out texts).

| variant | weights | activations | calibration | notes |
|---|---|---|---|---|
| `fp32` | fp32 | fp32 | – | the reference; checked against the torch model right after export |
| `dynamic-int8` | int8 per channel | uint8, range computed per batch at run time | none | `MatMulInteger` + `DynamicQuantizeLinear` |
| `static-int8` | int8 per channel | int8, fixed ranges | MinMax (or percentile / entropy) on calibration texts | QDQ graph; ORT fuses it into int8 kernels |
| `mixed-int8` | static int8, but chosen blocks fp32 | as static, fp32 in the kept blocks | shares the static calibration | default: MLM head fp32; `--mixed-auto-k` adds the most sensitive blocks |

What is quantized: MatMul / Gemm with constant weights. The attention products (Q K^T, softmax V) stay fp32 in
every variant (`--quantize-attention-matmuls` to quantize them in static / mixed), and so do the embeddings
unless `--quantize-embeddings` (weight-only int8 Gather; on a 50 k vocabulary the word embeddings are ~40 % of
a 6-layer base model, so this matters for file size, not speed).

Why the head is the default fp32 block: the vocabulary projection produces the logits that go through
`log1p(relu(.))` and a max over tokens, so its quantization error decides which terms exist at all (and the
bias form of the 0.05 threshold, `lm_head.bias −0.0513`, lives exactly there). One int8 scale per tensor for
50 k logits rounds small positive weights to zero or lifts zeros above the 0.005 index cut. The sensitivity probe
(`--mixed-auto-k`) quantizes one block at a time (everything else fp32) and ranks the blocks by 1 − cosine
against fp32 on the check texts; the ranking is written into the mixed variant's metadata either way.

`--mixed-method dynamic` builds the mixed variant on the dynamic graph instead (no calibration; the kept blocks
fp32, the rest `MatMulInteger`). On CPUs without VNNI (AVX2 only) prefer it, and pass `--weight-type u8` or
`--reduce-range` to every int8 variant: onnxruntime's u8s8 int8 GEMM there sums pairs of products in int16 and
saturates, which took r2v12b's first dynamic export to cosine 0.03 against fp32 (tiny10 0.25); with u8 weights
it is 0.967. Fixed static ranges stay weak on such a model (cosine ≤ 0.77, outlier activations before the head
and in one encoder layer), see `experiments/polish-splade-2026-10/results/onnx/fix_r2v12b.md`.

Graph kinds (`--graph`):

* `splade` (default): output `sparse_vector` [batch, vocab], the pooled `max_t log1p(relu(logits)) * mask`,
  identical to PIRB's encoder. Smallest output, nothing to post-process.
* `mlm`: output `logits` [batch, seq, vocab], the layout `sentence-transformers` / optimum load as
  `onnx/model.onnx` (pooling then happens outside). Not tested against the ST ONNX backend here (needs
  `optimum[onnxruntime]`, not installed); the PIRB encoder and `OnnxSplade` pool it themselves.

## Running on a busy box

The export, the calibration and the benchmark are CPU-only (`CUDA_VISIBLE_DEVICES` is cleared before torch is
imported), so they never touch the card a training run holds. RAM is the constraint:

* the export loads the fp32 checkpoint once (a 6-layer base: ~0.4 GB, plus the tracing overhead);
* MinMax calibration adds ReduceMin / ReduceMax nodes and returns two scalars per tensor, so its peak is one
  forward pass of a calibration batch (4 × ≤ 256 tokens: the fp32 logits alone are ~200 MB). The histogram
  methods return every calibrated tensor in full and are fed one batch per call;
* one calibration is cached (`calibration/ranges-<method>.pkl`) and reused by static, mixed and every
  sensitivity trial;
* the benchmark runs every variant in its own subprocess (peak RSS per variant, nothing left resident) and
  refuses to start a worker below `--min-free-gb` (default 4) of available RAM;
* the encoders cap rows × padded length per session run (`max_batch_tokens`, PIRB: `onnx_max_batch_tokens`,
  default 4096 ≈ 0.8 GB per [rows, seq, vocab] intermediate) and slice bigger batches.

Speed numbers are only meaningful on an idle machine; the benchmark records the load average and the GPU
compute processes at start and warns when the box is busy. Exporting next to a training run is possible
(`--threads 2`, `nice`), timing is not. On the 16 GB 3060 box keep it serial anyway (a CPU job next to a queue step has frozen the
machine before): `queue_onnx.sh` waits for an idle GPU and ≥ 8 GB of RAM.

```bash
# export (the calibration texts are a seeded sample of kd_v1)
CUDA_VISIBLE_DEVICES= .venv/bin/python -m release.onnx_splade --model models/arm-dense-kl-n5kj-ens400 \
    --out models/onnx/n5kj --variants fp32,dynamic,static,mixed \
    --calib-texts datasets/kd/kd_v1.jsonl --threads 4 --mixed-auto-k 1

# CPU benchmark (idle machine)
.venv/bin/python -m release.onnx_bench --variants models/onnx/n5kj/{fp32,dynamic-int8,static-int8,mixed-int8} \
    --torch-model models/arm-dense-kl-n5kj-ens400 --threads 1,6 \
    --out experiments/polish-splade-2026-10/results/onnx/n5kj_cpu.json

# PIRB: an export directory works wherever a checkpoint does (do not pass MAXLEN: the fork tokenizes at 512)
PIRB_QUERY_MODEL=models/onnx/n5kj/mixed-int8 bash scripts/pirb_regime_eval.sh quantized onnx-n5kj-q-mixed-int8 \
    models/arm-dense-kl-n5kj-ens400 tiny10          # torch index on the GPU, ONNX queries on the CPU
bash scripts/pirb_regime_eval.sh quantized onnx-n5kj-mixed-int8 models/onnx/n5kj/mixed-int8 tiny10   # both sides ONNX

# all of it, serially, when the box is free
MODEL=models/arm-dense-kl-n5kj-ens400 NAME=n5kj setsid nohup \
    bash experiments/polish-splade-2026-10/queues/queue_onnx.sh > logs/queue_onnx.out 2>&1 < /dev/null &
```

Using an export (what the model card can show):

```python
from release.onnx_encoder import OnnxSplade
enc = OnnxSplade("models/onnx/n5kj/mixed-int8", threads=4)
vectors = enc.encode(["jak działa SPLADE?"])            # float32 [n, vocab]
enc.term_weights(vectors[0], quantization_factor=100)   # {token: impact} as Anserini indexes it
```

## Benchmark plan

Per model (the publication candidates; first n5kj, then the ALBERT query encoder):

1. **Speed, CPU** (`release.onnx_bench`, idle box, 1 and 6 threads): query latency p50 / p90 at batch 1,
   passages/s at batch 8, RSS peak, file size; torch fp32 on the CPU as the baseline the exports must beat.
2. **Agreement with fp32** (same run, kd_v1 sample, seed 7, disjoint from the export's calibration seed 13):
   cosine, Jaccard of the indexed dimensions, **dims per query and per passage** (quantization tends to add
   junk dims; the index cost rule applies), top-10 term overlap, in-sample top-10 retrieval agreement.
3. **Retrieval, PIRB leaderboard regime** (`quantized`):
   * query side only (torch index, ONNX queries): tiny10 + heldout8 for all four variants. This is the
     deployment where users run only the query encoder;
   * both sides ONNX on the CPU: tiny10 for all four (heldout8 via `SYM_TASKSETS` if the time allows; the CPU
     encode of the corpora is the slow part).
   The int8 variants are compared with the **fp32 export**, not with the published torch numbers (fp16 on the GPU).
4. Reading (proposal, to confirm before the runs): a variant is releasable when it stays within the seed noise
   of the fp32 export on heldout8 and does not raise dims per query / per passage by more than a few per cent. Among those, the
   fastest at 1 thread wins; fp32 is published alongside as the reference.

Knobs worth one extra run each if a variant falls short: `--calib-method percentile` (MinMax ranges are set by
outliers), `--reduce-range` (the 5600X has AVX2 without VNNI, where u8 × s8 products can saturate int16;
7-bit weights avoid it; matters for dynamic), `--activation-type u8`, `--mixed-auto-k 2`.

## onnxruntime 1.23 quirks handled in the code

* The QDQ MatMul handler ignores `MatMulConstBOnly`; without the explicit exclusion the attention products
  were int8 in static and in the "fp32" blocks of mixed (`activation_matmul_nodes`).
* `MinMaxCalibrater(max_intermediate_outputs=n)` clears the collected outputs without merging them when the
  limit is hit (the ranges of those batches are lost); the code calls `collect_data` per chunk instead.
* The histogram calibrators stack a call's outputs with `np.asarray`: batches of different lengths fail, so
  they get one batch per call.
* Symbolic shape inference fails on some position-id graphs ("unsupported broadcast between Min(...)");
  the pre-processing then falls back to ONNX shape inference.
* `transformers` imports torch on `AutoTokenizer` (~0.7 GB RSS); `OnnxSplade` reads `tokenizer.json` with
  `tokenizers` directly (ids and masks checked identical to `AutoTokenizer` on 204 kd_v1 texts, empty and
  over-long ones included, at max length 64 and 512).

## Static query table as a Sentence Transformers model

A static query table (`distill.static_query`, the inference-free query side over a SPLADE index) is released as one
`SparseEncoder`: `release.st_export` puts a `StaticQueryRouter` in front of the table (`query` route,
`SparseStaticExpansion`: k (dimension, weight) slots per token, max over the tokens) and the SPLADE checkpoint's own
modules (`document` route). ST's built-in `SparseStaticEmbedding` holds one weight per token on its own dimension, so it
cannot carry the expansion slots; the two classes therefore ship as remote code (`static_query_expansion.py` in the
model root, a copy of `release/st_static.py`) and loading needs `trust_remote_code=True`.

```bash
.venv/bin/python -m release.st_export --doc-model models/arm-dense-kl-n5kj-ens400-r2v12-bias005 \
    --table models/static-r2v12b/credit-k8-kd --out models/st/r2v12b-static-credit-k8-kd
```

```python
from sentence_transformers import SparseEncoder
model = SparseEncoder("models/st/r2v12b-static-credit-k8-kd", trust_remote_code=True)
q = model.encode_query(["jak działa splade?"])     # token lookup, no transformer
d = model.encode_document(["SPLADE to ..."])       # the SPLADE document encoder
```

`tests/test_st_static.py` checks that the packaged query vectors equal `distill.static_query` and the document vectors
equal the plain checkpoint. PIRB evaluates the same table directly (`PIRB_QUERY_MODEL=<table dir>`).
