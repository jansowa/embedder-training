# Smaller SPLADE students by distillation (`distill/`)

`distill/` is the second module of the repository, next to the contrastive / ranking trainer in
`training/`. It compresses a SPLADE model into a smaller one **with the same tokenizer**, which is
what makes direct distillation possible: teacher and student write their weights into the same
50 001 vocabulary dimensions, so the student can copy the teacher's per-token logits and pooled
vectors on any text, without labels. The only shared code with the trainer is the sparse-gradient
decoder (`training/backends/splade_fused_head.py`).

## Pipeline

| stage | tool | input | what it learns |
|---|---|---|---|
| 0 | `python -m distill.student` | teacher folder | a student folder: layer subset, FFN pruned by importance, optionally the vocabulary factorised to rank *r* (ALBERT output), or a fresh ALBERT body with the inherited vocabulary |
| 1 | `python -m distill.train_kd --token-loss kl` | unlabeled text | per-token distribution over the vocabulary (tempered KL on the teacher's top-k entries + a rest bucket) |
| 2 | `python -m distill.train_kd --pooled-loss mse` | unlabeled text | the pooled SPLADE vector (sparse-gradient max over tokens); optional FLOPS |
| 3 | `python -m training.train` with `loss: sparse_distill_kl_div` | training rows with teacher scores | ranking, exactly as the v32 arms |
| 3' | `python -m training.train` with `loss: sparse_kl_plus_mnrl` | training rows with teacher scores | ranking: the same KL plus `mnrl_weight` x the multiple-negatives ranking loss (the row's positive against its negatives and every passage of the other rows in the physical batch; `scale` is the MNRL scale). Labels and `score_scale` as for the KL loss; the MNRL term needs no label beyond the row's positive. Untested as of 7.11 (meant for the A/B at the MLM start on the 5070) |
| check | `python -m distill.fidelity` | a text sample | cosine / active-dimension overlap with the teacher and seconds per text |
| check | `python -m distill.rank_agreement` | stage-3 rows with teacher scores | Kendall tau / top-1 agreement of the student's dot products with the teacher's per query |
| check | `python -m distill.vocab_usage` | a text sample | live / dead output dimensions and the mass covered by the top-K |

Every output folder is a complete sentence-transformers SparseEncoder (weights, tokenizer,
`modules.json`, `1_SpladePooling/`), so `scripts/pirb_regime_eval.sh`, `scripts/compare_splade_sparsity.py`
and the trainer accept it as a model path. A `student.json` next to the weights records the spec,
parameter counts and an estimate of multiply-accumulates per token.

## Student surgery (`distill/student.py`)

* `--keep-layers 0,1,2,5` keeps those teacher layers in order; attention and LayerNorms are copied.
* `--intermediate-size 1536` keeps the FFN neurons with the highest importance per layer:
  `weight` (default) = ‖W_in[j]‖·‖W_out[:, j]‖; `activation` = mean |GELU(W_in h + b)[j]|·‖W_out[:, j]‖
  over `--calibration-texts`.
* `--embedding-rank 256` projects the 50 001 × 768 embedding onto the top-256 principal subspace of
  its centred rows (an orthonormal basis `Q`, so the rank-space coordinates are a projection of the
  hidden vector) and emits an `AlbertForMaskedLM` (one group per layer, no weight sharing,
  `embedding_size = 256`, decoder tied to the embedding unless `--untie-head`). ALBERT applies its
  LayerNorm in the rank space; for an orthonormal projection that is the teacher's LayerNorm up to a
  constant, so the embedding path is a near-copy and the conversion is exact at full rank (a test
  checks this). A balanced SVD `E ≈ A B` is *not* usable here: it scales the rank coordinates by the
  singular values and the rank-space LayerNorm then computes something else (pooled cosine 0.07 with
  the teacher even at full rank). The head is the real approximation: ALBERT runs the gelu in the rank
  space, which no fit repairs (soup-v9b, rank 256 or 512: pooled cosine 0.13–0.31 after 800 steps).
* `--head-layer` (recommended with `--embedding-rank`) removes that approximation: the teacher's head
  transform `LN(gelu(W h + b))` becomes one extra full-width ALBERT layer (attention zeroed, FFN =
  `gelu(W' n + b') − n` through gelu's linear regime, final LayerNorm = the head's), and the ALBERT
  head is reduced to a linear projection: gelu pushed into its linear regime by a large bias, the
  decoder untied and fitted to the *distribution* of the teacher's pre-decoder vectors on the
  calibration texts (SVD of `E C^½`, `C` = their second moment). The vocabulary itself is not low-rank
  (rank 256 keeps 71 % of its energy), but the decoder on real vectors is: soup-v9b ceiling with this
  decoder is pooled cosine 0.97 at rank 256 / 0.88 at rank 128 before the rank-space LayerNorm and
  0.94 / 0.85 with it (`results/decoder_rank_ceiling.json`, `decoder_rank_layernorm.json` of the
  small-2026-11 series). Costs one extra layer with `intermediate_size ≥ 2H`.
* `--fresh --hidden-size 384 --embedding-rank 256` builds a random body of that width; only the
  vocabulary / decoder / bias come from the teacher.

Why ALBERT and not a custom class: the student must load through `AutoModelForMaskedLM` in
sentence-transformers, PIRB (`third_party/pirb/search/splade.py`) and ONNX exporters without
`trust_remote_code`.

Head-layer students carry `head_layer: true` in their `config.json`; `distill.splade.prepare_head_layer`
(called by the KD trainer, and by `fidelity` / `rank_agreement` with `--fp32-head`) runs the extra layer's
feed-forward and the head transform in fp32 and switches that layer's dropout off: the `gelu(n + b) − n`
cancellation relies on shifted values that fp16 rounds at 1/64–1/16 and that dropout would zero on 10 %
of the coordinates. Stock inference (sentence-transformers, PIRB) has no such patch and runs the fp16
path; `fidelity --dtype fp16` without `--fp32-head` measures what that costs.

## KD trainer (`distill/train_kd.py`)

Plain PyTorch loop: teacher in fp16 without gradients, student in fp32 under autocast, AdamW,
warm-up then constant / linear / cosine LR, gradient accumulation, length-bucketed batches
(`distill/data.py`), checkpoints every N steps with a fidelity sample compared at each
(`log.jsonl` with cosine / L1 / active dims per kind, `kd_summary.json`, optional W&B;
`scripts/kd_fidelity_curve.py` prints the curve). The pooled MSE is relative (divided by the teacher's mean
squared norm), so it combines with a token loss: the token KL is invariant to a shift of all logits of a
token and cannot fix sparsity, while the pooled loss through the sparse-gradient decoder has no gradient
on dimensions the student leaves inactive and can only switch dimensions off; the probe's second stage
therefore runs `--pooled-loss mse --token-loss kl --token-weight 0.3`. The token loss runs the decoder on position chunks
under activation checkpointing (`--head-chunk 64`), so at most one `[B, chunk, V]` logits tensor of
the student is alive; the pooled loss uses the sparse max decoder and never materialises `[B, L, V]`.
The corpus is held in memory (1.1 GB of text for the 2.26 M texts of `kd_v1`), because the data drive
is a spinning disk.

The unlabeled corpus is built by the data kitchen (`corpus/build_kd_corpus.py`, recipe
`corpus/recipes/kd_v1.yaml`): every unique query and passage of the retrieval training rows plus the
Wikipedia chunk pool, deduplicated across sources. Queue helpers: `kd` and `fidelity` in
`scripts/queue_lib.sh`; the first experiment series is `experiments/polish-splade-small-2026-11/`.

## Static query encoder (`distill/static_query.py`, `static_pool.py`, `train_static.py`)

The extreme student: no transformer on the query side at all. Every token id owns a row of `k`
(vocabulary dimension, weight) slots; a query vector is the max (or sum) of its tokens' rows, written into
the same 50 001 dimensions as the SPLADE index, so it is scored against an index built by a full SPLADE
model (asymmetric mode, `PIRB_QUERY_MODEL`). Encoding a query is a lookup and a scatter-max.

* `k = 1`, `indices[t] = t` (`--init idf` / `ones`): one weight per token, only the query's own tokens are
  active; the model of sentence-transformers' `SparseStaticEmbedding` (inference-free query side).
* `k > 1` (`--init credit` / `isolated`): a context-free SPLADE: a token also activates its expansions
  (inflected forms, synonyms). `credit` runs the document model on a query sample and credits every active
  dimension of each pooled vector to the token at the position where the max was reached; a row is the mean
  credit over the queries containing the token, top-k kept (on one query with large k this reproduces the
  teacher vector exactly). Tokens seen in fewer than `--min-count` queries take their row from `<s> t </s>`
  alone (`isolated`, Model2Vec style).

The document side stays frozen: the index of the best model is reused as it is. Training therefore needs no
transformer either: `distill.static_pool` encodes the passages of the training rows once (quantized as the
PIRB index stores them, `round(w * 100)`) and, with `--teacher-queries`, the model's own query vectors.
`distill.train_static` then fits only the table weights (the slots stay as initialised) with a KL over each
row's candidates, against the stored row scores (`--target rows`, e.g. the ensemble teacher of the
`-tens400` files), against the symmetric SPLADE (`--target splade`, optionally `--in-batch` over every
passage of the batch), or a mix. Validation reports, on held-out rows and with the query weights quantized,
Kendall tau / top-1 agreement with the target, positive-first, MRR, dims per query and the posting fraction
(sum of df / N over the query's dims, on the pool), next to the same numbers for the symmetric SPLADE.

```bash
# S0: idf weights, no training
.venv/bin/python -m distill.static_query build --init idf --teacher models/arm-dense-kl-n5kj-ens400 \
    --texts datasets/kd/kd_v1.jsonl --kinds passage --n 300000 --out models/static/idf
# pool of the n5 + kj rows against the frozen document model
.venv/bin/python -m distill.static_pool --doc-model models/arm-dense-kl-n5kj-ens400 \
    --rows datasets/arms/n5-plus-kj-tens400.jsonl --out cache/static_pools/n5kj --teacher-queries
# S2: expansion table from the document model, trained on the ensemble scores
.venv/bin/python -m distill.static_query build --init credit --teacher models/arm-dense-kl-n5kj-ens400 \
    --texts datasets/kd/kd_v1.jsonl --kinds query --top-k 32 --out models/static/credit-k32
.venv/bin/python -m distill.train_static --pool cache/static_pools/n5kj --init models/static/credit-k32 \
    --out runs/static-credit-k32 --target rows --epochs 3
PIRB_QUERY_MODEL=runs/static-credit-k32/final scripts/pirb_regime_eval.sh quantized \
    asym-n5kj-q-static-credit-k32 models/arm-dense-kl-n5kj-ens400 tiny10
```

The rows pool turned out too easy to train on (s4: validation MRR 0.97 before any step, idf-rows = idf on
tiny10). `distill.static_mine` builds the pool the way a retriever is distilled instead: a seeded sample of
unlabelled kd_v1 passages and queries, both encoded by the frozen model, and an exact GPU search of every
query over the passage sample with the model's own query vector and with each `--student` table. The
candidates are the teacher's top passages plus each student's top passages (its own false positives), the
targets are the teacher scores `q_splade . d`; no labels are used. The output is an ordinary pool folder,
trained with `--target rows`. `--learn-scale` multiplies the student scores inside the KL by one learned
scalar, so the KL does not spend the table's weights on matching the teacher's score scale (an idf table
scores several times higher than the SPLADE query it imitates); the lr has to follow the scale of the table
(idf weights are ~8, credit slots ~0.25).

```bash
.venv/bin/python -m distill.static_mine --doc-model models/arm-dense-kl-n5kj-ens400 --corpus datasets/kd/kd_v1.jsonl \
    --n-docs 300000 --n-queries 120000 --student models/static/idf --student models/static/credit-k32 \
    --out cache/static_pools/kd-n5kj
.venv/bin/python -m distill.train_static --pool cache/static_pools/kd-n5kj --init models/static/idf \
    --out runs/static-idf-kd --target rows --learn-scale --lr 5e-2 --epochs 2
```

`python -m distill.static_query stats --model DIR --show pies,psa` prints a few rows; `prune --min-weight
--top-k` cuts the table. The PIRB fork detects a static folder by its `static_query.json`
(`StaticSpladeQueryEncoder` in `third_party/pirb/search/splade.py`).

## Tests

`tests/test_distill_*.py` run on the CPU in a few seconds with a 3-layer, 16-dimensional RoBERTa
teacher and a word-level tokenizer (`tests/distill_fixtures.py`): pooling against the dense
reference, losses vanishing on identical models, chunked gradients equal to unchunked, pruning
that keeps every layer reproducing the teacher bit for bit, the ALBERT conversion loading and
calibrating, and a three-step KD run writing a loadable checkpoint.
