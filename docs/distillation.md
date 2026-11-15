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
--top-k` cuts the table and `prune --max-length N` changes how many tokens of a query are read (the tables were
built with 128, which cuts 91 % of the arguana-pl queries; the rows themselves do not depend on it). The PIRB fork detects a static folder by its `static_query.json`
(`StaticSpladeQueryEncoder` in `third_party/pirb/search/splade.py`).

## Improving the static query encoder: two paths

The released table leaves about 4.3 points to the full query side (60.74 / 39.24 against 65.06 / 43.64 on tiny10 /
heldout8, st1), while a small contextual student over the same index leaves 0.3: what the table lacks is the context
of the query, not expansion as such. Training its weights barely moves the agreement with the teacher (top-1 0.676 →
0.685) and 2.5 times more queries brought +0.09, so the limit is what the table can hold, not the data. Work on it
is split into two paths that share neither a format nor a name, so a result of one cannot be taken for the other.

| | path A: the released architecture, stronger | path B: a larger static encoder |
|---|---|---|
| format | `static-sparse-query/1` (`static_query.json`) | `static-sparse-query/2` (`static_query_v2.json`) |
| shape | the rows and the k of the released table | free: more rows, more slots, more parameters |
| arithmetic | maximum over rows of non-negative slots | sum, signed weights, threshold, context gates, length multipliers |
| loaders | unchanged: `StaticSpladeQueryEncoder` in the PIRB fork, `release/st_static.py`; the result replaces the table file of the package | `search/splade_static_ext.py` in the PIRB fork; no Sentence Transformers module until a variant is chosen |
| code | `distill.train_static` (options), `distill.static_slots`, `distill.static_diag` | `distill.static_v2`, `distill.train_static_vec`; the trainer and the pool miner are shared |
| names | `models/static-<tag>/…`, labels `asym-<tag>-q-static-…` | `models/static2-<tag>/…`, labels `asym-<tag>-q-static2-…` |

`distill.static_query.load_static_encoder` reads either folder; a folder never carries both formats. A path-B
encoder that ends up using nothing beyond format 1 (more rows or slots, the same arithmetic) goes back with
`python -m distill.static_v2 export-v1`.

### Diagnostics first (`distill/static_diag.py`)

Oracles are upper bounds measured without training anything. Each mixes the teacher's query vector `t` with the
table's `s`: `weights` = `t` on the dimensions the table activates (the ceiling of re-weighting the table's slots in
context: path B's gates; what is still missing to the teacher is the ceiling of new slots and rows), `support` = `s`
on the dimensions the teacher activates (the ceiling of noise suppression: residual rows, a threshold, replace mode),
`own` / `student-own` = either vector on the query's own tokens (what query-side expansion is worth).

```bash
# on the held-out rows of a mined pool (seconds on the GPU): agreement with the teacher's ranking
.venv/bin/python -m distill.static_diag oracles --pool cache/static_pools/kd-X-r2 --model models/static-X/phr2-k8-kd-r2 \
    --out experiments/polish-splade-small-2026-11/results/static-X/oracles_phr2-k8-kd-r2.json
# as nDCG@10 on a task set: an oracle folder is a query model of the PIRB fork
.venv/bin/python -m distill.static_diag oracle-model --table models/static-X/phr2-k8-kd-r2 --teacher models/X \
    --mode weights --out models/static-X/oracle-weights
PIRB_QUERY_MODEL=models/static-X/oracle-weights scripts/pirb_regime_eval.sh quantized asym-X-q-oracle-weights models/X tiny10
# which words fall back to subword pieces, which pairs own a row (CPU, tokenizer and table only)
.venv/bin/python -m distill.static_diag coverage --model models/static-X/phr2-k8-kd-r2 --texts queries.jsonl --field text
```

### Path A: the same table, fitted better

* `distill.train_static` gained options that change how the weights are fitted and never the format:
  `--optimizer sparse_adam | adagrad` (the default Adam treats the zero gradient of an absent row as a gradient, so a
  rare row keeps moving on momentum after its single occurrence), `--value-floor` (no slot dies behind the relu; slots
  under half a quantization step are written as zeros), `--dim-gain` (one multiplier per vocabulary dimension, shared
  by all rows and folded into the weights on saving, so rows too rare to be trained on their own still move),
  `--keep-best` / `--best-metric`, `--temperature`. The summary now also carries `train_final`, the validation
  measurement on rows the table was fitted on: no gap to `val_final` means capacity, a gap means over-fitting.
* `distill.static_slots` learns the slots, which were frozen at the credit init. For every pool query it takes the
  direction in which the query vector should move to lower the KL, `scale * (p_teacher - p_student) @ D`, keeps the
  dimensions the query does not write yet, credits them to every row of the query and, over the pool, picks per row
  the dimensions wanted consistently in its presence. They are written into empty or dead slots (the table keeps its
  shape) with a small weight; `distill.train_static` then decides their worth. Rounds can alternate.
* A query-side `min_weight` sweep needs no code: under max pooling `python -m distill.static_query prune
  --min-weight w` is exactly the pooled-vector threshold (`PIRB_MIN_WEIGHT` would also cut the index).

```bash
.venv/bin/python -m distill.static_slots --pool cache/static_pools/kd-X-r2 --model models/static-X/phr2-k8-kd-r2 \
    --out models/static-X/phr2-k8-kd-r2-g1 --new-slots 2 --device cuda
.venv/bin/python -m distill.train_static --pool cache/static_pools/kd-X-r2 --init models/static-X/phr2-k8-kd-r2-g1 \
    --out runs/static-X-phr2-k8-kd-r2-g1 --target rows --learn-scale --lr 5e-3 --epochs 3 --value-floor 1e-4 --keep-best
```

### Path B: a larger static encoder (`distill/static_v2.py`)

Still no transformer: lookups, one scatter and, with a gate, a dot product per row. Every switch is off by default
(a converted table computes exactly what format 1 computes, gates start at 1), so each is measured on its own:

| switch | what it adds | size |
|---|---|---|
| `--pooling sum --signed`, `--threshold` | `relu(sum of rows - threshold)`: rows vote, phrase rows can take back what their pieces write; `build --rows residual` initialises phrase rows as the phrase's credit minus what its parts already write | unchanged |
| `--phrase-mode replace` | a word that owns a row is encoded by that row alone | unchanged |
| `--pair-key first`, `--pair-window n` | pair rows keyed by the first piece of each word (one row for the inflected forms of a pair), matched up to n words apart | more rows |
| `--ctx-dim d` | context gate: a row's weights times `2 * sigmoid(g_row . c + b_row)`, `c` = mean of d-dimensional token embeddings of the query | + (tokens + rows) x d floats |
| `--slot-ctx-dim s` | a gate per slot on a projection of `c` (which expansions fit this query) | + rows x k x s floats |
| `--length-edges` | one learned multiplier of the expansion slots per query-length bucket | a few scalars |
| `--top-m` | only the m heaviest dimensions of the pooled vector are kept | unchanged |

More rows or more slots alone (`--top-k`, `--max-phrases`, `--kinds query,passage` at build time,
`static_slots --new-k`) are also path B: the arithmetic is the released one, the table is larger.

```bash
# the released table + a context gate; first fitted to the teacher's query vectors on plain text, then to the ranking
.venv/bin/python -m distill.static_v2 convert --model models/static-X/phr2-k8-kd-r2 --ctx-dim 32 --out models/static2-X/gate32
.venv/bin/python -m distill.train_static_vec --teacher models/X --init models/static2-X/gate32 --texts datasets/kd/kd_v1.jsonl \
    --kinds query --n 400000 --train extra --extra-lr 1e-2 --epochs 2 --out runs/static2-X-gate32-vec
.venv/bin/python -m distill.train_static --pool cache/static_pools/kd-X-r2 --init runs/static2-X-gate32-vec/final \
    --out runs/static2-X-gate32-kd --target rows --learn-scale --lr 5e-3 --extra-lr 5e-3 --epochs 3
# signed sum with residual phrase rows
.venv/bin/python -m distill.static_v2 build --rows residual --teacher models/X --texts datasets/kd/kd_v1.jsonl --kinds query \
    --top-k 8 --phrases 2 --out models/static2-X/res2-k8
PIRB_QUERY_MODEL=models/static2-X/res2-k8 scripts/pirb_regime_eval.sh quantized asym-X-q-static2-res2-k8 models/X tiny10
```

`distill.train_static_vec` minimises the squared distance to the teacher's query vector over the whole vocabulary,
on any text and without a pool: every dimension of every query is supervised, which is what parameters that depend
on the context need. It knows nothing about ranking, so `distill.train_static` on a mined pool follows it
(`--train values | extra | all` picks the table weights, the encoder's other parameters or both; `distill.static_mine`
accepts encoders of either format as students).

## Tests

`tests/test_distill_*.py` run on the CPU in a few seconds with a 3-layer, 16-dimensional RoBERTa
teacher and a word-level tokenizer (`tests/distill_fixtures.py`): pooling against the dense
reference, losses vanishing on identical models, chunked gradients equal to unchunked, pruning
that keeps every layer reproducing the teacher bit for bit, the ALBERT conversion loading and
calibrating, and a three-step KD run writing a loadable checkpoint.
