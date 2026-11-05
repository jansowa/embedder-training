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

## Tests

`tests/test_distill_*.py` run on the CPU in a few seconds with a 3-layer, 16-dimensional RoBERTa
teacher and a word-level tokenizer (`tests/distill_fixtures.py`): pooling against the dense
reference, losses vanishing on identical models, chunked gradients equal to unchunked, pruning
that keeps every layer reproducing the teacher bit for bit, the ALBERT conversion loading and
calibrating, and a three-step KD run writing a loadable checkpoint.
