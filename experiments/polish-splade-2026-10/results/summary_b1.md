# b1 (quantized regime, nDCG@10): a larger backbone after KD from the frozen full model

Student: `sdadas/polish-roberta-base-v2` (12 layers, 124 M) as a SparseEncoder (MLMTransformer with the model's own MLM head + SpladePooling max). KD: the recipe of the ALBERT student (token KL x 0.3 + pooled MSE on all of kd_v1, one epoch, lr 5e-5, warmup 300, linear, fp16, 256 tokens, batch 16, seed 42), teacher = the raw weights of the frozen full model (`models/soup-v20-dqf`). Ranking versions start from the KD model: **R-a** = the rows and the recipe of dose3 unchanged (only the start model differs from dose3; the "little data" reference); **R-c'** = stage 1 on 200 000 MS MARCO-pl rows alone (four segments of 50 000, the recipe of R-a with one epoch), then a round on the dose3 rows with the recipe of R-a. R-b (stage 3 on n5 + kj) was dropped on the user's decision of 15.11: the 82 M family started from a model trained on all of MS MARCO-pl, the 124 M student has seen no ranking data, so the amount of ranking data is the open question. Every model is measured in its first form (no shift, then bias steps of 0.05) with at most 410 dims/doc and at most 16807 postings/query.

## Precondition

- Tokenizer against the frozen model: OK vocab 50001, mapping identical True, special tokens identical True, texts with different ids 0 of 5000.
- Memory profile on the 8 GB card (peak = memory used on the whole card):

```
kd_plain	exit 0, peak 5170 MiB, 82.31 examples/s, OOM lines 0
kd_mode	plain
kd_eps	82.31
kd_peak	5170
rank_plain	exit 1, checkpoint , peak 7878 MiB, OOM lines 2, 35 s for 40 steps
rank_gc	exit 0, checkpoint yes, peak 3745 MiB, OOM lines 0, 377 s for 40 steps
rank_mode	gc
rank_peak	3745
rank_s40	377
```

- The first profile attempt (15.11 01:11) ran out of memory in both KD modes: the student before KD activates about 46 800 dimensions per text, so the backward of the sparse decoder asked for entries x hidden floats in one piece (2.3 GiB). It now goes in slices of 32 768 entries (the same sums, `training/backends/splade_fused_head.py`); the logs of that attempt are in `logs/b1-aside/`.
- KD mode: plain. Ranking mode: gc (the dose3 recipe with gradient checkpointing; batch 2 x 16 and 512 tokens unchanged). Training time: kd:rbase-v2-dqf-full 25986s; train:arm-b1-ra 29012s; 

## Fidelity to the teacher (`soup-v20-dqf`)

| student | texts | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |
|---|---:|---:|---:|---:|---:|---:|---:|
| roberta-base-v2 before KD, corpus texts | 256 | 0.0947 | 0.999 | 0.009 | 684.202 | 428 | 46797 |
| roberta-base-v2 before KD, queries | 512 | 0.0599 | 0.995 | 0.003 | 1337.988 | 108 | 39754 |
| roberta-base-v2 after KD, queries | 512 | 0.9831 | 0.846 | 0.755 | 0.195 | 108 | 105 |
| new ALBERT (50 M) before KD, queries | 512 | 0.5481 | 0.352 | 0.155 | 1.510 | 108 | 169 |
| new ALBERT (50 M) after KD, queries | 512 | 0.9722 | 0.804 | 0.691 | 0.258 | 108 | 105 |

Fidelity curve of the KD (256 corpus texts per checkpoint):

```
  step      0: all: cos 0.094 l1 687.951 dims 46895 | passage: cos 0.101 l1 570.758 dims 48197 | query: cos 0.061 l1 1302.500 dims 40065 (teacher dims 487)
  step   6000: all: cos 0.931 l1 0.401 dims 352 | passage: cos 0.930 l1 0.408 dims 401 | query: cos 0.941 l1 0.369 dims 94 (teacher dims 487)
  step  12000: all: cos 0.943 l1 0.367 dims 364 | passage: cos 0.940 l1 0.375 dims 414 | query: cos 0.954 l1 0.325 dims 97 (teacher dims 487)
  step  18000: all: cos 0.951 l1 0.343 dims 383 | passage: cos 0.948 l1 0.352 dims 436 | query: cos 0.963 l1 0.301 dims 106 (teacher dims 487)
  step  24000: all: cos 0.954 l1 0.330 dims 388 | passage: cos 0.952 l1 0.337 dims 442 | query: cos 0.965 l1 0.295 dims 108 (teacher dims 487)
  step  30000: all: cos 0.956 l1 0.332 dims 417 | passage: cos 0.954 l1 0.340 dims 475 | query: cos 0.966 l1 0.289 dims 113 (teacher dims 487)
  step  36000: all: cos 0.960 l1 0.309 dims 399 | passage: cos 0.958 l1 0.318 dims 455 | query: cos 0.970 l1 0.264 dims 108 (teacher dims 487)
  step  42000: all: cos 0.962 l1 0.303 dims 412 | passage: cos 0.960 l1 0.312 dims 470 | query: cos 0.972 l1 0.258 dims 109 (teacher dims 487)
  step  48000: all: cos 0.963 l1 0.292 dims 387 | passage: cos 0.961 l1 0.300 dims 440 | query: cos 0.972 l1 0.251 dims 106 (teacher dims 487)
  step  54000: all: cos 0.964 l1 0.294 dims 416 | passage: cos 0.962 l1 0.302 dims 474 | query: cos 0.974 l1 0.253 dims 115 (teacher dims 487)
  step  60000: all: cos 0.966 l1 0.284 dims 415 | passage: cos 0.964 l1 0.293 dims 473 | query: cos 0.976 l1 0.235 dims 109 (teacher dims 487)
  step  66000: all: cos 0.967 l1 0.283 dims 424 | passage: cos 0.965 l1 0.293 dims 483 | query: cos 0.976 l1 0.235 dims 111 (teacher dims 487)
  step  72000: all: cos 0.968 l1 0.274 dims 420 | passage: cos 0.967 l1 0.283 dims 479 | query: cos 0.978 l1 0.227 dims 111 (teacher dims 487)
  step  78000: all: cos 0.969 l1 0.271 dims 405 | passage: cos 0.967 l1 0.279 dims 461 | query: cos 0.978 l1 0.227 dims 109 (teacher dims 487)
  step  84000: all: cos 0.970 l1 0.265 dims 407 | passage: cos 0.968 l1 0.274 dims 465 | query: cos 0.979 l1 0.219 dims 108 (teacher dims 487)
  step  90000: all: cos 0.970 l1 0.262 dims 404 | passage: cos 0.968 l1 0.270 dims 460 | query: cos 0.979 l1 0.219 dims 108 (teacher dims 487)
  step  96000: all: cos 0.971 l1 0.259 dims 415 | passage: cos 0.970 l1 0.267 dims 473 | query: cos 0.980 l1 0.215 dims 112 (teacher dims 487)
  step 102000: all: cos 0.972 l1 0.256 dims 428 | passage: cos 0.971 l1 0.265 dims 488 | query: cos 0.981 l1 0.208 dims 113 (teacher dims 487)
  step 108000: all: cos 0.973 l1 0.250 dims 413 | passage: cos 0.971 l1 0.259 dims 470 | query: cos 0.982 l1 0.206 dims 113 (teacher dims 487)
  step 114000: all: cos 0.974 l1 0.248 dims 424 | passage: cos 0.972 l1 0.257 dims 483 | query: cos 0.982 l1 0.203 dims 115 (teacher dims 487)
  step 120000: all: cos 0.974 l1 0.246 dims 426 | passage: cos 0.973 l1 0.254 dims 486 | query: cos 0.982 l1 0.201 dims 112 (teacher dims 487)
  step 126000: all: cos 0.975 l1 0.241 dims 418 | passage: cos 0.973 l1 0.249 dims 476 | query: cos 0.983 l1 0.198 dims 113 (teacher dims 487)
  step 132000: all: cos 0.975 l1 0.238 dims 416 | passage: cos 0.974 l1 0.246 dims 474 | query: cos 0.984 l1 0.195 dims 112 (teacher dims 487)
  step 138000: all: cos 0.976 l1 0.236 dims 419 | passage: cos 0.974 l1 0.244 dims 477 | query: cos 0.984 l1 0.192 dims 113 (teacher dims 487)
  step 141543: all: cos 0.976 l1 0.235 dims 419 | passage: cos 0.974 l1 0.244 dims 477 | query: cos 0.984 l1 0.192 dims 113 (teacher dims 487)
```

After KD alone: `kd-rbase-v2-dqf-full-bias015` tiny10 64.69 at 372.7 dims/doc, 120.9 dims/query, 14064 postings/query (552.8 dims/doc without a shift). The new ALBERT after KD alone: 64.03 at 554.8 dims/doc without a shift (not within the limits).

## Ranking versions against dose3

| model | label | tiny10 | heldout8 | trec-covid | tiny10 vs dose3 | heldout8 vs dose3 | dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dose3 (82 M, v56), the reference | `arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015` | 64.96 | 43.63 | 72.82 | 0.0 | 0.0 | 374.7 | 121.9 | 14079 |
| frozen full model (82 M) | `soup-v20-dqf-bias015` | 65.06 | 43.64 | 72.48 | 0.1 | 0.01 | 371.4 | 123.0 | 13945 |
| R-a: roberta-base-v2 after KD, dose3 recipe (124 M) | `arm-b1-ra-bias015` | 65.34 | 43.76 | 67.39 | 0.38 | 0.13 | 361.8 | 130.8 | 15462 |
| ALBERT (50 M, a1) | `arm-hl-dqf-full-s3n5kj-r2d3-bias020` | 64.34 | 42.89 | 64.8 | -0.62 | -0.74 | 339.8 | 120.7 | 12725 |

Unbiased dims/doc: KD model 552.8, R-a 592.7, R-c' after stage 1 , R-c' after the dose3 round . One seed each; the seed noise of the ranking recipe is 0.13-0.2.

R-a against dose3: 0.38 on tiny10, 0.13 on heldout8 (neither >= +0.2 on both nor <= -0.3 on both).

## MS MARCO-pl rows of R-c'

Source: the MS MARCO-pl train rows (`speakleash/msmarco_pl`; the pairs of `datasets/msmarco-full-judged-qwen/pairs_msmarco_full.unique.jsonl` are the positives of the same rows) through the trainer's filter `splade_positive_score_gt_23_50_no_metricx.yaml`. Rows before the filter: 388699 (500732 query-positive pairs); after the filter: 317668 (418661 pairs); after the exclusions: **311080** (benchmark queries of pirb-queries, by normalised text or sorted tokens: 686; queries of the dose3 rows: 2; repeated query texts: 5900). One positive per query (the highest gate score; 95256 further positives left out, 39088 of the kept ones are positives added by the two-stage mining of the source). Shuffled once (seed 42) and cut into units of 25000 rows; stage 1 uses the first 8 units, the others are for the 5070.

Negatives and scores as in dose3 (`queue_v56.sh`): miner `arm-dense-kl-n5kj-ens400-r2v12`, 5 best + 5 random of its top-60 inside a length window of 0.5-2.0 of the positive; the pool of a unit is the positives of its own rows (about 25000 passages; dose3: 19 008 passages of its own rows); ensemble teacher soup-v9b + PolDense-400M (bf16) with the fixed dense scale 84.19441654567078, computed by `bridge/splade_mine_and_score.py` in one pass over the closed pool (the tool of dose3; the same mean of the two teachers as `scripts/reranker_rows.py ensemble`).

| rows | n | teacher P(pos) at T = 2, median | share above 0.95 | rows with a negative above the positive | negative / positive length, median | p90 | query chars, median | positive chars, median | student ranks the positive first |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dose3 rows (60 385, the rows of R-a) | 60385 | 0.855 | 0.298 | 6826 (11.3 %) | 1.02 | 1.66 | 63 | 734 | 0.867 |
| MS MARCO-pl, stage 1 (units 1-8) | 200000 | 0.963 | 0.555 | 11513 (5.8 %) | 0.99 | 1.66 | 33 | 327 | 0.868 |
| MS MARCO-pl, the other units (for the 5070) | 111080 | 0.964 | 0.563 | 6206 (5.6 %) | 0.99 | 1.66 | 33 | 327 | 0.871 |

## Dose curve of stage 1 (MS MARCO-pl rows alone)

Every segment is one epoch over its own 50 000 rows from the model of the segment before, with its own warm-up (10 %) and linear decay to zero (lr 2e-5), so each point is an annealed model and the points are comparable; in one run with a single schedule the intermediate points would be read at a high learning rate. tiny10 of the first form within both limits.

| examples | model | measured form | tiny10 | step | dims/doc | dims/query | postings/query | dims/doc without a shift | training time |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `kd-rbase-v2-dqf-full` | `kd-rbase-v2-dqf-full-bias015` | 64.69 | - | 372.7 | 120.9 | 14064 | 552.8 | KD alone |

## Result

| model | rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | dose3 (v56) | 64.96 | 43.63 | 72.82 | 374.7 | 381.5 | 121.9 | 14079 |
| soup-v20-dqf-bias015 | frozen full model | 65.06 | 43.64 | 72.48 | 371.4 | 378.5 | 123.0 | 13945 |
| **arm-b1-ra-bias015** | R-a | 65.34 | 43.76 | 67.39 | 361.8 | 367.4 | 130.8 | 15462 |

## Reading

arm-b1-ra-bias015 65.34 / 43.76 at 361.8 dims/doc against dose3 (v56) arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 64.96 / 43.63 at 374.7: delta +0.38 / +0.12, trec-covid 67.39 against 72.82 -> arm-b1-ra-bias015 is within the seed noise of dose3 set by set.

Seed noise of this recipe is 0.13-0.2 (one seed here).

## Task groups (mean delta against dose3 (v56); tasks both models have)

| group | tasks | frozen full model | R-a |
|---|---:|---:|---:|
| FAQ tasks (Allegro FAQ A, B, mfaq) | 3 | +0.38 | +0.67 |
| tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions) | 4 | -0.15 | +0.27 |
| scifact-pl | 1 | +0.02 | +0.34 |
| maupqa-mkqa | 1 | +0.11 | +0.05 |
| MAUPQA tasks | 8 | +0.09 | +0.15 |
| all tasks | 18 | +0.06 | +0.27 |

### tiny10 per task

| task | dose3 (v56) | frozen full model | R-a | R-a - dose3 (v56) |
|---|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.88 | 18.89 | 18.84 | -0.04 |
| maupqa-mkqa | 65.60 | 65.71 | 65.66 | +0.05 |
| maupqa-poleval2021-pairs | 83.95 | 84.07 | 84.33 | +0.39 |
| maupqa-templates | 76.57 | 76.72 | 76.73 | +0.16 |
| poleval-2022-test-A-allegro-faq | 78.80 | 78.93 | 79.40 | +0.60 |
| poleval-2022-test-B-allegro-faq | 74.77 | 75.48 | 75.82 | +1.05 |
| arguana-pl | 50.24 | 50.09 | 51.20 | +0.97 |
| nfcorpus-pl | 32.84 | 32.76 | 33.09 | +0.25 |
| scifact-pl | 68.84 | 68.86 | 69.19 | +0.34 |
| gpt-exams | 99.10 | 99.12 | 99.13 | +0.03 |

### heldout8 per task

| task | dose3 (v56) | frozen full model | R-a | R-a - dose3 (v56) |
|---|---:|---:|---:|---:|
| maupqa-gpt3.5-cc | 20.54 | 20.57 | 20.49 | -0.06 |
| maupqa-multilingual-NLI | 14.89 | 14.88 | 14.77 | -0.12 |
| mfaq | 63.56 | 63.85 | 63.92 | +0.36 |
| poleval-2022-test-A-legal-questions | 85.06 | 84.58 | 84.03 | -1.03 |
| maupqa-wiki-def | 67.87 | 68.05 | 68.44 | +0.56 |
| scidocs-pl | 14.81 | 14.73 | 15.02 | +0.21 |
| maupqa-1z10 | 49.46 | 49.60 | 49.72 | +0.26 |
| fiqa-pl | 32.88 | 32.87 | 33.68 | +0.80 |

## 23 tasks

23-task mean = (10 x tiny10 + 8 x heldout8 + the five held-out-2 tasks) / 23.

| model | tiny10 | heldout8 | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 | 23 tasks | dims/doc | cost: dims/query | cost: postings/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | 64.96 | 43.63 | 87.66 | 92.73 | 79.52 | 19.43 | 72.82 | 70.43 | 58.73 | 374.7 | 121.9 | 14079 |
| soup-v20-dqf-bias015 | 65.06 | 43.64 | 87.76 | 92.95 | 79.84 | 19.57 | 72.48 | 70.52 | 58.80 | 371.4 | 123.0 | 13945 |

## Per task, with the ALBERT

| task | dose3 | frozen full model | R-a | ALBERT (50 M) |
|---|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.88 | 18.89 | 18.84 | 18.62 |
| maupqa-mkqa | 65.60 | 65.71 | 65.66 | 64.75 |
| maupqa-poleval2021-pairs | 83.95 | 84.07 | 84.33 | 82.96 |
| maupqa-templates | 76.57 | 76.72 | 76.73 | 75.87 |
| poleval-2022-test-A-allegro-faq | 78.80 | 78.93 | 79.40 | 77.70 |
| poleval-2022-test-B-allegro-faq | 74.77 | 75.48 | 75.82 | 74.29 |
| arguana-pl | 50.24 | 50.09 | 51.20 | 49.15 |
| nfcorpus-pl | 32.84 | 32.76 | 33.09 | 32.78 |
| scifact-pl | 68.84 | 68.86 | 69.19 | 68.28 |
| gpt-exams | 99.10 | 99.12 | 99.13 | 99.01 |
| **tiny10 mean** | **64.96** | **65.06** | **65.34** | **64.34** |
| maupqa-gpt3.5-cc | 20.54 | 20.57 | 20.49 | 20.32 |
| maupqa-multilingual-NLI | 14.89 | 14.88 | 14.77 | 14.55 |
| mfaq | 63.56 | 63.85 | 63.92 | 63.41 |
| poleval-2022-test-A-legal-questions | 85.06 | 84.58 | 84.03 | 82.74 |
| maupqa-wiki-def | 67.87 | 68.05 | 68.44 | 66.85 |
| scidocs-pl | 14.81 | 14.73 | 15.02 | 14.84 |
| maupqa-1z10 | 49.46 | 49.60 | 49.72 | 49.07 |
| fiqa-pl | 32.88 | 32.87 | 33.68 | 31.36 |
| **heldout8 mean** | **43.63** | **43.64** | **43.76** | **42.89** |
| trec-covid-pl | 72.82 | 72.48 | 67.39 | 64.80 |

## Queue state (b1, b1c)

```
b1:tokenizer	ok	-	-
fidelity:student-rbase-v2.corpus	ok	-	16s
fidelity:student-rbase-v2.queries	ok	-	13s
b1:tokenizer	ok	-	-
kd:rbase-v2-dqf-full	ok	0.9758	25986s
fidelity:kd-rbase-v2-dqf-full.queries	ok	-	15s
sparsity:kd-rbase-v2-dqf-full	ok	552.8	28s
sparsity:kd-rbase-v2-dqf-full-bias005	ok	485.6	28s
sparsity:kd-rbase-v2-dqf-full-bias010	ok	425.7	28s
sparsity:kd-rbase-v2-dqf-full-bias015	ok	372.7	28s
regime:quantized:kd-rbase-v2-dqf-full-bias015:tiny10	ok	64.69 10	724s
train:arm-b1-ra	ok	-	29012s
sparsity:arm-b1-ra	ok	592.7	27s
sparsity:arm-b1-ra-bias005	ok	503.7	27s
sparsity:arm-b1-ra-bias010	ok	427.2	27s
sparsity:arm-b1-ra-bias015	ok	361.8	27s
regime:quantized:arm-b1-ra-bias015:tiny10	ok	65.34 10	699s
regime:quantized:arm-b1-ra-bias015:heldout8	ok	43.76 8	2544s
regime:quantized:arm-b1-ra-bias015:heldout2_treccovid	ok	67.39 1	1585s
chain:rc_smoke	ok	-	16s
chain:rc_unit-01	ok	-	256s
chain:rc_unit-02	ok	-	257s
chain:rc_unit-03	ok	-	256s
chain:rc_unit-04	ok	-	258s
chain:rc_unit-05	ok	-	256s
chain:rc_unit-06	ok	-	255s
chain:rc_unit-07	ok	-	257s
chain:rc_unit-08	ok	-	257s
chain:rc_unit-09	ok	-	257s
chain:rc_unit-10	ok	-	258s
chain:rc_unit-11	ok	-	257s
chain:rc_unit-12	ok	-	256s
chain:rc_unit-13	ok	-	117s
```
