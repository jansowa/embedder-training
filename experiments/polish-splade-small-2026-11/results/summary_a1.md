# a1 (quantized regime, nDCG@10): the ALBERT student descending from the frozen full model

The old lineage: student `b4-l0125-f1536-hl-r256-w` cut from soup-v9b -> KD from soup-v9b on kd_v1 -> stage 3 on n5 + kj (`arm-hl-full-s3n5kj`) -> round 2 in r1. The new lineage repeats it with one intended change: the source model and the KD teacher are the raw weights of the frozen full model (`models/soup-v20-dqf`; the frozen form `soup-v20-dqf-bias015` is the same model with the bias shift). Student: layers 0,1,2,5, FFN 1536, the teacher's head transform as an extra layer, rank-256 vocabulary weighted by the kd_v1 token counts. KD: token KL x 0.3 + pooled MSE on all of kd_v1, one epoch, lr 5e-5, linear, batch 16, seed 42. Stage 3 and round 2: the configs of the old lineage with the start model replaced; the ranking teacher is unchanged (the ensemble scores stored in the rows). Round 2 ran on the rows of the r1 arm that won tiny10: **dose3** (r1: dose3 64.14, dose3 + faq20 64.09).

One unintended difference: the old student was built before `distill.student` wrote the `head_layer` marker into config.json, so its KD ran the extra layer and the head transform on the stock fp16 path; the new student has the marker, so its KD ran them in fp32 with that layer's dropout off (`distill.splade.prepare_head_layer`). Evaluation (PIRB) uses the stock fp16 path for both.

## Forms

Every model of the new lineage is measured in its first bias form (steps of 0.05) with at most 410 dims/doc **and** at most 16807 postings/query. Queue a1 looked at dims/doc only and measured `arm-hl-dqf-full-s3n5kj-bias010` and `arm-hl-dqf-full-s3n5kj-r2d3-bias015`; both are above the postings limit (17573 and 17487), so a1 was stopped by hand on 14.11 at 19:10 and queue a1b took the next forms: stage 3 `arm-hl-dqf-full-s3n5kj-bias015`, round 2 `arm-hl-dqf-full-s3n5kj-r2d3-bias020`. The heldout8 of `arm-hl-dqf-full-s3n5kj-r2d3-bias015` is missing: its run lost 4 of 8 tasks to a CUDA out-of-memory error caused by a cost measurement started by hand next to the queue, and the partial file was moved out of the results.

## Choice

Rule: the highest mean of tiny10 and heldout8 within the two limits, no margin; the best arm of r1 is the reference, both models of the new lineage and the stage-3 model of the old lineage are candidates.

**ALBERT to release: `arm-hl-dqf-full-s3n5kj-r2d3-bias020`** — new lineage, round 2 on dose3 (a1) leads old lineage, round 2 on dose3 (r1) by 0.240 on the mean of tiny10 and heldout8 (any lead wins: margin 0) within the limits. The label is the first line of `logs/final_albert_model.txt`; the model is in `datasets/transfer/release-20261113/albert/arm-hl-dqf-full-s3n5kj-r2d3-bias020`.

## Old lineage against the new one

asym = the model as the query encoder over the index of the frozen full model (`soup-v20-dqf-bias015`, symmetric 65.06 / 43.64).

| model | label | tiny10 | heldout8 | trec-covid | asym tiny10 | asym heldout8 | dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old lineage, stage 3 on n5 + kj (v45); above the postings limit | `arm-hl-full-s3n5kj-bias010` | 63.77 | 42.33 | 64.23 | - | - | 374.6 | 146.1 | 17437 |
| old lineage, round 2 on dose3 (r1) | `arm-hl-full-s3n5kj-r2d3-bias020` | 64.14 | 42.61 | 61.35 | 64.75 | 43.2 | 359.6 | 124.3 | 13397 |
| old lineage, round 2, the other arm of r1 | `arm-hl-full-s3n5kj-r2d3faq-bias020` | 64.09 | - | - | - | - | 370.4 | 133.8 | 14790 |
| new lineage, stage 3, the form of a1 (above the postings limit) | `arm-hl-dqf-full-s3n5kj-bias010` | 64.32 | - | - | - | - | 378.4 | 152.0 | 17573 |
| new lineage, stage 3 on n5 + kj (a1) | `arm-hl-dqf-full-s3n5kj-bias015` | 64.29 | 42.76 | 63.39 | 64.73 | 43.26 | 329.4 | 133.8 | 13635 |
| new lineage, round 2, the form of a1 (above the postings limit) | `arm-hl-dqf-full-s3n5kj-r2d3-bias015` | 64.36 | - | 65.14 | - | - | 405.6 | 141.3 | 17487 |
| **new lineage, round 2 on dose3 (a1)** | `arm-hl-dqf-full-s3n5kj-r2d3-bias020` | 64.34 | 42.89 | 64.8 | 64.9 | 43.33 | 339.8 | 120.7 | 12725 |
| frozen full model (82 M), for scale | `soup-v20-dqf-bias015` | 65.06 | 43.64 | 72.48 | - | - | 371.4 | 123.0 | 13945 |

Unbiased dims/doc: old stage 3 487.5, old round 2 759.5, new stage 3 498.5, new round 2 697.1. One seed each; the seed noise of the ranking recipe is 0.13-0.2.

The change of the source model and KD teacher alone, in the forms of a1 (stage 3 against stage 3, tiny10; both forms are above the postings limit): 0.55. Within the limits, against the best arm of r1: new stage 3 tiny10 0.15, heldout8 0.15; new round 2 tiny10 0.2, heldout8 0.28.

After KD alone, before any ranking stage (no bias shift): new `kd-b4-l0125-f1536-hl-r256-w-dqf-full` tiny10 64.03 at 554.8 dims/doc, old `kd-b4-l0125-f1536-hl-r256-w-full` tiny10 63.02 at 418.7 dims/doc; after stage 3 without a bias shift the old lineage has 63.78 at 487.5.

Training time of the new lineage: KD 14 359 s (3 h 59 min), stage 3 3 175 s, round 2 9 305 s.

## Fidelity to the own teacher (old: soup-v9b; new: soup-v20-dqf)

Stock path = the whole model in fp16, as inference runs it; fp32 head = the extra layer's feed-forward and the head transform in fp32, as the KD of the new student ran them. The old student has no `head_layer` marker, so its files named fp32head are the stock path as well.

| student | texts | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |
|---|---:|---:|---:|---:|---:|---:|---:|
| old, before KD, corpus texts (stock path) | 256 | 0.4547 | 0.359 | 0.145 | 2.079 | 341 | 599 |
| new, before KD, corpus texts, stock path | 256 | 0.4950 | 0.298 | 0.163 | 1.461 | 428 | 433 |
| new, before KD, corpus texts, fp32 head | 256 | 0.4951 | 0.298 | 0.163 | 1.460 | 428 | 432 |
| old, before KD, queries (stock path) | 512 | 0.5129 | 0.373 | 0.142 | 1.873 | 108 | 220 |
| new, before KD, queries, stock path | 512 | 0.5481 | 0.352 | 0.155 | 1.510 | 108 | 169 |
| new, before KD, queries, fp32 head | 512 | 0.5482 | 0.352 | 0.155 | 1.509 | 108 | 168 |
| old, after KD, queries (stock path) | 512 | 0.9623 | 0.775 | 0.656 | 0.297 | 108 | 105 |
| new, after KD, queries, stock path | 512 | 0.9722 | 0.804 | 0.691 | 0.258 | 108 | 105 |
| new, after KD, queries, fp32 head | 512 | 0.9722 | 0.804 | 0.691 | 0.258 | 108 | 105 |

Fidelity curve of the new KD (256 corpus texts per checkpoint):

```
  step      0: all: cos 0.495 l1 1.460 dims 433 | passage: cos 0.486 l1 1.443 dims 480 | query: cos 0.542 l1 1.549 dims 184 (teacher dims 487)
  step   3000: all: cos 0.886 l1 0.554 dims 370 | passage: cos 0.883 l1 0.556 dims 418 | query: cos 0.903 l1 0.541 dims 119 (teacher dims 487)
  step   6000: all: cos 0.906 l1 0.514 dims 418 | passage: cos 0.903 l1 0.523 dims 476 | query: cos 0.921 l1 0.468 dims 114 (teacher dims 487)
  step   9000: all: cos 0.918 l1 0.486 dims 440 | passage: cos 0.916 l1 0.490 dims 499 | query: cos 0.927 l1 0.462 dims 131 (teacher dims 487)
  step  12000: all: cos 0.927 l1 0.420 dims 363 | passage: cos 0.925 l1 0.428 dims 413 | query: cos 0.941 l1 0.379 dims 98 (teacher dims 487)
  step  15000: all: cos 0.930 l1 0.423 dims 407 | passage: cos 0.928 l1 0.431 dims 463 | query: cos 0.944 l1 0.379 dims 110 (teacher dims 487)
  step  18000: all: cos 0.935 l1 0.398 dims 381 | passage: cos 0.933 l1 0.405 dims 434 | query: cos 0.948 l1 0.359 dims 106 (teacher dims 487)
  step  21000: all: cos 0.934 l1 0.438 dims 469 | passage: cos 0.931 l1 0.451 dims 536 | query: cos 0.948 l1 0.371 dims 119 (teacher dims 487)
  step  24000: all: cos 0.937 l1 0.411 dims 438 | passage: cos 0.935 l1 0.420 dims 500 | query: cos 0.950 l1 0.360 dims 114 (teacher dims 487)
  step  27000: all: cos 0.938 l1 0.406 dims 438 | passage: cos 0.936 l1 0.416 dims 500 | query: cos 0.951 l1 0.355 dims 115 (teacher dims 487)
  step  30000: all: cos 0.944 l1 0.367 dims 376 | passage: cos 0.942 l1 0.374 dims 428 | query: cos 0.955 l1 0.328 dims 101 (teacher dims 487)
  step  33000: all: cos 0.945 l1 0.367 dims 397 | passage: cos 0.943 l1 0.375 dims 452 | query: cos 0.955 l1 0.329 dims 108 (teacher dims 487)
  step  36000: all: cos 0.946 l1 0.371 dims 421 | passage: cos 0.944 l1 0.378 dims 480 | query: cos 0.955 l1 0.332 dims 114 (teacher dims 487)
  step  39000: all: cos 0.948 l1 0.367 dims 426 | passage: cos 0.946 l1 0.374 dims 485 | query: cos 0.958 l1 0.327 dims 114 (teacher dims 487)
  step  42000: all: cos 0.948 l1 0.368 dims 442 | passage: cos 0.947 l1 0.375 dims 504 | query: cos 0.957 l1 0.331 dims 118 (teacher dims 487)
  step  45000: all: cos 0.950 l1 0.353 dims 414 | passage: cos 0.948 l1 0.362 dims 471 | query: cos 0.960 l1 0.309 dims 110 (teacher dims 487)
  step  48000: all: cos 0.950 l1 0.351 dims 411 | passage: cos 0.948 l1 0.359 dims 468 | query: cos 0.960 l1 0.310 dims 113 (teacher dims 487)
  step  51000: all: cos 0.951 l1 0.346 dims 411 | passage: cos 0.949 l1 0.354 dims 468 | query: cos 0.961 l1 0.305 dims 110 (teacher dims 487)
  step  54000: all: cos 0.952 l1 0.343 dims 411 | passage: cos 0.950 l1 0.351 dims 469 | query: cos 0.961 l1 0.301 dims 107 (teacher dims 487)
  step  57000: all: cos 0.952 l1 0.341 dims 404 | passage: cos 0.950 l1 0.349 dims 461 | query: cos 0.962 l1 0.299 dims 106 (teacher dims 487)
  step  60000: all: cos 0.953 l1 0.341 dims 423 | passage: cos 0.951 l1 0.350 dims 483 | query: cos 0.964 l1 0.297 dims 113 (teacher dims 487)
  step  63000: all: cos 0.955 l1 0.333 dims 422 | passage: cos 0.953 l1 0.342 dims 481 | query: cos 0.967 l1 0.288 dims 114 (teacher dims 487)
  step  66000: all: cos 0.955 l1 0.335 dims 425 | passage: cos 0.953 l1 0.344 dims 483 | query: cos 0.967 l1 0.291 dims 116 (teacher dims 487)
  step  69000: all: cos 0.955 l1 0.331 dims 414 | passage: cos 0.953 l1 0.339 dims 472 | query: cos 0.965 l1 0.287 dims 111 (teacher dims 487)
  step  72000: all: cos 0.956 l1 0.332 dims 431 | passage: cos 0.954 l1 0.342 dims 492 | query: cos 0.968 l1 0.280 dims 112 (teacher dims 487)
  step  75000: all: cos 0.957 l1 0.329 dims 427 | passage: cos 0.954 l1 0.338 dims 487 | query: cos 0.967 l1 0.284 dims 114 (teacher dims 487)
  step  78000: all: cos 0.956 l1 0.327 dims 411 | passage: cos 0.954 l1 0.336 dims 468 | query: cos 0.967 l1 0.281 dims 110 (teacher dims 487)
  step  81000: all: cos 0.958 l1 0.320 dims 413 | passage: cos 0.955 l1 0.329 dims 471 | query: cos 0.969 l1 0.274 dims 110 (teacher dims 487)
  step  84000: all: cos 0.958 l1 0.324 dims 429 | passage: cos 0.956 l1 0.332 dims 488 | query: cos 0.969 l1 0.282 dims 119 (teacher dims 487)
  step  87000: all: cos 0.958 l1 0.325 dims 429 | passage: cos 0.956 l1 0.333 dims 489 | query: cos 0.968 l1 0.281 dims 114 (teacher dims 487)
  step  90000: all: cos 0.958 l1 0.320 dims 419 | passage: cos 0.956 l1 0.327 dims 477 | query: cos 0.968 l1 0.281 dims 114 (teacher dims 487)
  step  93000: all: cos 0.958 l1 0.324 dims 430 | passage: cos 0.956 l1 0.332 dims 490 | query: cos 0.968 l1 0.282 dims 118 (teacher dims 487)
  step  96000: all: cos 0.959 l1 0.316 dims 415 | passage: cos 0.957 l1 0.324 dims 473 | query: cos 0.970 l1 0.273 dims 113 (teacher dims 487)
  step  99000: all: cos 0.960 l1 0.310 dims 409 | passage: cos 0.958 l1 0.319 dims 466 | query: cos 0.970 l1 0.267 dims 112 (teacher dims 487)
  step 102000: all: cos 0.960 l1 0.312 dims 422 | passage: cos 0.958 l1 0.321 dims 481 | query: cos 0.970 l1 0.267 dims 112 (teacher dims 487)
  step 105000: all: cos 0.960 l1 0.309 dims 415 | passage: cos 0.958 l1 0.318 dims 473 | query: cos 0.971 l1 0.264 dims 111 (teacher dims 487)
  step 108000: all: cos 0.961 l1 0.308 dims 417 | passage: cos 0.959 l1 0.317 dims 475 | query: cos 0.971 l1 0.264 dims 112 (teacher dims 487)
  step 111000: all: cos 0.961 l1 0.308 dims 420 | passage: cos 0.959 l1 0.316 dims 478 | query: cos 0.971 l1 0.265 dims 112 (teacher dims 487)
  step 114000: all: cos 0.961 l1 0.310 dims 428 | passage: cos 0.959 l1 0.318 dims 487 | query: cos 0.971 l1 0.269 dims 118 (teacher dims 487)
  step 117000: all: cos 0.961 l1 0.309 dims 432 | passage: cos 0.960 l1 0.318 dims 493 | query: cos 0.972 l1 0.260 dims 112 (teacher dims 487)
  step 120000: all: cos 0.962 l1 0.305 dims 424 | passage: cos 0.960 l1 0.314 dims 483 | query: cos 0.972 l1 0.261 dims 114 (teacher dims 487)
  step 123000: all: cos 0.962 l1 0.305 dims 425 | passage: cos 0.960 l1 0.313 dims 485 | query: cos 0.972 l1 0.261 dims 115 (teacher dims 487)
  step 126000: all: cos 0.962 l1 0.301 dims 418 | passage: cos 0.960 l1 0.310 dims 477 | query: cos 0.973 l1 0.254 dims 112 (teacher dims 487)
  step 129000: all: cos 0.963 l1 0.301 dims 422 | passage: cos 0.961 l1 0.310 dims 481 | query: cos 0.973 l1 0.255 dims 115 (teacher dims 487)
  step 132000: all: cos 0.963 l1 0.301 dims 422 | passage: cos 0.961 l1 0.309 dims 480 | query: cos 0.973 l1 0.255 dims 113 (teacher dims 487)
  step 135000: all: cos 0.963 l1 0.300 dims 422 | passage: cos 0.961 l1 0.309 dims 481 | query: cos 0.973 l1 0.253 dims 113 (teacher dims 487)
  step 138000: all: cos 0.963 l1 0.299 dims 422 | passage: cos 0.961 l1 0.308 dims 480 | query: cos 0.973 l1 0.252 dims 113 (teacher dims 487)
  step 141000: all: cos 0.963 l1 0.298 dims 422 | passage: cos 0.961 l1 0.307 dims 480 | query: cos 0.973 l1 0.252 dims 113 (teacher dims 487)
  step 141543: all: cos 0.963 l1 0.298 dims 422 | passage: cos 0.961 l1 0.307 dims 480 | query: cos 0.973 l1 0.252 dims 113 (teacher dims 487)
```

## Result

| model | rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm-hl-full-s3n5kj-bias010 | old lineage, stage 3 (v45) | 63.77 | 42.33 | 64.23 | 374.6 | 381.5 | 146.1 | 17437 |
| arm-hl-full-s3n5kj-r2d3-bias020 | old lineage, round 2 (r1) | 64.14 | 42.61 | 61.35 | 359.6 | 365.4 | 124.3 | 13397 |
| arm-hl-dqf-full-s3n5kj-bias015 | new lineage, stage 3 (a1) | 64.29 | 42.76 | 63.39 | 329.4 | 336.1 | 133.8 | 13635 |
| **arm-hl-dqf-full-s3n5kj-r2d3-bias020** | new lineage, round 2 (a1) | 64.34 | 42.89 | 64.80 | 339.8 | 346.4 | 120.7 | 12725 |

## Reading

arm-hl-dqf-full-s3n5kj-r2d3-bias020 64.34 / 42.89 at 339.8 dims/doc against old lineage, round 2 (r1) arm-hl-full-s3n5kj-r2d3-bias020 64.14 / 42.61 at 359.6: delta +0.20 / +0.28, trec-covid 64.80 against 61.35 -> within the seed noise of the best arm of r1 set by set (the choice itself follows the rule above, on the mean of the two sets).

Seed noise of this recipe is 0.13-0.2 (one seed here).

## Task groups (mean delta against old lineage, stage 3 (v45); tasks both models have)

| group | tasks | old lineage, round 2 (r1) | new lineage, stage 3 (a1) | new lineage, round 2 (a1) |
|---|---:|---:|---:|---:|
| FAQ tasks (Allegro FAQ A, B, mfaq) | 3 | +0.75 | +0.66 | +0.93 |
| tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions) | 4 | +0.49 | +1.04 | +1.15 |
| MAUPQA tasks | 8 | +0.11 | +0.28 | +0.25 |
| all tasks | 18 | +0.33 | +0.48 | +0.56 |

### tiny10 per task

| task | old lineage, stage 3 (v45) | old lineage, round 2 (r1) | new lineage, stage 3 (a1) | new lineage, round 2 (a1) | new lineage, round 2 (a1) - old lineage, round 2 (r1) |
|---|---:|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.53 | 18.58 | 18.45 | 18.62 | +0.04 |
| maupqa-mkqa | 64.70 | 64.60 | 65.08 | 64.75 | +0.16 |
| maupqa-poleval2021-pairs | 82.68 | 82.99 | 83.07 | 82.96 | -0.03 |
| maupqa-templates | 75.32 | 75.57 | 75.67 | 75.87 | +0.29 |
| poleval-2022-test-A-allegro-faq | 76.43 | 77.85 | 76.78 | 77.70 | -0.15 |
| poleval-2022-test-B-allegro-faq | 73.19 | 73.80 | 74.49 | 74.29 | +0.49 |
| arguana-pl | 47.93 | 48.08 | 50.14 | 49.15 | +1.07 |
| nfcorpus-pl | 32.59 | 33.00 | 32.47 | 32.78 | -0.22 |
| scifact-pl | 67.67 | 68.04 | 67.90 | 68.28 | +0.24 |
| gpt-exams | 98.71 | 98.92 | 98.90 | 99.01 | +0.09 |

### heldout8 per task

| task | old lineage, stage 3 (v45) | old lineage, round 2 (r1) | new lineage, stage 3 (a1) | new lineage, round 2 (a1) | new lineage, round 2 (a1) - old lineage, round 2 (r1) |
|---|---:|---:|---:|---:|---:|
| maupqa-gpt3.5-cc | 20.21 | 20.29 | 20.22 | 20.32 | +0.03 |
| maupqa-multilingual-NLI | 14.43 | 14.46 | 14.50 | 14.55 | +0.09 |
| mfaq | 62.99 | 63.21 | 63.34 | 63.41 | +0.20 |
| poleval-2022-test-A-legal-questions | 81.46 | 82.37 | 82.59 | 82.74 | +0.37 |
| maupqa-wiki-def | 66.50 | 66.54 | 67.38 | 66.85 | +0.31 |
| scidocs-pl | 14.57 | 14.80 | 14.72 | 14.84 | +0.04 |
| maupqa-1z10 | 48.64 | 48.85 | 48.91 | 49.07 | +0.22 |
| fiqa-pl | 29.86 | 30.37 | 30.43 | 31.36 | +0.98 |

## 23 tasks

23-task mean = (10 x tiny10 + 8 x heldout8 + the five held-out-2 tasks) / 23.

| model | tiny10 | heldout8 | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 | 23 tasks | dims/doc | cost: dims/query | cost: postings/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **arm-hl-dqf-full-s3n5kj-r2d3-bias020** | 64.34 | 42.89 | 86.37 | 92.06 | 79.25 | 18.53 | 64.80 | 68.20 | 57.72 | 339.8 | 120.7 | 12725 |

Nothing further was started by this queue except queue x1 for the ONNX export of the chosen ALBERT.

## Queue state (a1, a1b)

```
train:arm-hl-full-s3n5kj-r2d3	ok	-	9686s
train:arm-hl-full-s3n5kj-r2d3faq	ok	-	12299s
sparsity:arm-hl-full-s3n5kj-r2d3	ok	759.5	20s
sparsity:arm-hl-full-s3n5kj-r2d3-bias005	ok	630.8	20s
sparsity:arm-hl-full-s3n5kj-r2d3-bias010	ok	522.7	20s
sparsity:arm-hl-full-s3n5kj-r2d3-bias015	ok	433.2	19s
sparsity:arm-hl-full-s3n5kj-r2d3-bias020	ok	359.6	19s
regime:quantized:arm-hl-full-s3n5kj-r2d3-bias020:tiny10	ok	64.14 10	462s
regime:quantized:arm-hl-full-s3n5kj-r2d3-bias020:heldout8	ok	42.61 8	finished after the stop of r1
sparsity:arm-hl-full-s3n5kj-r2d3faq	ok	803.6	18s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias005	ok	663.3	19s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias010	ok	545.9	19s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias015	ok	449.1	19s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias020	ok	370.4	18s
regime:quantized:arm-hl-full-s3n5kj-r2d3faq-bias020:tiny10	ok	64.09 10	462s
student:b4-l0125-f1536-hl-r256-w-dqf	ok	-	19s
fidelity:student-b4-l0125-f1536-hl-r256-w-dqf.queries	FAILED	-	12s
kd:b4-l0125-f1536-hl-r256-w-dqf-full	ok	0.9632	14359s
fidelity:kd-b4-l0125-f1536-hl-r256-w-dqf-full.queries	ok	-	14s
train:arm-hl-dqf-full-s3n5kj	ok	-	3175s
sparsity:arm-hl-dqf-full-s3n5kj	ok	498.5	19s
sparsity:arm-hl-dqf-full-s3n5kj-bias005	ok	434.5	18s
sparsity:arm-hl-dqf-full-s3n5kj-bias010	ok	378.4	18s
regime:quantized:arm-hl-dqf-full-s3n5kj-bias010:tiny10	ok	64.32 10	472s
train:arm-hl-dqf-full-s3n5kj-r2d3	ok	-	9305s
sparsity:arm-hl-dqf-full-s3n5kj-r2d3	ok	697.1	20s
sparsity:arm-hl-dqf-full-s3n5kj-r2d3-bias005	ok	582.2	19s
sparsity:arm-hl-dqf-full-s3n5kj-r2d3-bias010	ok	485.6	19s
sparsity:arm-hl-dqf-full-s3n5kj-r2d3-bias015	ok	405.6	18s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias015:tiny10	ok	64.36 10	475s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias015:heldout8	ok	45.41 4	1419s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias015:heldout2_treccovid	ok	65.14 1	finished after the stop of a1
sparsity:kd-b4-l0125-f1536-hl-r256-w-dqf-full	ok	554.8	19s
regime:quantized:kd-b4-l0125-f1536-hl-r256-w-dqf-full:tiny10	ok	64.03 10	543s
fidelity:b4-l0125-f1536-hl-r256-w-dqf.cuda-fp16-fp32head	ok	-	-
fidelity:b4-l0125-f1536-hl-r256-w-dqf.cuda-fp16	ok	-	-
fidelity:student-b4-l0125-f1536-hl-r256-w-dqf.queries	ok	-	13s
fidelity:student-b4-l0125-f1536-hl-r256-w-dqf.queries-stock	ok	-	13s
fidelity:kd-b4-l0125-f1536-hl-r256-w-dqf-full.queries-stock	ok	-	12s
sparsity:arm-hl-dqf-full-s3n5kj-r2d3-bias020	ok	339.8	19s
sparsity:arm-hl-dqf-full-s3n5kj-bias015	ok	329.4	19s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias020:tiny10	ok	64.34 10	453s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias020:heldout8	ok	42.89 8	1689s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias020:heldout2_treccovid	ok	64.8 1	942s
regime:quantized:arm-hl-dqf-full-s3n5kj-bias015:tiny10	ok	64.29 10	464s
regime:quantized:arm-hl-dqf-full-s3n5kj-bias015:heldout8	ok	42.76 8	1757s
regime:quantized:arm-hl-dqf-full-s3n5kj-bias015:heldout2_treccovid	ok	63.39 1	941s
regime:quantized:arm-hl-full-s3n5kj-r2d3-bias020:heldout2_treccovid	ok	61.35 1	918s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias020:heldout2_czywiesz	ok	86.37 1	575s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias020:heldout2_gpt35wiki	ok	92.06 1	463s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias020:heldout2_quora	ok	79.25 1	665s
regime:quantized:arm-hl-dqf-full-s3n5kj-r2d3-bias020:heldout2_poquad	ok	18.53 1	919s
regime:quantized:asym-soup-v20-dqf-bias015-q-arm-hl-dqf-full-s3n5kj-r2d3-bias020:tiny10	ok	64.9 10	563s
regime:quantized:asym-soup-v20-dqf-bias015-q-arm-hl-dqf-full-s3n5kj-r2d3-bias020:heldout8	ok	43.33 8	2079s
regime:quantized:asym-soup-v20-dqf-bias015-q-arm-hl-dqf-full-s3n5kj-bias015:tiny10	ok	64.73 10	596s
regime:quantized:asym-soup-v20-dqf-bias015-q-arm-hl-dqf-full-s3n5kj-bias015:heldout8	ok	43.26 8	2169s
regime:quantized:asym-soup-v20-dqf-bias015-q-arm-hl-full-s3n5kj-r2d3-bias020:tiny10	ok	64.75 10	585s
regime:quantized:asym-soup-v20-dqf-bias015-q-arm-hl-full-s3n5kj-r2d3-bias020:heldout8	ok	43.2 8	2099s
```
