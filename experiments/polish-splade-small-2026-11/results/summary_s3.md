# s3 (07.11.2026 06:07, leaderboard regime, nDCG@10): full KD of b4-l0125-f1536-hl-r256-w on kd_v1, teacher soup-v9b

Student hl-r256-w = layers 0,1,2,5 of soup-v9b with FFN 1536 as an ALBERT with the teacher's head transform as an extra layer and a rank-256 vocabulary / decoder weighted by kd_v1 token counts (49.8 M params, 0.45 of the teacher's MACs per token). KD: the s2 recipe of b4-full (token KL x0.3, T=2, top-128 + pooled MSE) on all 2264676 texts of datasets/kd/kd_v1.jsonl, one epoch, batch 16, lr 5e-5 with linear decay, seed 42; then stage 3 = ranking KL on the n4 rows (v32 recipe, fast head). References: hl s3 = the same student after 200 k texts (s1), b4-full-s3 = the wide student after the same full KD (s2).

### tiny10

| task | base-splade-wrapped | arm-b4-l0125-f1536-hl-r256-w-s3 | arm-b4-l0125-f1536-hl-r256-w-full-s3 | arm-b4-l0125-f3072-joint-s3 | arm-b4-l0125-f3072-full-s3 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 40.86 | 42.84 | 43.82 | 44.57 | 46.28 |
| gpt-exams | 98.63 | 97.83 | 98.48 | 98.43 | 98.56 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.26 | 18.48 | 18.61 | 18.64 | 18.53 |
| maupqa-mkqa | 63.92 | 62.93 | 64.91 | 64.77 | 65.31 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 81.40 | 82.78 | 82.95 | 83.25 | 83.32 |
| maupqa-templates | 75.02 | 74.40 | 75.56 | 75.46 | 75.86 | 75.80 |
| nfcorpus-pl | 30.07 | 31.19 | 32.28 | 32.56 | 32.45 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 71.90 | 74.98 | 75.73 | 77.26 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 69.37 | 72.54 | 73.92 | 74.30 | 73.48 |
| scifact-pl | 64.52 | 64.64 | 66.87 | 66.63 | 67.46 | 66.44 |
| **mean** | **61.60** | **61.28** | **62.97** | **63.29** | **63.77** | **63.65** |

### heldout8

| task | arm-b4-l0125-f1536-hl-r256-w-full-s3 | arm-b4-l0125-f3072-full-s3 | soup-v9b |
|---|---:|---:|---:|
| fiqa-pl | 30.06 | 31.00 | 30.37 |
| maupqa-1z10 | 48.72 | 49.00 | 48.49 |
| maupqa-gpt3.5-cc | 20.18 | 20.28 | 20.26 |
| maupqa-multilingual-NLI | 14.60 | 14.57 | 14.53 |
| maupqa-wiki-def | 65.86 | 66.19 | 66.35 |
| mfaq | 62.97 | 63.19 | 63.18 |
| poleval-2022-test-A-legal-questions | 80.36 | 81.59 | 82.72 |
| scidocs-pl | 14.08 | 14.49 | 14.55 |
| **mean** | **42.10** | **42.54** | **42.56** |


## Active document dimensions (base 436)

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 |  |  |  |  |  |
| arm-b4-l0125-f1536-hl-r256-w-s3 | 593.2 | ×1.36 | 875 |  |  |  |  |  |
| arm-b4-l0125-f1536-hl-r256-w-full-s3 | 512.0 | ×1.17 | 750 |  |  |  |  |  |
| arm-b4-l0125-f3072-full-s3 | 426.2 | ×0.98 | 617 |  |  |  |  |  |

## Index cost (one PIRB sample: 1 350 passages + 1 350 queries; doc dims, query dims, postings/query)

```
model                 dims/doc  dims/query  postings/q   flops
soup-v9b                 417.6       151.4       17805   3.008
b4-full-s3               436.6       156.3       20728   3.312
hl-r256-w-s3             605.7       200.7       38411   4.631
hl-full-s3               524.6       183.3       27501   3.733
```

## Fidelity curve of the full KD (256 corpus texts: cosine / L1 relative error / student dims, per kind)

```
  step      0: all: cos 0.455 l1 2.080 dims 599 | passage: cos 0.446 l1 2.102 dims 669 | query: cos 0.502 l1 1.960 dims 236 (teacher dims 383)
  step   3000: all: cos 0.859 l1 0.572 dims 239 | passage: cos 0.855 l1 0.577 dims 268 | query: cos 0.880 l1 0.550 dims 85 (teacher dims 383)
  step   6000: all: cos 0.882 l1 0.554 dims 319 | passage: cos 0.879 l1 0.558 dims 358 | query: cos 0.897 l1 0.537 dims 114 (teacher dims 383)
  step   9000: all: cos 0.891 l1 0.509 dims 284 | passage: cos 0.888 l1 0.517 dims 320 | query: cos 0.908 l1 0.469 dims 94 (teacher dims 383)
  step  12000: all: cos 0.904 l1 0.471 dims 278 | passage: cos 0.901 l1 0.477 dims 313 | query: cos 0.919 l1 0.437 dims 96 (teacher dims 383)
  step  15000: all: cos 0.903 l1 0.509 dims 355 | passage: cos 0.901 l1 0.515 dims 398 | query: cos 0.918 l1 0.476 dims 132 (teacher dims 383)
  step  18000: all: cos 0.912 l1 0.464 dims 318 | passage: cos 0.909 l1 0.472 dims 356 | query: cos 0.929 l1 0.427 dims 117 (teacher dims 383)
  step  21000: all: cos 0.913 l1 0.493 dims 378 | passage: cos 0.910 l1 0.506 dims 426 | query: cos 0.930 l1 0.427 dims 123 (teacher dims 383)
  step  24000: all: cos 0.918 l1 0.443 dims 313 | passage: cos 0.916 l1 0.450 dims 352 | query: cos 0.933 l1 0.403 dims 107 (teacher dims 383)
  step  27000: all: cos 0.920 l1 0.440 dims 321 | passage: cos 0.917 l1 0.448 dims 361 | query: cos 0.935 l1 0.398 dims 110 (teacher dims 383)
  step  30000: all: cos 0.924 l1 0.421 dims 306 | passage: cos 0.921 l1 0.428 dims 344 | query: cos 0.937 l1 0.382 dims 105 (teacher dims 383)
  step  33000: all: cos 0.924 l1 0.428 dims 325 | passage: cos 0.922 l1 0.435 dims 365 | query: cos 0.936 l1 0.396 dims 115 (teacher dims 383)
  step  36000: all: cos 0.927 l1 0.414 dims 314 | passage: cos 0.925 l1 0.422 dims 354 | query: cos 0.940 l1 0.375 dims 105 (teacher dims 383)
  step  39000: all: cos 0.929 l1 0.404 dims 309 | passage: cos 0.927 l1 0.412 dims 348 | query: cos 0.941 l1 0.364 dims 105 (teacher dims 383)
  step  42000: all: cos 0.931 l1 0.416 dims 346 | passage: cos 0.929 l1 0.424 dims 389 | query: cos 0.943 l1 0.377 dims 121 (teacher dims 383)
  step  45000: all: cos 0.932 l1 0.394 dims 296 | passage: cos 0.930 l1 0.401 dims 333 | query: cos 0.943 l1 0.357 dims 100 (teacher dims 383)
  step  48000: all: cos 0.933 l1 0.397 dims 320 | passage: cos 0.931 l1 0.405 dims 359 | query: cos 0.946 l1 0.357 dims 111 (teacher dims 383)
  step  51000: all: cos 0.935 l1 0.388 dims 311 | passage: cos 0.932 l1 0.396 dims 349 | query: cos 0.948 l1 0.347 dims 109 (teacher dims 383)
  step  54000: all: cos 0.933 l1 0.398 dims 324 | passage: cos 0.931 l1 0.407 dims 365 | query: cos 0.946 l1 0.352 dims 106 (teacher dims 383)
  step  57000: all: cos 0.936 l1 0.387 dims 320 | passage: cos 0.933 l1 0.396 dims 360 | query: cos 0.950 l1 0.340 dims 110 (teacher dims 383)
  step  60000: all: cos 0.938 l1 0.391 dims 344 | passage: cos 0.935 l1 0.401 dims 387 | query: cos 0.951 l1 0.341 dims 119 (teacher dims 383)
  step  63000: all: cos 0.938 l1 0.395 dims 354 | passage: cos 0.935 l1 0.404 dims 398 | query: cos 0.951 l1 0.346 dims 121 (teacher dims 383)
  step  66000: all: cos 0.939 l1 0.386 dims 338 | passage: cos 0.936 l1 0.393 dims 379 | query: cos 0.952 l1 0.349 dims 123 (teacher dims 383)
  step  69000: all: cos 0.938 l1 0.392 dims 344 | passage: cos 0.935 l1 0.402 dims 387 | query: cos 0.951 l1 0.344 dims 121 (teacher dims 383)
  step  72000: all: cos 0.940 l1 0.377 dims 329 | passage: cos 0.938 l1 0.385 dims 371 | query: cos 0.952 l1 0.334 dims 111 (teacher dims 383)
  step  75000: all: cos 0.940 l1 0.376 dims 328 | passage: cos 0.938 l1 0.385 dims 369 | query: cos 0.953 l1 0.333 dims 115 (teacher dims 383)
  step  78000: all: cos 0.941 l1 0.379 dims 340 | passage: cos 0.938 l1 0.387 dims 382 | query: cos 0.954 l1 0.333 dims 120 (teacher dims 383)
  step  81000: all: cos 0.942 l1 0.375 dims 344 | passage: cos 0.940 l1 0.384 dims 387 | query: cos 0.955 l1 0.327 dims 117 (teacher dims 383)
  step  84000: all: cos 0.943 l1 0.372 dims 341 | passage: cos 0.940 l1 0.381 dims 384 | query: cos 0.955 l1 0.324 dims 116 (teacher dims 383)
  step  87000: all: cos 0.943 l1 0.370 dims 334 | passage: cos 0.940 l1 0.379 dims 377 | query: cos 0.954 l1 0.325 dims 113 (teacher dims 383)
  step  90000: all: cos 0.943 l1 0.366 dims 335 | passage: cos 0.941 l1 0.375 dims 377 | query: cos 0.956 l1 0.324 dims 117 (teacher dims 383)
  step  93000: all: cos 0.944 l1 0.368 dims 343 | passage: cos 0.942 l1 0.376 dims 386 | query: cos 0.956 l1 0.325 dims 120 (teacher dims 383)
  step  96000: all: cos 0.944 l1 0.365 dims 333 | passage: cos 0.942 l1 0.374 dims 375 | query: cos 0.957 l1 0.318 dims 112 (teacher dims 383)
  step  99000: all: cos 0.945 l1 0.359 dims 327 | passage: cos 0.942 l1 0.368 dims 368 | query: cos 0.957 l1 0.313 dims 110 (teacher dims 383)
  step 102000: all: cos 0.945 l1 0.365 dims 348 | passage: cos 0.943 l1 0.373 dims 391 | query: cos 0.957 l1 0.322 dims 122 (teacher dims 383)
  step 105000: all: cos 0.946 l1 0.360 dims 340 | passage: cos 0.943 l1 0.369 dims 382 | query: cos 0.959 l1 0.313 dims 117 (teacher dims 383)
  step 108000: all: cos 0.946 l1 0.353 dims 321 | passage: cos 0.944 l1 0.361 dims 361 | query: cos 0.959 l1 0.308 dims 113 (teacher dims 383)
  step 111000: all: cos 0.946 l1 0.356 dims 336 | passage: cos 0.944 l1 0.365 dims 378 | query: cos 0.959 l1 0.310 dims 117 (teacher dims 383)
  step 114000: all: cos 0.947 l1 0.357 dims 342 | passage: cos 0.944 l1 0.366 dims 384 | query: cos 0.960 l1 0.312 dims 121 (teacher dims 383)
  step 117000: all: cos 0.947 l1 0.356 dims 345 | passage: cos 0.945 l1 0.365 dims 388 | query: cos 0.960 l1 0.305 dims 116 (teacher dims 383)
  step 120000: all: cos 0.948 l1 0.346 dims 325 | passage: cos 0.946 l1 0.356 dims 366 | query: cos 0.961 l1 0.298 dims 113 (teacher dims 383)
  step 123000: all: cos 0.948 l1 0.350 dims 335 | passage: cos 0.946 l1 0.358 dims 376 | query: cos 0.960 l1 0.305 dims 117 (teacher dims 383)
  step 126000: all: cos 0.949 l1 0.348 dims 336 | passage: cos 0.946 l1 0.357 dims 378 | query: cos 0.961 l1 0.300 dims 116 (teacher dims 383)
  step 129000: all: cos 0.949 l1 0.351 dims 343 | passage: cos 0.946 l1 0.359 dims 385 | query: cos 0.961 l1 0.306 dims 121 (teacher dims 383)
  step 132000: all: cos 0.949 l1 0.346 dims 335 | passage: cos 0.946 l1 0.355 dims 376 | query: cos 0.962 l1 0.299 dims 116 (teacher dims 383)
  step 135000: all: cos 0.949 l1 0.345 dims 335 | passage: cos 0.947 l1 0.354 dims 377 | query: cos 0.962 l1 0.298 dims 117 (teacher dims 383)
  step 138000: all: cos 0.949 l1 0.344 dims 333 | passage: cos 0.947 l1 0.353 dims 375 | query: cos 0.962 l1 0.297 dims 116 (teacher dims 383)
  step 141000: all: cos 0.950 l1 0.344 dims 336 | passage: cos 0.947 l1 0.353 dims 378 | query: cos 0.962 l1 0.297 dims 117 (teacher dims 383)
  step 141543: all: cos 0.950 l1 0.344 dims 336 | passage: cos 0.947 l1 0.353 dims 378 | query: cos 0.962 l1 0.297 dims 117 (teacher dims 383)
```

## Ranking agreement with the teacher scores (500 rows of mix3v2-6219-tv9b.jsonl) and fidelity on queries (512 corpus queries)

```
kd-b4-l0125-f1536-hl-r256-w-s2               tau 0.732  top1 0.904  pos-first 0.906  margin 8.88  | queries: cos 0.917 l1 0.471 dims 106/108
arm-b4-l0125-f1536-hl-r256-w-s3              tau 0.741  top1 0.956  pos-first 0.940  margin 11.67  | queries: cos 0.905 l1 0.576 dims 143/108
kd-b4-l0125-f1536-hl-r256-w-full             tau 0.816  top1 0.952  pos-first 0.948  margin 10.22  | queries: cos 0.962 l1 0.297 dims 105/108
arm-b4-l0125-f1536-hl-r256-w-full-s3         tau 0.806  top1 0.970  pos-first 0.956  margin 11.83  | queries: cos 0.950 l1 0.377 dims 128/108
kd-b4-l0125-f3072-full                       tau 0.866  top1 0.974  pos-first 0.956  margin 11.37  | queries: cos 0.980 l1 0.215 dims 106/108
arm-b4-l0125-f3072-full-s3                   tau 0.860  top1 0.978  pos-first 0.958  margin 12.22  | queries: cos 0.976 l1 0.241 dims 107/108
```

## Queue state (s3)

```
kd:b4-l0125-f1536-hl-r256-w-full	ok	0.9496	13973s
fidelity:kd-b4-l0125-f1536-hl-r256-w-full.queries	ok	-	12s
train:arm-b4-l0125-f1536-hl-r256-w-full-s3	ok	-	1088s
sparsity:arm-b4-l0125-f1536-hl-r256-w-full-s3	ok	512.0	20s
regime:quantized:arm-b4-l0125-f1536-hl-r256-w-full-s3:tiny10	ok	62.97 10	567s
regime:quantized:arm-b4-l0125-f1536-hl-r256-w-full-s3:heldout8	ok	42.1 8	2142s
fidelity:arm-b4-l0125-f1536-hl-r256-w-full-s3.queries	ok	-	13s
```
