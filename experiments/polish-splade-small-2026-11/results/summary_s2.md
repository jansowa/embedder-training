# s2 (06.11.2026 18:38, leaderboard regime, nDCG@10): full KD of b4-l0125-f3072 on kd_v1, teacher soup-v9b

Student b4-f3072 = layers 0,1,2,5 of soup-v9b with the full FFN (67.8 M params, 0.83 of the teacher's MACs per token). KD: the combined loss (token KL x0.3, T=2, top-128 + pooled MSE) on all 2264676 texts of datasets/kd/kd_v1.jsonl, one epoch, batch 16, lr 5e-5 with linear decay, checkpoints every 3 000 steps; then stage 3 = ranking KL on the n4 rows (v32 recipe, fast head). s1 arms for reference: joint-s3 = the same loss on 200 000 texts, s3 = staged 50 k + 150 k, ctrl = stage 3 only.

### tiny10

| task | base-splade-wrapped | arm-distil-v9b-kl-n4 | arm-b4-l0125-f3072-ctrl | arm-b4-l0125-f3072-s3 | arm-b4-l0125-f3072-joint-s3 | arm-b4-l0125-f3072-full-s3 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 42.77 | 37.80 | 43.86 | 43.82 | 44.57 | 46.28 |
| gpt-exams | 98.63 | 98.75 | 97.49 | 98.43 | 98.43 | 98.56 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.63 | 18.19 | 18.59 | 18.61 | 18.64 | 18.53 |
| maupqa-mkqa | 63.92 | 65.12 | 58.72 | 64.37 | 64.77 | 65.31 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.39 | 80.33 | 83.02 | 82.95 | 83.25 | 83.32 |
| maupqa-templates | 75.02 | 75.99 | 73.28 | 75.47 | 75.46 | 75.86 | 75.80 |
| nfcorpus-pl | 30.07 | 32.78 | 31.05 | 32.39 | 32.56 | 32.45 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 76.00 | 69.92 | 75.54 | 75.73 | 77.26 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.91 | 67.95 | 72.92 | 73.92 | 74.30 | 73.48 |
| scifact-pl | 64.52 | 66.85 | 63.63 | 66.89 | 66.63 | 67.46 | 66.44 |
| **mean** | **61.60** | **63.32** | **59.83** | **63.15** | **63.29** | **63.77** | **63.65** |

### heldout8

| task | arm-distil-v9b-kl-n4 | arm-b4-l0125-f3072-full-s3 | soup-v9b |
|---|---:|---:|---:|
| fiqa-pl | 30.04 | 31.00 | 30.37 |
| maupqa-1z10 | 48.73 | 49.00 | 48.49 |
| maupqa-gpt3.5-cc | 20.35 | 20.28 | 20.26 |
| maupqa-multilingual-NLI | 14.66 | 14.57 | 14.53 |
| maupqa-wiki-def | 66.15 | 66.19 | 66.35 |
| mfaq | 63.26 | 63.19 | 63.18 |
| poleval-2022-test-A-legal-questions | 82.67 | 81.59 | 82.72 |
| scidocs-pl | 14.08 | 14.49 | 14.55 |
| **mean** | **42.49** | **42.54** | **42.56** |


## Active document dimensions (base 436)

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 |  |  |  |  |  |
| arm-b4-l0125-f3072-joint-s3 | 449.1 | ×1.03 | 658 |  |  |  |  |  |
| arm-b4-l0125-f3072-full-s3 | 426.2 | ×0.98 | 617 |  |  |  |  |  |

## Fidelity curve of the full KD (256 corpus texts: cosine / L1 relative error / student dims, per kind)

```
  step      0: all: cos 0.706 l1 1.023 dims 285 | passage: cos 0.696 l1 1.061 dims 322 | query: cos 0.762 l1 0.827 dims 90 (teacher dims 383)
  step   3000: all: cos 0.935 l1 0.395 dims 308 | passage: cos 0.932 l1 0.402 dims 347 | query: cos 0.947 l1 0.355 dims 101 (teacher dims 383)
  step   6000: all: cos 0.941 l1 0.363 dims 289 | passage: cos 0.940 l1 0.370 dims 325 | query: cos 0.951 l1 0.331 dims 101 (teacher dims 383)
  step   9000: all: cos 0.947 l1 0.359 dims 332 | passage: cos 0.945 l1 0.367 dims 373 | query: cos 0.958 l1 0.316 dims 116 (teacher dims 383)
  step  12000: all: cos 0.950 l1 0.342 dims 325 | passage: cos 0.948 l1 0.351 dims 367 | query: cos 0.962 l1 0.295 dims 106 (teacher dims 383)
  step  15000: all: cos 0.951 l1 0.340 dims 329 | passage: cos 0.950 l1 0.347 dims 371 | query: cos 0.961 l1 0.299 dims 111 (teacher dims 383)
  step  18000: all: cos 0.954 l1 0.328 dims 315 | passage: cos 0.951 l1 0.334 dims 353 | query: cos 0.965 l1 0.293 dims 115 (teacher dims 383)
  step  21000: all: cos 0.954 l1 0.336 dims 348 | passage: cos 0.953 l1 0.344 dims 394 | query: cos 0.963 l1 0.293 dims 110 (teacher dims 383)
  step  24000: all: cos 0.956 l1 0.317 dims 318 | passage: cos 0.954 l1 0.325 dims 358 | query: cos 0.966 l1 0.279 dims 110 (teacher dims 383)
  step  27000: all: cos 0.956 l1 0.317 dims 320 | passage: cos 0.954 l1 0.325 dims 361 | query: cos 0.966 l1 0.277 dims 104 (teacher dims 383)
  step  30000: all: cos 0.957 l1 0.321 dims 343 | passage: cos 0.955 l1 0.329 dims 386 | query: cos 0.967 l1 0.277 dims 116 (teacher dims 383)
  step  33000: all: cos 0.957 l1 0.314 dims 332 | passage: cos 0.956 l1 0.321 dims 374 | query: cos 0.966 l1 0.281 dims 114 (teacher dims 383)
  step  36000: all: cos 0.958 l1 0.308 dims 323 | passage: cos 0.956 l1 0.317 dims 364 | query: cos 0.970 l1 0.260 dims 107 (teacher dims 383)
  step  39000: all: cos 0.959 l1 0.306 dims 325 | passage: cos 0.957 l1 0.313 dims 365 | query: cos 0.970 l1 0.266 dims 112 (teacher dims 383)
  step  42000: all: cos 0.960 l1 0.310 dims 341 | passage: cos 0.958 l1 0.316 dims 383 | query: cos 0.969 l1 0.276 dims 120 (teacher dims 383)
  step  45000: all: cos 0.961 l1 0.295 dims 311 | passage: cos 0.959 l1 0.303 dims 350 | query: cos 0.970 l1 0.256 dims 107 (teacher dims 383)
  step  48000: all: cos 0.961 l1 0.296 dims 315 | passage: cos 0.959 l1 0.303 dims 355 | query: cos 0.971 l1 0.255 dims 109 (teacher dims 383)
  step  51000: all: cos 0.962 l1 0.296 dims 332 | passage: cos 0.960 l1 0.303 dims 374 | query: cos 0.971 l1 0.258 dims 113 (teacher dims 383)
  step  54000: all: cos 0.961 l1 0.304 dims 346 | passage: cos 0.959 l1 0.313 dims 390 | query: cos 0.971 l1 0.258 dims 115 (teacher dims 383)
  step  57000: all: cos 0.962 l1 0.295 dims 334 | passage: cos 0.960 l1 0.302 dims 375 | query: cos 0.972 l1 0.257 dims 120 (teacher dims 383)
  step  60000: all: cos 0.963 l1 0.293 dims 337 | passage: cos 0.961 l1 0.301 dims 379 | query: cos 0.973 l1 0.250 dims 116 (teacher dims 383)
  step  63000: all: cos 0.963 l1 0.306 dims 362 | passage: cos 0.961 l1 0.315 dims 408 | query: cos 0.973 l1 0.258 dims 121 (teacher dims 383)
  step  66000: all: cos 0.964 l1 0.284 dims 325 | passage: cos 0.962 l1 0.292 dims 366 | query: cos 0.974 l1 0.241 dims 112 (teacher dims 383)
  step  69000: all: cos 0.965 l1 0.289 dims 343 | passage: cos 0.963 l1 0.297 dims 386 | query: cos 0.974 l1 0.245 dims 116 (teacher dims 383)
  step  72000: all: cos 0.965 l1 0.289 dims 342 | passage: cos 0.963 l1 0.297 dims 385 | query: cos 0.974 l1 0.245 dims 115 (teacher dims 383)
  step  75000: all: cos 0.965 l1 0.281 dims 332 | passage: cos 0.964 l1 0.289 dims 373 | query: cos 0.974 l1 0.244 dims 117 (teacher dims 383)
  step  78000: all: cos 0.965 l1 0.284 dims 340 | passage: cos 0.964 l1 0.291 dims 382 | query: cos 0.975 l1 0.245 dims 118 (teacher dims 383)
  step  81000: all: cos 0.966 l1 0.277 dims 328 | passage: cos 0.964 l1 0.285 dims 368 | query: cos 0.976 l1 0.233 dims 113 (teacher dims 383)
  step  84000: all: cos 0.967 l1 0.274 dims 329 | passage: cos 0.965 l1 0.282 dims 370 | query: cos 0.976 l1 0.233 dims 115 (teacher dims 383)
  step  87000: all: cos 0.967 l1 0.274 dims 332 | passage: cos 0.965 l1 0.282 dims 374 | query: cos 0.976 l1 0.231 dims 116 (teacher dims 383)
  step  90000: all: cos 0.967 l1 0.275 dims 335 | passage: cos 0.965 l1 0.283 dims 377 | query: cos 0.977 l1 0.229 dims 117 (teacher dims 383)
  step  93000: all: cos 0.967 l1 0.278 dims 347 | passage: cos 0.965 l1 0.287 dims 391 | query: cos 0.977 l1 0.232 dims 119 (teacher dims 383)
  step  96000: all: cos 0.968 l1 0.269 dims 327 | passage: cos 0.966 l1 0.277 dims 368 | query: cos 0.978 l1 0.226 dims 114 (teacher dims 383)
  step  99000: all: cos 0.968 l1 0.266 dims 321 | passage: cos 0.966 l1 0.274 dims 361 | query: cos 0.978 l1 0.224 dims 113 (teacher dims 383)
  step 102000: all: cos 0.968 l1 0.274 dims 348 | passage: cos 0.966 l1 0.282 dims 392 | query: cos 0.977 l1 0.230 dims 121 (teacher dims 383)
  step 105000: all: cos 0.969 l1 0.266 dims 332 | passage: cos 0.967 l1 0.274 dims 373 | query: cos 0.977 l1 0.226 dims 115 (teacher dims 383)
  step 108000: all: cos 0.969 l1 0.265 dims 334 | passage: cos 0.967 l1 0.273 dims 376 | query: cos 0.978 l1 0.224 dims 116 (teacher dims 383)
  step 111000: all: cos 0.969 l1 0.266 dims 339 | passage: cos 0.967 l1 0.274 dims 382 | query: cos 0.978 l1 0.223 dims 118 (teacher dims 383)
  step 114000: all: cos 0.969 l1 0.266 dims 345 | passage: cos 0.968 l1 0.274 dims 387 | query: cos 0.978 l1 0.225 dims 122 (teacher dims 383)
  step 117000: all: cos 0.970 l1 0.263 dims 344 | passage: cos 0.968 l1 0.272 dims 387 | query: cos 0.979 l1 0.218 dims 118 (teacher dims 383)
  step 120000: all: cos 0.970 l1 0.258 dims 330 | passage: cos 0.968 l1 0.267 dims 371 | query: cos 0.979 l1 0.214 dims 113 (teacher dims 383)
  step 123000: all: cos 0.970 l1 0.260 dims 340 | passage: cos 0.969 l1 0.268 dims 381 | query: cos 0.979 l1 0.223 dims 121 (teacher dims 383)
  step 126000: all: cos 0.970 l1 0.260 dims 338 | passage: cos 0.969 l1 0.268 dims 380 | query: cos 0.979 l1 0.217 dims 118 (teacher dims 383)
  step 129000: all: cos 0.971 l1 0.260 dims 342 | passage: cos 0.969 l1 0.268 dims 385 | query: cos 0.979 l1 0.220 dims 121 (teacher dims 383)
  step 132000: all: cos 0.971 l1 0.259 dims 340 | passage: cos 0.969 l1 0.267 dims 382 | query: cos 0.980 l1 0.215 dims 118 (teacher dims 383)
  step 135000: all: cos 0.971 l1 0.257 dims 339 | passage: cos 0.969 l1 0.265 dims 381 | query: cos 0.980 l1 0.217 dims 119 (teacher dims 383)
  step 138000: all: cos 0.971 l1 0.256 dims 337 | passage: cos 0.970 l1 0.264 dims 379 | query: cos 0.980 l1 0.214 dims 117 (teacher dims 383)
  step 141000: all: cos 0.971 l1 0.256 dims 340 | passage: cos 0.970 l1 0.264 dims 382 | query: cos 0.980 l1 0.214 dims 118 (teacher dims 383)
  step 141543: all: cos 0.971 l1 0.256 dims 340 | passage: cos 0.970 l1 0.264 dims 383 | query: cos 0.980 l1 0.215 dims 118 (teacher dims 383)
```

## Ranking agreement with the teacher scores (500 rows of mix3v2-6219-tv9b.jsonl) and fidelity on queries (512 corpus queries)

```
kd-b4-l0125-f3072-joint                      tau 0.824  top1 0.954  pos-first 0.952  margin 10.57  | queries: cos 0.961 l1 0.302 dims 98/108
arm-b4-l0125-f3072-joint-s3                  tau 0.824  top1 0.970  pos-first 0.960  margin 12.25  | queries: cos 0.957 l1 0.344 dims 112/108
kd-b4-l0125-f3072-full                       tau 0.866  top1 0.974  pos-first 0.956  margin 11.37  | queries: cos 0.980 l1 0.215 dims 106/108
arm-b4-l0125-f3072-full-s3                   tau 0.860  top1 0.978  pos-first 0.958  margin 12.22  | queries: cos 0.976 l1 0.241 dims 107/108
```

## Queue state (s2)

```
kd:b4-l0125-f3072-full	ok	0.9713	17830s
fidelity:kd-b4-l0125-f3072-full.queries	ok	-	14s
train:arm-b4-l0125-f3072-full-s3	ok	-	1334s
sparsity:arm-b4-l0125-f3072-full-s3	ok	426.2	23s
regime:quantized:arm-b4-l0125-f3072-full-s3:tiny10	ok	63.77 10	593s
regime:quantized:arm-b4-l0125-f3072-full-s3:heldout8	ok	42.54 8	2193s
fidelity:arm-b4-l0125-f3072-full-s3.queries	ok	-	14s
```
