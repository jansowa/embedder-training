# s1 (06.11.2026 02:00, leaderboard regime, nDCG@10): small-SPLADE probe, teacher soup-v9b

Students: b4-f3072 = layers 0,1,2,5 of soup-v9b with the full FFN (68 M params, 0.83 of the teacher's MACs per token); hl-r256-w = layers 0,1,2,5 with FFN 1536 as an ALBERT with the teacher's head transform as an extra layer and a rank-256 vocabulary / decoder (50 M, 0.45). Stages: s1 token KL on 50000 texts, s2 pooled MSE (relative) + token KL x0.3 on 150000 texts of datasets/kd/kd_v1.jsonl, joint = the combined loss on 200000 texts, s3 ranking KL on the n4 rows (v32 recipe, fast head). ctrl = s3 only. Memory check (fast head, no checkpointing): b4 2.27 GB, hl 2.00 GB (limit 7.0).

### tiny10

| task | base-splade-wrapped | arm-n4-3060 | arm-distil-v9b-kl-n4 | arm-b4-l0125-f3072-ctrl | arm-b4-l0125-f3072-s3 | arm-b4-l0125-f3072-joint-s3 | arm-b4-l0125-f1536-hl-r256-w-s3 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 42.96 | 42.77 | 37.80 | 43.86 | 43.82 | 40.86 | 46.28 |
| gpt-exams | 98.63 | 98.52 | 98.75 | 97.49 | 98.43 | 98.43 | 97.83 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.50 | 18.63 | 18.19 | 18.59 | 18.61 | 18.26 | 18.53 |
| maupqa-mkqa | 63.92 | 63.87 | 65.12 | 58.72 | 64.37 | 64.77 | 62.93 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.03 | 83.39 | 80.33 | 83.02 | 82.95 | 81.40 | 83.32 |
| maupqa-templates | 75.02 | 75.60 | 75.99 | 73.28 | 75.47 | 75.46 | 74.40 | 75.80 |
| nfcorpus-pl | 30.07 | 31.85 | 32.78 | 31.05 | 32.39 | 32.56 | 31.19 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.57 | 76.00 | 69.92 | 75.54 | 75.73 | 71.90 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 71.42 | 72.91 | 67.95 | 72.92 | 73.92 | 69.37 | 73.48 |
| scifact-pl | 64.52 | 67.60 | 66.85 | 63.63 | 66.89 | 66.63 | 64.64 | 66.44 |
| **mean** | **61.60** | **62.89** | **63.32** | **59.83** | **63.15** | **63.29** | **61.28** | **63.65** |


## Active document dimensions (base 436)

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 |  |  |  |  |  |
| arm-b4-l0125-f3072-ctrl | 529.9 | ×1.21 | 778 |  |  |  |  |  |
| arm-b4-l0125-f3072-s3 | 460.8 | ×1.06 | 676 |  |  |  |  |  |
| arm-b4-l0125-f3072-joint-s3 | 449.1 | ×1.03 | 658 |  |  |  |  |  |
| arm-b4-l0125-f1536-hl-r256-w-s3 | 593.2 | ×1.36 | 875 |  |  |  |  |  |

## Fidelity curve per KD checkpoint (256 corpus texts: cosine / L1 relative error / student dims, per kind)

```
kd-b4-l0125-f1536-hl-r256-w-s1:
  step      0: all: cos 0.455 l1 2.080 dims 599 | passage: cos 0.446 l1 2.102 dims 669 | query: cos 0.502 l1 1.960 dims 236 (teacher dims 383)
  step   1500: all: cos 0.599 l1 0.874 dims 59 | passage: cos 0.584 l1 0.882 dims 64 | query: cos 0.681 l1 0.830 dims 33 (teacher dims 383)
  step   3000: all: cos 0.658 l1 0.827 dims 78 | passage: cos 0.646 l1 0.833 dims 86 | query: cos 0.718 l1 0.795 dims 39 (teacher dims 383)
  step   3125: all: cos 0.652 l1 0.821 dims 70 | passage: cos 0.639 l1 0.832 dims 77 | query: cos 0.723 l1 0.765 dims 34 (teacher dims 383)
kd-b4-l0125-f1536-hl-r256-w-s2:
  step      0: all: cos 0.657 l1 0.812 dims 68 | passage: cos 0.641 l1 0.826 dims 75 | query: cos 0.733 l1 0.748 dims 31 (teacher dims 388)
  step   1500: all: cos 0.854 l1 0.646 dims 340 | passage: cos 0.850 l1 0.647 dims 383 | query: cos 0.876 l1 0.644 dims 133 (teacher dims 388)
  step   3000: all: cos 0.879 l1 0.539 dims 278 | passage: cos 0.874 l1 0.547 dims 315 | query: cos 0.905 l1 0.504 dims 101 (teacher dims 388)
  step   4500: all: cos 0.880 l1 0.574 dims 345 | passage: cos 0.876 l1 0.582 dims 391 | query: cos 0.902 l1 0.537 dims 122 (teacher dims 388)
  step   6000: all: cos 0.891 l1 0.514 dims 292 | passage: cos 0.886 l1 0.520 dims 330 | query: cos 0.913 l1 0.483 dims 109 (teacher dims 388)
  step   7500: all: cos 0.895 l1 0.499 dims 283 | passage: cos 0.890 l1 0.510 dims 322 | query: cos 0.918 l1 0.446 dims 93 (teacher dims 388)
  step   9000: all: cos 0.899 l1 0.498 dims 309 | passage: cos 0.895 l1 0.506 dims 352 | query: cos 0.919 l1 0.459 dims 104 (teacher dims 388)
  step   9375: all: cos 0.899 l1 0.502 dims 313 | passage: cos 0.895 l1 0.508 dims 353 | query: cos 0.920 l1 0.470 dims 115 (teacher dims 388)
kd-b4-l0125-f3072-joint:
  step      0: all: cos 0.706 l1 1.023 dims 285 | passage: cos 0.696 l1 1.061 dims 322 | query: cos 0.762 l1 0.827 dims 90 (teacher dims 383)
  step   1500: all: cos 0.925 l1 0.428 dims 304 | passage: cos 0.922 l1 0.435 dims 342 | query: cos 0.940 l1 0.394 dims 107 (teacher dims 383)
  step   3000: all: cos 0.935 l1 0.399 dims 319 | passage: cos 0.932 l1 0.407 dims 359 | query: cos 0.948 l1 0.358 dims 105 (teacher dims 383)
  step   4500: all: cos 0.940 l1 0.387 dims 333 | passage: cos 0.938 l1 0.395 dims 375 | query: cos 0.954 l1 0.344 dims 113 (teacher dims 383)
  step   6000: all: cos 0.945 l1 0.361 dims 317 | passage: cos 0.943 l1 0.369 dims 357 | query: cos 0.954 l1 0.322 dims 105 (teacher dims 383)
  step   7500: all: cos 0.946 l1 0.363 dims 334 | passage: cos 0.943 l1 0.372 dims 378 | query: cos 0.958 l1 0.313 dims 104 (teacher dims 383)
  step   9000: all: cos 0.947 l1 0.346 dims 295 | passage: cos 0.945 l1 0.353 dims 332 | query: cos 0.957 l1 0.309 dims 99 (teacher dims 383)
  step  10500: all: cos 0.948 l1 0.346 dims 311 | passage: cos 0.946 l1 0.352 dims 350 | query: cos 0.957 l1 0.313 dims 105 (teacher dims 383)
  step  12000: all: cos 0.949 l1 0.343 dims 315 | passage: cos 0.948 l1 0.349 dims 355 | query: cos 0.958 l1 0.310 dims 110 (teacher dims 383)
  step  12500: all: cos 0.950 l1 0.342 dims 326 | passage: cos 0.949 l1 0.350 dims 368 | query: cos 0.960 l1 0.303 dims 107 (teacher dims 383)
kd-b4-l0125-f3072-s1:
  step      0: all: cos 0.706 l1 1.023 dims 285 | passage: cos 0.696 l1 1.061 dims 322 | query: cos 0.762 l1 0.827 dims 90 (teacher dims 383)
  step   1500: all: cos 0.833 l1 0.710 dims 365 | passage: cos 0.830 l1 0.716 dims 411 | query: cos 0.843 l1 0.681 dims 123 (teacher dims 383)
  step   3000: all: cos 0.851 l1 0.846 dims 540 | passage: cos 0.848 l1 0.847 dims 607 | query: cos 0.867 l1 0.842 dims 187 (teacher dims 383)
  step   3125: all: cos 0.844 l1 0.874 dims 553 | passage: cos 0.841 l1 0.868 dims 621 | query: cos 0.857 l1 0.906 dims 199 (teacher dims 383)
kd-b4-l0125-f3072-s2:
  step      0: all: cos 0.844 l1 0.882 dims 565 | passage: cos 0.841 l1 0.871 dims 641 | query: cos 0.856 l1 0.937 dims 201 (teacher dims 388)
  step   1500: all: cos 0.931 l1 0.406 dims 309 | passage: cos 0.928 l1 0.416 dims 353 | query: cos 0.948 l1 0.357 dims 100 (teacher dims 388)
  step   3000: all: cos 0.937 l1 0.404 dims 341 | passage: cos 0.934 l1 0.411 dims 388 | query: cos 0.949 l1 0.371 dims 117 (teacher dims 388)
  step   4500: all: cos 0.941 l1 0.376 dims 323 | passage: cos 0.939 l1 0.382 dims 367 | query: cos 0.952 l1 0.344 dims 111 (teacher dims 388)
  step   6000: all: cos 0.944 l1 0.368 dims 326 | passage: cos 0.942 l1 0.375 dims 370 | query: cos 0.955 l1 0.337 dims 111 (teacher dims 388)
  step   7500: all: cos 0.946 l1 0.369 dims 345 | passage: cos 0.944 l1 0.377 dims 392 | query: cos 0.958 l1 0.330 dims 115 (teacher dims 388)
  step   9000: all: cos 0.946 l1 0.367 dims 335 | passage: cos 0.944 l1 0.373 dims 380 | query: cos 0.957 l1 0.337 dims 117 (teacher dims 388)
  step   9375: all: cos 0.948 l1 0.357 dims 333 | passage: cos 0.945 l1 0.364 dims 379 | query: cos 0.958 l1 0.324 dims 116 (teacher dims 388)
```

## Ranking agreement with the teacher scores (500 rows of mix3v2-6219-tv9b.jsonl) and fidelity on queries (512 corpus queries)

```
arm-b4-l0125-f1536-hl-r256-w-s3              tau 0.741  top1 0.956  pos-first 0.940  margin 11.67  | queries: cos 0.905 l1 0.576 dims 143/108
arm-b4-l0125-f3072-ctrl                      tau 0.712  top1 0.952  pos-first 0.944  margin 12.95  | queries: cos 0.844 l1 0.761 dims 129/108
arm-b4-l0125-f3072-joint-s3                  tau 0.824  top1 0.970  pos-first 0.960  margin 12.25  | queries: cos 0.957 l1 0.344 dims 112/108
arm-b4-l0125-f3072-s3                        tau 0.824  top1 0.974  pos-first 0.960  margin 12.32  | queries: cos 0.954 l1 0.361 dims 117/108
kd-b4-l0125-f1536-hl-r256-w-s1               tau 0.496  top1 0.762  pos-first 0.780  margin 2.53  | queries: cos 0.729 l1 0.815 dims 33/108
kd-b4-l0125-f1536-hl-r256-w-s2               tau 0.732  top1 0.904  pos-first 0.906  margin 8.88  | queries: cos 0.917 l1 0.471 dims 106/108
kd-b4-l0125-f3072-joint                      tau 0.824  top1 0.954  pos-first 0.952  margin 10.57  | queries: cos 0.961 l1 0.302 dims 98/108
kd-b4-l0125-f3072-s1                         tau 0.670  top1 0.912  pos-first 0.910  margin 13.26  | queries: cos 0.858 l1 0.901 dims 183/108
kd-b4-l0125-f3072-s2                         tau 0.816  top1 0.952  pos-first 0.934  margin 10.58  | queries: cos 0.959 l1 0.320 dims 108/108
student-b4-l0125-f1536-hl-r256-w             tau 0.353  top1 0.606  pos-first 0.644  margin 3.76  | queries: cos 0.513 l1 1.873 dims 220/108
student-b4-l0125-f3072                       tau 0.530  top1 0.814  pos-first 0.838  margin 10.28  | queries: cos 0.765 l1 0.820 dims 84/108
```

## Queue state (s1)

```
train:arm-distil-v9b-kl-n6-fresh38k	ok	-	9279s
sparsity:arm-distil-v9b-kl-n6-fresh38k	ok	433.2	24s
regime:quantized:arm-distil-v9b-kl-n6-fresh38k:tiny10	ok	63.42 10	644s
regime:quantized:arm-distil-v9b-kl-n6-fresh38k:heldout8	ok	42.6 8	2536s
memcheck:n4-3060-fast	ok	3.10	exact, no checkpointing
train:arm-n4-3060-fast	ok	-	1814s
sparsity:arm-n4-3060-fast	ok	416.4	24s
regime:quantized:arm-n4-3060-fast:tiny10	ok	62.99 10	632s
regime:quantized:arm-n4-3060-fast:heldout8	ok	42.02 8	2258s
train:arm-distil-v9b-kl-n1v2-qreg4x	ok	-	1237s
sparsity:arm-distil-v9b-kl-n1v2-qreg4x	ok	425.0	23s
regime:quantized:arm-distil-v9b-kl-n1v2-qreg4x:tiny10	ok	63.26 10	636s
train:arm-distil-v9b-kl-n1v2-qreg10x	ok	-	1225s
sparsity:arm-distil-v9b-kl-n1v2-qreg10x	ok	432.6	24s
regime:quantized:arm-distil-v9b-kl-n1v2-qreg10x:tiny10	ok	63.28 10	631s
train:arm-dense-kl-n1v2-pd150	FAILED	-	1222s
train:arm-dense-kl-n1v2-pd150	ok	-	1222s
sparsity:arm-dense-kl-n1v2-pd150	ok	451.9	23s
regime:quantized:arm-dense-kl-n1v2-pd150:tiny10	ok	63.41 10	624s
regime:quantized:arm-dense-kl-n1v2-pd150:heldout8	ok	42.28 8	2374s
train:arm-dense-kl-n1v2-mmlwB	ok	-	1257s
sparsity:arm-dense-kl-n1v2-mmlwB	ok	404.3	23s
regime:quantized:arm-dense-kl-n1v2-mmlwB:tiny10	ok	62.9 10	613s
train:arm-dense-kl-n1v2-ens	ok	-	1255s
sparsity:arm-dense-kl-n1v2-ens	ok	446.7	24s
regime:quantized:arm-dense-kl-n1v2-ens:tiny10	ok	63.56 10	634s
regime:quantized:arm-dense-kl-n1v2-ens:heldout8	ok	42.59 8	2285s
fidelity:hl-r256-w.cuda-fp32	ok	-	-
fidelity:hl-r256-w.cuda-fp16	ok	-	-
fidelity:hl-r256-w.cuda-fp16-fp32head	ok	-	-
memcheck:b4-l0125-f3072	2.27	-	-
memcheck:b4-l0125-f1536-hl-r256-w	2.00	-	-
fidelity:student-b4-l0125-f3072.queries	ok	-	13s
train:arm-b4-l0125-f3072-ctrl	ok	-	1288s
sparsity:arm-b4-l0125-f3072-ctrl	ok	529.9	20s
regime:quantized:arm-b4-l0125-f3072-ctrl:tiny10	ok	59.83 10	655s
fidelity:arm-b4-l0125-f3072-ctrl.queries	ok	-	14s
kd:b4-l0125-f3072-s1	ok	0.8436	367s
fidelity:kd-b4-l0125-f3072-s1.queries	ok	-	13s
kd:b4-l0125-f3072-s2	ok	0.9476	1211s
fidelity:kd-b4-l0125-f3072-s2.queries	ok	-	13s
train:arm-b4-l0125-f3072-s3	ok	-	1337s
sparsity:arm-b4-l0125-f3072-s3	ok	460.8	23s
regime:quantized:arm-b4-l0125-f3072-s3:tiny10	ok	63.15 10	612s
fidelity:arm-b4-l0125-f3072-s3.queries	ok	-	14s
kd:b4-l0125-f3072-joint	ok	0.9505	1604s
fidelity:kd-b4-l0125-f3072-joint.queries	ok	-	14s
train:arm-b4-l0125-f3072-joint-s3	ok	-	1338s
sparsity:arm-b4-l0125-f3072-joint-s3	ok	449.1	23s
regime:quantized:arm-b4-l0125-f3072-joint-s3:tiny10	ok	63.29 10	606s
fidelity:arm-b4-l0125-f3072-joint-s3.queries	ok	-	13s
fidelity:student-b4-l0125-f1536-hl-r256-w.queries	ok	-	12s
kd:b4-l0125-f1536-hl-r256-w-s1	ok	0.6524	297s
fidelity:kd-b4-l0125-f1536-hl-r256-w-s1.queries	ok	-	13s
kd:b4-l0125-f1536-hl-r256-w-s2	ok	0.8995	953s
fidelity:kd-b4-l0125-f1536-hl-r256-w-s2.queries	ok	-	13s
train:arm-b4-l0125-f1536-hl-r256-w-s3	ok	-	1087s
sparsity:arm-b4-l0125-f1536-hl-r256-w-s3	ok	593.2	20s
regime:quantized:arm-b4-l0125-f1536-hl-r256-w-s3:tiny10	ok	61.28 10	593s
fidelity:arm-b4-l0125-f1536-hl-r256-w-s3.queries	ok	-	13s
```
