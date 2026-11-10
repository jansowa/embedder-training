# v45 (day 9.11, quantized regime, nDCG@10): trec-covid around r2v12, the ALBERT student on the n5 + kj rows and from the r2v12 teacher, the n5-only round 2, one more static pool

References: soup-v9b 63.65 / 42.56, trec-covid 69.66; soup-v12-bias005 64.05 / 42.83, trec-covid 70.24; r2v12 64.20 / 43.15; r2v12-bias005 64.21 / 43.08, trec-covid 66.02; hl-full-s3 62.97 / 42.10 (bias010 63.17 / 42.01); b4-full-s3 63.77 / 42.54.

## Full models

| model | tiny10 | heldout8 | dims/doc | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soup-v9b | 63.65 | 42.56 | 409 | 86.88 | 91.97 | 79.32 | 19.25 | 69.66 | 69.42 |
| soup-v12 | 63.96 | 42.92 | 418 | - | - | - | - | 69.93 | - |
| soup-v12-bias005 | 64.05 | 42.83 | 378 | 87.12 | 92.09 | 79.50 | 19.16 | 70.24 | 69.62 |
| arm-dense-kl-n5v2-ens400 | 63.95 | 42.96 | 425 | - | - | - | - | 69.38 | - |
| arm-dense-kl-n5kj-ens400 | 64.14 | 43.01 | 422 | 87.29 | 92.45 | 79.57 | 19.16 | 68.49 | 69.39 |
| soup-v14 | 64.15 | 43.00 | 421 | - | - | - | - | 69.27 | - |
| soup-v14s | 64.18 | 43.11 | 426 | 87.36 | 92.53 | 79.50 | 19.23 | 70.00 | 69.73 |
| soup-v15 | 64.20 | 43.10 | 411 | - | - | - | - | 68.91 | - |
| arm-dense-kl-n5kj-ens400-r2v12 | 64.20 | 43.15 | 404 | 87.19 | 92.33 | 79.70 | 19.10 | 67.02 | 69.07 |
| arm-dense-kl-n5kj-ens400-r2v12-bias005 | 64.21 | 43.08 | 365 | 87.03 | 92.24 | 79.68 | 19.02 | 66.02 | 68.80 |
| arm-dense-kl-n5kj-ens400-r2v12-bias010 | 64.24 | 43.00 | 328 | 86.85 | 92.16 | 79.56 | 18.86 | 65.35 | 68.56 |

## ALBERT student (50 M, 0.45 of the teacher's cost)

| model | what | tiny10 | heldout8 | dims/doc |
|---|---|---:|---:|---:|
| arm-b4-l0125-f1536-hl-r256-w-full-s3 | s3: KD from soup-v9b, stage 3 on n4 (v9b scores) | 62.97 | 42.10 | 512 |
| arm-hl-full-s3n5kj | s3 KD student, stage 3 on n5 + kj (ensemble scores) | 63.78 | 44.34 | 487 |
| arm-hl-r2-s3n5kj | KD continued from r2v12 (600 k), stage 3 on n5 + kj | 63.72 | 44.48 | 449 |
| arm-hl-r2-s3n5kj-bias010 | KD-continued arm at bias 0.10 | 63.69 | 42.46 | 348 |
| asym-r2v12b-q-arm-hl-r2-s3n5kj | query encoder over the r2v12-bias005 index | 63.99 | 42.65 | - |

## Static query tables over r2v12-bias005 (symmetric 64.21 / 43.08)

| table | tiny10 | heldout8 | val top1 | dims/q | postings fraction |
|---|---:|---:|---:|---:|---:|
| credit-k8-kd-r2 | 59.70 | - | 0.617 | 68.5 | 8.324 |
| credit-k8-kd | 59.76 | 38.60 | 0.651 | 77.6 | 9.256 |
| phr1-k8-kd | 59.87 | - | 0.673 | 79.5 | 9.543 |
| phr2-k8-kd-r2-q300k | 60.29 | - | 0.670 | 62.1 | 7.690 |
| phr2-k8-kd-r2 | 60.20 | 38.60 | 0.671 | 69.8 | 8.440 |
| phr2-k8-kd-r3 | 60.04 | - | 0.674 | 65.5 | 7.985 |
| phr2-k8-kd | 60.01 | - | 0.680 | 79.9 | 9.592 |
| phr2m3-k8-kd-r2 | 60.12 | - | 0.684 | 69.8 | 8.427 |
| phr2m3-k8-kd | - | - | 0.694 | 79.8 | 9.594 |

## Queue state (v45)

```
fidelity:hl-r256-w.cuda-fp32	ok	-	-
fidelity:hl-r256-w.cuda-fp16	ok	-	-
fidelity:hl-r256-w.cuda-fp16-fp32head	ok	-	-
memcheck:b4-l0125-f1536-hl-r256-w	2.00	-	-
fidelity:student-b4-l0125-f1536-hl-r256-w.queries	ok	-	12s
kd:b4-l0125-f1536-hl-r256-w-s1	ok	0.6524	297s
fidelity:kd-b4-l0125-f1536-hl-r256-w-s1.queries	ok	-	13s
kd:b4-l0125-f1536-hl-r256-w-s2	ok	0.8995	953s
fidelity:kd-b4-l0125-f1536-hl-r256-w-s2.queries	ok	-	13s
train:arm-b4-l0125-f1536-hl-r256-w-s3	ok	-	1087s
sparsity:arm-b4-l0125-f1536-hl-r256-w-s3	ok	593.2	20s
regime:quantized:arm-b4-l0125-f1536-hl-r256-w-s3:tiny10	ok	61.28 10	593s
fidelity:arm-b4-l0125-f1536-hl-r256-w-s3.queries	ok	-	13s
kd:b4-l0125-f1536-hl-r256-w-full	ok	0.9496	13973s
fidelity:kd-b4-l0125-f1536-hl-r256-w-full.queries	ok	-	12s
train:arm-b4-l0125-f1536-hl-r256-w-full-s3	ok	-	1088s
sparsity:arm-b4-l0125-f1536-hl-r256-w-full-s3	ok	512.0	20s
regime:quantized:arm-b4-l0125-f1536-hl-r256-w-full-s3:tiny10	ok	62.97 10	567s
regime:quantized:arm-b4-l0125-f1536-hl-r256-w-full-s3:heldout8	ok	42.1 8	2142s
fidelity:arm-b4-l0125-f1536-hl-r256-w-full-s3.queries	ok	-	13s
regime:quantized:soup-v15:heldout2_treccovid	ok	68.91 1	1385s
regime:quantized:soup-v14:heldout2_treccovid	ok	69.27 1	1399s
sparsity:soup-v15	ok	410.8	23s
regime:quantized:soup-v15:tiny10	ok	64.2 10	606s
regime:quantized:soup-v15:heldout8	ok	43.1 8	2228s
train:arm-hl-full-s3n5kj	ok	-	3353s
sparsity:arm-hl-full-s3n5kj	ok	487.5	20s
regime:quantized:arm-hl-full-s3n5kj:tiny10	ok	63.78 10	544s
regime:quantized:arm-hl-full-s3n5kj:heldout8	ok	44.34 7	2131s
kd:hl-r2	ok	0.963	3717s
fidelity:kd-hl-r2.queries	ok	-	13s
train:arm-hl-r2-s3n5kj	ok	-	3346s
sparsity:arm-hl-r2-s3n5kj	ok	449.0	20s
regime:quantized:arm-hl-r2-s3n5kj:tiny10	ok	63.72 10	524s
regime:quantized:arm-hl-r2-s3n5kj:heldout8	ok	44.48 7	2028s
regime:quantized:asym-r2v12b-q-arm-hl-r2-s3n5kj:tiny10	ok	63.99 10	616s
regime:quantized:asym-r2v12b-q-arm-hl-r2-s3n5kj:heldout8	ok	42.65 8	2308s
sparsity:arm-hl-r2-s3n5kj-bias010	ok	347.5	20s
regime:quantized:arm-hl-r2-s3n5kj-bias010:tiny10	ok	63.69 10	481s
regime:quantized:arm-hl-r2-s3n5kj-bias010:heldout8	ok	42.46 8	1812s
static_mine:kd-r2v12b-q300k	ok	-	962s
static_train:r2v12b:phr2-k8-kd-r2-q300k	ok	-	543s
regime:quantized:asym-r2v12b-q-static-phr2-k8-kd-r2-q300k:tiny10	ok	60.29 10	562s
```
