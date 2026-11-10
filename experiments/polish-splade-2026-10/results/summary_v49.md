# v49 (day 10.11, quantized regime, nDCG@10): the cross-encoder teacher, student-mined negatives under the ensemble teacher, FLOPS with an L0 threshold, joint doc-side training with the static table, the repaired heldout8 runs

## Full models (heldout8 = 8/8 unless marked)

| model | tiny10 | heldout8 | dims/doc | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 | 23 tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soup-v9b | 63.65 | 42.56 | 409 | 86.88 | 91.97 | 79.32 | 19.25 | 69.66 | 69.42 | 57.57 |
| soup-v12-bias005 | 64.05 | 42.83 | 378 | 87.12 | 92.09 | 79.50 | 19.16 | 70.24 | 69.62 | 57.88 |
| arm-dense-kl-n5kj-ens400 | 64.14 | 43.01 | 422 | 87.29 | 92.45 | 79.57 | 19.16 | 68.49 | 69.39 | 57.93 |
| soup-v14s | 64.18 | 43.11 | 426 | 87.36 | 92.53 | 79.50 | 19.23 | 70.00 | 69.73 | 58.06 |
| soup-v14s-bias005 | 64.22 | 43.07 | 385 | 87.25 | 92.45 | 79.45 | 19.12 | 69.24 | 69.50 | 58.01 |
| soup-v15 | 64.20 | 43.10 | 411 | - | - | - | - | 68.91 | - | - |
| soup-v17 | 64.06 | - | 399 | - | - | - | - | 63.23 | - | - |
| arm-dense-kl-n5kj-ens400-r2v12 | 64.20 | 43.15 | 404 | 87.19 | 92.33 | 79.70 | 19.10 | 67.02 | 69.07 | 57.94 |
| arm-dense-kl-n5kj-ens400-r2v12-s43 | 64.11 | 43.20 | 406 | - | - | - | - | - | - | - |
| soup-v16 | 64.25 | 43.23 | 406 | - | - | - | - | 68.48 | - | - |
| arm-dense-kl-n5kj-ens400-r2v12-bias005 | 64.21 | 43.08 | 365 | 87.03 | 92.24 | 79.68 | 19.02 | 66.02 | 68.80 | 57.86 |
| arm-dense-kl-n5kj-ens3-r2v12 | 63.86 | 43.04 | 425 | - | - | - | - | 65.45 | - | - |
| arm-dense-kl-n5kj-mined-ens400-r2v12 | 64.48 | 43.60 | 515 | - | - | - | - | 71.48 | - | - |
| arm-dense-kl-n5kj-ens400-r2v12-l0t256 | 64.18 | 43.18 | 408 | - | - | - | - | 66.84 | - | - |

## ALBERT student (50 M)

| model | tiny10 | heldout8 | dims/doc |
|---|---:|---:|---:|
| arm-b4-l0125-f1536-hl-r256-w-full-s3 | 62.97 | 42.10 | 512 |
| arm-hl-full-s3n5kj | 63.78 | 42.55 | 487 |
| arm-hl-r2-s3n5kj | 63.72 | 42.67 | 449 |
| arm-hl-r2-s3n5kj-bias010 | 63.69 | 42.46 | 348 |
| asym-r2v12b-q-arm-hl-r2-s3n5kj | 63.99 | 42.65 | - |

## Static query table phr2-k8-kd-r2 over a jointly trained index (frozen index of r2v12-bias005: 60.20 / 38.6; symmetric 64.21 / 43.08)

| arm | asym tiny10 | asym heldout8 | doc model symmetric tiny10 | dims/doc | val top1 | val dims/doc | val dims/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| phr2-frozen | 59.87 | - | 63.84 | 406 | 0.990 | 431 | 67.6 |
| phr2-both | - | - | - | - | 0.000 | 0 | 0.0 |

## Student-mined negatives (r2v12 as the miner)

```
{
  "student": "models/arm-dense-kl-n5kj-ens400-r2v12",
  "rows": 20292,
  "negatives": 202920,
  "by_origin": {
    "hard": 101163,
    "random": 100342,
    "outside_window": 1415,
    "old": 0
  },
  "rows_short_of_negatives": 0,
  "negatives_also_in_old_rows": 8536,
  "median_mined_rank": 10.0,
  "top_k": 60,
  "hard": 5,
  "len_window": [
    0.5,
    2.0
  ],
  "group_keys": [
    "dataset",
    "translation"
  ],
  "pools": {
    "default": 11627,
    "squadv2-v2": 18569,
    "karoljez-strict_fast_v1": 24409
  },
  "seed": 42
}
```

## PolDense-400M on the mined rows

```
{
  "teacher": "OPI-PIB/PolDense-400M",
  "similarity": "cosine",
  "scale": 84.19441654567078,
  "scale_from": "datasets/arms/n5-plus-kj-mined-tv9b.jsonl",
  "query_prefix": "[query]: ",
  "passage_prefix": "",
  "max_seq_length": 512,
  "dtype": "bf16",
  "rows": 20292,
  "median_margin": 3.7361976922453373,
  "rows_with_negative_above_positive": 3113,
  "within_row_pearson_vs_reference": 0.7916453999684211,
  "within_row_kendall_vs_reference": 0.5393222444236434
}
```

## Cross-encoder teacher (sdadas/polish-reranker-roberta-v3) on n5 + kj

```
{
  "teacher": "sdadas/polish-reranker-roberta-v3",
  "scale": 1.5624579139247299,
  "rows": 20292,
  "median_margin": 12.840306601713948,
  "rows_with_negative_above_positive": 81,
  "within_row_pearson_vs_reference": 0.8476198649332329,
  "within_row_kendall_vs_reference": 0.41035482348510893,
  "scale_from": "datasets/arms/n5-plus-kj-tv9b.jsonl"
}
```

## Queue state (v49)

```
kd:b4-l0125-f3072-joint	ok	0.9505	1604s
fidelity:kd-b4-l0125-f3072-joint.queries	ok	-	14s
train:arm-b4-l0125-f3072-joint-s3	ok	-	1338s
sparsity:arm-b4-l0125-f3072-joint-s3	ok	449.1	23s
regime:quantized:arm-b4-l0125-f3072-joint-s3:tiny10	ok	63.29 10	606s
fidelity:arm-b4-l0125-f3072-joint-s3.queries	ok	-	13s
regime:quantized:soup-v14s-bias005:heldout8_fiqa	ok	30.32 1	433s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-s43:heldout8_fiqa	ok	30.95 1	445s
regime:quantized:soup-v16:heldout8_fiqa	ok	31.02 1	435s
regime:quantized:soup-v16:heldout8_1z10	ok	49.39 1	543s
regime:quantized:arm-hl-full-s3n5kj:heldout8_fiqa	ok	30.09 1	371s
regime:quantized:arm-hl-r2-s3n5kj:heldout8_fiqa	ok	30.03 1	357s
rerank:n5kj	ok	-	1044s
train:arm-dense-kl-n5kj-ens3-r2v12	ok	-	5126s
sparsity:arm-dense-kl-n5kj-ens3-r2v12	ok	424.6	24s
regime:quantized:arm-dense-kl-n5kj-ens3-r2v12:tiny10	ok	63.86 10	635s
regime:quantized:arm-dense-kl-n5kj-ens3-r2v12:heldout8	ok	43.04 8	2302s
regime:quantized:arm-dense-kl-n5kj-ens3-r2v12:heldout2_treccovid	ok	65.45 1	1393s
sparsity:soup-v17	ok	399.2	23s
regime:quantized:soup-v17:tiny10	ok	64.06 10	619s
regime:quantized:soup-v17:heldout2_treccovid	ok	63.23 1	1346s
chain:mine_smoke	ok	-	12s
chain:tscore_v9b_smoke	ok	-	14s
chain:tscore_pd400_smoke	ok	-	47s
chain:mine_full	ok	-	124s
chain:tscore_v9b_full	ok	-	538s
chain:tscore_pd400_full	ok	-	2397s
train:arm-dense-kl-n5kj-mined-ens400-r2v12	ok	-	5161s
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12	ok	515.4	24s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12:tiny10	ok	64.48 10	648s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12:heldout8	ok	43.6 8	2340s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12:heldout2_treccovid	ok	71.48 1	1431s
train:arm-dense-kl-n5kj-ens400-r2v12-l0t256	ok	-	5121s
sparsity:arm-dense-kl-n5kj-ens400-r2v12-l0t256	ok	407.8	23s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-l0t256:tiny10	ok	64.18 10	621s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-l0t256:heldout8	ok	43.18 8	2289s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-l0t256:heldout2_treccovid	ok	66.84 1	1356s
joint:r2v12b:phr2-frozen	ok	-	3688s
regime:quantized:asym-joint-phr2-frozen:tiny10	ok	59.87 10	569s
sparsity:joint-phr2-frozen-doc	ok	405.8	24s
regime:quantized:joint-phr2-frozen-doc-sym:tiny10	ok	63.84 10	609s
```
