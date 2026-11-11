# v52 (night 10/11.11, quantized regime, nDCG@10): refreshed negatives, hard positives, the second seed of the mined arm

Every new arm is measured in its first bias form with at most 410 dims/doc (biasT = lm_head.bias shifted by -(e^T - 1)). iter4 = 4 x 0.5 epoch from soup-v12, the current model mines its negatives before each segment; n5hpkj = the SQuAD-v2 part of n5 replaced by rows the teacher gets right and r2v12 does not; mined = v49 (negatives mined once by r2v12).

| model | tiny10 | heldout8 | trec-covid | dims/doc |
|---|---:|---:|---:|---:|
| arm-dense-kl-n5kj-ens400-r2v12 | 64.2 | 43.15 | 67.02 | 404.4 |
| arm-dense-kl-n5kj-mined-ens400-r2v12 | 64.48 | 43.6 | 71.48 | 515.4 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | 70.15 | 405.3 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 | 64.67 | 43.35 | 69.86 | 358.0 |
| arm-dense-kl-n5kj-iter4-r2v12-s4-bias015 | 64.47 | 43.35 | 71.06 | 367.4 |
| arm-dense-kl-n5hpkj-mined-ens400-r2v12-bias015 | 64.65 | 43.38 | 69.98 | 371.8 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias015 | 64.61 | 43.45 | 69.59 | 366.7 |
| soup-v19-bias010 | 64.74 | 43.55 | 70.17 | 409.2 |

## Refresh rounds (student = miner of the round)

| round | miner | dense scale | teacher P(pos) median | rows with a negative above a positive | student ranks a positive first |
|---|---|---:|---:|---:|---:|
| 1 | models/arm-dense-kl-n5kj-ens400-r2v12 | 84.20028818158012 | 0.8521524605428898 | 2663 | 0.8442243248570865 |
| 2 | models/arm-dense-kl-n5kj-iter4-r2v12-s1 | 83.20613633056357 | 0.8481191483727174 | 2665 | 0.8453084959589986 |
| 3 | models/arm-dense-kl-n5kj-iter4-r2v12-s2 | 83.75553867635678 | 0.849096713041201 | 2664 | 0.8480189237137788 |
| 4 | models/arm-dense-kl-n5kj-iter4-r2v12-s3 | 84.21087422783751 | 0.8505670325235802 | 2666 | 0.8509264734870885 |

## Hard-positive candidates (gated SQuAD-v2 rows outside n5, r2v12 as the student)

```
{
  "student": "models/arm-dense-kl-n5kj-ens400-r2v12",
  "teacher": "ensemble: mean(models/soup-v9b, OPI-PIB/PolDense-400M cosine x 84.20)",
  "dense_scale": 84.20028818158012,
  "rows": 40093,
  "by_origin": {
    "hard": 199994,
    "random": 198568,
    "outside_window": 2368,
    "old": 0
  },
  "teacher_positive_mass_median": 0.8558009655452511,
  "teacher_positive_mass_above_095": 0.2956625844910583,
  "rows_with_negative_above_positive": 4168,
  "student_ranks_a_positive_first": 0.8666600154640461,
  "teacher_ranks_a_positive_first": 0.8988850921607263,
  "median_negative_chars": 755.0,
  "top_k": 60,
  "hard": 5,
  "len_window": [
    0.5,
    2.0
  ],
  "pools": {
    "squadv2-v2": 19008,
    "default": 11627,
    "karoljez-strict_fast_v1": 24409
  },
  "dense_texts_encoded_now": 40436,
  "seed": 42
}
```

## Queue state (v52)

```
chain:refresh_smoke	ok	-	56s
chain:refresh_smoke2	ok	-	19s
chain:refresh_iter1	ok	-	727s
train:arm-dense-kl-n5kj-iter4-r2v12-s1	ok	-	1245s
chain:refresh_iter2	ok	-	234s
train:arm-dense-kl-n5kj-iter4-r2v12-s2	ok	-	1273s
chain:refresh_iter3	ok	-	231s
train:arm-dense-kl-n5kj-iter4-r2v12-s3	ok	-	1275s
chain:refresh_iter4	ok	-	237s
train:arm-dense-kl-n5kj-iter4-r2v12-s4	ok	-	1281s
sparsity:arm-dense-kl-n5kj-iter4-r2v12-s4	ok	524.1	23s
sparsity:arm-dense-kl-n5kj-iter4-r2v12-s4-bias005	ok	467.1	21s
sparsity:arm-dense-kl-n5kj-iter4-r2v12-s4-bias010	ok	414.9	22s
sparsity:arm-dense-kl-n5kj-iter4-r2v12-s4-bias015	ok	367.4	22s
regime:quantized:arm-dense-kl-n5kj-iter4-r2v12-s4-bias015:tiny10	ok	64.47 10	579s
regime:quantized:arm-dense-kl-n5kj-iter4-r2v12-s4-bias015:heldout8	ok	43.35 8	1988s
regime:quantized:arm-dense-kl-n5kj-iter4-r2v12-s4-bias015:heldout2_treccovid	ok	71.06 1	1251s
chain:refresh_hp-candidates	ok	-	205s
train:arm-dense-kl-n5hpkj-mined-ens400-r2v12	ok	-	4909s
sparsity:arm-dense-kl-n5hpkj-mined-ens400-r2v12	ok	535.4	22s
sparsity:arm-dense-kl-n5hpkj-mined-ens400-r2v12-bias005	ok	475.9	22s
sparsity:arm-dense-kl-n5hpkj-mined-ens400-r2v12-bias010	ok	421.3	21s
sparsity:arm-dense-kl-n5hpkj-mined-ens400-r2v12-bias015	ok	371.8	22s
regime:quantized:arm-dense-kl-n5hpkj-mined-ens400-r2v12-bias015:tiny10	ok	64.65 10	555s
regime:quantized:arm-dense-kl-n5hpkj-mined-ens400-r2v12-bias015:heldout8	ok	43.38 8	1982s
regime:quantized:arm-dense-kl-n5hpkj-mined-ens400-r2v12-bias015:heldout2_treccovid	ok	69.98 1	1324s
train:arm-dense-kl-n5kj-mined-ens400-r2v12-s43	ok	-	5026s
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12-s43	ok	527.1	23s
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias005	ok	468.6	23s
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias010	ok	415.2	22s
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias015	ok	366.7	22s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias015:tiny10	ok	64.61 10	577s
chain:refresh_smoke	ok	-	59s
chain:refresh_smoke2	ok	-	19s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias015:heldout2_treccovid	ok	69.59 1	1290s
sparsity:soup-v19	ok	519.9	22s
sparsity:soup-v19-bias005	ok	461.8	22s
sparsity:soup-v19-bias010	ok	409.2	22s
regime:quantized:soup-v19-bias010:tiny10	ok	64.74 10	587s
regime:quantized:soup-v19-bias010:heldout8	ok	43.55 8	2072s
regime:quantized:soup-v19-bias010:heldout2_treccovid	ok	70.17 1	1355s
```
