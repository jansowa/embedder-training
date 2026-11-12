# v53 (evening 11.11, quantized regime, nDCG@10): the student-mined arm with a large candidate pool

minedxl = the mined arm (v49) with negatives chosen by r2v12 from the rows' own passages plus 1000000 passages sampled from the distillation corpus kd_v1; same selection rule, teacher and recipe. New arms are measured in their first bias form with at most 410 dims/doc (biasT = lm_head.bias shifted by -(e^T - 1)).

| model | tiny10 | heldout8 | trec-covid | dims/doc |
|---|---:|---:|---:|---:|
| arm-dense-kl-n5kj-ens400-r2v12 | 64.2 | 43.15 | 67.02 | 404.4 |
| arm-dense-kl-n5kj-mined-ens400-r2v12 | 64.48 | 43.6 | 71.48 | 515.4 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | 70.15 | 405.3 |
| arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015 | 64.58 | 43.22 | 69.97 | 363.4 |

## Mining over the large pool (own pools: teacher P(pos) median 0.852, negative above positive in 2663 rows, student ranks a positive first in 0.844)

```
{
  "student": "models/arm-dense-kl-n5kj-ens400-r2v12",
  "teacher": "ensemble: mean(models/soup-v9b, OPI-PIB/PolDense-400M cosine x 84.20)",
  "dense_scale": 84.20028818158012,
  "rows": 20292,
  "by_origin": {
    "hard": 101103,
    "random": 99666,
    "outside_window": 2151,
    "old": 0
  },
  "negatives_by_pool": {
    "own": 64166,
    "nq": 27493,
    "fiqa-pool": 9590,
    "msmarco-clean": 10201,
    "msmarco-plain": 8683,
    "msmarco-rejected": 9079,
    "wiki-chunks": 10732,
    "hotpotqa": 30035,
    "fever": 16366,
    "fiqa-plus-squadv2": 16575
  },
  "negatives_from_shared_pool": 0.683786713975951,
  "teacher_positive_mass_median": 0.5209455810782921,
  "teacher_positive_mass_above_095": 0.1430120244431303,
  "rows_with_negative_above_positive": 6441,
  "student_ranks_a_positive_first": 0.6312832643406269,
  "teacher_ranks_a_positive_first": 0.6987975556869702,
  "median_negative_chars": 638.0,
  "top_k": 60,
  "hard": 5,
  "len_window": [
    0.5,
    2.0
  ],
  "pools": {
    "default": 11627,
    "squadv2-v2": 18569,
    "karoljez-strict_fast_v1": 24409
  },
  "extra_pool": "datasets/kd/kd_v1.jsonl",
  "extra_max": 1000000,
  "extra_chars": [
    50,
    3000
  ],
  "dense_texts_encoded_now": 93974,
  "seed": 42
}
```

## Queue state (v53)

```
chain:bigpool_smoke	ok	-	96s
chain:bigpool_full	ok	-	3148s
train:arm-dense-kl-n5kj-minedxl-ens400-r2v12	ok	-	5189s
sparsity:arm-dense-kl-n5kj-minedxl-ens400-r2v12	ok	529.4	24s
sparsity:arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias005	ok	468.5	23s
sparsity:arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias010	ok	413.2	24s
sparsity:arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015	ok	363.4	23s
regime:quantized:arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015:tiny10	ok	64.58 10	580s
regime:quantized:arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015:heldout8	ok	43.22 8	1976s
regime:quantized:arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015:heldout2_treccovid	ok	69.97 1	1241s
```
