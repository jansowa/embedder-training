# v56 (12.11, quantized regime, nDCG@10): the query dose of the student-mined arm

dose3 = the rows of the reference arm unchanged plus the gated SQuAD-v2 rows outside n5, negatives mined by r2v12 under the same rules, the same ensemble teacher with the fixed dense scale, the same recipe (from soup-v12, 2 epochs). Branch: **own** (rows' own passages, as in the mined arm). New arms are measured in their first bias form with at most 410 dims/doc. The cost columns come from the cost sample (1 350 passages + 1 350 queries).

Rows: 60385 = 20292 base + 40093 new (training pairs 61981 against 21476). Teacher P(pos) at T = 2, median: new rows 0.856, base rows 0.852; share above 0.95: 0.296 / 0.303; rows with a negative above a positive: 4168 of 40093 new, 2658 of 20292 base. Pools of the new rows: {'squadv2-v2': 19008, 'default': 11627, 'karoljez-strict_fast_v1': 24409}.

| model | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-ens400-r2v12 | 64.20 | 43.15 | 67.02 | 404.4 | - | - | - |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | 70.15 | 405.3 | 413.6 | 130.8 | 16461 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias011 | 64.62 | 43.48 | 69.70 | 395.5 | 403.5 | 127.9 | 15782 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias012 | 64.63 | 43.44 | 69.60 | 385.9 | 393.6 | 125.0 | 15110 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 64.64 | 43.40 | 69.74 | 376.5 | 384.0 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 | 64.67 | 43.35 | 69.86 | 358.0 | 365.2 | 116.7 | 13272 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias015 | 64.61 | 43.45 | 69.59 | 366.7 | 374.8 | 119.1 | 13658 |
| **arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015** | 64.96 | 43.63 | 72.82 | 374.7 | 381.5 | 121.9 | 14079 |

## Reading

arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 64.96 / 43.63 at 374.7 dims/doc against arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 64.64 / 43.40 at 376.5: delta +0.32 / +0.23, trec-covid 72.82 against 69.74 -> THE DOSE PAYS (>= +0.2 on both sets).

Seed noise of this recipe is 0.13-0.2 (one seed here). Nothing further was started by this queue.

## Mining of the new rows

```
{
  "student": "models/arm-dense-kl-n5kj-ens400-r2v12",
  "teacher": "ensemble: mean(models/soup-v9b, OPI-PIB/PolDense-400M cosine x 84.19)",
  "dense_scale": 84.19441654567078,
  "rows": 40093,
  "by_origin": {
    "hard": 199994,
    "random": 198568,
    "outside_window": 2368,
    "old": 0
  },
  "teacher_positive_mass_median": 0.8557928974452454,
  "teacher_positive_mass_above_095": 0.295612700471404,
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
  "dense_texts_encoded_now": 0,
  "seed": 42
}
```

## Branch decision (v53 result files)

```
{
  "branch": "own",
  "xl_forms": [
    {
      "label": "arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015",
      "tiny10": 64.57913257426662,
      "heldout8": 43.21549417217829,
      "dims_per_doc": 363.41925925925926
    }
  ],
  "rule": "large pool wins when its form is >= +0.2 on tiny10 and on heldout8 over the mined-arm form at the nearest dims/doc",
  "xl_pick": "arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015",
  "xl_reference": "arm-dense-kl-n5kj-mined-ens400-r2v12-bias015",
  "xl_delta_tiny10": -0.08933192740194329,
  "xl_delta_heldout8": -0.13213157935472708
}
```

## Queue state (v56)

```
chain:dose_smoke	ok	-	62s
chain:dose_full	ok	-	156s
train:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12	ok	-	14906s
sparsity:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12	ok	548.0	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias005	ok	484.1	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias010	ok	426.4	22s
sparsity:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015	ok	374.7	22s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:tiny10	ok	64.96 10	589s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:heldout8	ok	43.63 8	2072s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:heldout2_treccovid	ok	72.82 1	1302s
```
