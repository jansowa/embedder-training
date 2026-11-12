# v57 (12.11, quantized regime, nDCG@10): the large-pool arm with a false-negative filter

f95 = the rows of the large-pool arm (v53: own passages + 1 M passages of kd_v1, 5 best + 5 random of the student's top-60) without the negatives the ensemble teacher scores at or above 0.95 x the best positive of the row, refilled from the same row's own-pool negatives (the rows of the mined arm) under the same threshold. Queries, positives and the recipe (from soup-v12, 2 epochs) are unchanged; no new mining. The cost columns come from the cost sample (1 350 passages + 1 350 queries).

Rows: 20292. Negatives dropped: 25478 of 202920; refilled from the own-pool rows: 21285; kept above the threshold as the last resort: 4193 (in 977 rows); rows untouched: 10463. Negatives from the shared pool: 0.684 -> 0.597. Teacher P(pos) at T = 2, median: large pool 0.521, filtered 0.737, own pools 0.852. Rows with a negative above a positive: large pool 6441, filtered 932, own pools 2658.

| model | negatives | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-ens400-r2v12 | old (round 2) | 64.20 | 43.15 | 67.02 | 404.4 | - | - | - |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | own pools | 64.64 | 43.40 | 69.74 | 376.5 | 384.0 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 | own pools | 64.67 | 43.35 | 69.86 | 358.0 | 365.2 | 116.7 | 13272 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-s43-bias015 | own pools, seed 43 | 64.61 | 43.45 | 69.59 | 366.7 | - | - | - |
| arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015 | large pool | 64.58 | 43.22 | 69.97 | 363.4 | 369.9 | 119.2 | 13792 |
| **arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12-bias015** | large pool, filtered | 64.69 | 43.17 | 69.90 | 366.1 | 371.5 | 120.5 | 13897 |

## Reading

arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12-bias015 64.69 / 43.17 at 366.1 dims/doc against the own pools arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 64.67 / 43.35 at 358.0: delta +0.03 / -0.18, trec-covid 69.90 against 69.86 -> within the seed noise of the own pools: the large pool adds nothing under this filter. With this, the composition of the negative lists at fixed queries is exhausted: own pools (teacher P(pos) median 0.85), the large pool (0.52), this filter (0.74), refreshed negatives and hard positives (v52) all land within the noise of each other, so no further variant of negative selection is proposed; the open axes are new queries. Against the unfiltered large pool arm-dense-kl-n5kj-minedxl-ens400-r2v12-bias015 (64.58 / 43.22 at 363.4): delta +0.11 / -0.05.

Seed noise of this recipe is 0.13-0.2 (one seed here). The refill negatives were scored in another pass of the same teacher (dense factor 84.194 against 84.200; the positives of the two passes differ by the amounts in the block below). Nothing further was started by this queue.

## The filter

```
{
  "rows": 20292,
  "rule": "drop negatives with teacher score >= 0.95 x the best positive of the row",
  "source_rows": "datasets/arms/n5-plus-kj-minedxl.jsonl",
  "refill_rows": "datasets/arms/n5-plus-kj-mined-tens400.jsonl",
  "negatives_kept_from_large_pool_rows": 177442,
  "negatives_dropped": 25478,
  "negatives_refilled_from_own_pool_rows": 21285,
  "negatives_above_threshold_kept_as_last_resort": 4193,
  "rows_with_a_last_resort_negative": 977,
  "rows_untouched": 10463,
  "negatives_from_shared_pool_before": 0.683786713975951,
  "negatives_from_shared_pool_after": 0.5971565148827124,
  "dropped_share_by_pool": {
    "own": 0.109,
    "hotpotqa": 0.0794,
    "nq": 0.1304,
    "fiqa-plus-squadv2": 0.3322,
    "fever": 0.0841,
    "wiki-chunks": 0.0876,
    "msmarco-clean": 0.1116,
    "fiqa-pool": 0.1709,
    "msmarco-rejected": 0.1043,
    "msmarco-plain": 0.1113
  },
  "teacher_positive_mass_median": {
    "large pool": 0.5209455810782921,
    "filtered": 0.7367417203943293,
    "own pools": 0.8521623619963876
  },
  "rows_with_negative_above_positive": {
    "large pool": 6441,
    "filtered": 932,
    "own pools": 2658
  },
  "positive_score_abs_diff_between_the_two_scoring_passes": {
    "median": 0.019444129084208228,
    "max": 0.2840239997534546
  }
}
```

## Queue state (v57)

```
train:arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12	ok	-	5156s
sparsity:arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12	ok	537.8	24s
sparsity:arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12-bias015	ok	366.1	23s
regime:quantized:arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12-bias015:tiny10	ok	64.69 10	576s
regime:quantized:arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12-bias015:heldout8	ok	43.17 8	1980s
regime:quantized:arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12-bias015:heldout2_treccovid	ok	69.9 1	1269s
```
