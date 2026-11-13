# v61 (quantized regime, nDCG@10): translated factual sets (NQ, HotpotQA, FEVER, TriviaQA) under KD

fact20 = the rows of the dose arm of v56 (`datasets/arms/n5-plus-kj-mined-dose3.jsonl`, unchanged) plus 17402 rows of translated factual sets with one positive per query, negatives mined by r2v12 from a pool of each source's own passages and the scores of the ensemble teacher with the fixed dense scale; the recipe unchanged (from soup-v12, 2 epochs). The base is dose3, not dose3 + faq20, so the arm answers its own question and compares with v56, v58 and v60. The new arm is measured in the first bias form (steps of 0.01) with at most 1.02 x the dims/doc of the reference `arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015`.

## Result

| model | rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 20 k arm (v49) | 64.64 | 43.40 | 69.74 | 376.5 | 384.0 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | dose3 (v56) | 64.96 | 43.63 | 72.82 | 374.7 | 381.5 | 121.9 | 14079 |
| arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015 | faq20 (v58) | 64.84 | 43.54 | 70.39 | 377.9 | 385.8 | 127.6 | 15289 |
| arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016 | dose3 + faq20 (v60) | 64.93 | 43.60 | 70.79 | 375.5 | 382.8 | 129.4 | 15001 |
| **arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014** | dose3 + fact20 (v61) | 64.77 | 43.49 | 73.75 | 379.9 | 386.0 | 128.7 | 15113 |

## Reading

arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014 64.77 / 43.49 at 379.9 dims/doc against dose3 (v56) arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 64.96 / 43.63 at 374.7: delta -0.19 / -0.15, trec-covid 73.75 against 72.82 -> within the seed noise of the dose arm: these translated factual sources are neutral also under KD with the current recipe (ensemble teacher, student-mined negatives).

Seed noise of this recipe is 0.13-0.2 (one seed here).

## Task groups (mean delta against 20 k arm (v49); tasks both models have)

| group | tasks | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) |
|---|---:|---:|---:|---:|---:|
| wiki-factual tasks (MAUPQA incl. czywiesz, poquad, 1z10, wiki-def) | 10 | +0.10 | +0.02 (8) | +0.12 (8) | +0.15 |
| legal questions and mfaq (old TriviaQA pattern) | 2 | +0.36 | +0.33 | +0.25 | +0.05 |
| Allegro FAQ A, B (old NQ pattern) | 2 | +0.62 | +1.24 | +0.64 | +0.30 |
| arguana-pl (old HotpotQA pattern) | 1 | +0.86 | +0.26 | +0.67 | -0.42 |
| maupqa-czy-wiesz-v2 (old FEVER pattern) | 1 | +0.21 | - | - | +0.18 |
| maupqa-poquad | 1 | +0.01 | - | - | +0.27 |
| all tasks | 20 | +0.26 | +0.17 (18) | +0.25 (18) | +0.12 |

### tiny10 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + fact20 (v61) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.81 | 18.88 | 18.79 | 18.88 | 18.89 | +0.01 |
| maupqa-mkqa | 65.62 | 65.60 | 65.19 | 65.58 | 65.34 | -0.26 |
| maupqa-poleval2021-pairs | 83.84 | 83.95 | 84.01 | 83.96 | 84.17 | +0.23 |
| maupqa-templates | 76.31 | 76.57 | 76.54 | 76.67 | 76.78 | +0.21 |
| poleval-2022-test-A-allegro-faq | 77.85 | 78.80 | 79.30 | 78.39 | 77.84 | -0.96 |
| poleval-2022-test-B-allegro-faq | 74.47 | 74.77 | 75.51 | 75.22 | 75.07 | +0.30 |
| arguana-pl | 49.38 | 50.24 | 49.64 | 50.05 | 48.96 | -1.28 |
| nfcorpus-pl | 32.93 | 32.84 | 32.85 | 32.95 | 32.85 | +0.01 |
| scifact-pl | 68.16 | 68.84 | 67.58 | 68.56 | 68.73 | -0.11 |
| gpt-exams | 99.04 | 99.10 | 99.05 | 99.09 | 99.08 | -0.02 |

### heldout8 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + fact20 (v61) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3.5-cc | 20.51 | 20.54 | 20.60 | 20.60 | 20.49 | -0.05 |
| maupqa-multilingual-NLI | 14.86 | 14.89 | 14.72 | 14.71 | 14.87 | -0.02 |
| mfaq | 63.54 | 63.56 | 64.18 | 64.15 | 63.48 | -0.08 |
| poleval-2022-test-A-legal-questions | 84.35 | 85.06 | 84.38 | 84.25 | 84.52 | -0.55 |
| maupqa-wiki-def | 67.86 | 67.87 | 67.98 | 68.01 | 67.95 | +0.07 |
| scidocs-pl | 14.76 | 14.81 | 14.69 | 14.93 | 14.72 | -0.08 |
| maupqa-1z10 | 49.18 | 49.46 | 49.32 | 49.50 | 49.53 | +0.07 |
| fiqa-pl | 32.17 | 32.88 | 32.48 | 32.64 | 32.36 | -0.51 |

### heldout2_czywiesz per task

| task | 20 k arm (v49) | dose3 (v56) | dose3 + fact20 (v61) | dose3 + fact20 (v61) - dose3 (v56) |
|---|---:|---:|---:|---:|
| maupqa-czy-wiesz-v2 | 87.45 | 87.66 | 87.63 | -0.03 |

### heldout2_poquad per task

| task | 20 k arm (v49) | dose3 (v56) | dose3 + fact20 (v61) | dose3 + fact20 (v61) - dose3 (v56) |
|---|---:|---:|---:|---:|
| maupqa-poquad | 19.42 | 19.43 | 19.69 | +0.26 |

The group means include czywiesz and poquad only for the models that have them (the count is in brackets when it differs). Nothing further was started by this queue.

## Rows per source (left after each filter)

### fever: 5000 rows, pool 79276 passages

Rows from `/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/FlagEmbedding-data/speakleash/fever/train_pl.jsonl`; pool from `/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/FlagEmbedding-data/speakleash/fever/train_pl.jsonl`. Rules: excluded domains none; answer location any; broken-Polish veto yes (annotation records: 8706); positive at most 512 tokens (median of the chosen positives 269.0, max 512); positives per row in the source {'1': 89034, '2': 16074, '3': 2177, '4': 845, '5': 537, '6': 1070}.

| step | removed | left |
|---|---:|---:|
| rows read | 0 | 109810 |
| empty_query_or_positive | 73 | 109737 |
| polish_veto | 49 | 109688 |
| benchmark_query | 0 | 109688 |
| query_in_exclude_rows | 0 | 109688 |
| repeated_query | 10792 | 98896 |
| drawn at random (seed 42), examined for the token limit | 92120 | 6776 |
| no allowed positive within 512 tokens | 1776 | 5000 |

### hotpotqa: 3838 rows, pool 120000 passages

Rows from `datasets/arms/pool/hotpotqa-pool-8000.jsonl`; pool from `/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/FlagEmbedding-data/speakleash/hotpotqa/train_pl.jsonl`. Rules: excluded domains none; answer location ['explicit', 'inferred']; broken-Polish veto yes (annotation records: 13272); positive at most 512 tokens (median of the chosen positives 97.0, max 419); positives per row in the source {'1': 3056, '2': 4926, '3': 18}.

| step | removed | left |
|---|---:|---:|
| rows read | 0 | 8000 |
| no_positive_with_the_answer | 4077 | 3923 |
| polish_veto | 85 | 3838 |
| benchmark_query | 0 | 3838 |
| query_in_exclude_rows | 0 | 3838 |
| drawn at random (seed 42), examined for the token limit | 0 | 3838 |
| no allowed positive within 512 tokens | 0 | 3838 |

### nq: 3564 rows, pool 103298 passages

Rows from `datasets/arms/pool/nq-pool-8000.jsonl`; pool from `/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/FlagEmbedding-data/speakleash/nq/train_pl.jsonl`. Rules: excluded domains ['culture_entertainment', 'sport']; answer location any; broken-Polish veto yes (annotation records: 8128); positive at most 512 tokens (median of the chosen positives 125.0, max 510); positives per row in the source {'1': 8000}.

| step | removed | left |
|---|---:|---:|
| rows read | 0 | 8000 |
| domain_excluded | 4349 | 3651 |
| polish_veto | 67 | 3584 |
| benchmark_query | 12 | 3572 |
| benchmark_query_loose_key | 1 | 3571 |
| query_in_exclude_rows | 0 | 3571 |
| repeated_query | 7 | 3564 |
| drawn at random (seed 42), examined for the token limit | 0 | 3564 |
| no allowed positive within 512 tokens | 0 | 3564 |

### trivia: 5000 rows, pool 120000 passages

Rows from `/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/doc2query/data/raw/trivia-mined-negatives/train_pl.jsonl`; pool from `/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/doc2query/data/raw/trivia-mined-negatives/train_pl.jsonl`. Rules: excluded domains none; answer location any; broken-Polish veto no (annotation records: 0); positive at most 512 tokens (median of the chosen positives 146.0, max 279); positives per row in the source {'1': 5096, '2': 6101, '3': 5029, '4': 4419, '5': 3674, '6': 35978}.

| step | removed | left |
|---|---:|---:|
| rows read | 0 | 60413 |
| translation_missing | 116 | 60297 |
| benchmark_query | 56 | 60241 |
| query_in_exclude_rows | 14 | 60227 |
| repeated_query | 372 | 59855 |
| drawn at random (seed 42), examined for the token limit | 54855 | 5000 |
| no allowed positive within 512 tokens | 0 | 5000 |

## Mining per source (teacher P(pos) at T = 2)

| rows | n | teacher P(pos) median | share with P(pos) > 0.95 | rows with a negative above the positive | student ranks the positive first |
|---|---:|---:|---:|---:|---:|
| fever | 5000 | 0.967 | 0.575 | 475 (9.5 %) | 0.818 |
| hotpotqa | 3838 | 0.952 | 0.507 | 499 (13.0 %) | 0.789 |
| nq | 3564 | 0.689 | 0.213 | 892 (25.0 %) | 0.584 |
| trivia | 5000 | 0.664 | 0.223 | 1510 (30.2 %) | 0.632 |
| new SQuAD-v2 rows of dose3 (v56) | 40093 | 0.856 | 0.296 | 4168 (10.4 %) | 0.867 |
| FAQ rows (v58) | 60000 | 0.982 | 0.655 | 3864 (6.4 %) | 0.868 |
| rows of the 20 k arm | 20292 | 0.852 | 0.303 | 2658 (13.1 %) | 0.844 |

Memory: none of the factual rows were changed for memory.

## Rows of the arm

```
{
  "rows": 77787,
  "base_rows": 60385,
  "new_rows": 17402,
  "new_rows_read": 17402,
  "new_rows_left_out_duplicate_query": 0,
  "new_rows_with_a_query_id_of_a_base_row": 0,
  "new_rows_by_dataset": {
    "nq-kd": 3564,
    "hotpotqa-kd": 3838,
    "fever-kd": 5000,
    "trivia-kd": 5000
  },
  "training_pairs": 79383,
  "base_rows_file": "datasets/arms/n5-plus-kj-mined-dose3.jsonl",
  "new_rows_file": "datasets/arms/fact-kd-v1-mined.jsonl",
  "new_rows_group": null,
  "seed": 42
}
```

## Queue state (v61)

```
chain:v61_rows_nq	ok	-	48s
chain:v61_rows_hotpotqa	ok	-	50s
chain:v61_rows_fever	ok	-	95s
chain:v61_rows_trivia	ok	-	66s
chain:v61_mine_smoke	ok	-	70s
chain:v61_mine_nq	ok	-	417s
chain:v61_mine_hotpotqa	ok	-	407s
chain:v61_mine_fever	ok	-	588s
chain:v61_mine_trivia	ok	-	712s
train:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12	ok	-	19173s
sparsity:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12	ok	547.2	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias005	ok	481.9	22s
sparsity:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias010	ok	422.8	22s
sparsity:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias011	ok	411.7	22s
sparsity:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias012	ok	400.9	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias013	ok	390.2	22s
sparsity:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014	ok	379.9	23s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014:tiny10	ok	64.77 10	582s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014:heldout8	ok	43.49 8	2166s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014:heldout2_treccovid	ok	73.75 1	1346s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014:heldout2_czywiesz	ok	87.63 1	740s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014:heldout2_poquad	ok	19.69 1	1208s
```
