# v60 (quantized regime, nDCG@10): the SQuAD-v2 dose and the native FAQ rows together

dose3 + faq20 = the rows of the dose arm of v56 (`datasets/arms/n5-plus-kj-mined-dose3.jsonl`, unchanged) plus the same 20000 native Polish FAQ rows that went into the FAQ arm of v58, cut out of its row file by their group; no new mining. The recipe is unchanged (from soup-v12, 2 epochs, LR 2e-5, 2x16, 512 tokens, KL from the ensemble teacher's scores in the rows). The new arm is measured in the first bias form (steps of 0.01) with at most 1.02 x the dims/doc of the reference `arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015`.

## Result

| model | rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 20 k arm (v49) | 64.64 | 43.40 | 69.74 | 376.5 | 384.0 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | dose3 (v56) | 64.96 | 43.63 | 72.82 | 374.7 | 381.5 | 121.9 | 14079 |
| arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015 | faq20 (v58) | 64.84 | 43.54 | 70.39 | 377.9 | 385.8 | 127.6 | 15289 |
| **arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016** | dose3 + faq20 (v60) | 64.93 | 43.60 | 70.79 | 375.5 | 382.8 | 129.4 | 15001 |

## Reading

arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016 64.93 / 43.60 at 375.5 dims/doc against dose3 (v56) arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 64.96 / 43.63 at 374.7: delta -0.02 / -0.03, trec-covid 70.79 against 72.82 -> within the seed noise of the dose arm: by the rule dose3 stays the best arm; the task groups below say whether the FAQ gain was kept at no cost elsewhere.

Seed noise of this recipe is 0.13-0.2 (one seed here).

## Task groups (mean delta against 20 k arm (v49); tasks both models have)

| group | tasks | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) |
|---|---:|---:|---:|---:|
| FAQ tasks (Allegro FAQ A, B, mfaq) | 3 | +0.42 | +1.04 | +0.63 |
| tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions) | 4 | +0.74 | +0.00 | +0.36 |
| scifact-pl | 1 | +0.68 | -0.58 | +0.40 |
| maupqa-mkqa | 1 | -0.02 | -0.44 | -0.04 |
| MAUPQA tasks | 8 | +0.10 | +0.02 | +0.12 |
| all tasks | 18 | +0.28 | +0.17 | +0.25 |

### tiny10 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + faq20 (v60) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.81 | 18.88 | 18.79 | 18.88 | -0.01 |
| maupqa-mkqa | 65.62 | 65.60 | 65.19 | 65.58 | -0.02 |
| maupqa-poleval2021-pairs | 83.84 | 83.95 | 84.01 | 83.96 | +0.02 |
| maupqa-templates | 76.31 | 76.57 | 76.54 | 76.67 | +0.10 |
| poleval-2022-test-A-allegro-faq | 77.85 | 78.80 | 79.30 | 78.39 | -0.41 |
| poleval-2022-test-B-allegro-faq | 74.47 | 74.77 | 75.51 | 75.22 | +0.45 |
| arguana-pl | 49.38 | 50.24 | 49.64 | 50.05 | -0.19 |
| nfcorpus-pl | 32.93 | 32.84 | 32.85 | 32.95 | +0.11 |
| scifact-pl | 68.16 | 68.84 | 67.58 | 68.56 | -0.28 |
| gpt-exams | 99.04 | 99.10 | 99.05 | 99.09 | -0.01 |

### heldout8 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + faq20 (v60) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|
| maupqa-gpt3.5-cc | 20.51 | 20.54 | 20.60 | 20.60 | +0.06 |
| maupqa-multilingual-NLI | 14.86 | 14.89 | 14.72 | 14.71 | -0.18 |
| mfaq | 63.54 | 63.56 | 64.18 | 64.15 | +0.59 |
| poleval-2022-test-A-legal-questions | 84.35 | 85.06 | 84.38 | 84.25 | -0.81 |
| maupqa-wiki-def | 67.86 | 67.87 | 67.98 | 68.01 | +0.14 |
| scidocs-pl | 14.76 | 14.81 | 14.69 | 14.93 | +0.13 |
| maupqa-1z10 | 49.18 | 49.46 | 49.32 | 49.50 | +0.04 |
| fiqa-pl | 32.17 | 32.88 | 32.48 | 32.64 | -0.24 |

## 23 tasks

23-task mean = (10 x tiny10 + 8 x heldout8 + the five held-out-2 tasks) / 23.

| model | tiny10 | heldout8 | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 | 23 tasks | dims/doc | cost: dims/query | cost: postings/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 64.64 | 43.40 | 87.45 | 92.67 | 79.60 | 19.42 | 69.74 | 69.77 | 58.37 | 376.5 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | 64.96 | 43.63 | 87.66 | 92.73 | 79.52 | 19.43 | 72.82 | 70.43 | 58.73 | 374.7 | 121.9 | 14079 |
| **arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016** | 64.93 | 43.60 | - | - | - | - | 70.79 | - | - | 375.5 | 129.4 | 15001 |

## Winner of the pair

**arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015**: dose3 + faq20 does not win by the rule (reading: noise), so the dose arm stays. Its held-out-2 tasks and 23-task mean are in the table above; the row without them belongs to the arm that was not chosen. Queue v59b trains the hard MS MARCO rows (when H passed its threshold) on the rows of this winner.

Nothing further was started by this queue. All three arms with new rows (dose3, faq20, dose3 + faq20) also train for more steps than the 20 k arm; these numbers do not separate that from the data.

## Rows

```
{
  "rows": 80385,
  "base_rows": 60385,
  "new_rows": 20000,
  "new_rows_read": 20000,
  "new_rows_left_out_duplicate_query": 0,
  "new_rows_with_a_query_id_of_a_base_row": 0,
  "new_rows_by_dataset": {
    "webfaq-v1": 20000
  },
  "training_pairs": 81981,
  "base_rows_file": "datasets/arms/n5-plus-kj-mined-dose3.jsonl",
  "new_rows_file": "datasets/arms/n5-plus-kj-mined-faq20.jsonl",
  "new_rows_group": "webfaq-v1",
  "seed": 42
}
```

## Queue state (v60)

```
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:heldout2_treccovid	ok	72.82 1	1302s
train:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12	ok	-	18673s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12	ok	570.5	25s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias005	ok	502.5	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias010	ok	440.9	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias011	ok	429.3	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias012	ok	418.0	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias013	ok	407.0	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias014	ok	396.3	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias015	ok	385.8	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016	ok	375.5	24s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016:tiny10	ok	64.93 10	591s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016:heldout8	ok	43.6 8	2082s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016:heldout2_treccovid	ok	70.79 1	1303s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:heldout2_czywiesz	ok	87.66 1	727s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:heldout2_gpt35wiki	ok	92.73 1	589s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:heldout2_quora	ok	79.52 1	730s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015:heldout2_poquad	ok	19.43 1	1184s
```
