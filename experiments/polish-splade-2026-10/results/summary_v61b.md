# v61b (quantized regime, nDCG@10): translated GooAQ pairs under KD

dose3 + gooaq = the rows of the dose arm of v56 (`datasets/arms/n5-plus-kj-mined-dose3.jsonl`, unchanged) plus 17896 rows of GooAQ (`datasets/qooaq.jsonl`: questions of a search engine with the answer snippet, translated by Qwen3.5-27B), one positive per query, negatives mined by r2v12 from the answers plus 100 000 MS MARCO-pl passages and the scores of the ensemble teacher with the fixed dense scale; the recipe unchanged (from soup-v12, 2 epochs). The new arm is measured in the first bias form (steps of 0.01) with at most 1.02 x the dims/doc of the reference `arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015`.

## Result

| model | rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 20 k arm (v49) | 64.64 | 43.40 | 69.74 | 376.5 | 384.0 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | dose3 (v56) | 64.96 | 43.63 | 72.82 | 374.7 | 381.5 | 121.9 | 14079 |
| arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015 | faq20 (v58) | 64.84 | 43.54 | 70.39 | 377.9 | 385.8 | 127.6 | 15289 |
| arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016 | dose3 + faq20 (v60) | 64.93 | 43.60 | 70.79 | 375.5 | 382.8 | 129.4 | 15001 |
| arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014 | dose3 + fact20 (v61) | 64.77 | 43.49 | 73.75 | 379.9 | 386.0 | 128.7 | 15113 |
| **arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016** | dose3 + gooaq (v61b) | 64.82 | 43.49 | 70.87 | 381.2 | 389.3 | 125.9 | 14707 |

## Reading

arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016 64.82 / 43.49 at 381.2 dims/doc against dose3 (v56) arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 64.96 / 43.63 at 374.7: delta -0.14 / -0.15, trec-covid 70.87 against 72.82 -> within the seed noise of the dose arm; the soups of v62 take the arm when it is not more than 0.2 below dose3 on either set.

Seed noise of this recipe is 0.13-0.2 (one seed here).

## Task groups (mean delta against 20 k arm (v49); tasks both models have)

| group | tasks | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + gooaq (v61b) |
|---|---:|---:|---:|---:|---:|---:|
| FAQ tasks (Allegro FAQ A, B, mfaq) | 3 | +0.42 | +1.04 | +0.63 | +0.17 | +0.28 |
| tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions) | 4 | +0.74 | +0.00 | +0.36 | +0.13 | +0.21 |
| scifact-pl | 1 | +0.68 | -0.58 | +0.40 | +0.57 | -0.76 |
| maupqa-mkqa | 1 | -0.02 | -0.44 | -0.04 | -0.28 | +0.05 |
| MAUPQA tasks | 8 | +0.10 | +0.02 | +0.12 | +0.13 | +0.06 |
| all tasks | 18 | +0.28 | +0.17 | +0.25 | +0.11 | +0.14 |

### tiny10 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + gooaq (v61b) | dose3 + gooaq (v61b) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.81 | 18.88 | 18.79 | 18.88 | 18.89 | 18.84 | -0.05 |
| maupqa-mkqa | 65.62 | 65.60 | 65.19 | 65.58 | 65.34 | 65.67 | +0.07 |
| maupqa-poleval2021-pairs | 83.84 | 83.95 | 84.01 | 83.96 | 84.17 | 84.05 | +0.10 |
| maupqa-templates | 76.31 | 76.57 | 76.54 | 76.67 | 76.78 | 76.64 | +0.07 |
| poleval-2022-test-A-allegro-faq | 77.85 | 78.80 | 79.30 | 78.39 | 77.84 | 77.95 | -0.85 |
| poleval-2022-test-B-allegro-faq | 74.47 | 74.77 | 75.51 | 75.22 | 75.07 | 74.90 | +0.14 |
| arguana-pl | 49.38 | 50.24 | 49.64 | 50.05 | 48.96 | 50.50 | +0.26 |
| nfcorpus-pl | 32.93 | 32.84 | 32.85 | 32.95 | 32.85 | 33.17 | +0.33 |
| scifact-pl | 68.16 | 68.84 | 67.58 | 68.56 | 68.73 | 67.40 | -1.44 |
| gpt-exams | 99.04 | 99.10 | 99.05 | 99.09 | 99.08 | 99.07 | -0.03 |

### heldout8 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + gooaq (v61b) | dose3 + gooaq (v61b) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3.5-cc | 20.51 | 20.54 | 20.60 | 20.60 | 20.49 | 20.56 | +0.02 |
| maupqa-multilingual-NLI | 14.86 | 14.89 | 14.72 | 14.71 | 14.87 | 14.81 | -0.08 |
| mfaq | 63.54 | 63.56 | 64.18 | 64.15 | 63.48 | 63.85 | +0.29 |
| poleval-2022-test-A-legal-questions | 84.35 | 85.06 | 84.38 | 84.25 | 84.52 | 84.40 | -0.66 |
| maupqa-wiki-def | 67.86 | 67.87 | 67.98 | 68.01 | 67.95 | 67.46 | -0.42 |
| scidocs-pl | 14.76 | 14.81 | 14.69 | 14.93 | 14.72 | 14.77 | -0.04 |
| maupqa-1z10 | 49.18 | 49.46 | 49.32 | 49.50 | 49.53 | 49.48 | +0.01 |
| fiqa-pl | 32.17 | 32.88 | 32.48 | 32.64 | 32.36 | 32.57 | -0.30 |

Nothing further was started by this queue.

## Rows (left after each filter)

17896 rows of 18819 pairs read (file modified 2026-11-13 06:47:06; 0 unreadable lines). Questions 5 / 7 / 10 words (p10 / p50 / p90), answers 177 / 283 / 346 characters, positives median 63.0 tokens (max 228). Pool: 100000 of the 467514 distinct MS MARCO-pl passages of `datasets/msmarco-full-judged-qwen/pairs_msmarco_full.unique.jsonl` (seed 42) plus the answers of the rows.

| step | removed | left |
|---|---:|---:|
| pairs read | 0 | 18819 |
| empty_question_or_answer | 0 | 18819 |
| untranslated_side | 63 | 18756 |
| repeated_question | 50 | 18706 |
| answer_of_an_earlier_question | 332 | 18374 |
| debris:suggest_new_definition | 1 | 18373 |
| debris:cut_answer_trailing_ellipsis | 193 | 18180 |
| other_script_characters | 45 | 18135 |
| answer_shorter_than_80_chars | 225 | 17910 |
| positive_above_512_tokens | 0 | 17910 |
| benchmark_query | 14 | 17896 |
| benchmark_query_loose_key | 0 | 17896 |
| query_in_exclude_rows | 0 | 17896 |

Debris patterns (case-insensitive regular expressions on the answer): suggest_new_definition = `suggest new definition|zaproponuj nową definicję|zasugeruj nową definicję`; cut_answer_trailing_ellipsis = `(\.\.\.|…)\s*$`. Other-script characters (question or answer): `[぀-ヿ㐀-鿿가-힯Ѐ-ӿ؀-ۿ]`. An untranslated side = the question or the answer equal to its English original. No Qwen labels exist for these pairs, so the broken-Polish veto of v61 does not apply; the other-script check is the only cheap detector.

## Mining (teacher P(pos) at T = 2)

| rows | n | teacher P(pos) median | share with P(pos) > 0.95 | rows with a negative above the positive | student ranks the positive first |
|---|---:|---:|---:|---:|---:|
| gooaq (v61b) | 17896 | 0.740 | 0.276 | 4753 (26.6 %) | 0.630 |
| fever (v61) | 5000 | 0.967 | 0.575 | 475 (9.5 %) | 0.818 |
| hotpotqa (v61) | 3838 | 0.952 | 0.507 | 499 (13.0 %) | 0.789 |
| nq (v61) | 3564 | 0.689 | 0.213 | 892 (25.0 %) | 0.584 |
| trivia (v61) | 5000 | 0.664 | 0.223 | 1510 (30.2 %) | 0.632 |
| new SQuAD-v2 rows of dose3 (v56) | 40093 | 0.856 | 0.296 | 4168 (10.4 %) | 0.867 |
| FAQ rows (v58) | 60000 | 0.982 | 0.655 | 3864 (6.4 %) | 0.868 |
| rows of the 20 k arm | 20292 | 0.852 | 0.303 | 2658 (13.1 %) | 0.844 |

Negatives by pool: msmarco-pl 110324, own 68636.

## Rows of the arm

```
{
  "rows": 78281,
  "base_rows": 60385,
  "new_rows": 17896,
  "new_rows_read": 17896,
  "new_rows_left_out_duplicate_query": 0,
  "new_rows_with_a_query_id_of_a_base_row": 0,
  "new_rows_by_dataset": {
    "gooaq-kd": 17896
  },
  "training_pairs": 79877,
  "base_rows_file": "datasets/arms/n5-plus-kj-mined-dose3.jsonl",
  "new_rows_file": "datasets/arms/gooaq-kd-v1-mined.jsonl",
  "new_rows_group": null,
  "seed": 42
}
```

## Queue state (v61b)

```
chain:v61b_rows	ok	-	28s
chain:v61b_mine_smoke	ok	-	57s
chain:v61b_mine_gooaq	ok	-	651s
train:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12	ok	-	18717s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12	ok	576.9	25s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias005	ok	509.1	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias010	ok	447.1	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias011	ok	435.5	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias012	ok	424.2	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias013	ok	413.0	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias014	ok	402.2	23s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias015	ok	391.6	24s
sparsity:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016	ok	381.2	24s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016:tiny10	ok	64.82 10	597s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016:heldout8	ok	43.49 8	2085s
regime:quantized:arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016:heldout2_treccovid	ok	70.87 1	1323s
```
