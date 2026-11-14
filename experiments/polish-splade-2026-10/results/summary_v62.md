# v62 (quantized regime, nDCG@10): weight soups of the data arms and the freeze of the full model

The soups are uniform averages of the raw weights of arms that all start from soup-v12 with one recipe and differ only in their rows; each is measured in its first bias form with at most 410 dims/doc (steps of 0.05). fact20 joined the soups: 1; the GooAQ arm joined the soups: 1 (the bar: not more than 0.2 below dose3 on either set).

## Freeze

Rule: among dose3-bias015, fact20, the GooAQ arm and the soups, the highest mean of tiny10 and heldout8 with at most 410 dims/doc and at most 16807 postings/query; a lead below 0.1 over dose3-bias015 keeps dose3-bias015.

**Frozen full model: `soup-v20-dqf-bias015`** — soup-v20-dqf: dose3 + faq20 + fact20 leads dose3 (v56) by 0.041 on the mean over the tasks of tiny10, heldout8 and trec-covid (any lead wins: margin 0) within the limits. The label is the first line of `logs/final_full_model.txt`; the model is in `datasets/transfer/release-20261113/full/soup-v20-dqf-bias015`.

| candidate | label | mean of the 19 tasks (the rule) | mean of the two sets | tiny10 | heldout8 | trec-covid | dims/doc | dims/query | postings/query | within the limits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **soup-v20-dqf: dose3 + faq20 + fact20** | `soup-v20-dqf-bias015` | 56.435 | 54.353 | 65.06 | 43.64 | 72.48 | 371.4 | 123.0 | 13945 | yes |
| soup-v20-dqfg: dose3 + faq20 + fact20 + gooaq | `soup-v20-dqfg-bias015` | 56.421 | 54.362 | 65.12 | 43.61 | 71.98 | 375.1 | 123.8 | 14048 | yes |
| soup-v20-dqm: dose3 + faq20 + (dose3 + faq20) | `soup-v20-dqm-bias015` | 56.395 | 54.382 | 65.07 | 43.70 | 71.27 | 377.2 | 126.3 | 14529 | yes |
| dose3 (v56) | `arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015` | 56.394 | 54.297 | 64.96 | 43.63 | 72.82 | 374.7 | 121.9 | 14079 | yes |
| soup-v20-dqg: dose3 + faq20 + gooaq | `soup-v20-dqg-bias015` | 56.374 | 54.326 | 65.05 | 43.60 | 71.78 | 378.6 | 124.9 | 14339 | yes |
| soup-v20-dq: dose3 + faq20 | `soup-v20-dq-bias015` | 56.345 | 54.341 | 65.04 | 43.65 | 71.03 | 374.8 | 124.0 | 14312 | yes |
| dose3 + fact20 (v61) | `arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014` | 56.283 | 54.130 | 64.77 | 43.49 | 73.75 | 379.9 | 128.7 | 15113 | yes |
| dose3 + gooaq (v61b) | `arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016` | 56.156 | 54.153 | 64.82 | 43.49 | 70.87 | 381.2 | 125.9 | 14707 | yes |

## Result

| model | rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 20 k arm (v49) | 64.64 | 43.40 | 69.74 | 376.5 | 384.0 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | dose3 (v56) | 64.96 | 43.63 | 72.82 | 374.7 | 381.5 | 121.9 | 14079 |
| arm-dense-kl-n5kj-mined-faq20-ens400-r2v12-bias015 | faq20 (v58) | 64.84 | 43.54 | 70.39 | 377.9 | 385.8 | 127.6 | 15289 |
| arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12-bias016 | dose3 + faq20 (v60) | 64.93 | 43.60 | 70.79 | 375.5 | 382.8 | 129.4 | 15001 |
| arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12-bias014 | dose3 + fact20 (v61) | 64.77 | 43.49 | 73.75 | 379.9 | 386.0 | 128.7 | 15113 |
| arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12-bias016 | dose3 + gooaq (v61b) | 64.82 | 43.49 | 70.87 | 381.2 | 389.3 | 125.9 | 14707 |
| soup-v20-dq-bias015 | soup-v20-dq (dose3 + faq20) | 65.04 | 43.65 | 71.03 | 374.8 | 382.4 | 124.0 | 14312 |
| **soup-v20-dqf-bias015** | soup-v20-dqf (dose3 + faq20 + fact20) | 65.06 | 43.64 | 72.48 | 371.4 | 378.5 | 123.0 | 13945 |
| soup-v20-dqm-bias015 | soup-v20-dqm (dose3 + faq20 + (dose3 + faq20)) | 65.07 | 43.70 | 71.27 | 377.2 | 384.9 | 126.3 | 14529 |
| soup-v20-dqg-bias015 | soup-v20-dqg (dose3 + faq20 + gooaq) | 65.05 | 43.60 | 71.78 | 378.6 | 386.2 | 124.9 | 14339 |
| soup-v20-dqfg-bias015 | soup-v20-dqfg (dose3 + faq20 + fact20 + gooaq) | 65.12 | 43.61 | 71.98 | 375.1 | 382.2 | 123.8 | 14048 |

## Reading

soup-v20-dqf-bias015 65.06 / 43.64 at 371.4 dims/doc against dose3 (v56) arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 64.96 / 43.63 at 374.7: delta +0.10 / +0.01, trec-covid 72.48 against 72.82 -> the best candidate other than dose3 is within the seed noise of dose3-bias015 on the two sets taken one by one (the freeze itself follows the rule above).

Seed noise of this recipe is 0.13-0.2 (one seed here).

## Task groups (mean delta against 20 k arm (v49); tasks both models have)

| group | tasks | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + gooaq (v61b) | soup-v20-dq (dose3 + faq20) | soup-v20-dqf (dose3 + faq20 + fact20) | soup-v20-dqm (dose3 + faq20 + (dose3 + faq20)) | soup-v20-dqg (dose3 + faq20 + gooaq) | soup-v20-dqfg (dose3 + faq20 + fact20 + gooaq) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FAQ tasks (Allegro FAQ A, B, mfaq) | 3 | +0.42 | +1.04 | +0.63 | +0.17 | +0.28 | +0.94 | +0.80 | +1.07 | +0.93 | +0.99 |
| tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions) | 4 | +0.74 | +0.00 | +0.36 | +0.13 | +0.21 | +0.49 | +0.59 | +0.51 | +0.43 | +0.51 |
| scifact-pl | 1 | +0.68 | -0.58 | +0.40 | +0.57 | -0.76 | +0.41 | +0.70 | +0.41 | +0.12 | +0.44 |
| maupqa-mkqa | 1 | -0.02 | -0.44 | -0.04 | -0.28 | +0.05 | -0.06 | +0.09 | +0.07 | +0.03 | +0.11 |
| MAUPQA tasks | 8 | +0.10 | +0.02 | +0.12 | +0.13 | +0.06 | +0.11 | +0.19 | +0.15 | +0.13 | +0.17 |
| all tasks | 18 | +0.28 | +0.17 | +0.25 | +0.11 | +0.14 | +0.33 | +0.34 | +0.37 | +0.32 | +0.35 |

### tiny10 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + gooaq (v61b) | soup-v20-dq (dose3 + faq20) | soup-v20-dqf (dose3 + faq20 + fact20) | soup-v20-dqm (dose3 + faq20 + (dose3 + faq20)) | soup-v20-dqg (dose3 + faq20 + gooaq) | soup-v20-dqfg (dose3 + faq20 + fact20 + gooaq) | soup-v20-dqf (dose3 + faq20 + fact20) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.81 | 18.88 | 18.79 | 18.88 | 18.89 | 18.84 | 18.86 | 18.89 | 18.86 | 18.87 | 18.90 | +0.01 |
| maupqa-mkqa | 65.62 | 65.60 | 65.19 | 65.58 | 65.34 | 65.67 | 65.57 | 65.71 | 65.70 | 65.65 | 65.74 | +0.11 |
| maupqa-poleval2021-pairs | 83.84 | 83.95 | 84.01 | 83.96 | 84.17 | 84.05 | 83.83 | 84.07 | 83.84 | 83.90 | 84.01 | +0.12 |
| maupqa-templates | 76.31 | 76.57 | 76.54 | 76.67 | 76.78 | 76.64 | 76.65 | 76.72 | 76.70 | 76.68 | 76.73 | +0.15 |
| poleval-2022-test-A-allegro-faq | 77.85 | 78.80 | 79.30 | 78.39 | 77.84 | 77.95 | 79.55 | 78.93 | 79.31 | 79.45 | 79.13 | +0.13 |
| poleval-2022-test-B-allegro-faq | 74.47 | 74.77 | 75.51 | 75.22 | 75.07 | 74.90 | 75.13 | 75.48 | 75.68 | 75.24 | 75.84 | +0.71 |
| arguana-pl | 49.38 | 50.24 | 49.64 | 50.05 | 48.96 | 50.50 | 50.00 | 50.09 | 49.96 | 50.34 | 50.21 | -0.14 |
| nfcorpus-pl | 32.93 | 32.84 | 32.85 | 32.95 | 32.85 | 33.17 | 33.08 | 32.76 | 32.91 | 32.99 | 32.87 | -0.07 |
| scifact-pl | 68.16 | 68.84 | 67.58 | 68.56 | 68.73 | 67.40 | 68.57 | 68.86 | 68.58 | 68.29 | 68.60 | +0.02 |
| gpt-exams | 99.04 | 99.10 | 99.05 | 99.09 | 99.08 | 99.07 | 99.13 | 99.12 | 99.13 | 99.13 | 99.14 | +0.03 |

### heldout8 per task

| task | 20 k arm (v49) | dose3 (v56) | faq20 (v58) | dose3 + faq20 (v60) | dose3 + fact20 (v61) | dose3 + gooaq (v61b) | soup-v20-dq (dose3 + faq20) | soup-v20-dqf (dose3 + faq20 + fact20) | soup-v20-dqm (dose3 + faq20 + (dose3 + faq20)) | soup-v20-dqg (dose3 + faq20 + gooaq) | soup-v20-dqfg (dose3 + faq20 + fact20 + gooaq) | soup-v20-dqf (dose3 + faq20 + fact20) - dose3 (v56) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3.5-cc | 20.51 | 20.54 | 20.60 | 20.60 | 20.49 | 20.56 | 20.59 | 20.57 | 20.60 | 20.60 | 20.58 | +0.03 |
| maupqa-multilingual-NLI | 14.86 | 14.89 | 14.72 | 14.71 | 14.87 | 14.81 | 14.85 | 14.88 | 14.82 | 14.87 | 14.87 | -0.01 |
| mfaq | 63.54 | 63.56 | 64.18 | 64.15 | 63.48 | 63.85 | 63.99 | 63.85 | 64.09 | 63.98 | 63.87 | +0.29 |
| poleval-2022-test-A-legal-questions | 84.35 | 85.06 | 84.38 | 84.25 | 84.52 | 84.40 | 84.42 | 84.58 | 84.44 | 84.20 | 84.33 | -0.48 |
| maupqa-wiki-def | 67.86 | 67.87 | 67.98 | 68.01 | 67.95 | 67.46 | 68.06 | 68.05 | 68.13 | 67.89 | 67.92 | +0.18 |
| scidocs-pl | 14.76 | 14.81 | 14.69 | 14.93 | 14.72 | 14.77 | 14.73 | 14.73 | 14.81 | 14.76 | 14.78 | -0.08 |
| maupqa-1z10 | 49.18 | 49.46 | 49.32 | 49.50 | 49.53 | 49.48 | 49.49 | 49.60 | 49.57 | 49.56 | 49.58 | +0.14 |
| fiqa-pl | 32.17 | 32.88 | 32.48 | 32.64 | 32.36 | 32.57 | 33.03 | 32.87 | 33.12 | 32.94 | 32.92 | -0.01 |

## 23 tasks

23-task mean = (10 x tiny10 + 8 x heldout8 + the five held-out-2 tasks) / 23.

| model | tiny10 | heldout8 | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 | 23 tasks | dims/doc | cost: dims/query | cost: postings/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 64.64 | 43.40 | 87.45 | 92.67 | 79.60 | 19.42 | 69.74 | 69.77 | 58.37 | 376.5 | 122.2 | 14479 |
| arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015 | 64.96 | 43.63 | 87.66 | 92.73 | 79.52 | 19.43 | 72.82 | 70.43 | 58.73 | 374.7 | 121.9 | 14079 |
| **soup-v20-dqf-bias015** | 65.06 | 43.64 | 87.76 | 92.95 | 79.84 | 19.57 | 72.48 | 70.52 | 58.80 | 371.4 | 123.0 | 13945 |

The Reading above compares `soup-v20-dqf-bias015` with dose3-bias015 set by set; the soups are measured at up to 410 dims/doc, dose3-bias015 at 374.7, so the dims/doc and postings/query columns belong to the comparison. Nothing further was started by this queue: no new data arm, no seed soup.

## Queue state (v62)

```
sparsity:soup-v20-dq	ok	546.9	24s
sparsity:soup-v20-dq-bias005	ok	483.9	24s
sparsity:soup-v20-dq-bias010	ok	426.6	24s
sparsity:soup-v20-dq-bias015	ok	374.8	24s
regime:quantized:soup-v20-dq-bias015:tiny10	ok	65.04 10	594s
regime:quantized:soup-v20-dq-bias015:heldout8	ok	43.65 8	2111s
regime:quantized:soup-v20-dq-bias015:heldout2_treccovid	ok	71.03 1	1311s
sparsity:soup-v20-dqf	ok	543.8	22s
sparsity:soup-v20-dqf-bias005	ok	480.6	21s
sparsity:soup-v20-dqf-bias010	ok	423.1	25s
sparsity:soup-v20-dqf-bias015	ok	371.4	23s
regime:quantized:soup-v20-dqf-bias015:tiny10	ok	65.06 10	611s
regime:quantized:soup-v20-dqf-bias015:heldout8	ok	43.64 8	2139s
regime:quantized:soup-v20-dqf-bias015:heldout2_treccovid	ok	72.48 1	1300s
sparsity:soup-v20-dqm	ok	552.1	23s
sparsity:soup-v20-dqm-bias005	ok	488.1	23s
sparsity:soup-v20-dqm-bias010	ok	429.6	22s
sparsity:soup-v20-dqm-bias015	ok	377.2	21s
regime:quantized:soup-v20-dqm-bias015:tiny10	ok	65.07 10	572s
regime:quantized:soup-v20-dqm-bias015:heldout8	ok	43.7 8	2024s
regime:quantized:soup-v20-dqm-bias015:heldout2_treccovid	ok	71.27 1	1293s
sparsity:soup-v20-dqg	ok	553.3	21s
sparsity:soup-v20-dqg-bias005	ok	489.1	22s
sparsity:soup-v20-dqg-bias010	ok	430.8	21s
sparsity:soup-v20-dqg-bias015	ok	378.6	22s
regime:quantized:soup-v20-dqg-bias015:tiny10	ok	65.05 10	576s
regime:quantized:soup-v20-dqg-bias015:heldout8	ok	43.6 8	2016s
regime:quantized:soup-v20-dqg-bias015:heldout2_treccovid	ok	71.78 1	1272s
sparsity:soup-v20-dqfg	ok	549.6	23s
sparsity:soup-v20-dqfg-bias005	ok	485.6	24s
sparsity:soup-v20-dqfg-bias010	ok	427.4	23s
sparsity:soup-v20-dqfg-bias015	ok	375.1	23s
regime:quantized:soup-v20-dqfg-bias015:tiny10	ok	65.12 10	586s
regime:quantized:soup-v20-dqfg-bias015:heldout8	ok	43.61 8	2092s
regime:quantized:soup-v20-dqfg-bias015:heldout2_treccovid	ok	71.98 1	1317s
regime:quantized:soup-v20-dqf-bias015:heldout2_czywiesz	ok	87.76 1	721s
regime:quantized:soup-v20-dqf-bias015:heldout2_gpt35wiki	ok	92.95 1	585s
regime:quantized:soup-v20-dqf-bias015:heldout2_quora	ok	79.84 1	733s
regime:quantized:soup-v20-dqf-bias015:heldout2_poquad	ok	19.57 1	1170s
```
