# v41 (07.11.2026 16:10, leaderboard regime, nDCG@10): the colleague rows in the mix and in soups, thresholds on heldout8, the bias form, the ALBERT student asymmetric

n5kj = the 23.5-gated n5 rows plus all 2 219 colleague rows, the ens-n5v2 recipe. soup-v13a = uniform soup of soup-v9b, ens-n5v2 and kj-ens400; soup-v13b = the same at 0.4 / 0.4 / 0.2. min_weight 0.05 = terms below 0.05 dropped on both sides at indexing. soup-v12-bias005 = soup-v12 with lm_head.bias shifted by -(e^0.05 - 1), the threshold baked into the checkpoint as a soft threshold. asym-hlfull-q-v9b = queries encoded by the ALBERT student (0.45 of the teacher's cost), index built by soup-v9b.

### tiny10

| task | base-splade-wrapped | arm-dense-kl-kj-ens400 | arm-dense-kl-n5v2-ens400 | arm-dense-kl-n5kj-ens400 | soup-v12 | soup-v13a | soup-v13b | soup-v12-mw005 | soup-v12-bias005 | ens-n5v2-mw005 | asym-hlfull-q-v9b | hl-full-s3-mw005 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 39.43 | 47.39 | 47.73 | 46.80 | 45.01 | 45.93 | 46.89 | 47.40 | 47.44 | 45.17 | 42.87 | 46.28 |
| gpt-exams | 98.63 | 98.74 | 98.78 | 98.84 | 98.76 | 98.88 | 98.85 | 98.77 | 98.77 | 98.77 | 98.68 | 98.48 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.66 | 18.77 | 18.83 | 18.69 | 18.76 | 18.74 | 18.70 | 18.68 | 18.77 | 18.34 | 18.50 | 18.53 |
| maupqa-mkqa | 63.92 | 65.04 | 64.69 | 64.67 | 65.09 | 65.43 | 65.26 | 65.04 | 65.07 | 64.66 | 64.19 | 64.82 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.02 | 83.63 | 83.35 | 83.63 | 83.60 | 83.74 | 83.58 | 83.52 | 83.66 | 83.18 | 82.78 | 83.32 |
| maupqa-templates | 75.02 | 75.47 | 76.14 | 76.15 | 76.07 | 75.99 | 76.04 | 76.07 | 76.02 | 76.12 | 75.60 | 75.55 | 75.80 |
| nfcorpus-pl | 30.07 | 31.52 | 32.54 | 32.49 | 32.44 | 32.52 | 32.36 | 32.42 | 32.46 | 32.51 | 32.32 | 32.29 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.38 | 75.94 | 77.22 | 76.78 | 76.60 | 76.67 | 76.75 | 76.99 | 76.01 | 75.91 | 74.98 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.35 | 73.42 | 74.65 | 73.74 | 73.71 | 73.79 | 73.64 | 73.34 | 73.44 | 73.20 | 72.52 | 73.48 |
| scifact-pl | 64.52 | 65.58 | 68.23 | 67.52 | 67.65 | 67.01 | 67.42 | 67.64 | 68.26 | 68.33 | 66.90 | 66.88 | 66.44 |
| **mean** | **61.60** | **62.52** | **63.95** | **64.14** | **63.96** | **63.75** | **63.88** | **63.95** | **64.05** | **63.97** | **63.35** | **62.97** | **63.65** |

### heldout8

| task | arm-dense-kl-kj-ens400 | arm-dense-kl-n5v2-ens400 | arm-dense-kl-n5kj-ens400 | soup-v12 | soup-v13a | soup-v13b | soup-v12-mw005 | soup-v12-bias005 | ens-n5v2-mw005 | asym-hlfull-q-v9b | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 28.21 | 30.13 | 30.11 | 30.68 | 30.88 | 30.83 | 30.75 | 30.18 | 30.05 | 29.97 | 30.37 |
| maupqa-1z10 | 48.30 | 48.98 | 49.14 | 48.84 | 48.89 | 48.92 | 48.85 | 48.83 | 48.98 | 48.05 | 48.49 |
| maupqa-gpt3.5-cc | 20.35 | 20.40 | 20.42 | 20.39 | 20.45 | 20.44 | 20.39 | 20.39 | 20.40 | 20.12 | 20.26 |
| maupqa-multilingual-NLI | 14.52 | 14.66 | 14.73 | 14.67 | 14.67 | 14.68 | 14.66 | 14.62 | 14.66 | 14.41 | 14.53 |
| maupqa-wiki-def | 65.45 | 67.66 | 67.53 | 67.23 | 66.97 | 67.14 | 67.20 | 67.09 | 67.66 | 65.43 | 66.35 |
| mfaq | 63.09 | 63.07 | 63.24 | 63.23 | 63.31 | 63.29 | 63.22 | 63.21 | 63.07 | 62.98 | 63.18 |
| poleval-2022-test-A-legal-questions | 81.15 | 83.97 | 84.14 | 83.49 | 82.94 | 83.25 | 83.52 | 83.39 | 83.97 | 81.21 | 82.72 |
| scidocs-pl | 13.60 | 14.79 | 14.74 | 14.85 | 14.61 | 14.72 | 14.85 | 14.95 | 14.81 | 14.53 | 14.55 |
| **mean** | **41.84** | **42.96** | **43.01** | **42.92** | **42.84** | **42.91** | **42.93** | **42.83** | **42.95** | **42.09** | **42.56** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-dense-kl-kj-ens400 | 412.5 | ×0.95 | 639 |  |  |  |  |  |
| arm-dense-kl-n5v2-ens400 | 425.4 | ×0.97 | 593 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400 | 422.3 | ×0.97 | 588 |  |  |  |  |  |
| soup-v12 | 417.9 | ×0.96 | 591 |  |  |  |  |  |
| soup-v13a | 417.7 | ×0.96 | 604 |  |  |  |  |  |
| soup-v13b | 417.9 | ×0.96 | 598 |  |  |  |  |  |
| soup-v12-bias005 | 377.8 | ×0.87 | 535 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v41)

```
regime:quantized:ens-n5v2-mw005:tiny10	ok	63.97 10	636s
train:arm-dense-kl-n5kj-ens400	ok	-	5038s
sparsity:arm-dense-kl-n5kj-ens400	ok	422.3	22s
regime:quantized:arm-dense-kl-n5kj-ens400:tiny10	ok	64.14 10	605s
regime:quantized:arm-dense-kl-n5kj-ens400:heldout8	ok	43.01 8	2215s
sparsity:soup-v13a	ok	417.7	22s
regime:quantized:soup-v13a:tiny10	ok	63.75 10	613s
regime:quantized:soup-v13a:heldout8	ok	42.84 8	2194s
sparsity:soup-v13b	ok	417.9	23s
regime:quantized:soup-v13b:tiny10	ok	63.88 10	615s
regime:quantized:soup-v13b:heldout8	ok	42.91 8	2284s
regime:quantized:soup-v12-mw005:tiny10	ok	63.95 10	636s
regime:quantized:soup-v12-mw005:heldout8	ok	42.93 8	2318s
regime:quantized:ens-n5v2-mw005:heldout8	ok	42.95 8	2388s
sparsity:soup-v12-bias005	ok	377.8	23s
regime:quantized:soup-v12-bias005:tiny10	ok	64.05 10	614s
regime:quantized:soup-v12-bias005:heldout8	ok	42.83 8	2151s
regime:quantized:asym-hlfull-q-v9b:tiny10	ok	63.35 10	648s
regime:quantized:asym-hlfull-q-v9b:heldout8	ok	42.09 8	2362s
regime:quantized:hl-full-s3-mw005:tiny10	ok	62.97 10	597s
```
