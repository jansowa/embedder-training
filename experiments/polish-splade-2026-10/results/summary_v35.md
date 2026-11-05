# v35 (05.11.2026 06:25, leaderboard regime, nDCG@10): fresh rows instead of a second epoch in SPLADE distillation; index cost of the v11 soups

arm-distil-v9b-kl-n6-fresh38k = KL from soup-v9b on the kl-n5v2 rows (18 k) plus msmarco-clean-20000, one epoch: the same number of steps as kl-n5v2 (18 k x 2 epochs), every example seen once.

### tiny10

| task | base-splade-wrapped | arm-distil-v9b-kl-n4v2 | arm-distil-v9b-kl-n5v2 | arm-distil-v9b-kl-n6-fresh38k | soup-v11-v2 | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 41.44 | 44.80 | 44.21 | 46.41 | 46.13 | 46.28 |
| gpt-exams | 98.63 | 98.64 | 98.71 | 98.58 | 98.72 | 98.77 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.65 | 18.63 | 18.63 | 18.56 | 18.57 | 18.53 |
| maupqa-mkqa | 63.92 | 65.18 | 64.74 | 65.29 | 64.74 | 64.99 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.43 | 83.37 | 83.63 | 83.49 | 83.38 | 83.32 |
| maupqa-templates | 75.02 | 76.12 | 76.00 | 75.99 | 75.86 | 76.00 | 75.80 |
| nfcorpus-pl | 30.07 | 32.52 | 32.14 | 32.51 | 32.30 | 32.55 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.76 | 75.26 | 75.79 | 76.26 | 76.05 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 73.68 | 72.46 | 73.12 | 73.62 | 73.40 | 73.48 |
| scifact-pl | 64.52 | 67.33 | 67.75 | 66.42 | 66.44 | 66.55 | 66.44 |
| **mean** | **61.60** | **63.27** | **63.39** | **63.42** | **63.64** | **63.64** | **63.65** |

### heldout8

| task | arm-distil-v9b-kl-n4v2 | arm-distil-v9b-kl-n5v2 | arm-distil-v9b-kl-n6-fresh38k | soup-v11-v2 | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 30.56 | 30.29 | 30.23 | 30.43 | 30.48 | 30.37 |
| maupqa-1z10 | 48.71 | 49.09 | 48.61 | 48.58 | 48.54 | 48.49 |
| maupqa-gpt3.5-cc | 20.35 | 20.33 | 20.35 | 20.31 | 20.29 | 20.26 |
| maupqa-multilingual-NLI | 14.69 | 14.59 | 14.58 | 14.52 | 14.61 | 14.53 |
| maupqa-wiki-def | 65.96 | 66.97 | 66.58 | 66.19 | 66.42 | 66.35 |
| mfaq | 63.26 | 63.12 | 63.10 | 63.25 | 63.36 | 63.18 |
| poleval-2022-test-A-legal-questions | 83.31 | 83.37 | 82.91 | 82.71 | 83.23 | 82.72 |
| scidocs-pl | 14.20 | 14.66 | 14.45 | 14.55 | 14.69 | 14.55 |
| **mean** | **42.63** | **42.80** | **42.60** | **42.57** | **42.71** | **42.56** |


## Held-out-2 (czywiesz, gpt35wiki, poquad, quora, treccovid)

### heldout2_czywiesz

| task | soup-v11-v2 | soup-v11-nq | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|
| maupqa-czy-wiesz-v2 | 86.88 | 86.97 | 86.93 | 86.88 |
| **mean** | **86.88** | **86.97** | **86.93** | **86.88** |

### heldout2_gpt35wiki

| task | soup-v11-v2 | soup-v11-nq | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|
| maupqa-gpt3.5-wiki | 92.02 | 92.15 | 92.11 | 91.97 |
| **mean** | **92.02** | **92.15** | **92.11** | **91.97** |

### heldout2_poquad

| task | soup-v11-v2 | soup-v11-nq | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|
| maupqa-poquad | 19.20 | 19.37 | 19.22 | 19.25 |
| **mean** | **19.20** | **19.37** | **19.22** | **19.25** |

### heldout2_quora

| task | soup-v11-v2 | soup-v11-nq | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|
| quora-pl | 79.35 | 79.32 | 79.39 | 79.32 |
| **mean** | **79.35** | **79.32** | **79.39** | **79.32** |

### heldout2_treccovid

| task | soup-v11-v2 | soup-v11-nq | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|
| trec-covid-pl | 69.15 | 69.35 | 69.45 | 69.66 |
| **mean** | **69.15** | **69.35** | **69.45** | **69.66** |


## Active document dimensions (base 436) and index cost

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| soup-v9b | 408.7 | ×0.94 | 589 | 417.6 | 151.4 | ×1.09 | 17805 | ×1.08 |
| soup-v11-kl | 394.4 | ×0.90 | 566 | 402.8 | 147.6 | ×1.07 | 16832 | ×1.03 |
| soup-v11-v2 | 400.3 | ×0.92 | 573 | 408.9 | 148.6 | ×1.07 | 17122 | ×1.04 |
| soup-v11-nq | 404.8 | ×0.93 | 582 | 413.7 | 149.1 | ×1.08 | 17269 | ×1.05 |
| soup-v11-n5 | 411.9 | ×0.94 | 596 | 421.5 | 150.8 | ×1.09 | 18338 | ×1.12 |
| soup-v11-n6 |  |  |  |  |  |  |  |  |
| arm-distil-v9b-kl-n5v2 | 381.7 | ×0.87 | 548 |  |  |  |  |  |
| arm-distil-v9b-kl-n6-fresh38k | 433.2 | ×0.99 | 631 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań), index_cost_v11.json (1350 pasaży i tyle samo zapytań).

## Queue state (v35)

```
takeover:v33	ok	-	killed before nq-pl by v35
cost:v11	ok	-	72s
tscore:msmarco-clean-20000-tv9b.jsonl	ok	-	295s
train:arm-distil-v9b-kl-n6-fresh38k	ok	-	9279s
sparsity:arm-distil-v9b-kl-n6-fresh38k	ok	433.2	24s
regime:quantized:arm-distil-v9b-kl-n6-fresh38k:tiny10	ok	63.42 10	644s
regime:quantized:arm-distil-v9b-kl-n6-fresh38k:heldout8	ok	42.6 8	2536s
```
