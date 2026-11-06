# v39 (06.11.2026 12:28, leaderboard regime, nDCG@10): the ensemble teacher on the n4 and n5 rows

Student sdadas/polish-splade, the kl-n4v2 / kl-n5v2 recipes (2 epochs, KL T=2, FLOPS kept, 2 x 16, 512 tokens, 10 negatives, fast head). Teacher = per-pair mean of soup-v9b's dot products and OPI-PIB/PolDense-400M's cosines scaled to soup-v9b's units (sidecars datasets/arms/*-tpd400.jsonl.teacher.json: n4 scale 102.2, Pearson 0.79; n5 scale 114.5, Pearson 0.84). On the n1 rows (v38 / v38b) the ensemble gave 63.56 / 42.59 (pd150) and 63.43 / 42.58 (pd400) against 63.25 / 42.62 for soup-v9b alone and 63.4-63.5 / 42.2-42.3 for the dense teacher alone.

### tiny10

| task | base-splade-wrapped | arm-distil-v9b-kl-n4v2 | arm-dense-kl-n4v2-ens400 | arm-distil-v9b-kl-n5v2 | arm-dense-kl-n5v2-ens400 | arm-dense-kl-n1v2-ens400 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 41.44 | 43.08 | 44.80 | 47.39 | 45.50 | 46.28 |
| gpt-exams | 98.63 | 98.64 | 98.75 | 98.71 | 98.78 | 98.67 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.65 | 18.64 | 18.63 | 18.77 | 18.59 | 18.53 |
| maupqa-mkqa | 63.92 | 65.18 | 65.13 | 64.74 | 64.69 | 64.69 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.43 | 83.69 | 83.37 | 83.63 | 83.26 | 83.32 |
| maupqa-templates | 75.02 | 76.12 | 76.07 | 76.00 | 76.14 | 76.05 | 75.80 |
| nfcorpus-pl | 30.07 | 32.52 | 32.43 | 32.14 | 32.54 | 32.16 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.76 | 76.13 | 75.26 | 75.94 | 75.53 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 73.68 | 73.67 | 72.46 | 73.42 | 72.72 | 73.48 |
| scifact-pl | 64.52 | 67.33 | 66.83 | 67.75 | 68.23 | 67.10 | 66.44 |
| **mean** | **61.60** | **63.27** | **63.44** | **63.39** | **63.95** | **63.43** | **63.65** |

### heldout8

| task | arm-distil-v9b-kl-n4v2 | arm-dense-kl-n4v2-ens400 | arm-distil-v9b-kl-n5v2 | arm-dense-kl-n5v2-ens400 | arm-dense-kl-n1v2-ens400 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 30.56 | 30.01 | 30.29 | 30.13 | 30.20 | 30.37 |
| maupqa-1z10 | 48.71 | 48.41 | 49.09 | 48.98 | 48.45 | 48.49 |
| maupqa-gpt3.5-cc | 20.35 | 20.25 | 20.33 | 20.40 | 20.26 | 20.26 |
| maupqa-multilingual-NLI | 14.69 | 14.71 | 14.59 | 14.66 | 14.68 | 14.53 |
| maupqa-wiki-def | 65.96 | 66.10 | 66.97 | 67.66 | 66.51 | 66.35 |
| mfaq | 63.26 | 63.17 | 63.12 | 63.07 | 63.23 | 63.18 |
| poleval-2022-test-A-legal-questions | 83.31 | 83.17 | 83.37 | 83.97 | 82.82 | 82.72 |
| scidocs-pl | 14.20 | 14.46 | 14.66 | 14.79 | 14.51 | 14.55 |
| **mean** | **42.63** | **42.53** | **42.80** | **42.96** | **42.58** | **42.56** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-distil-v9b-kl-n4v2 | 459.4 | ×1.05 | 666 |  |  |  |  |  |
| arm-dense-kl-n4v2-ens400 | 481.1 | ×1.10 | 700 |  |  |  |  |  |
| arm-distil-v9b-kl-n5v2 | 381.7 | ×0.87 | 548 |  |  |  |  |  |
| arm-dense-kl-n5v2-ens400 | 425.4 | ×0.97 | 593 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v39)

```
train:arm-dense-kl-n4v2-ens400	ok	-	1664s
sparsity:arm-dense-kl-n4v2-ens400	ok	481.1	23s
regime:quantized:arm-dense-kl-n4v2-ens400:tiny10	ok	63.44 10	662s
regime:quantized:arm-dense-kl-n4v2-ens400:heldout8	ok	42.53 8	2342s
train:arm-dense-kl-n5v2-ens400	ok	-	4670s
sparsity:arm-dense-kl-n5v2-ens400	ok	425.4	23s
regime:quantized:arm-dense-kl-n5v2-ens400:tiny10	ok	63.95 10	640s
regime:quantized:arm-dense-kl-n5v2-ens400:heldout8	ok	42.96 8	2287s
```
