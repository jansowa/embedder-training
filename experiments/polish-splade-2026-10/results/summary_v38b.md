# v38b (06.11.2026 05:19, leaderboard regime, nDCG@10): PolDense-400M as the dense teacher on the n1 rows

Student sdadas/polish-splade, the kl-n1v2 recipe (4 146 rows: fiqa 2073 + squadv2-v2 2073, 2 epochs, KL T=2, FLOPS kept, 2 x 16, fast head). pd400 = OPI-PIB/PolDense-400M (dense ModernBERT 395 M, 61.8 on our 23 PIRB tasks; leaderboard tiny10 68.8 / heldout8 46.8), the same family and query prefix as pd150 (v38: 60.4; 63.41 / 42.28). Cosines scaled to soup-v9b's units by the within-row least-squares fit (sidecar datasets/arms/fiqa-plus-squadv2v2-tpd400.jsonl.teacher.json). The pd400 score files for the n4 rows (mix3v2-6219) and the n5 rows (fiqa-plus-squadv2v2-18k) are written here for the 5070.

## Teacher agreement with soup-v9b (within-row)

```
fiqa-plus-squadv2v2-tpd150.jsonl.teacher.json: scale 121.2, Pearson 0.833, Kendall tau 0.420, median margin 10.09, rows with neg > pos 151
fiqa-plus-squadv2v2-tpd400.jsonl.teacher.json: scale 106.5, Pearson 0.836, Kendall tau 0.402, median margin 10.73, rows with neg > pos 99
mix3v2-6219-tpd400.jsonl.teacher.json: scale 102.2, Pearson 0.794, Kendall tau 0.387, median margin 8.89, rows with neg > pos 166
fiqa-plus-squadv2v2-18k-tpd400.jsonl.teacher.json: scale 114.5, Pearson 0.842, Kendall tau 0.425, median margin 11.26, rows with neg > pos 472
```

### tiny10

| task | base-splade-wrapped | n1-5070 | arm-distil-v9b-kl-n1v2 | arm-dense-kl-n1v2-pd150 | arm-dense-kl-n1v2-mmlwB | arm-dense-kl-n1v2-ens | arm-dense-kl-n1v2-pd400 | arm-dense-kl-n1v2-ens400 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 42.34 | 43.32 | 47.26 | 43.64 | 45.56 | 48.54 | 45.50 | 46.28 |
| gpt-exams | 98.63 | 98.64 | 98.68 | 98.87 | 98.96 | 98.76 | 98.69 | 98.67 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.43 | 18.64 | 18.38 | 18.11 | 18.66 | 18.34 | 18.59 | 18.53 |
| maupqa-mkqa | 63.92 | 64.17 | 64.97 | 63.72 | 62.91 | 64.76 | 63.91 | 64.69 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 82.88 | 83.01 | 83.40 | 83.61 | 83.40 | 83.29 | 83.26 | 83.32 |
| maupqa-templates | 75.02 | 75.80 | 75.98 | 75.73 | 74.98 | 76.08 | 75.69 | 76.05 | 75.80 |
| nfcorpus-pl | 30.07 | 32.47 | 32.26 | 31.84 | 32.07 | 32.20 | 31.53 | 32.16 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.95 | 76.23 | 76.75 | 76.77 | 75.92 | 76.06 | 75.53 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.36 | 72.24 | 72.66 | 73.00 | 73.07 | 72.67 | 72.72 | 73.48 |
| scifact-pl | 64.52 | 66.74 | 67.17 | 65.44 | 64.99 | 67.24 | 66.23 | 67.10 | 66.44 |
| **mean** | **61.60** | **62.98** | **63.25** | **63.41** | **62.90** | **63.56** | **63.49** | **63.43** | **63.65** |

### heldout8

| task | n1-5070 | arm-distil-v9b-kl-n1v2 | arm-dense-kl-n1v2-pd150 | arm-dense-kl-n1v2-ens | arm-dense-kl-n1v2-pd400 | arm-dense-kl-n1v2-ens400 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 29.54 | 30.22 | 30.03 | 30.28 | 29.26 | 30.20 | 30.37 |
| maupqa-1z10 | 48.18 | 48.63 | 47.29 | 48.40 | 47.59 | 48.45 | 48.49 |
| maupqa-gpt3.5-cc | 20.15 | 20.37 | 20.11 | 20.29 | 20.10 | 20.26 | 20.26 |
| maupqa-multilingual-NLI | 14.59 | 14.64 | 14.47 | 14.69 | 14.54 | 14.68 | 14.53 |
| maupqa-wiki-def | 65.94 | 66.51 | 66.10 | 66.45 | 66.02 | 66.51 | 66.35 |
| mfaq | 62.92 | 63.21 | 63.26 | 63.35 | 63.18 | 63.23 | 63.18 |
| poleval-2022-test-A-legal-questions | 81.82 | 83.04 | 82.67 | 82.82 | 82.65 | 82.82 | 82.72 |
| scidocs-pl | 14.21 | 14.30 | 14.28 | 14.46 | 14.32 | 14.51 | 14.55 |
| **mean** | **42.17** | **42.62** | **42.28** | **42.59** | **42.21** | **42.58** | **42.56** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-distil-v9b-kl-n1v2 | 432.5 | ×0.99 | 629 |  |  |  |  |  |
| arm-dense-kl-n1v2-pd150 | 451.9 | ×1.04 | 637 |  |  |  |  |  |
| arm-dense-kl-n1v2-ens | 446.7 | ×1.02 | 633 |  |  |  |  |  |
| arm-dense-kl-n1v2-pd400 | 448.7 | ×1.03 | 638 |  |  |  |  |  |
| arm-dense-kl-n1v2-ens400 | 456.9 | ×1.05 | 647 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v38b)

```
teacher:pd400	ok	-	442s
train:arm-dense-kl-n1v2-pd400	ok	-	1267s
sparsity:arm-dense-kl-n1v2-pd400	ok	448.7	24s
regime:quantized:arm-dense-kl-n1v2-pd400:tiny10	ok	63.49 10	638s
regime:quantized:arm-dense-kl-n1v2-pd400:heldout8	ok	42.21 8	2406s
train:arm-dense-kl-n1v2-ens400	ok	-	1297s
sparsity:arm-dense-kl-n1v2-ens400	ok	456.9	24s
regime:quantized:arm-dense-kl-n1v2-ens400:tiny10	ok	63.43 10	652s
regime:quantized:arm-dense-kl-n1v2-ens400:heldout8	ok	42.58 8	2373s
teacher:pd400-n4	ok	-	561s
teacher:pd400-n5	ok	-	2083s
```
