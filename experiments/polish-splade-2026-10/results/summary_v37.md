# v37 (05.11.2026 10:12, leaderboard regime, nDCG@10): query-cost probe, kl-n1v2 recipe with the query regularizer x4 and x10

Same rows, recipe and teacher as arm-distil-v9b-kl-n1v2 (63.25 / 42.62); query_regularizer_weight 5e-5 -> 2e-4 (x4) and 5e-4 (x10); sparse-gradient head, no checkpointing.

### tiny10

| task | base-splade-wrapped | arm-distil-v9b-kl-n1v2 | arm-distil-v9b-kl-n1v2-qreg4x | arm-distil-v9b-kl-n1v2-qreg10x | soup-v9b |
|---|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 43.32 | 43.41 | 43.20 | 46.28 |
| gpt-exams | 98.63 | 98.68 | 98.68 | 98.70 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.64 | 18.62 | 18.67 | 18.53 |
| maupqa-mkqa | 63.92 | 64.97 | 64.90 | 65.02 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.01 | 83.03 | 82.89 | 83.32 |
| maupqa-templates | 75.02 | 75.98 | 75.97 | 76.02 | 75.80 |
| nfcorpus-pl | 30.07 | 32.26 | 32.30 | 32.30 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 76.23 | 75.97 | 75.98 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.24 | 72.52 | 72.70 | 73.48 |
| scifact-pl | 64.52 | 67.17 | 67.20 | 67.31 | 66.44 |
| **mean** | **61.60** | **63.25** | **63.26** | **63.28** | **63.65** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-distil-v9b-kl-n1v2 | 432.5 | ×0.99 | 629 |  |  |  |  |  |
| arm-distil-v9b-kl-n1v2-qreg4x | 425.0 | ×0.97 | 615 |  |  |  |  |  |
| arm-distil-v9b-kl-n1v2-qreg10x | 432.6 | ×0.99 | 627 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v37)

```
train:arm-distil-v9b-kl-n1v2-qreg4x	ok	-	1237s
sparsity:arm-distil-v9b-kl-n1v2-qreg4x	ok	425.0	23s
regime:quantized:arm-distil-v9b-kl-n1v2-qreg4x:tiny10	ok	63.26 10	636s
train:arm-distil-v9b-kl-n1v2-qreg10x	ok	-	1225s
sparsity:arm-distil-v9b-kl-n1v2-qreg10x	ok	432.6	24s
regime:quantized:arm-distil-v9b-kl-n1v2-qreg10x:tiny10	ok	63.28 10	631s
```
