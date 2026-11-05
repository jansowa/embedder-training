# v38 (05.11.2026 22:04, leaderboard regime, nDCG@10): distillation from dense teachers on the n1 rows

Student sdadas/polish-splade, the kl-n1v2 recipe (4 146 rows: fiqa 2073 + squadv2-v2 2073, 2 epochs, KL T=2, FLOPS kept, 2 x 16, fast head). Teachers: pd150 = OPI-PIB/PolDense-150M (dense, 60.4 on our 23 PIRB tasks), mmlwB = sdadas/mmlw-retrieval-roberta-base (57.8, parity with soup-v9b), ens = mean of soup-v9b and PolDense per pair. Dense cosines were put on the SPLADE teacher's scale by a within-row least-squares fit against the -tv9b file (sidecars datasets/arms/*.teacher.json: scale, agreement with soup-v9b, margins).

## Teacher agreement with soup-v9b (within-row, over the 4 146 rows)

```
pd150: scale 121.2, Pearson 0.833, Kendall tau 0.420, median margin 10.09, rows with neg > pos 151
mmlwB: scale 144.9, Pearson 0.842, Kendall tau 0.459, median margin 9.78, rows with neg > pos 233
```

### tiny10

| task | base-splade-wrapped | n1-5070 | arm-distil-v9b-kl-n1v2 | arm-dense-kl-n1v2-pd150 | arm-dense-kl-n1v2-mmlwB | arm-dense-kl-n1v2-ens | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 42.34 | 43.32 | 47.26 | 43.64 | 45.56 | 46.28 |
| gpt-exams | 98.63 | 98.64 | 98.68 | 98.87 | 98.96 | 98.76 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.43 | 18.64 | 18.38 | 18.11 | 18.66 | 18.53 |
| maupqa-mkqa | 63.92 | 64.17 | 64.97 | 63.72 | 62.91 | 64.76 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 82.88 | 83.01 | 83.40 | 83.61 | 83.40 | 83.32 |
| maupqa-templates | 75.02 | 75.80 | 75.98 | 75.73 | 74.98 | 76.08 | 75.80 |
| nfcorpus-pl | 30.07 | 32.47 | 32.26 | 31.84 | 32.07 | 32.20 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.95 | 76.23 | 76.75 | 76.77 | 75.92 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.36 | 72.24 | 72.66 | 73.00 | 73.07 | 73.48 |
| scifact-pl | 64.52 | 66.74 | 67.17 | 65.44 | 64.99 | 67.24 | 66.44 |
| **mean** | **61.60** | **62.98** | **63.25** | **63.41** | **62.90** | **63.56** | **63.65** |

### heldout8

| task | n1-5070 | arm-distil-v9b-kl-n1v2 | arm-dense-kl-n1v2-pd150 | arm-dense-kl-n1v2-ens | soup-v9b |
|---|---:|---:|---:|---:|---:|
| fiqa-pl | 29.54 | 30.22 | 30.03 | 30.28 | 30.37 |
| maupqa-1z10 | 48.18 | 48.63 | 47.29 | 48.40 | 48.49 |
| maupqa-gpt3.5-cc | 20.15 | 20.37 | 20.11 | 20.29 | 20.26 |
| maupqa-multilingual-NLI | 14.59 | 14.64 | 14.47 | 14.69 | 14.53 |
| maupqa-wiki-def | 65.94 | 66.51 | 66.10 | 66.45 | 66.35 |
| mfaq | 62.92 | 63.21 | 63.26 | 63.35 | 63.18 |
| poleval-2022-test-A-legal-questions | 81.82 | 83.04 | 82.67 | 82.82 | 82.72 |
| scidocs-pl | 14.21 | 14.30 | 14.28 | 14.46 | 14.55 |
| **mean** | **42.17** | **42.62** | **42.28** | **42.59** | **42.56** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-distil-v9b-kl-n1v2 | 432.5 | ×0.99 | 629 |  |  |  |  |  |
| arm-dense-kl-n1v2-pd150 | 451.9 | ×1.04 | 637 |  |  |  |  |  |
| arm-dense-kl-n1v2-mmlwB | 404.3 | ×0.93 | 578 |  |  |  |  |  |
| arm-dense-kl-n1v2-ens | 446.7 | ×1.02 | 633 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v38)

```
teacher:pd150	ok	-	191s
teacher:mmlwB	ok	-	83s
train:arm-dense-kl-n1v2-pd150	FAILED	-	1222s
train:arm-dense-kl-n1v2-pd150	ok	-	1222s
sparsity:arm-dense-kl-n1v2-pd150	ok	451.9	23s
regime:quantized:arm-dense-kl-n1v2-pd150:tiny10	ok	63.41 10	624s
regime:quantized:arm-dense-kl-n1v2-pd150:heldout8	ok	42.28 8	2374s
train:arm-dense-kl-n1v2-mmlwB	ok	-	1257s
sparsity:arm-dense-kl-n1v2-mmlwB	ok	404.3	23s
regime:quantized:arm-dense-kl-n1v2-mmlwB:tiny10	ok	62.9 10	613s
train:arm-dense-kl-n1v2-ens	ok	-	1255s
sparsity:arm-dense-kl-n1v2-ens	ok	446.7	24s
regime:quantized:arm-dense-kl-n1v2-ens:tiny10	ok	63.56 10	634s
regime:quantized:arm-dense-kl-n1v2-ens:heldout8	ok	42.59 8	2285s
```
