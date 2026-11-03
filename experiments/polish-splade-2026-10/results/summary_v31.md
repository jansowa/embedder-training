# v31 (03.11.2026 14:54, leaderboard regime, nDCG@10): data probe S2 from the MLM, anchors, arm-n4x-nq on nq-pl

S2: 1 875 steps (60 k examples) from sdadas/polish-distilroberta, settings of mlm-path-ep3; only the data differs:
mlm-s2a-msmarco = msmarco judged 38 k rows, mlm-s2b-squadfiqa = squadv2 18 k + fiqa 2 k, mlm-s2c-mix = both (60 k).

### tiny10

| task | base-splade-wrapped | mlm-path-ep3 | mlm-msmarco-ep1 | arm-mlm-s2a-msmarco | arm-mlm-s2b-squadfiqa | arm-mlm-s2c-mix | arm-n4x-nq | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 24.82 | 29.85 | 18.22 | 35.64 | 27.23 | 40.80 | 46.28 |
| gpt-exams | 98.63 | 96.28 | 98.12 | 97.23 | 95.74 | 96.80 | 98.56 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.23 | 18.69 | 18.46 | 17.74 | 18.71 | 18.15 | 18.53 |
| maupqa-mkqa | 63.92 | 53.30 | 61.66 | 55.98 | 45.57 | 55.48 | 62.29 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 78.53 | 80.67 | 78.64 | 75.42 | 78.88 | 83.23 | 83.32 |
| maupqa-templates | 75.02 | 70.37 | 74.16 | 71.10 | 68.78 | 72.73 | 75.80 | 75.80 |
| nfcorpus-pl | 30.07 | 29.55 | 32.51 | 30.67 | 28.63 | 29.41 | 31.50 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 65.87 | 71.36 | 66.03 | 64.50 | 67.09 | 75.45 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 62.24 | 67.80 | 62.94 | 60.94 | 63.78 | 70.81 | 73.48 |
| scifact-pl | 64.52 | 57.09 | 58.11 | 57.27 | 62.03 | 57.14 | 66.62 | 66.44 |
| **mean** | **61.60** | **55.63** | **59.29** | **55.66** | **55.50** | **56.72** | **62.32** | **63.65** |

### heldout8

| task | mlm-path-ep3 | mlm-msmarco-ep1 | arm-n4x-nq | soup-v9b |
|---|---:|---:|---:|---:|
| fiqa-pl |  |  | 28.55 | 30.37 |
| maupqa-1z10 | 44.02 |  | 47.50 | 48.49 |
| maupqa-gpt3.5-cc | 19.69 | 20.14 | 20.02 | 20.26 |
| maupqa-multilingual-NLI |  |  | 14.09 | 14.53 |
| maupqa-multilingual-nli | 12.84 | 13.72 |  |  |
| maupqa-wiki-def | 62.36 | 63.94 | 66.26 | 66.35 |
| mfaq | 58.23 | 62.36 | 61.89 | 63.18 |
| poleval-2022-test-A-legal-questions |  |  | 79.74 | 82.72 |
| poleval-2022-test-a-legal-questions | 80.86 | 82.70 |  |  |
| scidocs-pl | 10.45 |  | 13.88 | 14.55 |
| **mean** | **41.21** | **48.57** | **41.49** | **42.56** |

### nqpl

| task | base-splade-wrapped | arm-n4x-nq | soup-v9b |
|---|---:|---:|---:|
| nq-pl | 40.07 | 43.79 | 41.46 |
| **mean** | **40.07** | **43.79** | **41.46** |


## Active document dimensions (300 PIRB passages per tiny task, base 436)

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| mlm-path-ep3 | 503.3 | ×1.15 | 766 |  |  |  |  |  |
| mlm-msmarco-ep1 | 798.9 | ×1.83 | 1236 |  |  |  |  |  |
| arm-mlm-s2a-msmarco | 599.0 | ×1.37 | 883 |  |  |  |  |  |
| arm-mlm-s2b-squadfiqa | 193.5 | ×0.44 | 287 |  |  |  |  |  |
| arm-mlm-s2c-mix | 563.1 | ×1.29 | 847 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v31)

```
sparsity:mlm-msmarco-ep1	ok	798.9	24s
sparsity:mlm-path-ep3	ok	503.3	24s
train:arm-mlm-s2a-msmarco	ok	-	3899s
sparsity:arm-mlm-s2a-msmarco	ok	599.0	24s
regime:quantized:arm-mlm-s2a-msmarco:tiny10	ok	55.66 10	745s
train:arm-mlm-s2b-squadfiqa	ok	-	6315s
sparsity:arm-mlm-s2b-squadfiqa	ok	193.5	23s
regime:quantized:arm-mlm-s2b-squadfiqa:tiny10	ok	55.5 10	524s
train:arm-mlm-s2c-mix	ok	-	5079s
sparsity:arm-mlm-s2c-mix	ok	563.1	22s
regime:quantized:arm-mlm-s2c-mix:tiny10	ok	56.72 10	715s
regime:quantized:mlm-path-ep3:tiny10	ok	55.63 10	717s
regime:quantized:mlm-msmarco-ep1:tiny10	ok	59.29 10	777s
regime:quantized:arm-n4x-nq:nqpl	ok	43.79 1	13531s
regime:quantized:mlm-path-ep3:heldout8	ok	41.21 7	2166s
regime:quantized:mlm-msmarco-ep1:heldout8	ok	48.57 5	1822s
```
