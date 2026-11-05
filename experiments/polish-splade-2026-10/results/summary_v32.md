# v32 (03.11.2026 23:18, leaderboard regime, nDCG@10): SPLADE-to-SPLADE distillation, teacher soup-v9b, student polish-splade

### tiny10

| task | base-splade-wrapped | arm-n4-3060 | n1-5070 | arm-distil-v9b-kl-n1 | arm-distil-v9b-mmse-n1 | arm-distil-v9b-kl-n4 | arm-distil-v9b-mmse-n4 | soup-v11-distil | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 42.96 | 42.34 | 43.22 | 41.71 | 42.77 | 42.19 | 45.21 | 46.28 |
| gpt-exams | 98.63 | 98.52 | 98.64 | 98.63 | 98.68 | 98.75 | 98.73 | 98.70 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.50 | 18.43 | 18.54 | 18.67 | 18.63 | 18.72 | 18.59 | 18.53 |
| maupqa-mkqa | 63.92 | 63.87 | 64.17 | 64.96 | 65.37 | 65.12 | 65.49 | 65.01 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.03 | 82.88 | 83.28 | 83.66 | 83.39 | 83.42 | 83.34 | 83.32 |
| maupqa-templates | 75.02 | 75.60 | 75.80 | 75.96 | 75.95 | 75.99 | 75.99 | 75.86 | 75.80 |
| nfcorpus-pl | 30.07 | 31.85 | 32.47 | 32.24 | 32.43 | 32.78 | 32.27 | 32.38 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.57 | 75.95 | 76.12 | 75.58 | 76.00 | 76.14 | 76.64 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 71.42 | 72.36 | 72.52 | 72.49 | 72.91 | 73.13 | 73.54 | 73.48 |
| scifact-pl | 64.52 | 67.60 | 66.74 | 66.84 | 66.18 | 66.85 | 67.17 | 66.26 | 66.44 |
| **mean** | **61.60** | **62.89** | **62.98** | **63.23** | **63.07** | **63.32** | **63.32** | **63.55** | **63.65** |

### heldout8

| task | arm-n4-3060 | n1-5070 | arm-distil-v9b-kl-n1 | arm-distil-v9b-mmse-n1 | arm-distil-v9b-kl-n4 | arm-distil-v9b-mmse-n4 | soup-v11-distil | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 29.18 | 29.54 | 30.22 |  | 30.04 | 30.43 | 30.66 | 30.37 |
| maupqa-1z10 | 48.05 | 48.18 | 48.71 | 48.77 | 48.73 | 48.79 | 48.57 | 48.49 |
| maupqa-gpt3.5-cc | 20.07 | 20.15 | 20.25 | 20.39 | 20.35 | 20.38 | 20.27 | 20.26 |
| maupqa-multilingual-NLI | 14.50 | 14.59 | 14.59 |  | 14.66 | 14.65 | 14.58 | 14.53 |
| maupqa-multilingual-nli |  |  |  | 14.69 |  |  |  |  |
| maupqa-wiki-def | 65.43 | 65.94 | 66.27 | 66.59 | 66.15 | 66.57 | 66.44 | 66.35 |
| mfaq | 62.67 | 62.92 | 63.02 | 63.38 | 63.26 | 63.36 | 63.26 | 63.18 |
| poleval-2022-test-A-legal-questions | 82.01 | 81.82 | 82.88 |  | 82.67 | 83.03 | 82.75 | 82.72 |
| poleval-2022-test-a-legal-questions |  |  |  | 82.73 |  |  |  |  |
| scidocs-pl | 14.20 | 14.21 | 14.23 | 14.25 | 14.08 | 14.31 | 14.45 | 14.55 |
| **mean** | **42.01** | **42.17** | **42.52** | **44.40** | **42.49** | **42.69** | **42.62** | **42.56** |


## Active document dimensions (base 436)

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-distil-v9b-kl-n1 | 412.1 | ×0.94 | 588 |  |  |  |  |  |
| arm-distil-v9b-mmse-n1 | 501.2 | ×1.15 | 732 |  |  |  |  |  |
| arm-distil-v9b-kl-n4 | 451.7 | ×1.04 | 654 |  |  |  |  |  |
| arm-distil-v9b-mmse-n4 | 476.8 | ×1.09 | 698 |  |  |  |  |  |
| soup-v11-distil | 423.0 | ×0.97 | 610 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v32)

```
train:arm-distil-v9b-kl-n1	ok	-	3158s
sparsity:arm-distil-v9b-kl-n1	ok	412.1	24s
regime:quantized:arm-distil-v9b-kl-n1:tiny10	ok	63.23 10	630s
regime:quantized:arm-distil-v9b-kl-n1:heldout8	ok	42.52 8	2212s
train:arm-distil-v9b-mmse-n1	ok	-	3188s
sparsity:arm-distil-v9b-mmse-n1	ok	501.2	24s
regime:quantized:arm-distil-v9b-mmse-n1:tiny10	ok	63.07 10	662s
regime:quantized:arm-distil-v9b-mmse-n1:heldout8	ok	44.4 7	2352s
train:arm-distil-v9b-kl-n4	ok	-	3905s
sparsity:arm-distil-v9b-kl-n4	ok	451.7	23s
regime:quantized:arm-distil-v9b-kl-n4:tiny10	ok	63.32 10	635s
regime:quantized:arm-distil-v9b-kl-n4:heldout8	ok	42.49 8	2289s
train:arm-distil-v9b-mmse-n4	ok	-	3933s
sparsity:arm-distil-v9b-mmse-n4	ok	476.8	23s
regime:quantized:arm-distil-v9b-mmse-n4:tiny10	ok	63.32 10	634s
regime:quantized:arm-distil-v9b-mmse-n4:heldout8	ok	42.69 8	3175s
sparsity:soup-v11-distil	ok	423.0	23s
regime:quantized:soup-v11-distil:tiny10	FAILED	-	629s
regime:quantized:soup-v11-distil:tiny10	ok	63.55 10	reconstructed from the PIRB log
regime:quantized:soup-v11-distil:heldout8	ok	42.62 8	2232s
```
