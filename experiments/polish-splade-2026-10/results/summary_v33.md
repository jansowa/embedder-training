# v33 (05.11.2026 02:49, leaderboard regime, nDCG@10): better SQuAD-v2 translation (A/B) and SPLADE-to-SPLADE distillation on more rows

Paired files: *v2* = the same rows, order, ids and scores as the old arm, only the SQuAD texts come from the better translation. Written by queue v35, which took over before v33's nq-pl step (2 x 3.8 h for one task each).

### tiny10

| task | base-splade-wrapped | n3-5070 | arm-n3v2-squad8k | arm-n5-squad16k | arm-n5v2-squad16k | arm-distil-v9b-kl-n1 | arm-distil-v9b-kl-n1v2 | arm-distil-en-kl-n1v2 | arm-distil-en-kl-n1v2-raw | arm-distil-v9b-mmse-n1 | arm-distil-v9b-kl-n4 | arm-distil-v9b-kl-n4v2 | arm-n4x-nq | arm-distil-v9b-kl-n4xnq | arm-distil-v9b-kl-n5v2 | soup-v11-distil | soup-v11-kl | soup-v11-v2 | soup-v11-nq | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 44.29 | 43.42 | 44.13 | 43.94 | 43.22 | 43.32 | 43.61 | 44.19 | 41.71 | 42.77 | 41.44 | 40.80 | 42.86 | 44.80 | 45.21 | 46.44 | 46.41 | 45.71 | 46.13 | 46.28 |
| gpt-exams | 98.63 | 98.35 | 98.24 | 97.53 | 97.82 | 98.63 | 98.68 | 98.78 | 98.82 | 98.68 | 98.75 | 98.64 | 98.56 | 98.70 | 98.71 | 98.70 | 98.69 | 98.72 | 98.73 | 98.77 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.44 | 18.18 | 18.25 | 18.14 | 18.54 | 18.64 | 18.05 | 18.21 | 18.67 | 18.63 | 18.65 | 18.15 | 18.63 | 18.63 | 18.59 | 18.55 | 18.56 | 18.59 | 18.57 | 18.53 |
| maupqa-mkqa | 63.92 | 63.29 | 62.26 | 58.61 | 59.51 | 64.96 | 64.97 | 63.22 | 63.79 | 65.37 | 65.12 | 65.18 | 62.29 | 65.15 | 64.74 | 65.01 | 64.86 | 64.74 | 64.94 | 64.99 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 82.66 | 82.43 | 80.15 | 80.59 | 83.28 | 83.01 | 83.55 | 83.76 | 83.66 | 83.39 | 83.43 | 83.23 | 83.59 | 83.37 | 83.34 | 83.65 | 83.49 | 83.65 | 83.38 | 83.32 |
| maupqa-templates | 75.02 | 75.24 | 74.48 | 72.91 | 73.27 | 75.96 | 75.98 | 74.97 | 75.65 | 75.95 | 75.99 | 76.12 | 75.80 | 76.18 | 76.00 | 75.86 | 75.79 | 75.86 | 75.98 | 76.00 | 75.80 |
| nfcorpus-pl | 30.07 | 32.03 | 31.61 | 30.79 | 30.33 | 32.24 | 32.26 | 32.41 | 32.23 | 32.43 | 32.78 | 32.52 | 31.50 | 32.34 | 32.14 | 32.38 | 32.70 | 32.30 | 32.37 | 32.55 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 74.32 | 73.55 | 71.32 | 70.62 | 76.12 | 76.23 | 74.89 | 74.50 | 75.58 | 76.00 | 75.76 | 75.45 | 76.40 | 75.26 | 76.64 | 76.14 | 76.26 | 76.45 | 76.05 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.21 | 71.19 | 69.71 | 69.13 | 72.52 | 72.24 | 71.59 | 71.26 | 72.49 | 72.91 | 73.68 | 70.81 | 72.81 | 72.46 | 73.54 | 73.70 | 73.62 | 73.80 | 73.40 | 73.48 |
| scifact-pl | 64.52 | 66.36 | 65.82 | 66.44 | 66.25 | 66.84 | 67.17 | 67.71 | 66.15 | 66.18 | 66.85 | 67.33 | 66.62 | 67.41 | 67.75 | 66.26 | 66.55 | 66.44 | 66.77 | 66.55 | 66.44 |
| **mean** | **61.60** | **62.72** | **62.12** | **60.99** | **60.96** | **63.23** | **63.25** | **62.88** | **62.86** | **63.07** | **63.32** | **63.27** | **62.32** | **63.41** | **63.39** | **63.55** | **63.71** | **63.64** | **63.70** | **63.64** | **63.65** |

### heldout8

| task | n3-5070 | arm-n3v2-squad8k | arm-n5-squad16k | arm-distil-v9b-kl-n1 | arm-distil-v9b-kl-n1v2 | arm-distil-en-kl-n1v2 | arm-distil-en-kl-n1v2-raw | arm-distil-v9b-mmse-n1 | arm-distil-v9b-kl-n4 | arm-distil-v9b-kl-n4v2 | arm-n4x-nq | arm-distil-v9b-kl-n4xnq | arm-distil-v9b-kl-n5v2 | soup-v11-distil | soup-v11-kl | soup-v11-v2 | soup-v11-nq | soup-v11-n5 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 28.18 | 27.38 | 25.50 | 30.22 | 30.22 | 29.74 | 29.31 | 30.78 | 30.04 | 30.56 | 28.55 | 29.84 | 30.29 | 30.66 | 30.30 | 30.43 | 30.47 | 30.48 | 30.37 |
| maupqa-1z10 | 48.79 | 48.31 | 47.48 | 48.71 | 48.63 | 47.87 | 48.12 | 48.77 | 48.73 | 48.71 | 47.50 | 48.87 | 49.09 | 48.57 | 48.58 | 48.58 | 48.67 | 48.54 | 48.49 |
| maupqa-gpt3.5-cc | 20.24 | 19.93 | 19.89 | 20.25 | 20.37 | 19.94 | 20.02 | 20.39 | 20.35 | 20.35 | 20.02 | 20.41 | 20.33 | 20.27 | 20.28 | 20.31 | 20.33 | 20.29 | 20.26 |
| maupqa-multilingual-NLI | 14.00 | 13.42 | 11.76 | 14.59 | 14.64 | 14.49 | 14.63 | 14.69 | 14.66 | 14.69 | 14.09 | 14.66 | 14.59 | 14.58 | 14.51 | 14.52 | 14.56 | 14.61 | 14.53 |
| maupqa-wiki-def | 65.90 | 65.87 | 64.31 | 66.27 | 66.51 | 66.23 | 66.19 | 66.59 | 66.15 | 65.96 | 66.26 | 66.90 | 66.97 | 66.44 | 66.12 | 66.19 | 66.36 | 66.42 | 66.35 |
| mfaq | 62.35 | 61.74 | 59.69 | 63.02 | 63.21 | 62.84 | 63.19 | 63.38 | 63.26 | 63.26 | 61.89 | 63.15 | 63.12 | 63.26 | 63.19 | 63.25 | 63.26 | 63.36 | 63.18 |
| poleval-2022-test-A-legal-questions | 81.94 | 81.85 | 81.63 | 82.88 | 83.04 | 82.32 | 82.47 | 82.73 | 82.67 | 83.31 | 79.74 | 82.76 | 83.37 | 82.75 | 82.72 | 82.71 | 83.01 | 83.23 | 82.72 |
| scidocs-pl | 14.11 | 14.03 | 13.73 | 14.23 | 14.30 | 14.81 | 14.63 | 14.25 | 14.08 | 14.20 | 13.88 | 14.35 | 14.66 | 14.45 | 14.47 | 14.55 | 14.51 | 14.69 | 14.55 |
| **mean** | **41.94** | **41.57** | **40.50** | **42.52** | **42.62** | **42.28** | **42.32** | **42.70** | **42.49** | **42.63** | **41.49** | **42.62** | **42.80** | **42.62** | **42.52** | **42.57** | **42.64** | **42.71** | **42.56** |


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


## Active document dimensions (base 436)

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-n3v2-squad8k | 270.9 | ×0.62 | 375 |  |  |  |  |  |
| arm-n5v2-squad16k | 255.6 | ×0.59 | 357 |  |  |  |  |  |
| arm-distil-v9b-kl-n1v2 | 432.5 | ×0.99 | 629 |  |  |  |  |  |
| arm-distil-en-kl-n1v2 | 394.8 | ×0.90 | 573 |  |  |  |  |  |
| arm-distil-en-kl-n1v2-raw | 456.2 | ×1.05 | 647 |  |  |  |  |  |
| arm-distil-v9b-kl-n4v2 | 459.4 | ×1.05 | 666 |  |  |  |  |  |
| arm-distil-v9b-kl-n4xnq | 425.7 | ×0.98 | 616 |  |  |  |  |  |
| arm-distil-v9b-kl-n5v2 | 381.7 | ×0.87 | 548 |  |  |  |  |  |
| soup-v9b | 408.7 | ×0.94 | 589 |  |  |  |  |  |
| soup-v11-kl | 394.4 | ×0.90 | 566 |  |  |  |  |  |
| soup-v11-v2 | 400.3 | ×0.92 | 573 |  |  |  |  |  |
| soup-v11-nq | 404.8 | ×0.93 | 582 |  |  |  |  |  |
| soup-v11-n5 | 411.9 | ×0.94 | 596 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v33)

```
tscore:fiqa-plus-squadv2v2-18k-tv9b.jsonl	ok	-	486s
train:arm-distil-v9b-kl-n1v2	ok	-	3155s
sparsity:arm-distil-v9b-kl-n1v2	ok	432.5	24s
regime:quantized:arm-distil-v9b-kl-n1v2:tiny10	ok	63.25 10	637s
regime:quantized:arm-distil-v9b-kl-n1v2:heldout8	ok	42.62 8	2271s
tscore:fiqa-plus-squadv2v2-ten.jsonl	ok	-	82s
train:arm-distil-en-kl-n1v2	ok	-	3145s
sparsity:arm-distil-en-kl-n1v2	ok	394.8	23s
regime:quantized:arm-distil-en-kl-n1v2:tiny10	ok	62.88 10	584s
regime:quantized:arm-distil-en-kl-n1v2:heldout8	ok	42.28 8	2158s
train:arm-distil-en-kl-n1v2-raw	ok	-	3155s
sparsity:arm-distil-en-kl-n1v2-raw	ok	456.2	24s
regime:quantized:arm-distil-en-kl-n1v2-raw:tiny10	ok	62.86 10	660s
regime:quantized:arm-distil-en-kl-n1v2-raw:heldout8	ok	42.32 8	2382s
train:arm-n3v2-squad8k	ok	-	6797s
sparsity:arm-n3v2-squad8k	ok	270.9	24s
regime:quantized:arm-n3v2-squad8k:tiny10	ok	62.12 10	566s
regime:quantized:arm-n3v2-squad8k:heldout8	ok	41.57 8	2035s
train:arm-distil-v9b-kl-n4xnq	ok	-	3999s
sparsity:arm-distil-v9b-kl-n4xnq	ok	425.7	23s
regime:quantized:arm-distil-v9b-kl-n4xnq:tiny10	ok	63.41 10	630s
regime:quantized:arm-distil-v9b-kl-n4xnq:heldout8	ok	42.62 8	2214s
train:arm-distil-v9b-kl-n4v2	ok	-	3850s
sparsity:arm-distil-v9b-kl-n4v2	ok	459.4	23s
regime:quantized:arm-distil-v9b-kl-n4v2:tiny10	ok	63.27 10	646s
regime:quantized:arm-distil-v9b-kl-n4v2:heldout8	ok	42.63 8	2246s
train:arm-n5v2-squad16k	ok	-	11539s
sparsity:arm-n5v2-squad16k	ok	255.6	24s
regime:quantized:arm-n5v2-squad16k:tiny10	ok	60.96 10	565s
train:arm-distil-v9b-kl-n5v2	ok	-	11571s
sparsity:arm-distil-v9b-kl-n5v2	ok	381.7	23s
regime:quantized:arm-distil-v9b-kl-n5v2:tiny10	ok	63.39 10	615s
regime:quantized:arm-distil-v9b-kl-n5v2:heldout8	ok	42.8 8	2150s
sparsity:soup-v11-kl	ok	394.4	23s
regime:quantized:soup-v11-kl:tiny10	ok	63.71 10	614s
regime:quantized:soup-v11-kl:heldout8	ok	42.52 8	2194s
sparsity:soup-v11-v2	ok	400.3	23s
regime:quantized:soup-v11-v2:tiny10	ok	63.64 10	629s
regime:quantized:soup-v11-v2:heldout8	ok	42.57 8	2248s
regime:quantized:soup-v11-v2:heldout2_czywiesz	ok	86.88 1	803s
regime:quantized:soup-v11-v2:heldout2_gpt35wiki	ok	92.02 1	676s
regime:quantized:soup-v11-v2:heldout2_poquad	ok	19.2 1	1339s
regime:quantized:soup-v11-v2:heldout2_quora	ok	79.35 1	861s
regime:quantized:soup-v11-v2:heldout2_treccovid	ok	69.15 1	1390s
sparsity:soup-v11-nq	ok	404.8	25s
regime:quantized:soup-v11-nq:tiny10	ok	63.7 10	707s
regime:quantized:soup-v11-nq:heldout8	ok	42.64 8	2519s
regime:quantized:soup-v11-nq:heldout2_czywiesz	ok	86.97 1	806s
regime:quantized:soup-v11-nq:heldout2_gpt35wiki	ok	92.15 1	663s
regime:quantized:soup-v11-nq:heldout2_poquad	ok	19.37 1	1353s
regime:quantized:soup-v11-nq:heldout2_quora	ok	79.32 1	864s
regime:quantized:soup-v11-nq:heldout2_treccovid	ok	69.35 1	1414s
sparsity:soup-v11-n5	ok	411.9	24s
regime:quantized:soup-v11-n5:tiny10	ok	63.64 10	628s
regime:quantized:soup-v11-n5:heldout8	ok	42.71 8	2247s
regime:quantized:soup-v11-n5:heldout2_czywiesz	ok	86.93 1	831s
regime:quantized:soup-v11-n5:heldout2_gpt35wiki	ok	92.11 1	686s
regime:quantized:soup-v11-n5:heldout2_poquad	ok	19.22 1	1389s
regime:quantized:soup-v11-n5:heldout2_quora	ok	79.39 1	862s
regime:quantized:soup-v11-n5:heldout2_treccovid	ok	69.45 1	1406s
```
