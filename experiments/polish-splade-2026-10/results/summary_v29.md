# v28 + v29 (02.11.2026 18:20, leaderboard regime, nDCG@10)

Leaderboard (polish-splade 2023 / best standalone): maupqa-mqa 18.78 / 26.09,
poleval-2022-test-B-legal-questions 78.39 / 84.73, nq-pl 40.42 / 61.14.

### mqa

| task | base-splade-wrapped | soup-v9b |
|---|---:|---:|
| maupqa-mqa | 18.68 | 19.39 |
| **mean** | **18.68** | **19.39** |

### legalB

| task | base-splade-wrapped | soup-v9b |
|---|---:|---:|
| poleval-2022-test-B-legal-questions | 78.67 | 82.23 |
| **mean** | **78.67** | **82.23** |

### nqpl

| task | base-splade-wrapped | soup-v9b |
|---|---:|---:|
| nq-pl | 40.07 | 41.46 |
| **mean** | **40.07** | **41.46** |

### tiny10

| task | base-splade-wrapped | soup-v9b | arm-n4x-nq | arm-n4-3060 | arm-n4-msclean | arm-n4-3060-seed1 | arm-n4-3060-seed2 | soup-n4seeds | arm-n4-msrand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 46.28 | 40.80 | 42.96 | 41.84 | 41.06 | 42.67 | 42.21 | 41.67 |
| gpt-exams | 98.63 | 98.71 | 98.56 | 98.52 | 98.67 | 98.60 | 98.57 | 98.60 | 98.64 |
| maupqa-gpt3-cc | 18.43 | 18.53 | 18.15 | 18.50 | 18.56 | 18.52 | 18.37 | 18.51 | 18.49 |
| maupqa-mkqa | 63.92 | 64.79 | 62.29 | 63.87 | 64.28 | 64.29 | 63.59 | 64.35 | 64.55 |
| maupqa-poleval2021-pairs | 82.67 | 83.32 | 83.23 | 83.03 | 82.98 | 83.02 | 82.94 | 83.02 | 82.86 |
| maupqa-templates | 75.02 | 75.80 | 75.80 | 75.60 | 75.62 | 75.66 | 75.55 | 75.71 | 75.46 |
| nfcorpus-pl | 30.07 | 32.34 | 31.50 | 31.85 | 32.15 | 32.17 | 32.03 | 32.26 | 31.79 |
| poleval-2022-test-A-allegro-faq | 73.76 | 76.85 | 75.45 | 75.57 | 75.67 | 75.30 | 75.19 | 74.67 | 74.83 |
| poleval-2022-test-B-allegro-faq | 71.81 | 73.48 | 70.81 | 71.42 | 72.20 | 71.94 | 72.02 | 72.12 | 71.68 |
| scifact-pl | 64.52 | 66.44 | 66.62 | 67.60 | 66.49 | 67.16 | 67.21 | 66.58 | 66.97 |
| **mean** | **61.60** | **63.65** | **62.32** | **62.89** | **62.85** | **62.77** | **62.81** | **62.80** | **62.70** |

### heldout8

| task | soup-v9b | arm-n4x-nq | arm-n4-3060 | arm-n4-msclean | arm-n4-3060-seed1 | arm-n4-3060-seed2 | soup-n4seeds | arm-n4-msrand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 30.37 | 28.55 | 29.18 | 29.33 | 29.82 | 28.91 | 29.56 | 29.52 |
| maupqa-1z10 | 48.49 | 47.50 | 48.05 | 48.38 | 48.13 | 48.10 | 48.24 | 48.13 |
| maupqa-gpt3.5-cc | 20.26 | 20.02 | 20.07 | 20.16 | 20.15 | 20.07 | 20.15 | 20.15 |
| maupqa-multilingual-NLI | 14.53 | 14.09 | 14.50 | 14.59 | 14.60 | 14.39 | 14.53 | 14.51 |
| maupqa-wiki-def | 66.35 | 66.26 | 65.43 | 66.01 | 65.28 | 65.67 | 65.64 | 65.72 |
| mfaq | 63.18 | 61.89 | 62.67 | 62.70 | 62.69 | 62.57 | 62.73 | 62.64 |
| poleval-2022-test-A-legal-questions | 82.72 | 79.74 | 82.01 | 81.91 | 81.78 | 81.67 | 82.40 | 81.93 |
| scidocs-pl | 14.55 | 13.88 | 14.20 | 13.77 | 13.99 | 14.04 | 14.20 | 13.90 |
| **mean** | **42.56** | **41.49** | **42.01** | **42.11** | **42.06** | **41.93** | **42.18** | **42.06** |


## Queue state (v28, v29)

```
regime:quantized:arm-n4x-nq:tiny10	ok	62.32 10	618s
regime:quantized:arm-n4x-nq:heldout8	ok	41.49 8	2203s
train:arm-n4x-hotpotqa-1pos	ok	-	4008s
sparsity:arm-n4x-hotpotqa-1pos	ok	336.9	23s
regime:quantized:arm-n4x-hotpotqa-1pos:tiny10	ok	62.35 10	592s
regime:quantized:arm-n4x-hotpotqa-1pos:heldout8	ok	42.01 8	2141s
train:arm-n4x-fever-short	ok	-	4622s
sparsity:arm-n4x-fever-short	ok	302.5	24s
regime:quantized:arm-n4x-fever-short:tiny10	ok	62.25 10	573s
regime:quantized:arm-n4x-fever-short:heldout8	ok	41.29 8	2078s
regime:quantized:base-splade-wrapped:mqa	ok	18.68 1	1402s
regime:quantized:soup-v9b:mqa	ok	19.39 1	1632s
regime:quantized:base-splade-wrapped:legalB	ok	78.67 1	203s
regime:quantized:soup-v9b:legalB	ok	82.23 1	208s
regime:quantized:soup-v9b:nqpl	ok	41.46 1	13883s
train:arm-n4-3060-seed1	ok	-	3865s
sparsity:arm-n4-3060-seed1	ok	440.6	24s
regime:quantized:arm-n4-3060-seed1:tiny10	ok	62.77 10	645s
regime:quantized:arm-n4-3060-seed1:heldout8	ok	42.06 8	2235s
train:arm-n4-3060-seed2	ok	-	3869s
sparsity:arm-n4-3060-seed2	ok	435.3	23s
regime:quantized:arm-n4-3060-seed2:tiny10	ok	62.81 10	646s
regime:quantized:arm-n4-3060-seed2:heldout8	ok	41.93 8	2268s
sparsity:soup-n4seeds	ok	434.7	24s
regime:quantized:soup-n4seeds:tiny10	ok	62.8 10	638s
regime:quantized:soup-n4seeds:heldout8	ok	42.18 8	2304s
train:arm-n4-msrand	ok	-	3737s
sparsity:arm-n4-msrand	ok	434.8	24s
regime:quantized:arm-n4-msrand:tiny10	ok	62.7 10	654s
regime:quantized:arm-n4-msrand:heldout8	ok	42.06 8	2303s
```
