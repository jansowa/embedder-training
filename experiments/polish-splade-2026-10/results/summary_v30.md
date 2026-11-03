# v30 (03.11.2026 04:44, leaderboard regime, nDCG@10): soup-v9b at 256 tokens, arm-n4x-nq on nq-pl

### tiny10

| task | base-splade-wrapped | soup-v9b | soup-v9b-len256 | arm-n4x-nq |
|---|---:|---:|---:|---:|
| arguana-pl | 37.14 | 46.28 | 49.56 | 40.80 |
| gpt-exams | 98.63 | 98.71 | 98.64 | 98.56 |
| maupqa-gpt3-cc | 18.43 | 18.53 | 18.53 | 18.15 |
| maupqa-mkqa | 63.92 | 64.79 | 64.80 | 62.29 |
| maupqa-poleval2021-pairs | 82.67 | 83.32 | 83.32 | 83.23 |
| maupqa-templates | 75.02 | 75.80 | 75.81 | 75.80 |
| nfcorpus-pl | 30.07 | 32.34 | 32.01 | 31.50 |
| poleval-2022-test-A-allegro-faq | 73.76 | 76.85 | 76.78 | 75.45 |
| poleval-2022-test-B-allegro-faq | 71.81 | 73.48 | 73.32 | 70.81 |
| scifact-pl | 64.52 | 66.44 | 65.11 | 66.62 |
| **mean** | **61.60** | **63.65** | **63.79** | **62.32** |

### heldout8

| task | soup-v9b | soup-v9b-len256 | arm-n4x-nq |
|---|---:|---:|---:|
| fiqa-pl | 30.37 | 29.92 | 28.55 |
| maupqa-1z10 | 48.49 | 48.49 | 47.50 |
| maupqa-gpt3.5-cc | 20.26 | 20.26 | 20.02 |
| maupqa-multilingual-NLI | 14.53 | 14.53 | 14.09 |
| maupqa-wiki-def | 66.35 | 66.35 | 66.26 |
| mfaq | 63.18 | 63.18 | 61.89 |
| poleval-2022-test-A-legal-questions | 82.72 | 82.40 | 79.74 |
| scidocs-pl | 14.55 | 14.65 | 13.88 |
| **mean** | **42.56** | **42.47** | **41.49** |

### heldout2_czywiesz

| task | soup-v9b | soup-v9b-len256 |
|---|---:|---:|
| maupqa-czy-wiesz-v2 | 86.88 | 86.88 |
| **mean** | **86.88** | **86.88** |

### heldout2_gpt35wiki

| task | soup-v9b | soup-v9b-len256 |
|---|---:|---:|
| maupqa-gpt3.5-wiki | 91.97 | 91.95 |
| **mean** | **91.97** | **91.95** |

### heldout2_poquad

| task | soup-v9b | soup-v9b-len256 |
|---|---:|---:|
| maupqa-poquad | 19.25 | 19.17 |
| **mean** | **19.25** | **19.17** |

### heldout2_quora

| task | soup-v9b | soup-v9b-len256 |
|---|---:|---:|
| quora-pl | 79.32 | 79.32 |
| **mean** | **79.32** | **79.32** |

### heldout2_treccovid

| task | soup-v9b | soup-v9b-len256 |
|---|---:|---:|
| trec-covid-pl | 69.66 | 67.59 |
| **mean** | **69.66** | **67.59** |

### nqpl

| task | base-splade-wrapped | soup-v9b |
|---|---:|---:|
| nq-pl | 40.07 | 41.46 |
| **mean** | **40.07** | **41.46** |


## Queue state (v30)

```
regime:quantized:arm-n4x-nq:tiny10	ok	62.32 10	618s
regime:quantized:arm-n4x-nq:heldout8	ok	41.49 8	2203s
regime:quantized:soup-v9b-len256:tiny10	ok	63.79 10	584s
regime:quantized:soup-v9b-len256:heldout8	ok	42.47 8	2072s
regime:quantized:soup-v9b-len256:heldout2_czywiesz	ok	86.88 1	827s
regime:quantized:soup-v9b-len256:heldout2_gpt35wiki	ok	91.95 1	669s
regime:quantized:soup-v9b-len256:heldout2_poquad	ok	19.17 1	1345s
regime:quantized:soup-v9b-len256:heldout2_quora	ok	79.32 1	873s
regime:quantized:soup-v9b-len256:heldout2_treccovid	ok	67.59 1	954s
```
