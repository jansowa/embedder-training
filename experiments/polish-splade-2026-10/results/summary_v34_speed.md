# v34-speed (05.11.2026 07:48, leaderboard regime, nDCG@10): the fast training path on the arm-n4-3060 recipe

arm-n4-3060-fast = the rows, recipe and seed of arm-n4-3060 trained with the sparse-gradient SPLADE head (`splade_fused_head: exact`, exact, no checkpointing). Reference: three stock seeds.

Verdict (tolerance +-0.30 vs the seed mean on both task sets): **PASS: tiny10: fast 62.99 vs seed mean 62.83 (range 62.77-62.89), diff +0.16 -> within +-0.3; heldout8: fast 42.02 vs seed mean 42.00 (range 41.93-42.06), diff +0.03 -> within +-0.3; train 1814s vs 4125s stock = 2.27x**

Memory check (12 x 2 x 512, fp16, with optimizer states): peak 3.10 GB (limit 7.0 GB).

### tiny10

| task | base-splade-wrapped | arm-n4-3060 | arm-n4-3060-seed1 | arm-n4-3060-seed2 | arm-n4-3060-fast |
|---|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 42.96 | 41.06 | 42.67 | 42.49 |
| gpt-exams | 98.63 | 98.52 | 98.60 | 98.57 | 98.57 |
| maupqa-gpt3-cc | 18.43 | 18.50 | 18.52 | 18.37 | 18.52 |
| maupqa-mkqa | 63.92 | 63.87 | 64.29 | 63.59 | 64.13 |
| maupqa-poleval2021-pairs | 82.67 | 83.03 | 83.02 | 82.94 | 83.30 |
| maupqa-templates | 75.02 | 75.60 | 75.66 | 75.55 | 75.51 |
| nfcorpus-pl | 30.07 | 31.85 | 32.17 | 32.03 | 31.88 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.57 | 75.30 | 75.19 | 76.23 |
| poleval-2022-test-B-allegro-faq | 71.81 | 71.42 | 71.94 | 72.02 | 71.96 |
| scifact-pl | 64.52 | 67.60 | 67.16 | 67.21 | 67.29 |
| **mean** | **61.60** | **62.89** | **62.77** | **62.81** | **62.99** |

### heldout8

| task | arm-n4-3060 | arm-n4-3060-seed1 | arm-n4-3060-seed2 | arm-n4-3060-fast |
|---|---:|---:|---:|---:|
| fiqa-pl | 29.18 | 29.82 | 28.91 | 29.19 |
| maupqa-1z10 | 48.05 | 48.13 | 48.10 | 48.13 |
| maupqa-gpt3.5-cc | 20.07 | 20.15 | 20.07 | 20.10 |
| maupqa-multilingual-NLI | 14.50 | 14.60 | 14.39 | 14.49 |
| maupqa-wiki-def | 65.43 | 65.28 | 65.67 | 65.50 |
| mfaq | 62.67 | 62.69 | 62.57 | 62.67 |
| poleval-2022-test-A-legal-questions | 82.01 | 81.78 | 81.67 | 81.84 |
| scidocs-pl | 14.20 | 13.99 | 14.04 | 14.27 |
| **mean** | **42.01** | **42.06** | **41.93** | **42.02** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-n4-3060 | 415.5 | ×0.95 | 601 |  |  |  |  |  |
| arm-n4-3060-seed1 | 440.6 | ×1.01 | 639 |  |  |  |  |  |
| arm-n4-3060-seed2 | 435.3 | ×1.00 | 629 |  |  |  |  |  |
| arm-n4-3060-fast | 416.4 | ×0.95 | 601 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v34-speed)

```
memcheck:n4-3060-fast	ok	3.10	exact, no checkpointing
train:arm-n4-3060-fast	ok	-	1814s
sparsity:arm-n4-3060-fast	ok	416.4	24s
regime:quantized:arm-n4-3060-fast:tiny10	ok	62.99 10	632s
regime:quantized:arm-n4-3060-fast:heldout8	ok	42.02 8	2258s
verdict:n4-3060-fast	PASS	-	tiny10: fast 62.99 vs seed mean 62.83 (range 62.77-62.89), diff +0.16 -> within +-0.3; heldout8: fast 42.02 vs seed mean 42.00 (range 41.93-42.06), diff +0.03 -> within +-0.3; train 1814s vs 4125s stock = 2.27x
```
