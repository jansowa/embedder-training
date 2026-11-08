# v43 (08.11.2026 11:59, leaderboard regime, nDCG@10): the bias form of the threshold at 0.05 and 0.10

biasT = the checkpoint with lm_head.bias shifted by -(e^T - 1): a dimension stays active only when its pooled weight would exceed T, the other weights drop slightly (the threshold inside the checkpoint, no custom layer). hl-full-s3 = the ALBERT student (50 M params, 0.45 of the teacher's cost); the best full model of the night = arm-dense-kl-n5kj-ens400-r2v12 (43.15 on heldout8).

### tiny10

| task | base-splade-wrapped | hl-full-s3-bias005 | hl-full-s3-bias010 | soup-v12 | soup-v12-bias005 | arm-dense-kl-n5kj-ens400-r2v12 | arm-dense-kl-n5kj-ens400-r2v12-bias005 | arm-dense-kl-n5kj-ens400-r2v12-bias010 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 43.52 | 44.35 | 46.80 | 47.40 | 48.79 | 49.03 | 49.37 | 46.28 |
| gpt-exams | 98.63 | 98.50 | 98.52 | 98.76 | 98.77 | 98.85 | 98.82 | 98.81 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.50 | 18.48 | 18.69 | 18.68 | 18.78 | 18.77 | 18.75 | 18.53 |
| maupqa-mkqa | 63.92 | 65.03 | 64.92 | 65.09 | 65.07 | 64.77 | 64.73 | 64.55 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 82.86 | 82.81 | 83.63 | 83.52 | 83.66 | 83.33 | 83.09 | 83.32 |
| maupqa-templates | 75.02 | 75.53 | 75.49 | 76.07 | 76.02 | 76.10 | 76.01 | 75.90 | 75.80 |
| nfcorpus-pl | 30.07 | 32.47 | 32.59 | 32.44 | 32.46 | 32.44 | 32.53 | 32.50 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 74.74 | 75.06 | 76.78 | 76.99 | 77.24 | 77.31 | 77.53 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.70 | 73.05 | 73.74 | 73.34 | 73.74 | 73.83 | 73.77 | 73.48 |
| scifact-pl | 64.52 | 66.78 | 66.43 | 67.65 | 68.26 | 67.68 | 67.71 | 68.08 | 66.44 |
| **mean** | **61.60** | **63.06** | **63.17** | **63.96** | **64.05** | **64.20** | **64.21** | **64.24** | **63.65** |

### heldout8

| task | hl-full-s3-bias005 | hl-full-s3-bias010 | soup-v12 | soup-v12-bias005 | arm-dense-kl-n5kj-ens400-r2v12 | arm-dense-kl-n5kj-ens400-r2v12-bias005 | arm-dense-kl-n5kj-ens400-r2v12-bias010 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 29.85 | 29.84 | 30.68 | 30.18 | 30.76 | 30.59 | 30.34 | 30.37 |
| maupqa-1z10 | 48.70 | 48.66 | 48.84 | 48.83 | 49.22 | 49.18 | 49.06 | 48.49 |
| maupqa-gpt3.5-cc | 20.17 | 20.15 | 20.39 | 20.39 | 20.44 | 20.42 | 20.40 | 20.26 |
| maupqa-multilingual-NLI | 14.56 | 14.53 | 14.67 | 14.62 | 14.72 | 14.68 | 14.62 | 14.53 |
| maupqa-wiki-def | 65.77 | 65.59 | 67.23 | 67.09 | 67.49 | 67.39 | 67.25 | 66.35 |
| mfaq | 62.94 | 62.89 | 63.23 | 63.21 | 63.22 | 63.19 | 63.15 | 63.18 |
| poleval-2022-test-A-legal-questions | 80.49 | 80.30 | 83.49 | 83.39 | 84.38 | 84.36 | 84.35 | 82.72 |
| scidocs-pl | 14.13 | 14.14 | 14.85 | 14.95 | 14.93 | 14.82 | 14.81 | 14.55 |
| **mean** | **42.08** | **42.01** | **42.92** | **42.83** | **43.15** | **43.08** | **43.00** | **42.56** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-b4-l0125-f1536-hl-r256-w-full-s3 |  |  |  |  |  |  |  |  |
| hl-full-s3-bias005 | 454.9 | ×1.04 | 669 |  |  |  |  |  |
| hl-full-s3-bias010 | 403.8 | ×0.93 | 592 |  |  |  |  |  |
| soup-v12 | 417.9 | ×0.96 | 591 |  |  |  |  |  |
| soup-v12-bias005 | 377.8 | ×0.87 | 535 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400-r2v12 | 404.4 | ×0.93 | 558 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400-r2v12-bias005 | 364.9 | ×0.84 | 502 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400-r2v12-bias010 | 328.4 | ×0.75 | 452 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Index cost (one PIRB sample: 1 350 passages + 1 350 queries; doc dims, query dims, postings/query; thresholds applied at indexing on top of each checkpoint)

```
model                               dims/doc  dims/query  postings/q   flops
soup-v9b                               417.6       151.4       17805   3.008
soup-v9b min_weight=0.05               381.9       139.3       15228   2.984
soup-v9b min_weight=0.1                345.0       126.7       12722   2.915
soup-v12                               426.1       155.8       18733   3.128
soup-v12 min_weight=0.05               389.2       143.1       15971   3.103
soup-v12 min_weight=0.1                351.1       130.0       13332   3.031
hl-full-s3                             524.6       183.3       27501   3.733
hl-full-s3 min_weight=0.05             471.9       165.9       22643   3.693
hl-full-s3 min_weight=0.1              418.7       148.1       18144   3.582
soup-v12-bias005                       385.3       141.8       15700   2.725
soup-v12-bias005 min_weight=0.05       352.8       130.6       13448   2.704
soup-v12-bias005 min_weight=0.1        318.9       118.8       11266   2.644
arm-dense-kl-n5kj-ens400-r2v12         411.8       152.0       19411   3.085
arm-dense-kl-n5kj-ens400-r2v12 min_weight=0.05     375.2       139.5       16446   3.058
arm-dense-kl-n5kj-ens400-r2v12 min_weight=0.1     337.6       126.6       13614   2.983
arm-dense-kl-n5kj-ens400-r2v12-bias005     371.4       138.2       16156   2.675
arm-dense-kl-n5kj-ens400-r2v12-bias005 min_weight=0.05     339.3       127.2       13739   2.653
arm-dense-kl-n5kj-ens400-r2v12-bias005 min_weight=0.1     306.3       115.6       11426   2.590
arm-dense-kl-n5kj-ens400-r2v12-bias010     334.4       125.4       13389   2.312
arm-dense-kl-n5kj-ens400-r2v12-bias010 min_weight=0.05     306.3       115.6       11426   2.293
arm-dense-kl-n5kj-ens400-r2v12-bias010 min_weight=0.1     277.5       105.4        9558   2.242
hl-full-s3-bias005                     466.4       164.2       22164   3.190
hl-full-s3-bias005 min_weight=0.05     421.0       148.9       18340   3.157
hl-full-s3-bias005 min_weight=0.1      375.1       133.5       14804   3.068
hl-full-s3-bias010                     414.1       146.5       17784   2.722
hl-full-s3-bias010 min_weight=0.05     375.1       133.5       14804   2.696
hl-full-s3-bias010 min_weight=0.1      335.8       120.2       12059   2.624
```

## Queue state (v43)

```
sparsity:hl-full-s3-bias005	ok	454.9	20s
regime:quantized:hl-full-s3-bias005:tiny10	ok	63.06 10	578s
regime:quantized:hl-full-s3-bias005:heldout8	ok	42.08 8	1979s
regime:quantized:hl-full-s3-bias010:tiny10	ok	63.17 10	508s
regime:quantized:hl-full-s3-bias010:heldout8	ok	42.01 8	1864s
sparsity:arm-dense-kl-n5kj-ens400-r2v12-bias010	ok	328.4	24s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias010:tiny10	ok	64.24 10	589s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias010:heldout8	ok	43.0 8	2145s
```
