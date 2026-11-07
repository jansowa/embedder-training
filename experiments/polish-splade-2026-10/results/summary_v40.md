# v40 (07.11.2026 01:06, leaderboard regime, nDCG@10): colleague rows, soup-v12, weight threshold, asymmetric evaluation

1. arm-dense-kl-kj-ens400: the ensemble teacher (soup-v9b + PolDense-400M, scale 111.8) on the colleague's 2 219 rows (datasets/karoljez/train, 64 % legal), the n1v2-ens400 recipe. 2. soup-v12 = uniform soup of soup-v9b and arm-dense-kl-n5v2-ens400. 3. min_weight = terms with weight below the threshold dropped on both sides at indexing (PIRB fork config key min_weight; the usual rule is round(w*100) > 0, i.e. 0.005). 4. asym-X-q-v9b = queries encoded by X, index built by soup-v9b (PIRB config key query_model).

### tiny10

| task | base-splade-wrapped | arm-dense-kl-n1v2-ens400 | arm-dense-kl-kj-ens400 | arm-dense-kl-n5v2-ens400 | soup-v12 | ens-n5v2-mw002 | ens-n5v2-mw005 | soup-v9b-mw002 | asym-base-q-v9b | asym-b4full-q-v9b | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 45.50 | 39.43 | 47.39 | 46.80 | 47.42 | 47.44 | 46.28 | 45.66 | 45.84 | 46.28 |
| gpt-exams | 98.63 | 98.67 | 98.74 | 98.78 | 98.76 | 98.77 | 98.77 | 98.70 | 98.84 | 98.75 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.59 | 18.66 | 18.77 | 18.69 | 18.77 | 18.77 | 18.53 | 18.60 | 18.52 | 18.53 |
| maupqa-mkqa | 63.92 | 64.69 | 65.04 | 64.69 | 65.09 | 64.66 | 64.66 | 64.77 | 64.77 | 64.75 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.26 | 83.02 | 83.63 | 83.63 | 83.64 | 83.66 | 83.32 | 83.28 | 83.18 | 83.32 |
| maupqa-templates | 75.02 | 76.05 | 75.47 | 76.14 | 76.07 | 76.14 | 76.12 | 75.80 | 75.46 | 75.73 | 75.80 |
| nfcorpus-pl | 30.07 | 32.16 | 31.52 | 32.54 | 32.44 | 32.52 | 32.51 | 32.33 | 32.44 | 32.53 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.53 | 75.38 | 75.94 | 76.78 | 76.08 | 76.01 | 76.83 | 75.77 | 77.08 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 72.72 | 72.35 | 73.42 | 73.74 | 73.42 | 73.44 | 73.48 | 73.39 | 73.64 | 73.48 |
| scifact-pl | 64.52 | 67.10 | 65.58 | 68.23 | 67.65 | 68.23 | 68.33 | 66.24 | 66.50 | 66.93 | 66.44 |
| **mean** | **61.60** | **63.43** | **62.52** | **63.95** | **63.96** | **63.96** | **63.97** | **63.63** | **63.47** | **63.70** | **63.65** |

### heldout8

| task | arm-dense-kl-n1v2-ens400 | arm-dense-kl-kj-ens400 | arm-dense-kl-n5v2-ens400 | soup-v12 | asym-b4full-q-v9b | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 30.20 | 28.21 | 30.13 | 30.68 | 30.37 | 30.37 |
| maupqa-1z10 | 48.45 | 48.30 | 48.98 | 48.84 | 48.42 | 48.49 |
| maupqa-gpt3.5-cc | 20.26 | 20.35 | 20.40 | 20.39 | 20.25 | 20.26 |
| maupqa-multilingual-NLI | 14.68 | 14.52 | 14.66 | 14.67 | 14.53 | 14.53 |
| maupqa-wiki-def | 66.51 | 65.45 | 67.66 | 67.23 | 66.05 | 66.35 |
| mfaq | 63.23 | 63.09 | 63.07 | 63.23 | 63.10 | 63.18 |
| poleval-2022-test-A-legal-questions | 82.82 | 81.15 | 83.97 | 83.49 | 81.97 | 82.72 |
| scidocs-pl | 14.51 | 13.60 | 14.79 | 14.85 | 14.65 | 14.55 |
| **mean** | **42.58** | **41.84** | **42.96** | **42.92** | **42.42** | **42.56** |


| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-dense-kl-kj-ens400 | 412.5 | ×0.95 | 639 |  |  |  |  |  |
| arm-dense-kl-n5v2-ens400 | 425.4 | ×0.97 | 593 |  |  |  |  |  |
| soup-v12 | 417.9 | ×0.96 | 591 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Index cost with thresholds (one PIRB sample: 1 350 passages + 1 350 queries; doc dims, query dims, postings/query)

```
model                               dims/doc  dims/query  postings/q   flops
soup-v9b                               417.6       151.4       17805   3.008
soup-v9b min_weight=0.01               413.6       150.0       17507   3.007
soup-v9b min_weight=0.02               405.4       147.3       16913   3.004
soup-v9b min_weight=0.03               397.4       144.5       16324   2.999
soup-v9b min_weight=0.05               381.9       139.3       15228   2.984
ens-n5v2                               432.3       161.1       20590   3.270
ens-n5v2 min_weight=0.01               427.9       159.6       20214   3.269
ens-n5v2 min_weight=0.02               419.2       156.6       19486   3.266
ens-n5v2 min_weight=0.03               410.7       153.6       18797   3.260
ens-n5v2 min_weight=0.05               394.1       147.7       17462   3.242
b4-full-s3                             436.6       156.3       20728   3.312
b4-full-s3 min_weight=0.01             432.2       154.9       20351   3.310
b4-full-s3 min_weight=0.02             423.3       151.9       19619   3.307
b4-full-s3 min_weight=0.03             414.5       149.1       18903   3.301
b4-full-s3 min_weight=0.05             397.8       143.5       17523   3.283
soup-v12                               426.1       155.8       18733   3.128
soup-v12 min_weight=0.01               421.9       154.3       18401   3.127
soup-v12 min_weight=0.02               413.6       151.4       17769   3.124
soup-v12 min_weight=0.03               405.3       148.7       17154   3.119
soup-v12 min_weight=0.05               389.2       143.1       15971   3.103
```

## Queue state (v40)

```
train:arm-dense-kl-kj-ens400	ok	-	423s
sparsity:arm-dense-kl-kj-ens400	ok	412.5	23s
regime:quantized:arm-dense-kl-kj-ens400:tiny10	ok	62.52 10	609s
regime:quantized:arm-dense-kl-kj-ens400:heldout8	ok	41.84 8	2092s
sparsity:soup-v12	ok	417.9	22s
regime:quantized:soup-v12:tiny10	ok	63.96 10	605s
regime:quantized:soup-v12:heldout8	ok	42.92 8	2178s
regime:quantized:ens-n5v2-mw002:tiny10	ok	63.96 10	627s
regime:quantized:soup-v9b-mw002:tiny10	ok	63.63 10	620s
regime:quantized:ens-n5v2-mw005:tiny10	ok	63.97 10	636s
regime:quantized:asym-base-q-v9b:tiny10	ok	63.47 10	632s
regime:quantized:asym-b4full-q-v9b:tiny10	ok	63.7 10	639s
regime:quantized:asym-b4full-q-v9b:heldout8	ok	42.42 8	2295s
```
