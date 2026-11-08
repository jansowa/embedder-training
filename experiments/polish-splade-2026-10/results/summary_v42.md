# v42 (08.11.2026 06:37, leaderboard regime, nDCG@10): around n5 + kj — seed, soups, round 2 from soup-v12, held-out-2 for the best, regulator

n5kj = the 23.5-gated n5 rows (fiqa 2 073 + 16 000 squadv2-v2) plus all 2 219 colleague rows, KL from the ensemble teacher (soup-v9b + PolDense-400M), the ens-n5v2 recipe, seeds 42 and 43. soup-v14 = uniform(soup-v12, n5kj); soup-v14s = uniform(n5kj s42, n5kj s43). r2v12 = the same recipe with the student starting from soup-v12. reg = regularizer_control keeping dims/doc inside 0.90-1.02 of soup-v9b. Best of the night: arm-dense-kl-n5kj-ens400-r2v12; publication candidate: arm-dense-kl-n5kj-ens400-r2v12-bias005 (bias005 = lm_head.bias shifted by -0.0513, the threshold inside the checkpoint); held-out-2 = the five tasks that complete the 23 leaderboard tasks.

### tiny10

| task | base-splade-wrapped | arm-dense-kl-n5v2-ens400 | arm-dense-kl-n5kj-ens400 | arm-dense-kl-n5kj-ens400-s43 | soup-v14 | soup-v14s | arm-dense-kl-n5kj-ens400-r2v12 | arm-dense-kl-n7-ens400 | arm-dense-kl-n5kj-ens400-reg | soup-v12 | arm-dense-kl-n5kj-ens400-r2v12-bias005 | soup-v12-bias005 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 47.39 | 47.73 | 47.19 | 47.56 | 47.49 | 48.79 | 47.43 | 48.61 | 46.80 | 49.03 | 47.40 | 46.28 |
| gpt-exams | 98.63 | 98.78 | 98.84 | 98.75 | 98.83 | 98.80 | 98.85 | 98.76 | 98.81 | 98.76 | 98.82 | 98.77 | 98.71 |
| maupqa-gpt3-cc | 18.43 | 18.77 | 18.83 | 18.88 | 18.77 | 18.87 | 18.78 | 18.78 | 18.72 | 18.69 | 18.77 | 18.68 | 18.53 |
| maupqa-mkqa | 63.92 | 64.69 | 64.67 | 64.63 | 64.99 | 64.88 | 64.77 | 64.91 | 64.55 | 65.09 | 64.73 | 65.07 | 64.79 |
| maupqa-poleval2021-pairs | 82.67 | 83.63 | 83.35 | 83.71 | 83.65 | 83.62 | 83.66 | 83.72 | 83.41 | 83.63 | 83.33 | 83.52 | 83.32 |
| maupqa-templates | 75.02 | 76.14 | 76.15 | 76.29 | 76.21 | 76.31 | 76.10 | 76.19 | 76.01 | 76.07 | 76.01 | 76.02 | 75.80 |
| nfcorpus-pl | 30.07 | 32.54 | 32.49 | 32.52 | 32.66 | 32.53 | 32.44 | 32.70 | 32.36 | 32.44 | 32.53 | 32.46 | 32.34 |
| poleval-2022-test-A-allegro-faq | 73.76 | 75.94 | 77.22 | 76.02 | 76.77 | 76.74 | 77.24 | 75.21 | 76.90 | 76.78 | 77.31 | 76.99 | 76.85 |
| poleval-2022-test-B-allegro-faq | 71.81 | 73.42 | 74.65 | 74.26 | 74.15 | 74.74 | 73.74 | 74.22 | 73.98 | 73.74 | 73.83 | 73.34 | 73.48 |
| scifact-pl | 64.52 | 68.23 | 67.52 | 67.98 | 67.96 | 67.88 | 67.68 | 67.71 | 67.95 | 67.65 | 67.71 | 68.26 | 66.44 |
| **mean** | **61.60** | **63.95** | **64.14** | **64.02** | **64.15** | **64.18** | **64.20** | **63.96** | **64.13** | **63.96** | **64.21** | **64.05** | **63.65** |

### heldout8

| task | arm-dense-kl-n5v2-ens400 | arm-dense-kl-n5kj-ens400 | arm-dense-kl-n5kj-ens400-s43 | soup-v14 | soup-v14s | arm-dense-kl-n5kj-ens400-r2v12 | arm-dense-kl-n7-ens400 | arm-dense-kl-n5kj-ens400-reg | soup-v12 | arm-dense-kl-n5kj-ens400-r2v12-bias005 | soup-v12-bias005 | soup-v9b |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fiqa-pl | 30.13 | 30.11 | 30.43 | 30.31 | 30.44 | 30.76 | 30.40 | 29.87 | 30.68 | 30.59 | 30.18 | 30.37 |
| maupqa-1z10 | 48.98 | 49.14 | 49.39 | 49.07 | 49.34 | 49.22 | 49.14 | 49.06 | 48.84 | 49.18 | 48.83 | 48.49 |
| maupqa-gpt3.5-cc | 20.40 | 20.42 | 20.49 | 20.44 | 20.47 | 20.44 | 20.43 | 20.37 | 20.39 | 20.42 | 20.39 | 20.26 |
| maupqa-multilingual-NLI | 14.66 | 14.73 | 14.69 | 14.73 | 14.72 | 14.72 | 14.75 | 14.58 | 14.67 | 14.68 | 14.62 | 14.53 |
| maupqa-wiki-def | 67.66 | 67.53 | 67.48 | 67.44 | 67.53 | 67.49 | 67.39 | 67.40 | 67.23 | 67.39 | 67.09 | 66.35 |
| mfaq | 63.07 | 63.24 | 63.27 | 63.31 | 63.30 | 63.22 | 63.21 | 63.13 | 63.23 | 63.19 | 63.21 | 63.18 |
| poleval-2022-test-A-legal-questions | 83.97 | 84.14 | 84.23 | 83.88 | 84.29 | 84.38 | 83.70 | 84.04 | 83.49 | 84.36 | 83.39 | 82.72 |
| scidocs-pl | 14.79 | 14.74 | 14.75 | 14.87 | 14.82 | 14.93 | 14.91 | 14.83 | 14.85 | 14.82 | 14.95 | 14.55 |
| **mean** | **42.96** | **43.01** | **43.09** | **43.00** | **43.11** | **43.15** | **42.99** | **42.91** | **42.92** | **43.08** | **42.83** | **42.56** |


## Held-out-2 (quantized; soup-v9b in old-pyserini: czywiesz 86.88, gpt35wiki 91.97, quora 79.32, poquad 19.25, treccovid 69.66, mean 69.42)

- arm-dense-kl-n5kj-ens400-r2v12-bias005: czywiesz 87.03, gpt35wiki 92.24, quora 79.68, poquad 19.02, treccovid 66.02; mean 68.80 (5/5)
- soup-v12-bias005: czywiesz 87.12, gpt35wiki 92.09, quora 79.5, poquad 19.16, treccovid 70.24; mean 69.62 (5/5)

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| arm-dense-kl-n5kj-ens400 | 422.3 | ×0.97 | 588 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400-s43 | 427.5 | ×0.98 | 603 |  |  |  |  |  |
| soup-v14 | 421.3 | ×0.97 | 593 |  |  |  |  |  |
| soup-v14s | 426.4 | ×0.98 | 599 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400-r2v12 | 404.4 | ×0.93 | 558 |  |  |  |  |  |
| arm-dense-kl-n7-ens400 | 438.0 | ×1.00 | 612 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400-reg | 377.5 | ×0.87 | 526 |  |  |  |  |  |
| soup-v12 | 417.9 | ×0.96 | 591 |  |  |  |  |  |
| soup-v12-bias005 | 377.8 | ×0.87 | 535 |  |  |  |  |  |
| arm-dense-kl-n5kj-ens400-r2v12-bias005 | 364.9 | ×0.84 | 502 |  |  |  |  |  |

Źródło próbki kosztu: index_cost_best.json (1350 pasaży i tyle samo zapytań).

## Queue state (v42)

```
regime:quantized:soup-v6a:heldout2_czywiesz	ok	87.17 1	800s
regime:quantized:soup-v6a:heldout2_gpt35wiki	ok	92.3 1	643s
regime:quantized:soup-v6a:heldout2_poquad	ok	19.86 1	1333s
regime:quantized:soup-v6a:heldout2_quora	ok	79.16 1	851s
regime:quantized:soup-v6a:heldout2_treccovid	ok	66.07 1	1449s
regime:quantized:soup-v7a:heldout2_czywiesz	ok	86.99 1	812s
regime:quantized:soup-v7a:heldout2_gpt35wiki	ok	92.01 1	663s
regime:quantized:soup-v7a:heldout2_poquad	ok	19.37 1	1349s
regime:quantized:soup-v7a:heldout2_quora	ok	79.15 1	870s
regime:quantized:soup-v7a:heldout2_treccovid	ok	68.29 1	1407s
regime:quantized:n4-5070:heldout2_czywiesz	ok	86.81 1	879s
regime:quantized:n4-5070:heldout2_gpt35wiki	ok	92.13 1	709s
regime:quantized:n4-5070:heldout2_poquad	ok	19.85 1	1463s
regime:quantized:n4-5070:heldout2_quora	ok	78.73 1	856s
regime:quantized:n4-5070:heldout2_treccovid	ok	67.46 1	1493s
regime:quantized:arm-mnrl-fiqa-ldose2:heldout2_treccovid	ok	69.87 1	1415s
regime:quantized:soup-v8a:heldout2_czywiesz	ok	87.09 1	818s
regime:quantized:soup-v8a:heldout2_gpt35wiki	ok	92.23 1	663s
regime:quantized:soup-v8a:heldout2_poquad	ok	19.8 1	1358s
regime:quantized:soup-v8a:heldout2_quora	ok	79.29 1	862s
regime:quantized:soup-v8a:heldout2_treccovid	ok	68.05 1	1441s
regime:quantized:soup-v9b:heldout2_czywiesz	ok	86.88 1	801s
regime:quantized:soup-v9b:heldout2_gpt35wiki	ok	91.97 1	647s
regime:quantized:soup-v9b:heldout2_poquad	ok	19.25 1	1342s
regime:quantized:soup-v9b:heldout2_quora	ok	79.32 1	850s
regime:quantized:soup-v9b:heldout2_treccovid	ok	69.66 1	1338s
regime:quantized:soup-v10-squad16k:heldout2_czywiesz	ok	86.92 1	775s
regime:quantized:soup-v10-squad16k:heldout2_gpt35wiki	ok	92.01 1	636s
regime:quantized:soup-v10-squad16k:heldout2_poquad	ok	19.26 1	1317s
regime:quantized:soup-v10-squad16k:heldout2_quora	ok	79.19 1	860s
regime:quantized:soup-v10-squad16k:heldout2_treccovid	ok	67.48 1	1391s
regime:quantized:soup-v9b-len256:heldout2_czywiesz	ok	86.88 1	827s
regime:quantized:soup-v9b-len256:heldout2_gpt35wiki	ok	91.95 1	669s
regime:quantized:soup-v9b-len256:heldout2_poquad	ok	19.17 1	1345s
regime:quantized:soup-v9b-len256:heldout2_quora	ok	79.32 1	873s
regime:quantized:soup-v9b-len256:heldout2_treccovid	ok	67.59 1	954s
regime:quantized:soup-v11-v2:heldout2_czywiesz	ok	86.88 1	803s
regime:quantized:soup-v11-v2:heldout2_gpt35wiki	ok	92.02 1	676s
regime:quantized:soup-v11-v2:heldout2_poquad	ok	19.2 1	1339s
regime:quantized:soup-v11-v2:heldout2_quora	ok	79.35 1	861s
regime:quantized:soup-v11-v2:heldout2_treccovid	ok	69.15 1	1390s
regime:quantized:soup-v11-nq:heldout2_czywiesz	ok	86.97 1	806s
regime:quantized:soup-v11-nq:heldout2_gpt35wiki	ok	92.15 1	663s
regime:quantized:soup-v11-nq:heldout2_poquad	ok	19.37 1	1353s
regime:quantized:soup-v11-nq:heldout2_quora	ok	79.32 1	864s
regime:quantized:soup-v11-nq:heldout2_treccovid	ok	69.35 1	1414s
regime:quantized:soup-v11-n5:heldout2_czywiesz	ok	86.93 1	831s
regime:quantized:soup-v11-n5:heldout2_gpt35wiki	ok	92.11 1	686s
regime:quantized:soup-v11-n5:heldout2_poquad	ok	19.22 1	1389s
regime:quantized:soup-v11-n5:heldout2_quora	ok	79.39 1	862s
regime:quantized:soup-v11-n5:heldout2_treccovid	ok	69.45 1	1406s
train:arm-dense-kl-n5kj-ens400	ok	-	5038s
sparsity:arm-dense-kl-n5kj-ens400	ok	422.3	22s
regime:quantized:arm-dense-kl-n5kj-ens400:tiny10	ok	64.14 10	605s
regime:quantized:arm-dense-kl-n5kj-ens400:heldout8	ok	43.01 8	2215s
train:arm-dense-kl-n5kj-ens400-s43	ok	-	5048s
sparsity:arm-dense-kl-n5kj-ens400-s43	ok	427.5	22s
regime:quantized:arm-dense-kl-n5kj-ens400-s43:tiny10	ok	64.02 10	602s
regime:quantized:arm-dense-kl-n5kj-ens400-s43:heldout8	ok	43.09 8	2303s
sparsity:soup-v14	ok	421.3	24s
regime:quantized:soup-v14:tiny10	ok	64.15 10	634s
regime:quantized:soup-v14:heldout8	ok	43.0 8	2283s
sparsity:soup-v14s	ok	426.4	24s
regime:quantized:soup-v14s:tiny10	ok	64.18 10	638s
regime:quantized:soup-v14s:heldout8	ok	43.11 8	2335s
train:arm-dense-kl-n5kj-ens400-r2v12	ok	-	5079s
sparsity:arm-dense-kl-n5kj-ens400-r2v12	ok	404.4	22s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12:tiny10	ok	64.2 10	633s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12:heldout8	ok	43.15 8	2288s
train:arm-dense-kl-n7-ens400	ok	-	6054s
sparsity:arm-dense-kl-n7-ens400	ok	438.0	24s
regime:quantized:arm-dense-kl-n7-ens400:tiny10	ok	63.96 10	648s
regime:quantized:arm-dense-kl-n7-ens400:heldout8	ok	42.99 8	2294s
sparsity:arm-dense-kl-n5kj-ens400-r2v12-bias005	ok	364.9	24s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias005:tiny10	ok	64.21 10	606s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias005:heldout8	ok	43.08 8	2227s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias005:heldout2_czywiesz	ok	87.03 1	760s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias005:heldout2_gpt35wiki	ok	92.24 1	617s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias005:heldout2_quora	ok	79.68 1	847s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias005:heldout2_poquad	ok	19.02 1	1238s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-bias005:heldout2_treccovid	ok	66.02 1	1323s
regime:quantized:soup-v12-bias005:heldout2_czywiesz	ok	87.12 1	784s
regime:quantized:soup-v12-bias005:heldout2_gpt35wiki	ok	92.09 1	640s
regime:quantized:soup-v12-bias005:heldout2_quora	ok	79.5 1	844s
regime:quantized:soup-v12-bias005:heldout2_poquad	ok	19.16 1	1275s
regime:quantized:soup-v12-bias005:heldout2_treccovid	ok	70.24 1	1347s
train:arm-dense-kl-n5kj-ens400-reg	ok	-	5164s
sparsity:arm-dense-kl-n5kj-ens400-reg	ok	377.5	24s
regime:quantized:arm-dense-kl-n5kj-ens400-reg:tiny10	ok	64.13 10	600s
regime:quantized:arm-dense-kl-n5kj-ens400-reg:heldout8	ok	42.91 8	2203s
```
