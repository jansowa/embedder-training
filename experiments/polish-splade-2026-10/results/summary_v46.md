# v46 (9.11, quantized regime, nDCG@10): the publication candidate after v44 — the bias form of the seed soup, round-2 seed 43 and its seed soup

23-task mean = (10 x tiny10 + 8 x heldout8 + 5 x held-out-2) / 23. polish-splade (leaderboard) 55.77, soup-v9b 57.58.

| model | tiny10 | heldout8 | dims/doc | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 | 23 tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soup-v9b | 63.65 | 42.56 | 409 | 86.88 | 91.97 | 79.32 | 19.25 | 69.66 | 69.42 | 57.57 |
| soup-v12 | 63.96 | 42.92 | 418 | - | - | - | - | 69.93 | - | - |
| soup-v12-bias005 | 64.05 | 42.83 | 378 | 87.12 | 92.09 | 79.50 | 19.16 | 70.24 | 69.62 | 57.88 |
| arm-dense-kl-n5kj-ens400 | 64.14 | 43.01 | 422 | 87.29 | 92.45 | 79.57 | 19.16 | 68.49 | 69.39 | 57.93 |
| arm-dense-kl-n5kj-ens400-s43 | 64.02 | 43.09 | 428 | - | - | - | - | - | - | - |
| soup-v14 | 64.15 | 43.00 | 421 | - | - | - | - | 69.27 | - | - |
| soup-v14s | 64.18 | 43.11 | 426 | 87.36 | 92.53 | 79.50 | 19.23 | 70.00 | 69.73 | 58.06 |
| soup-v14s-bias005 | 64.22 | 44.89 | 385 | 87.25 | 92.45 | 79.45 | 19.12 | 69.24 | 69.50 | 58.64 |
| soup-v15 | 64.20 | 43.10 | 411 | - | - | - | - | 68.91 | - | - |
| arm-dense-kl-n5kj-ens400-r2v12 | 64.20 | 43.15 | 404 | 87.19 | 92.33 | 79.70 | 19.10 | 67.02 | 69.07 | 57.94 |
| arm-dense-kl-n5kj-ens400-r2v12-s43 | 64.11 | 44.95 | 406 | - | - | - | - | - | - | - |
| soup-v16 | 64.25 | 44.24 | 406 | - | - | - | - | 68.48 | - | - |
| arm-dense-kl-n5kj-ens400-r2v12-bias005 | 64.21 | 43.08 | 365 | 87.03 | 92.24 | 79.68 | 19.02 | 66.02 | 68.80 | 57.86 |
| arm-dense-kl-n5kj-ens400-r2v12-bias010 | 64.24 | 43.00 | 328 | 86.85 | 92.16 | 79.56 | 18.86 | 65.35 | 68.56 | 57.79 |

## Queue state (v46)

```
sparsity:soup-v14s-bias005	ok	385.1	23s
regime:quantized:soup-v14s-bias005:tiny10	ok	64.22 10	616s
regime:quantized:soup-v14s-bias005:heldout8	ok	44.89 7	2160s
regime:quantized:soup-v14s-bias005:heldout2_treccovid	ok	69.24 1	1347s
regime:quantized:soup-v14s-bias005:heldout2_czywiesz	ok	87.25 1	777s
regime:quantized:soup-v14s-bias005:heldout2_gpt35wiki	ok	92.45 1	638s
regime:quantized:soup-v14s-bias005:heldout2_quora	ok	79.45 1	852s
regime:quantized:soup-v14s-bias005:heldout2_poquad	ok	19.12 1	1274s
train:arm-dense-kl-n5kj-ens400-r2v12-s43	ok	-	5152s
sparsity:arm-dense-kl-n5kj-ens400-r2v12-s43	ok	406.2	24s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-s43:tiny10	ok	64.11 10	625s
regime:quantized:arm-dense-kl-n5kj-ens400-r2v12-s43:heldout8	ok	44.95 7	2257s
sparsity:soup-v16	ok	405.6	23s
regime:quantized:soup-v16:tiny10	ok	64.25 10	626s
regime:quantized:soup-v16:heldout8	ok	44.24 6	2117s
regime:quantized:soup-v16:heldout2_treccovid	ok	68.48 1	1370s
```
