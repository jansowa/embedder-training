# v44 (night 8/9.11, quantized regime, nDCG@10): held-out-2 per task before the publication decision

soup-v9b in old-pyserini: czywiesz 86.88, gpt35wiki 91.97, quora 79.32, poquad 19.25, treccovid 69.66, mean 69.42 (quantized = old-pyserini within 0.3 except trec-covid, -0.83).

| model | tiny10 | heldout8 | czywiesz | gpt35wiki | quora | poquad | treccovid | mean (5/5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| soup-v9b | 63.65 | 42.56 | 86.88 | 91.97 | 79.32 | 19.25 | 69.66 | 69.42 |
| soup-v12 | 63.96 | 42.92 | - | - | - | - | 69.93 | - |
| soup-v12-bias005 | 64.05 | 42.83 | 87.12 | 92.09 | 79.50 | 19.16 | 70.24 | 69.62 |
| arm-dense-kl-n5v2-ens400 | 63.95 | 42.96 | - | - | - | - | 69.38 | - |
| arm-dense-kl-n5kj-ens400 | 64.14 | 43.01 | 87.29 | 92.45 | 79.57 | 19.16 | 68.49 | 69.39 |
| soup-v14s | 64.18 | 43.11 | 87.36 | 92.53 | 79.50 | 19.23 | 70.00 | 69.73 |
| arm-dense-kl-n5kj-ens400-r2v12 | 64.20 | 43.15 | 87.19 | 92.33 | 79.70 | 19.10 | 67.02 | 69.07 |
| arm-dense-kl-n5kj-ens400-r2v12-bias005 | 64.21 | 43.08 | 87.03 | 92.24 | 79.68 | 19.02 | 66.02 | 68.80 |
| arm-dense-kl-n5kj-ens400-r2v12-bias010 | 64.24 | 43.00 | 86.85 | 92.16 | 79.56 | 18.86 | 65.35 | 68.56 |
