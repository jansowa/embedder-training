# s6 (8.11): static query table credit-k8-kd over the r2v12-bias005 index (quantized regime, nDCG@10)

Pool: 120000 kd_v1 queries over 250000 kd_v1 passages, 43.21 candidates per query; the s5 recipe (KL, learned scale, lr 5e-3, 2 epochs).

| query side | index | tiny10 | val top1 | dims/q | postings fraction |
|---|---|---:|---:|---:|---:|
| credit-k8-kd (s5) | n5kj | 59.65 | | 83.3 | 10.945 |
| credit-k8-kd (s6) | r2v12-bias005 | 59.76 | 0.651 | 77.6 | 9.256 |
| r2v12-bias005 symmetric | r2v12-bias005 | 64.21 | 1.000 | 101.9 | 10.101 |
