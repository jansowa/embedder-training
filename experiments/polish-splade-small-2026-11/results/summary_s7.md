# s7 (8.11): phrase rows and round-2 mining for the static query table over r2v12-bias005 (quantized regime, tiny10 nDCG@10)

| table | what | tiny10 | val top1 | dims/q | postings fraction |
|---|---|---:|---:|---:|---:|
| r2v12-bias005 symmetric | full SPLADE query side | 64.21 | 1.000 | | |
| credit-k8-kd | s6: token rows, k8, s6 pool | 59.76 | 0.651 | 77.6 | 9.256 |
| phr1-k8-kd | + rows for multi-piece words | 59.87 | 0.673 | 79.5 | 9.543 |
| phr2-k8-kd | + words and word pairs | 60.01 | 0.680 | 79.9 | 9.592 |
| credit-k8-kd-r2 | s6 table + round-2 pool, 3 epochs | 59.70 | 0.617 | 68.5 | 8.324 |
| phr2-k8-kd-r2 | phr2 + round-2 pool, 3 epochs | 60.20 | 0.671 | 69.8 | 8.440 |

val columns: 2 000 held-out kd_v1 queries of the pool each table trained on (the r2 rows on the round-2 pool, not comparable 1:1).
