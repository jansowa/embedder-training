# s8 (night 8/9.11): static query tables over r2v12-bias005, heldout8 check and two more steps (quantized regime, nDCG@10)

References: r2v12-bias005 symmetric 64.21 / 43.08; BM25 pyserini (leaderboard) 50.07 / 33.34, bm25s 53.75 / 36.17; polish-splade 61.53 / 41.06.

| table | what | tiny10 | heldout8 | val top1 | dims/q | postings fraction |
|---|---|---:|---:|---:|---:|---:|
| credit-k8-kd | s6: token rows | 59.76 | 38.60 | 0.651 | 77.6 | 9.256 |
| phr2-k8-kd-r2 | s7: + words and pairs (min 5), round-2 pool | 60.20 | 38.60 | 0.671 | 69.8 | 8.440 |
| phr2m3-k8-kd-r2 | phrases from 3 queries, round-2 pool | 60.12 | - | 0.684 | 69.8 | 8.427 |
| phr2-k8-kd-r3 | s7 best + round-3 pool | 60.04 | - | 0.674 | 65.5 | 7.985 |

val columns: 2 000 held-out kd_v1 queries of the pool each table last trained on (not comparable across pools).
