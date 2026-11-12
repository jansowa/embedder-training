# v55 (night 11/12.11, quantized regime, nDCG@10): held-out-2 and the 23-task mean of the student-mined arm

Publication form: **arm-dense-kl-n5kj-mined-ens400-r2v12-bias013**, chosen by the rule (sparsest form within 0.1 of bias010 on both sets). 23-task mean = (10 x tiny10 + 8 x heldout8 + 5 x held-out-2) / 23. dims/doc = the sparsity sample (300 passages of each tiny task); the cost columns = the cost sample (1 350 passages + 1 350 queries, scripts/splade_index_cost.py).

| model | tiny10 | heldout8 | czywiesz | gpt35wiki | quora | poquad | treccovid | held-out-2 | 23 tasks | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polish-splade (base-splade-wrapped), old-pyserini regime* | 61.59 | 41.03 | - | - | - | - | - | 67.53 | 55.73 | 436 | 450.0 | 138.6 | 16415 |
| soup-v12-bias005 | 64.05 | 42.83 | 87.12 | 92.09 | 79.50 | 19.16 | 70.24 | 69.62 | 57.88 | 378 | 385.3 | 141.8 | 15700 |
| soup-v14s | 64.18 | 43.11 | 87.36 | 92.53 | 79.50 | 19.23 | 70.00 | 69.73 | 58.06 | 426 | 435.7 | 154.9 | 19807 |
| arm-dense-kl-n5kj-ens400-r2v12 | 64.20 | 43.15 | 87.19 | 92.33 | 79.70 | 19.10 | 67.02 | 69.07 | 57.94 | 404 | 411.8 | 152.0 | 19411 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | - | - | - | - | 70.15 | - | - | 405 | 413.6 | 130.8 | 16461 |
| **arm-dense-kl-n5kj-mined-ens400-r2v12-bias013** | 64.64 | 43.40 | 87.45 | 92.67 | 79.60 | 19.42 | 69.74 | 69.77 | 58.37 | 376 | 384.0 | 122.2 | 14479 |

\* polish-splade has its three sets only in the old-pyserini regime (summary_v36_leaderboard.md); the other rows are quantized. Its quantized tiny10 is 61.60.

## Bias forms of the mined arm considered by the rule (reference bias010: 64.63 / 43.45)

| form | tiny10 | heldout8 | trec-covid | dims/doc | within 0.1 on both |
|---|---:|---:|---:|---:|---|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | 70.15 | 405.3 | yes |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias011 | 64.62 | 43.48 | 69.70 | 395.5 | yes |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias012 | 64.63 | 43.44 | 69.60 | 385.9 | yes |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias013 | 64.64 | 43.40 | 69.74 | 376.5 | yes |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 | 64.67 | 43.35 | 69.86 | 358.0 | no |

The rule's pick: arm-dense-kl-n5kj-mined-ens400-r2v12-bias013.

## Queue state (v55)

```
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias013:tiny10	ok	64.64 10	573s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias013:heldout8	ok	43.4 8	2039s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias013:heldout2_treccovid	ok	69.74 1	1281s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias013:heldout2_czywiesz	ok	87.45 1	700s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias013:heldout2_gpt35wiki	ok	92.67 1	584s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias013:heldout2_quora	ok	79.6 1	721s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias013:heldout2_poquad	ok	19.42 1	1149s
```
