# b2 (quantized regime, nDCG@10): a 124 M replacement of the frozen full model

The release of 26.11 has frozen versions; a stronger 124 M model (sdadas/polish-roberta-base-v2 after KD from the frozen model, `summary_b1.md`) found by Thursday 19.11 would replace the full model. Rule of the replacement: the highest mean over the 19 tasks of tiny10, heldout8 and trec-covid (every task with the same weight) with at most 410 dims/doc and at most 16807 postings/query, and a lead of at least 0.2 over `soup-v20-dqf-bias015` (56.435): the architecture changes and the encoding is about twice as slow. The winner then needs its 23 tasks not below the frozen model's (58.799). The decision is the user's. Every candidate is measured in its first form within both limits (no shift, 0.05, 0.10, then steps of 0.01). One seed each; the seed noise of the ranking recipe is 0.13-0.2 on tiny10.

## Candidates

| candidate | measured form | tiny10 | heldout8 | trec-covid | mean of 19 tasks | vs the frozen model | mean of 11 tasks (tiny10 + trec-covid) | vs the frozen model | dims/doc | dims/query | postings/query | within the limits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frozen full model (82 M) | `soup-v20-dqf-bias015` | 65.06 | 43.64 | 72.48 | 56.435 | reference | 65.738 | reference | 371.4 | 123.0 | 13945 | yes |
| **R-c' after the dose3 round** | `arm-b1-rc-d3-bias010` | 65.59 | 44.01 | 68.42 | 56.652 | +0.217 | 65.843 | +0.104 | 389.3 | 132.7 | 16355 | yes |
| R-a, bias015 | `arm-b1-ra-bias015` | 65.34 | 43.76 | 67.39 | 56.359 | -0.076 | 65.525 | -0.214 | 361.8 | 130.8 | 15462 | yes |
| KD alone (124 M) | `kd-rbase-v2-dqf-full-bias015` | 64.69 | - | - | - | - | - | - | 372.7 | 120.9 | 14064 | yes |

Bold = within the limits and at least 0.2 above the frozen model on the 19 tasks. A dash in the 19-task column = no heldout8 yet (the candidate was read on tiny10 and trec-covid only).

## p0: the release package of 17.11

`datasets/transfer/release-20261117` (2,3G, 126 files; `sha256sum -c` passes) next to `release-20261113`, which was not changed. The swaps the user approved on 16.11:

| path | release-20261113 | this version |
|---|---|---|
| `onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/mixed-int8` | broken int8 export of the ALBERT: cosine to fp32 0.7499, tiny10 with int8 queries 62.1 (torch 64.34), 120 MB | **replaced** by the repaired export (the extra head layer and the FFN output of group 2 stay fp32 next to the MLM head): cosine 0.9980, tiny10 64.3, 132 MB; report `onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/int8_sensitivity.json` |
| `onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/mixed-int8-int8head` | - | **added**: the repaired export with the MLM head in int8 as well: cosine 0.9948, tiny10 64.29, 95 MB |
| `onnx/full-soup-v20-dqf-bias015/dynamic-int8` | - | **added**: the full model with every node in int8, the MLM head included: cosine 0.9700, tiny10 64.88, 226 MB (mixed-int8 with the head in fp32, unchanged: cosine 0.9741, tiny10 64.91, 338 MB; torch 65.06) |
| `static/` (the table, the Sentence Transformers model, the query side alone) | table `phr2-k8-kd-r2`: asymmetric tiny10 / heldout8 60.74 / 39.24, 89.7 dims/query, 11764 postings/query | **replaced** by table `phr2-k8-kd-r2-mw020` (the same table with weights below 0.20 dropped, no training): 60.67 / 39.20, 75.9 dims/query, 9651 postings/query (-18 %) |
| `SHA256SUMS` | lists a stray `./SHA256SUMS.tmp`, so `sha256sum -c` reports one unreadable file | no entries of `SHA256SUMS*`; `sha256sum -c` passes |

`logs/final_static_table.txt` names `models/static-soup-v20-dqf-bias015/phr2-k8-kd-r2-mw020`. Difference for the 5070: `datasets/transfer/release-20261117-diff-from-20261113.tar` (823M, 62 files; how to apply it is in the README of the package). The static-table benchmark on the 5070 has to be run again for the new table.

## Queue state (b2)

```
```
