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
