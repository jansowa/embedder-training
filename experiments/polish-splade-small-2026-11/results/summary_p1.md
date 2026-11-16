# p1 (15.11.2026, quantized regime, nDCG@10): the static query table in its released format, in the pause of queue b1

Run in the pause of queue b1 after its KD (14:26-17:02, 155 minutes on the 3060 Ti; every step one job on the card). Query side: a static table `static-sparse-query/1` (153 343 rows x 8 slots); index: the frozen full model `soup-v20-dqf-bias015` (symmetric 65.06 / 43.64, 123.0 dims/query, 13 945 postings/query). Released table: `phr2-k8-kd-r2` (st1: 60.74 / 39.24). Cost columns: the cost sample of `scripts/splade_index_cost_asym.py` (1 350 passages + 1 350 queries), now also per task; limit 16 807 postings/query.

**Result: no candidate for a swap in the release package.** The best table within the limit is +0.06 on tiny10 over the released one; the rule asks for +0.3. The released table stays.

## Candidates

| table | what | tiny10 | vs released | dims/query | postings/query | arguana-pl: dims / postings | other tasks: postings | within the limit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `phr2-k8-kd-r2` | the released table (128 query tokens) | 60.74 | | 89.7 | 11 764 | 375.9 / 43 673 | 7 775 | yes |
| `phr2-k8-kd-r2-len512` | W1: reads 512 query tokens, rows unchanged | 60.74 | 0.00 | 118.2 | 14 253 | 632.2 / 66 078 | 7 775 | yes |
| `a1-sparseadam` | A1: sparse_adam | 60.79 | +0.05 | 94.0 | 12 258 | 394.8 / 45 727 | 8 074 | yes |
| `a1-adagrad` | A1: adagrad, lr 5e-2 | 60.75 | +0.01 | 98.0 | 12 780 | 410.7 / 47 632 | 8 424 | yes |
| `a1-floor` | A1: value floor 1e-4 | 60.77 | +0.03 | 133.8 | 17 682 | 521.1 / 60 613 | 12 316 | NO |
| `a1-sparseadam-mw005` | A3: weights below 0.05 dropped | 60.80 | +0.06 | 92.4 | 12 091 | 387.5 / 44 991 | 7 979 | yes |
| `a1-sparseadam-mw010` | A3: weights below 0.10 dropped | 60.80 | +0.06 | 88.4 | 11 271 | 372.5 / 43 202 | 7 280 | yes |
| `a1-sparseadam-mw020` | A3: weights below 0.20 dropped | 60.74 | 0.00 | 77.5 | 9 881 | 328.2 / 38 325 | 6 326 | yes |
| `a1-sparseadam-mw005-g1-kd` | A2 round 1: up to 2 new slots per row, 3 epochs | 60.71 | -0.03 | 97.6 | 12 050 | 410.5 / 46 282 | 7 771 | yes |

Seed noise on tiny10 is 0.13-0.2, so every difference in this table is inside it.

## W1: 512 query tokens

Only arguana-pl has queries longer than 128 tokens. Reading 512 tokens moves arguana-pl from 42.73 to 42.76 and nothing else, while its queries grow from 376 to 632 active dimensions and from 43 673 to 66 078 postings (all tasks: 11 764 -> 14 253 postings/query, still within the limit). The truncation is not what costs arguana-pl 7.4 points against the full model; the longer reading only adds cost. The next steps therefore started from the 128-token reading.

## W2 / W3: oracles

On 2 000 held-out queries of pool r2 (agreement with the teacher's ranking):

| query side | top-1 | Kendall tau | MRR | dims/query | postings fraction |
|---|---:|---:|---:|---:|---:|
| the table (student) | 0.6850 | 0.4233 | 0.7879 | 61.0 | 7.203 |
| oracle `weights` (the table's dimensions with the teacher's weights) | 0.7905 | 0.6552 | 0.8672 | 41.7 | 4.739 |
| oracle `support` (the table's weights on the dimensions the teacher also has) | 0.7140 | 0.4799 | 0.8086 | 41.7 | 4.739 |
| oracle `own` (the teacher's weights on the query's own tokens only) | 0.3950 | 0.2576 | 0.5329 | 7.9 | 0.503 |
| oracle `student-own` (the table's weights on the query's own tokens only) | 0.3855 | 0.2257 | 0.5231 | 10.1 | 0.765 |
| teacher | 1.0000 | 1.0000 | 1.0000 | 83.4 | 8.390 |

On tiny10 (the oracles stand on the 512-token table, 60.74; the full model 65.06):

| oracle | tiny10 | vs the table |
|---|---:|---:|
| `weights` | 62.98 | +2.24 |
| `support` | 61.57 | +0.83 |
| `own` | 55.97 | -4.77 |
| `student-own` | 52.05 | -8.69 |

Reading by the rules of the budget file: `weights` is below 64, so context-dependent weights on the table's own dimensions recover about half of the 4.3 points and the other half (65.06 - 62.98 = 2.1) needs dimensions the table does not produce. `support` is below 63: removing false expansions alone is worth less than a point, so sums with signs, thresholds and pruning have a low ceiling (A3 confirms: +0.01 at best). `own` is 9 points below the full model: the expansion on the query side matters a lot, weighting the query's own tokens is far from enough.

## A1-A4: the released architecture

- **A1** (six trainings of 3 epochs from `phr2-k8-kd` on pool r2; validation top-1 against the teacher on 2 000 queries; the released table has 0.6850): baseline 0.6865 (best step 3 500 of 5 529), sparse_adam 0.6925, adagrad 0.6920, value floor 0.6910, dimension gain 0.6875, temperature 2 0.6870. On the training rows the tables reach 0.69-0.76, so there is some over-fitting (baseline 0.762 against 0.687), but removing it does not move tiny10: the three arms measured on tiny10 are within +0.05 of the released table.
- **A3** (table weights below a threshold dropped, on `a1-sparseadam`): 0.05 and 0.10 keep tiny10 (60.80), 0.20 gives 60.74 with 16 % fewer postings than the released table (9 881 against 11 764). A cheaper table at the same quality is available; a better one is not.
- **A2** (the condition held: `weights` is 2.2 above the best table): slot learning, round 1, gives 60.71 (-0.09 against its start). Round 2 (a fresh pool) was not run.
- **A4** was skipped: no candidate reaches +0.3.

What this says about the next step: the released format is at its ceiling at this size (hygiene, thresholds and new slots all end within the noise). The oracles point to context-dependent weighting (path B3, the context gate; ceiling about +2.2) rather than more rows or pruning. Path B was not run here.

## GPU speed of the frozen full model (3060 Ti, short list)

`release.gpu_bench`, 1 024 queries and 4 096 passages of kd_v1, 3 repeats; agreement against fp16 eager.

| variant | query p50 ms | queries/s (batch 64) | passages/s (batch 16) | passages/s (batch 32) | speed-up | CUDA MB | cosine q / p | Jaccard p | dims/query | dims/passage | postings/query | top-10 sym | top-10 q-only |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| fp16 eager | 3.8 | 2 685 | 561 | 553 | 1.00x | 3 313 | ref | ref | 84.4 | 334.1 | 8 371 | ref | ref |
| fp16 compiled | 2.9 | 3 040 | 627 | 619 | 1.12x | 3 313 | 1.0000 / 1.0000 | 0.997 | 84.4 | 334.1 | 8 369 | 0.998 | 0.999 |
| fp16 fast head, eager | 3.8 | 2 909 | 604 | 595 | 1.08x | 1 784 | 1.0000 / 1.0000 | 1.000 | 84.4 | 334.1 | 8 371 | 1.000 | 1.000 |
| fp16 fast head, compiled body | 3.0 | 2 976 | 616 | 603 | 1.10x | 1 783 | 1.0000 / 1.0000 | 0.998 | 84.4 | 334.1 | 8 370 | 0.999 | 0.999 |
| torchao int8 (dynamic) | failed | | | | | | | | | | | | |

- The fast head halves the memory on the card (1 784 against 3 313 MiB) with identical vectors and 8 % more passages per second; compiling adds about 12 % and a warm-up of 33-48 s.
- torchao `int8-dyn` failed on the MLM decoder (`mat2.size(1) needs to be ... a multiple of 8, but got 50001`: the vocabulary size). The variants that leave the head in fp16 (`int8-dyn:nohead`) were not on the short list.
- ONNX fp32 with the CUDA provider failed at passage batch 32: the graph of the release builds the logits tensor (rows x tokens x 50 001) before the maximum and asked for 4.06 GB in one piece. It is a memory limit of the 8 GB card, not a driver problem; the TensorRT steps were skipped by the queue's rule.
- Second pass (16.11 01:42-01:48, in the pause of queue b1c after the R-a training): it added no number, both faults are in the benchmark code.
  - torchao `int8-dyn:nohead` (with and without the fast head): `RuntimeError: _apply(): Couldn't swap Linear.weight` ("Cannot swap t1 because it has weakref associated with it"). `SparseEncoder.encode` calls `self.to(device)` on every call, and after the body is quantized and compiled the weights can no longer be swapped. The benchmark has to stop that call for a quantized model; not fixed yet. fp16 eager repeated its numbers (3.8 ms, 2 805 queries/s, 581 passages/s).
  - ONNX with the CUDA provider asked for the same 4 057 681 152 bytes at passage batch 8 and with the max-first graph. The request comes from the agreement pass of `release.onnx_bench`, which encodes in chunks of 16 texts whatever `--passage-batch` says: 16 rows x 317 tokens x 50 001 floats are 1.01 GB and `ReduceMax` of the CUDA provider asks for four times its input. The chunk has to follow the passage batch (4 rows would ask for about 1 GB); not fixed yet. TensorRT was not reached.
- Nothing here is a model-card number yet: variants faster than fp16 eager need PIRB (tiny10) first.

## ONNX int8 of the released ALBERT (`arm-hl-dqf-full-s3n5kj-r2d3-bias020`)

`release.int8_sensitivity` on the fp32 export of the package, 128 texts of kd_v1 (dynamic int8, u8 weights; the MLM head stays fp32 in every row):

| form | int8 nodes | cosine to fp32 | active Jaccard | dims ratio | size |
|---|---:|---:|---:|---:|---:|
| the released mixed int8 | 29 | 0.7499 | 0.2103 | 4.732 | 126 MB |
| only the extra head layer (group 4) in int8 | 4 | 0.7272 | 0.1855 | 5.419 | 183 MB |
| group 4 back in fp32 | 25 | 0.9931 | 0.8503 | 0.889 | 135 MB |
| group 4 and the FFN output of group 2 back in fp32 (**the repaired variant**) | 24 | 0.9980 | 0.9197 | 1.000 | 139 MB |

- Cause: the layer added on top of the four shared ALBERT groups ("head layer", group 4). Every other group alone in int8 keeps cosine >= 0.994. The second, small fault is the FFN output of group 2: its input range is [-0.17, 54.4] with 97 % of the values below one quantization step, so most of the signal rounds to zero (cosine 0.9943, 12 % fewer active dimensions).
- The quantizer of onnxruntime cannot quantize a single node of group 4 alone (an assertion on the scale), so the tool repairs that group as a whole.
- tiny10 with the repaired int8 queries over the torch index: **64.30** (torch 64.34; the released int8 62.10).
- The repaired variant is in `runs/onnx-int8-diag/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/mixed-int8-repaired` (139 MB against 126 MB). The package was not touched; replacing its int8 files is the user's decision.

## p2 (16.11 01:48-03:24, 96 minutes, in the pause of queue b1c after the R-a training)

Queue `queues/queue_p2.sh`, run by `logs/p2_chain.sh`. Everything it produced for the release lies next to the package, in `datasets/transfer/release-20261113-update1` (497 MB, 42 files in `SHA256SUMS`, a README with the differences); the package itself was not touched and applying a path is the user's decision.

### The repaired int8 export of the ALBERT

`onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/mixed-int8` of the update directory is the repaired variant of the table above (24 int8 nodes, cosine 0.9980, 136 MB against 124 MB, tiny10 with int8 queries 64.30 against 62.10), with `int8_sensitivity.json` next to it. heldout8 with int8 queries was not measured for either version.

### A cheaper static table of the same quality

Rule: tiny10 and heldout8 each at most 0.10 below the released table (60.74 / 39.24) and at least 500 postings/query cheaper. Both candidates pass; the cheaper one is the released table itself with weights below 0.20 dropped, no training.

| table | tiny10 | heldout8 | dims/query | postings/query | arguana-pl: dims / postings | other tasks: postings | postings/query on the heldout8 tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `phr2-k8-kd-r2` (released) | 60.74 | 39.24 | 89.7 | 11 764 | 375.9 / 43 673 | 7 775 | 7 558 |
| `phr2-k8-kd-r2-mw020` (**chosen**) | 60.67 | 39.20 | 75.9 | 9 651 | 322.3 / 37 628 | 6 154 | 6 115 |
| `a1-sparseadam-mw020` | 60.74 | 39.17 | 77.5 | 9 881 | 328.2 / 38 325 | 6 326 | 6 206 |

- 18 % fewer postings per query on the cost sample (19 % on the heldout8 tasks) at -0.07 / -0.04, inside the seed noise. No task moves by more than 0.20 (poleval allegro-faq B 71.84 -> 71.64, legal questions 76.47 -> 76.67).
- Query encoding time does not change (a table lookup); the gain is the number of postings the index has to read.
- The table, its Sentence Transformers model (`release.st_export`, loaded with `trust_remote_code=True` as a check) and the query side alone are in `static/` of the update directory. If the table is swapped, the static-table benchmark on the 5070 has to be run again for it.

### One probe of the context gate (path B3, format 2, d = 32)

On the released table: `distill.static_v2 convert --ctx-dim 32`, a fit of the gate to the teacher's query vectors (400 000 kd_v1 queries, 2 epochs: cosine 0.8929, recall of the teacher's dimensions 0.537), then 3 epochs of ranking training on pool r2.

| query side | val top-1 | tiny10 | dims/query (median) | postings/query (median, p95) | folder |
|---|---:|---:|---:|---:|---:|
| `phr2-k8-kd-r2` (released) | 0.6850 | 60.74 | 89.7 | 11 764 | 16 MB |
| `gate32-kd` | 0.7105 | 60.85 | 84.9 (49) | 11 222 (7 303, 42 333) | 42 MB |
| oracle `weights` | 0.7905 | 62.98 | | | |

- The gate moves the agreement with the teacher on the validation queries (+0.0255 top-1, a quarter of the way to the oracle) but tiny10 only by +0.11, inside the seed noise; of the oracle's +2.24 it recovers nothing measurable. Per task: allegro-faq A +0.66, scifact-pl +0.72, allegro-faq B -0.39, the rest within 0.25.
- One probe with one size; it says that d = 32 with this training does not reach the oracle, not that context weighting cannot. The format-2 encoder has no Sentence Transformers module, so it could not enter the 26.11 package in any case.

## Files

State and cost files (p2: `cost_p2.json`, `cost_h8.json`, `cost_b3.json`, `static2-gate32-kd.json`): `results/static-soup-v20-dqf-bias015/p1/`; training summaries `results/static-soup-v20-dqf-bias015/a1-*.json`; oracles `oracles_phr2-k8-kd-r2.json`; GPU table `experiments/polish-splade-2026-10/results/gpu-quant/soup-v20-dqf-bias015_3060ti.json`. Queue: `queues/queue_p1{a,b,c}.sh` with `p1_common.sh`, run by `logs/p1_chain.sh`.
