# st1 (quantized regime, nDCG@10): the static query table over the index of the frozen full model

The recipe of phr2-k8-kd-r2 (s7 / s8) repeated with `soup-v20-dqf-bias015` as the document side and the teacher: tables credited on the kd_v1 queries (token rows; rows for whole words and adjacent word pairs), a pool of 250 000 passages and 120 000 queries mined with the new index, 2 epochs, a second pool whose student candidates come from the trained tables, 3 epochs. The document encoder is frozen; the joint training of v54 is not repeated.

| query side | index | tiny10 | heldout8 | val top1 | dims/query | postings fraction |
|---|---|---:|---:|---:|---:|---:|
| **static phr2-k8-kd-r2 (st1)** | soup-v20-dqf-bias015 | 60.74 | 39.24 | 0.685 | 61.040 | 7.203 |
| static phr2-k8-kd (first pool only) | soup-v20-dqf-bias015 | - | - | 0.688 | 69.900 | 8.151 |
| static phr2-k8-kd-r2 (s7 / s8) | r2v12-bias005 | 60.20 | 38.60 | 0.671 | 69.8 | 8.440 |
| the full model itself (symmetric) | soup-v20-dqf-bias015 | 65.06 | 43.64 | 1.000 | | |
| r2v12-bias005 (symmetric) | r2v12-bias005 | 64.21 | 43.08 | 1.000 | | |

val columns: 2 000 held-out kd_v1 queries of the pool each table trained on (top-1 agreement with the full model; postings fraction as reported by distill.train_static, relative to its own reference). The val numbers of different pools are not comparable 1:1.

Released table: `models/static-soup-v20-dqf-bias015/phr2-k8-kd-r2` (first line of `logs/final_static_table.txt`). Sentence Transformers package: `models/st/soup-v20-dqf-bias015-static-phr2-k8-kd-r2` (router: query = the table, document = the frozen model; load with `trust_remote_code=True`) and `models/st/soup-v20-dqf-bias015-static-phr2-k8-kd-r2-query` (the query side alone). Package check: loaded with trust_remote_code; active dims per query [28, 33, 21], per document [266], score of the first query 9.54

## Query-side cost on the cost sample (added by hand on 14.11)

`scripts/splade_index_cost_asym.py` on the sample of `scripts/splade_index_cost.py` (1 350 passages + 1 350 queries of the tiny tasks; `results/index_cost_st1_asym.json`), the index built by `soup-v20-dqf-bias015`:

| query side | dims/query, mean | dims/query, median | postings/query, mean | median | p95 |
|---|---:|---:|---:|---:|---:|
| static phr2-k8-kd-r2 (st1) | 89.7 | 52 | 11 764 | 7 670 | 44 399 |
| the full model itself | 123.0 | | 13 945 | | |

The limit is 16 807 postings/query. The 61.0 dims/query of the table above is measured on the kd_v1 validation queries; the cost sample contains the long arguana-pl queries, hence the higher mean here.

## Queue state (st1)

```
static_build:soup-v20-dqf-bias015:credit	ok	-	137s
static_build:soup-v20-dqf-bias015:phr2	ok	-	191s
static_mine:kd-soup-v20-dqf-bias015	ok	-	693s
static_train:soup-v20-dqf-bias015:credit-k8-kd	ok	-	161s
static_train:soup-v20-dqf-bias015:phr2-k8-kd	ok	-	180s
static_mine:kd-soup-v20-dqf-bias015-r2	ok	-	669s
static_train:soup-v20-dqf-bias015:phr2-k8-kd-r2	ok	-	207s
regime:quantized:asym-soup-v20-dqf-bias015-q-static-phr2-k8-kd-r2:tiny10	ok	60.74 10	558s
regime:quantized:asym-soup-v20-dqf-bias015-q-static-phr2-k8-kd-r2:heldout8	ok	39.24 8	1934s
```
