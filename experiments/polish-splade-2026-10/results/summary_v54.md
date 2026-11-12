# v54 (morning 12.11, quantized regime, nDCG@10): joint training with the static query table, a query prefix

## Query prefix "zapytanie: " on the mined arm (trained and measured with the prefix on queries only; first bias form with at most 410 dims/doc)

| model | tiny10 | heldout8 | trec-covid | dims/doc |
|---|---:|---:|---:|---:|
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | 70.15 | 405.3 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 | 64.67 | 43.35 | 69.86 | 358.0 |
| arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias015 | 64.78 | 43.41 | 68.98 | 372.3 |
| arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias015, queries without the prefix | 64.7 | - | - | 372.3 |

## Static query table phr2-k8-kd-r2 over a jointly trained index (untouched index of r2v12-bias005: 60.20; that model alone 64.21)

| joint arm | static queries, tiny10 | static queries, heldout8 | document model alone, tiny10 | dims/doc |
|---|---:|---:|---:|---:|
| phr2-frozen | 59.87 |  | 63.84 | 405.8 |
| phr2-both | 59.86 |  | 63.83 | 406.3 |

```
{
  "dims_per_doc": 431.3,
  "dims_per_query": 65.8,
  "mrr": 1.0,
  "positive_first": 1.0,
  "rows": 400,
  "top1_agreement": 0.9925
}
```

## Queue state (v54)

```
joint:r2v12b:phr2-both	ok	-	3713s
regime:quantized:asym-joint-phr2-both:tiny10	ok	59.86 10	546s
sparsity:joint-phr2-both-doc	ok	406.3	23s
regime:quantized:joint-phr2-both-doc-sym:tiny10	ok	63.83 10	582s
train:arm-dense-kl-n5kj-mined-qp-ens400-r2v12	ok	-	4910s
sparsity:arm-dense-kl-n5kj-mined-qp-ens400-r2v12	ok	536.2	22s
sparsity:arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias005	ok	476.4	21s
sparsity:arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias010	ok	421.8	22s
sparsity:arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias015	ok	372.3	21s
regime:quantized:arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias015:tiny10	ok	64.78 10	563s
regime:quantized:arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias015:heldout8	ok	43.41 8	2023s
regime:quantized:arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias015:heldout2_treccovid	ok	68.98 1	1271s
regime:quantized:arm-dense-kl-n5kj-mined-qp-ens400-r2v12-bias015-noprefix:tiny10	ok	64.7 10	593s
```
