# x1: ONNX exports of the released models and the package for the 5070

| model | export | fp32 vs torch: max abs diff | mixed int8 vs fp32: cosine | active Jaccard | top-10 terms | dims ratio | fp32 MB | mixed MB | tiny10, ONNX mixed-int8 queries over the torch index | tiny10 torch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `soup-v20-dqf-bias015` | `onnx/full-soup-v20-dqf-bias015` | 0.0001 | 0.9741 | 0.7140 | 0.9047 | 0.8416 | 459 | 338 | 64.91 | 65.06 |

Mixed int8 = dynamic int8 with u8 weights and the MLM head kept fp32 (`release.onnx_splade --variants fp32,mixed --mixed-method dynamic --weight-type u8`), no calibration. Earlier, for r2v12-bias005: cosine 0.9700, tiny10 64.14 against 64.21 in torch. No CPU timing was run; the commands are in the README of the package.

## Package

`datasets/transfer/release-20261113` (a link to the data drive), 1,5G:

```
bench	1,1M	benchmark inputs: 3 000 queries + 1 500 passages of kd_v1 for release.onnx_bench; the PIRB task sets
full/soup-v20-dqf-bias015	317M	full model (82 M), release form with the bias baked in; 65.06 / 43.64 / trec-covid 72.48, 371.4 dims/doc
onnx/full-soup-v20-dqf-bias015	804M	ONNX exports of soup-v20-dqf-bias015: fp32 and dynamic mixed int8 (u8 weights, MLM head fp32); mixed against fp32 cosine 0.9741; tiny10 with mixed-int8 queries 64.91
results	36K	summaries and choices of v62 / r1 / st1, the ONNX export reports
static/soup-v20-dqf-bias015-static-phr2-k8-kd-r2	16M	static query table phr2-k8-kd-r2 over the index of soup-v20-dqf-bias015 (static-sparse-query/1); asymmetric 60.74 / 39.24, dims/query 61.040
static/st-soup-v20-dqf-bias015-static-phr2-k8-kd-r2	331M	Sentence Transformers SparseEncoder: query route = the table, document route = soup-v20-dqf-bias015 (load with trust_remote_code=True)
static/st-soup-v20-dqf-bias015-static-phr2-k8-kd-r2-query	15M	the query side alone as a Sentence Transformers model (no document encoder)
```

## Queue state (x1)

```
onnx-export:full-soup-v20-dqf-bias015	ok	-	48s
regime:quantized:onnx-full-soup-v20-dqf-bias015-q-mixed-int8:tiny10	ok	64.91 10	1295s
```
