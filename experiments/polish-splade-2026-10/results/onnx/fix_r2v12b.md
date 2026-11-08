# Int8 repair attempt for r2v12b

fp32 checks on 128 held-out kd_v1 texts (64 queries, 64 passages). orig = the first export (no reduce_range, s8 weights, minmax).

| config | variant | cosine | active Jaccard | top-10 terms | dims ratio | MB | fp32 blocks |
|---|---|---:|---:|---:|---:|---:|---|
| orig | dynamic-int8 | 0.0283 | 0.017 | 0.005 | 2.715 | 227 |  |
| orig | static-int8 | 0.1433 | 0.022 | 0.174 | 26.489 | 227 |  |
| orig | mixed-int8 | 0.9173 | 0.553 | 0.836 | 0.874 | 358 | layer.4,head |
| rr | dynamic-int8 | 0.9627 | 0.668 | 0.898 | 0.817 | 227 |  |
| rr | static-int8 | 0.7474 | 0.306 | 0.636 | 0.852 | 227 |  |
| rr | mixed-int8 | 0.9163 | 0.547 | 0.820 | 0.853 | 358 | layer.4,head |
| u8 | dynamic-int8 | 0.9671 | 0.691 | 0.898 | 0.873 | 227 |  |
| u8 | static-int8 | 0.7480 | 0.304 | 0.627 | 0.892 | 227 |  |
| u8 | mixed-int8 | 0.9188 | 0.552 | 0.837 | 0.874 | 358 | layer.4,head |
| rr-pct | dynamic-int8 | 0.9627 | 0.668 | 0.898 | 0.817 | 227 |  |
| rr-pct | static-int8 | 0.7697 | 0.334 | 0.594 | 0.906 | 227 |  |
| rr-pct | mixed-int8 | 0.9361 | 0.596 | 0.853 | 0.889 | 358 | layer.4,head |

## PIRB tiny10, queries ONNX on the CPU, index torch (leaderboard regime)

| config | variant | tiny10 |
|---|---|---:|
| torch fp16 | - | 64.21 |
| orig | fp32 | 64.22 |
| orig | dynamic-int8 | 0.25 |
| orig | static-int8 | 13.66 |
| orig | mixed-int8 | 63.69 |

## Dynamic mixed (u8 dynamic int8, MLM head fp32, no calibration)

| cosine | active Jaccard | top-10 terms | dims ratio | MB | tiny10 (q-only) |
|---:|---:|---:|---:|---:|---:|
| 0.9700 | 0.702 | 0.895 | 0.833 | 338 | 64.14 |
