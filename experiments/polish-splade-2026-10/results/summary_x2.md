# x2 (16.11): int8 exports of the release models with the MLM head in int8

Queue `queues/queue_x2.sh`, run by `logs/p3_chain.sh` at a pause point of queue b1c (08:48-09:23, 35 min). The released
mixed int8 variants keep the MLM head (the projection onto the 50 001 vocabulary entries, about half of the
multiplications of the 6-layer model) in fp32. On the earlier model of the same architecture the full dynamic int8
encoded a query in 4.9 ms against 9.0 ms at 6 threads with the same PIRB numbers (`onnx/summary_r2v12b.md`), so the
same form was built and measured for the two release models. Built by `release.int8_sensitivity --build` from the fp32
graphs of the package (u8 weights, per channel, no calibration); agreement against the fp32 export on 128 texts of
kd_v1; tiny10 (quantized regime, nDCG@10) with the int8 queries over the torch index of the same model. **No CPU timing
was run**, so the speed gain on the release models is not measured.

## Result

| model | variant | int8 nodes | fp32 nodes | file | cosine to fp32 | active Jaccard | top-10 terms | dims ratio | tiny10 with int8 queries | torch |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `soup-v20-dqf-bias015` | mixed int8, head fp32 (released) | 36 | head (2) | 354.4 MB | 0.9741 | 0.7140 | 0.9047 | 0.8416 | 64.91 | 65.06 |
| `soup-v20-dqf-bias015` | **dynamic int8, every node** | 38 | none | 237.8 MB | 0.9700 | 0.6932 | 0.8992 | 0.8803 | **64.88** | 65.06 |
| `arm-hl-dqf-full-s3n5kj-r2d3-bias020` | mixed int8, head fp32 (released, broken) | 29 | head | 126.4 MB | 0.7499 | 0.2103 | 0.7945 | 4.7317 | 62.10 | 64.34 |
| `arm-hl-dqf-full-s3n5kj-r2d3-bias020` | repaired, head fp32 (p2) | 24 | head, 4 nodes of group 4, FFN output of group 2 (7) | 138.8 MB | 0.9980 | 0.9197 | - | 1.000 | 64.30 | 64.34 |
| `arm-hl-dqf-full-s3n5kj-r2d3-bias020` | **repaired, head int8** | 26 | 4 nodes of group 4, FFN output of group 2 (5) | 100.0 MB | 0.9948 | 0.8637 | 0.9539 | 1.0166 | **64.29** | 64.34 |

Both new variants are within 0.03 of the head-fp32 variant on tiny10 (the rule of the queue: at most 0.15 below), so
both were copied next to the package as additional paths:
`datasets/transfer/release-20261113-update1/onnx/full-soup-v20-dqf-bias015/dynamic-int8` and
`.../onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/mixed-int8-int8head`, each with its report (`.json` next to the
directory). Nothing in the package was replaced. The files are smaller by 33 % (full model, against the released mixed
variant) and 28 % (ALBERT, against the repaired one).

### tiny10 per task

| task | full: torch | head fp32 | every node int8 | int8 - head fp32 | ALBERT: torch | repaired, head fp32 | repaired, head int8 | int8 - head fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| maupqa-gpt3-cc | 18.89 | 18.84 | 18.84 | +0.00 | 18.62 | 18.62 | 18.62 | +0.00 |
| maupqa-mkqa | 65.71 | 65.39 | 65.32 | -0.08 | 64.75 | 64.80 | 64.74 | -0.06 |
| maupqa-poleval2021-pairs | 84.07 | 83.76 | 83.64 | -0.12 | 82.96 | 83.01 | 82.95 | -0.06 |
| maupqa-templates | 76.72 | 76.66 | 76.58 | -0.08 | 75.87 | 75.90 | 75.86 | -0.04 |
| poleval-2022-test-A-allegro-faq | 78.93 | 78.44 | 78.88 | +0.43 | 77.70 | 77.56 | 77.51 | -0.05 |
| poleval-2022-test-B-allegro-faq | 75.48 | 75.08 | 75.03 | -0.05 | 74.29 | 74.23 | 74.60 | +0.37 |
| arguana-pl | 50.09 | 49.95 | 49.53 | -0.43 | 49.15 | 49.16 | 49.17 | +0.01 |
| nfcorpus-pl | 32.76 | 32.84 | 32.97 | +0.13 | 32.78 | 32.75 | 32.70 | -0.05 |
| scifact-pl | 68.86 | 69.05 | 68.91 | -0.13 | 68.28 | 68.06 | 67.77 | -0.29 |
| gpt-exams | 99.12 | 99.10 | 99.10 | -0.01 | 99.01 | 98.96 | 98.97 | +0.01 |
| **mean** | **65.06** | **64.91** | **64.88** | **-0.03** | **64.34** | **64.30** | **64.29** | **-0.02** |

## Reading

- The head in int8 costs nothing measurable on tiny10 for either model (-0.03 and -0.02). The largest single moves are
  arguana-pl -0.43 for the full model (long queries; against torch it is -0.57) and scifact-pl -0.29 for the ALBERT
  (-0.52 against torch).
- The whole int8 cost of the full model stays where it was: -0.18 against torch with every node in int8, -0.15 with the
  head in fp32, and about 12 % fewer active dimensions than fp32 either way.
- The ALBERT keeps its agreement with fp32 (cosine 0.9948, dims ratio 1.017) as long as the extra head layer (group 4)
  and the FFN output of group 2 stay fp32; the MLM head was never the sensitive part.
- Not measured: CPU timing of the new variants (only on the user's go, on an idle machine), heldout8 with ONNX queries,
  int8 documents.

## Checksums of the update directory

The `SHA256SUMS` written by `queue_p2.sh` listed its own temporary file (`./SHA256SUMS.tmp`), so `sha256sum -c` reported
one unreadable file; the "42 files" of the p2 summary were 41 files and this line. The line was removed on 16.11
(`release-20261113-update1/SHA256SUMS`: 55 files, `sha256sum -c` passes) and the listing of both queues now skips
`SHA256SUMS*`. The `SHA256SUMS` of the package itself (`pack_sums` of `queues/release_common.sh`) has the same stray
line (line 74; 106 files + the line): the package is not changed without the user's decision, and
`release_common.sh` is fixed after queue b1c ends.

## Queue state (x2)

```
x2:build:full-soup-v20-dqf-bias015	ok	-	38s
regime:quantized:onnx-full-soup-v20-dqf-bias015-q-dynamic-int8:tiny10	ok	64.88 10	1169s
x2:build:albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020	ok	-	26s
regime:quantized:onnx-albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020-q-mixed-int8-int8head:tiny10	ok	64.29 10	877s
```
