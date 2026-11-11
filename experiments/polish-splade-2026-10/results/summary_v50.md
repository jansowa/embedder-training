# v50 (evening 10.11, quantized regime, nDCG@10): the student-mined arm at lower index cost (bias forms)

biasT = the checkpoint with lm_head.bias shifted by -(e^T - 1). r2v12 = the same recipe on the old negatives.

| model | tiny10 | heldout8 | trec-covid | dims/doc |
|---|---:|---:|---:|---:|
| arm-dense-kl-n5kj-ens400-r2v12 | 64.2 | 43.15 | 67.02 | 404.4 |
| arm-dense-kl-n5kj-ens400-r2v12-bias005 | 64.21 | 43.08 | 66.02 | 364.9 |
| arm-dense-kl-n5kj-ens400-r2v12-bias010 | 64.24 | 43.0 | 65.35 | 328.4 |
| arm-dense-kl-n5kj-mined-ens400-r2v12 | 64.48 | 43.6 | 71.48 | 515.4 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias010 | 64.63 | 43.45 | 70.15 | 405.3 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias015 | 64.67 | 43.35 | 69.86 | 358.0 |
| arm-dense-kl-n5kj-mined-ens400-r2v12-bias020 |  |  |  | 315.3 |

## Queue state (v50)

```
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12-bias010	ok	405.3	23s
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12-bias015	ok	358.0	22s
sparsity:arm-dense-kl-n5kj-mined-ens400-r2v12-bias020	ok	315.3	22s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias015:tiny10	ok	64.67 10	580s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias015:heldout8	ok	43.35 8	2070s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias015:heldout2_treccovid	ok	69.86 1	1286s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias010:tiny10	ok	64.63 10	611s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias010:heldout8	ok	43.45 8	2099s
regime:quantized:arm-dense-kl-n5kj-mined-ens400-r2v12-bias010:heldout2_treccovid	ok	70.15 1	1285s
```
