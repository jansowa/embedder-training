| variant | warm-up s | graphs (+timed) | q p50 ms | q/s (b64) | passages/s (b16) | passages/s (b32) | speed-up | CUDA MB | cos q | cos p | Jacc p | dims q | dims p | postings q | top10 sym | top10 q-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16:eager | 17 | 0 (+0) | 3.8 | 2685 | 561 | 553 | 1.00x | 3313 | ref | ref | ref | 84.4 | 334.1 | 8371 | ref | ref |
| fp16:compile | 48 | 8 (+0) | 2.9 | 3040 | 627 | 619 | 1.12x | 3313 | 1.0000 | 1.0000 | 0.997 | 84.4 | 334.1 | 8369 | 0.998 | 0.999 |
| fp16:fasthead,eager | 16 | 0 (+0) | 3.8 | 2909 | 604 | 595 | 1.08x | 1784 | 1.0000 | 1.0000 | 1.000 | 84.4 | 334.1 | 8371 | 1.000 | 1.000 |
| fp16:fasthead,body,compile | 33 | 8 (+0) | 3.0 | 2976 | 616 | 603 | 1.10x | 1783 | 1.0000 | 1.0000 | 0.998 | 84.4 | 334.1 | 8370 | 0.999 | 0.999 |
| int8-dyn:compile | failed: exit 1: RuntimeError: mat2.size(1) needs to be greater than 0 and a multiple of 8, but got 50001 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

GPU NVIDIA GeForce RTX 3060 Ti (cc 8.6), torch 2.7.1+cu128, torchao 0.15.0; speed-up = best passages/s over the reference's best; postings q = in-sample, symmetric.
