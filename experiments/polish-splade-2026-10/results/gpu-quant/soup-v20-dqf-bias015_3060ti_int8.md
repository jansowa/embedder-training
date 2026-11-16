| variant | warm-up s | graphs (+timed) | q p50 ms | q/s (b64) | passages/s (b16) | passages/s (b32) | speed-up | CUDA MB | cos q | cos p | Jacc p | dims q | dims p | postings q | top10 sym | top10 q-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16:eager | 16 | 0 (+0) | 3.8 | 2805 | 581 | 577 | 1.00x | 3313 | ref | ref | ref | 84.4 | 334.1 | 8371 | ref | ref |
| int8-dyn:nohead,compile | failed: exit 1: RuntimeError: _apply(): Couldn't swap Linear.weight |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| int8-dyn:fasthead,body,nohead,compile | failed: exit 1: RuntimeError: _apply(): Couldn't swap Linear.weight |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

GPU NVIDIA GeForce RTX 3060 Ti (cc 8.6), torch 2.7.1+cu128, torchao 0.15.0; speed-up = best passages/s over the reference's best; postings q = in-sample, symmetric.
