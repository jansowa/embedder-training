# s5 (08.11.2026 10:13, quantized regime, nDCG@10): static query tables trained on a mined pool, index of arm-dense-kl-n5kj-ens400 frozen

Pool (cache/static_pools/kd-n5kj): 120000 kd_v1 queries over 250000 kd_v1 passages; per query the teacher's top 32 plus the top 16 of each of idf, credit-k32 (43.67 candidates, 11.67 found by the students only); recall of the teacher's top 10 in the student's top 16: {'idf': 0.4115, 'credit-k32': 0.7246}.
Training: KL(teacher || student) over each query's candidates, teacher = q_splade . d of n5kj (no labels), a learned global score scale, 2 epochs; -kd = trained on this pool. Validation: 2 000 held-out queries, top-1 agreement = the student's top candidate is the teacher's, quantized query weights.

```
query side                  tiny10  heldout8  val top1  val tau  dims/q  post.frac   scale
BM25 pyserini (leaderb.)     50.07     33.34
BM25 bm25s (leaderb.)        53.75     36.17
polish-splade (leaderb.)     61.53     41.06
n5kj symmetric               64.14     43.01
idf                          55.76         -     0.365   -0.046    13.5      0.853       -
idf-kd                       55.62         -     0.379    0.020    12.0      0.749   0.261
credit-k8                    59.19         -     0.604    0.406   102.1     15.660       -
credit-k8-kd                 59.65         -     0.629    0.444    83.3     10.945   0.844
credit-k32                   59.20         -     0.620    0.425   333.0     40.037       -
credit-k32-rows              59.31         -     0.978    0.669   213.8     23.980       -
credit-k32-kd                59.48         -     0.644    0.470   145.5     16.035   0.844
n5kj symmetric (val)                             1.000    1.000   115.4     12.813
```

credit-k32-rows is s4's table (trained on the n5 + kj rows); its val numbers come from the rows pool, the others from the mined pool. Postings fraction = sum of df / N over the query's dims on the pool's passages (x corpus size = postings per query).

## Queue state (s5)

```
static_mine:smoke	ok	-	16s
static_mine:smoke	ok	-	15s
static_mine:kd-n5kj	ok	-	682s
static_train:idf-kd	ok	-	165s
static_train:credit-k32-kd	ok	-	168s
static_train:credit-k8-kd	ok	-	163s
regime:quantized:asym-n5kj-q-static-idf-kd:tiny10	ok	55.62 10	529s
regime:quantized:asym-n5kj-q-static-credit-k32-kd:tiny10	ok	59.48 10	609s
regime:quantized:asym-n5kj-q-static-credit-k8-kd:tiny10	ok	59.65 10	574s
regime:quantized:asym-n5kj-q-static-credit-k8:tiny10	ok	59.19 10	602s
```
