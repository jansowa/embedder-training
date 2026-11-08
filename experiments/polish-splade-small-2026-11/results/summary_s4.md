# s4 (08.11.2026 08:45, quantized regime, nDCG@10): static query encoders over the index of arm-dense-kl-n5kj-ens400

Document side frozen (models/arm-dense-kl-n5kj-ens400, symmetric tiny10 64.14). Query side = a lookup table (distill.static_query): idf = BM25 idf weights of 300 k kd_v1 passages (k = 1, only the query's own tokens); credit-k32 = expansion table credited from the document model on the kd_v1 queries (32 slots per token). -rows = trained on the ensemble scores of n5-plus-kj-tens400.jsonl (KL over each row's candidates); -mix = 0.5 symmetric SPLADE with in-batch passages + 0.5 row scores. Validation on 1 000 held-out pool rows with quantized query weights; postings fraction = sum of df / N over the query's dims on the pool's 54605 passages (x corpus size = postings per query).

```
query side              tiny10  heldout8  val mrr    tau  dims/q  post.frac
idf                      55.76         -        -      -       -          -
idf-rows                 55.76         -   0.9657  0.495    13.3      0.729
credit-k32               59.20         -        -      -       -          -
credit-k32-rows          59.31         -   0.9912  0.669   213.8     23.980
credit-k32-mix           59.41         -   0.9932  0.669   183.9     21.970
symmetric n5kj           64.14         -   0.9962  0.737   105.0     12.586
```

Init tables (idf, credit-k32) carry no validation of their own; their val numbers are the val_init of the -rows runs.

## Queue state (s4)

```
static_build:idf	ok	-	38s
static_build:credit-k32	ok	-	143s
static_pool:n5kj	ok	-	106s
static_train:idf-rows	ok	-	18s
static_train:credit-k32-rows	ok	-	17s
static_train:credit-k32-mix	ok	-	17s
regime:quantized:asym-n5kj-q-static-idf:tiny10	ok	55.76 10	525s
regime:quantized:asym-n5kj-q-static-idf-rows:tiny10	ok	55.76 10	522s
regime:quantized:asym-n5kj-q-static-credit-k32:tiny10	ok	59.2 10	755s
regime:quantized:asym-n5kj-q-static-credit-k32-rows:tiny10	ok	59.31 10	680s
regime:quantized:asym-n5kj-q-static-credit-k32-mix:tiny10	ok	59.41 10	663s
regime:quantized:asym-n5kj-q-static-credit-k32-mix:heldout8	FAILED	-	1214s
```
