# r1 (quantized regime, nDCG@10): the second round of the ALBERT student — partial result

Round 2 = from `arm-hl-full-s3n5kj` (student cut from soup-v9b, KD from soup-v9b, stage 3 on n5 + kj) with the recipe of the full model's second round (2 epochs, lr 2e-5, 2 x 16, 512 tokens, KL to the ensemble scores stored in the rows) on the rows of the full model's dose arm (`datasets/arms/n5-plus-kj-mined-dose3.jsonl`) and, as a second arm, on dose3 + faq20 (`datasets/arms/n5-plus-kj-mined-dose3-faq20.jsonl`). Every model is measured in its first bias form with at most 410 dims/doc (steps of 0.05).

**The queue was cut short on 14.11.** The user decided that the ALBERT to release must descend from the frozen full model (`soup-v20-dqf-bias015`), not from soup-v9b: a new lineage (queue a1: student cut from the frozen model, KD from it, stage 3, round 2). r1 was stopped by hand at 08:41 during the heldout8 of the first arm, and queue r1b added only the tiny10 of the second arm and the query-side cost. Not measured for the second arm: heldout8, trec-covid; for neither arm: held-out-2, the asymmetric run. No ALBERT was chosen here and `logs/final_albert_model.txt` was not written; the choice is made by a1, which measures what is missing for the better arm of this table.

| model | label | rows of the last training | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage 3 on n5 + kj (v45), the reference | `arm-hl-full-s3n5kj-bias010` | n5 + kj (20 292) | 63.77 | 42.33 | 64.23 | 374.6 | 381.5 | 146.1 | 17437 |
| round 2 on dose3 (r1) | `arm-hl-full-s3n5kj-r2d3-bias020` | dose3 (60 385) | 64.14 | 42.61 | - | 359.6 | 365.4 | 124.3 | 13397 |
| round 2 on dose3 + faq20 (r1) | `arm-hl-full-s3n5kj-r2d3faq-bias020` | dose3 + faq20 | 64.09 | - | - | 370.4 | 376.9 | 133.8 | 14790 |
| frozen full model (82 M), for scale | `soup-v20-dqf-bias015` | - | 65.06 | 43.64 | 72.48 | 371.4 | 378.5 | 123.0 | 13945 |

Unbiased dims/doc: reference 487.5, round 2 on dose3 759.5, round 2 on dose3 + faq20 803.6: the second round makes the student much denser, so its measured form needs a larger bias shift than the reference (bias010).

tiny10 against the reference at <= 410 dims/doc: round 2 on dose3 0.37, round 2 on dose3 + faq20 0.32. The better arm on tiny10 gives the rows of the second round of a1: **dose3**. One seed; the seed noise of this recipe is 0.13-0.2.

Training time: round 2 on dose3 9 686 s, on dose3 + faq20 12 299 s.

## Queue state (r1, r1b)

```
train:arm-hl-full-s3n5kj-r2d3	ok	-	9686s
train:arm-hl-full-s3n5kj-r2d3faq	ok	-	12299s
sparsity:arm-hl-full-s3n5kj-bias005	ok	427.8	20s
sparsity:arm-hl-full-s3n5kj-bias010	ok	374.6	19s
regime:quantized:arm-hl-full-s3n5kj-bias010:tiny10	ok	63.77 10	497s
regime:quantized:arm-hl-full-s3n5kj-bias010:heldout8	ok	42.33 8	1883s
regime:quantized:arm-hl-full-s3n5kj-bias010:heldout2_treccovid	ok	64.23 1	1007s
sparsity:arm-hl-full-s3n5kj-r2d3	ok	759.5	20s
sparsity:arm-hl-full-s3n5kj-r2d3-bias005	ok	630.8	20s
sparsity:arm-hl-full-s3n5kj-r2d3-bias010	ok	522.7	20s
sparsity:arm-hl-full-s3n5kj-r2d3-bias015	ok	433.2	19s
sparsity:arm-hl-full-s3n5kj-r2d3-bias020	ok	359.6	19s
regime:quantized:arm-hl-full-s3n5kj-r2d3-bias020:tiny10	ok	64.14 10	462s
regime:quantized:arm-hl-full-s3n5kj-r2d3-bias020:heldout8	ok	42.61 8	finished after the stop of r1
sparsity:arm-hl-full-s3n5kj-r2d3faq	ok	803.6	18s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias005	ok	663.3	19s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias010	ok	545.9	19s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias015	ok	449.1	19s
sparsity:arm-hl-full-s3n5kj-r2d3faq-bias020	ok	370.4	18s
regime:quantized:arm-hl-full-s3n5kj-r2d3faq-bias020:tiny10	ok	64.09 10	462s
```
