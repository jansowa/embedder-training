# b2 (quantized regime, nDCG@10): a 124 M replacement of the frozen full model

The release of 26.11 has frozen versions; a stronger 124 M model (sdadas/polish-roberta-base-v2 after KD from the frozen model, `summary_b1.md`) found by Thursday 19.11 would replace the full model. Rule of the replacement: the highest mean over the 19 tasks of tiny10, heldout8 and trec-covid (every task with the same weight) with at most 410 dims/doc and at most 16807 postings/query, and a lead of at least 0.2 over `soup-v20-dqf-bias015` (56.435): the architecture changes and the encoding is about twice as slow. The winner then needs its 23 tasks not below the frozen model's (58.799). The decision is the user's. Every candidate is measured in its first form within both limits (no shift, 0.05, 0.10, then steps of 0.01). One seed each; the seed noise of the ranking recipe is 0.13-0.2 on tiny10.

## Candidates

| candidate | measured form | tiny10 | heldout8 | trec-covid | mean of 19 tasks | vs the frozen model | mean of 11 tasks (tiny10 + trec-covid) | vs the frozen model | dims/doc | dims/query | postings/query | within the limits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frozen full model (82 M) | `soup-v20-dqf-bias015` | 65.06 | 43.64 | 72.48 | 56.435 | reference | 65.738 | reference | 371.4 | 123.0 | 13945 | yes |
| **R-c' after the dose3 round** | `arm-b1-rc-d3-bias010` | 65.59 | 44.01 | 68.42 | 56.652 | +0.217 | 65.843 | +0.104 | 389.3 | 132.7 | 16355 | yes |
| R-a, denser form (bias014) | `arm-b1-ra-bias014` | 65.39 | 43.77 | 67.97 | 56.425 | -0.010 | 65.628 | -0.110 | 374.0 | 134.7 | 16400 | yes |
| R-a, bias015 | `arm-b1-ra-bias015` | 65.34 | 43.76 | 67.39 | 56.359 | -0.076 | 65.525 | -0.214 | 361.8 | 130.8 | 15462 | yes |
| KD alone (124 M) | `kd-rbase-v2-dqf-full-bias015` | 64.69 | - | 68.89 | - | - | 65.075 | -0.664 | 372.7 | 120.9 | 14064 | yes |

Bold = within the limits and at least 0.2 above the frozen model on the 19 tasks. A dash in the 19-task column = no heldout8 yet (the candidate was read on tiny10 and trec-covid only).

## p0: the release package of 17.11

`datasets/transfer/release-20261117` (2,3G, 126 files; `sha256sum -c` passes) next to `release-20261113`, which was not changed. The swaps the user approved on 16.11:

| path | release-20261113 | this version |
|---|---|---|
| `onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/mixed-int8` | broken int8 export of the ALBERT: cosine to fp32 0.7499, tiny10 with int8 queries 62.1 (torch 64.34), 120 MB | **replaced** by the repaired export (the extra head layer and the FFN output of group 2 stay fp32 next to the MLM head): cosine 0.9980, tiny10 64.3, 132 MB; report `onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/int8_sensitivity.json` |
| `onnx/albert-arm-hl-dqf-full-s3n5kj-r2d3-bias020/mixed-int8-int8head` | - | **added**: the repaired export with the MLM head in int8 as well: cosine 0.9948, tiny10 64.29, 95 MB |
| `onnx/full-soup-v20-dqf-bias015/dynamic-int8` | - | **added**: the full model with every node in int8, the MLM head included: cosine 0.9700, tiny10 64.88, 226 MB (mixed-int8 with the head in fp32, unchanged: cosine 0.9741, tiny10 64.91, 338 MB; torch 65.06) |
| `static/` (the table, the Sentence Transformers model, the query side alone) | table `phr2-k8-kd-r2`: asymmetric tiny10 / heldout8 60.74 / 39.24, 89.7 dims/query, 11764 postings/query | **replaced** by table `phr2-k8-kd-r2-mw020` (the same table with weights below 0.20 dropped, no training): 60.67 / 39.20, 75.9 dims/query, 9651 postings/query (-18 %) |
| `SHA256SUMS` | lists a stray `./SHA256SUMS.tmp`, so `sha256sum -c` reports one unreadable file | no entries of `SHA256SUMS*`; `sha256sum -c` passes |

`logs/final_static_table.txt` names `models/static-soup-v20-dqf-bias015/phr2-k8-kd-r2-mw020`. Difference for the 5070: `datasets/transfer/release-20261117-diff-from-20261113.tar` (823M, 62 files; how to apply it is in the README of the package). The static-table benchmark on the 5070 has to be run again for the new table.

## p1: cheap diagnosis and cheap repairs of trec-covid (no training)

**The KD model alone** (`kd-rbase-v2-dqf-full-bias015`): tiny10 64.69, trec-covid **68.89** (R-a 67.39, frozen model 72.48).

Fidelity of the KD student (`kd-rbase-v2-dqf-full`, no shift) to its teacher (the raw frozen model), 512 tokens:

| texts | n | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |
|---|---:|---:|---:|---:|---:|---:|---:|
| kd_v1, passages | 512 | 0.9725 | 0.834 | 0.725 | 0.252 | 495 | 491 |
| trec-covid-pl, passages | 512 | 0.9657 | 0.833 | 0.706 | 0.292 | 581 | 598 |
| nfcorpus-pl, passages | 512 | 0.9684 | 0.862 | 0.721 | 0.288 | 686 | 729 |
| scifact-pl, passages | 512 | 0.9701 | 0.860 | 0.723 | 0.279 | 700 | 739 |
| kd_v1, queries | 512 | 0.9831 | 0.847 | 0.756 | 0.195 | 108 | 105 |
| trec-covid-pl, queries | 50 | 0.9674 | 0.816 | 0.708 | 0.274 | 104 | 101 |
| nfcorpus-pl, queries | 323 | 0.9592 | 0.751 | 0.653 | 0.301 | 75 | 69 |
| scifact-pl, queries | 300 | 0.9678 | 0.810 | 0.682 | 0.288 | 252 | 251 |

The two measured forms on the same passages (`arm-b1-ra-bias015` as the student, the frozen model as the teacher), i.e. the document density side by side:

| texts | n | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |
|---|---:|---:|---:|---:|---:|---:|---:|
| kd_v1, passages | 512 | 0.9653 | 0.802 | 0.688 | 0.320 | 342 | 329 |
| trec-covid-pl, passages | 512 | 0.9618 | 0.812 | 0.683 | 0.331 | 397 | 386 |
| nfcorpus-pl, passages | 512 | 0.9661 | 0.817 | 0.709 | 0.316 | 470 | 454 |
| scifact-pl, passages | 512 | 0.9673 | 0.814 | 0.709 | 0.315 | 480 | 461 |

Cosine on trec-covid-pl against the lowest of kd_v1 and the two biomedical controls: passage -0.0027; query +0.0082. No fidelity gap on trec-covid by the bar of 0.03 below every control.

**Density of the forms of R-a** (limit 410 dims/doc):

| form | dims/doc |
|---|---:|
| `arm-b1-ra-bias010` | 427.2 |
| `arm-b1-ra-bias011` | 413.5 |
| `arm-b1-ra-bias012` | 399.8 |
| `arm-b1-ra-bias013` | 386.7 |
| `arm-b1-ra-bias014` | 374.0 |
| `arm-b1-ra-bias015` | 361.8 |

Measured: `arm-b1-ra-bias014`: tiny10 65.39, trec-covid 67.97 (bias015: 65.34, 67.39).

**Interpolation with the KD model**: not made (trec-covid of the KD model alone 68.89, the bar was 70).

**Soup R-a + R-c' after the dose3 round**: made (R-c' `arm-b1-rc-d3-bias010` 65.59 / 44.01 / trec-covid 68.42 is within 0.2 of R-a on both sets). R-c' started from a stage-1 model whose bias had been shifted in place (see p1b).

heldout8 was measured for: arm-b1-ra-bias014.

**Start of the data arms of p2 / p3** (the user, 17.11: start from the better checkpoint): **rc** = `arm-b1-rc-s1p4`. Rule: the lineage of R-c' when R-c' after the dose3 round (`arm-b1-rc-d3-bias010`, 19 tasks 56.652) leads every measured form of R-a (bias015 56.359, bias014 56.425) on the 19 tasks; then the arms start where R-c' started (the end of the MS MARCO-pl stage as it is on the disk, with the bias shift of b1c: the start R-c' proves), and R-c', fact20 and faq20 are arms from one start. R-a stays out of the soups of that lineage (no soup R-a + R-c', or it is below R-c' alone on the 11 tasks).

New biomedical data: not proposed (the bar: a fidelity gap of 0.03 on trec-covid AND the KD model alone losing trec-covid).

## p1b: erratum to the dose curve of R-c' stage 1 (summary_b1.md)

Queue b1c moved every stage-1 model to the data drive before it wrote the bias forms, and the copy of the forms copied the link instead of the files. The three shifts (thresholds 0.05, 0.10, 0.15) therefore went into the raw model itself and added up: every `arm-b1-rc-s1pK` on the disk has its MLM head bias lower by 0.3183 (threshold 0.276), the form labelled `bias015` in summary_b1.md is that model, and the labels `bias005` / `bias010` are the sums of one and two shifts. Consequences: (1) the dose curve (64.60 / 64.50 / 64.65 / 64.62) and the end of stage 1 (64.62 / 43.00 / trec-covid 67.33 at 315.6 dims/doc, "postings -20 %") were read in a far sparser form than the first form within the limits; (2) segments 2-4 each started from a model shifted that way, and so did the dose3 round `arm-b1-rc-d3`; (3) the reading "the dose curve is flat" does not follow from those numbers. R-a, the KD model and every 82 M number are not affected (their forms were copies of real directories).

The shift added back on real copies (the copy reproduces the density b1c measured before its first shift), first form within both limits:

| MS MARCO-pl rows | restored model, measured form | tiny10 | trec-covid | dims/doc | dims/query | postings/query | tiny10 reported by b1c | at dims/doc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 50000 | `arm-b1-rc-s1p1-unshifted-bias025` | 64.6 | - | 363.6 | 120.8 | 15472 | 64.6 | 333.7 |
| 200000 | `arm-b1-rc-s1p4-unshifted-bias020` | 64.78 | 67.84 | 396.9 | 128.4 | 16511 | 64.62 | 315.6 |
| 0 | `kd-rbase-v2-dqf-full-bias015` (KD alone) | 64.69 | 68.89 | 372.7 | 120.9 | 14064 | | |

The point at 50 000 rows is clean (its training started from the KD model). The point at 200 000 rows is the end of a stage whose segments 2-4 each started from a shifted model, so it is not a clean measurement of what 200 000 rows give. heldout8 was not measured here.

## Queue state (b2)

```
regime:quantized:kd-rbase-v2-dqf-full-bias015:heldout2_treccovid	ok	68.89 1	1640s
fidelity:b2-kd-rbase-v2-dqf-full.kd_v1.passage	ok	-	18s
fidelity:b2-kd-rbase-v2-dqf-full.kd_v1.query	ok	-	13s
fidelity:b2-arm-b1-ra-bias015-vs-frozen.kd_v1.passage	ok	-	17s
fidelity:b2-kd-rbase-v2-dqf-full.trec-covid-pl.passage	ok	-	11s
fidelity:b2-kd-rbase-v2-dqf-full.trec-covid-pl.query	ok	-	5s
fidelity:b2-arm-b1-ra-bias015-vs-frozen.trec-covid-pl.passage	ok	-	11s
fidelity:b2-kd-rbase-v2-dqf-full.nfcorpus-pl.passage	ok	-	11s
fidelity:b2-kd-rbase-v2-dqf-full.nfcorpus-pl.query	ok	-	6s
fidelity:b2-arm-b1-ra-bias015-vs-frozen.nfcorpus-pl.passage	ok	-	12s
fidelity:b2-kd-rbase-v2-dqf-full.scifact-pl.passage	ok	-	11s
fidelity:b2-kd-rbase-v2-dqf-full.scifact-pl.query	ok	-	6s
fidelity:b2-arm-b1-ra-bias015-vs-frozen.scifact-pl.passage	ok	-	12s
sparsity:arm-b1-ra-bias011	ok	413.5	27s
sparsity:arm-b1-ra-bias012	ok	399.8	27s
sparsity:arm-b1-ra-bias013	ok	386.7	26s
sparsity:arm-b1-ra-bias014	ok	374.0	27s
regime:quantized:arm-b1-ra-bias014:tiny10	ok	65.39 10	701s
regime:quantized:arm-b1-ra-bias014:heldout2_treccovid	ok	67.97 1	1602s
sparsity:soup-b1-ra-rcd	ok	3344.5	27s
sparsity:soup-b1-ra-rcd-bias010	ok	2492.9	27s
sparsity:soup-b1-ra-rcd-bias030	ok	1175.5	27s
regime:quantized:arm-b1-ra-bias014:heldout8	ok	43.77 8	2549s
sparsity:arm-b1-rc-s1p1-unshifted	ok	793.5	28s
sparsity:arm-b1-rc-s1p1-unshifted-bias010	ok	586.8	27s
sparsity:arm-b1-rc-s1p1-unshifted-bias019	ok	441.4	27s
sparsity:arm-b1-rc-s1p1-unshifted-bias020	ok	427.4	26s
sparsity:arm-b1-rc-s1p1-unshifted-bias022	ok	400.7	27s
sparsity:arm-b1-rc-s1p1-unshifted-bias025	ok	363.6	26s
regime:quantized:arm-b1-rc-s1p1-unshifted-bias025:tiny10	ok	64.6 10	696s
sparsity:arm-b1-rc-s1p4-unshifted	ok	704.9	27s
sparsity:arm-b1-rc-s1p4-unshifted-bias010	ok	532.8	26s
sparsity:arm-b1-rc-s1p4-unshifted-bias018	ok	421.4	27s
sparsity:arm-b1-rc-s1p4-unshifted-bias019	ok	409.0	27s
sparsity:arm-b1-rc-s1p4-unshifted-bias020	ok	396.9	26s
regime:quantized:arm-b1-rc-s1p4-unshifted-bias020:tiny10	ok	64.78 10	704s
regime:quantized:arm-b1-rc-s1p4-unshifted-bias020:heldout2_treccovid	ok	67.84 1	1662s
```
