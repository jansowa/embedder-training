## p1b: erratum to the dose curve of R-c' stage 1 (summary_b1.md)

Queue b1c moved every stage-1 model to the data drive before it wrote the bias forms, and the copy of the forms copied the link instead of the files. The three shifts (thresholds 0.05, 0.10, 0.15) therefore went into the raw model itself and added up: every `arm-b1-rc-s1pK` on the disk has its MLM head bias lower by 0.3183 (threshold 0.276), the form labelled `bias015` in summary_b1.md is that model, and the labels `bias005` / `bias010` are the sums of one and two shifts. Consequences: (1) the dose curve (64.60 / 64.50 / 64.65 / 64.62) and the end of stage 1 (64.62 / 43.00 / trec-covid 67.33 at 315.6 dims/doc, "postings -20 %") were read in a far sparser form than the first form within the limits; (2) segments 2-4 each started from a model shifted that way, and so did the dose3 round `arm-b1-rc-d3`; (3) the reading "the dose curve is flat" does not follow from those numbers. R-a, the KD model and every 82 M number are not affected (their forms were copies of real directories).

The shift added back on real copies (the copy reproduces the density b1c measured before its first shift), first form within both limits:

| MS MARCO-pl rows | restored model, measured form | tiny10 | trec-covid | dims/doc | dims/query | postings/query | tiny10 reported by b1c | at dims/doc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 50000 | `arm-b1-rc-s1p1-unshifted-bias025` | 64.6 | - | 363.6 | 120.8 | 15472 | 64.6 | 333.7 |
| 200000 | `arm-b1-rc-s1p4-unshifted-bias020` | 64.78 | 67.84 | 396.9 | 128.4 | 16511 | 64.62 | 315.6 |
| 0 | `kd-rbase-v2-dqf-full-bias015` (KD alone) | 64.69 | 68.89 | 372.7 | 120.9 | 14064 | | |

The point at 50 000 rows is clean (its training started from the KD model). The point at 200 000 rows is the end of a stage whose segments 2-4 each started from a shifted model, so it is not a clean measurement of what 200 000 rows give. heldout8 was not measured here.
