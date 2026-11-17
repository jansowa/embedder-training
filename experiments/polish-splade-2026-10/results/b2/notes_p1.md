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
