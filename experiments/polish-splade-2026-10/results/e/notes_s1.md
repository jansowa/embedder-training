## s1 (E1): model A with R-c' in the teacher

**Teacher.** The 60 385 rows of the dose arm were scored by the raw R-c' (`models/arm-b1-rc-d3-bias000`: the raw weights with both bias keys, so that the fp32 loader of the scoring tool reads the bias) and again by soup-v9b, with the tool and the settings of ens400 (kitchen `bridge/splade_teacher_scores.py`, 512 tokens, batch 16, fp16). The rows keep only the ensemble, so the PolDense-400M part of a pair is 2 x ensemble - soup-v9b and the new score is ensemble + (R-c' - soup-v9b) / 2 (`scripts/swap_splade_teacher.py`); combining, the dense scale 84.19 and the weights are unchanged. Rows: `datasets/arms/n5-plus-kj-mined-dose3-tensrcd.jsonl`.

| scores | median margin min(pos) - max(neg) | rows with a negative above a positive | P(pos) at T = 2, median | share above 0.95 |
|---|---:|---:|---:|---:|
| ens400: soup-v9b + PolDense-400M (the teacher of the frozen model) | 5.28 | 6826 (11.3 %) | 0.855 | 0.298 |
| **R-c' + PolDense-400M (new)** | 5.01 | 6648 (11.0 %) | 0.834 | 0.252 |
| soup-v9b alone | 5.96 | 9300 (15.4 %) | 0.901 | 0.403 |
| R-c' alone (raw) | 5.80 | 6987 (11.6 %) | 0.884 | 0.346 |
| PolDense-400M alone (x 84.19) | 3.69 | 8702 (14.4 %) | 0.730 | 0.139 |

Within-row agreement (60385 rows): new ensemble against ens400 Pearson 0.9878, Kendall 0.8896; R-c' against soup-v9b Pearson 0.9601, Kendall 0.7935; against PolDense-400M: R-c' Pearson 0.8703, soup-v9b 0.7986. Within-row spread of the R-c' scores against soup-v9b's (median ratio of the standard deviations): 0.885, so with unchanged weights R-c' weighs a little less in the mean than soup-v9b did.

Check of the reconstruction: soup-v9b scored again against its scores of the base rows from queue v49 (224862 pairs): median |difference| 0.0020, 99th percentile 0.0206, max 12.6418.

**Arm and soups.** `arm-dense-kl-n5kj-mined-dose3-ensrcd-r2v12`: the recipe of the dose3 arm unchanged (from soup-v12, 2 epochs, lr 2e-5), training time 14875 s. Soups are uniform averages of the raw weights of arms from soup-v12.

| candidate | measured form | tiny10 | heldout8 | trec-covid | mean of 19 tasks | vs the frozen model | mean of 11 tasks (tiny10 + trec-covid) | vs the frozen model | dims/doc | dims/query | postings/query | within the limits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frozen full model A (82 M) | `soup-v20-dqf-bias015` | 65.06 | 43.64 | 72.48 | 56.435 | reference | 65.738 | reference | 371.4 | 123.0 | 13945 | yes |
| soup new arm + faq20 + fact20 | `soup-v21-nqf-bias015` | 65.17 | 43.72 | 72.22 | 56.510 | +0.075 | 65.814 | +0.076 | 380.3 | 125.2 | 14665 | yes |
| dose3 arm of the frozen model (teacher ens400) | `arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015` | 64.96 | 43.63 | 72.82 | 56.394 | -0.041 | 65.674 | -0.065 | 374.7 | 121.9 | 14079 | yes |
| soup dose3 + faq20 + fact20 + new arm | `soup-v21-dqfn-bias015` | 65.14 | - | 71.76 | - | - | 65.738 | -0.001 | 377.8 | 123.8 | 14359 | yes |
| new arm: dose3 rows, teacher R-c' + PolDense | `arm-dense-kl-n5kj-mined-dose3-ensrcd-r2v12-bias015` | 64.97 | - | 71.01 | - | - | 65.521 | -0.217 | 401.5 | 128.5 | 16588 | yes |

The frozen model and its dose3 arm are in the forms they were measured in (bias015); the candidates are in their first form within the limits on the same grid of 0.05. heldout8 went to the best candidate on the 11 tasks only.

**Gate for A** (>= +0.2 over the frozen model on the 19 tasks, within the limits): `soup-v21-nqf-bias015` 56.510 against 56.435 -> **does not pass**.
