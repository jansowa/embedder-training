# Probe M diagnosis — 4 November 2026

Two independent failures were confirmed: the retrieval evaluator could not convert
BF16 tensors to NumPy, and the trained SPLADE representations had already
collapsed to zero before the first saved checkpoint. The pretrained ModernBERT
checkpoint itself works. The saved training weights contain no NaN or infinity.

## Direct model checks, outside PIRB

The same four Polish texts were encoded with `AutoModelForMaskedLM` in FP32 and
BF16. Outputs were pooled as `max(log1p(relu(logits)))`, with attention masking.
Counts below include only dimensions with a real tokenizer token and use the
same `round(weight * 100) > 0` definition as retrieval. Tests were repeated with
max-length settings 256 and 512; these four texts are shorter than either limit.

| Weights | Precision | Active dimensions for the four texts | Maximum raw logit |
|---|---|---|---:|
| original MLM | FP32 | 11732, 8122, 25829, 32949 | +25.95 |
| original MLM | BF16 | 12236, 9091, 25530, 33444 | +25.75 |
| step 625 | FP32 | 0, 0, 0, 0 | -3.55 |
| step 625 | BF16 | 0, 0, 0, 0 | -3.52 |
| step 5000 | FP32 | 0, 0, 0, 0 | -3.55 |
| step 5000 | BF16 | 0, 0, 0, 0 | -3.52 |

Both trained checkpoints return finite, exactly zero pooled vectors. Negative
logits are removed by ReLU before quantization or retrieval. This is a training
collapse, not a BF16 conversion artifact or a damaged checkpoint file. Existing
FP32 sparsity measurements also found zero dimensions on 2700 PIRB passages.

## The collapse was visible during training

| Step | Mean active document dimensions on the training sample | Base loss | Gradient norm |
|---:|---:|---:|---:|
| 25 | not sampled | 3457.6489 | 282915.59375 |
| 250 | 3.2891 | 3.8666 | 48.7706 |
| 500 | 0 | 4.0254 | 0 |
| 625 | not sampled at this step | 4.0254 | 0 |
| 1000 onward | 0 on the density-control sample | 4.0254 | 0 |

The final base loss 4.0254 matches `log(56)`: equal scores for the 56 candidate
documents in a physical batch of 8 with six negatives per row. This is consistent
with zero representations and dead ReLU gradients. The density regulator only
started controlling at step 1000, after the collapse, and reducing lambda did not
restore gradients. The exact contribution of learning rate, score scale and
regularization needs a short controlled test from the original MLM; it has not
been established by this diagnosis.

## Evaluation fixes

`third_party/pirb/search/splade.py` converted BF16 tensors directly with
`.cpu().numpy()` for documents and queries. NumPy does not support this tensor
dtype. Both paths now cast BF16 to FP32 before conversion, retaining the original
FP16/FP32 conversion behavior. Four CPU tests passed, including real BF16 query
and document conversion and preservation of FP16 rounding.

The queue also parsed an empty mean followed by a zero task count as if it were
a successful evaluation. Missing means now print an explicit `nan` field so the
zero count remains in the expected column. Two tests distinguish a failed
zero-task evaluation from a valid score of zero on ten tasks.

The prior `n_tasks: 0, mean_ndcg@10: null` files are failed measurements, not
tiny10 scores of zero. The corrected end-to-end evaluation of s05000 ran from
10:13 to 10:15. The BF16 conversion failure disappeared. On the first corpus,
all **10,176 documents** had empty vectors: Anserini reported 10,176 empty
documents and zero indexed documents. Opening this empty index then failed in
Pyserini's IDF calculation (`MultiTerms.getTerms(...)` returned null). This
independently confirms the unusable embeddings in the actual retrieval path.
The queue now correctly reports this run as FAILED. The original invalid result
was retained beside the result file; no numerical tiny10 score was invented.

## Conclusion

The current probe M checkpoints are unusable as retrieval models. Continuing the
same checkpoints will not address the observed zero-gradient state. This run
does not establish the quality ceiling of ModernBERT-SPLADE. No night-3 training
has been started or prepared on the basis of these invalid comparisons.

The technical outcome of this recipe is FAIL (collapsed representations). A
comparative GM nDCG score remains unavailable, rather than a measured zero.

Detailed raw measurements: `results/data/modernbert-probe-m-diagnosis.json`.
