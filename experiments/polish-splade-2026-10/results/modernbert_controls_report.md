# ModernBERT: bounded collapse controls

Written: 2026-11-04T22:02:54+01:00

Both controls start from the original MLM, with zero query/document FLOPS weights, the same five training files, seed, physical batch 8 and accumulation 4. The 500-step cap preserves the original 250-step learning-rate warmup (ratio 0.5). Only ranking scale differs between the two controls: 1 versus 0.0001. The original failed run used nonzero, scheduled regularization. These controls do not test a complete corrected regularized recipe.

| Control | Last step | FP32 active dimensions (four texts) | BF16 active dimensions | Verdict |
|---|---:|---|---|---|
| noreg-scale1 | 300 | [0, 0, 0, 0] | [0, 0, 0, 0] | COLLAPSED: zero vectors in direct FP32/BF16 checks |
| noreg-scale1e-4 | 500 | [33425, 31413, 28042, 31274] | [33295, 31241, 27926, 31267] | NONZERO on four diagnostic texts; retrieval quality remains unmeasured |

## Interpretation

Collapse is reproduced without FLOPS regularization: regularization is not necessary for this failure under the original ranking scale and learning rate.
Reducing ranking scale while holding the no-regularization control fixed prevents an all-zero result on the four texts. This supports score scale as a contributing factor; retrieval evaluation and a regularized control are still required.

No full retraining, night-3 preparation, commit or push is scheduled by these diagnostic queues. All-zero and incomplete measurements are not reported as nDCG scores. Inspect `logs/mlm_control_*.log`, `logs/mlm_controls_diagnosis.log` and `logs/queue_5070_mlm_controls_finish.out` for errors. Raw results are under `results/data/`.
