# SPLADE initialization: D0, E1–E4

Updated: 2026-11-04T23:58:45+01:00

All new trials: stock head, BF16, physical batch 8 × accumulation 4, 256 tokens, six negatives, same five data files. E2–E4: scale=1, backbone/head LR=2e-5, LR warmup 250 steps. E1 is diagnostic scale=0.0001, LR=5e-5, no regularization. Sparsity: FP32, 512 tokens, 300 passages per tiny task; tiny10: BF16, quantized, 512 tokens. Only complete 10-task means count as tiny10 scores.

## D0: identical microbatch, no training

| Model | Scores min / median / max | Median active doc dims | MNRL scale=1 |
|---|---|---:|---:|
| sdadas/polish-distilroberta | 28787.979 / 57783.781 / 111333.188 | 45832.000 | 2787.762 |
| OPI-PIB/pl-ModernBERT-512-base | 2802.814 / 21152.203 / 176011.250 | 49085.000 | 9895.867 |

D0 uses the same first eight eligible prepared rows for both models, eval mode (no dropout), BF16 forward and float32 score diagnostics. The input SHA-256 is in the raw JSON. This is a controlled initial-condition comparison, not proof of the entire causal mechanism.

## Trials

| run | c | initial scores median / max | initial loss | dims s250 / s500 | rank loss s500 | tiny10 s500 | verdict |
|---|---:|---|---:|---|---:|---:|---|
| e1 | — | — / — | — | — / 36290.978 | 2.179 | — | nonzero; quality/density assessed separately |
| e2 | 11.745 | 2.450 / 148.438 | 3.808 | 3725.275 / 4014.277 | 0.875 | — | nonzero; quality/density assessed separately |
| e3 | 11.745 | 2.450 / 148.438 | 3.808 | 518.573 / 362.843 | 0.811 | 54.205 | nonzero; quality/density assessed separately |

Recipe verdict: **pending**

Stable E3 requires: completed 500 steps; finite nonzero FP32/BF16 diagnostic outputs; mean document dimensions in [200,600]; complete tiny10@s500 at least 1.0 nDCG percentage point above E1; and E3 tiny10@s500 > tiny10@s250. The +1.0 threshold is the operational interpretation of “clearly above E1”.

Head-only E4 preserves tied decoder/input-embedding parameters: those shared weights remain trainable; no architecture change or untying is performed. E4-unfrozen starts a new optimizer and 250-step LR warmup from E4-head weights and does not apply the bias shift again.

E1 initialization scores, if not explicitly measured before its original training, are not invented or substituted with D0 scale=1 loss. Full process results and exceptions are in `results/data/splade-init-state.json` and logs/init_*. No night-3 training is launched.
