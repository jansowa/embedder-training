# P1 v2: KL od MLM

Stan: completed; aktualizacja: 2026-11-10T08:02:06+01:00

347 687 zapytań, jeden pozytyw i 10 negatywów; start sdadas/polish-distilroberta, batch 8×4, LR 2e-5 stały po rozgrzewce, 3 epoki.

S PolDense: 97.41069280910885; zastosowane przed uśrednieniem nauczycieli. score_scale treningu: 1.0.

| Model | tiny10 | heldout8 | wym./dok | wym./zapytanie | postings/zapytanie |
|---|---:|---:|---:|---:|---:|
| baza | 61.60 | 41.03 | — | — | — |
| soup-v12 | 63.96 | 42.92 | — | — | — |
| sondy MLM | 55.5–56.7 | — | — | — | — |
| epoka 1 | 59.24 | — | 464.24 | 92.87 | 5119.15 |
| epoka 2 | 60.33 | — | 418.24 | 87.11 | 5860.83 |
| epoka 3 | 60.51 | 41.27358518240606 | 464.48 | 94.93 | 5969.69 |

G1 (informacyjnie): density_outside_band (tiny10 ≥59.5, gęstość dokumentów 330–460; nie zatrzymuje epok).

Pomiary quantized przy 512 tokenach, po zakończeniu treningu, z zachowanych checkpointów; dodatkowy profil min_weight=0.05 w plikach JSON.

```json
{
  "status": "completed",
  "resumed": true,
  "checkpoint_policy": {
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 2,
    "keep_epoch_checkpoints": true
  },
  "teacher": {
    "rows": 347687,
    "examples": 347687,
    "ab_rows": 40000,
    "dense_scale": 97.41069280910885,
    "dense_sidecar": "/mnt/c/work/embedder-training-v2/datasets/arms/mlm-p1-kl-v2-tpd400.jsonl.teacher.json",
    "recipe": "mean(soup-v12 dot product, already scaled PolDense cosine)",
    "training_score_scale": 1.0,
    "input_pool_sha256": "d1bdfea47314829bf3ba5c2caa5a1da49cece837298c7d35f08ffaf9d46885eb",
    "sha256": "1a6055aad2e4d7eae7bcc228d4a1874245f87474d50d412fd4f600ecbef646da",
    "ab_sha256": "9979c907979cb8783c1f7e62954d2d82555b94ca5438ac9a0b7b57ebd29e1f83",
    "mnrl_rows": 301042,
    "mnrl_sha256": "2184196c430d0ea6edb047bafa9888e123d8248128cf73b8118a05fd7122d6d9"
  },
  "config_sha256": "c145a7fc06c7e4e10387a86ad0911840bb5a22fe17a288abdf7ff3b18eac3e9e",
  "measurements_after_training": true,
  "pre_resume_tiny10": [
    {
      "epoch": 1,
      "checkpoint": "/mnt/c/work/embedder-training-v2/cache/mlm-p1-loadable-exports/epoch-1",
      "source_checkpoint": "runs/arm-mlm-p1/sentence-transformers/splade/sdadas_polish-distilroberta-lr-2e-05-ep-3-loss-sparse_distill_kl_div-docreg-0_003-20261107-160555/epoch-checkpoints/epoch-0001-step-10866",
      "label": "mlm-p1-ep1",
      "tiny10": 59.239718948349335,
      "verified_pirb_original_loader": true
    },
    {
      "epoch": 2,
      "checkpoint": "/mnt/c/work/embedder-training-v2/cache/mlm-p1-loadable-exports/epoch-2",
      "source_checkpoint": "runs/arm-mlm-p1/sentence-transformers/splade/sdadas_polish-distilroberta-lr-2e-05-ep-3-loss-sparse_distill_kl_div-docreg-0_003-20261107-160555/epoch-checkpoints/epoch-0002-step-21732",
      "label": "mlm-p1-ep2",
      "tiny10": 60.33142117087981,
      "verified_pirb_original_loader": true
    }
  ],
  "g1": "density_outside_band",
  "g1_policy": "informational; continue all three epochs",
  "finished_at": "2026-11-10T08:02:06+01:00",
  "best_checkpoint": "/mnt/c/work/embedder-training-v2/cache/mlm-p1-loadable-exports/epoch-3",
  "best_within_density_band": false
}
```
