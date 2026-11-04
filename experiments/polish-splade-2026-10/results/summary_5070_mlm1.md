# Sonda MLM — 2026-11-04T01:30+01:00

Reżim leaderboardowy quantized, pomiary przy 512 tokenach.

**Wariant zapasowy: stałe λ**, λ_d=0.003, λ_q=0.00015; bez regulatora i progów L0.

Batch fizyczny: **8**, akumulacja: 4, efektywny batch: 32; gradient checkpointing: True, pooling chunk: 8.

Przebieg z batchem 16 przerwano z powodu przekroczenia VRAM i używania pamięci współdzielonej. Jego checkpointy usunięto; wyniki sondy dotyczą świeżego przebiegu z mniejszym batchem.

| krok | przykłady | tiny10 | wym./dok (PIRB) | λ_d | gęstość próbki / baza | mkqa | pairs | templates |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 625.00 | 20000.00 | 54.71 | 418.49 | 0.003 | 215.54 / brak pomiaru bazy | 47.81 | 74.23 | 67.37 |
| 1250.00 | 40000.00 | 54.73 | 434.86 | 0.003 | 338.83 / brak pomiaru bazy | 53.16 | 75.69 | 70.59 |
| 1875.00 | 60000.00 | 55.70 | 390.82 | 0.003 | 329.93 / brak pomiaru bazy | 51.80 | 76.50 | 70.31 |
| 2500.00 | 80000.00 | 56.33 | 297.15 | 0.003 | 236.86 / brak pomiaru bazy | 51.55 | 76.26 | 71.02 |
| 5000.00 | 160000.00 | 55.37 | 310.58 | 0.003 | 230.68 / brak pomiaru bazy | 52.11 | 75.62 | 69.90 |
| 10000.00 | 320000.00 | 55.62 | 249.16 | 0.003 | 178.88 / brak pomiaru bazy | 51.54 | 75.61 | 69.33 |
| 15000.00 | 480000.00 | 52.94 | 222.40 | 0.003 | 151.50 / brak pomiaru bazy | 45.52 | 72.40 | 64.47 |
| 18750.00 | 600000.00 | 52.44 | 221.02 | 0.003 | 146.09 / brak pomiaru bazy | 44.91 | 68.82 | 64.43 |

## Kotwice

| model | tiny10 | wym./dok |
|---|---:|---:|
| base-splade-wrapped | 61.60 | — |
| mlm-msmarco-ep1 | 59.30 | 798.94 |
| mlm-path-ep3 | 55.62 | 503.26 |

## G1

REVIEW — stałe λ, gęstość poza 300–480; decyzja użytkownika, próg 59.5 nie obowiązuje wprost. tiny10(s18750)=52.44, s10000−s05000=+0.24; gęstości=249.16, 222.40, 221.02

## Czas i monitoring

```json
{
  "queue_started": "2026-11-03T19:47:30+01:00",
  "training_started_unix": 1793731700,
  "training_seconds": 16555,
  "return_code": 0,
  "first625": {
    "attempt": "arm-mlm-probe-a",
    "wallclock_seconds_from_train_call_including_loading": 1462.7511959075928,
    "wallclock_examples_per_second": 13.672865252788672,
    "training_loop_seconds_from_progress_bar": 1392,
    "checkpoint": "runs/arm-mlm-probe-a/sentence-transformers/splade/sdadas_polish-distilroberta-lr-5e-5-ep-12-loss-sparse_multiple_negatives_ranking-docreg-0_003-20261103-062910/step-checkpoints/step-625/trainer_state.json"
  },
  "timeout_limit_seconds": 43200,
  "ended_at_timeout_limit": false
}
```

λ_d: bez zmian (0 wzrostów, 0 spadków), końcowo 0.003. Gęstość względem bazy: niezmierzona w wariancie zapasowym.

## FAILED

```
[14:53:45] [mlm1] FAILED step (1): probe_train
[15:03:11] [mlm1] FAILED: physical C: free below 35 GiB
[15:03:11] [mlm1] FAILED step (1): probe_train
```

## Kontrola po restarcie z mniejszym batchem

```json
{
  "recorded_at": "2026-11-03T06:54:07.5623304+01:00",
  "physical_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 32,
  "gradient_checkpointing": true,
  "splade_pooling_chunk_size": 8,
  "training_started": "2026-11-03T06:29:05+01:00",
  "queue_deadline": "2026-11-03T20:00:00+01:00",
  "first625_training_loop_seconds": 1392,
  "first625_wallclock_seconds_including_loading_and_checkpoint": 1462.7511959075928,
  "training_loop_examples_per_second": 14.368,
  "target_625_within_900_seconds_met": false,
  "nvidia_smi_used_mib": 10904,
  "windows_dedicated_peak_gib": 10.704,
  "windows_shared_peak_mib": 187.75,
  "windows_shared_latest_mib": 127.996,
  "nvidia_snapshot": "logs/mlm_gpu_step625.txt",
  "status": "VRAM below 15 GiB; shared memory currently 128 MiB, peak 188 MiB. The 625-step time target was missed. Training continues under the requested 9-hour limit."
}
```

## Per-task table

### tiny10

| task | base-splade-wrapped | mlm-msmarco-ep1 | mlm-path-ep3 | mlm-probe-a-s00625 | mlm-probe-a-s01250 | mlm-probe-a-s01875 | mlm-probe-a-s02500 | mlm-probe-a-s05000 | mlm-probe-a-s10000 | mlm-probe-a-s15000 | mlm-probe-a-s18750 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arguana-pl | 37.14 | 29.94 | 24.76 | 31.74 | 25.66 | 29.67 | 30.82 | 28.25 | 32.72 | 34.89 | 34.01 |
| gpt-exams | 98.63 | 98.12 | 96.28 | 95.70 | 95.83 | 95.28 | 96.06 | 94.85 | 94.52 | 93.97 | 94.71 |
| maupqa-gpt3-cc | 18.43 | 18.69 | 18.23 | 17.90 | 18.37 | 17.96 | 17.79 | 17.61 | 17.03 | 15.97 | 16.41 |
| maupqa-mkqa | 63.92 | 61.66 | 53.26 | 47.81 | 53.16 | 51.80 | 51.55 | 52.11 | 51.54 | 45.52 | 44.91 |
| maupqa-poleval2021-pairs | 82.67 | 80.65 | 78.53 | 74.23 | 75.69 | 76.50 | 76.26 | 75.62 | 75.61 | 72.40 | 68.82 |
| maupqa-templates | 75.02 | 74.17 | 70.36 | 67.37 | 70.59 | 70.31 | 71.02 | 69.90 | 69.33 | 64.47 | 64.43 |
| nfcorpus-pl | 30.07 | 32.57 | 29.56 | 29.11 | 27.66 | 29.50 | 29.64 | 28.96 | 28.92 | 26.92 | 27.06 |
| poleval-2022-test-A-allegro-faq | 73.76 | 71.52 | 65.88 | 66.07 | 64.61 | 64.28 | 67.21 | 63.94 | 66.04 | 61.38 | 59.31 |
| poleval-2022-test-B-allegro-faq | 71.81 | 67.76 | 62.31 | 61.21 | 59.04 | 62.12 | 63.30 | 60.41 | 60.82 | 57.70 | 57.12 |
| scifact-pl | 64.52 | 57.95 | 57.01 | 55.89 | 56.72 | 59.61 | 59.64 | 62.08 | 59.62 | 56.17 | 57.62 |
| **mean** | **61.60** | **59.30** | **55.62** | **54.71** | **54.73** | **55.70** | **56.33** | **55.37** | **55.62** | **52.94** | **52.44** |


## Sparsity table

| model | wymiary/dokument, średnia (3000 pasaży tiny) | ×baza | p95 | wymiary/dokument (próbka kosztu) | wymiary/zapytanie | ×baza | postings/zapytanie | ×baza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 436.3 | ×1.00 | 695 | 450.0 | 138.6 | ×1.00 | 16415 | ×1.00 |
| mlm-probe-a-s00625 | 418.5 | ×0.96 | 610 |  |  |  |  |  |
| mlm-probe-a-s01250 | 434.9 | ×1.00 | 624 |  |  |  |  |  |
| mlm-probe-a-s01875 | 390.8 | ×0.90 | 555 |  |  |  |  |  |
| mlm-probe-a-s02500 | 297.2 | ×0.68 | 437 |  |  |  |  |  |
| mlm-probe-a-s05000 | 310.6 | ×0.71 | 458 |  |  |  |  |  |
| mlm-probe-a-s10000 | 249.2 | ×0.57 | 375 |  |  |  |  |  |
| mlm-probe-a-s15000 | 222.4 | ×0.51 | 333 |  |  |  |  |  |
| mlm-probe-a-s18750 | 221.0 | ×0.51 | 332 |  |  |  |  |  |

Źródło próbki kosztu: index_cost.json (1350 pasaży i tyle samo zapytań).

## Zastrzeżenie do G1: powtarzanie danych

Sonda A to około 9.8 epoki po tych samych 61 tys. przykładów. Spłaszczenie krzywej w ostatnich podwojeniach może wynikać z powtórzeń danych; nie stanowi samo w sobie dowodu osiągnięcia sufitu architektury.
