# scripts/

Generic tools around the trainer and the PIRB evaluation. Run them from the
repository root (`python scripts/<name>.py`, `bash scripts/<name>.sh`).
Experiment-specific scripts (config generators, queues) live in
`experiments/<series>/`; dataset building and annotation live in the data
kitchen repository.

## Queues and guards

| script | purpose |
|---|---|
| `queue_lib.sh` | helpers sourced by every queue: guarded `train`, `bench`, `holdout`, `sparsity`, `regime` steps; see [docs/queues.md](../docs/queues.md) |
| `orphan_guard.py` | kills GPU workers that lost their queue (no terminal, no `bash .../scripts/*.sh` or `.../queues/*.sh` ancestor, no `ORPHAN_GUARD_EXEMPT=1`) |
| `disk_guard.sh` | prunes PIRB index directories whose scores are already saved when `/` drops below 20 GB |

## Evaluation

| script | purpose |
|---|---|
| `pirb_regime_eval.sh` | one SPLADE model on one task set (`configs/pirb_tasks/`) in one regime: `quantized` (leaderboard), `truncated`, `old-pyserini`, `new-backend` |
| `pirb_ab_check.sh` | A/B check of the vendored PIRB fork against the last upstream commit before our changes, on CPU |
| `pirb_resume_e2e.py` | end-to-end proof that an interrupted PIRB benchmark resumes |
| `compare_splade_sparsity.py` | active document dimensions of SPLADE models on PIRB passages, without building an index |
| `splade_index_cost.py` | index size and postings per query of SPLADE models on one PIRB sample |

## Result tables

All read `$RESULTS_ROOT`, `$REGIMES_ROOT`, `$HOLDOUT_ROOT`, `$HOLDOUT2_ROOT`
(defaults in `result_roots.py`, under the current experiment series).

| script | purpose |
|---|---|
| `queue_summary.py` | one table of every model: tiny, held-out-8, held-out-2, index cost, with deltas against the base |
| `regime_task_table.py` | per-task table of leaderboard-regime results for chosen models and task sets |
| `seed_noise_table.py` | per-task table of several models (seeds, arms) with mean, spread and std |
| `sparsity_table.py` | active dimensions of documents and queries, postings per query |
| `leaderboard_compare.py` | our regime results next to the public PIRB leaderboard export |
| `update_leaderboard_doc.sh` | regenerates the tables of the series' `docs/benchmark-vs-leaderboard.md` |
| `compare_results.py` | NDCG@10 across `metrics.json` files and leaderboard columns |
| `paired_delta.py` | paired comparison against a base with a bootstrap CI over tasks |
| `aggregate_task_runs.py` | combines one-task-per-process PIRB runs into one summary |

## Models

| script | purpose |
|---|---|
| `merge_splade_weights.py` | weight-averaged soup of SPLADE checkpoints (`--models ... --weights ... --output models/<name>`) |
