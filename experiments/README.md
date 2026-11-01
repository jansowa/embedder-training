# Experiments

One directory per experiment series. A series keeps everything that only makes
sense for it: configs, the generators that wrote them, queues, results and the
notes that interpret them. The generic trainer, evaluator and queue helpers stay
at the repository root.

```
experiments/<series>/
  configs/          training configs of the series' arms
  scripts/          config generators and other series-only scripts
  queues/           queue_vN.sh (see docs/queues.md)
  results/          tiny benchmark (truncated regime), sparsity, summaries, data diagnostics
  results-regimes/  leaderboard-regime runs (scripts/pirb_regime_eval.sh)
  results-holdout/  held-out-8, truncated regime
  docs/             plans, findings, runbooks, model cards
  tests/            tests of the series' configs
```

Everything runs from the repository root. The queue helpers and the table
scripts read and write under `EXP_ROOT` (default
`experiments/polish-splade-2026-10`); a new series sets
`EXP_ROOT=experiments/<new-series>` and gets its own result tree.

## polish-splade-2026-10

Fine-tuning `sdadas/polish-splade` for Polish retrieval (PIRB), 23.10 to
1.11.2026, on an RTX 3060 Ti 8 GB (and a 5070 Ti 16 GB for some arms,
`docs/RUNBOOK-5070.md`). The queues v2 to v25 are in `queues/`.

### What was studied

- **Recipe.** MNRL + FLOPS (document regularizer 0.001, query 5e-5), lr 2e-5,
  2 epochs, 512 tokens, 10 hard negatives, batch 2 x 16 by accumulation, fp16,
  gradient checkpointing, a no-duplicates batch sampler, and a gate filter on the
  strong reranker score of the pair (`configs/dataset_filters/splade_positive_score_gt_23_50_no_metricx.yaml`,
  gemma2-lightweight > 23.5 on the English pair).
- **Data dose.** Small gated cuts (2073 rows per source) beat whole datasets;
  FiQA (own translation) + SQuAD-v2 is the common denominator of the best arms;
  trivia, wikisynth and whole nq / fever / hotpotqa hurt. The new speakleash sets
  (v25) did not pass as n4-pattern components (nq), or are still open (fever was
  killed by the train timeout; a one-positive hotpotqa cut is prepared).
- **Negatives.** Length relative to the positives is a real knob (long-easy
  negatives: arguana +10, fiqa-pl -8; a 2/10 dose is neutral on averages). The
  student's own hardest negatives (SPLADE margin > -3, the "tail") hurt
  everywhere, while the far band is neutral. Details: `docs/PLAN-2026-10-31.md`
  section 4.
- **Batch and loss.** A true batch of 32 through GradCache
  (`CachedSpladeLoss`) is worse than 2 x 16 at this regularizer and 2.5x slower;
  raw margin-MSE distillation shrinks the model (score scale mismatch).
- **Soups.** Uniform weight averages of the best arms
  (`scripts/merge_splade_weights.py`) are the best models; variants are saturated
  (within 0.03 on held-out-8), gains come only from new ingredients.
- **LLM judge on FiQA** pairs did not help.

### Best models (leaderboard regime, nDCG@10)

| model | composition | tiny10 | heldout8 | held-out-2 |
|---|---|---:|---:|---:|
| soup-v9b | n1 + n3 + n4 + long-easy + lenmatch-far, uniform | 63.65 | 42.56 | 69.42 |
| soup-v7a | n1 + n3 + n4 + long-easy | 63.54 | 42.53 | 69.16 |
| n4-5070 | FiQA + SQuAD-v2 + MS MARCO-clean, 2073 rows each | 62.86 | 42.30 | 69.00 |
| base | `sdadas/polish-splade` | 61.59 | 41.03 | 67.53 |

The full ranking and the reasoning are in `docs/PLAN-2026-10-31.md` (sections 2
and 3); the earlier plans (`PLAN-2026-10-26`, `-27`, `-29`) are the history.

### Where the models are

Checkpoints are not in git. On the 3060 box they are under `models/<name>` in
the repository root (about 42 GB); the name is the same as the result directory:

- `models/arm-<tag>`: trained by a queue's `train <tag> <config>`; the config is
  `configs/next/3060/<tag>.yaml` (or `configs/weekend/...` for the early arms);
- `models/<name>-5070`: trained on the 5070 Ti from `configs/next/5070/` and
  copied over;
- `models/soup-*`: weight averages; the ingredients are in the
  `merge_splade_weights.py` calls of the queue that built them and in the plan's ranking;
- `models/base-splade-wrapped`: the base model saved as a SentenceTransformers
  sparse encoder (the reference of every delta).

Training runs (checkpoints, resolved configs) go to `runs/`, a symlink to the
media drive. A model card draft is in `docs/model_card_soup-of-ft.md`. Training
sets (`datasets/arms/`) are generated, not tracked; `results/data/` keeps their
manifests and diagnostics.

### How to read the summaries

Each queue ends by writing `results/summary_vN.md`:

1. The first table is `scripts/queue_summary.py`: every model measured so far,
   sorted by tiny. Columns: **tiny** (10 PIRB tasks, truncated-query regime,
   delta against `base-splade-wrapped` = 60.14), **held-out-8** (8 further PIRB tasks,
   base 38.09), **held-out-2** (5 more tasks),
   **doc dims x base** and **postings/query x base** (index cost, where profiled).
2. Then per-task tables: `scripts/regime_task_table.py` for the leaderboard
   regime (`results-regimes/quantized/<model>/<taskset>.json`) and
   `scripts/seed_noise_table.py` for arms in the truncated regime.
3. Then queue-specific blocks (data diagnostics, probes) and the queue state:
   the last lines of `logs/queue_state.tsv` (`step`, `ok`/`FAILED`/`SKIPPED`,
   value, duration).

How to judge a difference:

- The **truncated** regime (`results/`, `results-holdout/`) is what
  `run_experiments.py` reports; it underestimates and is only for ranking arms
  against each other. Report numbers in the **leaderboard** regime
  (`results-regimes/quantized/`), which matches the public PIRB leaderboard.
- Seed noise of one recipe: std 0.20 on tiny, 0.18 on held-out-8 (spread
  0.39 / 0.34), so differences up to 0.4 between single arms are noise.
- The maupqa / wiki-def / legal tasks have per-task std <= 0.15 and are a
  sensitive detector of damage; the BEIR-like tasks (allegro-faq, fiqa-pl,
  nfcorpus, arguana) are noisy and need the leaderboard regime or seeds.

Other files in `results/`: `REPORT-2026-10-24.md` (the first written report),
`summary.md` (an early `scripts/compare_results.py` table), `sparsity/`,
`index_cost*.json`, `data/` (cut manifests, pool diagnostics, length bias, probes).

### Historical scripts

The queues in `queues/` and the generators in `scripts/` were moved here
unchanged, so they still use the paths from before the move (`configs/next/...`,
`results/...`, `scripts/build_*`, `cd "$(dirname "$0")/.."`). To rerun one as it
was, check out commit `61cf632` (the last commit before the move); for new work
copy the patterns into a new queue as described in `docs/queues.md`.
