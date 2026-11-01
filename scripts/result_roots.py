"""Where the benchmark results live, shared by the table scripts and scripts/queue_lib.sh.

Every root can be overridden by an environment variable of the same name;
relative paths are resolved against the working directory (the repository root
for every script in scripts/). The defaults point at the current experiment
series; scripts/queue_lib.sh exports the same variables, so a table printed from
a queue reads exactly what that queue wrote.

  EXP_ROOT       experiments/polish-splade-2026-10
  RESULTS_ROOT   $EXP_ROOT/results            tiny benchmark, sparsity, index cost, summaries
  REGIMES_ROOT   $EXP_ROOT/results-regimes    leaderboard-regime runs (scripts/pirb_regime_eval.sh)
  HOLDOUT_ROOT   $EXP_ROOT/results-holdout    held-out-8, truncated regime
  HOLDOUT2_ROOT  $EXP_ROOT/results-holdout2   held-out-2 per-task aggregates (scripts/aggregate_task_runs.py)
"""

from __future__ import annotations

import os
from pathlib import Path

EXP_ROOT = Path(os.environ.get("EXP_ROOT") or "experiments/polish-splade-2026-10")
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT") or EXP_ROOT / "results")
REGIMES_ROOT = Path(os.environ.get("REGIMES_ROOT") or EXP_ROOT / "results-regimes")
HOLDOUT_ROOT = Path(os.environ.get("HOLDOUT_ROOT") or EXP_ROOT / "results-holdout")
HOLDOUT2_ROOT = Path(os.environ.get("HOLDOUT2_ROOT") or EXP_ROOT / "results-holdout2")
