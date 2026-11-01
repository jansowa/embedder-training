# Writing a queue

A queue is a bash script that runs a series of GPU measurements one after
another, unattended, on a single small machine (the reference box is an 8 GB
RTX 3060 Ti with 16 GB RAM). The helpers live in `scripts/queue_lib.sh`; a queue
sources them and is mostly a list of calls such as `train`, `bench`, `holdout`
and `regime`. Each step:

- waits for an idle GPU, enough free RAM and enough free disk before starting;
- runs under `timeout`, logs its outcome and never aborts the queue on failure;
- is skipped when its result already exists, so a crashed or killed queue is
  resumed by starting it again.

Queues belong to an experiment series: put them in
`experiments/<series>/queues/queue_vN.sh` next to the series' configs and
results, never in `scripts/` (that directory holds only the generic helpers).

## Skeleton

```bash
#!/usr/bin/env bash
# What this queue answers, the steps in order, and the start command:
# setsid nohup bash experiments/<series>/queues/queue_v26.sh > logs/queue_v26.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1      # repository root: every path below is relative to it
export QUEUE_TAG=v26 MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=16000                   # optional: a GradCache run needs more than the default 7200 s
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v26.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"

# The previous queue owns the GPU and the git index until it exits.
if [ -f "$LOGS/queue_v25.pid" ] && kill -0 "$(cat "$LOGS/queue_v25.pid")" 2>/dev/null; then
    log "waiting for queue v25 to exit"
    while kill -0 "$(cat "$LOGS/queue_v25.pid")" 2>/dev/null; do sleep 60; done
    sleep 30
fi

log "=== v26 start: base $(tiny base-splade-wrapped), $(free_gb) GB free ==="
train n4x-fever "$C/n4x_fever_cb32.yaml"                     # -> models/arm-n4x-fever
bench arm-n4x-fever models/arm-n4x-fever                     # tiny, truncated regime
sparsity arm-n4x-fever models/arm-n4x-fever
holdout arm-n4x-fever models/arm-n4x-fever                   # held-out-8, truncated regime
regime quantized arm-n4x-fever models/arm-n4x-fever tiny10 16     # leaderboard regime
regime quantized arm-n4x-fever models/arm-n4x-fever heldout8 16

"$PY" scripts/queue_summary.py > "$RESULTS_ROOT/summary_v26.md"
git add "$RESULTS_ROOT/arm-n4x-fever" "$HOLDOUT_ROOT/arm-n4x-fever" "$REGIMES_ROOT/quantized" \
        "$RESULTS_ROOT/sparsity" "$RESULTS_ROOT/summary_v26.md"
git commit -q -m "Add the v26 results" && \
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v26.log" 2>&1 && break; sleep 120; done ) \
    > /dev/null 2>&1 < /dev/null &
log "=== v26 done ==="
```

Start it detached so it survives the terminal and the tool shell that launched
it: `setsid nohup bash experiments/<series>/queues/queue_v26.sh > logs/queue_v26.out 2>&1 &`.

## Settings

Set these before `source scripts/queue_lib.sh` (or in the environment of the
launching shell).

| variable | default | meaning |
|---|---|---|
| `QUEUE_TAG` | `q` | prefix of every log line |
| `MIN_RAM_GB` | `6` | a step waits (up to 2 h) until this much RAM is available |
| `TRAIN_TIMEOUT` | `7200` | wall-clock limit of one `train` call, seconds |
| `EXP_ROOT` | `experiments/polish-splade-2026-10` | the experiment series |
| `RESULTS_ROOT` | `$EXP_ROOT/results` | tiny results, sparsity, summaries |
| `REGIMES_ROOT` | `$EXP_ROOT/results-regimes` | leaderboard-regime results |
| `HOLDOUT_ROOT` | `$EXP_ROOT/results-holdout` | held-out-8 results |
| `HOLDOUT2_ROOT` | `$EXP_ROOT/results-holdout2` | held-out-2 aggregates |
| `JAVA_HOME` | first `~/opt/jdk-21*` or `/usr/lib/jvm/*21*` | JDK 21 for Anserini (PIRB) |

The result roots are exported, so `scripts/queue_summary.py`,
`scripts/regime_task_table.py`, `scripts/seed_noise_table.py`,
`scripts/sparsity_table.py` and `scripts/leaderboard_compare.py` called from the
queue read the same directories (their defaults are in `scripts/result_roots.py`).
A new experiment series sets `EXP_ROOT=experiments/<new-series>` and gets its own
result tree.

## Helpers

Measurement steps (each skips itself when its output exists):

| call | does | output | timeout |
|---|---|---|---|
| `train TAG CONFIG` | `run_experiments.py --grid-configuration-file CONFIG`, then copies the newest `runs/arm-TAG/sentence-transformers/splade/*/final` to `models/arm-TAG` | `models/arm-TAG`, `logs/arm_TAG.log` | `$TRAIN_TIMEOUT` |
| `bench NAME MODEL [args]` | PIRB tiny (10 tasks), truncated-query regime, 512 tokens | `$RESULTS_ROOT/NAME/final/metrics.json` | 5400 s |
| `holdout NAME MODEL` | PIRB held-out-8 (`$HOLDOUT_TASKS`), truncated regime, encoding batch 16 | `$HOLDOUT_ROOT/NAME/final/metrics.json` | 9000 s |
| `sparsity NAME MODEL` | active document dimensions on 300 passages of each tiny task, next to the base | `$RESULTS_ROOT/sparsity/NAME.json` | 1800 s |
| `sparsity_missing` | `sparsity` for every model with a tiny result and no sparsity file | | |
| `regime REGIME LABEL MODEL TASKSET [BATCH] [MAXLEN]` | `scripts/pirb_regime_eval.sh`; TASKSET is `configs/pirb_tasks/<name>.json` (`tiny10`, `heldout8`, `heldout2_<task>`), REGIME is `quantized` (leaderboard), `truncated`, `old-pyserini` or `new-backend` | `$REGIMES_ROOT/REGIME/LABEL/TASKSET.json` | 10800 s |
| `split_block SRC TAG...` | trains `$EXP_ROOT/configs/weekend/splits/arm_SRC_TAG.yaml` for each tag, benches all, held-out for the control (`rand`) and the best split when it wins by 0.4 | | |

`train` requires the config's `runs_dir` to be `runs/arm-TAG`: it looks for the
newest `runs/arm-TAG/sentence-transformers/splade/<run>/final` (the layout the
SentenceTransformers SPLADE backend writes with `timestamp_output_dir: true`).
`models/arm-TAG` existing means "trained", so delete it to retrain.

Reading results back:

| call | returns |
|---|---|
| `tiny NAME` | tiny nDCG@10 average of `$RESULTS_ROOT/NAME`, rounded to 2 places |
| `hold NAME` | held-out-8 average of `$HOLDOUT_ROOT/NAME` |
| `regime_mean FILE` | `mean n_tasks` of a regime result |
| `sparsity_mean FILE LABEL` | mean active dims per document |
| `ge A B` | `1` when A >= B (both non-empty), else `0` |
| `fsub A B` | A - B, rounded |

Guards and bookkeeping:

| call | does |
|---|---|
| `log MSG` | timestamped line to stdout and `logs/queue.log` (shared by all queues) |
| `record STEP STATUS VALUE [DURATION]` | one row in `logs/queue_state.tsv`; every measurement step records itself |
| `need_space` | waits up to 1 h for 10 GB free on `/`, else the step is skipped |
| `wait_gpu_free` | waits until no compute process is on the GPU (gives up after 40 min and proceeds) |
| `wait_ram` | waits until `MIN_RAM_GB` is available (gives up after 2 h and proceeds) |
| `guard_alive` | restarts `scripts/orphan_guard.py` when it is not running |
| `drop_indexes MODEL` | removes the PIRB index directories a failed benchmark left in `third_party/pirb/cache` |
| `free_gb`, `gpu_busy` | free GB on `/`, number of GPU compute processes |

## Patterns worth copying

From `experiments/polish-splade-2026-10/queues/` (historical: those scripts
still use the paths from before the move to `experiments/`, so copy patterns,
not paths):

- a deadline: `step() { before_deadline || return 1; "$@"; }` and every call as
  `step bench ...` (`queue_v23.sh`, `queue_v25.sh`);
- a retry with a fallback config when a 2 x 16 run hits CUDA OOM (`train2` in
  `queue_v25.sh`);
- gates between steps: compare `tiny` / `hold` / regime means with `ge` and
  `fsub` and run the expensive held-out or leaderboard step only for arms that
  pass (`queue_v25.sh` step 4);
- weight-averaged soups with `scripts/merge_splade_weights.py` (`merge_lb` in
  `queue_v25.sh`);
- a summary at the end: `scripts/queue_summary.py`, then per-task tables from
  `scripts/regime_task_table.py` and `scripts/seed_noise_table.py`, then
  `git add` of exactly the result paths, a commit, and a push with retries in
  the background.

## Rules

- One GPU job at a time on the machine, including CPU-heavy jobs: with 16 GB RAM
  a job next to a queue step can freeze the box. Chain queues by waiting on the
  previous queue's pid file, never run two.
- Never edit a running queue script: bash reads the file as it executes, so an
  edit changes or breaks the steps it has not reached. Write a new queue that
  waits for the old one instead.
- `scripts/orphan_guard.py` kills, after 30 s, worker processes
  (`run_experiments.py`, `training.train`, `run_benchmark.py`,
  `compare_splade_sparsity`, ...) that have no terminal and no ancestor matching
  `bash .../scripts/*.sh` or `bash .../queues/*.sh`. Start queues with
  `bash <path>` as above; run an ad-hoc GPU job from a terminal, or with
  `ORPHAN_GUARD_EXEMPT=1` in its environment.
- `scripts/disk_guard.sh` prunes PIRB index directories whose task score is
  already saved under `$RESULTS_ROOT` / `$HOLDOUT_ROOT` when `/` drops below
  20 GB free.
- Use `WANDB_MODE=offline` (or `disabled`) in queues; nothing should block on the
  network.
- Commit only the result paths the queue produced; checkpoints (`models/`,
  `runs/`), per-arm training sets (`datasets/`) and logs are ignored.
