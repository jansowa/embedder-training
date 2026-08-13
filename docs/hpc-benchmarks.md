# Running post-training benchmarks on HPC

This page documents how the PIRB benchmark uses node resources when several
workers share one machine, and what a Slurm submission script should declare so
the workers do not fight for the same cores.

## How the CPU budget is split

`convert_utils.run_pirb` writes the model config consumed by
`third_party/pirb/run_benchmark.py`. Two resource keys are derived from the
allocation rather than hardcoded:

- `threads` — Lucene/Anserini indexing and search threads for one worker.
- `batch_size` — encoding batch size for one worker.

When `threads` is not set explicitly, it is computed as:

```
cpu_budget       = SLURM_CPUS_PER_TASK  (falls back to len(os.sched_getaffinity(0)))
threads_per_worker = max(1, cpu_budget // parallel_workers)
```

`parallel_workers` is the number of PIRB subprocesses the benchmark scheduler
runs at the same time, which equals the number of GPUs it hands out
(`training/benchmarks.py`). With `--cpus-per-task=8` and four GPUs each worker
gets two threads instead of the previous default of eight, so the node sees
8 Java threads instead of 32.

The same number is exported to each PIRB subprocess as `OMP_NUM_THREADS` and
`MKL_NUM_THREADS`, and `TOKENIZERS_PARALLELISM=false` is exported as well, so
torch and the fast tokenizers do not each start a node-sized pool on top of the
Java threads. Variables already present in the parent environment are never
overwritten — exporting them in the batch script keeps full control.

## Overriding the derived values

YAML:

```yaml
benchmark:
  run_pirb: true
  scope: small
  pirb_threads: 4
  pirb_batch_size: 64
```

CLI (takes precedence over YAML):

```bash
python -m training.train \
  --backend sentence-transformers \
  --training-type splade \
  --config configs/your_config.yaml \
  --run-pirb \
  --pirb-threads 4 \
  --pirb-batch-size 64
```

`benchmark.batch_size` remains the MTEB encoding batch size and is intentionally
not reused for PIRB: PIRB corpora are encoded at `pirb_max_seq_length`, so the
two knobs have different memory profiles.

## Recommended sbatch fragment

The snippet below is documentation only — this repository does not ship or edit
Slurm scripts. Adjust the account, partition, and paths to your site.

```bash
#!/bin/bash
#SBATCH --job-name=embedder-train
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32      # ~8 cores per GPU worker; 8 total starves Lucene
#SBATCH --mem=200G
#SBATCH --time=24:00:00

# One thread pool per worker, not one per node. Leave these unset to let
# run_pirb derive them from SLURM_CPUS_PER_TASK and the worker count.
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / 4 ))
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export TOKENIZERS_PARALLELISM=false

# Anserini requires Java 21.
module load Java/21

srun python -m training.train \
  --backend sentence-transformers \
  --training-type splade \
  --config configs/your_config.yaml \
  --run-pirb --pirb-scope small
```

Key points:

- `--cpus-per-task` should scale with the number of GPU workers. Four workers on
  8 cores leaves 2 cores per worker for Lucene indexing, which is the dominant
  CPU cost of a PIRB run on large datasets.
- Divide the exported thread counts by the number of parallel workers, not by
  one. The four PIRB subprocesses run concurrently.
- If you export nothing, the derived values apply, which is the intended default.
