#!/usr/bin/env bash
# Unattended weekend queue: soup alphas, three fine-tunes and their WiSE-FT
# merges, each benchmarked on PIRB tiny.
#
# Deliberately not 'set -e': one failed step must not cancel the independent
# steps behind it. Every step records its own status in $STATE and the summary
# at the end reports what ran and what did not.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$ROOT/.venv/bin/python"
# training/dataset_sources.py downloads with urlopen and reads the token only
# from the environment, so the cached huggingface_hub login is not enough for
# the private speakleash datasets.
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
LOGS="$ROOT/logs"
STATE="$LOGS/queue_state.tsv"
SUMMARY="$LOGS/queue_summary.md"
mkdir -p "$LOGS"
# Append: a restart after a crash must not erase what already ran.
printf '# --- queue started %s ---\n' "$(date -Is)" >> "$STATE"

BASE_AVG=60.14
RUN_ROOT="$ROOT/runs/weekend-3060/sentence-transformers/splade"

log()   { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

# --- helpers ---------------------------------------------------------------

# merge <output-name> <model-a> <model-b> <alpha>
merge() {
    local out="models/$1" a="$2" b="$3" alpha="$4"
    if [ -d "$out" ]; then log "merge $1: already present, skipping"; return 0; fi
    log "merge $1 (alpha=$alpha)"
    "$PY" scripts/merge_splade_weights.py --models "$a" "$b" --alpha "$alpha" \
        --output "$out" > "$LOGS/merge_$1.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && record "merge:$1" ok "-" || record "merge:$1" FAILED "see logs/merge_$1.log"
    return $rc
}

# bench <run-name> <model-path>
bench() {
    local name="$1" model="$2"
    if [ -f "results/$name/final/metrics.json" ]; then
        log "bench $name: metrics already present, skipping"; return 0
    fi
    if [ ! -e "$model" ]; then
        log "bench $name: model $model missing, skipping"
        record "bench:$name" SKIPPED "model missing"; return 1
    fi
    log "bench $name"
    local s=$SECONDS
    "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-max-seq-length 512 --remove-pirb-indexes \
        --benchmark-target "${model}::" \
        --benchmark-output-dir "results/$name" > "$LOGS/bench_$name.log" 2>&1
    local rc=$? e=$((SECONDS - s))
    if [ -f "results/$name/final/metrics.json" ]; then
        local avg n
        avg=$("$PY" -c "import json;print(round(json.load(open('results/$name/final/metrics.json'))['epoch0/pirb_average_ndcg@10'],2))" 2>/dev/null)
        n=$("$PY" -c "import json;print(json.load(open('results/$name/final/metrics.json'))['epoch0/pirb_datasets'])" 2>/dev/null)
        log "bench $name -> ndcg=$avg tasks=$n in ${e}s"
        [ "$n" = "10" ] || log "WARNING: $name evaluated $n tasks, expected 10"
        record "bench:$name" ok "$avg" "${e}s tasks=$n"
        "$PY" scripts/compare_results.py "results/*/*/metrics.json" \
            --leaderboard results/leaderboard_data.json --format markdown \
            --output results/summary.md >> "$LOGS/compare.log" 2>&1
    else
        log "bench $name FAILED (rc=$rc) after ${e}s"
        record "bench:$name" FAILED "rc=$rc" "see logs/bench_$name.log"
    fi
    return $rc
}

# Did this training log hit a real CUDA OOM? The 7.6 GB soft ceiling is
# deliberately not enforced: the user asked to let a tight run finish.
hit_oom() {
    grep -qiE "CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError" "$1"
}

# Sample GPU memory for the first 10 minutes of a training run.
sample_gpu() {
    local out="$1" i=0
    while [ $i -lt 10 ]; do
        printf '%s\t%s\n' "$(date +%H:%M:%S)" \
            "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)" >> "$out"
        sleep 60; i=$((i + 1))
    done
}

# train <tag> <config> <results-name>
#
# The configs carry no 'benchmark.output_dir', and without one no metrics.json
# is written at all. Passing it on the command line wins over the config and
# puts the run's own benchmark next to every other result.
train() {
    # Split: bash expands every word of a 'local' before running it, so a
    # self-reference like logfile="$LOGS/${tag}.log" would read tag unset.
    local tag="$1" cfg="$2" results="$3"
    local logfile="$LOGS/${tag}.log"
    log "train $tag with $cfg (benchmark -> results/$results)"
    sample_gpu "$LOGS/${tag}_gpu.tsv" &
    local sampler=$!
    local s=$SECONDS
    # run_experiments always forwards --backend/--training-type with their
    # defaults, and train.py lets the CLI win over the config, so the configs'
    # own 'backend: sentence-transformers' would otherwise be ignored.
    "$PY" run_experiments.py --grid-configuration-file "$cfg" \
        --backend sentence-transformers --training-type splade \
        --run_pirb --pirb_scope tiny --remove-pirb-indexes \
        --benchmark-output-dir "results/$results" > "$logfile" 2>&1
    local rc=$? e=$((SECONDS - s))
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    local peak
    peak=$(awk -F'\t' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END{print m+0}' "$LOGS/${tag}_gpu.tsv" 2>/dev/null)
    if hit_oom "$logfile"; then
        log "train $tag: CUDA OOM after ${e}s (peak ${peak} MiB)"
        record "train:$tag" OOM "peak=${peak}MiB" "${e}s"
        return 2
    fi
    if [ $rc -ne 0 ]; then
        log "train $tag FAILED rc=$rc after ${e}s (peak ${peak} MiB)"
        record "train:$tag" FAILED "rc=$rc peak=${peak}MiB" "${e}s"
        return 1
    fi
    log "train $tag ok in ${e}s (peak ${peak} MiB)"
    record "train:$tag" ok "peak=${peak}MiB" "${e}s"
    return 0
}

# Newest <run>/final whose run name contains <pattern>. The pattern guards
# against a failed run silently handing the previous run's checkpoint to the
# WiSE-FT merges.
latest_final() {
    local pattern="$1"
    ls -1dt "$RUN_ROOT"/*"$pattern"*/final 2>/dev/null | head -1
}

# Did this training run leave metrics behind? Returns the path, or nothing.
run_metrics() {
    ls -1 "results/$1"/*/metrics.json 2>/dev/null | head -1
}

# Benchmark a training checkpoint that produced no metrics of its own, through
# a short path so the PIRB cache name cannot hit the 255-byte filename limit.
bench_checkpoint_copy() {
    local name="$1" src="$2"
    [ -d "$src" ] || return 1
    log "$name: no metrics from the in-config benchmark, benching a short copy"
    rm -rf "models/$name"
    cp -r "$src" "models/$name" && bench "$name" "models/$name"
}

# wiseft <label> <ft-dir> <alpha...>
wiseft() {
    local label="$1" ft="$2"; shift 2
    for alpha in "$@"; do
        local name="wiseft-${label}-a${alpha}"
        merge "$name" sdadas/polish-splade "$ft" "$alpha" || continue
        bench "$name" "models/$name"
    done
}

# --- phase 1: remaining soup alphas ---------------------------------------

log "=== phase 1: soup alphas 0.15 and 0.35 ==="
for alpha in 0.15 0.35; do
    merge "soup-a${alpha}" sdadas/polish-splade models/mlm-msmarco-ep1 "$alpha" \
        && bench "soup-a${alpha}" "models/soup-a${alpha}"
done

# --- phase 2: R1, fiqa reproduction ---------------------------------------

log "=== phase 2: R1 fiqa reproduction ==="
train r1_fiqa configs/weekend/ft_fiqa_3060_repro.yaml ft-fiqa-repro
R1_RC=$?
FIQA_FT_DIR=""
if [ $R1_RC -eq 0 ]; then
    FIQA_FT_DIR="$(latest_final lr-5e-5-ep-2)"
    log "R1 final: $FIQA_FT_DIR"
    record "r1:final_dir" ok "$FIQA_FT_DIR"
    # Rows before/after the filter and the optimizer step count, for the report.
    grep -ohiE "[0-9]+ rows|filtered[^.]*[0-9]+|optimization steps[ =:]+[0-9]+|num_train_steps[ =:]+[0-9]+" \
        "$LOGS/r1_fiqa.log" 2>/dev/null | sort -u > "$LOGS/r1_counts.txt"
    [ -n "$(run_metrics ft-fiqa-repro)" ] || bench_checkpoint_copy ft-fiqa-repro "$FIQA_FT_DIR"
else
    log "R1 did not finish (rc=$R1_RC): skipping its WiSE-FT merges"
fi

# --- phase 3: WiSE-FT on the fiqa checkpoint ------------------------------

if [ -n "$FIQA_FT_DIR" ]; then
    log "=== phase 3: WiSE-FT fiqa ==="
    wiseft fiqa "$FIQA_FT_DIR" 0.5 0.3 0.7
else
    record "phase3" SKIPPED "no fiqa checkpoint"
fi

# --- phase 4: N1, rehearsal grid at lr 2e-5 -------------------------------

# No longer gated on R1: that gate assumed R1 probes the memory N1 needs, but
# N1 is the one config that already had gradient checkpointing on, so it has
# its own memory profile and deserves its own attempt.
if true; then
    log "=== phase 4: N1 rehearsal grid lr 2e-5 ==="
    train n1_grid configs/weekend/ft_grid_3060_lr2e-5.yaml ft-3060-2e-5
    if [ $? -eq 0 ]; then
        N1_FINAL="$(latest_final lr-2e-5-ep-1)"
        N1_RUN="$(dirname "$N1_FINAL")"
        log "N1 final: $N1_FINAL"
        record "n1:final_dir" ok "$N1_FINAL"
        if [ -d "$N1_RUN/frac-0.80" ]; then
            bench ft3060-2e-5-frac080 "$N1_RUN/frac-0.80"
        else
            log "N1: no frac-0.80 checkpoint"
            record "bench:ft3060-2e-5-frac080" SKIPPED "checkpoint missing"
        fi
        [ -n "$(run_metrics ft-3060-2e-5)" ] || bench_checkpoint_copy ft-3060-2e-5 "$N1_FINAL"
        wiseft 3060 "$N1_FINAL" 0.5 0.3 0.7
    else
        log "N1 did not finish; not resuming, as instructed"
    fi
else
    log "=== phase 4 skipped: R1 hit OOM ==="
    record "phase4" SKIPPED "R1 OOM"
fi

# --- phase 5: R2, fiqa with warmup ----------------------------------------

log "=== phase 5: R2 fiqa with warmup ==="
train r2_fiqa_warmup configs/weekend/ft_fiqa_3060_warmup.yaml ft-fiqa-warmup
if [ $? -eq 0 ]; then
    R2_FINAL="$(latest_final lr-2e-5-ep-2)"
    log "R2 final: $R2_FINAL"
    record "r2:final_dir" ok "$R2_FINAL"
    [ -n "$(run_metrics ft-fiqa-warmup)" ] || bench_checkpoint_copy ft-fiqa-warmup "$R2_FINAL"
    wiseft fiqa-warmup "$R2_FINAL" 0.5 0.3
fi

# --- phase 6: summary ------------------------------------------------------

log "=== building summary ==="
"$PY" - "$BASE_AVG" <<'PYEOF' > "$SUMMARY" 2>>"$LOGS/queue.log"
import glob, json, sys
base = float(sys.argv[1])
rows = []
for path in sorted(glob.glob("results/*/*/metrics.json")):
    try:
        m = json.load(open(path))
    except Exception:
        continue
    avg = m.get("epoch0/pirb_average_ndcg@10")
    if avg is None:
        continue
    tasks = {
        k.replace("epoch0/pirb_", ""): v
        for k, v in m.items()
        if k.startswith("epoch0/pirb_") and isinstance(v, float) and "@" not in k and not k.endswith("ndcg_tasks")
    }
    no_arg = [v for k, v in tasks.items() if k != "arguana-pl"]
    rows.append((
        path.split("/")[1],
        avg,
        sum(no_arg) / len(no_arg) if no_arg else float("nan"),
        m.get("epoch0/pirb_datasets"),
        path,
    ))
rows.sort(key=lambda r: -r[1])
print("# Weekend queue results (PIRB tiny, NDCG@10)\n")
print(f"Reference B = {base:.2f}\n")
print("| run | mean | without arguana | tasks | delta vs B |")
print("| --- | ---: | ---: | ---: | ---: |")
for name, avg, noarg, n, _ in rows:
    print(f"| {name} | {avg:.2f} | {noarg:.2f} | {n} | {avg - base:+.2f} |")
PYEOF

log "=== queue finished ==="
cat "$STATE" >> "$LOGS/queue.log"
