#!/usr/bin/env bash
# A/B check of the PIRB evaluation code: run the same model on the same task
# with (A) the vendored fork at HEAD (third_party/pirb) and (B) the last upstream
# commit before our changes (7a0a8b6, "Support for late-interaction models"),
# checked out as a git worktree. Runs on CPU so it does not touch the GPU queue.
#
#   scripts/pirb_ab_check.sh [TASK=scifact-pl] [MODEL=sdadas/polish-splade]
#
# Writes logs/pirb_ab/<run>.log and prints the nDCG@10 lines side by side.
# A third run (C) evaluates models/base-splade-wrapped with the upstream code,
# which checks the wrapper directory itself.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TASKS=(${1:-scifact-pl arguana-pl nfcorpus-pl})
MODEL="${2:-sdadas/polish-splade}"
REPO="$(pwd)"
PY="$REPO/.venv/bin/python"
UP_COMMIT=7a0a8b6
UP_DIR=/home/jsowa/work/pirb-upstream-$UP_COMMIT
SCRATCH="$REPO/logs/pirb_ab"
mkdir -p "$SCRATCH"
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${AB_CUDA:-}"   # empty = CPU; AB_CUDA=0 uses the card
export ORPHAN_GUARD_EXEMPT=1
OLDPY=/home/jsowa/work/venv-pyserini021/bin/python   # same lock as .venv but pyserini 0.21.0 (JFloat query weights)
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
log() { echo "[$(date +%H:%M:%S)] [ab] $*" | tee -a "$SCRATCH/ab.log"; }

[ -d "$UP_DIR" ] || git -C third_party/pirb worktree add "$UP_DIR" "$UP_COMMIT" >> "$SCRATCH/ab.log" 2>&1 || { log "worktree failed"; exit 1; }


mk_models() {  # mk_models MODEL OUT [quantize_queries]
    echo "[{\"name\": \"$1\", \"fp16\": false, \"type\": \"splade\", \"threads\": 6, \"batch_size\": 16, \"quantize_queries\": ${3:-false}}]" > "$2"
}

# the 16 GB machine is already swapping with the desktop apps open: start a run
# only when >= 6 GB are available (the run itself stays under the cgroup cap)
wait_ram() {
    local waited=0 avail
    while :; do
        avail=$(free -g | awk '/^Mem:/ {print $7}')
        [ "${avail:-0}" -ge "${AB_MIN_RAM_GB:-8}" ] && return 0
        [ "$waited" -ge 10800 ] && { log "RAM still ${avail} GB after 3 h; giving up"; exit 1; }
        sleep 60; waited=$((waited + 60))
    done
}

run() {  # run LABEL DIR MODEL [quantize_queries] [python]
    local label="$1" dir="$2" model="$3" qq="${4:-false}" py="${5:-$PY}" cfg="$SCRATCH/models_${1}_$TASK.json"
    mk_models "$model" "$cfg" "$qq"
    wait_ram
    log "run $label ($TASK): $(git -C "$dir" log -1 --format='%h %s' | cut -c1-60) quantize_queries=$qq python=$(basename "$(dirname "$(dirname "$py")")")"
    local s=$SECONDS
    ( cd "$dir" && nice -n 10 "$py" run_benchmark.py --models_config "$cfg" --benchmark_config "$SCRATCH/bench_$TASK.json" \
        --data_dir "$REPO/third_party/pirb/data" --cache_dir "$SCRATCH/cache_${label}_$TASK" --scope full --overwrite ) > "$SCRATCH/${label}_$TASK.log" 2>&1
    rm -rf "$SCRATCH/cache_${label}_$TASK"
    log "run $label ($TASK) finished in $((SECONDS - s))s: $(grep -E "Average NDCG|Killed|Error" "$SCRATCH/${label}_$TASK.log" | tail -n 1 | cut -c1-100)"
}

# leaderboard (29/09/2023): scifact-pl 64.70, arguana-pl 36.44, nfcorpus-pl 29.84
# ours (fork, float query weights truncated by pyserini 0.39): 62.75, 41.72, 30.79
for TASK in "${TASKS[@]}"; do
    echo "[{\"task_id\": \"$TASK\", \"type\": \"beir\"}]" > "$SCRATCH/bench_$TASK.json"
    run A_fork_head        "$REPO/third_party/pirb" "$MODEL" false
    run D_fork_quantized   "$REPO/third_party/pirb" "$MODEL" true
    run E_fork_oldpyserini "$REPO/third_party/pirb" "$MODEL" false "$OLDPY"
    run B_upstream         "$UP_DIR"                "$MODEL" false
done
log "done"
