#!/usr/bin/env bash
# Redo what the full disk killed around 02:55, out-of-sample checks first.
#
# Those checks are the only measurement that can contradict the headline
# result, and they were last in the previous queue, so they were exactly what
# got lost. They go first here.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$(pwd)/.venv/bin/python"
LOGS="$(pwd)/logs"
STATE="$LOGS/queue_state.tsv"
HOLDOUT_DIR="$(pwd)/results-holdout"
HOLDOUT_TASKS="maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022-test-A-legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl"
mkdir -p "$LOGS"

log()    { echo "[$(date +%H:%M:%S)] [recover] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

# Stop before the disk fills again rather than after.
need_space() {
    local gb
    gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    if [ "${gb:-0}" -lt 12 ]; then
        log "only ${gb} GB free - stopping instead of filling the disk"
        record "recover" ABORTED "low disk ${gb}GB"
        exit 1
    fi
}

bench() {
    local name="$1" model="$2"
    need_space
    [ -f "results/$name/final/metrics.json" ] && { log "bench $name: already present"; return 0; }
    [ -e "$model" ] || { log "bench $name: $model missing"; record "bench:$name" SKIPPED "missing"; return 1; }
    log "bench $name"
    local s=$SECONDS
    "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-max-seq-length 512 --remove-pirb-indexes \
        --benchmark-target "${model}::" --benchmark-output-dir "results/$name" \
        > "$LOGS/bench_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "results/$name/final/metrics.json" ]; then
        local avg
        avg=$("$PY" -c "import json;print(round(json.load(open('results/$name/final/metrics.json'))['epoch0/pirb_average_ndcg@10'],2))" 2>/dev/null)
        log "bench $name -> ndcg=$avg in ${e}s"; record "bench:$name" ok "$avg" "${e}s"
    else
        log "bench $name FAILED after ${e}s"; record "bench:$name" FAILED "-" "${e}s"
    fi
}

holdout() {
    local name="$1" model="$2"
    need_space
    [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ] && { log "holdout $name: already present"; return 0; }
    [ -e "$model" ] || { log "holdout $name: $model missing"; record "holdout:$name" SKIPPED "missing"; return 1; }
    log "holdout $name"
    local s=$SECONDS
    "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-tasks "$HOLDOUT_TASKS" --pirb-max-seq-length 512 --remove-pirb-indexes \
        --benchmark-target "${model}::" --benchmark-output-dir "$HOLDOUT_DIR/$name" \
        > "$LOGS/holdout_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ]; then
        local avg n
        avg=$("$PY" -c "import json;print(round(json.load(open('$HOLDOUT_DIR/$name/final/metrics.json'))['epoch0/pirb_average_ndcg@10'],2))" 2>/dev/null)
        n=$("$PY" -c "import json;print(json.load(open('$HOLDOUT_DIR/$name/final/metrics.json'))['epoch0/pirb_datasets'])" 2>/dev/null)
        log "holdout $name -> ndcg=$avg tasks=$n in ${e}s"; record "holdout:$name" ok "$avg" "${e}s tasks=$n"
    else
        log "holdout $name FAILED after ${e}s"; record "holdout:$name" FAILED "-" "${e}s"
    fi
}

run() {
    local tag="$1" cfg="$2"
    need_space
    [ -f "results/sw-$tag/final/metrics.json" ] && { log "run $tag: already done"; return 0; }
    log "train $tag"
    local s=$SECONDS
    "$PY" run_experiments.py --grid-configuration-file "$cfg" \
        --backend sentence-transformers --training-type splade \
        --run_pirb --pirb_scope tiny --remove-pirb-indexes \
        --benchmark-output-dir "results/sw-$tag-inconfig" > "$LOGS/sweep_$tag.log" 2>&1
    local e=$((SECONDS - s))
    local ck
    ck=$(ls -1dt "runs/sweep-$tag/sentence-transformers/splade"/*/final 2>/dev/null | head -1)
    [ -z "$ck" ] && { log "train $tag: no checkpoint after ${e}s"; record "train:sw-$tag" FAILED "-" "${e}s"; return 1; }
    log "train $tag done in ${e}s"; record "train:sw-$tag" ok "-" "${e}s"
    rm -rf "models/sw-$tag"; cp -r "$ck" "models/sw-$tag" || return 1
    bench "sw-$tag" "models/sw-$tag"
}

# --- 1. the out-of-sample check, this time first --------------------------

log "=== block 1: out-of-sample check (8 tasks outside tiny) ==="
holdout base            models/base-splade-wrapped
holdout soup-of-ft      models/soup-of-ft
holdout ck-warmup-l     models/ck-warmup-l
holdout ck-warmup-s     models/ck-warmup-s

# --- 2. the one benchmark that died -------------------------------------

log "=== block 2: missing benchmark ==="
bench ck-warmup-s-ep1 models/ck-warmup-s-ep1

# --- 3. the four trainings that died ------------------------------------

log "=== block 3: trainings lost to the full disk ==="
run seed45  configs/weekend/sweep_seed45.yaml
run lr5e-5w configs/weekend/sweep_lr5e-5w.yaml
run ep4     configs/weekend/sweep_ep4.yaml
run mix1to1 configs/weekend/sweep_mix1to1.yaml

log "=== recover queue finished ==="
