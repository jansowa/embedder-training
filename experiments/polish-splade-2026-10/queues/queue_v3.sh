#!/usr/bin/env bash
# Sunday queue, in order of how much each measurement can change a decision:
#   1. is encoding batch size 16 harmless (needed for the big held-out corpora)
#   2. held-out for the two models whose held-out runs died on OOM
#   3. tiny gate for the two new seed soups
#   4-6. data-source arms at the FiQA dose and recipe: SQuAD, judged MS MARCO,
#        FiQA+SQuAD, each measured on tiny AND held-out-8, plus sparsity
#   7. held-out for the better soup
#   8. synthetic-query arms, once the generation pipeline drops its marker
#
# Waits for scripts/recover_queue.sh (pid 141715) to finish before touching
# the GPU.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$(pwd)/.venv/bin/python"
LOGS="$(pwd)/logs"
STATE="$LOGS/queue_state.tsv"
HOLDOUT_DIR="$(pwd)/results-holdout"
HOLDOUT_TASKS="maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022-test-A-legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl"
PREV_PID="${PREV_QUEUE_PID:-141715}"
mkdir -p "$LOGS" results/sparsity datasets/synth

log()    { echo "[$(date +%H:%M:%S)] [v3] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

need_space() {
    local gb
    gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    if [ "${gb:-0}" -lt 12 ]; then
        log "only ${gb} GB free - stopping instead of filling the disk"
        record "today" ABORTED "low disk ${gb}GB"
        exit 1
    fi
}

avg_of() { "$PY" -c "import json;print(round(json.load(open('$1'))['epoch0/pirb_average_ndcg@10'],2))" 2>/dev/null; }

# bench NAME MODEL [extra run_experiments args...]
bench() {
    local name="$1"
    local model="$2"
    shift 2
    need_space
    [ -f "results/$name/final/metrics.json" ] && { log "bench $name: already present"; return 0; }
    [ -e "$model" ] || { log "bench $name: $model missing"; record "bench:$name" SKIPPED "missing"; return 1; }
    log "bench $name"
    local s=$SECONDS
    "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-max-seq-length 512 --remove-pirb-indexes "$@" \
        --benchmark-target "${model}::" --benchmark-output-dir "results/$name" \
        > "$LOGS/bench_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "results/$name/final/metrics.json" ]; then
        local avg
        avg=$(avg_of "results/$name/final/metrics.json")
        log "bench $name -> ndcg=$avg in ${e}s"; record "bench:$name" ok "$avg" "${e}s"
    else
        log "bench $name FAILED after ${e}s"; record "bench:$name" FAILED "-" "${e}s"
    fi
}

# holdout NAME MODEL  (always encoding batch 16: the held-out corpora OOM at the default)
holdout() {
    local name="$1"
    local model="$2"
    need_space
    [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ] && { log "holdout $name: already present"; return 0; }
    [ -e "$model" ] || { log "holdout $name: $model missing"; record "holdout:$name" SKIPPED "missing"; return 1; }
    log "holdout $name"
    local s=$SECONDS
    "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-tasks "$HOLDOUT_TASKS" --pirb-max-seq-length 512 --pirb-batch-size 16 \
        --remove-pirb-indexes \
        --benchmark-target "${model}::" --benchmark-output-dir "$HOLDOUT_DIR/$name" \
        > "$LOGS/holdout_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ]; then
        local avg n
        avg=$(avg_of "$HOLDOUT_DIR/$name/final/metrics.json")
        n=$("$PY" -c "import json;print(json.load(open('$HOLDOUT_DIR/$name/final/metrics.json'))['epoch0/pirb_datasets'])" 2>/dev/null)
        log "holdout $name -> ndcg=$avg tasks=$n in ${e}s"; record "holdout:$name" ok "$avg" "${e}s tasks=$n"
    else
        log "holdout $name FAILED after ${e}s"; record "holdout:$name" FAILED "-" "${e}s"
    fi
}

# sparsity NAME MODEL: active document dimensions on tiny passages, next to the base
sparsity() {
    local name="$1"
    local model="$2"
    [ -f "results/sparsity/$name.json" ] && { log "sparsity $name: already present"; return 0; }
    [ -e "$model" ] || return 1
    log "sparsity $name"
    local s=$SECONDS
    "$PY" scripts/compare_splade_sparsity.py --model "base=models/base-splade-wrapped" --model "$name=$model" \
        --scope tiny --max-docs-per-task 500 --device cuda --batch-size 16 \
        --output-json "results/sparsity/$name.json" > "$LOGS/sparsity_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "results/sparsity/$name.json" ]; then
        log "sparsity $name done in ${e}s"; record "sparsity:$name" ok "-" "${e}s"
    else
        log "sparsity $name FAILED after ${e}s"; record "sparsity:$name" FAILED "-" "${e}s"
    fi
}

# arm TAG CONFIG: train, copy final/ to models/arm-TAG, then tiny + held-out + sparsity
arm() {
    local tag="$1"
    local cfg="$2"
    need_space
    if [ ! -d "models/arm-$tag" ]; then
        [ -f "$cfg" ] || { log "arm $tag: config $cfg missing"; record "train:arm-$tag" SKIPPED "no config"; return 1; }
        log "train arm-$tag"
        local s=$SECONDS
        "$PY" run_experiments.py --grid-configuration-file "$cfg" \
            --backend sentence-transformers --training-type splade \
            > "$LOGS/arm_$tag.log" 2>&1
        local e=$((SECONDS - s))
        local ck
        ck=$(ls -1dt "runs/arm-$tag/sentence-transformers/splade"/*/final 2>/dev/null | head -1)
        [ -z "$ck" ] && { log "train arm-$tag: no checkpoint after ${e}s"; record "train:arm-$tag" FAILED "-" "${e}s"; return 1; }
        log "train arm-$tag done in ${e}s"; record "train:arm-$tag" ok "-" "${e}s"
        cp -r "$ck" "models/arm-$tag" || return 1
    fi
    bench "arm-$tag" "models/arm-$tag"
    holdout "arm-$tag" "models/arm-$tag"
    sparsity "arm-$tag" "models/arm-$tag"
}

# --- take over after the fiqa+wikisynth arm (its last step writes the sparsity json)
log "waiting for arm-fiqa-plus-wikisynth to finish (sparsity json)"
until [ -f results/sparsity/arm-fiqa-plus-wikisynth.json ] || grep -q "sparsity arm-fiqa-plus-wikisynth" logs/queue.log; do sleep 30; done
sleep 60
kill 194881 2>/dev/null; sleep 3; pkill -P 194881 2>/dev/null; sleep 2
log "took over from queue_v2: reordered remaining arms"
sparsity arm-fiqa-plus-wikisynth models/arm-fiqa-plus-wikisynth
log "=== v3 block 1: fiqa+squadv2 ==="
arm fiqa-plus-squadv2  configs/weekend/arm_fiqa_plus_squadv2.yaml
log "=== v3 block 2: trivia at the FiQA dose ==="
arm trivia-2073        configs/weekend/arm_trivia_2073.yaml
log "=== v3 block 3: msmarco-clean-2073 (dose test) ==="
arm msmarco-clean-2073 configs/weekend/arm_msmarco_clean_2073.yaml
log "=== v3 block 4: index-cost profile for all new arms ==="
"$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 \
    --model base=models/base-splade-wrapped --model soup-of-ft=models/soup-of-ft \
    --model squadv2-2073=models/arm-squadv2-2073 --model wikisynth-2073=models/arm-wikisynth-2073 \
    --model fiqa-plus-wikisynth=models/arm-fiqa-plus-wikisynth --model fiqa-plus-squadv2=models/arm-fiqa-plus-squadv2 \
    --model trivia-2073=models/arm-trivia-2073 --model msmarco-clean-2073=models/arm-msmarco-clean-2073 \
    --output-json results/index_cost_arms.json > "$LOGS/index_cost_arms.log" 2>&1 && log "index-cost profile written" || log "index-cost profile FAILED"
log "=== v3 block 5 (optional): held-out for soup-top8 ==="
holdout soup-top8 models/soup-top8
log "=== v3 queue finished ==="
