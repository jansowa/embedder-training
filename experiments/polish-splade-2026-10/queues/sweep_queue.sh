#!/usr/bin/env bash
# Fills the night out to morning, built on what the evening actually showed:
# a short fine-tune on domain data beats the base by +1.35, while every
# weight-averaging variant managed at most +0.28.
#
# Order is by what the result would change, not by what is quick:
#
#   1. Seed repeats. Two recipes differ by 0.39 and nobody knows the run-to-run
#      spread, so every recipe comparison made today rests on an unmeasured
#      quantity. Four observations of one recipe give it.
#   2. Learning rate and warmup around the leader.
#   3. Epochs, including the one-epoch checkpoint that already exists.
#   4. Whether judge-approved data rescues mixing, which hurt when unfiltered.
#   5. Out-of-sample checks and soups, once there is a winner worth checking.
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
SENTINELS_AT_START="${1:-0}"

log()    { echo "[$(date +%H:%M:%S)] [sweep] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

log "waiting for the judge queue (marker count at start: $SENTINELS_AT_START)"
missing=0
while true; do
    now=$(grep -c '=== judge queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)
    [ "$now" -gt "$SENTINELS_AT_START" ] && { log "judge queue reported finished"; break; }
    if pgrep -f "bash scripts/judge_queue.sh" >/dev/null; then
        missing=0
    else
        missing=$((missing + 1))
        [ "$missing" -ge 3 ] && { log "judge queue process gone; continuing"; break; }
    fi
    sleep 60
done

bench() {
    local name="$1" model="$2"
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
        "$PY" scripts/compare_results.py "results/*/*/metrics.json" \
            --leaderboard results/leaderboard_data.json --format markdown \
            --output results/summary.md >> "$LOGS/compare.log" 2>&1
    else
        log "bench $name FAILED after ${e}s"; record "bench:$name" FAILED "-" "${e}s"
    fi
}

bench_holdout() {
    local name="$1" model="$2"
    [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ] && { log "holdout $name: already present"; return 0; }
    [ -e "$model" ] || return 1
    log "holdout $name"
    local s=$SECONDS
    "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-tasks "$HOLDOUT_TASKS" --pirb-max-seq-length 512 --remove-pirb-indexes \
        --benchmark-target "${model}::" --benchmark-output-dir "$HOLDOUT_DIR/$name" \
        > "$LOGS/holdout_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ]; then
        local avg
        avg=$("$PY" -c "import json;print(round(json.load(open('$HOLDOUT_DIR/$name/final/metrics.json'))['epoch0/pirb_average_ndcg@10'],2))" 2>/dev/null)
        log "holdout $name -> ndcg=$avg in ${e}s"; record "holdout:$name" ok "$avg" "${e}s"
    else
        log "holdout $name FAILED after ${e}s"; record "holdout:$name" FAILED "-" "${e}s"
    fi
}

# run <tag> <config>: train, then benchmark whatever checkpoint it leaves.
# The exit code is uninformative here, because the benchmark the training
# config runs next always dies for lack of VRAM; final/ is what counts.
run() {
    local tag="$1" cfg="$2"
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
    if [ -z "$ck" ]; then
        log "train $tag produced no checkpoint after ${e}s"; record "train:sw-$tag" FAILED "-" "${e}s"; return 1
    fi
    log "train $tag done in ${e}s"; record "train:sw-$tag" ok "-" "${e}s"
    rm -rf "models/sw-$tag"
    cp -r "$ck" "models/sw-$tag" || return 1
    bench "sw-$tag" "models/sw-$tag"
}

# --- 1. how much does the seed alone move the number? ---------------------

log "=== block 1: seed repeats of the leading recipe ==="
run seed43 configs/weekend/sweep_seed43.yaml
run seed44 configs/weekend/sweep_seed44.yaml

# --- 2. learning rate and warmup around the leader ------------------------

log "=== block 2: learning rate ==="
run lr1e-5  configs/weekend/sweep_lr1e-5.yaml
run lr3e-5  configs/weekend/sweep_lr3e-5.yaml
# 5e-5 with warmup, against the 5e-5 without warmup already measured: isolates
# warmup from the learning rate.
run lr5e-5w configs/weekend/sweep_lr5e-5w.yaml

# --- 3. epochs ------------------------------------------------------------

log "=== block 3: epochs ==="
# One epoch of the leading recipe costs nothing: the checkpoint is on disk.
EP1=$(ls -1dt runs/weekend-3060/sentence-transformers/splade/*lr-2e-5-ep-2*/epoch-checkpoints/epoch-0001-step-98 2>/dev/null | head -1)
if [ -n "$EP1" ] && [ ! -d models/ck-warmup-s-ep1 ]; then
    cp -r "$EP1" models/ck-warmup-s-ep1 \
        && rm -f models/ck-warmup-s-ep1/{optimizer.pt,scheduler.pt,rng_state.pth,trainer_state.json}
fi
bench ck-warmup-s-ep1 models/ck-warmup-s-ep1
run ep4 configs/weekend/sweep_ep4.yaml

# --- 4. does the judge make mixing work? ----------------------------------

log "=== block 4: fiqa plus judge-approved MS MARCO ==="
run mix1to1 configs/weekend/sweep_mix1to1.yaml

# --- 5. a third seed, for a steadier spread -------------------------------

log "=== block 5: third seed repeat ==="
run seed45 configs/weekend/sweep_seed45.yaml

# --- 6. soup the best fine-tunes, then check the winner out of sample -----

log "=== block 6: soup of the best fine-tunes ==="
mapfile -t TOP < <("$PY" - <<'PYEOF'
import glob, json
rows = []
for path in glob.glob("results/*/final/metrics.json"):
    run = path.split("/")[1]
    if not (run.startswith("ck-") or run.startswith("sw-") or run.startswith("judge-")):
        continue
    try:
        rows.append((json.load(open(path))["epoch0/pirb_average_ndcg@10"], run))
    except Exception:
        pass
rows.sort(reverse=True)
for _, run in rows[:3]:
    print("models/" + run)
PYEOF
)
log "top fine-tunes: ${TOP[*]:-none}"
if [ "${#TOP[@]}" -ge 2 ]; then
    [ -d models/soup-top3 ] || "$PY" scripts/merge_splade_weights.py --models "${TOP[@]}" \
        --output models/soup-top3 > "$LOGS/merge_soup-top3.log" 2>&1
    bench soup-top3 models/soup-top3
    for alpha in 0.5 0.7; do
        out="models/soup-top3-wiseft-a${alpha}"
        [ -d "$out" ] || "$PY" scripts/merge_splade_weights.py --models sdadas/polish-splade models/soup-top3 \
            --alpha "$alpha" --output "$out" > "$LOGS/merge_soup-top3-a${alpha}.log" 2>&1
        bench "soup-top3-wiseft-a${alpha}" "$out"
    done
fi

log "=== block 7: out-of-sample checks for the best models ==="
mapfile -t BEST2 < <("$PY" - <<'PYEOF'
import glob, json
rows = []
for path in glob.glob("results/*/final/metrics.json"):
    run = path.split("/")[1]
    if run in {"base-splade", "base-splade-wrapped"}:
        continue
    try:
        rows.append((json.load(open(path))["epoch0/pirb_average_ndcg@10"], run))
    except Exception:
        pass
rows.sort(reverse=True)
for _, run in rows[:2]:
    print(run)
PYEOF
)
log "checking out of sample: ${BEST2[*]:-none}"
bench_holdout base "models/base-splade-wrapped"
for run in "${BEST2[@]}"; do
    bench_holdout "$run" "models/$run"
done

log "=== sweep queue finished ==="
