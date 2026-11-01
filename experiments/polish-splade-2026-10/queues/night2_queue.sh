#!/usr/bin/env bash
# Rebuilt night plan, after the fiqa fine-tune turned out to beat the base by
# +0.96 while every weight-averaging variant gained at most +0.28.
#
# The earlier plan assumed fine-tuning hurts and spent the night on merges
# pulled hard toward the base. That assumption came from a single checkpoint
# trained on a different recipe. The order below measures the recipes we
# already trained, then explores the part of the alpha curve that matters when
# the fine-tune is the better model, then checks the whole thing out of sample.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/logs"
STATE="$LOGS/queue_state.tsv"
HOLDOUT_DIR="$ROOT/results-holdout"
HOLDOUT_TASKS="maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022-test-A-legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl"
SHORT="runs/weekend-3060/sentence-transformers/splade"
LONG="runs/weekend-3060-long/sentence-transformers/splade"
mkdir -p "$LOGS"

log()    { echo "[$(date +%H:%M:%S)] [n2] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

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
    [ -e "$model" ] || { log "holdout $name: $model missing"; return 1; }
    log "holdout $name (8 tasks outside tiny)"
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

# copy_ckpt <name> <source dir>
copy_ckpt() {
    local name="$1" src="$2"
    [ -d "$src" ] || { log "copy $name: source $src missing"; return 1; }
    [ -d "models/$name" ] && return 0
    cp -r "$src" "models/$name" || return 1
    # Optimizer and RNG state are training artifacts; the benchmark only needs
    # the model, and they are 300+ MB each.
    rm -f "models/$name"/{optimizer.pt,scheduler.pt,rng_state.pth,trainer_state.json}
    return 0
}

# --- step 1: the recipes we trained but never measured ---------------------

log "=== step 1: benchmark the remaining checkpoints ==="
copy_ckpt ck-grid-s   "$(ls -d $SHORT/*lr-2e-5-ep-1*/final 2>/dev/null | head -1)" && bench ck-grid-s   models/ck-grid-s
copy_ckpt ck-warmup-s "$(ls -d $SHORT/*lr-2e-5-ep-2*/final 2>/dev/null | head -1)" && bench ck-warmup-s models/ck-warmup-s
copy_ckpt ck-fiqa-l   "$(ls -d $LONG/*lr-5e-5-ep-2*/final  2>/dev/null | head -1)" && bench ck-fiqa-l   models/ck-fiqa-l
copy_ckpt ck-warmup-l "$(ls -d $LONG/*lr-2e-5-ep-2*/final  2>/dev/null | head -1)" && bench ck-warmup-l models/ck-warmup-l

# --- step 2: is two epochs already too many? -------------------------------

log "=== step 2: fiqa after one epoch ==="
copy_ckpt ck-fiqa-s-ep1 "$(ls -d $SHORT/*lr-5e-5-ep-2*/epoch-checkpoints/epoch-0001-step-98 2>/dev/null | head -1)" \
    && bench ck-fiqa-s-ep1 models/ck-fiqa-s-ep1

# --- step 3: the alpha range that matters when the fine-tune wins ----------

log "=== step 3: alpha sweep around the best checkpoint ==="
BEST=$("$PY" - <<'PYEOF'
import glob, json
best, name = None, None
for path in glob.glob("results/ck-*/final/metrics.json"):
    try:
        v = json.load(open(path))["epoch0/pirb_average_ndcg@10"]
    except Exception:
        continue
    if best is None or v > best:
        best, name = v, path.split("/")[1]
print(name or "")
PYEOF
)
log "best fine-tune on tiny: ${BEST:-none}"
record "n2:best_ckpt" ok "${BEST:-none}"
if [ -n "$BEST" ]; then
    # 0.15 and 1.00 are already measured for ck-fiqa-s; this fills the gap
    # between them, where WiSE-FT usually peaks.
    for alpha in 0.3 0.5 0.7; do
        out="models/wiseft-${BEST}-a${alpha}"
        if [ ! -d "$out" ]; then
            "$PY" scripts/merge_splade_weights.py --models sdadas/polish-splade "models/$BEST" \
                --alpha "$alpha" --output "$out" > "$LOGS/merge_wiseft-${BEST}-a${alpha}.log" 2>&1 \
                || { record "merge:wiseft-${BEST}-a${alpha}" FAILED "-"; continue; }
        fi
        bench "wiseft-${BEST}-a${alpha}" "$out"
    done
fi

# --- step 4: does any of this survive outside the tuning set? --------------

log "=== step 4: out-of-sample check ==="
WINNER=$("$PY" - <<'PYEOF'
import glob, json
best, name = None, None
for path in glob.glob("results/*/final/metrics.json"):
    run = path.split("/")[1]
    if run in {"base-splade", "base-splade-wrapped"}:
        continue
    try:
        v = json.load(open(path))["epoch0/pirb_average_ndcg@10"]
    except Exception:
        continue
    if best is None or v > best:
        best, name = v, run
print(name or "")
PYEOF
)
log "overall tiny winner: ${WINNER:-none}"
record "n2:winner" ok "${WINNER:-none}"
bench_holdout base "models/base-splade-wrapped"
[ -n "$WINNER" ] && bench_holdout "$WINNER" "models/$WINNER"

# --- step 5: a soup of the fine-tunes that actually beat the base ----------

log "=== step 5: soup over the fine-tunes that beat the base ==="
mapfile -t GOOD < <("$PY" - <<'PYEOF'
import glob, json
base = json.load(open("results/base-splade-wrapped/final/metrics.json"))["epoch0/pirb_average_ndcg@10"]
for path in sorted(glob.glob("results/ck-*/final/metrics.json")):
    try:
        v = json.load(open(path))["epoch0/pirb_average_ndcg@10"]
    except Exception:
        continue
    if v > base:
        print("models/" + path.split("/")[1])
PYEOF
)
log "fine-tunes above the base: ${#GOOD[@]}"
record "n2:good_ckpts" ok "${#GOOD[@]}"
if [ "${#GOOD[@]}" -ge 2 ]; then
    [ -d models/soup-of-ft ] || "$PY" scripts/merge_splade_weights.py --models "${GOOD[@]}" \
        --output models/soup-of-ft > "$LOGS/merge_soup-of-ft.log" 2>&1
    bench soup-of-ft models/soup-of-ft
    for alpha in 0.5 0.7; do
        out="models/soup-of-ft-wiseft-a${alpha}"
        [ -d "$out" ] || "$PY" scripts/merge_splade_weights.py --models sdadas/polish-splade models/soup-of-ft \
            --alpha "$alpha" --output "$out" > "$LOGS/merge_soup-of-ft-a${alpha}.log" 2>&1
        bench "soup-of-ft-wiseft-a${alpha}" "$out"
    done
else
    log "fewer than two fine-tunes beat the base; nothing to soup"
fi

log "=== night2 queue finished ==="
