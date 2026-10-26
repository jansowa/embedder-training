#!/usr/bin/env bash
# Third shift. Two jobs the first two queues do not cover:
#
#   A. Out-of-sample check. Every number so far comes from the same ten tasks,
#      and the winning alpha was chosen on them, so the headline +0.25 is a
#      within-sample result. This re-measures the tiny-set winner and the base
#      on eight PIRB tasks that are already downloaded but were never used.
#   B. A soup over several fine-tunes at once, which is the recipe that is
#      supposed to pay off, rather than one checkpoint at a time.
#
# Held-out results go outside results/ on purpose: they cover 8 tasks, not 10,
# and must never land in a table that averages the 10-task runs.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/logs"
STATE="$LOGS/queue_state.tsv"
mkdir -p "$LOGS"

BASE_MODEL="models/base-splade-wrapped"
HOLDOUT_DIR="$ROOT/results-holdout"
# Mid-sized PIRB tasks that are on disk and are not part of tiny. '--pirb-tasks'
# selects by id, which bypasses the 20 MB rule that defines tiny.
HOLDOUT_TASKS="maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022-test-A-legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl"
SENTINELS_AT_START="${1:-$(grep -c '=== night queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)}"

log()    { echo "[$(date +%H:%M:%S)] [sunday] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

log "waiting for the night queue (marker count at start: $SENTINELS_AT_START)"
missing=0
while true; do
    now=$(grep -c '=== night queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)
    [ "$now" -gt "$SENTINELS_AT_START" ] && { log "night queue reported finished"; break; }
    if pgrep -f "bash scripts/night_queue.sh" >/dev/null; then
        missing=0
    else
        missing=$((missing + 1))
        [ "$missing" -ge 3 ] && { log "night queue process gone; continuing"; break; }
    fi
    sleep 60
done

# --- helpers ---------------------------------------------------------------

# bench_tiny <run-name> <model>
bench_tiny() {
    local name="$1" model="$2"
    [ -f "results/$name/final/metrics.json" ] && { log "bench $name: already present"; return 0; }
    [ -e "$model" ] || { log "bench $name: $model missing"; record "bench:$name" SKIPPED "model missing"; return 1; }
    log "bench $name (tiny)"
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

# bench_holdout <run-name> <model>
bench_holdout() {
    local name="$1" model="$2"
    [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ] && { log "holdout $name: already present"; return 0; }
    [ -e "$model" ] || { log "holdout $name: $model missing"; record "holdout:$name" SKIPPED "model missing"; return 1; }
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

# --- phase A: out-of-sample check -----------------------------------------

log "=== phase A: out-of-sample check ==="
BEST_SOUP=$("$PY" - <<'PYEOF'
import glob, json, os
best, name = None, None
for path in glob.glob("results/soup-a*/final/metrics.json"):
    try:
        v = json.load(open(path))["epoch0/pirb_average_ndcg@10"]
    except Exception:
        continue
    if best is None or v > best:
        best, name = v, path.split("/")[1]
print(name or "")
PYEOF
)
log "tiny-set winner among the soups: ${BEST_SOUP:-none}"
record "sunday:best_soup" ok "${BEST_SOUP:-none}"

bench_holdout base "$BASE_MODEL"
[ -n "$BEST_SOUP" ] && bench_holdout "$BEST_SOUP" "models/$BEST_SOUP"

# --- phase B: soup over every fine-tune we managed to train ----------------

log "=== phase B: multi-checkpoint soup ==="
mapfile -t CKPTS < <(ls -1d runs/weekend-3060/sentence-transformers/splade/*/final \
                            runs/weekend-3060-long/sentence-transformers/splade/*/final 2>/dev/null)
log "fine-tuned checkpoints found: ${#CKPTS[@]}"
record "sunday:checkpoints" ok "${#CKPTS[@]}"

if [ "${#CKPTS[@]}" -ge 2 ]; then
    if [ ! -d models/multisoup-ft ]; then
        log "merging ${#CKPTS[@]} checkpoints into models/multisoup-ft (uniform)"
        "$PY" scripts/merge_splade_weights.py --models "${CKPTS[@]}" \
            --output models/multisoup-ft > "$LOGS/merge_multisoup-ft.log" 2>&1 \
            && record "merge:multisoup-ft" ok "${#CKPTS[@]} models" \
            || record "merge:multisoup-ft" FAILED "see logs/merge_multisoup-ft.log"
    fi
    bench_tiny multisoup-ft models/multisoup-ft
    # Then pull that soup toward the base, at the alphas the tiny curve says
    # matter. Large alphas are skipped: the 7-point curve already showed them
    # to be monotonically worse.
    for alpha in 0.10 0.15 0.20; do
        out="models/multisoup-wiseft-a${alpha}"
        if [ ! -d "$out" ]; then
            "$PY" scripts/merge_splade_weights.py --models sdadas/polish-splade models/multisoup-ft \
                --alpha "$alpha" --output "$out" > "$LOGS/merge_multisoup-a${alpha}.log" 2>&1 \
                || { record "merge:multisoup-a${alpha}" FAILED "-"; continue; }
        fi
        bench_tiny "multisoup-wiseft-a${alpha}" "$out"
    done
else
    log "fewer than two checkpoints; nothing to soup"
    record "phaseB" SKIPPED "need >=2 checkpoints"
fi

# --- summaries -------------------------------------------------------------

log "=== building summaries ==="
"$PY" scripts/compare_results.py "results/*/*/metrics.json" \
    --leaderboard results/leaderboard_data.json --format markdown \
    --output results/summary.md >> "$LOGS/compare.log" 2>&1

"$PY" - <<'PYEOF' > "$LOGS/holdout_summary.md" 2>>"$LOGS/queue.log"
import glob, json
rows = []
for path in sorted(glob.glob("results-holdout/*/*/metrics.json")):
    try:
        m = json.load(open(path))
    except Exception:
        continue
    avg = m.get("epoch0/pirb_average_ndcg@10")
    if avg is None:
        continue
    tasks = {k.replace("epoch0/pirb_", ""): v for k, v in m.items()
             if k.startswith("epoch0/pirb_") and isinstance(v, float)
             and "@" not in k and not k.endswith("ndcg_tasks")}
    rows.append((path.split("/")[1], avg, m.get("epoch0/pirb_datasets"), tasks))
print("# Out-of-sample check (PIRB tasks outside tiny, NDCG@10)\n")
print("These runs cover a different task set from the tiny results and the two")
print("must never be averaged together.\n")
if rows:
    names = [r[0] for r in rows]
    all_tasks = sorted(rows[0][3])
    print("| task | " + " | ".join(names) + " |")
    print("| --- |" + " ---: |" * len(names))
    for t in all_tasks:
        print(f"| {t} | " + " | ".join(f"{r[3].get(t, float('nan')):.2f}" for r in rows) + " |")
    print("| **mean** | " + " | ".join(f"**{r[1]:.2f}**" for r in rows) + " |")
    if len(rows) == 2:
        a, b = rows[0], rows[1]
        d = [b[3][t] - a[3][t] for t in all_tasks if t in b[3] and t in a[3]]
        if d:
            import statistics
            mean = sum(d) / len(d)
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            print(f"\n{b[0]} minus {a[0]}: mean {mean:+.2f}, SD {sd:.2f}, "
                  f"improved on {sum(1 for x in d if x > 0)}/{len(d)} tasks")
PYEOF

log "=== sunday queue finished ==="
