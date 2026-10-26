#!/usr/bin/env bash
# Does the LLM judge earn its keep?
#
# Three arms, identical in every respect except which MS MARCO pairs they see.
# All three are pre-gated on pos_scores > 23.50 and re-checked by the same
# filter profile, so the verdict is the only variable:
#
#   plain     natural mix, i.e. what the current recipe trains on
#   clean     only pairs the judge passed on all four checks
#   rejected  only pairs the judge flagged
#
# plain vs clean answers whether filtering by the judge helps. rejected is the
# control: if it trains just as well, the judge is not separating anything.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$(pwd)/.venv/bin/python"
LOGS="$(pwd)/logs"
STATE="$LOGS/queue_state.tsv"
mkdir -p "$LOGS"
SENTINELS_AT_START="${1:-$(grep -c '=== night2 queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)}"

log()    { echo "[$(date +%H:%M:%S)] [judge] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

log "waiting for the night2 queue (marker count at start: $SENTINELS_AT_START)"
missing=0
while true; do
    now=$(grep -c '=== night2 queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)
    [ "$now" -gt "$SENTINELS_AT_START" ] && { log "night2 reported finished"; break; }
    if pgrep -f "bash scripts/night2_queue.sh" >/dev/null; then
        missing=0
    else
        missing=$((missing + 1))
        [ "$missing" -ge 3 ] && { log "night2 process gone; continuing"; break; }
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

for arm in plain clean rejected; do
    log "=== arm: $arm ==="
    logfile="$LOGS/judge_${arm}.log"
    s=$SECONDS
    "$PY" run_experiments.py --grid-configuration-file "configs/weekend/ft_msmarco_judged_${arm}.yaml" \
        --backend sentence-transformers --training-type splade \
        --run_pirb --pirb_scope tiny --remove-pirb-indexes \
        --benchmark-output-dir "results/judge-${arm}" > "$logfile" 2>&1
    e=$((SECONDS - s))
    # Training succeeds and the benchmark it runs next fails for lack of VRAM,
    # so the exit code says nothing; what counts is whether final/ was written.
    ck=$(ls -1dt "runs/judged-${arm}/sentence-transformers/splade"/*/final 2>/dev/null | head -1)
    if [ -z "$ck" ]; then
        log "arm $arm produced no checkpoint after ${e}s"; record "train:judge-$arm" FAILED "-" "${e}s"
        continue
    fi
    log "arm $arm trained in ${e}s -> $ck"; record "train:judge-$arm" ok "-" "${e}s"
    rm -rf "models/judge-${arm}"
    cp -r "$ck" "models/judge-${arm}" || continue
    bench "judge-${arm}" "models/judge-${arm}"
done

log "=== judge queue finished ==="
