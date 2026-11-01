#!/usr/bin/env bash
# Fourth shift: benchmark the checkpoints the earlier queues trained but could
# not measure.
#
# Training succeeds; the benchmark the training config runs straight afterwards
# does not, because PIRB starts encoding while the trainer still holds the card
# (third_party/pirb/search/splade.py:125 asking for 1.53 GiB with 315 MiB free).
# The earlier queues read that non-zero exit as "training failed" and skipped
# both the benchmark and the WiSE-FT merges, even though <run>/final was
# written and is a complete SparseEncoder. A standalone benchmark has the card
# to itself and works, so this picks the checkpoints back up.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/logs"
STATE="$LOGS/queue_state.tsv"
mkdir -p "$LOGS"

# The alpha the 7-point soup curve put the optimum at: +0.25 with a bootstrap
# CI of [+0.04, +0.51] and 8 of 10 tasks improved.
WISEFT_ALPHA=0.15
SENTINELS_AT_START="${1:-$(grep -c '=== sunday queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)}"

log()    { echo "[$(date +%H:%M:%S)] [rescue] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

log "waiting for the sunday queue (marker count at start: $SENTINELS_AT_START)"
missing=0
while true; do
    now=$(grep -c '=== sunday queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)
    [ "$now" -gt "$SENTINELS_AT_START" ] && { log "sunday queue reported finished"; break; }
    if pgrep -f "bash scripts/sunday_queue.sh" >/dev/null; then
        missing=0
    else
        missing=$((missing + 1))
        [ "$missing" -ge 3 ] && { log "sunday queue process gone; continuing"; break; }
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

# A short, stable label per checkpoint: the long run directory name would blow
# past the filename limit PIRB hits when it builds its cache key.
label_for() {
    local dir="$1" ctx="s"
    [[ "$dir" == *weekend-3060-long* ]] && ctx="l"
    case "$dir" in
        *lr-5e-5-ep-2*) echo "ck-fiqa-${ctx}" ;;
        *lr-2e-5-ep-1*) echo "ck-grid-${ctx}" ;;
        *lr-2e-5-ep-2*) echo "ck-warmup-${ctx}" ;;
        *)              echo "ck-other-${ctx}-$(basename "$(dirname "$dir")" | tail -c 8)" ;;
    esac
}

mapfile -t CKPTS < <(ls -1d runs/weekend-3060/sentence-transformers/splade/*/final \
                            runs/weekend-3060-long/sentence-transformers/splade/*/final 2>/dev/null)
log "checkpoints to rescue: ${#CKPTS[@]}"
record "rescue:checkpoints" ok "${#CKPTS[@]}"

for ckpt in "${CKPTS[@]}"; do
    name="$(label_for "$ckpt")"
    log "rescuing $ckpt as $name"
    if [ ! -d "models/$name" ]; then
        cp -r "$ckpt" "models/$name" || { record "copy:$name" FAILED "-"; continue; }
    fi
    bench "$name" "models/$name"

    wname="wiseft-${name}-a${WISEFT_ALPHA}"
    if [ ! -d "models/$wname" ]; then
        "$PY" scripts/merge_splade_weights.py --models sdadas/polish-splade "models/$name" \
            --alpha "$WISEFT_ALPHA" --output "models/$wname" \
            > "$LOGS/merge_$wname.log" 2>&1 \
            || { record "merge:$wname" FAILED "see logs/merge_$wname.log"; continue; }
    fi
    bench "$wname" "models/$wname"
done

log "=== rescue queue finished ==="
