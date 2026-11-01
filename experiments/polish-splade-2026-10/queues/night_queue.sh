#!/usr/bin/env bash
# Second shift: waits for scripts/weekend_queue.sh to finish, then runs the
# long-context counterparts of its three fine-tunes plus the two remaining
# soup alphas.
#
# Standalone on purpose: editing or sourcing the running script could disturb
# it, because bash reads a script incrementally while executing it.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$ROOT/.venv/bin/python"
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi
LOGS="$ROOT/logs"
STATE="$LOGS/queue_state.tsv"
SUMMARY="$LOGS/queue_summary.md"
mkdir -p "$LOGS"

BASE_AVG=60.14
RUN_ROOT="$ROOT/runs/weekend-3060-long/sentence-transformers/splade"
# Baseline count of the first queue's completion marker, taken before waiting.
SENTINELS_AT_START="${1:-$(grep -c '=== queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)}"

log()    { echo "[$(date +%H:%M:%S)] [night] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }

# --- wait for the day queue ------------------------------------------------

log "waiting for the weekend queue to finish (marker count at start: $SENTINELS_AT_START)"
missing=0
while true; do
    now=$(grep -c '=== queue finished ===' "$LOGS/queue.log" 2>/dev/null || echo 0)
    [ "$now" -gt "$SENTINELS_AT_START" ] && { log "weekend queue reported finished"; break; }
    # Fallback: the queue could die without printing its marker. This pattern
    # cannot match this script, whose own command line names night_queue.sh.
    if pgrep -f "bash scripts/weekend_queue.sh" >/dev/null; then
        missing=0
    else
        missing=$((missing + 1))
        [ "$missing" -ge 3 ] && { log "weekend queue process gone; continuing"; break; }
    fi
    sleep 60
done

# --- helpers ---------------------------------------------------------------

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

hit_oom() { grep -qiE "CUDA out of memory|OutOfMemoryError" "$1"; }

train() {
    local tag="$1" cfg="$2" results="$3"
    local logfile="$LOGS/${tag}.log"
    log "train $tag with $cfg (benchmark -> results/$results)"
    local s=$SECONDS
    "$PY" run_experiments.py --grid-configuration-file "$cfg" \
        --backend sentence-transformers --training-type splade \
        --run_pirb --pirb_scope tiny --remove-pirb-indexes \
        --benchmark-output-dir "results/$results" > "$logfile" 2>&1
    local rc=$? e=$((SECONDS - s))
    if hit_oom "$logfile"; then
        log "train $tag: CUDA OOM after ${e}s"; record "train:$tag" OOM "-" "${e}s"; return 2
    fi
    if [ $rc -ne 0 ]; then
        log "train $tag FAILED rc=$rc after ${e}s"; record "train:$tag" FAILED "rc=$rc" "${e}s"; return 1
    fi
    log "train $tag ok in ${e}s"; record "train:$tag" ok "-" "${e}s"
    return 0
}

latest_final() { ls -1dt "$RUN_ROOT"/*"$1"*/final 2>/dev/null | head -1; }
run_metrics()  { ls -1 "results/$1"/*/metrics.json 2>/dev/null | head -1; }

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

# --- phase A: close the soup alpha curve -----------------------------------

log "=== phase A: soup alphas 0.05 and 0.10 ==="
for alpha in 0.05 0.10; do
    merge "soup-a${alpha}" sdadas/polish-splade models/mlm-msmarco-ep1 "$alpha" \
        && bench "soup-a${alpha}" "models/soup-a${alpha}"
done

# --- phase B: long-context fine-tunes --------------------------------------
#
# Same 8192-token budget as their 256-token counterparts, spent on full-length
# passages. Each is followed by WiSE-FT at the two alphas that worked best on
# the soup curve.

log "=== phase B: long-context fiqa (lr 5e-5) ==="
train r1_fiqa_long configs/weekend/ft_fiqa_3060_long.yaml ft-fiqa-long
if [ $? -eq 0 ]; then
    D="$(latest_final lr-5e-5-ep-2)"
    record "r1_long:final_dir" ok "$D"
    [ -n "$(run_metrics ft-fiqa-long)" ] || bench_checkpoint_copy ft-fiqa-long "$D"
    [ -n "$D" ] && wiseft fiqa-long "$D" 0.15 0.3
fi

log "=== phase C: long-context fiqa with warmup (lr 2e-5) ==="
train r2_fiqa_warmup_long configs/weekend/ft_fiqa_3060_warmup_long.yaml ft-fiqa-warmup-long
if [ $? -eq 0 ]; then
    D="$(latest_final lr-2e-5-ep-2)"
    record "r2_long:final_dir" ok "$D"
    [ -n "$(run_metrics ft-fiqa-warmup-long)" ] || bench_checkpoint_copy ft-fiqa-warmup-long "$D"
    [ -n "$D" ] && wiseft fiqa-warmup-long "$D" 0.15 0.3
fi

log "=== phase D: long-context rehearsal grid (lr 2e-5) ==="
train n1_grid_long configs/weekend/ft_grid_3060_lr2e-5_long.yaml ft-3060-2e-5-long
if [ $? -eq 0 ]; then
    D="$(latest_final lr-2e-5-ep-1)"
    R="$(dirname "$D")"
    record "n1_long:final_dir" ok "$D"
    [ -d "$R/frac-0.80" ] && bench ft3060-2e-5-long-frac080 "$R/frac-0.80"
    [ -n "$(run_metrics ft-3060-2e-5-long)" ] || bench_checkpoint_copy ft-3060-2e-5-long "$D"
    [ -n "$D" ] && wiseft 3060-long "$D" 0.15 0.3
fi

# --- summary ---------------------------------------------------------------

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
    rows.append((path.split("/")[1], avg,
                 sum(no_arg) / len(no_arg) if no_arg else float("nan"),
                 m.get("epoch0/pirb_datasets")))
rows.sort(key=lambda r: -r[1])
print("# Weekend queue results (PIRB tiny, NDCG@10)\n")
print(f"Reference B = {base:.2f}\n")
print("| run | mean | without arguana | tasks | delta vs B |")
print("| --- | ---: | ---: | ---: | ---: |")
for name, avg, noarg, n in rows:
    print(f"| {name} | {avg:.2f} | {noarg:.2f} | {n} | {avg - base:+.2f} |")
PYEOF

log "=== night queue finished ==="
