#!/usr/bin/env bash
# Step A of docs/PLAN-2026-10-27.md: re-mine and re-score the gated FiQA-PL rows
# (datasets/arms/fiqa-2073.jsonl, 2073 queries) with sdadas/polish-reranker-roberta-v3
# as the final scorer, in ~/work/speakleash/mine-hard-negatives. The corpus is our
# own translation only (every positive and negative of those rows, 11.6k passages);
# the PIRB fiqa-pl corpus is a weaker translation and is not used.
#
# Outputs: datasets/arms/fiqa-2073-v3.jsonl       (export without synthetic positives)
#          datasets/arms/fiqa-2073-v3-plus.jsonl  (export with MINE_POSITIVES)
#          results/data/fiqa-2073-v3*.json        (small reports, versioned)
# Working files: datasets/arms/fiqa-v3-mine/ (LanceDB, parquet, sqlite; ignored).
#
# One job at a time on this machine: the script waits for an idle GPU and >= 6 GB
# of free RAM before starting, exactly like scripts/queue_lib.sh. Start with:
#   setsid nohup bash scripts/fiqa_v3_mine.sh > logs/fiqa_v3_mine.out 2>&1 &
# THEN_QUEUE=scripts/queue_v11.sh starts that queue when both exports are done.
# SMOKE=30 runs the whole chain on 30 queries and a 2k-passage corpus into
# datasets/arms/fiqa-v3-mine-smoke/ (nothing under datasets/arms/*.jsonl is touched).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
MINE=/home/jsowa/work/speakleash/mine-hard-negatives
MPY="$MINE/.venv/bin/python"
PY="$REPO/.venv/bin/python"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" -gt 0 ]; then
    export MINE_OUT="$REPO/datasets/arms/fiqa-v3-mine-smoke"
    ARM="$MINE_OUT/fiqa-smoke-v3.jsonl"; ARM_PLUS="$MINE_OUT/fiqa-smoke-v3-plus.jsonl"
    REPORT="$MINE_OUT/report-v3.json"; REPORT_PLUS="$MINE_OUT/report-v3-plus.json"
    TAG=fiqa-v3-smoke
elif [ "${VARIANT:-}" = hard ]; then
    # hard variant (27.10 night): the small reranker lets anything below the median of its
    # positive scores through, v3 keeps negatives below the 5th percentile of its positives
    export MINE_OUT="$REPO/datasets/arms/fiqa-v3hard-mine"
    ARM="$REPO/datasets/arms/fiqa-2073-v3hard.jsonl"; ARM_PLUS="$REPO/datasets/arms/fiqa-2073-v3hard-plus.jsonl"
    REPORT="$REPO/results/data/fiqa-2073-v3hard.json"; REPORT_PLUS="$REPO/results/data/fiqa-2073-v3hard-plus.json"
    TAG=fiqa-v3hard
else
    export MINE_OUT="$REPO/datasets/arms/fiqa-v3-mine"
    ARM="$REPO/datasets/arms/fiqa-2073-v3.jsonl"; ARM_PLUS="$REPO/datasets/arms/fiqa-2073-v3-plus.jsonl"
    REPORT="$REPO/results/data/fiqa-2073-v3.json"; REPORT_PLUS="$REPO/results/data/fiqa-2073-v3-plus.json"
    TAG=fiqa-v3
fi
mkdir -p "$MINE_OUT" logs results/data
log() { echo "[$(date +%H:%M:%S)] [$TAG] $*"; }
echo $$ > "logs/$TAG.pid"

# the machine rules: nothing starts next to another GPU job or with < 6 GB RAM free
waited=0
while :; do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    avail=$(free -g | awk '/^Mem:/ {print $7}')
    [ "$busy" -eq 0 ] && [ "${avail:-0}" -ge 6 ] && break
    [ "$waited" -ge 7200 ] && { log "GPU busy ($busy procs) or RAM low (${avail} GB) for 2 h; giving up"; exit 1; }
    [ $((waited % 600)) -eq 0 ] && log "waiting: $busy GPU procs, ${avail} GB RAM available"
    sleep 60; waited=$((waited + 60))
done

log "=== start: $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GB free on /, ${avail} GB RAM, out $MINE_OUT ==="
build_args=(--rows datasets/arms/fiqa-2073.jsonl --out-dir "$MINE_OUT")
[ "$SMOKE" -gt 0 ] && build_args+=(--smoke "$SMOKE")
"$PY" scripts/fiqa_v3_build_mining_inputs.py "${build_args[@]}" || { log "building inputs FAILED"; exit 1; }

set -a; source "$REPO/scripts/fiqa_v3_mine.env"; set +a
if [ "${VARIANT:-}" = hard ]; then export CANDIDATE_U_FLOOR=0.5 FINAL_U_FLOOR=0.05; log "hard variant: CANDIDATE_U_FLOOR=0.5 FINAL_U_FLOOR=0.05"; fi
cd "$MINE" || exit 1
T0=$SECONDS
step() {  # step LABEL DONE_FILE CMD...: skip when DONE_FILE exists (re-runs continue where they stopped)
    local label="$1" done_file="$2"; shift 2
    [ -n "$done_file" ] && [ -e "$done_file" ] && { log "$label: already done ($done_file)"; return 0; }
    log "$label"; local s=$SECONDS
    timeout -k 60 10800 "$@" || { log "$label FAILED after $((SECONDS - s))s: $*"; exit 1; }
    log "$label done in $((SECONDS - s))s"
}
step "ingest corpus (PolDense-150M -> LanceDB)" "" "$MPY" app_code/add_documents_to_db.py
step "score positives, candidate reranker" "$MINE_OUT/relevant_with_candidate_score.parquet" "$MPY" app_code/add_positives_ranks.py
step "mine candidates (hybrid + candidate reranker)" "$MINE_OUT/negative_candidates.parquet" "$MPY" app_code/find_negatives.py
step "score positives, roberta-v3" "$MINE_OUT/relevant_with_score.parquet" "$MPY" app_code/add_positives_ranks.py --final-step \
    --relevant_path "$MINE_OUT/relevant_with_candidate_score.parquet"
step "final rerank, roberta-v3 (adaptive)" "$MINE_OUT/negatives.parquet" "$MPY" app_code/rerank_negative_candidates.py
step "export v3 (no synthetic positives)" "$MINE_OUT/output.jsonl" "$MPY" app_code/create_flag_embedding_jsonl.py \
    --no_mine_positives --output_path "$MINE_OUT/output.jsonl" --report_path "$MINE_OUT/output.jsonl.report.json" \
    --negcount_sqlite_path "$MINE_OUT/negcount.sqlite"
step "export v3-plus (MINE_POSITIVES)" "$MINE_OUT/output-plus.jsonl" "$MPY" app_code/create_flag_embedding_jsonl.py \
    --mine_positives --output_path "$MINE_OUT/output-plus.jsonl" --report_path "$MINE_OUT/output-plus.jsonl.report.json" \
    --negcount_sqlite_path "$MINE_OUT/negcount-plus.sqlite"
cd "$REPO" || exit 1

rows=datasets/arms/fiqa-2073.jsonl
if [ "$SMOKE" -gt 0 ]; then  # the finalizer compares against the source rows; restrict them to the smoke window
    head -n "$SMOKE" datasets/arms/fiqa-2073.jsonl > "$MINE_OUT/source_rows.jsonl"; rows="$MINE_OUT/source_rows.jsonl"
fi
log "finalize v3"
"$PY" scripts/fiqa_v3_finalize.py --mined "$MINE_OUT/output.jsonl" --rows "$rows" --out "$ARM" --report "$REPORT" > "logs/$TAG.finalize.log" 2>&1 \
    && log "wrote $ARM" || log "finalize v3: rows dropped or failed, see logs/$TAG.finalize.log"
log "finalize v3-plus"
"$PY" scripts/fiqa_v3_finalize.py --mined "$MINE_OUT/output-plus.jsonl" --rows "$rows" --out "$ARM_PLUS" --report "$REPORT_PLUS" > "logs/$TAG.finalize-plus.log" 2>&1 \
    && log "wrote $ARM_PLUS" || log "finalize v3-plus: rows dropped or failed, see logs/$TAG.finalize-plus.log"
log "=== done in $((SECONDS - T0))s ==="

if [ -n "${THEN_QUEUE:-}" ] && [ "$SMOKE" -eq 0 ] && [ -f "$ARM" ]; then
    name=$(basename "$THEN_QUEUE" .sh)
    log "starting $THEN_QUEUE (log logs/$name.out)"
    setsid nohup bash "$THEN_QUEUE" > "logs/$name.out" 2>&1 < /dev/null &
fi
