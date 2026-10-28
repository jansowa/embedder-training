#!/usr/bin/env bash
# Student-mined negatives for the gated FiQA-PL rows (28.10): a SPLADE model proposes the
# candidates (scripts/fiqa_splade_mine.py, top-100 per query by dot product), then the final
# rerank and the export of ~/work/speakleash/mine-hard-negatives (roberta-v3, the two
# percentile thresholds FINAL_BETA / FINAL_U_FLOOR on the ECDF of the positives' scores)
# keep only the safe ones. Variants (VARIANT=...):
#   own       corpus = the rows' own 11.6k passages, candidates from base polish-splade (default)
#   plus      corpus = own passages + the 57.6k PIRB fiqa-pl passages (clarin-knext translation)
#   own-f02   the own candidates, FINAL_U_FLOOR 0.02 instead of 0.005 (looser teacher floor)
#   own-full  the own candidates, roberta-v3 scores all top-100 of every query (fixed budget
#             100 instead of 20/+10/80): the input of scripts/build_fiqa_band_negatives.py
#   soup-own  own corpus, candidates from models/soup-v2b (hard for the current best model)
#   long-easy, short-easy  length-chosen, student-easy candidates prepared beforehand by
#             scripts/build_fiqa_length_negatives.py in the variant's mine directory
#             (the mining step is skipped because negative_candidates.parquet exists)
# MODEL overrides the SPLADE model directory. SMOKE=N: N queries and 2000 passages into
# datasets/arms/fiqa-splade-smoke/ (nothing under datasets/arms/*.jsonl is touched).
# Outputs: datasets/arms/fiqa-2073-splade-<variant>.jsonl and, in results/data/,
# fiqa-2073-splade-<variant>.json (finalizer report), -mining.json (the student's view),
# -teacher-filter.json (what roberta-v3 rejected among the student's top candidates).
# One job at a time: waits for an idle GPU and >= 6 GB RAM like scripts/queue_lib.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
MINE=/home/jsowa/work/speakleash/mine-hard-negatives
MPY="$MINE/.venv/bin/python"
PY="$REPO/.venv/bin/python"
VARIANT="${VARIANT:-own}"
SMOKE="${SMOKE:-0}"
SRC="$REPO/datasets/arms/fiqa-v3-mine"
MODEL="${MODEL:-}"
EXTRA=()
FLOOR=""
FULL=""
case "$VARIANT" in
    own) ;;
    plus) EXTRA=(--extra-corpus third_party/pirb/data/fiqa-pl/passages/passages.jsonl --extra-id-prefix pirb:) ;;
    own-f02) FLOOR=0.02 ;;
    own-full) FULL=1 ;;
    soup-own) [ -n "$MODEL" ] || MODEL=models/soup-v2b ;;
    long-easy|short-easy) ;;  # candidates prepared by scripts/build_fiqa_length_negatives.py; only the teacher pass + export run
    *) echo "unknown VARIANT=$VARIANT (own, plus, own-f02, own-full, soup-own, long-easy, short-easy)" >&2; exit 1 ;;
esac
[ -n "$MODEL" ] || MODEL=models/base-splade-wrapped
smoke_args=()
if [ "$SMOKE" -gt 0 ]; then
    export MINE_OUT="$REPO/datasets/arms/fiqa-splade-smoke"
    ARM="$MINE_OUT/fiqa-smoke-splade.jsonl"; REPORT="$MINE_OUT/report.json"; REPORT_STEM="$MINE_OUT/report"
    TAG=fiqa-splade-smoke; smoke_args=(--smoke "$SMOKE" --corpus-limit 2000)
else
    export MINE_OUT="$REPO/datasets/arms/fiqa-splade-$VARIANT-mine"
    ARM="$REPO/datasets/arms/fiqa-2073-splade-$VARIANT.jsonl"
    REPORT="$REPO/results/data/fiqa-2073-splade-$VARIANT.json"; REPORT_STEM="$REPO/results/data/fiqa-2073-splade-$VARIANT"
    TAG=fiqa-splade-$VARIANT
fi
mkdir -p "$MINE_OUT" logs results/data
log() { echo "[$(date +%H:%M:%S)] [$TAG] $*"; }
echo $$ > "logs/$TAG.pid"
[ -f "$SRC/relevant_with_score.parquet" ] || { log "missing $SRC/relevant_with_score.parquet (the v3 positive scores)"; exit 1; }
[ -d "$MODEL" ] || { log "model directory $MODEL missing"; exit 1; }

waited=0
while :; do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    avail=$(free -g | awk '/^Mem:/ {print $7}')
    [ "$busy" -eq 0 ] && [ "${avail:-0}" -ge 6 ] && break
    [ "$waited" -ge 7200 ] && { log "GPU busy ($busy procs) or RAM low (${avail} GB) for 2 h; giving up"; exit 1; }
    [ $((waited % 600)) -eq 0 ] && log "waiting: $busy GPU procs, ${avail} GB RAM available"
    sleep 60; waited=$((waited + 60))
done
log "=== start: variant $VARIANT, model $MODEL, $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GB free on /, out $MINE_OUT ==="
T0=$SECONDS
step() {  # step LABEL DONE_FILE CMD...: skip when DONE_FILE exists (re-runs continue where they stopped)
    local label="$1" done_file="$2"; shift 2
    [ -n "$done_file" ] && [ -e "$done_file" ] && { log "$label: already done ($done_file)"; return 0; }
    log "$label"; local s=$SECONDS
    timeout -k 60 10800 "$@" || { log "$label FAILED after $((SECONDS - s))s: $*"; exit 1; }
    log "$label done in $((SECONDS - s))s"
}

# own-f02 / own-full differ from own only in the teacher pass: reuse the own candidates when present
OWN_DIR="$REPO/datasets/arms/fiqa-splade-own-mine"
if { [ "$VARIANT" = own-f02 ] || [ "$VARIANT" = own-full ]; } && [ ! -f "$MINE_OUT/negative_candidates.parquet" ] && [ -f "$OWN_DIR/negative_candidates.parquet" ]; then
    for f in queries corpus relevant relevant_with_candidate_score relevant_with_score negative_candidates; do cp "$OWN_DIR/$f.parquet" "$MINE_OUT/"; done
    cp "$OWN_DIR/splade_mining_report.json" "$MINE_OUT/"
    log "reused the own candidates from $OWN_DIR"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step "mine candidates with $MODEL (top-100 by SPLADE)" "$MINE_OUT/negative_candidates.parquet" \
    "$PY" scripts/fiqa_splade_mine.py --source-dir "$SRC" --out-dir "$MINE_OUT" --model "$MODEL" --top-k 100 "${EXTRA[@]}" "${smoke_args[@]}"

set -a; source "$REPO/scripts/fiqa_v3_mine.env"; set +a
[ -n "$FLOOR" ] && { export FINAL_U_FLOOR="$FLOOR"; log "FINAL_U_FLOOR=$FLOOR"; }
[ -n "$FULL" ] && { export FINAL_RERANK_INITIAL_BUDGET=100 FINAL_RERANK_MAX_BUDGET=100; log "teacher scores all top-100 candidates (FINAL_RERANK_INITIAL_BUDGET=100)"; }
cd "$MINE" || exit 1
step "final rerank, roberta-v3 (adaptive, student order)" "$MINE_OUT/negatives.parquet" "$MPY" app_code/rerank_negative_candidates.py
step "export (no synthetic positives)" "$MINE_OUT/output.jsonl" "$MPY" app_code/create_flag_embedding_jsonl.py \
    --no_mine_positives --output_path "$MINE_OUT/output.jsonl" --report_path "$MINE_OUT/output.jsonl.report.json" \
    --negcount_sqlite_path "$MINE_OUT/negcount.sqlite"
cd "$REPO" || exit 1

rows=datasets/arms/fiqa-2073.jsonl
if [ "$SMOKE" -gt 0 ]; then
    head -n "$SMOKE" datasets/arms/fiqa-2073.jsonl > "$MINE_OUT/source_rows.jsonl"; rows="$MINE_OUT/source_rows.jsonl"
fi
log "finalize"
"$PY" scripts/fiqa_v3_finalize.py --mined "$MINE_OUT/output.jsonl" --rows "$rows" --out "$ARM" --report "$REPORT" > "logs/$TAG.finalize.log" 2>&1 \
    && log "wrote $ARM" || log "finalize: rows dropped or failed, see logs/$TAG.finalize.log"
cp "$MINE_OUT/splade_mining_report.json" "$REPORT_STEM-mining.json"
MINE_OUT="$MINE_OUT" OUT="$REPORT_STEM-teacher-filter.json" "$PY" - <<'EOF' && log "teacher filter: $(cat "$REPORT_STEM-teacher-filter.json" | tr -d '\n ' | cut -c1-300)"
import json, os
import pandas as pd
d = os.environ["MINE_OUT"]
cand = pd.read_parquet(f"{d}/negative_candidates.parquet", columns=["query_id", "document_id", "retrieval_rank"])
neg = pd.read_parquet(f"{d}/negatives.parquet", columns=["query_id", "document_id", "final_ranking", "final_selected", "final_rerank_budget"])
m = cand.merge(neg, on=["query_id", "document_id"], how="left")
m["scored"] = m["final_ranking"].notna()
def share(frame, col):
    return round(float(frame[col].fillna(False).astype(bool).mean()), 4) if len(frame) else None
rep = {"candidates": int(len(cand)), "scored_by_teacher": int(m["scored"].sum())}
for k in (10, 20):
    top = m[m["retrieval_rank"] <= k]
    sc = top[top["scored"]]
    rep[f"top{k}_scored_share"] = share(top, "scored")
    rep[f"top{k}_rejected_by_teacher_share"] = round(1 - float(sc["final_selected"].astype(bool).mean()), 4) if len(sc) else None
budget = neg.groupby("query_id")["final_rerank_budget"].max()
rep["queries"] = int(budget.size)
rep["final_budget_histogram"] = {str(int(k)): int(v) for k, v in budget.value_counts().sort_index().items()}
rep["teacher_score_of_rejected_top10"] = {k: round(float(v), 3) for k, v in m[(m["retrieval_rank"] <= 10) & m["scored"] & ~m["final_selected"].fillna(False).astype(bool)]["final_ranking"].quantile([0.1, 0.5, 0.9]).items()}
json.dump(rep, open(os.environ["OUT"], "w"), indent=2)
EOF
log "=== done in $((SECONDS - T0))s ==="
