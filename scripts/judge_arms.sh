#!/usr/bin/env bash
# Judge the positives of every arm file with the exact MS MARCO-pl audit prompt
# on the inference server, then build clean / rejected arms + configs.
#
#   JUDGE_BASE_URL=http://server:8000/v1 JUDGE_MODEL=<served model name> \
#   [JUDGE_API_KEY=...] [JUDGE_CONCURRENCY=8] [JUDGE_NEGATIVES=0] \
#   scripts/judge_arms.sh [arm ...]          # default: trivia-2073 squadv2-2073 fiqa-2073 wikisynth-2073 msmarco-clean-2073
#
# Safe to rerun: the journal under results/judge/ is the source of truth, only
# unjudged pairs are sent. JUDGE_NEGATIVES=K also judges the first K negatives
# per row (false-negative estimate; costs K times more calls).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="$(pwd)/.venv/bin/python"
: "${JUDGE_BASE_URL:?set JUDGE_BASE_URL, e.g. http://server:8000/v1}"
: "${JUDGE_MODEL:?set JUDGE_MODEL to the served model name}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-}"
CONC="${JUDGE_CONCURRENCY:-8}"
NEGS="${JUDGE_NEGATIVES:-0}"
ARMS=("$@"); [ "${#ARMS[@]}" -eq 0 ] && ARMS=(trivia-2073 squadv2-2073 fiqa-2073 wikisynth-2073 msmarco-clean-2073)
mkdir -p results/judge logs
log() { echo "[$(date +%H:%M:%S)] [judge] $*" | tee -a logs/judge_arms.log; }

# smoke call first: one pair, so a wrong URL / model name fails in seconds
log "smoke test against $JUDGE_BASE_URL ($JUDGE_MODEL)"
"$PY" scripts/judge_pairs.py --input "datasets/arms/${ARMS[0]}.jsonl" --output /tmp/judge_smoke.jsonl \
    --journal results/judge/smoke.journal.jsonl --base-url "$JUDGE_BASE_URL" --model "$JUDGE_MODEL" \
    --concurrency 1 --limit 1 2>&1 | tail -n 2 | tee -a logs/judge_arms.log
grep -q '"verdict"' results/judge/smoke.journal.jsonl 2>/dev/null || { log "smoke test produced no verdict; check the server"; exit 1; }

judged=()
for arm in "${ARMS[@]}"; do
    [ -f "datasets/arms/$arm.jsonl" ] || { log "$arm: datasets/arms/$arm.jsonl missing"; continue; }
    log "judging $arm (concurrency $CONC, negatives $NEGS)"
    s=$SECONDS
    "$PY" scripts/judge_pairs.py --input "datasets/arms/$arm.jsonl" --output "results/judge/$arm.judged.jsonl" \
        --journal "results/judge/$arm.journal.jsonl" --base-url "$JUDGE_BASE_URL" --model "$JUDGE_MODEL" \
        --concurrency "$CONC" --negatives "$NEGS" > "logs/judge_$arm.log" 2>&1
    log "$arm: $(tail -n 1 "logs/judge_$arm.log") [$((SECONDS - s))s]"
    judged+=("results/judge/$arm.judged.jsonl")
done
[ "${#judged[@]}" -gt 0 ] && "$PY" scripts/build_clean_arms.py --judged "${judged[@]}" 2>&1 | tee -a logs/judge_arms.log
log "done; arms under datasets/arms/judged, configs under configs/weekend/judged, reports under results/judge"
