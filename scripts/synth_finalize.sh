#!/usr/bin/env bash
# Autonomous finaliser for the synthetic set. Every few minutes it merges the
# generation and verification batches that have landed so far; once enough
# verified-clean pairs exist it writes datasets/synth/pairs_final.jsonl and
# the marker that scripts/today_queue.sh waits for, then exits.
#
#   TARGET   clean pairs wanted before finalising (default 2073)
#   DEADLINE HH:MM after which it finalises with whatever is >= MIN_ROWS
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="$(pwd)/.venv/bin/python"
TARGET="${TARGET:-2073}"
MIN_ROWS="${MIN_ROWS:-1500}"
DEADLINE="${DEADLINE:-12:30}"
LOG=logs/synth_finalize.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

while true; do
    "$PY" scripts/synth_validate_gen.py > /dev/null 2>&1
    "$PY" scripts/synth_verify_io.py split > /dev/null 2>&1
    "$PY" scripts/synth_verify_io.py merge > /tmp/synth_merge.out 2>&1
    clean=$("$PY" - <<'PYEOF'
import json
n = 0
try:
    for l in open("datasets/synth/verified.jsonl", encoding="utf-8"):
        if l.strip() and json.loads(l).get("clean"):
            n += 1
except FileNotFoundError:
    pass
print(n)
PYEOF
)
    now=$(date +%H:%M)
    log "clean verified pairs: $clean (target $TARGET, deadline $DEADLINE)"
    # The lexical band is NOT applied: the calibration (results/lexical_difficulty_calibration.md)
    # showed idf_coverage does not separate FiQA (works) from trivia (collapses), AUC 0.54,
    # so it is not a validated gate. The verification verdicts and the base-model loss
    # check in synth_mine.sh are the gates. Set APPLY_BAND=1 to override.
    if [ "$clean" -ge "$TARGET" ] || { [[ "$now" > "$DEADLINE" ]] && [ "$clean" -ge "$MIN_ROWS" ]; }; then
        "$PY" - <<'PYEOF'
import json, os, random
from pathlib import Path
rows = [json.loads(l) for l in open("datasets/synth/verified.jsonl", encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("clean")]
band = None
if os.environ.get("APPLY_BAND") == "1" and os.path.exists("results/lexical_band.json"):
    band = json.load(open("results/lexical_band.json"))
kept = rows
if band and os.path.exists("datasets/synth/lexical_metrics.jsonl"):
    metric, lo, hi = band["metric"], band["lo"], band["hi"]
    m = {}
    for l in open("datasets/synth/lexical_metrics.jsonl", encoding="utf-8"):
        if l.strip():
            d = json.loads(l); m[d.get("chunk_id") or d.get("id")] = d
    kept = [r for r in rows if r["chunk_id"] in m and lo <= m[r["chunk_id"]].get(metric, 0) <= hi]
    print(f"band {metric} [{lo},{hi}] kept {len(kept)} of {len(rows)}")
    if len(kept) < 1500:
        print("band too strict for the dose; falling back to all clean rows")
        kept = rows
else:
    print("no band applied (no results/lexical_band.json or metrics); using all clean rows")
random.seed(42); random.shuffle(kept)
with open("datasets/synth/pairs_final.jsonl", "w", encoding="utf-8") as fh:
    for r in kept:
        fh.write(json.dumps({"chunk_id": r["chunk_id"], "query": r["query"], "form": r.get("form", ""),
                             "title": r.get("title", ""), "text": r["text"]}, ensure_ascii=False) + "\n")
print("pairs_final rows:", len(kept))
PYEOF
        touch datasets/synth/READY_FOR_MINING
        log "finalised: datasets/synth/pairs_final.jsonl written, marker set"
        exit 0
    fi
    sleep 180
done
