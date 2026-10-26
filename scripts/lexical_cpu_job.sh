#!/usr/bin/env bash
# CPU-only lemma job (Stanza, doc2query venv). Runs unattended, nice'd, in order:
#   1. wait for the calibration run (results/lexical_difficulty_calibration.md) or 45 min
#   2. metrics for the verified synthetic set  -> datasets/synth/lexical_metrics.jsonl
#      (re-run later for rows verified after the first pass)
#   3. metrics for fiqa-2073 -> results/lexical_difficulty/full/fiqa-2073.jsonl
#      and the FiQA band (p10..p90 of idf_coverage) -> results/lexical_band.json
#   4. squadv2-2073, msmarco-clean-2073 (for the HF upload files)
# The band is what scripts/synth_finalize.sh applies before mining.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=""
DPY=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/doc2query/.venv/bin/python
PY="$(pwd)/.venv/bin/python"
OUT=results/lexical_difficulty/full
CACHE=results/lexical_difficulty/lemma_cache.jsonl
LOG=logs/lexical_cpu_job.log
mkdir -p "$OUT" datasets/synth
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# The machine has 16 GB RAM and one Stanza pipeline takes ~4 GB; never run two,
# and never start one when the PIRB benchmark + JVM leave too little headroom.
wait_ram() {
    while true; do
        other=$(ps -eo pid,args | awk -v me="$$" '/lexical_difficulty\.py/ && $1 != me {c++} END {print c+0}')
        avail=$(free -g | awk '/^Mem:/ {print $7}')
        if [ "$other" -eq 0 ] && [ "${avail:-0}" -ge 5 ]; then return 0; fi
        log "waiting: other lexical jobs=$other, available RAM=${avail}GB"
        sleep 120
    done
}
wait_ram
log "calibration present: $([ -f results/lexical_difficulty_calibration.md ] && echo yes || echo no), starting"

# shared IDF pool: positives of the three natural arms + 3000 synthetic chunks
"$PY" - <<'PYEOF'
import json, itertools
out = open("datasets/synth/idf_pool.jsonl", "w", encoding="utf-8")
n = 0
for p in ("datasets/arms/fiqa-2073.jsonl", "datasets/arms/squadv2-2073.jsonl", "datasets/arms/msmarco-clean-2073.jsonl"):
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); out.write(json.dumps({"text": r["pos"][0]}, ensure_ascii=False) + "\n"); n += 1
for l in itertools.islice(open("datasets/synth/chunks_pool.jsonl", encoding="utf-8"), 3000):
    r = json.loads(l); out.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n"); n += 1
print("idf pool rows", n)
PYEOF

synth_pass() {
    # verified.jsonl -> query/pos layout with chunk_id as query_id
    "$PY" - <<'PYEOF'
import json
n = 0
with open("datasets/synth/for_lexical.jsonl", "w", encoding="utf-8") as out:
    for l in open("datasets/synth/verified.jsonl", encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l)
        if not r.get("clean"): continue
        out.write(json.dumps({"query_id": r["chunk_id"], "query": r["query"], "pos": [r["text"]]}, ensure_ascii=False) + "\n"); n += 1
print("synthetic clean rows for lexical:", n)
PYEOF
    wait_ram; nice -n 10 "$DPY" scripts/lexical_difficulty.py --input synth=datasets/synth/for_lexical.jsonl \
        --output datasets/synth/lexical_metrics.jsonl --idf-pool datasets/synth/idf_pool.jsonl \
        --cache "$CACHE" --threads 2 --batch-size 16 >> "$LOG" 2>&1
    "$PY" - <<'PYEOF'
import json
# finaliser looks rows up by chunk_id; the script writes query_id
rows = [json.loads(l) for l in open("datasets/synth/lexical_metrics.jsonl", encoding="utf-8") if l.strip()]
with open("datasets/synth/lexical_metrics.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        r["chunk_id"] = r.get("query_id"); fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("synthetic lexical metrics rows:", len(rows))
PYEOF
}

log "pass 1: synthetic set"
synth_pass

log "fiqa-2073"
wait_ram; nice -n 10 "$DPY" scripts/lexical_difficulty.py --input fiqa=datasets/arms/fiqa-2073.jsonl \
    --output "$OUT/fiqa-2073.jsonl" --idf-pool datasets/synth/idf_pool.jsonl --cache "$CACHE" --threads 2 >> "$LOG" 2>&1
"$PY" - <<'PYEOF'
import json
vals = sorted(json.loads(l)["idf_coverage"] for l in open("results/lexical_difficulty/full/fiqa-2073.jsonl", encoding="utf-8") if l.strip())
lo, hi = vals[int(0.10 * len(vals))], vals[int(0.90 * len(vals)) - 1]
json.dump({"metric": "idf_coverage", "lo": lo, "hi": hi, "source": "fiqa-2073 p10..p90", "n": len(vals)},
          open("results/lexical_band.json", "w"), indent=2)
print(f"FiQA band idf_coverage [{lo:.3f}, {hi:.3f}] over {len(vals)} rows")
PYEOF

log "pass 2: synthetic set (rows verified since pass 1)"
synth_pass

for arm in squadv2-2073 msmarco-clean-2073; do
    log "$arm"
    wait_ram; nice -n 10 "$DPY" scripts/lexical_difficulty.py --input "$arm=datasets/arms/$arm.jsonl" \
        --output "$OUT/$arm.jsonl" --idf-pool datasets/synth/idf_pool.jsonl --cache "$CACHE" --threads 2 >> "$LOG" 2>&1
done
log "lexical CPU job finished"
