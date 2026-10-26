#!/usr/bin/env bash
# Mine hard negatives for the verified synthetic PolQA pairs, then build the two
# synthetic training arms. Needs the GPU; scripts/today_queue.sh calls it once
# datasets/synth/READY_FOR_MINING exists.
#
# Inputs : datasets/synth/pairs_final.jsonl  (chunk_id, query)   - verified + band-filtered
#          datasets/synth/chunks_pool.jsonl   (chunk_id, text)    - corpus for mining
# Outputs: datasets/arms/wikisynth-2073.jsonl, datasets/arms/fiqa-plus-wikisynth.jsonl
#          results/synth/pair_loss.json (base-model difficulty of synth vs FiQA)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
MINE=/home/jsowa/work/speakleash/mine-hard-negatives
MPY="$MINE/.venv/bin/python"
PY="$REPO/.venv/bin/python"
OUT="$REPO/datasets/synth/mine"
mkdir -p "$OUT" results/synth

# 16 GB machine: do not start the embedder/reranker stack while a Stanza job
# or anything else leaves less than 5 GB available.
while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt 5 ]; do
    echo "[$(date +%H:%M:%S)] waiting for RAM (available $(free -g | awk '/^Mem:/ {print $7}') GB)"; sleep 120
done
echo "[$(date +%H:%M:%S)] building mining inputs"
"$PY" scripts/synth_build_mining_inputs.py --pairs datasets/synth/pairs_final.jsonl \
    --pool datasets/synth/chunks_pool.jsonl --out-dir "$OUT" || exit 1

set -a; source "$REPO/scripts/synth_mine.env"; set +a
cd "$MINE" || exit 1
step() { echo "[$(date +%H:%M:%S)] $1"; shift; "$@" || { echo "FAILED: $*"; exit 1; }; }
step "ingest corpus"            "$MPY" app_code/add_documents_to_db.py
step "score positives (cand.)"  "$MPY" app_code/add_positives_ranks.py
step "mine candidates"          "$MPY" app_code/find_negatives.py
step "score positives (final)"  "$MPY" app_code/add_positives_ranks.py --final-step
step "final rerank"             "$MPY" app_code/rerank_negative_candidates.py
step "export jsonl"             "$MPY" app_code/create_flag_embedding_jsonl.py
cd "$REPO" || exit 1

echo "[$(date +%H:%M:%S)] building arms"
"$PY" - <<'PYEOF'
import json, random
from pathlib import Path
random.seed(42)
rows = [json.loads(l) for l in open("datasets/synth/mine/output.jsonl", encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("query") and r.get("pos") and len(r.get("neg", [])) >= 6]
print("mined rows usable:", len(rows))
random.shuffle(rows)
synth = rows[:2073]
for i, r in enumerate(synth):
    r["query_id"] = f"wikisynth-{i:05d}"
    r["type"] = "retrieval"; r["prompt"] = ""
Path("datasets/arms").mkdir(exist_ok=True)
with open("datasets/arms/wikisynth-2073.jsonl", "w", encoding="utf-8") as fh:
    for r in synth: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
fiqa = [l for l in open("datasets/arms/fiqa-2073.jsonl", encoding="utf-8") if l.strip()]
mix = fiqa + [json.dumps(r, ensure_ascii=False) + "\n" for r in synth]
random.shuffle(mix)
with open("datasets/arms/fiqa-plus-wikisynth.jsonl", "w", encoding="utf-8") as fh:
    for l in mix: fh.write(l if l.endswith("\n") else l + "\n")
print("wikisynth rows:", len(synth), "mix rows:", len(mix))
PYEOF

echo "[$(date +%H:%M:%S)] base-model difficulty check (synth vs fiqa vs squad)"
"$PY" scripts/splade_pair_loss.py --model models/base-splade-wrapped \
    --input datasets/arms/wikisynth-2073.jsonl --input datasets/arms/fiqa-2073.jsonl \
    --input datasets/arms/squadv2-2073.jsonl --input datasets/arms/msmarco-clean-2073.jsonl --input datasets/arms/trivia-2073.jsonl \
    --n 400 --output-json results/synth/pair_loss.json
echo "[$(date +%H:%M:%S)] synth_mine done"
