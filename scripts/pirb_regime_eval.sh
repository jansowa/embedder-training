#!/usr/bin/env bash
# Evaluate one SPLADE model on one PIRB task set in one evaluation "regime":
#
#   old-pyserini  our fork + pyserini 0.21.0 (query weights reach Anserini as
#                 floats, the regime of the 2023 leaderboard)      venv-pyserini021
#   quantized     our fork + pyserini 0.39 + quantize_queries=true  .venv
#   truncated     our fork + pyserini 0.39, floats truncated to ints (what every
#                 number measured so far used)                       .venv
#   new-backend   upstream sdadas/pirb HEAD, backend "splade" (splade-index,
#                 exact float scoring on a SparseEncoder)             venv-pirb-new
#
#   scripts/pirb_regime_eval.sh REGIME LABEL MODEL_DIR TASKSET [BATCH] [MAXLEN]
#     TASKSET = name of configs/pirb_tasks/<TASKSET>.json (tiny10, heldout8, ...)
#     MAXLEN  = evaluate with this max sequence length instead of the one the
#               checkpoint carries. Our fork always tokenizes at 512 unless the
#               model config says otherwise; the upstream backend loads a
#               SparseEncoder and so uses the checkpoint's own max_seq_length
#               (256 for everything trained at 256, 512 for the soups and the
#               base), which is why the two regimes can disagree on arguana-pl.
#
# Writes results-regimes/<REGIME>/<LABEL>/<TASKSET>.json
#   {"regime","model","taskset","max_seq_length","tasks":{task_id: ndcg@10},"mean_ndcg@10","n_tasks"}
# and the raw log under logs/regimes/. Indexes are removed after the run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
REGIME="$1"; LABEL="$2"; MODEL="$3"; TASKSET="$4"; BATCH="${5:-16}"; MAXLEN="${6:-}"
BENCH="$REPO/configs/pirb_tasks/$TASKSET.json"
[ -f "$BENCH" ] || { echo "no task set $BENCH"; exit 1; }
[ -e "$MODEL" ] || { echo "model $MODEL missing"; exit 1; }
MODEL_ABS="$(cd "$MODEL" && pwd)"
OUT_DIR="$REPO/results-regimes/$REGIME/$LABEL"; mkdir -p "$OUT_DIR" "$REPO/logs/regimes"
OUT="$OUT_DIR/$TASKSET.json"
LOG="$REPO/logs/regimes/${REGIME}_${LABEL}_${TASKSET}.log"
CACHE="$REPO/cache/regimes/${REGIME}_${LABEL}_${TASKSET}"; mkdir -p "$CACHE"
CFG="$CACHE/models_config.json"
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
export ORPHAN_GUARD_EXEMPT=1
THREADS=6

# With MAXLEN: a view of the checkpoint (symlinks) whose sentence_bert_config.json
# and tokenizer_config.json carry that length, so the SparseEncoder path honours
# it too; the fork regimes additionally get it as the max_seq_length config key.
EVAL_MODEL="$MODEL_ABS"; LEN_KEY=""
if [ -n "$MAXLEN" ]; then
    EVAL_MODEL="$CACHE/model-len$MAXLEN"; rm -rf "$EVAL_MODEL"; mkdir -p "$EVAL_MODEL"
    for f in "$MODEL_ABS"/*; do ln -s "$f" "$EVAL_MODEL/"; done
    rm -f "$EVAL_MODEL/sentence_bert_config.json" "$EVAL_MODEL/tokenizer_config.json"
    "$REPO/.venv/bin/python" - "$MODEL_ABS" "$EVAL_MODEL" "$MAXLEN" <<'PYEOF'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(f"{src}/sentence_bert_config.json")); c["max_seq_length"] = n
json.dump(c, open(f"{dst}/sentence_bert_config.json", "w"), indent=2)
t = json.load(open(f"{src}/tokenizer_config.json")); t["model_max_length"] = n
json.dump(t, open(f"{dst}/tokenizer_config.json", "w"), indent=2, ensure_ascii=False)
PYEOF
    LEN_KEY=", \"max_seq_length\": $MAXLEN"
fi

case "$REGIME" in
  old-pyserini)
    PYBIN=/home/jsowa/work/venv-pyserini021/bin/python; DIR="$REPO/third_party/pirb"
    echo "[{\"name\": \"$EVAL_MODEL\", \"fp16\": true, \"type\": \"splade\", \"batch_size\": $BATCH, \"threads\": $THREADS$LEN_KEY}]" > "$CFG" ;;
  quantized)
    PYBIN="$REPO/.venv/bin/python"; DIR="$REPO/third_party/pirb"
    echo "[{\"name\": \"$EVAL_MODEL\", \"fp16\": true, \"type\": \"splade\", \"batch_size\": $BATCH, \"threads\": $THREADS, \"quantize_queries\": true$LEN_KEY}]" > "$CFG" ;;
  truncated)
    PYBIN="$REPO/.venv/bin/python"; DIR="$REPO/third_party/pirb"
    echo "[{\"name\": \"$EVAL_MODEL\", \"fp16\": true, \"type\": \"splade\", \"batch_size\": $BATCH, \"threads\": $THREADS$LEN_KEY}]" > "$CFG" ;;
  new-backend)
    PYBIN=/home/jsowa/work/venv-pirb-new/bin/python; DIR=/home/jsowa/work/pirb-upstream
    echo "[{\"name\": \"$EVAL_MODEL\", \"fp16\": true, \"type\": \"splade\", \"backend\": \"splade\", \"encoder_type\": \"sentence_transformer\", \"batch_size\": $BATCH}]" > "$CFG" ;;
  *) echo "unknown regime $REGIME"; exit 1 ;;
esac

RAW="$CACHE/results_raw.json"
extra=()
[ "$REGIME" != new-backend ] && extra=(--results_json "$RAW")
( cd "$DIR" && "$PYBIN" run_benchmark.py --models_config "$CFG" --benchmark_config "$BENCH" \
    --data_dir "$REPO/third_party/pirb/data" --cache_dir "$CACHE" --scope full --overwrite "${extra[@]}" ) > "$LOG" 2>&1
rc=$?

"$REPO/.venv/bin/python" - "$REGIME" "$LABEL" "$TASKSET" "$MODEL_ABS" "$LOG" "$RAW" "$OUT" "$MAXLEN" <<'PYEOF'
import json, os, re, sys
regime, label, taskset, model, log, raw, out, maxlen = sys.argv[1:9]
tasks = {}
if os.path.exists(raw):
    for mr in json.load(open(raw)).get("results", []):
        for k, v in mr.items():
            if k in ("model", "ndcg_tasks", "datasets") or k.startswith("average"):
                continue
            if isinstance(v, dict):
                nd = next((v[kk] for kk in v if kk.lower().startswith("ndcg@10")), None)
            else:
                nd = v
            if isinstance(nd, (int, float)):
                tasks[k] = float(nd)
if not tasks:  # upstream prints "Results for task: X" then "NDCG@10: 64.7000%, ..."
    cur = None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"Results for task: (\S+)", line.strip())
        if m:
            cur = m.group(1).lower(); continue
        m = re.search(r"NDCG@10: ([0-9.]+)%", line)
        if m and cur:
            tasks[cur] = float(m.group(1)); cur = None
rep = {"regime": regime, "label": label, "model": model, "taskset": taskset, "max_seq_length": int(maxlen) if maxlen else None,
       "tasks": tasks, "n_tasks": len(tasks),
       "mean_ndcg@10": (sum(tasks.values()) / len(tasks)) if tasks else None}
json.dump(rep, open(out, "w"), indent=2)
print(f"{regime}/{label}/{taskset}: {len(tasks)} tasks, mean nDCG@10 = {rep['mean_ndcg@10']}")
for t, v in sorted(tasks.items()):
    print(f"  {t:40s} {v:6.2f}")
PYEOF
rm -rf "$CACHE"
exit $rc
