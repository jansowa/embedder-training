#!/usr/bin/env bash
# Queue v51 (3060 Ti, evening 10.11 after v50, ~10 min): the query side of the student-mined arm. dims/doc alone
# does not give the search cost; scripts/splade_index_cost.py reports active dims per query and postings per query
# on 150 documents + 150 queries per tiny task for r2v12 (old negatives), its bias form, the mined arm and the bias
# forms v50 measured. The bias shift acts on the shared head, so it trims queries as well as documents.
#   WAIT_FOR="v50" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v51.sh > logs/queue_v51.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v51 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v51.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v50}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
COST="$RESULTS_ROOT/index_cost_v51.json"
args=(--model base=models/base-splade-wrapped)
for m in $R2 $R2-bias005 $M $M-bias010 $M-bias015; do [ -d "models/$m" ] && args+=(--model "$m=models/$m"); done
log "=== v51 start: index cost (query dims, postings per query): ${args[*]} ==="
need_space; wait_gpu_free; wait_ram; s=$SECONDS
if ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" --output-json "$COST" > "$LOGS/index_cost_v51.log" 2>&1; then
    record "cost:v51" ok "-" "$((SECONDS - s))s"
    "$PY" - "$COST" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d["models"].items():
    print("%s: dims/doc %.1f, dims/query %.1f (median %.0f), postings/query %.0f, flops %.3f" % (k, v["doc_active_mean"], v["query_active_mean"], v["query_active_median"], v["postings_per_query"], v["flops"]))
PYEOF
    git add "$COST" "$EXP_ROOT/queues/queue_v51.sh" 2>/dev/null
    git commit -q -m "Add the v51 results: query-side index cost of the student-mined arm" 2>/dev/null && log "committed $(git log --oneline -1)"
    (for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v51.log" 2>&1 && { echo pushed >> "$LOGS/push_v51.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
    disown
else log "index-cost profile FAILED (logs/index_cost_v51.log)"; record "cost:v51" FAILED "-" "$((SECONDS - s))s"; fi
log "=== v51 done ($(date '+%d.%m %H:%M')) ==="
