#!/usr/bin/env bash
# v37 (5.11, short, ~75 min): the query-cost probe. The v11 soups have sparser documents than the base but denser
# queries, so their postings per query are above the base's (results/index_cost_v11.json). Two arms of the
# kl-n1v2 recipe (teacher soup-v9b, 4 k rows, 2 epochs; 63.25 / 42.62) with the query regularizer x4 and x10,
# trained with the sparse-gradient head without checkpointing (v34-speed PASS, 2.27x), then sparsity, tiny10
# and the index-cost sample of base, kl-n1v2 and both arms. Reading: tiny10 within 0.3 of kl-n1v2 at 10-20 %
# fewer postings per query = a free lever for every future arm; a drop > 0.5 = the query side is not free.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v37.sh > logs/queue_v37.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v37 MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-3600}   # kl-n1v2 took 3155 s stock; 2.27x faster expected
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v37.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
train_secs() { awk -F'\t' -v k="train:$1" '$1 == k && $2 == "ok" {v = $4} END {print v}' "$STATE"; }

log "=== v37 start: query-cost probe on the kl-n1v2 recipe (qreg x4, x10), fast head; $(free_gb) GB free ==="
for a in qreg4x qreg10x; do
    TAG="distil-v9b-kl-n1v2-$a"
    train "$TAG" "$C/distil_v9b_kl_n1v2_$a.yaml"
    if [ -d "models/arm-$TAG" ]; then
        sparsity "arm-$TAG" "models/arm-$TAG"
        regime quantized "arm-$TAG" "models/arm-$TAG" tiny10 16
        log "  arm-$TAG: tiny10 $(lbv "arm-$TAG" tiny10) (kl-n1v2 63.25), $(dims "arm-$TAG") dims/doc, train $(train_secs "arm-$TAG") (stock kl-n1v2 3155s)"
    fi
done

log "index cost: base, kl-n1v2, qreg4x, qreg10x"
wait_gpu_free; wait_ram
MODELS="--model base=models/base-splade-wrapped --model kl-n1v2=models/arm-distil-v9b-kl-n1v2"
for a in qreg4x qreg10x; do [ -d "models/arm-distil-v9b-kl-n1v2-$a" ] && MODELS="$MODELS --model $a=models/arm-distil-v9b-kl-n1v2-$a"; done
s=$SECONDS
timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 $MODELS \
    --output-json "$RESULTS_ROOT/index_cost_v37.json" > "$LOGS/index_cost_v37.log" 2>&1 \
    && log "index-cost profile written ($RESULTS_ROOT/index_cost_v37.json) in $((SECONDS - s))s" || log "index-cost profile FAILED (logs/index_cost_v37.log)"

"$PY" - "$RESULTS_ROOT/index_cost_v37.json" <<'EOF' | while read -r l; do log "  $l"; done
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print(f"no index cost: {e}"); sys.exit(0)
for m, v in d["models"].items():
    print(f"{m}: postings/query {v['postings_per_query']:.0f}, query dims {v['query_active_mean']:.1f}, doc dims {v['doc_active_mean']:.1f}")
EOF

SUM="$RESULTS_ROOT/summary_v37.md"
{
    echo "# v37 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): query-cost probe, kl-n1v2 recipe with the query regularizer x4 and x10"
    echo
    echo "Same rows, recipe and teacher as arm-distil-v9b-kl-n1v2 (63.25 / 42.62); query_regularizer_weight 5e-5 -> 2e-4 (x4) and 5e-4 (x10); sparse-gradient head, no checkpointing."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-distil-v9b-kl-n1v2 arm-distil-v9b-kl-n1v2-qreg4x arm-distil-v9b-kl-n1v2-qreg10x soup-v9b --tasksets tiny10 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-distil-v9b-kl-n1v2 --model arm-distil-v9b-kl-n1v2-qreg4x --model arm-distil-v9b-kl-n1v2-qreg10x --cost-json "$RESULTS_ROOT/index_cost_v37.json" 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Queue state (v37)"; echo; echo '```'
    grep -E "^(train|sparsity|regime:quantized):arm-distil-v9b-kl-n1v2-qreg" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$RESULTS_ROOT/index_cost_v37.json" "$RESULTS_ROOT"/sparsity/arm-distil-v9b-kl-n1v2-qreg*.json "$REGIMES_ROOT"/quantized/arm-distil-v9b-kl-n1v2-qreg* 2>/dev/null
git commit -q -m "Add the v37 results: the query regularizer against the postings per query" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin master > "$LOGS/push_v37.log" 2>&1 && { echo pushed >> "$LOGS/push_v37.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v37 done ($(date '+%d.%m %H:%M')) ==="
