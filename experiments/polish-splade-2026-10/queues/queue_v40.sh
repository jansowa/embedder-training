#!/usr/bin/env bash
# v40 (6/7.11 night, 3060 Ti, ~3.7 h, no training beyond one 12-minute arm): four cheap questions around the current
# best models (ens-n5v2 63.95 / 42.96, b4-full-s3 63.77 / 42.54, soup-v9b 63.65 / 42.56; tiny10 / heldout8, quantized).
#   1  the colleague's rows (datasets/karoljez/train, 2 219 audited rows, 64 % legal) with the ensemble teacher:
#      arm-dense-kl-kj-ens400 (the n1v2-ens400 recipe, 63.43 / 42.58 on 4 146 rows) -> sparsity, tiny10, heldout8.
#      Reading: heldout8 per task, legal-questions apart from the other seven (64 % legal rows, leak checked "small" only).
#   2  soup-v12 = uniform soup of soup-v9b and ens-n5v2 (the first arm above the teacher on both sets); soups were
#      sparser than their members so far. Reading: >= 63.95 / 42.96 = the new reference; below ens-n5v2 = keep the arm.
#   3  the small-weight threshold at indexing (zero training): index-cost profile with the doc AND query side for
#      min_weight 0.01 / 0.02 / 0.03 / 0.05 (results/index_cost_v40.json), then tiny10 of ens-n5v2 at 0.02 and 0.05
#      and of soup-v9b at 0.02 (PIRB min_weight, both sides). Reading: the threshold that brings ens-n5v2 back to
#      v9b's postings/query at < 0.2 tiny10 = the density fix costs nothing; else regulator / query regulariser (day).
#   4  asymmetric evaluation: queries from the b4-full-s3 student over the soup-v9b index (PIRB query_model), tiny10 +
#      heldout8; control = the base polish-splade as the query encoder over the same index (must differ from v9b's own
#      63.65). Reading: asym within 0.3 of v9b = the student serves as the query encoder with the big index (the
#      intended deployment); far below = the student needs its own index or a cross loss in KD.
#   5  summary_v40.md, commit (results, configs, scripts touched: pirb_regime_eval.sh, splade_index_cost.py, PIRB fork).
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v40.sh > logs/queue_v40.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v40 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-5400}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v40.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
task() { "$PY" -c "import json;print(round(json.load(open('$REGIMES_ROOT/quantized/$1/$2.json'))['tasks'].get('$3', float('nan')), 2))" 2>/dev/null; }
ENS=models/arm-dense-kl-n5v2-ens400
B4=models/arm-b4-l0125-f3072-full-s3
V9B=models/soup-v9b

log "=== v40 start: colleague rows, soup-v12, weight threshold, asymmetric evaluation; $(free_gb) GB free ==="

# 1. the colleague's rows with the ensemble teacher
if [ -s datasets/arms/karoljez-2219-tens400.jsonl ]; then
    train dense-kl-kj-ens400 "$C/distil_dense_kl_kj_ens400.yaml"
    if [ -d models/arm-dense-kl-kj-ens400 ]; then
        sparsity arm-dense-kl-kj-ens400 models/arm-dense-kl-kj-ens400
        regime quantized arm-dense-kl-kj-ens400 models/arm-dense-kl-kj-ens400 tiny10 16
        regime quantized arm-dense-kl-kj-ens400 models/arm-dense-kl-kj-ens400 heldout8 16
        log "  arm-dense-kl-kj-ens400: $(lbs arm-dense-kl-kj-ens400) (n1v2-ens400 63.43 / 42.58, soup-v9b 63.65 / 42.56), $(dims arm-dense-kl-kj-ens400) dims/doc; legal-questions $(task arm-dense-kl-kj-ens400 heldout8 poleval-2022-test-A-legal-questions) (soup-v9b $(task soup-v9b heldout8 poleval-2022-test-A-legal-questions))"
    fi
else log "colleague rows datasets/arms/karoljez-2219-tens400.jsonl missing; step 1 skipped"; fi

# 2. soup-v12 = uniform(soup-v9b, ens-n5v2)
if [ ! -d models/soup-v12 ] && [ -d "$ENS" ]; then
    wait_gpu_free
    timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$V9B" "$ENS" --output models/soup-v12 > "$LOGS/soup_v12.log" 2>&1 \
        && log "soup-v12 written (soup-v9b + ens-n5v2, uniform)" || { log "soup-v12 FAILED (logs/soup_v12.log)"; rm -rf models/soup-v12; }
fi
if [ -d models/soup-v12 ]; then
    sparsity soup-v12 models/soup-v12
    regime quantized soup-v12 models/soup-v12 tiny10 16
    regime quantized soup-v12 models/soup-v12 heldout8 16
    log "  soup-v12: $(lbs soup-v12) (ens-n5v2 63.95 / 42.96, soup-v9b 63.65 / 42.56), $(dims soup-v12) dims/doc"
fi

# 3. the small-weight threshold: profile (doc + query side) and tiny10 at two thresholds
COST="$RESULTS_ROOT/index_cost_v40.json"
if [ ! -s "$COST" ]; then
    wait_gpu_free; s=$SECONDS
    MODELS="--model soup-v9b=$V9B --model ens-n5v2=$ENS --model b4-full-s3=$B4"
    [ -d models/soup-v12 ] && MODELS+=" --model soup-v12=models/soup-v12"
    # shellcheck disable=SC2086
    timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 $MODELS \
        --min-weight 0.01 --min-weight 0.02 --min-weight 0.03 --min-weight 0.05 --output-json "$COST" > "$LOGS/index_cost_v40.log" 2>&1 \
        && log "index-cost profile with thresholds written ($COST) in $((SECONDS - s))s" || log "index-cost profile FAILED (logs/index_cost_v40.log)"
    grep -E "doc_active" "$LOGS/index_cost_v40.log" | while read -r l; do log "  $l"; done
fi
asym_or_thr() {  # asym_or_thr LABEL MODEL TASKSET: regime with PIRB_QUERY_MODEL / PIRB_MIN_WEIGHT taken from the environment
    regime quantized "$1" "$2" "$3" 16
}
export PIRB_MIN_WEIGHT=0.02
asym_or_thr ens-n5v2-mw002 "$ENS" tiny10
asym_or_thr soup-v9b-mw002 "$V9B" tiny10
export PIRB_MIN_WEIGHT=0.05
asym_or_thr ens-n5v2-mw005 "$ENS" tiny10
unset PIRB_MIN_WEIGHT
log "  threshold tiny10: ens-n5v2 $(lbv arm-dense-kl-n5v2-ens400 tiny10) -> mw0.02 $(lbv ens-n5v2-mw002 tiny10), mw0.05 $(lbv ens-n5v2-mw005 tiny10); soup-v9b 63.65 -> mw0.02 $(lbv soup-v9b-mw002 tiny10)"

# 4. asymmetric evaluation: student queries over the soup-v9b index
if [ -d "$B4" ]; then
    export PIRB_QUERY_MODEL=models/base-splade-wrapped
    asym_or_thr asym-base-q-v9b "$V9B" tiny10          # control: must differ from soup-v9b's own 63.65
    export PIRB_QUERY_MODEL="$B4"
    asym_or_thr asym-b4full-q-v9b "$V9B" tiny10
    asym_or_thr asym-b4full-q-v9b "$V9B" heldout8
    unset PIRB_QUERY_MODEL
    log "  asymmetric (queries from X over the soup-v9b index): base $(lbv asym-base-q-v9b tiny10) / -, b4-full-s3 $(lbs asym-b4full-q-v9b) (b4-full-s3 alone 63.77 / 42.54, soup-v9b 63.65 / 42.56)"
fi

# 5. summary
SUM="$RESULTS_ROOT/summary_v40.md"
{
    echo "# v40 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): colleague rows, soup-v12, weight threshold, asymmetric evaluation"
    echo
    echo "1. arm-dense-kl-kj-ens400: the ensemble teacher (soup-v9b + PolDense-400M, scale 111.8) on the colleague's 2 219 rows (datasets/karoljez/train, 64 % legal), the n1v2-ens400 recipe. 2. soup-v12 = uniform soup of soup-v9b and arm-dense-kl-n5v2-ens400. 3. min_weight = terms with weight below the threshold dropped on both sides at indexing (PIRB fork config key min_weight; the usual rule is round(w*100) > 0, i.e. 0.005). 4. asym-X-q-v9b = queries encoded by X, index built by soup-v9b (PIRB config key query_model)."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-dense-kl-n1v2-ens400 arm-dense-kl-kj-ens400 arm-dense-kl-n5v2-ens400 soup-v12 ens-n5v2-mw002 ens-n5v2-mw005 soup-v9b-mw002 asym-base-q-v9b asym-b4full-q-v9b arm-b4-l0125-f3072-full-s3 soup-v9b --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-dense-kl-kj-ens400 --model arm-dense-kl-n5v2-ens400 --model soup-v12 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Index cost with thresholds (one PIRB sample: 1 350 passages + 1 350 queries; doc dims, query dims, postings/query)"; echo; echo '```'
    [ -s "$COST" ] && "$PY" - "$COST" <<'EOF' 2>> "$LOGS/queue_summary.log"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{'model':34s} {'dims/doc':>9s} {'dims/query':>11s} {'postings/q':>11s} {'flops':>7s}")
for name, r in d["models"].items():
    rows = [(name, r)] + [(f"{name} min_weight={t}", s) for t, s in r.get("min_weight", {}).items()]
    for label, s in rows:
        print(f"{label:34s} {s['doc_active_mean']:9.1f} {s['query_active_mean']:11.1f} {s['postings_per_query']:11.0f} {s['flops']:7.3f}")
EOF
    echo '```'
    echo; echo "## Queue state (v40)"; echo; echo '```'
    grep -E "^(train|sparsity|regime:quantized):(arm-dense-kl-kj-ens400|soup-v12|ens-n5v2-mw|soup-v9b-mw|asym-)" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$COST" "$RESULTS_ROOT"/sparsity/arm-dense-kl-kj-ens400.json "$RESULTS_ROOT"/sparsity/soup-v12.json \
    "$REGIMES_ROOT"/quantized/arm-dense-kl-kj-ens400 "$REGIMES_ROOT"/quantized/soup-v12 "$REGIMES_ROOT"/quantized/ens-n5v2-mw* "$REGIMES_ROOT"/quantized/soup-v9b-mw* "$REGIMES_ROOT"/quantized/asym-* \
    "$C/distil_dense_kl_kj_ens400.yaml" "$EXP_ROOT/queues/queue_v40.sh" scripts/pirb_regime_eval.sh scripts/splade_index_cost.py \
    third_party/pirb/search/splade.py third_party/pirb/tests/test_splade_query_model.py 2>/dev/null
git commit -q -m "Add the v40 results: colleague rows, soup-v12, the weight threshold and asymmetric evaluation" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v40.log" 2>&1 && { echo pushed >> "$LOGS/push_v40.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v40 done ($(date '+%d.%m %H:%M')) ==="
