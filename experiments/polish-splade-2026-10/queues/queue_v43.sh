#!/usr/bin/env bash
# v43 (8.11 morning, 3060 Ti, after v42, ~2.8 h): the bias form of the threshold at two strengths, for the ALBERT student
# and for the full model. v41: the threshold 0.05 is free on tiny10 and heldout8 for soup-v12 and ens-n5v2, and the bias
# form (lm_head.bias = -(e^0.05 - 1)) on soup-v12 gave 64.05 / 42.83 at 378 dims/doc; hl-full-s3 at the hard threshold
# 0.05 kept tiny10 (62.97) but its index is the densest we have (525 dims/doc, 183 dims/query, 27 501 postings).
#   1  hl-full-s3-bias005 and hl-full-s3-bias010 (bias -(e^0.05 - 1) = -0.0513 and -(e^0.10 - 1) = -0.1052): sparsity,
#      tiny10, heldout8 each. Reading: how much of the ALBERT's density is junk weight; which strength still holds 62.97 / 42.10.
#   2  the best full model of the night (v42's choice by heldout8; its bias005 is measured in v42) at bias 0.10. Reading: 0.10
#      within 0.15 of the untreated model on heldout8 = the publication recipe can use the stronger cut.
#   3  index-cost profile (doc + query side, thresholds 0.05 and 0.10) for soup-v9b, soup-v12, soup-v12-bias005, the best model and
#      its bias005 / bias010, hl-full-s3, hl-full-s3-bias005, hl-full-s3-bias010.
#   4  summary_v43.md, commit (no third_party paths), push.
#   WAIT_FOR="v42" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v43.sh > logs/queue_v43.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v43 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v43.pid"
guard_alive

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v42}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
HL=models/arm-b4-l0125-f1536-hl-r256-w-full-s3

log "=== v43 start: the bias form of the threshold at 0.05 and 0.10 for the ALBERT student and soup-v12; $(free_gb) GB free ==="

biased() {  # biased SRC_DIR OUT_NAME THRESHOLD: models/OUT_NAME = SRC with lm_head.bias shifted by -(e^THRESHOLD - 1)
    local src="$1" out="$2" thr="$3"
    [ -d "models/$out" ] && return 0
    [ -d "$src" ] || { log "$out: $src missing"; return 1; }
    cp -r "$src" "models/$out" && "$PY" - "models/$out" "$thr" <<'EOF' && log "$out written (lm_head.bias = -(e^$thr - 1))" || { log "$out FAILED"; rm -rf "models/$out"; return 1; }
import math, sys
from safetensors.torch import load_file, save_file
d = sys.argv[1] + "/model.safetensors"; thr = float(sys.argv[2])
t = load_file(d); b = math.exp(thr) - 1.0; n = 0
# both the head bias and its tied decoder copy are written: transformers 4.48 replaces the head bias by the
# decoder bias on an fp32 load (sentence-transformers), so a file with only lm_head.bias loses the shift there
for base in ("lm_head.bias", "predictions.bias"):
    if base in t:
        shifted = t[base] - b
        t[base] = shifted; t[base.replace(".bias", ".decoder.bias")] = shifted.clone(); n += 1
        print(base, "and its decoder copy shifted by", -b)
assert n, "no MLM head bias tensor found: " + ", ".join(k for k in t if "bias" in k and ("head" in k or "pred" in k or "decoder" in k))
save_file(t, d, metadata={"format": "pt"})
EOF
}
measure() {  # measure LABEL MODEL REFLINE
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    sparsity "$label" "$model"
    regime quantized "$label" "$model" tiny10 16
    regime quantized "$label" "$model" heldout8 16
    log "  $label: $(lbs "$label") ($ref), $(dims "$label") dims/doc"
}

# 1. the ALBERT student at two bias strengths
biased "$HL" hl-full-s3-bias005 0.05 && measure hl-full-s3-bias005 models/hl-full-s3-bias005 "hl-full-s3 62.97 / 42.10 at 512 dims/doc; hard 0.05 tiny10 62.97"
biased "$HL" hl-full-s3-bias010 0.10 && measure hl-full-s3-bias010 models/hl-full-s3-bias010 "hl-full-s3 62.97 / 42.10; bias005 $(lbs hl-full-s3-bias005)"

# 2. the best full model of the night (v42's choice by heldout8) at 0.10; its bias005 comes from v42
gte() { "$PY" -c "import sys;a,b=sys.argv[1:3];print(int(bool(a) and bool(b) and float(a)>=float(b)))" "$1" "$2"; }
BEST=""; BEST_H=""
for m in arm-dense-kl-n5kj-ens400 arm-dense-kl-n5kj-ens400-s43 soup-v14 soup-v14s arm-dense-kl-n5kj-ens400-r2v12 arm-dense-kl-n7-ens400 arm-dense-kl-n5kj-ens400-reg soup-v12; do
    h=$(lbv "$m" heldout8); [ -n "$h" ] || continue
    if [ -z "$BEST_H" ] || [ "$(gte "$h" "$BEST_H")" = 1 ]; then BEST="$m"; BEST_H="$h"; fi
done
log "best full model by heldout8: ${BEST:-none} (${BEST_H:-?}); bias005 $(lbs "$BEST-bias005")"
[ -n "$BEST" ] && biased "models/$BEST" "$BEST-bias010" 0.10 && measure "$BEST-bias010" "models/$BEST-bias010" "$BEST $(lbs "$BEST") at $(dims "$BEST") dims/doc; bias005 $(lbs "$BEST-bias005") at $(dims "$BEST-bias005")"

# 3. index-cost profile with both thresholds
COST="$RESULTS_ROOT/index_cost_v43.json"
if [ ! -s "$COST" ]; then
    wait_gpu_free; s=$SECONDS
    MODELS="--model soup-v9b=models/soup-v9b --model soup-v12=models/soup-v12 --model hl-full-s3=$HL"
    for m in soup-v12-bias005 ${BEST:+$BEST $BEST-bias005 $BEST-bias010} hl-full-s3-bias005 hl-full-s3-bias010; do [ -d "models/$m" ] && MODELS+=" --model $m=models/$m"; done
    # shellcheck disable=SC2086
    timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 $MODELS \
        --min-weight 0.05 --min-weight 0.10 --output-json "$COST" > "$LOGS/index_cost_v43.log" 2>&1 \
        && log "index-cost profile (doc + query side, thresholds 0.05 / 0.10) written ($COST) in $((SECONDS - s))s" || log "index-cost profile FAILED"
    grep -E "doc_active" "$LOGS/index_cost_v43.log" | while read -r l; do log "  $l"; done
fi

# 4. summary
SUM="$RESULTS_ROOT/summary_v43.md"
{
    echo "# v43 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): the bias form of the threshold at 0.05 and 0.10"
    echo
    echo "biasT = the checkpoint with lm_head.bias shifted by -(e^T - 1): a dimension stays active only when its pooled weight would exceed T, the other weights drop slightly (the threshold inside the checkpoint, no custom layer). hl-full-s3 = the ALBERT student (50 M params, 0.45 of the teacher's cost); the best full model of the night = ${BEST:-none} (${BEST_H:-?} on heldout8)."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-b4-l0125-f1536-hl-r256-w-full-s3 hl-full-s3-bias005 hl-full-s3-bias010 soup-v12 soup-v12-bias005 ${BEST:+$BEST $BEST-bias005 $BEST-bias010} soup-v9b --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-b4-l0125-f1536-hl-r256-w-full-s3 --model hl-full-s3-bias005 --model hl-full-s3-bias010 --model soup-v12 --model soup-v12-bias005 ${BEST:+--model $BEST --model $BEST-bias005 --model $BEST-bias010} 2>> "$LOGS/queue_summary.log"
    echo; echo "## Index cost (one PIRB sample: 1 350 passages + 1 350 queries; doc dims, query dims, postings/query; thresholds applied at indexing on top of each checkpoint)"; echo; echo '```'
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
    echo; echo "## Queue state (v43)"; echo; echo '```'
    grep -E "^(sparsity|regime:quantized):(hl-full-s3-bias|.*-bias010)" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$COST" "$RESULTS_ROOT"/sparsity/hl-full-s3-bias005.json "$RESULTS_ROOT"/sparsity/hl-full-s3-bias010.json ${BEST:+"$RESULTS_ROOT/sparsity/$BEST-bias010.json"} \
    "$REGIMES_ROOT"/quantized/hl-full-s3-bias005 "$REGIMES_ROOT"/quantized/hl-full-s3-bias010 ${BEST:+"$REGIMES_ROOT/quantized/$BEST-bias010"} "$EXP_ROOT/queues/queue_v43.sh" 2>/dev/null
git commit -q -m "Add the v43 results: the bias form of the threshold at two strengths for the ALBERT student and soup-v12" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v43.log" 2>&1 && { echo pushed >> "$LOGS/push_v43.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v43 done ($(date '+%d.%m %H:%M')) ==="
