#!/usr/bin/env bash
# v41 (7.11 day, 3060 Ti, ~7 h): the colleague rows as a mix component and as a soup member first, then the density
# questions v40 / s3 left open.
#   1  n5 + kj: the 23.5-gated n5 rows (fiqa + 16 k squadv2-v2) plus all 2 219 colleague rows, the ens-n5v2 recipe
#      -> arm-dense-kl-n5kj-ens400 -> sparsity, tiny10, heldout8. Reading: above ens-n5v2 (63.95 / 42.96) on heldout8
#      with legal-questions > 83.97 = the colleague rows earn a seat in the mix; at or below = redundant with SQuAD.
#   2  soups with the colleague arm (arm-dense-kl-kj-ens400, 62.52 / 41.84): soup-v13a = uniform(soup-v9b, ens-n5v2,
#      kj-ens400); soup-v13b = (0.4, 0.4, 0.2) of the same. Reading against soup-v12 (63.96 / 42.92, 426 dims/doc): a soup
#      at or above v12 with kj inside = the arm is a usable merge member even though it is weak alone (as lenmatch-far was).
#   3  the weight threshold on heldout8: soup-v12 at min_weight 0.05 (tiny10 + heldout8), ens-n5v2 at 0.05 (heldout8).
#      v40: ens-n5v2 at 0.05 kept tiny10 (63.97) with -15 % postings. Reading: heldout8 within 0.1 = the threshold is
#      real; then the bias form: soup-v12-bias005 = soup-v12 with lm_head.bias = -(e^0.05 - 1) = -0.0513 (the threshold
#      baked into the checkpoint as a soft threshold, zero cost, no custom layer) -> sparsity, tiny10, heldout8.
#   4  the ALBERT student hl-full-s3 as the query encoder over the soup-v9b index (tiny10 + heldout8) and at 0.05 (tiny10).
#   5  summary_v41.md, commit (no third_party paths: the PIRB fork is a submodule).
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v41.sh > logs/queue_v41.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v41 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v41.pid"
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
KJ=models/arm-dense-kl-kj-ens400
V12=models/soup-v12
V9B=models/soup-v9b
HL=models/arm-b4-l0125-f1536-hl-r256-w-full-s3

log "=== v41 start: n5 + kj, soups with kj, thresholds on heldout8 and the bias form, ALBERT asymmetric; $(free_gb) GB free ==="

measure() {  # measure LABEL MODEL REFLINE: sparsity, tiny10, heldout8
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    sparsity "$label" "$model"
    regime quantized "$label" "$model" tiny10 16
    regime quantized "$label" "$model" heldout8 16
    log "  $label: $(lbs "$label") ($ref), $(dims "$label") dims/doc; legal-questions $(task "$label" heldout8 poleval-2022-test-A-legal-questions), arguana $(task "$label" tiny10 arguana-pl)"
}
soup() {  # soup NAME WEIGHTS MODEL...: merge (uniform when WEIGHTS is empty), then measure
    local name="$1" weights="$2"; shift 2
    if [ ! -d "models/$name" ]; then
        for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
        wait_gpu_free
        # shellcheck disable=SC2086
        timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" ${weights:+--weights $weights} --output "models/$name" > "$LOGS/soup_$name.log" 2>&1 \
            && log "soup $name written ($*; weights ${weights:-uniform})" || { log "soup $name FAILED (logs/soup_$name.log)"; rm -rf "models/$name"; return 1; }
    fi
    measure "$name" "models/$name" "soup-v12 63.96 / 42.92 with legal-questions 83.49, ens-n5v2 63.95 / 42.96"
}

# 1. n5 + kj
N5KJ=datasets/arms/n5-plus-kj-tens400.jsonl
if [ ! -s "$N5KJ" ]; then
    "$PY" - datasets/arms/fiqa-plus-squadv2v2-18k-tens400.jsonl datasets/arms/karoljez-2219-tens400.jsonl "$N5KJ" <<'EOF' | while read -r l; do log "  $l"; done
import json, sys
n5, kj, out = sys.argv[1:4]
kept = dropped = nk = 0
with open(out, "w", encoding="utf-8") as o:
    for line in open(n5, encoding="utf-8"):   # the 23.5 gate of the n5 recipe (gemma score of the English pair), one positive per row
        r = json.loads(line)
        s = r.get("pos_scores_stronger_reranker") or r.get("pos_scores")
        if s and s[0] is not None and s[0] > 23.5:
            o.write(line.rstrip("\n") + "\n"); kept += 1
        else:
            dropped += 1
    for line in open(kj, encoding="utf-8"):
        o.write(line.rstrip("\n") + "\n"); nk += 1
print(f"n5 + kj file: {kept} gated n5 rows ({dropped} dropped by the 23.5 gate) + {nk} colleague rows -> {out}")
EOF
fi
if [ -s "$N5KJ" ]; then
    train dense-kl-n5kj-ens400 "$C/distil_dense_kl_n5kj_ens400.yaml"
    measure arm-dense-kl-n5kj-ens400 models/arm-dense-kl-n5kj-ens400 "ens-n5v2 63.95 / 42.96 with legal-questions 83.97, kj-ens400 62.52 / 41.84"
fi

# 2. soups with the colleague arm
soup soup-v13a "" "$V9B" "$ENS" "$KJ"
soup soup-v13b "0.4 0.4 0.2" "$V9B" "$ENS" "$KJ"

# 3. the weight threshold on heldout8, then its bias form
export PIRB_MIN_WEIGHT=0.05
regime quantized soup-v12-mw005 "$V12" tiny10 16
regime quantized soup-v12-mw005 "$V12" heldout8 16
regime quantized ens-n5v2-mw005 "$ENS" heldout8 16
unset PIRB_MIN_WEIGHT
log "  threshold 0.05: soup-v12 63.96 / 42.92 -> $(lbs soup-v12-mw005); ens-n5v2 63.95 / 42.96 -> $(lbs ens-n5v2-mw005)"
if [ ! -d models/soup-v12-bias005 ]; then
    cp -r "$V12" models/soup-v12-bias005 && "$PY" - models/soup-v12-bias005 <<'EOF' && log "soup-v12-bias005 written (lm_head.bias = -0.0513)" || { log "soup-v12-bias005 FAILED"; rm -rf models/soup-v12-bias005; }
import math, sys, torch
from safetensors.torch import load_file, save_file
d = sys.argv[1] + "/model.safetensors"
t = load_file(d)
b = math.exp(0.05) - 1.0          # log(1 + relu(x)) >= 0.05  <=>  x >= e^0.05 - 1
for k in ("lm_head.bias", "lm_head.decoder.bias"):
    if k in t:
        t[k] = t[k] - b
        print(k, "shifted by", -b, "mean now", float(t[k].mean()))
save_file(t, d, metadata={"format": "pt"})
EOF
fi
[ -d models/soup-v12-bias005 ] && measure soup-v12-bias005 models/soup-v12-bias005 "soup-v12 63.96 / 42.92 at 426 dims/doc; threshold 0.05 $(lbs soup-v12-mw005)"

# 4. the ALBERT student as the query encoder over the soup-v9b index, and at the threshold
if [ -d "$HL" ]; then
    export PIRB_QUERY_MODEL="$HL"
    regime quantized asym-hlfull-q-v9b "$V9B" tiny10 16
    regime quantized asym-hlfull-q-v9b "$V9B" heldout8 16
    unset PIRB_QUERY_MODEL
    export PIRB_MIN_WEIGHT=0.05
    regime quantized hl-full-s3-mw005 "$HL" tiny10 16
    unset PIRB_MIN_WEIGHT
    log "  hl-full-s3 62.97 / 42.10: as query encoder over v9b $(lbs asym-hlfull-q-v9b) (b4-full asym 63.70 / 42.42, soup-v9b 63.65 / 42.56); threshold 0.05 tiny10 $(lbv hl-full-s3-mw005 tiny10)"
fi

# 5. summary
SUM="$RESULTS_ROOT/summary_v41.md"
{
    echo "# v41 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): the colleague rows in the mix and in soups, thresholds on heldout8, the bias form, the ALBERT student asymmetric"
    echo
    echo "n5kj = the 23.5-gated n5 rows plus all 2 219 colleague rows, the ens-n5v2 recipe. soup-v13a = uniform soup of soup-v9b, ens-n5v2 and kj-ens400; soup-v13b = the same at 0.4 / 0.4 / 0.2. min_weight 0.05 = terms below 0.05 dropped on both sides at indexing. soup-v12-bias005 = soup-v12 with lm_head.bias shifted by -(e^0.05 - 1), the threshold baked into the checkpoint as a soft threshold. asym-hlfull-q-v9b = queries encoded by the ALBERT student (0.45 of the teacher's cost), index built by soup-v9b."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-dense-kl-kj-ens400 arm-dense-kl-n5v2-ens400 arm-dense-kl-n5kj-ens400 soup-v12 soup-v13a soup-v13b soup-v12-mw005 soup-v12-bias005 ens-n5v2-mw005 arm-b4-l0125-f1536-hl-r256-w-full-s3 asym-hlfull-q-v9b hl-full-s3-mw005 soup-v9b --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-dense-kl-kj-ens400 --model arm-dense-kl-n5v2-ens400 --model arm-dense-kl-n5kj-ens400 --model soup-v12 --model soup-v13a --model soup-v13b --model soup-v12-bias005 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v41)"; echo; echo '```'
    grep -E "^(train|sparsity|regime:quantized):(.*(n5kj|soup-v13|mw005|bias005|asym-hlfull).*)" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$RESULTS_ROOT"/sparsity/arm-dense-kl-n5kj-ens400.json "$RESULTS_ROOT"/sparsity/soup-v13a.json "$RESULTS_ROOT"/sparsity/soup-v13b.json "$RESULTS_ROOT"/sparsity/soup-v12-bias005.json \
    "$REGIMES_ROOT"/quantized/arm-dense-kl-n5kj-ens400 "$REGIMES_ROOT"/quantized/soup-v13a "$REGIMES_ROOT"/quantized/soup-v13b "$REGIMES_ROOT"/quantized/soup-v12-mw005 \
    "$REGIMES_ROOT"/quantized/soup-v12-bias005 "$REGIMES_ROOT"/quantized/ens-n5v2-mw005 "$REGIMES_ROOT"/quantized/hl-full-s3-mw005 "$REGIMES_ROOT"/quantized/asym-hlfull-q-v9b \
    "$C/distil_dense_kl_n5kj_ens400.yaml" "$EXP_ROOT/queues/queue_v41.sh" 2>/dev/null
git commit -q -m "Add the v41 results: the colleague rows in the mix and in soups, thresholds on heldout8 and the ALBERT student as the query encoder" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v41.log" 2>&1 && { echo pushed >> "$LOGS/push_v41.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v41 done ($(date '+%d.%m %H:%M')) ==="
