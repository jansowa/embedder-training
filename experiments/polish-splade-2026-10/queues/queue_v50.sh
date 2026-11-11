#!/usr/bin/env bash
# Queue v50 (3060 Ti, evening 10.11, ~2.5 h): is the gain of the student-mined arm more than index density?
# v49: arm-dense-kl-n5kj-mined-ens400-r2v12 = 64.48 / 43.60, trec-covid 71.48 at 515 dims/doc, against r2v12
# 64.20 / 43.15 / 67.02 at 404. The bias form (lm_head.bias shifted by -(e^T - 1)) cut r2v12 by 10 % at T = 0.05 and
# 19 % at 0.10 for -0.07 and -0.15 heldout8. Steps: 1 bias forms at 0.10, 0.15, 0.20 with dims/doc each;
# 2 tiny10, heldout8, trec-covid for 0.10 and 0.15 (0.20 only when 0.15 still stays above 420 dims/doc);
# 3 summary_v50.md, commit, push loop. Reading at about 400 dims/doc: heldout8 >= 43.45 = the gain is real at equal
# cost; 43.1-43.2 = the mined rows bought density only.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v50.sh > logs/queue_v50.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v50 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v50.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
slbv() { regime_mean "$SREG/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
slbs() { echo "$(slbv "$1" tiny10) / $(slbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
small() { EXP_ROOT=$SEXP RESULTS_ROOT=$SRES REGIMES_ROOT=$SREG "$@"; }
h2task() { local bs=16; case "$2" in heldout2_poquad|heldout2_treccovid) bs=8;; esac; [ -d "models/$1" ] && regime quantized "$1" "models/$1" "$2" "$bs"; }
tc() { lbv "$1" heldout2_treccovid; }
measure() {  # measure LABEL MODEL REFLINE (main series)
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    sparsity "$label" "$model"; regime quantized "$label" "$model" tiny10 16; regime quantized "$label" "$model" heldout8 16
    log "  $label: $(lbs "$label") ($ref), $(dims "$label") dims/doc"
}
biased() {  # biased SRC_DIR OUT_NAME THRESHOLD: models/OUT_NAME = SRC with lm_head.bias (and its decoder copy) shifted by -(e^THRESHOLD - 1)
    local src="$1" out="$2" thr="$3"
    [ -d "models/$out" ] && return 0
    [ -d "$src" ] || { log "$out: $src missing"; return 1; }
    cp -r "$src" "models/$out" && "$PY" - "models/$out" "$thr" <<'EOF' && log "$out written (lm_head.bias = -(e^$thr - 1))" || { log "$out FAILED"; rm -rf "models/$out"; return 1; }
import math, sys
from safetensors.torch import load_file, save_file
d = sys.argv[1] + "/model.safetensors"; thr = float(sys.argv[2])
t = load_file(d); b = math.exp(thr) - 1.0; n = 0
for base in ("lm_head.bias", "predictions.bias"):
    if base in t:
        shifted = t[base] - b
        t[base] = shifted; t[base.replace(".bias", ".decoder.bias")] = shifted.clone(); n += 1
        print(base, "and its decoder copy shifted by", -b)
assert n, "no MLM head bias tensor found: " + ", ".join(k for k in t if "bias" in k and ("head" in k or "pred" in k or "decoder" in k))
save_file(t, d, metadata={"format": "pt"})
EOF
}
M=arm-dense-kl-n5kj-mined-ens400-r2v12
log "=== v50 start: bias forms of $M ($(lbs $M), trec-covid $(tc $M), $(dims $M) dims/doc); $(free_gb) GB free ==="
for t in 010 015 020; do biased "models/$M" "$M-bias$t" "0.${t#0}" && sparsity "$M-bias$t" "models/$M-bias$t"; done
log "  dims/doc: full $(dims $M), bias010 $(dims $M-bias010), bias015 $(dims $M-bias015), bias020 $(dims $M-bias020) (r2v12 404)"
full() {  # full TAG: tiny10, heldout8, trec-covid of one bias form
    local m="$M-bias$1"
    measure "$m" "models/$m" "full $(lbs $M) at $(dims $M); r2v12 64.20 / 43.15 at 404"
    h2task "$m" heldout2_treccovid; log "  $m trec-covid $(tc "$m") (full $(tc $M), r2v12 67.02), $(dims "$m") dims/doc"
}
full 015
full 010
if [ "$(gte "$(dims $M-bias015)" 420)" = 1 ]; then full 020; else log "  bias015 is at $(dims $M-bias015) dims/doc: bias020 not measured"; fi

SUM="$RESULTS_ROOT/summary_v50.md"
{
    echo "# v50 (evening 10.11, quantized regime, nDCG@10): the student-mined arm at lower index cost (bias forms)"; echo
    echo "biasT = the checkpoint with lm_head.bias shifted by -(e^T - 1). r2v12 = the same recipe on the old negatives."; echo
    echo "| model | tiny10 | heldout8 | trec-covid | dims/doc |"; echo "|---|---:|---:|---:|---:|"
    for m in arm-dense-kl-n5kj-ens400-r2v12 arm-dense-kl-n5kj-ens400-r2v12-bias005 arm-dense-kl-n5kj-ens400-r2v12-bias010 $M $M-bias010 $M-bias015 $M-bias020; do
        echo "| $m | $(lbv $m tiny10) | $(lbv $m heldout8) | $(tc $m) | $(dims $m) |"
    done
    echo; echo "## Queue state (v50)"; echo; echo '```'; grep -E "mined-ens400-r2v12-bias" "$STATE"; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_v50.sh" 2>/dev/null
for t in 010 015 020; do for p in "$REGIMES_ROOT/quantized/$M-bias$t" "$RESULTS_ROOT/sparsity/$M-bias$t.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git commit -q -m "Add the v50 results: the student-mined arm in the bias form at lower index cost" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v50.log" 2>&1 && { echo pushed >> "$LOGS/push_v50.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v50 done ($(date '+%d.%m %H:%M')) ==="
