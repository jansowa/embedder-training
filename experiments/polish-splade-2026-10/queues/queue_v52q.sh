#!/usr/bin/env bash
# Queue v52q (3060 Ti, 11.11 afternoon, ~2.2 h, PRIORITY for the presentation, follows v52p): the bias forms 0.12 and
# 0.13 of the mined arm in full (v52p measured 0.11). Index cost of all forms: results/index_cost_v52p.json.
# Waits for v52p to exit, pauses the queue v52 again within a second (v52 is then inside its 10-second GPU wait, so
# no step of it starts), measures tiny10 + heldout8 of both forms first, then trec-covid of both, and resumes v52.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v52q.sh > logs/queue_v52q.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v52q MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v52q.pid"
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
kscript() {  # kscript TAG TIMEOUT OUTPUT COMMAND...: a GPU helper script; OUTPUT is removed when it fails
    local tag="$1" to="$2" out="$3"; shift 3
    wait_gpu_free; wait_ram; local s=$SECONDS rc
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { rm -f "$out"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/$tag.log" | cut -c1-300)"
    record "chain:$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
soup() {
    local name="$1"; shift
    [ -d "models/$name" ] && return 0
    for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
    wait_gpu_free
    timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/soup_$name.log" 2>&1 \
        && log "soup $name written ($*, uniform)" || { log "soup $name FAILED"; rm -rf "models/$name"; return 1; }
}
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
A=datasets/arms
ROWS=$A/n5-plus-kj-tens400.jsonl
PD=OPI-PIB/PolDense-400M
DCACHE=cache/dense_pool/pd400
BASECFG="$C/distil_dense_kl_n5kj_mined_ens400_r2v12.yaml"

refresh() {  # refresh TAG STUDENT ROWS OUT [tool args]: the student mines, the ensemble teacher scores
    local tag="$1" student="$2" rows="$3" out="$4"; shift 4
    [ -s "$out" ] && return 0
    kscript "refresh_$tag" 7200 "$out" "$PY" "$KITCHEN/bridge/splade_mine_and_score.py" --student "$student" --splade-teacher models/soup-v9b \
        --dense-teacher "$PD" --dense-cache "$DCACHE" --rows "$rows" --output "$out" "$@"
}
side() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]+'.mined.json'));print(d[sys.argv[2]])" "$1" "$2" 2>/dev/null; }
at_cost() {  # at_cost MODEL REFLINE: first bias form with <= 410 dims/doc -> tiny10, heldout8, trec-covid; PICK holds its label
    local m="$1" ref="$2" t; PICK="$m"
    [ -d "models/$m" ] || { log "$m missing: not measured"; return 1; }
    sparsity "$m" "models/$m"
    if [ "$(gte 410 "$(dims "$m")")" != 1 ]; then
        for t in 005 010 015 020; do
            biased "models/$m" "$m-bias$t" "0.${t#0}" || return 1
            sparsity "$m-bias$t" "models/$m-bias$t"; PICK="$m-bias$t"
            [ "$(gte 410 "$(dims "$PICK")")" = 1 ] && break
        done
        for t in 005 010 015 020; do [ "$m-bias$t" != "$PICK" ] && [ "$m" != "$M" ] && rm -rf "models/$m-bias$t"; done
    fi
    log "  $m: $(dims "$m") dims/doc unbiased; measured form $PICK at $(dims "$PICK")"
    measure "$PICK" "models/$PICK" "$ref"
    h2task "$PICK" heldout2_treccovid; log "  $PICK trec-covid $(tc "$PICK") (mined arm full 71.48, r2v12 67.02), $(dims "$PICK") dims/doc"
}
REF="mined arm bias010 $(lbs $M-bias010) at $(dims $M-bias010); bias015 $(lbs $M-bias015) at $(dims $M-bias015); r2v12 64.20 / 43.15 at 404"
HOLD=$(cat "$LOGS/queue_v52.pid" 2>/dev/null)
PREV=$(cat "$LOGS/queue_v52p.pid" 2>/dev/null)
while [ -n "$PREV" ] && ps -p "$PREV" -o args= 2>/dev/null | grep -q queue_v52p.sh; do sleep 1; done
if [ -n "$HOLD" ] && ps -p "$HOLD" -o args= 2>/dev/null | grep -q queue_v52.sh; then
    kill -STOP "$HOLD" && log "queue v52 (pid $HOLD) paused again"
    trap 'kill -CONT "$HOLD" 2>/dev/null; log "queue v52 (pid $HOLD) resumed"' EXIT
fi
log "=== v52q start: bias forms 0.12 and 0.13 of the mined arm ==="
for t in 012 013; do biased "models/$M" "$M-bias$t" "0.${t#0}"; done
for t in 012 013; do measure "$M-bias$t" "models/$M-bias$t" "bias011 $(lbs $M-bias011); $REF"; done
for t in 012 013; do
    h2task "$M-bias$t" heldout2_treccovid
    log "  $M-bias$t: $(lbs "$M-bias$t") / trec-covid $(tc "$M-bias$t") at $(dims "$M-bias$t") dims/doc (bias010 64.63 / 43.45 / 70.15 at 405; bias011 $(lbs $M-bias011) / $(tc $M-bias011); bias015 64.67 / 43.35 / 69.86 at 358)"
done
for t in 012 013; do git add "$REGIMES_ROOT/quantized/$M-bias$t" "$RESULTS_ROOT/sparsity/$M-bias$t.json" 2>/dev/null; done
git add "$EXP_ROOT/queues/queue_v52q.sh" 2>/dev/null
git commit -q -m "Add the bias forms 0.12 and 0.13 of the student-mined arm" 2>/dev/null && log "committed $(git log --oneline -1)"
log "=== v52q done ($(date '+%d.%m %H:%M')) ==="
