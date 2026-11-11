#!/usr/bin/env bash
# Queue v52p (3060 Ti, 11.11 afternoon, ~70 min, PRIORITY for the presentation): an intermediate bias form of the
# mined arm. bias010 = 64.63 / 43.45 / 70.15 at 16 461 postings/query (polish-splade 16 415); bias015 = 64.67 / 43.35 /
# 69.86 at 13 272. Wanted: postings clearly below polish-splade with as little loss as possible.
# The queue v52 is paused with SIGSTOP while its current training keeps running (nothing is interrupted); this script
# waits for the GPU, measures and resumes v52 with SIGCONT (also on any exit).
#   1  bias forms 0.11, 0.12, 0.13: index cost on the cost sample (1 350 passages + 1 350 queries) next to base,
#      bias010 and bias015 -> results/index_cost_v52p.json
#   2  the smallest of them with at most 16 000 postings/query: sparsity, tiny10, heldout8, trec-covid
#   3  commit
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v52p.sh > logs/queue_v52p.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v52p MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v52p.pid"
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
log "=== v52p start: mined arm $(lbs $M) at $(dims $M) dims/doc, bias010 $(lbs $M-bias010) at $(dims $M-bias010), bias015 $(lbs $M-bias015) at $(dims $M-bias015); $(free_gb) GB free ==="

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
if [ -n "$HOLD" ] && ps -p "$HOLD" -o args= 2>/dev/null | grep -q queue_v52.sh; then
    kill -STOP "$HOLD" && log "queue v52 (pid $HOLD) paused; its running step continues"
    trap 'kill -CONT "$HOLD" 2>/dev/null; log "queue v52 (pid $HOLD) resumed"' EXIT
fi
COST="$RESULTS_ROOT/index_cost_v52p.json"
for t in 011 012 013; do biased "models/$M" "$M-bias$t" "0.${t#0}"; done
if [ ! -s "$COST" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    args=(--model base=models/base-splade-wrapped)
    for t in 010 011 012 013 015; do [ -d "models/$M-bias$t" ] && args+=(--model "$M-bias$t=models/$M-bias$t"); done
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" --output-json "$COST" > "$LOGS/index_cost_v52p.log" 2>&1 \
        && log "index cost written ($COST) in $((SECONDS - s))s" || { log "index cost FAILED (logs/index_cost_v52p.log)"; rm -f "$COST"; }
fi
[ -s "$COST" ] && "$PY" - "$COST" <<'EOF' | while read -r l; do log "  $l"; done
import json, sys
d = json.load(open(sys.argv[1]))
for name, s in d["models"].items():
    print("%s: dims/doc %.1f, dims/query %.1f (median %.0f), postings/query %.0f" % (name, s["doc_active_mean"], s["query_active_mean"], s["query_active_median"], s["postings_per_query"]))
EOF
PICK=$("$PY" - "$COST" "$M" <<'EOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))["models"]
for t in ("011", "012", "013"):
    s = d.get(f"{sys.argv[2]}-bias{t}")
    if s and s["postings_per_query"] <= 16000:
        print(f"{sys.argv[2]}-bias{t}"); break
EOF
)
PICK=${PICK:-$M-bias012}
log "=== v52p: measured form $PICK (the smallest bias with at most 16 000 postings/query) ==="
measure "$PICK" "models/$PICK" "$REF"
h2task "$PICK" heldout2_treccovid
log "  $PICK: $(lbs "$PICK") / trec-covid $(tc "$PICK") at $(dims "$PICK") dims/doc (bias010 64.63 / 43.45 / 70.15 at 405; bias015 64.67 / 43.35 / 69.86 at 358)"
for t in 011 012 013; do [ "$M-bias$t" != "$PICK" ] && rm -rf "models/$M-bias$t"; done
git add "$COST" "$EXP_ROOT/queues/queue_v52p.sh" "$REGIMES_ROOT/quantized/$PICK" "$RESULTS_ROOT/sparsity/$PICK.json" 2>/dev/null
git commit -q -m "Add an intermediate bias form of the student-mined arm with its index cost" 2>/dev/null && log "committed $(git log --oneline -1)"
log "=== v52p done ($(date '+%d.%m %H:%M')) ==="
