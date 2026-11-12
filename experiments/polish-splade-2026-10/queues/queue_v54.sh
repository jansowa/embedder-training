#!/usr/bin/env bash
# Queue v54 (3060 Ti, morning 12.11 after v52, v53 and s9, ~5.5 h): two questions asked by the user on 11.11.
#   1  JOINT TRAINING OF THE FULL SPLADE AND THE STATIC QUERY TABLE (phr2-both): v49 trained only the document
#      encoder against a frozen table (phr2-frozen: static queries over the joint index 59.87 against 60.20 over the
#      untouched index of r2v12-bias005; the document model alone 63.84 against 64.21). Here both sides move
#      (distill.train_joint_static without --freeze-table, same rows, steps and rates). Measured: the static table
#      over the jointly trained index (tiny10; heldout8 when >= 61) and the document model alone (tiny10, dims/doc).
#   2  QUERY PREFIX: the mined arm (64.63 / 43.45 / 70.15 at 405 dims/doc) trained with "zapytanie: " in front of
#      every query (sentence_transformers.query_prefix; documents without a prefix) and measured with the same prefix
#      (PIRB_QUERY_PREFIX -> q_prefix of the PIRB SPLADE encoder), in its first bias form with at most 410 dims/doc
#      (tiny10, heldout8, trec-covid); tiny10 once more without the prefix as a control.
#      Reading: >= +0.2 on both sets = the role signal pays; less = not worth a prefix every user has to remember.
#   3  summary_v54.md, commit, push loop.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v54.sh > logs/queue_v54.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v54 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v54.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v52 v53 s9}; do
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
log "=== v54 start: mined arm $(lbs $M) at $(dims $M) dims/doc, bias010 $(lbs $M-bias010) at $(dims $M-bias010), bias015 $(lbs $M-bias015) at $(dims $M-bias015); $(free_gb) GB free ==="

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
R2B="models/$R2-bias005"
TABLE=models/static-r2v12b/phr2-k8-kd-r2

# 1. joint training of the document encoder and the static query table
log "=== v54 step 1: joint training of the full SPLADE and the static query table (phr2-both) ==="
joint() {  # joint NAME [extra args]: train, copy to models/joint-r2v12b-NAME/{doc,query}, asym tiny10 (+ heldout8 when >= 61), symmetric tiny10 of the doc model
    local name="$1"; shift
    local dir="models/joint-r2v12b-$name"
    if [ ! -f "$dir/doc/model.safetensors" ]; then
        need_space || return 1; wait_gpu_free; wait_ram; local s=$SECONDS; log "joint $name ($*)"
        timeout -k 60 "$TRAIN_TIMEOUT" "$PY" -m distill.train_joint_static --rows "$ROWS" --doc-model "$R2B" --table "$TABLE" --out "runs/joint-r2v12b-$name" \
            --epochs 2 --rows-per-step 2 --grad-accum 16 --lr 2e-5 --table-lr 5e-3 --temperature 2.0 --doc-flops 1e-3 --query-flops 5e-5 \
            --val-rows 400 --eval-every 200 --log-every 20 --seed 42 "$@" > "$LOGS/joint_$name.log" 2>&1
        if [ -f "runs/joint-r2v12b-$name/final/doc/model.safetensors" ]; then
            mkdir -p "$dir"; cp -r "runs/joint-r2v12b-$name/final/doc" "$dir/doc"; cp -r "runs/joint-r2v12b-$name/final/query" "$dir/query"
            cp "runs/joint-r2v12b-$name/joint_summary.json" "$SRES/joint/$name.json"
            log "joint $name done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$SRES/joint/$name.json'));v=d['val_final'];print('val top1 %.3f pos-first %.3f dims/doc %.1f dims/q %.1f' % (v['top1_agreement'], v['positive_first'], v['dims_per_doc'], v['dims_per_query']))")"
            record "joint:r2v12b:$name" ok "-" "$((SECONDS - s))s"
        else log "joint $name FAILED (logs/joint_$name.log)"; record "joint:r2v12b:$name" FAILED "-" "$((SECONDS - s))s"; return 1; fi
    fi
    export PIRB_QUERY_MODEL="$dir/query"
    small regime quantized "asym-joint-$name" "$dir/doc" tiny10 16
    local v; v=$(slbv "asym-joint-$name" tiny10)
    log "  static table over the jointly trained index ($name): tiny10 $v (frozen index: phr2-k8-kd-r2 60.20; symmetric r2v12-bias005 64.21)"
    if [ "$(gte "$v" 61.0)" = 1 ]; then small regime quantized "asym-joint-$name" "$dir/doc" heldout8 16
        log "  asym-joint-$name heldout8 $(slbv "asym-joint-$name" heldout8) (frozen index 38.6; symmetric 43.08)"
    else log "  asym-joint-$name tiny10 $v < 61.0: no heldout8"; fi
    unset PIRB_QUERY_MODEL
    small sparsity "joint-$name-doc" "$dir/doc"
    small regime quantized "joint-$name-doc-sym" "$dir/doc" tiny10 16
    log "  joint-$name doc model symmetric tiny10 $(slbv "joint-$name-doc-sym" tiny10) (r2v12-bias005 64.21), $(sparsity_mean "$SRES/sparsity/joint-$name-doc.json" "joint-$name-doc" 2>/dev/null) dims/doc"
}
if [ -d "$TABLE" ] && [ -d "$R2B" ]; then joint phr2-both; else log "  $TABLE or $R2B missing: joint step skipped"; fi

# 2. query prefix
log "=== v54 step 2: the mined arm with a query prefix ==="
QP=arm-dense-kl-n5kj-mined-qp-ens400-r2v12
PREFIX="zapytanie: "
sed -e "4s|.*|# The mined arm with a query prefix (queue v54): every query starts with \"$PREFIX\" in training (documents without a prefix); evaluation must use the same prefix. Otherwise the mined arm: negatives mined once by r2v12 from the rows' own passages, ensemble teacher (soup-v9b + PolDense-400M), the r2v12 recipe from soup-v12.|" \
    -e "s|^runs_dir: .*|runs_dir: runs/$QP|" -e "/^sentence_transformers:/a\\  query_prefix: \"$PREFIX\"" "$BASECFG" > "$C/distil_dense_kl_n5kj_mined_qp_ens400_r2v12.yaml"
train "${QP#arm-}" "$C/distil_dense_kl_n5kj_mined_qp_ens400_r2v12.yaml"
if [ -d "models/$QP" ]; then
    export PIRB_QUERY_PREFIX="$PREFIX"
    at_cost "$QP" "$REF"; QP_PICK=$PICK
    unset PIRB_QUERY_PREFIX
    regime quantized "$QP_PICK-noprefix" "models/$QP_PICK" tiny10 16
    log "  $QP_PICK: with the prefix $(lbs "$QP_PICK") / trec-covid $(tc "$QP_PICK") at $(dims "$QP_PICK") dims/doc; tiny10 without the prefix $(lbv "$QP_PICK-noprefix" tiny10); mined arm bias010 64.63 / 43.45 / 70.15 at 405"
fi

# 3. summary
log "=== v54 step 3: summary ==="
SUM="$RESULTS_ROOT/summary_v54.md"
{
    echo "# v54 (morning 12.11, quantized regime, nDCG@10): joint training with the static query table, a query prefix"; echo
    echo "## Query prefix \"$PREFIX\" on the mined arm (trained and measured with the prefix on queries only; first bias form with at most 410 dims/doc)"; echo
    echo "| model | tiny10 | heldout8 | trec-covid | dims/doc |"; echo "|---|---:|---:|---:|---:|"
    for m in $M-bias010 $M-bias015 ${QP_PICK:-}; do
        [ -n "$(lbv $m tiny10)" ] && echo "| $m | $(lbv $m tiny10) | $(lbv $m heldout8) | $(tc $m) | $(dims $m) |"
    done
    [ -n "${QP_PICK:-}" ] && echo "| $QP_PICK, queries without the prefix | $(lbv "$QP_PICK-noprefix" tiny10) | - | - | $(dims "$QP_PICK") |"
    echo; echo "## Static query table phr2-k8-kd-r2 over a jointly trained index (untouched index of r2v12-bias005: 60.20; that model alone 64.21)"; echo
    echo "| joint arm | static queries, tiny10 | static queries, heldout8 | document model alone, tiny10 | dims/doc |"; echo "|---|---:|---:|---:|---:|"
    for n in phr2-frozen phr2-both; do
        echo "| $n | $(slbv "asym-joint-$n" tiny10) | $(slbv "asym-joint-$n" heldout8) | $(slbv "joint-$n-doc-sym" tiny10) | $(sparsity_mean "$SRES/sparsity/joint-$n-doc.json" "joint-$n-doc" 2>/dev/null) |"
    done
    [ -s "$SRES/joint/phr2-both.json" ] && { echo; echo '```'; "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(d.get('val_final'), indent=2))" "$SRES/joint/phr2-both.json"; echo '```'; }
    echo; echo "## Queue state (v54)"; echo; echo '```'; grep -E "mined-qp|phr2-both" "$STATE"; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_v54.sh" "$C/distil_dense_kl_n5kj_mined_qp_ens400_r2v12.yaml" scripts/pirb_regime_eval.sh 2>/dev/null
for m in ${QP_PICK:-} ${QP_PICK:+$QP_PICK-noprefix} $QP; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$QP-bias0??.json "$SREG/quantized/asym-joint-phr2-both" "$SREG/quantized/joint-phr2-both-doc-sym" "$SRES/sparsity/joint-phr2-both-doc.json" "$SRES/joint/phr2-both.json" 2>/dev/null
git commit -q -m "Add the v54 results: joint training of SPLADE with the static query table and a query prefix" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v54.log" 2>&1 && { echo pushed >> "$LOGS/push_v54.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v54 done ($(date '+%d.%m %H:%M')) ==="
