#!/usr/bin/env bash
# Queue v53 (3060 Ti, evening 11.11 after v52, ~4.5 h): the student-mined arm with a LARGE candidate pool.
# The mined arm (v49/v50: 64.63 / 43.45 / trec-covid 70.15 at 405 dims/doc with bias 0.10) took its negatives from the
# rows' own passages: 11.6 k (fiqa), 18.6 k (SQuAD-v2) and 24.4 k (kj) texts. v52 showed that re-mining during training
# hardly changes the lists' difficulty. Here the pool grows instead: every group also searches EXTRA_MAX (1 M) passages
# sampled from the distillation corpus datasets/kd/kd_v1.jsonl (msmarco, nq, hotpotqa, fever, wiki chunks, fiqa pool;
# passages only, 50-3000 characters, texts of the own pools removed). Everything else as in the mined arm: miner r2v12,
# 5 best + 5 random of the top-60 inside the length window, ensemble teacher soup-v9b + PolDense-400M with the dense
# scale of the v52 round 1, 2 epochs from soup-v12 with the r2v12 recipe.
#   1  SMOKE of the kitchen tool bridge/splade_mine_big_pool.py (300 rows, 20 k shared texts); a failure ends the queue.
#   2  MINING over the large pool -> datasets/arms/n5-plus-kj-minedxl.jsonl (the sidecar has the share of negatives
#      that come from the shared pool and the teacher's P(pos)).
#   3  ARM arm-dense-kl-n5kj-minedxl-ens400-r2v12, measured in its first bias form with at most 410 dims/doc
#      (tiny10, heldout8, trec-covid). Reading against the mined arm at equal cost (and its seed 43 from v52):
#      >= +0.2 on both sets = candidates are still the limit; equal = the own pools are enough.
#   4  summary_v53.md, commit, push loop.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v53.sh > logs/queue_v53.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v53 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v53.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v52}; do
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
log "=== v53 start: mined arm $(lbs $M) at $(dims $M) dims/doc, bias010 $(lbs $M-bias010) at $(dims $M-bias010), bias015 $(lbs $M-bias015) at $(dims $M-bias015); $(free_gb) GB free ==="

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
REF="mined arm seed 43: $(lbs $M-s43-bias010) at $(dims $M-s43-bias010); mined arm bias010 $(lbs $M-bias010) at $(dims $M-bias010); bias015 $(lbs $M-bias015) at $(dims $M-bias015); r2v12 64.20 / 43.15 at 404"
XL=arm-dense-kl-n5kj-minedxl-ens400-r2v12
XLROWS=$A/n5-plus-kj-minedxl.jsonl
EXTRA=datasets/kd/kd_v1.jsonl
EXTRA_MAX=${EXTRA_MAX:-1000000}
SCALE=$(side "$A/n5-plus-kj-iter1.jsonl" dense_scale); SCALE=${SCALE:-84.2}
bigpool() {  # bigpool TAG OUT [tool args]: r2v12 mines from the own pools + the shared pool, the ensemble teacher scores
    local tag="$1" out="$2"; shift 2
    [ -s "$out" ] && return 0
    kscript "bigpool_$tag" 14400 "$out" "$PY" "$KITCHEN/bridge/splade_mine_big_pool.py" --student "models/$R2" --splade-teacher models/soup-v9b \
        --dense-teacher "$PD" --rows "$ROWS" --extra-pool "$EXTRA" --dense-scale "$SCALE" --output "$out" "$@"
}

# 1. smoke
log "=== v53 step 1: smoke of the big-pool tool (300 rows, 20 000 shared texts) ==="
[ -s "$EXTRA" ] || { log "  $EXTRA missing (is the data drive mounted?): v53 stops"; exit 1; }
rm -rf cache/smoke_v53; mkdir -p cache/smoke_v53
if bigpool smoke cache/smoke_v53/rows.jsonl --limit 300 --extra-max 20000 --extra-shard 8000 --dense-cache cache/smoke_v53/dense \
    && [ "$(grep -c . cache/smoke_v53/rows.jsonl)" = 300 ]; then
    log "  smoke passed: negatives from the shared pool $(side cache/smoke_v53/rows.jsonl negatives_from_shared_pool), teacher P(pos) median $(side cache/smoke_v53/rows.jsonl teacher_positive_mass_median)"
else log "  smoke FAILED (logs/bigpool_smoke.log): v53 stops"; rm -rf cache/smoke_v53; exit 1; fi
rm -rf cache/smoke_v53

# 2. mining over the large pool
log "=== v53 step 2: mining over the own pools + $EXTRA_MAX texts of $EXTRA (dense scale $SCALE) ==="
bigpool full "$XLROWS" --extra-max "$EXTRA_MAX" --dense-cache "$DCACHE"
if [ -s "$XLROWS" ]; then
    log "  mined: negatives from the shared pool $(side "$XLROWS" negatives_from_shared_pool) ($(side "$XLROWS" negatives_by_pool)); teacher P(pos) median $(side "$XLROWS" teacher_positive_mass_median) (own pools 0.852); negative above positive in $(side "$XLROWS" rows_with_negative_above_positive) rows (own pools 2663); student ranks a positive first in $(side "$XLROWS" student_ranks_a_positive_first) (own pools 0.844); teacher $(side "$XLROWS" teacher_ranks_a_positive_first)"
    # 3. the arm
    log "=== v53 step 3: arm $XL ==="
    sed -e "4s|.*|# The mined arm with a large candidate pool (queue v53): negatives mined once by r2v12 from the rows' own passages plus $EXTRA_MAX passages of the distillation corpus kd_v1 (5 best + 5 random of its top-60, length window 0.5-2.0), scored by the ensemble teacher (soup-v9b + PolDense-400M). The r2v12 recipe from soup-v12.|" \
        -e "s|^runs_dir: .*|runs_dir: runs/$XL|" -e "s|^train_data: .*|train_data: $XLROWS|" "$BASECFG" > "$C/distil_dense_kl_n5kj_minedxl_ens400_r2v12.yaml"
    train "${XL#arm-}" "$C/distil_dense_kl_n5kj_minedxl_ens400_r2v12.yaml"
    at_cost "$XL" "$REF"; XL_PICK=$PICK
else log "  mining FAILED (logs/bigpool_full.log): arm skipped"; fi

# 4. summary
log "=== v53 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_v53.md"
{
    echo "# v53 (evening 11.11, quantized regime, nDCG@10): the student-mined arm with a large candidate pool"; echo
    echo "minedxl = the mined arm (v49) with negatives chosen by r2v12 from the rows' own passages plus $EXTRA_MAX passages sampled from the distillation corpus kd_v1; same selection rule, teacher and recipe. New arms are measured in their first bias form with at most 410 dims/doc (biasT = lm_head.bias shifted by -(e^T - 1))."; echo
    echo "| model | tiny10 | heldout8 | trec-covid | dims/doc |"; echo "|---|---:|---:|---:|---:|"
    for m in $R2 $M $M-bias010 $M-s43-bias010 ${XL_PICK:-}; do
        [ -n "$(lbv $m tiny10)" ] && echo "| $m | $(lbv $m tiny10) | $(lbv $m heldout8) | $(tc $m) | $(dims $m) |"
    done
    [ -s "$XLROWS.mined.json" ] && { echo; echo "## Mining over the large pool (own pools: teacher P(pos) median 0.852, negative above positive in 2663 rows, student ranks a positive first in 0.844)"; echo; echo '```'; cat "$XLROWS.mined.json"; echo '```'; }
    echo; echo "## Queue state (v53)"; echo; echo '```'; grep -E "minedxl|chain:bigpool" "$STATE"; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_v53.sh" "$C/distil_dense_kl_n5kj_minedxl_ens400_r2v12.yaml" 2>/dev/null
for m in ${XL_PICK:-} $XL; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$XL-bias0??.json 2>/dev/null
git commit -q -m "Add the v53 results: the student-mined arm with a large candidate pool" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v53.log" 2>&1 && { echo pushed >> "$LOGS/push_v53.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v53 done ($(date '+%d.%m %H:%M')) ==="
