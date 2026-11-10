#!/usr/bin/env bash
# Queue v49a (3060 Ti, day 10.11 from ~06:00, ~4.5 h): v47 and v48 died with the machine shutdown of 9.11 22:38 inside
# v47 step 0. This queue restarts the v47 steps that serve a stronger full model, ahead of the static-table line:
#   0  the one heldout8 repair v47 did not finish (arm-hl-r2-s3n5kj fiqa-pl); the other four are already 8/8.
#   2  CROSS-ENCODER TEACHER (v47 step 2 unchanged): sdadas/polish-reranker-roberta-v3 logits on the n5 + kj rows,
#      ens3 = mean(v9b, pd400, reranker) -> arm-dense-kl-n5kj-ens3-r2v12: sparsity, tiny10, heldout8, trec-covid.
#      Reading against r2v12 64.20 / 43.15 / 67.02 and soup-v14s 64.18 / 43.11 / 70.00.
#   4  soup-v17 = soup-v12 + 1.5 (r2v12 - soup-v12): tiny10, trec-covid (v47 step 4 unchanged).
# The rest (student-mined candidates, FLOPS with a threshold, the reranker-only arm, joint static training, the ALBERT
# line, summary and commit) is queue v49b, which waits for this one.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v49a.sh > logs/queue_v49a.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v49a MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v49a.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
DEADLINE=$(date -d '2026-11-10 17:00' +%s)
late() { [ "$(date +%s)" -ge "$DEADLINE" ] && { log "past the 10.11 17:00 deadline: $1 skipped"; return 0; }; return 1; }

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
smeasure() {  # smeasure LABEL MODEL REFLINE (small series)
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    small sparsity "$label" "$model"; small regime quantized "$label" "$model" tiny10 16; small regime quantized "$label" "$model" heldout8 16
    log "  $label: $(slbs "$label") ($ref), $(sparsity_mean "$SRES/sparsity/$label.json" "$label" 2>/dev/null) dims/doc"
}
R2=arm-dense-kl-n5kj-ens400-r2v12
R2B="models/$R2-bias005"
log "=== v49a start: $(free_gb) GB free on / ==="

# 0. the unfinished heldout8 repair
log "=== v49a step 0: fiqa-pl for arm-hl-r2-s3n5kj ==="
refill() {  # refill ROOT LABEL MODEL TASK...: run each single task set and fold it into heldout8.json
    local root="$1" label="$2" model="$3"; shift 3
    local ts srcs=()
    for ts in "$@"; do REGIMES_ROOT=$root regime quantized "$label" "$model" "$ts" 16 && srcs+=("$root/quantized/$label/$ts.json"); done
    [ ${#srcs[@]} -gt 0 ] && "$PY" scripts/merge_regime_task.py --into "$root/quantized/$label/heldout8.json" --from "${srcs[@]}" --n-tasks 8 | while read -r l; do log "  $l"; done
}
refill "$SREG" arm-hl-r2-s3n5kj models/arm-hl-r2-s3n5kj heldout8_fiqa
log "  heldout8 (8/8): hl-r2-s3n5kj $(slbv arm-hl-r2-s3n5kj heldout8) (hl-full-s3n5kj 42.55, hl-full-s3 42.10)"

ROWS=datasets/arms/n5-plus-kj-tens400.jsonl
# 2. the cross-encoder teacher
log "=== v49a step 2: cross-encoder teacher scores (sdadas/polish-reranker-roberta-v3) on n5 + kj ==="
A=datasets/arms
[ -s "$A/n5-plus-kj-tv9b.jsonl" ] || cat "$A/fiqa-plus-squadv2v2-18k-tv9b.jsonl" "$A/karoljez-2219-tv9b.jsonl" > "$A/n5-plus-kj-tv9b.jsonl"
[ -s "$A/n5-plus-kj-tpd400.jsonl" ] || cat "$A/fiqa-plus-squadv2v2-18k-tpd400.jsonl" "$A/karoljez-2219-tpd400.jsonl" > "$A/n5-plus-kj-tpd400.jsonl"
[ -s "$A/n5-plus-kj.pairs.parquet" ] || "$PY" scripts/reranker_rows.py pairs --rows "$ROWS" --out "$A/n5-plus-kj.pairs.parquet" | while read -r l; do log "  $l"; done
if [ ! -s "$A/n5-plus-kj.rr.parquet" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS; log "reranker scoring of n5 + kj pairs"
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 14400 "$PY" "$KITCHEN/bridge/reranker_pair_scores.py" --input "$A/n5-plus-kj.pairs.parquet" --output "$A/n5-plus-kj.rr.parquet" --batch-size 32 > "$LOGS/rerank_n5kj.log" 2>&1
    [ -s "$A/n5-plus-kj.rr.parquet" ] && { log "reranker scoring done in $((SECONDS - s))s"; record "rerank:n5kj" ok "-" "$((SECONDS - s))s"; } \
        || { log "reranker scoring FAILED (logs/rerank_n5kj.log)"; record "rerank:n5kj" FAILED "-" "$((SECONDS - s))s"; }
fi
if [ -s "$A/n5-plus-kj.rr.parquet" ]; then
    [ -s "$A/n5-plus-kj-trr.jsonl" ] || "$PY" scripts/reranker_rows.py merge --rows "$ROWS" --scores "$A/n5-plus-kj.rr.parquet" --reference "$A/n5-plus-kj-tv9b.jsonl" --out "$A/n5-plus-kj-trr.jsonl" > "$LOGS/rerank_merge.log" 2>&1 \
        && log "  reranker rows: $(tr -d '\n' < "$A/n5-plus-kj-trr.jsonl.teacher.json" | tr -s ' ')"
    [ -s "$A/n5-plus-kj-tens3.jsonl" ] || "$PY" scripts/reranker_rows.py ensemble --inputs "$A/n5-plus-kj-tv9b.jsonl" "$A/n5-plus-kj-tpd400.jsonl" "$A/n5-plus-kj-trr.jsonl" --out "$A/n5-plus-kj-tens3.jsonl" | while read -r l; do log "  ens3 $l"; done
fi
ENS3=arm-dense-kl-n5kj-ens3-r2v12; TRR=arm-dense-kl-n5kj-trr-r2v12
if [ -s "$A/n5-plus-kj-tens3.jsonl" ]; then
    train dense-kl-n5kj-ens3-r2v12 "$C/distil_dense_kl_n5kj_ens3_r2v12.yaml"
    measure $ENS3 models/$ENS3 "r2v12 (two-way teacher) 64.20 / 43.15; soup-v14s 64.18 / 43.11"
    h2task $ENS3 heldout2_treccovid; log "  $ENS3 trec-covid $(tc $ENS3) (r2v12 67.02, soup-v14s 70.00)"
fi

# 4. task-vector extrapolation
log "=== v49a step 4: soup-v17 = soup-v12 + 1.5 (r2v12 - soup-v12) ==="
if [ ! -d models/soup-v17 ] && ! late "soup-v17"; then
    cp -r models/soup-v12 models/soup-v17 && "$PY" - models/soup-v12 "models/$R2" models/soup-v17 1.5 <<'EOF' && log "soup-v17 written" || { log "soup-v17 FAILED"; rm -rf models/soup-v17; }
import sys, torch
from safetensors.torch import load_file, save_file
base, arm, out, alpha = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
b, a = load_file(base + "/model.safetensors"), load_file(arm + "/model.safetensors")
t = {k: (b[k].float() + alpha * (a[k].float() - b[k].float())).to(b[k].dtype) if k in a and b[k].dtype.is_floating_point else b[k] for k in b}
save_file(t, out + "/model.safetensors", metadata={"format": "pt"})
print("extrapolated", len(t), "tensors, alpha", alpha)
EOF
fi
if [ -d models/soup-v17 ]; then
    sparsity soup-v17 models/soup-v17; regime quantized soup-v17 models/soup-v17 tiny10 16; h2task soup-v17 heldout2_treccovid
    log "  soup-v17: tiny10 $(lbv soup-v17 tiny10), trec-covid $(tc soup-v17) (soup-v15 at 0.5: 64.20 / 68.91; r2v12 64.20 / 67.02), $(dims soup-v17) dims/doc"
fi
log "=== v49a done ($(date '+%d.%m %H:%M')) ==="
