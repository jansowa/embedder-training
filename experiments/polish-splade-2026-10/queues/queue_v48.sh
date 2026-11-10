#!/usr/bin/env bash
# Queue v48 (3060 Ti, 10.11 afternoon/evening, after v47, ~7 h; steps start only before 10.11 19:00): follow-ups that
# depend on v47's readings, in the order of expected value.
#   1  when the three-way teacher wins (ens3 heldout8 >= r2v12 43.15): ens3 seed 43 (85 min) + soup-v18 = uniform(s42, s43)
#      -> sparsity, tiny10, heldout8, trec-covid; the other four held-out-2 tasks when trec-covid >= 69.0 (the publication
#      candidate gets its 23 tasks). Otherwise: the joint arm with both sides trained gets a longer run (4 epochs) when its
#      asym tiny10 held >= 61.0; else the ALBERT stage 3 seed 43 + a soup of the two students (noise floor of 63.72).
#   2  the ALBERT student as the DOCUMENT side of the static pipeline: joint training of arm-hl-r2-s3n5kj-bias010 with the
#      frozen phr2 table (the cheapest end-to-end stack: 50 M documents, lookup queries) -> asym tiny10 (+ heldout8 >= 61),
#      symmetric tiny10, dims/doc.
#   3  when ens3 wins: KD continuation of the student from the ens3 full model (600 k texts) + stage 3 on ens3 rows ->
#      sparsity, tiny10, heldout8.
#   4  summary_v48.md, commit, push loop.
#   WAIT_FOR="v47" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v48.sh > logs/queue_v48.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v48 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-14400} KD_TIMEOUT=${KD_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v48.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
DEADLINE=$(date -d '2026-11-10 19:00' +%s)
late() { [ "$(date +%s)" -ge "$DEADLINE" ] && { log "past the 10.11 19:00 deadline: $1 skipped"; return 0; }; return 1; }

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v47}; do
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
measure() {
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    sparsity "$label" "$model"; regime quantized "$label" "$model" tiny10 16; regime quantized "$label" "$model" heldout8 16
    log "  $label: $(lbs "$label") ($ref), $(dims "$label") dims/doc"
}
smeasure() {
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    small sparsity "$label" "$model"; small regime quantized "$label" "$model" tiny10 16; small regime quantized "$label" "$model" heldout8 16
    log "  $label: $(slbs "$label") ($ref), $(sparsity_mean "$SRES/sparsity/$label.json" "$label" 2>/dev/null) dims/doc"
}
soup() {
    local name="$1"; shift
    [ -d "models/$name" ] && return 0
    for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
    wait_gpu_free
    timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/soup_$name.log" 2>&1 \
        && log "soup $name written ($*, uniform)" || { log "soup $name FAILED"; rm -rf "models/$name"; return 1; }
}
joint() {  # joint NAME DOC_MODEL TABLE [extra args]: as in v47, the document start and the table given explicitly
    local name="$1" docm="$2" table="$3"; shift 3
    local dir="models/joint-$name"
    if [ ! -f "$dir/doc/model.safetensors" ]; then
        need_space || return 1; wait_gpu_free; wait_ram; local s=$SECONDS; log "joint $name from $docm with $table ($*)"
        timeout -k 60 "$TRAIN_TIMEOUT" "$PY" -m distill.train_joint_static --rows datasets/arms/n5-plus-kj-tens400.jsonl --doc-model "$docm" --table "$table" --out "runs/joint-$name" \
            --rows-per-step 2 --grad-accum 16 --lr 2e-5 --table-lr 5e-3 --temperature 2.0 --doc-flops 1e-3 --query-flops 5e-5 \
            --val-rows 400 --eval-every 200 --log-every 20 --seed 42 "$@" > "$LOGS/joint_$name.log" 2>&1
        if [ -f "runs/joint-$name/final/doc/model.safetensors" ]; then
            mkdir -p "$dir"; cp -r "runs/joint-$name/final/doc" "$dir/doc"; cp -r "runs/joint-$name/final/query" "$dir/query"
            cp "runs/joint-$name/joint_summary.json" "$SRES/joint/$name.json"
            log "joint $name done in $((SECONDS - s))s"; record "joint:$name" ok "-" "$((SECONDS - s))s"
        else log "joint $name FAILED (logs/joint_$name.log)"; record "joint:$name" FAILED "-" "$((SECONDS - s))s"; return 1; fi
    fi
    export PIRB_QUERY_MODEL="$dir/query"
    small regime quantized "asym-joint-$name" "$dir/doc" tiny10 16
    local v; v=$(slbv "asym-joint-$name" tiny10)
    log "  static table over the jointly trained index ($name): tiny10 $v (frozen r2v12-bias005 index 60.20; v47 arms $(slbv asym-joint-phr2-frozen tiny10) / $(slbv asym-joint-phr2-both tiny10))"
    if [ "$(gte "$v" 61.0)" = 1 ]; then small regime quantized "asym-joint-$name" "$dir/doc" heldout8 16; log "  asym-joint-$name heldout8 $(slbv "asym-joint-$name" heldout8)"; fi
    unset PIRB_QUERY_MODEL
    small sparsity "joint-$name-doc" "$dir/doc"
    small regime quantized "joint-$name-doc-sym" "$dir/doc" tiny10 16
    log "  joint-$name doc model symmetric tiny10 $(slbv "joint-$name-doc-sym" tiny10), $(sparsity_mean "$SRES/sparsity/joint-$name-doc.json" "joint-$name-doc" 2>/dev/null) dims/doc"
}

R2=arm-dense-kl-n5kj-ens400-r2v12
ENS3=arm-dense-kl-n5kj-ens3-r2v12
A=datasets/arms
ENS3_H=$(lbv $ENS3 heldout8)
ENS3_WINS=$(gte "$ENS3_H" 43.15)
log "=== v48 start: $(free_gb) GB free; ens3 $(lbs $ENS3) trec-covid $(tc $ENS3) (r2v12 64.20 / 43.15 / 67.02) -> ens3 wins = $ENS3_WINS; joint arms $(slbv asym-joint-phr2-frozen tiny10) / $(slbv asym-joint-phr2-both tiny10) ==="

# 1. the branch by v47's reading
log "=== v48 step 1 ==="
if [ "$ENS3_WINS" = 1 ] && [ -s "$A/n5-plus-kj-tens3.jsonl" ]; then
    late "ens3 seed 43" || {
        train dense-kl-n5kj-ens3-r2v12-s43 "$C/distil_dense_kl_n5kj_ens3_r2v12_s43.yaml"
        measure $ENS3-s43 models/$ENS3-s43 "seed 42: $(lbs $ENS3)"
        soup soup-v18 "models/$ENS3" "models/$ENS3-s43" && measure soup-v18 models/soup-v18 "ens3 $(lbs $ENS3); soup-v14s 64.18 / 43.11; soup-v16 $(lbs soup-v16)"
        h2task soup-v18 heldout2_treccovid; log "  soup-v18 trec-covid $(tc soup-v18) (ens3 $(tc $ENS3), soup-v14s 70.00)"
        if [ "$(gte "$(tc soup-v18)" 69.0)" = 1 ]; then for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do late "held-out-2 $ts" || h2task soup-v18 "$ts"; done
        else log "  soup-v18 trec-covid $(tc soup-v18) < 69.0: no further held-out-2"; fi
    }
elif [ "$(gte "$(slbv asym-joint-phr2-both tiny10)" 61.0)" = 1 ]; then
    log "  ens3 does not win; the joint arm (both sides) held $(slbv asym-joint-phr2-both tiny10): a 4-epoch run"
    late "joint phr2-both 4 epochs" || joint r2v12b-phr2-both-ep4 "models/$R2-bias005" models/static-r2v12b/phr2-k8-kd-r2 --epochs 4
else
    log "  ens3 does not win and the joint arms stayed below 61: ALBERT stage 3 seed 43 + a soup of the two students"
    if ! late "student seed 43"; then
        sed -e 's#^runs_dir: .*#runs_dir: runs/arm-hl-r2-s3n5kj-s43#' -e '/^    loss: sparse_distill_kl_div/a\    seed: 43' "$SC/stage3_hl_r2_n5kj_ens400.yaml" > "$SC/stage3_hl_r2_n5kj_ens400_s43.yaml"
        small train hl-r2-s3n5kj-s43 "$SC/stage3_hl_r2_n5kj_ens400_s43.yaml"
        smeasure arm-hl-r2-s3n5kj-s43 models/arm-hl-r2-s3n5kj-s43 "seed 42: $(slbs arm-hl-r2-s3n5kj)"
        soup soup-hl-s3n5kj models/arm-hl-r2-s3n5kj models/arm-hl-r2-s3n5kj-s43 && smeasure soup-hl-s3n5kj models/soup-hl-s3n5kj "seeds $(slbs arm-hl-r2-s3n5kj), $(slbs arm-hl-r2-s3n5kj-s43)"
    fi
fi

# 2. the student as the document side of the static pipeline
log "=== v48 step 2: joint training of the ALBERT student (bias010) with the frozen phr2 table ==="
late "joint student" || joint hl-phr2-frozen models/arm-hl-r2-s3n5kj-bias010 models/static-r2v12b/phr2-k8-kd-r2 --epochs 2 --freeze-table

# 3. the student from the ens3 teacher
log "=== v48 step 3: KD continuation of the student from the ens3 full model ==="
if [ "$ENS3_WINS" = 1 ] && [ -d "models/$ENS3" ] && ! late "student KD from ens3"; then
    small kd hl-r2b models/kd-hl-r2 --teacher "models/$ENS3" --corpus "$CORPUS" --max-items 600000 --max-seq-length 256 --batch-size 16 \
        --lr 2e-5 --warmup-steps 200 --schedule linear --epochs 1 --precision fp16 --head-chunk 64 --save-every 5000 --keep-last 1 \
        --fidelity-texts "$CORPUS" --fidelity-n 256 --log-every 50 --token-loss kl --token-weight 0.3 --temperature 2.0 --top-k 128 --pooled-loss mse --pooled-weight 1.0 --seed 42
    small train hl-r2b-s3n5kj-ens3 "$SC/stage3_hl_r2b_n5kj_ens3.yaml"
    smeasure arm-hl-r2b-s3n5kj-ens3 models/arm-hl-r2b-s3n5kj-ens3 "from r2v12: arm-hl-r2-s3n5kj $(slbs arm-hl-r2-s3n5kj); on ens3 rows $(slbs arm-hl-r2-s3n5kj-ens3)"
else log "  skipped (ens3 wins = $ENS3_WINS)"; fi

# 4. summary
log "=== v48 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_v48.md"
{
    echo "# v48 (10.11, quantized regime, nDCG@10): follow-ups of v47 (ens3 wins = $ENS3_WINS)"; echo
    echo "## Full models"; echo
    "$PY" scripts/regime_task_table.py --models soup-v14s soup-v14s-bias005 $R2 $ENS3 $ENS3-s43 soup-v18 --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo; for m in $ENS3 $ENS3-s43 soup-v18; do [ -d "models/$m" ] && echo "- $m: trec-covid $(tc "$m"), czywiesz $(lbv "$m" heldout2_czywiesz), gpt35wiki $(lbv "$m" heldout2_gpt35wiki), quora $(lbv "$m" heldout2_quora), poquad $(lbv "$m" heldout2_poquad)"; done
    echo; echo "## ALBERT student"; echo
    small "$PY" scripts/regime_task_table.py --regime-dir "$SREG/quantized" --models arm-hl-r2-s3n5kj arm-hl-r2-s3n5kj-s43 soup-hl-s3n5kj arm-hl-r2-s3n5kj-ens3 arm-hl-r2b-s3n5kj-ens3 --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo; echo "## Joint arms (static table over a trained index)"; echo
    for name in phr2-frozen phr2-both r2v12b-phr2-both-ep4 hl-phr2-frozen; do
        [ -f "$SREG/quantized/asym-joint-$name/tiny10.json" ] && echo "- $name: asym tiny10 $(slbv asym-joint-$name tiny10), asym heldout8 $(slbv asym-joint-$name heldout8), doc symmetric tiny10 $(slbv joint-$name-doc-sym tiny10), $(sparsity_mean "$SRES/sparsity/joint-$name-doc.json" "joint-$name-doc" 2>/dev/null) dims/doc"
    done
    echo; echo "## Queue state (v48)"; echo; echo '```'; grep -E "ens3-r2v12-s43|soup-v18|joint:|hl-r2b|s3n5kj-s43|soup-hl" "$STATE"; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_v48.sh" "$C/distil_dense_kl_n5kj_ens3_r2v12_s43.yaml" "$SC/stage3_hl_r2b_n5kj_ens3.yaml" 2>/dev/null
for p in "$REGIMES_ROOT"/quantized/$ENS3-s43 "$REGIMES_ROOT"/quantized/soup-v18 "$RESULTS_ROOT"/sparsity/$ENS3-s43.json "$RESULTS_ROOT"/sparsity/soup-v18.json \
    "$SC/stage3_hl_r2_n5kj_ens400_s43.yaml" "$SREG"/quantized/arm-hl-r2-s3n5kj-s43 "$SREG"/quantized/soup-hl-s3n5kj "$SREG"/quantized/arm-hl-r2b-s3n5kj-ens3 \
    "$SREG"/quantized/asym-joint-hl-phr2-frozen "$SREG"/quantized/joint-hl-phr2-frozen-doc-sym "$SREG"/quantized/asym-joint-r2v12b-phr2-both-ep4 "$SREG"/quantized/joint-r2v12b-phr2-both-ep4-doc-sym \
    "$SRES"/sparsity "$SRES"/joint "$SRES"/fidelity/kd-hl-r2b.queries.json; do [ -e "$p" ] && git add "$p" 2>/dev/null; done
git commit -q -m "Add the v48 results: the follow-ups of the cross-encoder teacher and the joint static arms" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v48.log" 2>&1 && { echo pushed >> "$LOGS/push_v48.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v48 done ($(date '+%d.%m %H:%M')) ==="
