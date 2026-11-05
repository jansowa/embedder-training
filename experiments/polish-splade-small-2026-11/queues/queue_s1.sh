#!/usr/bin/env bash
# Queue s1 (3060 Ti; DRAFT, not started): the small-SPLADE probe night (docs/PLAN-2026-11-04.md §5).
# Does representation distillation on unlabeled text (token logits, then pooled vectors) turn a pruned
# soup-v9b into a usable student, and does a fresh body with the inherited vocabulary come anywhere close?
#   0  reference results copied into this series' results-regimes (base, arm-n4-3060, arm-distil-v9b-kl-n4, soup-v9b)
#   1  control: student b4-l0125-f1536 (4 layers, FFN 1536, no KD) -> stage 3 only (ranking KL on n4 rows) -> tiny10
#   2  b4-l0125-f1536: stage 1 (token KL, KD_ITEMS texts) -> stage 2 (pooled MSE) -> stage 3 -> sparsity + tiny10
#   3  b4-l0125-f1536-hl-r256-w: the same body as an ALBERT with the head layer and the rank-256 vocabulary
#      (orthonormal input projection weighted by kd_v1 token counts, untied decoder fitted to real tokens; 0.45 MACs)
#   4  a6-h384-r256: fresh ALBERT body (hidden 384, 6 layers, FFN 1536), only the vocabulary from the teacher
#   5  summary_s1.md (tiny10 table, active dims, fidelity after every stage, queue state)
# Budget (3060, estimates): KD stage ~25 min each at 150 k texts, stage 3 ~40 min, tiny10 ~10 min -> ~6 h in all.
# Reading (plan §5): control vs b4 after stages 1-3 < 0.5 on tiny10 = the unlabeled corpus adds nothing over
# ranking KD; a6 more than 3 points under b4 = the fresh-body variant is dropped.
# Start: setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_s1.sh > logs/queue_s1.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s1 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline
export KD_TIMEOUT=${KD_TIMEOUT:-5400} TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-6000}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s1.pid"
guard_alive
C="$EXP_ROOT/configs/3060"
OLD=experiments/polish-splade-2026-10/results-regimes/quantized
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
KD_ITEMS=${KD_ITEMS:-150000}
B_KD=5400; B_TRAIN=6000; B_TINY=900
S1_END=${S1_END:-2026-11-05 08:00}
END=$(date -d "$S1_END" +%s)

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() { local need=$1; shift; fits "$need" || { log "no time before $S1_END for a ${need}s step; skipping: $*"; return 1; }; "$@"; }
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in v34 v33; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }
[ -f "$CORPUS" ] || { log "corpus $CORPUS missing (kitchen corpus/build_kd_corpus.py corpus/recipes/kd_v1.yaml)"; exit 1; }

log "=== s1 start: end $S1_END; corpus $CORPUS ($KD_ITEMS texts per KD stage); $(free_gb) GB free on / ==="
log "=== s1 step 0: reference results and students ==="
mkdir -p "$REGIMES_ROOT/quantized"
for m in base-splade-wrapped arm-n4-3060 arm-distil-v9b-kl-n4 soup-v9b; do
    [ -d "$REGIMES_ROOT/quantized/$m" ] || cp -r "$OLD/$m" "$REGIMES_ROOT/quantized/"
done
build_student() {  # build_student NAME ARGS...: distill.student from soup-v9b on the CPU (a minute)
    local name="$1"; shift
    [ -d "models/students/$name" ] && return 0
    log "student $name ($*)"
    "$PY" -m distill.student --teacher models/soup-v9b --out "models/students/$name" "$@" --threads 4 > "$LOGS/student_$name.log" 2>&1 \
        || { log "student $name FAILED"; record "student:$name" FAILED "-" "-"; return 1; }
    record "student:$name" ok "-" "-"
}
build_student b4-l0125-f1536 --keep-layers 0,1,2,5 --intermediate-size 1536
build_student b4-l0125-f1536-hl-r256-w --keep-layers 0,1,2,5 --intermediate-size 1536 --embedding-rank 256 --head-layer \
    --token-counts "$EXP_ROOT/results/token_counts_kd_v1_sample.json" --calibration-texts datasets/synth/chunks_pool.jsonl --calibration-sample 96
build_student a6-h384-r256 --keep-layers 0,1,2,3,4,5 --intermediate-size 1536 --embedding-rank 256 --hidden-size 384 --fresh

KD_COMMON=(--teacher models/soup-v9b --corpus "$CORPUS" --max-items "$KD_ITEMS" --max-seq-length 256 --batch-size 16
           --lr 5e-5 --warmup-steps 300 --schedule constant --epochs 1 --precision fp16 --head-chunk 64
           --fidelity-texts "$CORPUS" --fidelity-n 256 --log-every 50)
measure() {  # measure TAG: sparsity + leaderboard tiny10 of models/arm-TAG
    step 120 sparsity "arm-$1" "models/arm-$1"
    step $B_TINY regime quantized "arm-$1" "models/arm-$1" tiny10 16
    log "  arm-$1: tiny10 $(lbv "arm-$1" tiny10) (base 61.59, arm-n4-3060 62.89, arm-distil-v9b-kl-n4 63.32, soup-v9b 63.65), $(sparsity_mean "$RESULTS_ROOT/sparsity/arm-$1.json" "arm-$1") dims/doc"
}

log "=== s1 step 1: control (pruned student, ranking KD only) ==="
step $B_TRAIN train b4-l0125-f1536-ctrl "$C/stage3_b4_l0125_f1536_ctrl.yaml" && measure b4-l0125-f1536-ctrl

for name in b4-l0125-f1536 b4-l0125-f1536-hl-r256-w a6-h384-r256; do
    log "=== s1: $name, stages 1-3 ==="
    step $B_KD kd "$name-s1" "models/students/$name" "${KD_COMMON[@]}" --token-loss kl --pooled-loss none --temperature 2.0 --top-k 128 --seed 42 || continue
    step $B_KD kd "$name-s2" "models/kd-$name-s1" "${KD_COMMON[@]}" --token-loss none --pooled-loss mse --seed 43 || continue
    step $B_TRAIN train "$name-s3" "$C/stage3_${name//-/_}_s3.yaml" && measure "$name-s3"
done

log "=== s1 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_s1.md"
{
    echo "# s1 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): small-SPLADE probe, teacher soup-v9b"
    echo
    echo "Students: b4 = layers 0,1,2,5 of soup-v9b with FFN 1536 (58 M params, 0.71 of the teacher's MACs per token); hl-r256-w = the same body as an ALBERT with the teacher's head transform as an extra layer and a rank-256 vocabulary / decoder (50 M, 0.45); a6 = fresh ALBERT body, hidden 384, 6 layers (vocabulary from the teacher). Stages: s1 token KL, s2 pooled MSE ($KD_ITEMS texts of $CORPUS each), s3 ranking KL on the n4 rows (v32 recipe). ctrl = s3 only."
    echo
    "$PY" scripts/regime_task_table.py --regime-dir "$REGIMES_ROOT/quantized" --tasksets tiny10 \
        --models base-splade-wrapped arm-n4-3060 arm-distil-v9b-kl-n4 arm-b4-l0125-f1536-ctrl arm-b4-l0125-f1536-s3 arm-b4-l0125-f1536-hl-r256-w-s3 arm-a6-h384-r256-s3 soup-v9b 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-b4-l0125-f1536-ctrl --model arm-b4-l0125-f1536-s3 --model arm-b4-l0125-f1536-hl-r256-w-s3 --model arm-a6-h384-r256-s3 2>> "$LOGS/queue_summary.log"
    echo; echo "## Fidelity to the teacher after every KD stage (cosine of pooled vectors, 256 corpus texts)"; echo
    echo '```'
    for r in runs/kd-*; do [ -f "$r/kd_summary.json" ] && echo "$(basename "$r"): $("$PY" -c "import json;d=json.load(open('$r/kd_summary.json'));print(json.dumps(d.get('fidelity')), d['steps'], 'steps', d['examples'], 'examples', d['seconds'], 's')")"; done
    echo '```'
    echo; echo "## Queue state (s1)"; echo; echo '```'
    grep -E "^(student|kd|train|sparsity|regime:quantized):" "$STATE" | tail -40
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_s1.out"
log "summary written to $SUM"
log "=== s1 done ==="
