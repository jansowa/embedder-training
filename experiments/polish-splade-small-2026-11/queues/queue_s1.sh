#!/usr/bin/env bash
# Queue s1 (3060 Ti 8 GB; NOT STARTED, needs the user's go): the small-SPLADE probe night (docs/PLAN-2026-11-04.md §5,
# revised 5.11 after the review in STATUS-DISTILL-2026-11-05.md).
# Does representation distillation on unlabeled text turn a pruned soup-v9b into a usable student, and does the
# staged recipe (short token-KL repair, then pooled-MSE + token-KL) beat the combined loss from the start?
#   0  reference results copied into this series' results-regimes; students; GPU memory check of the fast head
#   1  control: b4-l0125-f3072 (layers 0,1,2,5, FFN 3072, no KD) -> stage 3 only (ranking KL on n4 rows) -> tiny10
#   2  b4-l0125-f3072 staged: s1 token KL (KD_S1_ITEMS texts) -> s2 pooled MSE + token KL x0.3 (KD_S2_ITEMS) -> s3 -> tiny10
#   3  b4-l0125-f3072 joint: one KD stage with the combined loss on KD_S1_ITEMS + KD_S2_ITEMS texts -> s3 -> tiny10
#   4  b4-l0125-f1536-hl-r256-w (ALBERT with the head layer, rank-256 vocabulary weighted by kd_v1 token counts;
#      0.45 of the teacher's MACs): staged as 2 -> s3 -> tiny10
#   5  summary_s1.md: tiny10, active dims, fidelity curve per checkpoint (cosine / L1 / dims, passages and queries),
#      ranking agreement with the teacher scores on the stage-3 rows, queue state
# Every KD stage checkpoints every 1500 steps with fidelity on 256 corpus texts (a curve, not one number): "the
# corpus adds nothing over ranking KD" can only be read from a flat curve. At this dose the probe decides
# "b4 vs control" and "staged vs joint", not "ALBERT vs b4".
# Budget (3060 Ti, estimates): KD ~35 min per arm at 200 k texts, stage 3 with the fast head ~20-40 min, tiny10 ~10 min,
# measurements ~5 min per arm -> 4-5 h in all.
# Start (only when the user says so): setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_s1.sh > logs/queue_s1.out 2>&1 < /dev/null &
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
ROWS=${ROWS:-datasets/arms/mix3v2-6219-tv9b.jsonl}
KD_S1_ITEMS=${KD_S1_ITEMS:-50000}
KD_S2_ITEMS=${KD_S2_ITEMS:-150000}
B_KD=5400; B_TRAIN=6000; B_TINY=900
S1_END=${S1_END:-2026-11-06 08:00}
END=$(date -d "$S1_END" +%s)
MEM_LIMIT_GB=7.0

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() { local need=$1; shift; fits "$need" || { log "no time before $S1_END for a ${need}s step; skipping: $*"; return 1; }; "$@"; }
other_queues() {  # tags of other queues whose pid file points at a live queue process
    local pf tag pid
    for pf in "$LOGS"/queue_*.pid; do
        tag=$(basename "$pf" .pid); tag=${tag#queue_}
        [ "$tag" = s1 ] && continue
        pid=$(cat "$pf" 2>/dev/null) || continue
        ps -p "$pid" -o args= 2>/dev/null | grep -q "queue_$tag.sh" && echo "$tag"
    done
}
while :; do
    others=$(other_queues | tr '\n' ' ')
    [ -z "$others" ] && break
    log "waiting for queue(s) $others to exit"
    sleep 120
done
sleep 30
wait_gpu_free
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" | awk '{print $1}'; }
[ -f "$CORPUS" ] || { log "corpus $CORPUS missing (kitchen corpus/build_kd_corpus.py corpus/recipes/kd_v1.yaml)"; exit 1; }
[ -f "$ROWS" ] || { log "stage-3 rows $ROWS missing"; exit 1; }

log "=== s1 start: end $S1_END; corpus $CORPUS ($KD_S1_ITEMS + $KD_S2_ITEMS texts per arm); $(free_gb) GB free on / ==="
log "=== s1 step 0: reference results, students, memory check ==="
mkdir -p "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/rank_agreement" "$RESULTS_ROOT/fidelity"
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
build_student b4-l0125-f3072 --keep-layers 0,1,2,5
build_student b4-l0125-f1536-hl-r256-w --keep-layers 0,1,2,5 --intermediate-size 1536 --embedding-rank 256 --head-layer \
    --token-counts "$EXP_ROOT/results/token_counts_kd_v1_sample.json" --calibration-texts datasets/synth/chunks_pool.jsonl --calibration-sample 96

memcheck() {  # memcheck STUDENT: peak GB of one stage-3 step with the fast head and no checkpointing; "OOM" on failure
    "$PY" scripts/splade_head_memcheck.py --model "$1" --head exact --no-checkpointing --max-seq-length 512 --columns 12 --micro-batch 2 \
        2>> "$LOGS/memcheck_s1.log" | grep -o "peak_gb=.*" | tail -1 | cut -d= -f2
}
stage3_config() {  # stage3_config NAME PEAK -> the stage-3 config, with checkpointing switched back on when the memory check failed
    local cfg="$C/stage3_$1.yaml" peak="$2"
    if [ -z "$peak" ] || [ "$peak" = OOM ] || [ "$(awk -v p="$peak" -v l="$MEM_LIMIT_GB" 'BEGIN{print (p > l) ? 1 : 0}')" = 1 ]; then
        mkdir -p "$C/derived"
        sed 's/gradient_checkpointing: false/gradient_checkpointing: true/' "$cfg" > "$C/derived/stage3_$1.ckpt.yaml"
        echo "$C/derived/stage3_$1.ckpt.yaml"
    else echo "$cfg"; fi
}
PEAK_B4=$(memcheck models/students/b4-l0125-f3072); log "memcheck b4-l0125-f3072 fast head, no checkpointing: peak ${PEAK_B4:-?} GB (limit $MEM_LIMIT_GB)"
PEAK_HL=$(memcheck models/students/b4-l0125-f1536-hl-r256-w); log "memcheck b4-l0125-f1536-hl-r256-w: peak ${PEAK_HL:-?} GB"
record "memcheck:b4-l0125-f3072" "${PEAK_B4:-?}" "-" "-"; record "memcheck:b4-l0125-f1536-hl-r256-w" "${PEAK_HL:-?}" "-" "-"

KD_COMMON=(--teacher models/soup-v9b --corpus "$CORPUS" --max-seq-length 256 --batch-size 16
           --lr 5e-5 --warmup-steps 300 --schedule constant --epochs 1 --precision fp16 --head-chunk 64
           --save-every 1500 --keep-last 1 --fidelity-texts "$CORPUS" --fidelity-n 256 --log-every 50)
S2_LOSS=(--token-loss kl --token-weight 0.3 --temperature 2.0 --top-k 128 --pooled-loss mse --pooled-weight 1.0)
agree() {  # agree LABEL MODEL [args]: ranking agreement with the teacher scores on 500 stage-3 rows + fidelity on queries
    local label="$1" model="$2"; shift 2
    local out="$RESULTS_ROOT/rank_agreement/$label.json"
    if [ ! -f "$out" ]; then
        wait_gpu_free
        timeout -k 60 1800 "$PY" -m distill.rank_agreement --model "$model" --rows "$ROWS" --n 500 --device cuda --dtype fp16 --output "$out" "$@" > "$LOGS/agree_$label.log" 2>&1 \
            && log "  $label: $("$PY" -c "import json;d=json.load(open('$out'));print('tau %.3f top1 %.3f pos-first %.3f' % (d['kendall_tau'], d['top1_agreement'], d['positive_first']))")" \
            || log "  $label: rank agreement FAILED"
    fi
    fidelity "$label.queries" "$model" "$CORPUS" --kinds query --n 512 "$@"
}
measure() {  # measure TAG: sparsity + leaderboard tiny10 + agreement of models/arm-TAG
    step 120 sparsity "arm-$1" "models/arm-$1"
    step $B_TINY regime quantized "arm-$1" "models/arm-$1" tiny10 16
    log "  arm-$1: tiny10 $(lbv "arm-$1" tiny10) (base 61.59, arm-n4-3060 62.89, arm-distil-v9b-kl-n4 63.32, soup-v9b 63.65), $(sparsity_mean "$RESULTS_ROOT/sparsity/arm-$1.json" "arm-$1") dims/doc"
    step 600 agree "arm-$1" "models/arm-$1"
}

log "=== s1 step 1: control (b4-l0125-f3072, ranking KD only) ==="
step 600 agree student-b4-l0125-f3072 models/students/b4-l0125-f3072
step $B_TRAIN train b4-l0125-f3072-ctrl "$(stage3_config b4_l0125_f3072_ctrl "$PEAK_B4")" && measure b4-l0125-f3072-ctrl

log "=== s1 step 2: b4-l0125-f3072 staged (s1 token KL $KD_S1_ITEMS -> s2 pooled MSE + token KL x0.3 $KD_S2_ITEMS -> s3) ==="
name=b4-l0125-f3072
step $B_KD kd "$name-s1" "models/students/$name" "${KD_COMMON[@]}" --max-items "$KD_S1_ITEMS" --token-loss kl --pooled-loss none --temperature 2.0 --top-k 128 --seed 42 \
    && step 600 agree "kd-$name-s1" "models/kd-$name-s1" \
    && step $B_KD kd "$name-s2" "models/kd-$name-s1" "${KD_COMMON[@]}" --max-items "$KD_S2_ITEMS" "${S2_LOSS[@]}" --seed 43 \
    && step 600 agree "kd-$name-s2" "models/kd-$name-s2" \
    && step $B_TRAIN train "$name-s3" "$(stage3_config b4_l0125_f3072_s3 "$PEAK_B4")" && measure "$name-s3"

log "=== s1 step 3: b4-l0125-f3072 joint (one stage, combined loss, $((KD_S1_ITEMS + KD_S2_ITEMS)) texts -> s3) ==="
step $B_KD kd "$name-joint" "models/students/$name" "${KD_COMMON[@]}" --max-items "$((KD_S1_ITEMS + KD_S2_ITEMS))" "${S2_LOSS[@]}" --seed 42 \
    && step 600 agree "kd-$name-joint" "models/kd-$name-joint" \
    && step $B_TRAIN train "$name-joint-s3" "$(stage3_config b4_l0125_f3072_joint_s3 "$PEAK_B4")" && measure "$name-joint-s3"

log "=== s1 step 4: b4-l0125-f1536-hl-r256-w staged ==="
name=b4-l0125-f1536-hl-r256-w
# head-layer student: the measurements run the extra layer's FFN and the head transform in fp32, as the KD trainer does
step 600 agree "student-$name" "models/students/$name" --fp32-head
step $B_KD kd "$name-s1" "models/students/$name" "${KD_COMMON[@]}" --max-items "$KD_S1_ITEMS" --token-loss kl --pooled-loss none --temperature 2.0 --top-k 128 --seed 42 \
    && step 600 agree "kd-$name-s1" "models/kd-$name-s1" --fp32-head \
    && step $B_KD kd "$name-s2" "models/kd-$name-s1" "${KD_COMMON[@]}" --max-items "$KD_S2_ITEMS" "${S2_LOSS[@]}" --seed 43 \
    && step 600 agree "kd-$name-s2" "models/kd-$name-s2" --fp32-head \
    && step $B_TRAIN train "$name-s3" "$(stage3_config b4_l0125_f1536_hl_r256_w_s3 "$PEAK_HL")" && measure "$name-s3"

log "=== s1 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_s1.md"
{
    echo "# s1 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): small-SPLADE probe, teacher soup-v9b"
    echo
    echo "Students: b4-f3072 = layers 0,1,2,5 of soup-v9b with the full FFN (68 M params, 0.83 of the teacher's MACs per token); hl-r256-w = layers 0,1,2,5 with FFN 1536 as an ALBERT with the teacher's head transform as an extra layer and a rank-256 vocabulary / decoder (50 M, 0.45). Stages: s1 token KL on $KD_S1_ITEMS texts, s2 pooled MSE (relative) + token KL x0.3 on $KD_S2_ITEMS texts of $CORPUS, joint = the combined loss on $((KD_S1_ITEMS + KD_S2_ITEMS)) texts, s3 ranking KL on the n4 rows (v32 recipe, fast head). ctrl = s3 only. Memory check (fast head, no checkpointing): b4 ${PEAK_B4:-?} GB, hl ${PEAK_HL:-?} GB (limit $MEM_LIMIT_GB)."
    echo
    "$PY" scripts/regime_task_table.py --regime-dir "$REGIMES_ROOT/quantized" --tasksets tiny10 \
        --models base-splade-wrapped arm-n4-3060 arm-distil-v9b-kl-n4 arm-b4-l0125-f3072-ctrl arm-b4-l0125-f3072-s3 arm-b4-l0125-f3072-joint-s3 arm-b4-l0125-f1536-hl-r256-w-s3 soup-v9b 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-b4-l0125-f3072-ctrl --model arm-b4-l0125-f3072-s3 --model arm-b4-l0125-f3072-joint-s3 --model arm-b4-l0125-f1536-hl-r256-w-s3 2>> "$LOGS/queue_summary.log"
    echo; echo "## Fidelity curve per KD checkpoint (256 corpus texts: cosine / L1 relative error / student dims, per kind)"; echo
    echo '```'
    for r in runs/kd-*; do
        [ -f "$r/log.jsonl" ] || continue
        echo "$(basename "$r"):"
        "$PY" scripts/kd_fidelity_curve.py "$r/log.jsonl" 2>> "$LOGS/queue_summary.log"
    done
    echo '```'
    echo; echo "## Ranking agreement with the teacher scores (500 rows of $(basename "$ROWS")) and fidelity on queries (512 corpus queries)"; echo
    echo '```'
    for f in "$RESULTS_ROOT"/rank_agreement/*.json; do
        [ -f "$f" ] || continue
        label=$(basename "$f" .json)
        "$PY" scripts/kd_fidelity_curve.py --agreement "$f" --queries "$RESULTS_ROOT/fidelity/$label.queries.json" 2>> "$LOGS/queue_summary.log"
    done
    echo '```'
    echo; echo "## Queue state (s1)"; echo; echo '```'
    grep -E "^(student|memcheck|kd|train|sparsity|regime:quantized|fidelity):" "$STATE" | tail -60
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_s1.out"
log "summary written to $SUM"
log "=== s1 done ==="
