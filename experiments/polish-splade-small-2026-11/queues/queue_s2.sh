#!/usr/bin/env bash
# Queue s2 (3060 Ti, 6.11, after v36 and v39, ~6.5 h): full representation distillation of the b4 student on the whole
# kd_v1 corpus. s1 (STATUS-DISTILL-2026-11-05.md §8, threshold 2) read: b4 after KD on 200 k texts 63.15 / 63.29 on tiny10
# against 59.83 for the control (ranking KD only) and 63.65 for the teacher; the fidelity curve was still rising at
# 200 k (cosine 0.950 at step 12 500). Joint >= staged (threshold 3), so the recipe is the combined loss from the start.
#   1  kd b4-l0125-f3072-full: token KL x0.3 + pooled MSE on all 2 264 676 texts, one epoch, linear decay, checkpoints
#      every 3 000 steps with fidelity on 256 corpus texts (the curve); ~5 h
#   2  ranking agreement + query fidelity of the KD student; stage 3 (ranking KL on the n4 rows, fast head) ->
#      sparsity, tiny10, heldout8, agreement
#   3  summary_s2.md next to the s1 arms; commit of the s1 and s2 results, push
# Reading (tiny10 / heldout8, base 61.60 / 41.9, joint-s3 63.29, teacher 63.65 / 42.56): full-s3 >= 63.5 and heldout8
# >= 42.3 = the corpus keeps paying -> the ALBERT student gets the same full KD (with --flops-weight: 593 dims/doc after
# s1); full-s3 within 0.2 of joint-s3 = the curve saturated at 200 k, next move = dose on the ranking rows (n5) for the
# student, not more corpus.
#   WAIT_FOR="v36 v39" setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_s2.sh > logs/queue_s2.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s2 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export KD_TIMEOUT=${KD_TIMEOUT:-28800} TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-6000}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s2.pid"
guard_alive
C="$EXP_ROOT/configs/3060"
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
ROWS=${ROWS:-datasets/arms/mix3v2-6219-tv9b.jsonl}

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v36 v39}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
[ -f "$CORPUS" ] || { log "corpus $CORPUS missing"; exit 1; }
[ -f "$ROWS" ] || { log "stage-3 rows $ROWS missing"; exit 1; }
[ -d models/students/b4-l0125-f3072 ] || { log "student models/students/b4-l0125-f3072 missing (queue s1 builds it)"; exit 1; }

log "=== s2 start: full KD of b4-l0125-f3072 on $CORPUS ($(grep -c . "$CORPUS") texts); $(free_gb) GB free on / ==="

KD_COMMON=(--teacher models/soup-v9b --corpus "$CORPUS" --max-seq-length 256 --batch-size 16
           --lr 5e-5 --warmup-steps 300 --schedule linear --epochs 1 --precision fp16 --head-chunk 64
           --save-every 3000 --keep-last 1 --fidelity-texts "$CORPUS" --fidelity-n 256 --log-every 50)
KD_LOSS=(--token-loss kl --token-weight 0.3 --temperature 2.0 --top-k 128 --pooled-loss mse --pooled-weight 1.0)
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

log "=== s2 step 1: KD b4-l0125-f3072 on the whole corpus ==="
name=b4-l0125-f3072
kd "$name-full" "models/students/$name" "${KD_COMMON[@]}" "${KD_LOSS[@]}" --seed 42 \
    && agree "kd-$name-full" "models/kd-$name-full"

log "=== s2 step 2: stage 3 and measurements ==="
if [ -d "models/kd-$name-full" ]; then
    train "$name-full-s3" "$C/stage3_b4_l0125_f3072_full_s3.yaml"
    if [ -d "models/arm-$name-full-s3" ]; then
        sparsity "arm-$name-full-s3" "models/arm-$name-full-s3"
        regime quantized "arm-$name-full-s3" "models/arm-$name-full-s3" tiny10 16
        log "  arm-$name-full-s3: tiny10 $(lbv "arm-$name-full-s3" tiny10) (joint-s3 63.29, s3 63.15, ctrl 59.83, base 61.59, soup-v9b 63.65), $(sparsity_mean "$RESULTS_ROOT/sparsity/arm-$name-full-s3.json" "arm-$name-full-s3") dims/doc"
        regime quantized "arm-$name-full-s3" "models/arm-$name-full-s3" heldout8 16
        log "  arm-$name-full-s3: heldout8 $(lbv "arm-$name-full-s3" heldout8) (arm-distil-v9b-kl-n4 $(lbv arm-distil-v9b-kl-n4 heldout8), soup-v9b $(lbv soup-v9b heldout8))"
        agree "arm-$name-full-s3" "models/arm-$name-full-s3"
    fi
else
    log "no full-KD student; skipping stage 3"
fi

log "=== s2 step 3: summary ==="
SUM="$RESULTS_ROOT/summary_s2.md"
{
    echo "# s2 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): full KD of b4-l0125-f3072 on kd_v1, teacher soup-v9b"
    echo
    echo "Student b4-f3072 = layers 0,1,2,5 of soup-v9b with the full FFN (67.8 M params, 0.83 of the teacher's MACs per token). KD: the combined loss (token KL x0.3, T=2, top-128 + pooled MSE) on all $(grep -c . "$CORPUS") texts of $CORPUS, one epoch, batch 16, lr 5e-5 with linear decay, checkpoints every 3 000 steps; then stage 3 = ranking KL on the n4 rows (v32 recipe, fast head). s1 arms for reference: joint-s3 = the same loss on 200 000 texts, s3 = staged 50 k + 150 k, ctrl = stage 3 only."
    echo
    "$PY" scripts/regime_task_table.py --regime-dir "$REGIMES_ROOT/quantized" --tasksets tiny10 heldout8 \
        --models base-splade-wrapped arm-distil-v9b-kl-n4 arm-b4-l0125-f3072-ctrl arm-b4-l0125-f3072-s3 arm-b4-l0125-f3072-joint-s3 "arm-$name-full-s3" soup-v9b 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-b4-l0125-f3072-joint-s3 --model "arm-$name-full-s3" 2>> "$LOGS/queue_summary.log"
    echo; echo "## Fidelity curve of the full KD (256 corpus texts: cosine / L1 relative error / student dims, per kind)"; echo
    echo '```'
    [ -f "runs/kd-$name-full/log.jsonl" ] && "$PY" scripts/kd_fidelity_curve.py "runs/kd-$name-full/log.jsonl" 2>> "$LOGS/queue_summary.log"
    echo '```'
    echo; echo "## Ranking agreement with the teacher scores (500 rows of $(basename "$ROWS")) and fidelity on queries (512 corpus queries)"; echo
    echo '```'
    for label in kd-b4-l0125-f3072-joint arm-b4-l0125-f3072-joint-s3 "kd-$name-full" "arm-$name-full-s3"; do
        f="$RESULTS_ROOT/rank_agreement/$label.json"; [ -f "$f" ] || continue
        "$PY" scripts/kd_fidelity_curve.py --agreement "$f" --queries "$RESULTS_ROOT/fidelity/$label.queries.json" 2>> "$LOGS/queue_summary.log"
    done
    echo '```'
    echo; echo "## Queue state (s2)"; echo; echo '```'
    grep -E "^(kd|train|sparsity|regime:quantized|fidelity):.*full" "$STATE"
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_s2.out"
log "summary written to $SUM"
git add "$RESULTS_ROOT"/summary_s1.md "$SUM" "$RESULTS_ROOT"/sparsity "$RESULTS_ROOT"/fidelity "$RESULTS_ROOT"/rank_agreement "$REGIMES_ROOT"/quantized "$C" "$EXP_ROOT/queues/queue_s2.sh" 2>/dev/null
git commit -q -m "Add the s1 and s2 results: the small-SPLADE probe and the full KD of the b4 student" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_s2.log" 2>&1 && { echo pushed >> "$LOGS/push_s2.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== s2 done ($(date '+%d.%m %H:%M')) ==="
