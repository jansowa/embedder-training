#!/usr/bin/env bash
# Queue s3 (3060 Ti, 6/7.11 night, after v40, ~5.2 h): full representation distillation of the ALBERT student
# b4-l0125-f1536-hl-r256-w on the whole kd_v1 corpus, the s2 recipe of b4-full (threshold 2 of STATUS-DISTILL §8 read
# on 6.11: b4 full-s3 63.77 / 42.54 against joint-s3 63.29 and the teacher 63.65 / 42.56, so the corpus keeps paying).
# Same loss, schedule and seed as b4-full so the two students differ only in architecture (0.45 vs 0.83 of the
# teacher's cost); no --flops-weight in KD (593 dims/doc after s1 came with cosine 0.905; b4 went 449 -> 426 with
# fidelity 0.950 -> 0.971, so the density is read again after this run before any regulariser is added).
#   1  kd hl-r256-w-full: token KL x0.3 + pooled MSE on all 2 264 676 texts, one epoch, linear decay, checkpoints every
#      3 000 steps with fidelity on 256 corpus texts; ~3.9 h (162 texts/s in s1)
#   2  ranking agreement + query fidelity of the KD student (fp32 head); stage 3 (ranking KL on the n4 rows, fast head)
#      -> sparsity, index-cost profile with the query side (v9b, b4-full-s3, hl-full-s3), tiny10, heldout8, agreement
#   3  summary_s3.md, commit, push
# Reading (tiny10 / heldout8; b4-full-s3 63.77 / 42.54, hl s3 at 200 k 61.28, teacher 63.65 / 42.56): hl-full-s3 within
# 1 point of b4-full-s3 = the narrow vocabulary is cheap (threshold 4): next student = b4-f3072-hl-r256-w (0.60) or
# 6+1 with r256 (0.77); more than 1 point below with the curve still rising = another epoch on the 5070; dims/doc
# outside 330-480 after stage 3 = --flops-weight in the next KD pass.
#   WAIT_FOR="v40" setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_s3.sh > logs/queue_s3.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=s3 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export KD_TIMEOUT=${KD_TIMEOUT:-25200} TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-6000}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_s3.pid"
guard_alive
C="$EXP_ROOT/configs/3060"
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
ROWS=${ROWS:-datasets/arms/mix3v2-6219-tv9b.jsonl}

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v40}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
name=b4-l0125-f1536-hl-r256-w
[ -f "$CORPUS" ] || { log "corpus $CORPUS missing"; exit 1; }
[ -f "$ROWS" ] || { log "stage-3 rows $ROWS missing"; exit 1; }
[ -d "models/students/$name" ] || { log "student models/students/$name missing (queue s1 builds it)"; exit 1; }

log "=== s3 start: full KD of $name on $CORPUS ($(grep -c . "$CORPUS") texts); $(free_gb) GB free on / ==="

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

log "=== s3 step 1: KD $name on the whole corpus ==="
# head-layer student: the measurements run the extra layer's FFN and the head transform in fp32, as the KD trainer does
kd "$name-full" "models/students/$name" "${KD_COMMON[@]}" "${KD_LOSS[@]}" --seed 42 \
    && agree "kd-$name-full" "models/kd-$name-full" --fp32-head

log "=== s3 step 2: stage 3 and measurements ==="
if [ -d "models/kd-$name-full" ]; then
    train "$name-full-s3" "$C/stage3_b4_l0125_f1536_hl_r256_w_full_s3.yaml"
    if [ -d "models/arm-$name-full-s3" ]; then
        sparsity "arm-$name-full-s3" "models/arm-$name-full-s3"
        COST="$RESULTS_ROOT/index_cost_s3.json"
        if [ ! -s "$COST" ]; then
            wait_gpu_free; s=$SECONDS
            timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 \
                --model soup-v9b=models/soup-v9b --model b4-full-s3=models/arm-b4-l0125-f3072-full-s3 --model hl-r256-w-s3=models/arm-$name-s3 --model hl-full-s3="models/arm-$name-full-s3" \
                --output-json "$COST" > "$LOGS/index_cost_s3.log" 2>&1 \
                && log "index-cost profile (doc + query side) written ($COST) in $((SECONDS - s))s" || log "index-cost profile FAILED (logs/index_cost_s3.log)"
            grep -E "doc_active" "$LOGS/index_cost_s3.log" | while read -r l; do log "  $l"; done
        fi
        regime quantized "arm-$name-full-s3" "models/arm-$name-full-s3" tiny10 16
        log "  arm-$name-full-s3: tiny10 $(lbv "arm-$name-full-s3" tiny10) (hl s3 at 200 k 61.28, b4-full-s3 63.77, base 61.59, soup-v9b 63.65), $(sparsity_mean "$RESULTS_ROOT/sparsity/arm-$name-full-s3.json" "arm-$name-full-s3") dims/doc"
        regime quantized "arm-$name-full-s3" "models/arm-$name-full-s3" heldout8 16
        log "  arm-$name-full-s3: heldout8 $(lbv "arm-$name-full-s3" heldout8) (b4-full-s3 $(lbv arm-b4-l0125-f3072-full-s3 heldout8), soup-v9b $(lbv soup-v9b heldout8))"
        agree "arm-$name-full-s3" "models/arm-$name-full-s3" --fp32-head
    fi
else
    log "no full-KD student; skipping stage 3"
fi

log "=== s3 step 3: summary ==="
SUM="$RESULTS_ROOT/summary_s3.md"
{
    echo "# s3 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): full KD of $name on kd_v1, teacher soup-v9b"
    echo
    echo "Student hl-r256-w = layers 0,1,2,5 of soup-v9b with FFN 1536 as an ALBERT with the teacher's head transform as an extra layer and a rank-256 vocabulary / decoder weighted by kd_v1 token counts (49.8 M params, 0.45 of the teacher's MACs per token). KD: the s2 recipe of b4-full (token KL x0.3, T=2, top-128 + pooled MSE) on all $(grep -c . "$CORPUS") texts of $CORPUS, one epoch, batch 16, lr 5e-5 with linear decay, seed 42; then stage 3 = ranking KL on the n4 rows (v32 recipe, fast head). References: hl s3 = the same student after 200 k texts (s1), b4-full-s3 = the wide student after the same full KD (s2)."
    echo
    "$PY" scripts/regime_task_table.py --regime-dir "$REGIMES_ROOT/quantized" --tasksets tiny10 heldout8 \
        --models base-splade-wrapped arm-b4-l0125-f1536-hl-r256-w-s3 "arm-$name-full-s3" arm-b4-l0125-f3072-joint-s3 arm-b4-l0125-f3072-full-s3 soup-v9b 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-b4-l0125-f1536-hl-r256-w-s3 --model "arm-$name-full-s3" --model arm-b4-l0125-f3072-full-s3 2>> "$LOGS/queue_summary.log"
    echo; echo "## Index cost (one PIRB sample: 1 350 passages + 1 350 queries; doc dims, query dims, postings/query)"; echo; echo '```'
    [ -s "$RESULTS_ROOT/index_cost_s3.json" ] && "$PY" - "$RESULTS_ROOT/index_cost_s3.json" <<'EOF' 2>> "$LOGS/queue_summary.log"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{'model':20s} {'dims/doc':>9s} {'dims/query':>11s} {'postings/q':>11s} {'flops':>7s}")
for name, s in d["models"].items():
    print(f"{name:20s} {s['doc_active_mean']:9.1f} {s['query_active_mean']:11.1f} {s['postings_per_query']:11.0f} {s['flops']:7.3f}")
EOF
    echo '```'
    echo; echo "## Fidelity curve of the full KD (256 corpus texts: cosine / L1 relative error / student dims, per kind)"; echo
    echo '```'
    [ -f "runs/kd-$name-full/log.jsonl" ] && "$PY" scripts/kd_fidelity_curve.py "runs/kd-$name-full/log.jsonl" 2>> "$LOGS/queue_summary.log"
    echo '```'
    echo; echo "## Ranking agreement with the teacher scores (500 rows of $(basename "$ROWS")) and fidelity on queries (512 corpus queries)"; echo
    echo '```'
    for label in kd-$name-s2 arm-$name-s3 "kd-$name-full" "arm-$name-full-s3" kd-b4-l0125-f3072-full arm-b4-l0125-f3072-full-s3; do
        f="$RESULTS_ROOT/rank_agreement/$label.json"; [ -f "$f" ] || continue
        "$PY" scripts/kd_fidelity_curve.py --agreement "$f" --queries "$RESULTS_ROOT/fidelity/$label.queries.json" 2>> "$LOGS/queue_summary.log"
    done
    echo '```'
    echo; echo "## Queue state (s3)"; echo; echo '```'
    grep -E "^(kd|train|sparsity|regime:quantized|fidelity):.*hl-r256-w-full" "$STATE"
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_s3.out"
log "summary written to $SUM"
git add "$SUM" "$RESULTS_ROOT"/index_cost_s3.json "$RESULTS_ROOT"/sparsity "$RESULTS_ROOT"/fidelity "$RESULTS_ROOT"/rank_agreement "$REGIMES_ROOT"/quantized "$C" "$EXP_ROOT/queues/queue_s3.sh" 2>/dev/null
git commit -q -m "Add the s3 results: the full KD of the ALBERT student" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_s3.log" 2>&1 && { echo pushed >> "$LOGS/push_s3.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== s3 done ($(date '+%d.%m %H:%M')) ==="
