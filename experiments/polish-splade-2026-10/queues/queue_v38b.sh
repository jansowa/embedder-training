#!/usr/bin/env bash
# v38b (night 5/6.11, ~3.3 h, after s1): is the dense teacher's strength what the n1 distillation lacks? v38 gave
# PolDense-150M 63.41 on tiny10 but 42.28 on heldout8 (kl-n1v2 from soup-v9b: 63.25 / 42.62), so the dense gain did
# not carry over. OPI-PIB/PolDense-400M is the same family and prefix ("[query]: "), only stronger (our 23 PIRB tasks
# 61.8 vs 60.4 for pd150; tiny10 68.8 / heldout8 46.8 on the leaderboard), so the difference measures teacher
# strength alone. Steps: 1 score the n1 rows with pd400 (kitchen bridge/dense_teacher_scores.py, scaled to soup-v9b's
# units as in v38), 2 arm-dense-kl-n1v2-pd400 (kl-n1v2 recipe, fast head) -> sparsity, tiny10, heldout8 (unconditional:
# heldout8 is the question), 3 pd400 scores for the n4 rows (mix3v2-6219) and the n5 rows (fiqa-plus-squadv2v2-18k)
# so that the 5070 can train the bigger arms on 6.11 without waiting for this machine, 4 summary_v38b.md, commit.
# Reading: pd400 heldout8 >= 42.62 and tiny10 >= 63.4 = teacher strength matters (next: pd400 on n4 / n5, ensemble);
# heldout8 still below kl-n1v2 = dense KD alone loses the lexical side regardless of the teacher (next: KL + MNRL).
#   WAIT_FOR="v38 s1" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v38b.sh > logs/queue_v38b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v38b MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-3600}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v38b.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN=${KITCHEN:-$HOME/work/speakleash/retrieval-data-kitchen}
TEACHER=OPI-PIB/PolDense-400M

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v38 s1}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
[ -f "$KITCHEN/bridge/dense_teacher_scores.py" ] || { log "kitchen script missing: $KITCHEN/bridge/dense_teacher_scores.py"; exit 1; }
teacher_cached() { ls "$HOME/.cache/huggingface/hub/models--OPI-PIB--PolDense-400M/snapshots"/*/model.safetensors > /dev/null 2>&1; }
if ! teacher_cached; then
    log "$TEACHER not in the HF cache yet (logs/download_pd400.log); waiting up to 30 min"
    for i in $(seq 1 30); do teacher_cached && break; sleep 60; done
    teacher_cached || { log "$TEACHER missing; nothing to do"; exit 1; }
fi

log "=== v38b start: PolDense-400M as the dense teacher on the n1 rows (+ scores for n4 / n5); $(free_gb) GB free ==="

dense_scores() {  # dense_scores TAG ROWS REF OUT: pd400 scores for ROWS, scaled against REF (soup-v9b), written to OUT
    local tag="$1" rows="$2" ref="$3" out="$4"
    [ -s "$out" ] && [ -f "$out.teacher.json" ] && { log "teacher scores $tag: already present"; return 0; }
    [ -f "$rows" ] && [ -f "$ref" ] || { log "teacher scores $tag: rows or reference missing ($rows, $ref)"; record "teacher:$tag" SKIPPED "-" "missing input"; return 1; }
    wait_gpu_free; wait_ram
    log "teacher scores $tag ($TEACHER, preset poldense, bf16) -> $out"
    local s=$SECONDS
    timeout -k 60 5400 "$PY" "$KITCHEN/bridge/dense_teacher_scores.py" --teacher "$TEACHER" --preset poldense --dtype bf16 \
        --rows "$rows" --reference "$ref" --output "$out" --max-seq-length 512 --batch-size 32 --device cuda \
        > "$LOGS/teacher_scores_$tag.log" 2>&1
    local rc=$? line
    line=$(grep -E "^[0-9]+ rows ->" "$LOGS/teacher_scores_$tag.log" | tail -1)
    if [ "$rc" -eq 0 ] && [ -s "$out" ]; then log "teacher scores $tag done in $((SECONDS - s))s: ${line#* -> }"; record "teacher:$tag" ok "-" "$((SECONDS - s))s"
    else log "teacher scores $tag FAILED (rc=$rc, logs/teacher_scores_$tag.log)"; record "teacher:$tag" FAILED "-" "$((SECONDS - s))s"; rm -f "$out"; return 1; fi
}

# 1. n1 rows
dense_scores pd400 datasets/arms/fiqa-plus-squadv2v2.jsonl datasets/arms/fiqa-plus-squadv2v2-tv9b.jsonl datasets/arms/fiqa-plus-squadv2v2-tpd400.jsonl

# 2. the arm
TAG=dense-kl-n1v2-pd400
if [ -s datasets/arms/fiqa-plus-squadv2v2-tpd400.jsonl ]; then
    train "$TAG" "$C/distil_dense_kl_n1v2_pd400.yaml"
    if [ -d "models/arm-$TAG" ]; then
        sparsity "arm-$TAG" "models/arm-$TAG"
        regime quantized "arm-$TAG" "models/arm-$TAG" tiny10 16
        regime quantized "arm-$TAG" "models/arm-$TAG" heldout8 16
        log "  arm-$TAG: $(lbs "arm-$TAG") (pd150 $(lbs arm-dense-kl-n1v2-pd150), kl-n1v2 63.25 / 42.62, n1-5070 MNRL 62.98 / 42.17), $(dims "arm-$TAG") dims/doc"
    fi
else
    log "no pd400 teacher file; skipping the arm"
fi

# 2b. the ensemble with pd400 (v38: the pd150 ensemble gave 63.56 / 42.59, the best n1 arm so far; is the stronger dense
#     member better inside the ensemble? mean of soup-v9b's dot products and pd400's scaled cosines, pair by pair)
ENS=datasets/arms/fiqa-plus-squadv2v2-tens400.jsonl
if [ ! -s "$ENS" ] && [ -s datasets/arms/fiqa-plus-squadv2v2-tpd400.jsonl ]; then
    "$PY" - datasets/arms/fiqa-plus-squadv2v2-tv9b.jsonl datasets/arms/fiqa-plus-squadv2v2-tpd400.jsonl "$ENS" <<'EOF' && log "ensemble teacher file written: $ENS" || { log "ensemble file FAILED"; rm -f "$ENS"; }
import json, sys
ref, dense, out = sys.argv[1:4]
n = 0
with open(ref, encoding="utf-8") as a, open(dense, encoding="utf-8") as b, open(out, "w", encoding="utf-8") as o:
    for la, lb in zip(a, b):
        ra, rb = json.loads(la), json.loads(lb)
        assert str(ra["query_id"]) == str(rb["query_id"]) and len(ra["pos"]) == len(rb["pos"]) and len(ra["neg"]) == len(rb["neg"])
        row = dict(ra)
        row["pos_scores"] = [(x + y) / 2 for x, y in zip(ra["pos_scores"], rb["pos_scores"])]
        row["neg_scores"] = [(x + y) / 2 for x, y in zip(ra["neg_scores"], rb["neg_scores"])]
        row["teacher_model"] = f"ensemble: mean({ra.get('teacher_model')}, {rb.get('teacher_model')})"
        o.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
print(n, "rows")
EOF
fi
TAG=dense-kl-n1v2-ens400
if [ -s "$ENS" ]; then
    train "$TAG" "$C/distil_dense_kl_n1v2_ens400.yaml"
    if [ -d "models/arm-$TAG" ]; then
        sparsity "arm-$TAG" "models/arm-$TAG"
        regime quantized "arm-$TAG" "models/arm-$TAG" tiny10 16
        regime quantized "arm-$TAG" "models/arm-$TAG" heldout8 16
        log "  arm-$TAG: $(lbs "arm-$TAG") (ens with pd150 $(lbs arm-dense-kl-n1v2-ens), kl-n1v2 63.25 / 42.62)"
    fi
else
    log "no ens400 teacher file; skipping the ensemble arm"
fi

# 3. scores for the bigger row sets (for the 5070 on 6.11; nothing is trained here)
dense_scores pd400-n4 datasets/arms/mix3v2-6219.jsonl datasets/arms/mix3v2-6219-tv9b.jsonl datasets/arms/mix3v2-6219-tpd400.jsonl
dense_scores pd400-n5 datasets/arms/fiqa-plus-squadv2v2-18k.jsonl datasets/arms/fiqa-plus-squadv2v2-18k-tv9b.jsonl datasets/arms/fiqa-plus-squadv2v2-18k-tpd400.jsonl

# 4. summary
SUM="$RESULTS_ROOT/summary_v38b.md"
{
    echo "# v38b ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): PolDense-400M as the dense teacher on the n1 rows"
    echo
    echo "Student sdadas/polish-splade, the kl-n1v2 recipe (4 146 rows: fiqa 2073 + squadv2-v2 2073, 2 epochs, KL T=2, FLOPS kept, 2 x 16, fast head). pd400 = OPI-PIB/PolDense-400M (dense ModernBERT 395 M, 61.8 on our 23 PIRB tasks; leaderboard tiny10 68.8 / heldout8 46.8), the same family and query prefix as pd150 (v38: 60.4; 63.41 / 42.28). Cosines scaled to soup-v9b's units by the within-row least-squares fit (sidecar datasets/arms/fiqa-plus-squadv2v2-tpd400.jsonl.teacher.json). The pd400 score files for the n4 rows (mix3v2-6219) and the n5 rows (fiqa-plus-squadv2v2-18k) are written here for the 5070."
    echo
    echo "## Teacher agreement with soup-v9b (within-row)"; echo; echo '```'
    for f in datasets/arms/fiqa-plus-squadv2v2-tpd150.jsonl.teacher.json datasets/arms/fiqa-plus-squadv2v2-tpd400.jsonl.teacher.json datasets/arms/mix3v2-6219-tpd400.jsonl.teacher.json datasets/arms/fiqa-plus-squadv2v2-18k-tpd400.jsonl.teacher.json; do
        [ -f "$f" ] && "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print('%s: scale %.1f, Pearson %.3f, Kendall tau %.3f, median margin %.2f, rows with neg > pos %d' % (sys.argv[1].split('/')[-1], d['scale'], d['within_row_pearson_vs_reference'] or 0, d['within_row_kendall_vs_reference'] or 0, d['median_margin'], d['rows_with_negative_above_positive']))" "$f"
    done
    echo '```'; echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped n1-5070 arm-distil-v9b-kl-n1v2 arm-dense-kl-n1v2-pd150 arm-dense-kl-n1v2-mmlwB arm-dense-kl-n1v2-ens arm-dense-kl-n1v2-pd400 arm-dense-kl-n1v2-ens400 soup-v9b --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-distil-v9b-kl-n1v2 --model arm-dense-kl-n1v2-pd150 --model arm-dense-kl-n1v2-ens --model arm-dense-kl-n1v2-pd400 --model arm-dense-kl-n1v2-ens400 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Queue state (v38b)"; echo; echo '```'
    grep -E "^(teacher|train|sparsity|regime:quantized):(pd400|arm-dense-kl-n1v2-pd400|arm-dense-kl-n1v2-ens400)" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$RESULTS_ROOT"/sparsity/arm-dense-kl-n1v2-pd400.json "$RESULTS_ROOT"/sparsity/arm-dense-kl-n1v2-ens400.json "$REGIMES_ROOT"/quantized/arm-dense-kl-n1v2-pd400 "$REGIMES_ROOT"/quantized/arm-dense-kl-n1v2-ens400 "$C/distil_dense_kl_n1v2_pd400.yaml" "$C/distil_dense_kl_n1v2_ens400.yaml" "$EXP_ROOT/queues/queue_v38b.sh" 2>/dev/null
git commit -q -m "Add the v38b results: PolDense-400M as the dense teacher on the n1 rows" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v38b.log" 2>&1 && { echo pushed >> "$LOGS/push_v38b.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v38b done ($(date '+%d.%m %H:%M')) ==="
