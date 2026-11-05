#!/usr/bin/env bash
# v38 (5.11, ~2 h): distillation from DENSE teachers. Until now the SPLADE student was distilled from SPLADE
# teachers (works: kl-n1v2 63.25 vs MNRL 62.98 on the n1 rows) and from Polish cross-encoders (roberta-v3 on
# fiqa: -0.4 / -0.3 vs MNRL after score scaling, PLAN-2026-10-27 §8). A dense bi-encoder scores with the same
# functional form as the student (dot product of two vectors) and gives graded similarities over all 10 negatives
# instead of the cross-encoder's near one-hot rows. Teachers available offline (the HF CDN is unreachable today):
#   OPI-PIB/PolDense-150M            dense ModernBERT, 60.4 on our 23 PIRB tasks (soup-v9b ~57.6)  "strong"
#   sdadas/mmlw-retrieval-roberta-base  dense RoBERTa, 57.8 = parity with the soup                 "near"
# Steps: 1 score the n1 rows (fiqa 2073 + squadv2-v2 2073) with both teachers (kitchen bridge/dense_teacher_scores.py,
# cosines scaled to the SPLADE teacher's units by a within-row least-squares fit against the -tv9b file; the sidecar
# records the agreement with soup-v9b), 2 the ensemble file (mean of v9b and PolDense per pair), 3 three arms of the
# kl-n1v2 recipe (fast head): pd150, mmlwB, ens -> sparsity, tiny10; heldout8 for any arm >= 63.4 on tiny10,
# 4 summary_v38.md, commit. Reading: an arm above kl-n1v2 (63.25) by > 0.3 on tiny10 = dense knowledge transfers;
# per task: gains expected on arguana / nfcorpus / scifact, losses on gpt-exams / templates would mean the lexical
# side needs the MNRL anchor (next: KL + MNRL combined loss).
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v38.sh > logs/queue_v38.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v38 MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-3600}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v38.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN=${KITCHEN:-$HOME/work/speakleash/retrieval-data-kitchen}
ROWS=datasets/arms/fiqa-plus-squadv2v2.jsonl
REF=datasets/arms/fiqa-plus-squadv2v2-tv9b.jsonl
H8_MIN=${H8_MIN:-63.4}

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v37}; do
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
[ -f "$ROWS" ] && [ -f "$REF" ] || { log "rows or reference missing ($ROWS, $REF)"; exit 1; }

log "=== v38 start: dense-teacher distillation on the n1 rows (PolDense-150M, mmlw-retrieval-roberta-base, ensemble); $(free_gb) GB free ==="

# 1. teacher scores
dense_scores() {  # dense_scores TAG TEACHER PRESET DTYPE
    local out="datasets/arms/fiqa-plus-squadv2v2-t$1.jsonl"
    [ -s "$out" ] && [ -f "$out.teacher.json" ] && { log "teacher scores $1: already present"; return 0; }
    wait_gpu_free; wait_ram
    log "teacher scores $1 ($2, preset $3, $4) -> $out"
    local s=$SECONDS
    timeout -k 60 3600 "$PY" "$KITCHEN/bridge/dense_teacher_scores.py" --teacher "$2" --preset "$3" --dtype "$4" \
        --rows "$ROWS" --reference "$REF" --output "$out" --max-seq-length 512 --batch-size 32 --device cuda \
        > "$LOGS/teacher_scores_$1.log" 2>&1
    local rc=$? line
    line=$(grep -E "^[0-9]+ rows ->" "$LOGS/teacher_scores_$1.log" | tail -1)
    if [ "$rc" -eq 0 ] && [ -s "$out" ]; then log "teacher scores $1 done in $((SECONDS - s))s: ${line#* -> }"; record "teacher:$1" ok "-" "$((SECONDS - s))s"
    else log "teacher scores $1 FAILED (rc=$rc, logs/teacher_scores_$1.log)"; record "teacher:$1" FAILED "-" "$((SECONDS - s))s"; rm -f "$out"; return 1; fi
}
dense_scores pd150 OPI-PIB/PolDense-150M poldense bf16
dense_scores mmlwB sdadas/mmlw-retrieval-roberta-base mmlw fp16

# 2. the ensemble file: mean of soup-v9b's dot products and PolDense's scaled cosines, pair by pair
ENS=datasets/arms/fiqa-plus-squadv2v2-tens.jsonl
if [ ! -s "$ENS" ] && [ -s datasets/arms/fiqa-plus-squadv2v2-tpd150.jsonl ]; then
    "$PY" - "$REF" datasets/arms/fiqa-plus-squadv2v2-tpd150.jsonl "$ENS" <<'EOF' && log "ensemble teacher file written: $ENS" || log "ensemble file FAILED"
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

# 3. the arms
for a in pd150 mmlwB ens; do
    TAG="dense-kl-n1v2-$a"
    [ -s "datasets/arms/fiqa-plus-squadv2v2-t$a.jsonl" ] || { log "no teacher file for $a; skipping the arm"; continue; }
    train "$TAG" "$C/distil_dense_kl_n1v2_$a.yaml"
    if [ -d "models/arm-$TAG" ]; then
        sparsity "arm-$TAG" "models/arm-$TAG"
        regime quantized "arm-$TAG" "models/arm-$TAG" tiny10 16
        log "  arm-$TAG: tiny10 $(lbv "arm-$TAG" tiny10) (kl-n1v2 63.25, n1-5070 MNRL 62.98), $(dims "arm-$TAG") dims/doc"
        t=$(lbv "arm-$TAG" tiny10)
        if [ -n "$t" ] && [ "$(ge "$t" "$H8_MIN")" = 1 ]; then
            regime quantized "arm-$TAG" "models/arm-$TAG" heldout8 16
            log "  arm-$TAG: $(lbs "arm-$TAG") (kl-n1v2 63.25 / 42.62)"
        fi
    fi
done

# 4. summary
SUM="$RESULTS_ROOT/summary_v38.md"
{
    echo "# v38 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): distillation from dense teachers on the n1 rows"
    echo
    echo "Student sdadas/polish-splade, the kl-n1v2 recipe (4 146 rows: fiqa 2073 + squadv2-v2 2073, 2 epochs, KL T=2, FLOPS kept, 2 x 16, fast head). Teachers: pd150 = OPI-PIB/PolDense-150M (dense, 60.4 on our 23 PIRB tasks), mmlwB = sdadas/mmlw-retrieval-roberta-base (57.8, parity with soup-v9b), ens = mean of soup-v9b and PolDense per pair. Dense cosines were put on the SPLADE teacher's scale by a within-row least-squares fit against the -tv9b file (sidecars datasets/arms/*.teacher.json: scale, agreement with soup-v9b, margins)."
    echo
    echo "## Teacher agreement with soup-v9b (within-row, over the 4 146 rows)"; echo; echo '```'
    for a in pd150 mmlwB; do f="datasets/arms/fiqa-plus-squadv2v2-t$a.jsonl.teacher.json"; [ -f "$f" ] && "$PY" -c "import json;d=json.load(open('$f'));print('$a: scale %.1f, Pearson %.3f, Kendall tau %.3f, median margin %.2f, rows with neg > pos %d' % (d['scale'], d['within_row_pearson_vs_reference'] or 0, d['within_row_kendall_vs_reference'] or 0, d['median_margin'], d['rows_with_negative_above_positive']))"; done
    echo '```'; echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped n1-5070 arm-distil-v9b-kl-n1v2 arm-dense-kl-n1v2-pd150 arm-dense-kl-n1v2-mmlwB arm-dense-kl-n1v2-ens soup-v9b --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-distil-v9b-kl-n1v2 --model arm-dense-kl-n1v2-pd150 --model arm-dense-kl-n1v2-mmlwB --model arm-dense-kl-n1v2-ens 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Queue state (v38)"; echo; echo '```'
    grep -E "^(teacher|train|sparsity|regime:quantized):(pd150|mmlwB|arm-dense-kl-n1v2)" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$RESULTS_ROOT"/sparsity/arm-dense-kl-n1v2-*.json "$REGIMES_ROOT"/quantized/arm-dense-kl-n1v2-* 2>/dev/null
git commit -q -m "Add the v38 results: SPLADE distillation from dense teachers" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin master > "$LOGS/push_v38.log" 2>&1 && { echo pushed >> "$LOGS/push_v38.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v38 done ($(date '+%d.%m %H:%M')) ==="
