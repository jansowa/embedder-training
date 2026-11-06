#!/usr/bin/env bash
# v39 (6.11, after v36, ~4.5 h): the ensemble teacher (mean of soup-v9b dot products and PolDense-400M cosines scaled
# to soup-v9b's units) on the bigger row sets. v38 / v38b on the n1 rows: dense KD alone lifts tiny10 (63.4-63.5) but
# loses heldout8 (42.2-42.3 vs 42.6 for kl-n1v2); the ensemble keeps both (63.56 / 42.59 with pd150, 63.43 / 42.58 with
# pd400). The pd400 score files for n4 (mix3v2-6219) and n5 (fiqa-plus-squadv2v2-18k) were written by v38b, so no
# scoring here. Steps: 1 ensemble file n4 -> arm-dense-kl-n4v2-ens400 (kl-n4v2 recipe, fast head) -> sparsity, tiny10,
# heldout8; 2 the same on n5 -> arm-dense-kl-n5v2-ens400; 3 summary_v39.md, commit. Reading: ens on n4 >= 63.27 / 42.63
# and on n5 >= 63.39 / 42.80 = the ensemble pays with the dose -> soup-v12 member and the teacher for the judged MS MARCO
# rows; ens below kl on heldout8 at these doses = the dense member only helps on the small set (keep soup-v9b alone).
#   WAIT_FOR="v36" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v39.sh > logs/queue_v39.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v39 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v39.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v36}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }

log "=== v39 start: the ensemble teacher (soup-v9b + PolDense-400M) on the n4 and n5 rows; $(free_gb) GB free ==="

ensemble() {  # ensemble REF DENSE OUT: per-pair mean of the soup-v9b scores and the scaled dense scores (the v38 recipe)
    local ref="$1" dense="$2" out="$3"
    [ -s "$out" ] && { log "ensemble file present: $out"; return 0; }
    [ -s "$ref" ] && [ -s "$dense" ] || { log "ensemble: input missing ($ref, $dense)"; return 1; }
    "$PY" - "$ref" "$dense" "$out" <<'EOF' && log "ensemble teacher file written: $out" || { log "ensemble file FAILED: $out"; rm -f "$out"; return 1; }
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
}

arm() {  # arm TAG CONFIG REFLINE
    local tag="$1" cfg="$2" ref="$3"
    train "$tag" "$cfg"
    if [ -d "models/arm-$tag" ]; then
        sparsity "arm-$tag" "models/arm-$tag"
        regime quantized "arm-$tag" "models/arm-$tag" tiny10 16
        regime quantized "arm-$tag" "models/arm-$tag" heldout8 16
        log "  arm-$tag: $(lbs "arm-$tag") ($ref), $(dims "arm-$tag") dims/doc"
    fi
}

# 1. n4 rows (mix3v2-6219: fiqa 2073 + squadv2-v2 2073 + hotpotqa 2073)
ensemble datasets/arms/mix3v2-6219-tv9b.jsonl datasets/arms/mix3v2-6219-tpd400.jsonl datasets/arms/mix3v2-6219-tens400.jsonl \
    && arm dense-kl-n4v2-ens400 "$C/distil_dense_kl_n4v2_ens400.yaml" "kl-n4v2 from soup-v9b 63.27 / 42.63, n1 ens400 63.43 / 42.58"

# 2. n5 rows (fiqa 2073 + 16 000 gated squadv2-v2)
ensemble datasets/arms/fiqa-plus-squadv2v2-18k-tv9b.jsonl datasets/arms/fiqa-plus-squadv2v2-18k-tpd400.jsonl datasets/arms/fiqa-plus-squadv2v2-18k-tens400.jsonl \
    && arm dense-kl-n5v2-ens400 "$C/distil_dense_kl_n5v2_ens400.yaml" "kl-n5v2 from soup-v9b 63.39 / 42.80"

# 3. summary
SUM="$RESULTS_ROOT/summary_v39.md"
{
    echo "# v39 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): the ensemble teacher on the n4 and n5 rows"
    echo
    echo "Student sdadas/polish-splade, the kl-n4v2 / kl-n5v2 recipes (2 epochs, KL T=2, FLOPS kept, 2 x 16, 512 tokens, 10 negatives, fast head). Teacher = per-pair mean of soup-v9b's dot products and OPI-PIB/PolDense-400M's cosines scaled to soup-v9b's units (sidecars datasets/arms/*-tpd400.jsonl.teacher.json: n4 scale 102.2, Pearson 0.79; n5 scale 114.5, Pearson 0.84). On the n1 rows (v38 / v38b) the ensemble gave 63.56 / 42.59 (pd150) and 63.43 / 42.58 (pd400) against 63.25 / 42.62 for soup-v9b alone and 63.4-63.5 / 42.2-42.3 for the dense teacher alone."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-distil-v9b-kl-n4v2 arm-dense-kl-n4v2-ens400 arm-distil-v9b-kl-n5v2 arm-dense-kl-n5v2-ens400 arm-dense-kl-n1v2-ens400 soup-v9b --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-distil-v9b-kl-n4v2 --model arm-dense-kl-n4v2-ens400 --model arm-distil-v9b-kl-n5v2 --model arm-dense-kl-n5v2-ens400 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Queue state (v39)"; echo; echo '```'
    grep -E "^(train|sparsity|regime:quantized):(arm-dense-kl-n4v2-ens400|arm-dense-kl-n5v2-ens400)" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$RESULTS_ROOT"/sparsity/arm-dense-kl-n4v2-ens400.json "$RESULTS_ROOT"/sparsity/arm-dense-kl-n5v2-ens400.json "$REGIMES_ROOT"/quantized/arm-dense-kl-n4v2-ens400 "$REGIMES_ROOT"/quantized/arm-dense-kl-n5v2-ens400 "$C/distil_dense_kl_n4v2_ens400.yaml" "$C/distil_dense_kl_n5v2_ens400.yaml" "$EXP_ROOT/queues/queue_v39.sh" 2>/dev/null
git commit -q -m "Add the v39 results: the ensemble teacher on the n4 and n5 rows" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v39.log" 2>&1 && { echo pushed >> "$LOGS/push_v39.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v39 done ($(date '+%d.%m %H:%M')) ==="
