#!/usr/bin/env bash
# 28.10, after queue_v13: negatives chosen by the student. Every file so far had negatives
# that are hard for a dense retriever / reranker pair and easy for SPLADE (the base model
# ranks the positive first in 90.6 % of the old rows). Here base polish-splade proposes the
# top-100 passages per query and roberta-v3 with the two percentile thresholds of
# mine-hard-negatives vetoes the false negatives (scripts/fiqa_splade_mine.sh). Two corpora
# as agreed with the user: the rows' own passages, and own + the PIRB fiqa-pl corpus
# (weaker translation, but 5x larger). Waits for "=== v13 done ===". Blocks:
#   D   data: own, plus, own-f02 (looser v3 floor)
#   T1  MNRL (cluster-like recipe) on own, KL scaleauto on own, MNRL on plus, MNRL on own-f02
#   H   held-out-8 for the best T1 arm when it reaches the base on tiny; leaderboard regime
#       (quantized tiny10 + heldout8) when it is within 0.3 of arm-cluster-like-fiqa
#   T2  candidates from soup-v2b (soup-own) + MNRL arm; soups of the best new arm with
#       soup-v2b (soup-v4a) and with models/n1-5070, the 5070 Ti FiQA+SQuAD run (soup-v4b)
#   N   tiny for the 5070 Ti runs n1-5070 / n2-5070 in our regime, held-out-8 for n1
# Start with: setsid nohup bash scripts/queue_v14.sh > logs/queue_v14.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v14 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v14.pid"

waited=0
until grep -qE '=== v13 done ===' logs/queue_v13.out 2>/dev/null; do
    [ "$waited" -ge 7200 ] && { log "v14: v13 not finished after 2 h; exiting"; exit 1; }
    sleep 60; waited=$((waited + 60))
done
wait_gpu_free

arm() {  # arm NAME: train configs/next/3060/<NAME with _>.yaml, tiny, sparsity, log the fitted scale
    local cfg="configs/next/3060/${1//-/_}.yaml"
    train "$1" "$cfg" && { bench "arm-$1" "models/arm-$1"; sparsity "arm-$1" "models/arm-$1"; }
    log "  arm-$1: tiny $(tiny "arm-$1") $(scale_of "arm-$1")"
}
best_arm() {  # best_arm NAME...: the name with the highest tiny result (empty when none has one)
    local c v best="" best_v=""
    for c in "$@"; do
        v=$(tiny "$c"); [ -n "$v" ] || continue
        if [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; then best="$c"; best_v="$v"; fi
    done
    echo "$best"
}
scale_of() {  # scale_of NAME: the fitted teacher score scale of a scaleauto arm, from its run dir
    local f; f=$(ls -1dt "runs/$1/sentence-transformers/splade"/*/score_scale.json 2>/dev/null | head -1)
    [ -n "$f" ] && "$PY" -c "
import json; d = json.load(open('$f'))
print('(scale alpha %.3f, student margin %.2f vs teacher %.2f, r %.2f)' % (d['alpha'], d['student_margin_mean'], d['teacher_margin_mean'], d['pearson']))" 2>/dev/null
}
reaches() { [ -n "$(tiny "$1")" ] && [ -n "$2" ] && [ "$(ge "$(tiny "$1")" "$2")" = 1 ]; }
data_report() {  # data_report VARIANT: one log line from the finalizer, mining and teacher-filter reports
    local v="$1" f="results/data/fiqa-2073-splade-$1"
    [ -f "$f.json" ] || { log "  $v: no report"; return; }
    "$PY" - "$f" <<'EOF' 2>/dev/null | while read -r line; do log "  $line"; done
import json, sys
f = sys.argv[1]
r = json.load(open(f + ".json")); m = json.load(open(f + "-mining.json")); t = json.load(open(f + "-teacher-filter.json"))
print(f"{f.split('/')[-1]}: rows {r.get('rows_written')} dropped {r.get('dropped')} tiers {r.get('negative_tiers')} "
      f"v3 neg median {r['v3_neg_scores']['median']:.2f} margin median {r['v3_min_pos_minus_max_neg']['median']:.2f} "
      f"neg>pos {r.get('queries_where_a_negative_outscores_a_positive')}")
print(f"student view: positive first {m['positive_is_first_share']:.3f}, in top10 {m['positive_in_top10_share']:.3f}, "
      f"extra share top10 {m.get('extra_share_top10')}; teacher rejected top10 {t.get('top10_rejected_by_teacher_share')}, "
      f"top20 {t.get('top20_rejected_by_teacher_share')}, budgets {t.get('final_budget_histogram')}")
EOF
}
mine() {  # mine VARIANT: scripts/fiqa_splade_mine.sh unless its file exists
    local v="$1" f="datasets/arms/fiqa-2073-splade-$1.jsonl" s=$SECONDS
    if [ -f "$f" ]; then log "data $v already present"; else
        VARIANT="$v" bash scripts/fiqa_splade_mine.sh > "$LOGS/fiqa_splade_$v.out" 2>&1 \
            && log "data $v done in $((SECONDS - s))s" || log "data $v FAILED after $((SECONDS - s))s (logs/fiqa_splade_$v.out)"
    fi
    data_report "$v"
}
soup() {  # soup NAME MODEL... : equal-weight soup into models/NAME, then tiny + sparsity
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    bench "$name" "models/$name"; sparsity "$name" "models/$name"
}

REF=$(tiny arm-cluster-like-fiqa); BASE=$(tiny base-splade-wrapped); NEAR=$(fsub "$REF" 0.3)
log "=== v14 start: MNRL reference ${REF:-?}, base ${BASE:-?}, $(free_gb) GB free on / ==="

log "=== v14 block D: student-mined negatives (own, plus, own-f02) ==="
for v in own plus own-f02; do mine "$v"; done

log "=== v14 block T1: arms on the student-mined negatives ==="
T1=()
[ -f datasets/arms/fiqa-2073-splade-own.jsonl ] && T1+=(mnrl-fiqa-splade-own kl-fiqa-splade-own-scaleauto)
[ -f datasets/arms/fiqa-2073-splade-plus.jsonl ] && T1+=(mnrl-fiqa-splade-plus)
[ -f datasets/arms/fiqa-2073-splade-own-f02.jsonl ] && T1+=(mnrl-fiqa-splade-own-f02)
for a in "${T1[@]}"; do arm "$a"; done
names=(); for a in "${T1[@]}"; do names+=("arm-$a"); done
top=$(best_arm "${names[@]}")
log "=== v14 block H: best T1 arm ${top:-none} (tiny $(tiny "${top:-x}"); MNRL reference ${REF:-?}, base ${BASE:-?}) ==="
if [ -n "$top" ] && reaches "$top" "$BASE"; then
    holdout "$top" "models/$top"
    if reaches "$top" "$NEAR"; then regime quantized "$top" "models/$top" tiny10 16; regime quantized "$top" "models/$top" heldout8 16
    else log "leaderboard regime skipped: $top is more than 0.3 below $REF on tiny"; fi
else log "block H skipped: no T1 arm reaches the base on tiny"; fi

log "=== v14 block T2: candidates from soup-v2b, soups ==="
mine soup-own
[ -f datasets/arms/fiqa-2073-splade-soup-own.jsonl ] && { arm mnrl-fiqa-splade-soup-own; names+=(arm-mnrl-fiqa-splade-soup-own); }
top2=$(best_arm "${names[@]}")
for c in "${names[@]}"; do log "  $c tiny $(tiny "$c")"; done
if [ -n "$top2" ]; then
    log "best new arm: $top2 (tiny $(tiny "$top2"))"
    soup "soup-v4a" "models/$top2" models/soup-v2b
    [ -d models/n1-5070 ] && soup "soup-v4b" "models/$top2" models/soup-v2b models/n1-5070 || log "models/n1-5070 absent; no soup-v4b"
    for s in soup-v4a soup-v4b; do
        [ -f "results/$s/final/metrics.json" ] || continue
        if reaches "$s" "$(tiny arm-cluster-like-fiqa)"; then holdout "$s" "models/$s"; regime quantized "$s" "models/$s" tiny10 16; regime quantized "$s" "models/$s" heldout8 16; fi
    done
    [ "$top2" != "$top" ] && reaches "$top2" "$BASE" && { holdout "$top2" "models/$top2"; reaches "$top2" "$NEAR" && { regime quantized "$top2" "models/$top2" tiny10 16; regime quantized "$top2" "models/$top2" heldout8 16; }; }
fi

log "=== v14 block N: the 5070 Ti runs in our regime ==="
for m in n1-5070 n2-5070; do [ -d "models/$m" ] && { bench "$m" "models/$m"; sparsity "$m" "models/$m"; }; done
[ -d models/n1-5070 ] && reaches n1-5070 "$NEAR" && holdout n1-5070 models/n1-5070

"$PY" scripts/queue_summary.py > results/summary_v14.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v14.md)"
log "=== v14 done ==="
