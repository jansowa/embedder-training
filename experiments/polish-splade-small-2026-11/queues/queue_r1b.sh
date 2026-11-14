#!/usr/bin/env bash
# Queue r1b (3060 Ti, 14.11, ~25 min): the short end of r1. On 14.11 the user decided that the ALBERT to release must
# descend from the frozen full model (a new lineage, queue a1), so r1 was stopped by hand at 08:41 during the heldout8
# of its first arm. Both arms of r1 are trained; what is still needed from them is the tiny10 number, which picks the
# rows of the second round of a1, and the query-side cost.
#   1  wait for the evaluation r1 left running (heldout8 of arm-hl-full-s3n5kj-r2d3-bias020), keep its result
#   2  arm-hl-full-s3n5kj-r2d3faq: the first bias form with <= 410 dims/doc, tiny10 only
#   3  the query-side index cost of the reference form and of both arms
#   4  summary_r1.md as a partial result, commit. No heldout8 / trec-covid / held-out-2 / asymmetric run of the second
#      arm, no choice: logs/final_albert_model.txt is NOT written here.
#   setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_r1b.sh > logs/queue_r1b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=r1b MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
MAIN=experiments/polish-splade-2026-10
source "$MAIN/queues/arm_common.sh"
source "$MAIN/queues/release_common.sh"
echo $$ > "$LOGS/queue_r1b.pid"
guard_alive
C="$EXP_ROOT/configs/3060"

mlbv() { regime_mean "$MAIN/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
REF=arm-hl-full-s3n5kj; REF_PICK=arm-hl-full-s3n5kj-bias010
A1=arm-hl-full-s3n5kj-r2d3; A1_PICK=arm-hl-full-s3n5kj-r2d3-bias020; A2=arm-hl-full-s3n5kj-r2d3faq
FULL=$(final_full)
COST="$RESULTS_ROOT/index_cost_r1.json"
log "=== r1b start: the short end of r1 (tiny10 and the cost of both round-2 arms); frozen full model $FULL; $(free_gb) GB free on / ==="

# 1. the evaluation r1 left behind
if pgrep -f "pirb_regime_eval.sh quantized $A1_PICK" > /dev/null; then
    log "  waiting for the heldout8 of $A1_PICK that r1 left running"
    while pgrep -f "pirb_regime_eval.sh quantized $A1_PICK" > /dev/null; do sleep 20; done; sleep 20
fi
H8=$(regime_mean "$REGIMES_ROOT/quantized/$A1_PICK/heldout8.json" 2>/dev/null)
if [ -n "$H8" ]; then log "  $A1_PICK: $(lbs "$A1_PICK"), $(dims "$A1_PICK") dims/doc (heldout8 tasks: $(echo "$H8" | awk '{print $2}'))"; record "regime:quantized:$A1_PICK:heldout8" ok "$H8" "finished after the stop of r1"
else log "  $A1_PICK: no heldout8 result (not repeated here)"; rm -rf "cache/regimes/quantized_${A1_PICK}_heldout8"; fi

# 2. the second arm: tiny10 only
log "=== r1b step 2: tiny10 of $A2 at <= 410 dims/doc ==="
A2_PICK=""
if [ -d "models/$A2" ]; then
    sparsity "$A2" "models/$A2"; A2_PICK="$A2"
    if [ "$(gte 410 "$(dims "$A2")")" != 1 ]; then
        LIST="005 010 015 020 025 030"
        for t in $LIST; do
            biased "models/$A2" "$A2-bias$t" "0.${t#0}" || break
            sparsity "$A2-bias$t" "models/$A2-bias$t"; A2_PICK="$A2-bias$t"
            [ "$(gte 410 "$(dims "$A2_PICK")")" = 1 ] && break
        done
        for t in $LIST; do [ "$A2-bias$t" != "$A2_PICK" ] && rm -rf "models/$A2-bias$t"; done
    fi
    log "  $A2: $(dims "$A2") dims/doc unbiased; measured form $A2_PICK at $(dims "$A2_PICK") (cap 410)"
    regime quantized "$A2_PICK" "models/$A2_PICK" tiny10 16
    log "  $A2_PICK: tiny10 $(lbv "$A2_PICK" tiny10), $(dims "$A2_PICK") dims/doc (round 2 on dose3: $(lbv "$A1_PICK" tiny10) at $(dims "$A1_PICK"); reference $(lbv "$REF_PICK" tiny10) at $(dims "$REF_PICK"))"
else log "  models/$A2 missing"; fi

# 3. the cost
index_cost "$COST" "$REF_PICK" "$A1_PICK" "$A2_PICK"
cget() { "$PY" -c "import json,sys;m=json.load(open(sys.argv[1]))['models'].get(sys.argv[2],{});v=m.get(sys.argv[3]);print(sys.argv[4] % v if v is not None else '-')" "$COST" "$1" "$2" "$3" 2>/dev/null || echo "-"; }
MCOST="$MAIN/results/index_cost_v62.json"
fcget() { "$PY" -c "import json,sys;m=json.load(open(sys.argv[1]))['models'].get(sys.argv[2],{});v=m.get(sys.argv[3]);print(sys.argv[4] % v if v is not None else '-')" "$MCOST" "$1" "$2" "$3" 2>/dev/null || echo "-"; }
v() { local x; x=$(lbv "$1" "$2"); echo "${x:--}"; }

# 4. summary (partial)
log "=== r1b step 3: summary ==="
SUM="$RESULTS_ROOT/summary_r1.md"
T1=$(lbv "$A1_PICK" tiny10); T2=""; [ -n "$A2_PICK" ] && T2=$(lbv "$A2_PICK" tiny10)
ROWS="none measured"
if [ -n "$T1" ] && [ -n "$T2" ]; then if [ "$(gte "$T2" "$T1")" = 1 ] && [ "$T1" != "$T2" ]; then ROWS="dose3 + faq20"; else ROWS="dose3"; fi; elif [ -n "$T1" ]; then ROWS="dose3"; fi
row() {  # row NAME LABEL ROWS
    echo "| $1 | \`$2\` | $3 | $(v "$2" tiny10) | $(v "$2" heldout8) | $(v "$2" heldout2_treccovid) | $(dims "$2") | $(cget "$2" doc_active_mean %.1f) | $(cget "$2" query_active_mean %.1f) | $(cget "$2" postings_per_query %.0f) |"
}
{
    echo "# r1 (quantized regime, nDCG@10): the second round of the ALBERT student — partial result"; echo
    echo "Round 2 = from \`$REF\` (student cut from soup-v9b, KD from soup-v9b, stage 3 on n5 + kj) with the recipe of the full model's second round (2 epochs, lr 2e-5, 2 x 16, 512 tokens, KL to the ensemble scores stored in the rows) on the rows of the full model's dose arm (\`datasets/arms/n5-plus-kj-mined-dose3.jsonl\`) and, as a second arm, on dose3 + faq20 (\`datasets/arms/n5-plus-kj-mined-dose3-faq20.jsonl\`). Every model is measured in its first bias form with at most 410 dims/doc (steps of 0.05)."; echo
    echo "**The queue was cut short on 14.11.** The user decided that the ALBERT to release must descend from the frozen full model (\`$FULL\`), not from soup-v9b: a new lineage (queue a1: student cut from the frozen model, KD from it, stage 3, round 2). r1 was stopped by hand at 08:41 during the heldout8 of the first arm, and queue r1b added only the tiny10 of the second arm and the query-side cost. Not measured for the second arm: heldout8, trec-covid; for neither arm: held-out-2, the asymmetric run. No ALBERT was chosen here and \`logs/final_albert_model.txt\` was not written; the choice is made by a1, which measures what is missing for the better arm of this table."; echo
    echo "| model | label | rows of the last training | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |"
    echo "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    row "stage 3 on n5 + kj (v45), the reference" "$REF_PICK" "n5 + kj (20 292)"
    row "round 2 on dose3 (r1)" "$A1_PICK" "dose3 (60 385)"
    [ -n "$A2_PICK" ] && row "round 2 on dose3 + faq20 (r1)" "$A2_PICK" "dose3 + faq20"
    echo "| frozen full model (82 M), for scale | \`$FULL\` | - | $(mlbv "$FULL" tiny10) | $(mlbv "$FULL" heldout8) | $(mlbv "$FULL" heldout2_treccovid) | $(sparsity_mean "$MAIN/results/sparsity/$FULL.json" "$FULL" 2>/dev/null) | $(fcget "$FULL" doc_active_mean %.1f) | $(fcget "$FULL" query_active_mean %.1f) | $(fcget "$FULL" postings_per_query %.0f) |"
    echo
    echo "Unbiased dims/doc: reference $(dims "$REF"), round 2 on dose3 $(dims "$A1"), round 2 on dose3 + faq20 $(dims "$A2"): the second round makes the student much denser, so its measured form needs a larger bias shift than the reference (bias010)."; echo
    echo "tiny10 against the reference at <= 410 dims/doc: round 2 on dose3 $([ -n "$T1" ] && fsub "$T1" "$(lbv "$REF_PICK" tiny10)" || echo -), round 2 on dose3 + faq20 $([ -n "$T2" ] && fsub "$T2" "$(lbv "$REF_PICK" tiny10)" || echo -). The better arm on tiny10 gives the rows of the second round of a1: **$ROWS**. One seed; the seed noise of this recipe is 0.13-0.2."; echo
    echo "Training time: round 2 on dose3 9 686 s, on dose3 + faq20 12 299 s."
    echo; echo "## Queue state (r1, r1b)"; echo; echo '```'; grep -E "hl-full-s3n5kj-r2d3|hl-full-s3n5kj-bias" "$STATE" | tail -n 40; echo '```'
} > "$SUM"
log "summary written to $SUM (rows for the second round of a1: $ROWS)"
git add "$SUM" "$EXP_ROOT/queues/queue_r1.sh" "$EXP_ROOT/queues/queue_r1b.sh" "$C/stage3_hl_full_r2_dose3.yaml" "$C/stage3_hl_full_r2_dose3_faq20.yaml" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null
for m in $REF_PICK $A1_PICK $A2_PICK $A1 $A2; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/arm-hl-full-s3n5kj*.json 2>/dev/null
git commit -q -m "Add the partial r1 results: tiny10 and the cost of the two round-2 ALBERT arms" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop r1b
log "=== r1b done ($(date '+%d.%m %H:%M')); tiny10: dose3 ${T1:--}, dose3 + faq20 ${T2:--}; rows for a1: $ROWS ==="
