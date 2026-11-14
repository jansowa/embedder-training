#!/usr/bin/env bash
# Queue r1 (3060 Ti, after v62; ~8 h with one arm): the second round of the ALBERT student (50 M) and the choice of the
# ALBERT to release. The best ALBERT so far is arm-hl-full-s3n5kj (63.78 / 42.55 at 487 dims/doc: KD from soup-v9b,
# stage 3 on n5 + kj with the ensemble scores). The full model gained +1.2 / +1.1 from that point by a second round on
# rows with student-mined negatives and the SQuAD-v2 dose; the student gets the same round.
#   1  ARM arm-hl-full-s3n5kj-r2d3: from arm-hl-full-s3n5kj on datasets/arms/n5-plus-kj-mined-dose3.jsonl (the same
#      negatives and ensemble scores), the recipe of the full model's second round (2 epochs, LR 2e-5, 2 x 16, 512
#      tokens), the head path of the earlier ALBERT arms. When the frozen full model is a soup with the FAQ arm, a
#      second arm arm-hl-full-s3n5kj-r2d3faq on the dose3 + faq20 rows.
#   2  MEASURE each arm and the reference in the first bias form with <= 410 dims/doc: tiny10, heldout8, trec-covid,
#      the query-side index cost.
#   3  CHOICE by the rule of v62 (highest mean of tiny10 and heldout8 with <= 410 dims/doc and <= 16 807
#      postings/query; a lead below 0.1 keeps the reference arm-hl-full-s3n5kj). The winner: held-out-2 and the 23-task
#      number, and asymmetrically as the query encoder over the index of the frozen full model (earlier: 63.99 / 42.65
#      for arm-hl-r2-s3n5kj over r2v12-bias005). logs/final_albert_model.txt (first line = the label; the user may
#      overwrite it), the package for the 5070.
#   4  summary_r1.md, commit, push loop.
#   setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_r1.sh > logs/queue_r1.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=r1 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-21600}
source scripts/queue_lib.sh
MAIN=experiments/polish-splade-2026-10
source "$MAIN/queues/arm_common.sh"
source "$MAIN/queues/release_common.sh"
echo $$ > "$LOGS/queue_r1.pid"
guard_alive
C="$EXP_ROOT/configs/3060"
wait_for_queues ${WAIT_FOR:-v61 v61b v62}

mlbv() { regime_mean "$MAIN/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }   # the full models live in the main series
REF=arm-hl-full-s3n5kj
A1=arm-hl-full-s3n5kj-r2d3; A2=arm-hl-full-s3n5kj-r2d3faq
D3B=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015
FULL=$(final_full); [ -n "$FULL" ] && [ -d "models/$FULL" ] || { log "  no frozen full model in $FINAL_FULL_FILE (${FULL:-empty}): dose3-bias015 is used"; FULL="$D3B"; }
FTAG=$(echo "$FULL" | sed -e 's/^arm-dense-kl-n5kj-mined-//' -e 's/-ens400-r2v12//')
FAQ=0; case "$FULL" in soup-v20-*) FAQ=1 ;; esac; FAQ=${R1_FAQ_ARM:-$FAQ}
COST="$RESULTS_ROOT/index_cost_r1.json"; FREEZE="$RESULTS_ROOT/r1_freeze.json"; VERDICT="$RESULTS_ROOT/r1_reading.json"
MAX_POSTINGS=${MAX_POSTINGS:-16807}; MAX_DIMS=410; MARGIN=0.1
log "=== r1 start: round 2 of the ALBERT student; reference $REF $(lbs $REF) at $(dims $REF) dims/doc; frozen full model $FULL $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8); FAQ arm: $FAQ; $(free_gb) GB free on / ==="
for f in "$C/stage3_hl_full_r2_dose3.yaml" datasets/arms/n5-plus-kj-mined-dose3.jsonl scripts/freeze_choice.py scripts/arm_reading.py; do [ -s "$f" ] || { log "  $f missing: r1 stops"; exit 1; }; done
[ -d "models/$REF" ] || { log "  models/$REF missing: r1 stops"; exit 1; }
need_space || { log "  not enough space on /: r1 stops"; exit 1; }

fix8() {  # fix8 LABEL: when heldout8 came back with 7 tasks (fiqa-pl failed), run fiqa-pl alone and fold it in
    local label="$1" n; n=$(regime_mean "$REGIMES_ROOT/quantized/$label/heldout8.json" 2>/dev/null | awk '{print $2}')
    [ "${n:-8}" = 7 ] || return 0
    log "  $label: heldout8 has 7 tasks; fiqa-pl alone"
    regime quantized "$label" "models/$label" heldout8_fiqa 16 && "$PY" scripts/merge_regime_task.py --into "$REGIMES_ROOT/quantized/$label/heldout8.json" \
        --from "$REGIMES_ROOT/quantized/$label/heldout8_fiqa.json" --n-tasks 8 | while read -r l; do log "  $l"; done
}

# 1. the arms
log "=== r1 step 1: round 2 on the dose rows ==="
train "${A1#arm-}" "$C/stage3_hl_full_r2_dose3.yaml"
if [ "$FAQ" = 1 ] && [ -s datasets/arms/n5-plus-kj-mined-dose3-faq20.jsonl ]; then
    log "  the frozen full model is a soup with the FAQ arm: the second ALBERT arm on dose3 + faq20"
    train "${A2#arm-}" "$C/stage3_hl_full_r2_dose3_faq20.yaml"
fi

# 2. measure
log "=== r1 step 2: measure at <= 410 dims/doc ==="
at_cost "$REF" "unbiased $(lbs $REF) at $(dims $REF)"; REF_PICK=${PICK:-}; [ -n "$REF_PICK" ] && fix8 "$REF_PICK"
A1_PICK=""; A2_PICK=""
[ -d "models/$A1" ] && { at_cost "$A1" "reference $REF_PICK $(lbs "$REF_PICK") / $(tc "$REF_PICK") at $(dims "$REF_PICK")"; A1_PICK=${PICK:-}; fix8 "$A1_PICK"; }
[ -d "models/$A2" ] && { at_cost "$A2" "reference $REF_PICK $(lbs "$REF_PICK"); round 2 on dose3 $(lbs "$A1_PICK")"; A2_PICK=${PICK:-}; fix8 "$A2_PICK"; }
index_cost "$COST" "$REF_PICK" "$A1_PICK" "$A2_PICK" "$FULL"

# 3. the choice
log "=== r1 step 3: the ALBERT to release ==="
CANDS=("stage 3 on n5 + kj (v45)=$REF_PICK=$(dims "$REF_PICK")=n5kj")
[ -n "$A1_PICK" ] && [ -n "$(lbv "$A1_PICK" heldout8)" ] && CANDS+=("round 2 on dose3 (r1)=$A1_PICK=$(dims "$A1_PICK")=dose3")
[ -n "$A2_PICK" ] && [ -n "$(lbv "$A2_PICK" heldout8)" ] && CANDS+=("round 2 on dose3 + faq20 (r1)=$A2_PICK=$(dims "$A2_PICK")=dose3 + faq20")
"$PY" scripts/freeze_choice.py --regimes "$REGIMES_ROOT/quantized" --cost "$COST" --out "$FREEZE" --max-dims "$MAX_DIMS" --max-postings "$MAX_POSTINGS" --margin "$MARGIN" \
    "${CANDS[@]}" 2>&1 | while read -r l; do log "  $l"; done
WIN=$(jget "$FREEZE" winner)
[ -n "$WIN" ] && [ -d "models/$WIN" ] || { log "  no choice ($FREEZE): the reference form stays"; WIN="$REF_PICK"; }
[ -n "$WIN" ] && [ -d "models/$WIN" ] || { log "  no measured ALBERT form: r1 stops"; exit 1; }
{ echo "$WIN"; echo "# first line = the label of the ALBERT to release (models/<label>); written by queue r1 on $(date '+%d.%m %H:%M') by its rule: $(jget "$FREEZE" why)"; echo "# overwrite the first line to choose another model before queue x1 starts"; } > "$FINAL_ALBERT_FILE"
log "  ALBERT to release: $WIN -> $FINAL_ALBERT_FILE"
for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do h2task "$WIN" "$ts"; log "  $WIN $ts $(lbv "$WIN" "$ts")"; done
ASYM="asym-$FTAG-q-$WIN"
PIRB_QUERY_MODEL="models/$WIN" regime quantized "$ASYM" "models/$FULL" tiny10 16
PIRB_QUERY_MODEL="models/$WIN" regime quantized "$ASYM" "models/$FULL" heldout8 16
log "  $WIN as the query encoder over the index of $FULL: $(lbs "$ASYM") (symmetric: ALBERT $(lbs "$WIN"), full model $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8); earlier arm-hl-r2-s3n5kj over r2v12-bias005 63.99 / 42.65)"
pack_add "albert/$WIN" "models/$WIN" "ALBERT student (50 M), release form with the bias baked in; $(lbs "$WIN") / trec-covid $(tc "$WIN"), $(dims "$WIN") dims/doc; as the query encoder over $FULL: $(lbs "$ASYM")"

# 4. summary
log "=== r1 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_r1.md"
COLS=(--column "stage 3 on n5 + kj, unbiased (v45)=$REF" --column "stage 3 on n5 + kj (v45)=$REF_PICK")
[ -n "$A1_PICK" ] && COLS+=(--column "round 2 on dose3 (r1)=$A1_PICK")
[ -n "$A2_PICK" ] && COLS+=(--column "round 2 on dose3 + faq20 (r1)=$A2_PICK")
PICKL=${A1_PICK:-$REF_PICK}; [ "$WIN" != "$REF_PICK" ] && PICKL="$WIN"
{
    echo "# r1 (quantized regime, nDCG@10): the second round of the ALBERT student and the ALBERT to release"; echo
    echo "Round 2 = from \`$REF\` (KD from soup-v9b, stage 3 on n5 + kj) on the rows of the full model's dose arm (\`datasets/arms/n5-plus-kj-mined-dose3.jsonl\`: student-mined negatives, the SQuAD-v2 dose, ensemble teacher scores), the recipe of the full model's second round. Every model is measured in its first bias form with at most 410 dims/doc (steps of 0.05). Frozen full model: \`$FULL\` ($(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8)). The second arm on dose3 + faq20 runs only when the frozen full model is a soup with the FAQ arm: $FAQ."; echo
    echo "## Choice"; echo
    echo "Rule (as for the full model in v62): the highest mean of tiny10 and heldout8 with at most $MAX_DIMS dims/doc and at most $MAX_POSTINGS postings/query; a lead below $MARGIN keeps the reference."; echo
    echo "**ALBERT to release: \`$WIN\`** — $(jget "$FREEZE" why). The label is the first line of \`logs/final_albert_model.txt\`; the model is in \`$PACK/albert/$WIN\`."; echo
    "$PY" scripts/freeze_choice.py --table "$FREEZE"
    echo; echo "As the query encoder over the index of the frozen full model (\`$ASYM\`): **$(lbs "$ASYM")** (tiny10 / heldout8); symmetric ALBERT $(lbs "$WIN"), symmetric full model $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8). Earlier: arm-hl-r2-s3n5kj over r2v12-bias005 63.99 / 42.65."; echo
    "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" \
        "${COLS[@]}" --pick "$PICKL" --ref "$REF_PICK" \
        --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
        --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
        --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
        --win "THE SECOND ROUND PAYS FOR THE STUDENT (>= +0.2 on both sets over the stage-3 arm at <= 410 dims/doc)" \
        --noise "within the seed noise of the stage-3 arm set by set (the choice itself follows the rule above, on the mean of the two sets)" \
        --loss "a LOSS on both sets against the stage-3 arm" --m23 "$WIN" --json "$VERDICT"
    echo; echo "Nothing further was started by this queue."
    echo; echo "## Queue state (r1)"; echo; echo '```'; grep -E "hl-full-s3n5kj-r2d3|hl-full-s3n5kj-bias|asym-$FTAG-q-" "$STATE" | tail -n 60; echo '```'
} > "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-900 | while read -r l; do log "  READING: $l"; done
git add "$SUM" "$EXP_ROOT/queues/queue_r1.sh" "$C/stage3_hl_full_r2_dose3.yaml" "$C/stage3_hl_full_r2_dose3_faq20.yaml" "$FREEZE" "$VERDICT" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null
for m in $REF_PICK $A1_PICK $A2_PICK $A1 $A2 "$ASYM"; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/arm-hl-full-s3n5kj*.json 2>/dev/null
git commit -q -m "Add the r1 results: the second round of the ALBERT student on the dose rows" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop r1
log "=== r1 done ($(date '+%d.%m %H:%M')); ALBERT to release $WIN ==="
