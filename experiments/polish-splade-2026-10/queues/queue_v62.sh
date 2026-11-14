#!/usr/bin/env bash
# Queue v62 (3060 Ti, after v61b; ~1.2 h per soup, up to ~7 h): weight soups of the data arms and the FREEZE of the full model.
# The arms dose3 (v56), faq20 (v58), dose3 + faq20 (v60), dose3 + fact20 (v61) and dose3 + gooaq (v61b) all start from
# soup-v12 with one recipe and differ only in their rows. Training on mixed rows diluted both gains (v60: FAQ tasks
# +0.63 instead of +1.04, the SQuAD group +0.36 instead of +0.74); the question is whether averaging the weights keeps
# them. Candidates (uniform average of the raw weights, before any bias):
#   soup-v20-dq   = dose3 + faq20
#   soup-v20-dqf  = dose3 + faq20 + fact20   (only when fact20 is not more than 0.2 below dose3 on either set)
#   soup-v20-dqm  = dose3 + faq20 + (dose3 + faq20 of v60)
#   soup-v20-dqg  = dose3 + faq20 + gooaq    (only when the GooAQ arm is not more than 0.2 below dose3 on either set)
#   soup-v20-dqfg = dose3 + faq20 + fact20 + gooaq (only when both passed)
# Each: the first bias form with <= 410 dims/doc (0.05 steps, as for the earlier soups), tiny10, heldout8, trec-covid,
# the query-side index cost, the task groups of summary_v60.md.
# FREEZE RULE: among dose3-bias015, fact20, the GooAQ arm and the soups, the highest mean of tiny10 and heldout8 with
# <= 410 dims/doc and <= 16 807 postings/query; a lead below 0.1 over dose3-bias015 = dose3-bias015 stays (the simpler
# recipe). A winner other than dose3 gets held-out-2 and the 23-task number (dose3: 58.73). The choice goes to
# logs/final_full_model.txt (first line = the label; the user may overwrite it before the next queue starts), to
# results/v62_freeze.json and summary_v62.md; the frozen model is copied into the package for the 5070.
# No new data arm, no seed soup. Nothing else is started here.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v62.sh > logs/queue_v62.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v62 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
echo $$ > "$LOGS/queue_v62.pid"
guard_alive
wait_for_queues ${WAIT_FOR:-v61 v61b}

M=arm-dense-kl-n5kj-mined-ens400-r2v12
D3=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12; FQ=arm-dense-kl-n5kj-mined-faq20-ens400-r2v12; DF=arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12
FA=arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12; G=arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12
COST="$RESULTS_ROOT/index_cost_v62.json"; FREEZE="$RESULTS_ROOT/v62_freeze.json"; VERDICT="$RESULTS_ROOT/v62_reading.json"
MAX_POSTINGS=${MAX_POSTINGS:-16807}; MAX_DIMS=410; MARGIN=0.1
D3_PICK=$(measured_form "$D3"); FQ_PICK=$(measured_form "$FQ"); DF_PICK=$(measured_form "$DF"); FA_PICK=$(measured_form "$FA"); G_PICK=$(measured_form "$G")
log "=== v62 start: soups and the freeze of the full model; dose3 ${D3_PICK:-none} $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK"); fact20 ${FA_PICK:-not measured} $(lbs "$FA_PICK"); gooaq ${G_PICK:-not measured} $(lbs "$G_PICK"); $(free_gb) GB free on / ==="
[ -n "$D3_PICK" ] && [ -d "models/$D3_PICK" ] && [ -d "models/$D3" ] && [ -d "models/$FQ" ] || { log "  the dose arm or the FAQ arm is missing: v62 stops"; exit 1; }
need_space || { log "  not enough space on /: v62 stops"; exit 1; }

near() {  # near PICK: 1 when the arm is measured and not more than 0.2 below dose3 on tiny10 and on heldout8
    [ -n "$1" ] && [ -d "models/${1%-bias0??}" ] && [ "$(gte "$(lbv "$1" tiny10)" "$(lbv "$D3_PICK" tiny10)" -0.2)" = 1 ] && [ "$(gte "$(lbv "$1" heldout8)" "$(lbv "$D3_PICK" heldout8)" -0.2)" = 1 ] && echo 1 || echo 0
}
FA_OK=$(near "$FA_PICK"); G_OK=$(near "$G_PICK")
log "  fact20 in the soups: $FA_OK ($(lbs "$FA_PICK")); gooaq in the soups: $G_OK ($(lbs "$G_PICK")); the bar is dose3 - 0.2 on both sets ($(lbs "$D3_PICK"))"

# 1. soups
log "=== v62 step 1: soups ==="
SOUPS=""
mk() {  # mk NAME DESCRIPTION MODEL...: soup, bias form, tiny10, heldout8, trec-covid
    local name="$1" desc="$2"; shift 2
    need_space || { log "  not enough space on /: $name skipped"; return 1; }
    soup "$name" "$@" || return 1
    at_cost "$name" "$desc; dose3 $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK")" || return 1
    SOUPS="$SOUPS $name=$PICK"; echo "$desc" > "$LOGS/soup_$name.desc"
}
mk soup-v20-dq "dose3 + faq20" "models/$D3" "models/$FQ"
[ "$FA_OK" = 1 ] && mk soup-v20-dqf "dose3 + faq20 + fact20" "models/$D3" "models/$FQ" "models/$FA"
[ -d "models/$DF" ] && mk soup-v20-dqm "dose3 + faq20 + (dose3 + faq20)" "models/$D3" "models/$FQ" "models/$DF"
[ "$G_OK" = 1 ] && mk soup-v20-dqg "dose3 + faq20 + gooaq" "models/$D3" "models/$FQ" "models/$G"
[ "$FA_OK" = 1 ] && [ "$G_OK" = 1 ] && mk soup-v20-dqfg "dose3 + faq20 + fact20 + gooaq" "models/$D3" "models/$FQ" "models/$FA" "models/$G"
SOUP_LABELS=""; for s in $SOUPS; do SOUP_LABELS="$SOUP_LABELS ${s#*=}"; done
index_cost "$COST" "$M-bias013" "$D3_PICK" "$FQ_PICK" "$DF_PICK" "$FA_PICK" "$G_PICK" $SOUP_LABELS

# 2. the freeze
log "=== v62 step 2: the freeze rule ==="
CANDS=("dose3 (v56)=$D3_PICK=$(dims "$D3_PICK")=dose3")
[ -n "$FA_PICK" ] && [ -n "$(lbv "$FA_PICK" heldout8)" ] && CANDS+=("dose3 + fact20 (v61)=$FA_PICK=$(dims "$FA_PICK")=dose3-fact20")
[ -n "$G_PICK" ] && [ -n "$(lbv "$G_PICK" heldout8)" ] && CANDS+=("dose3 + gooaq (v61b)=$G_PICK=$(dims "$G_PICK")=dose3-gooaq")
for s in $SOUPS; do n=${s%%=*}; l=${s#*=}; [ -n "$(lbv "$l" heldout8)" ] && CANDS+=("$n: $(cat "$LOGS/soup_$n.desc")=$l=$(dims "$l")=$(cat "$LOGS/soup_$n.desc")"); done
"$PY" scripts/freeze_choice.py --regimes "$REGIMES_ROOT/quantized" --cost "$COST" --out "$FREEZE" --max-dims "$MAX_DIMS" --max-postings "$MAX_POSTINGS" --margin "$MARGIN" \
    "${CANDS[@]}" 2>&1 | while read -r l; do log "  $l"; done
WIN=$(jget "$FREEZE" winner)
[ -n "$WIN" ] && [ -d "models/$WIN" ] || { log "  no freeze decision ($FREEZE): dose3 stays"; WIN="$D3_PICK"; }
BEST_OTHER=$(jget "$FREEZE" best_other)
if [ "$WIN" != "$D3_PICK" ]; then
    log "=== v62 step 3: held-out-2 of the frozen model $WIN ==="
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do h2task "$WIN" "$ts"; log "  $WIN $ts $(lbv "$WIN" "$ts")"; done
fi
{ echo "$WIN"; echo "# first line = the label of the frozen full model (models/<label>); written by queue v62 on $(date '+%d.%m %H:%M') by its rule: $(jget "$FREEZE" why)"; echo "# overwrite the first line to choose another model before queue r1 starts"; } > "$FINAL_FULL_FILE"
log "  frozen full model: $WIN -> $FINAL_FULL_FILE"
pack_add "full/$WIN" "models/$WIN" "full model (82 M), release form with the bias baked in; $(lbs "$WIN") / trec-covid $(tc "$WIN"), $(dims "$WIN") dims/doc"

# 4. summary
log "=== v62 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_v62.md"
COLS=(--column "dose3 (v56)=$D3_PICK" --column "faq20 (v58)=${FQ_PICK:-$FQ}" --column "dose3 + faq20 (v60)=${DF_PICK:-$DF}")
[ -n "$FA_PICK" ] && COLS+=(--column "dose3 + fact20 (v61)=$FA_PICK")
[ -n "$G_PICK" ] && COLS+=(--column "dose3 + gooaq (v61b)=$G_PICK")
for s in $SOUPS; do COLS+=(--column "${s%%=*} ($(cat "$LOGS/soup_${s%%=*}.desc"))=${s#*=}"); done
M23=(--m23 "$M-bias013" --m23 "$D3_PICK"); [ "$WIN" != "$D3_PICK" ] && M23+=(--m23 "$WIN")
PICKL=${BEST_OTHER:-$D3_PICK}; [ "$WIN" != "$D3_PICK" ] && PICKL="$WIN"
{
    echo "# v62 (quantized regime, nDCG@10): weight soups of the data arms and the freeze of the full model"; echo
    echo "The soups are uniform averages of the raw weights of arms that all start from soup-v12 with one recipe and differ only in their rows; each is measured in its first bias form with at most 410 dims/doc (steps of 0.05). fact20 joined the soups: $FA_OK; the GooAQ arm joined the soups: $G_OK (the bar: not more than 0.2 below dose3 on either set)."; echo
    echo "## Freeze"; echo
    echo "Rule: among dose3-bias015, fact20, the GooAQ arm and the soups, the highest mean of tiny10 and heldout8 with at most $MAX_DIMS dims/doc and at most $MAX_POSTINGS postings/query; a lead below $MARGIN over dose3-bias015 keeps dose3-bias015."; echo
    echo "**Frozen full model: \`$WIN\`** — $(jget "$FREEZE" why). The label is the first line of \`logs/final_full_model.txt\`; the model is in \`$PACK/full/$WIN\`."; echo
    "$PY" scripts/freeze_choice.py --table "$FREEZE"
    echo
    "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" \
        --column "20 k arm (v49)=$M-bias013" "${COLS[@]}" --pick "$PICKL" --ref "$D3_PICK" \
        --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
        --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
        --group "scifact-pl=scifact-pl" --group "maupqa-mkqa=maupqa-mkqa" --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
        --win "the best candidate other than dose3 is >= +0.2 on both sets over dose3-bias015 (the freeze itself follows the rule above, on the mean of the two sets)" \
        --noise "the best candidate other than dose3 is within the seed noise of dose3-bias015 on the two sets taken one by one (the freeze itself follows the rule above)" \
        --loss "the best candidate other than dose3 is a LOSS on both sets against dose3-bias015" "${M23[@]}" --json "$VERDICT"
    echo; echo "The Reading above compares \`$PICKL\` with dose3-bias015 set by set; the soups are measured at up to 410 dims/doc, dose3-bias015 at $(dims "$D3_PICK"), so the dims/doc and postings/query columns belong to the comparison. Nothing further was started by this queue: no new data arm, no seed soup."
    echo; echo "## Queue state (v62)"; echo; echo '```'; grep -E "soup-v20" "$STATE" | tail -n 60; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_v62.sh" "$EXP_ROOT/queues/release_common.sh" scripts/freeze_choice.py "$FREEZE" "$VERDICT" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null
for m in $SOUP_LABELS $WIN; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/soup-v20-*.json 2>/dev/null
git commit -q -m "Add the v62 results: weight soups of the data arms and the frozen full model" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop v62
log "=== v62 done ($(date '+%d.%m %H:%M')); frozen full model $WIN ==="
