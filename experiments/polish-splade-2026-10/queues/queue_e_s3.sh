#!/usr/bin/env bash
# Queue e, stage s3 (about 1.5 h, up to 5 h): what was left open for the presentation deadline.
#   1  the 23 tasks of a new model A that passed its gate in s1, when s1 had no time for them
#   2  the soup R-c' + fact20 of the 124 M line, merged from copies with both bias keys (queue b2 merged it with the bias
#      of the R-c' lineage read as zero; see queue_e.sh), first form within the limits, the 19 tasks
#   3  the 23 tasks of the best 124 M candidate when it beats R-c' after the dose3 round on the 19 tasks
# The 124 M numbers go into summary_b2.md. Run by queue_e.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=es3 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
source "$EXP_ROOT/queues/e_common.sh"
guard_alive
FA=arm-b1-rc-fact20; S2=soup2-b1-rcd-fact20; RC=$(measured_form "$RCD")
log "=== e s3 start: the 23 tasks of a new A (when one passed), the soup R-c' + fact20 with its bias; $(left_min) min before the deadline ==="
AW=$(first_line "$EDIR/a_winner.txt" 2>/dev/null)
if [ -n "$AW" ] && [ -d "models/$AW" ] && [ -z "$(m23 "$AW")" ]; then
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$AW" "$ts"; done
    log "  23 tasks of $AW: $(m23 "$AW") against $(m23 "$FULL") of the frozen model"
    { echo "## s3: the 23 tasks of the candidate for A"; echo; echo "\`$AW\`: $(m23 "$AW") against $(m23 "$FULL") of the frozen model (\`$FULL\`)."; } > "$EDIR/notes_s3.md"
    EXTRA_LABELS="$AW" esummary "Add the e results: the 23 tasks of the candidate for model A"
fi
pause_point
SP=""
if [ -d "models/$FA" ] && [ -d "models/$RCD" ]; then
    biased2 "$RCD" "$RCD-bias000" 0.00 && to_dd "$RCD-bias000"
    biased2 "$FA" "$FA-bias000" 0.00 && to_dd "$FA-bias000"
    if mksoup "$S2" uniform "$RCD-bias000" "$FA-bias000"; then
        log "  $(grep -E "^<merged>" "$LOGS/soup_$S2.log" | head -n 1 | cut -c1-160) (active dims of the merged model on the check sentences; b2's merge without the bias: 786 to 2793)"
        to_dd "$S2"
        if form_fit2 "$S2"; then SP="$PICK"; measure19 "$SP"; cand p2 "soup R-c' + fact20 (merged with the bias kept)" "$SP" "$S2" "uniform average of R-c' after the dose3 round and the fact20 arm, both from the start of R-c'"; fi
    fi
fi
pause_point
BEST=$(best_cand --metric m19 --exclude "$FULL")
BL=$(echo "$BEST" | cut -f1); N23=""
if [ -n "$BL" ] && [ "$BL" != "$RC" ] && [ "$(gte "$(m19 "$BL")" "$(m19 "$RC")" 0.0001)" = 1 ]; then
    log "  $BL ($(m19 "$BL")) beats R-c' ($(m19 "$RC")) on the 19 tasks: its 23 tasks"
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$BL" "$ts"; done
    N23="23 tasks of \`$BL\`: $(m23 "$BL") (R-c' $(m23 "$RC"), frozen model $(m23 "$FULL"))."
else log "  no 124 M candidate beats R-c' ($RC $(m19 "$RC")) on the 19 tasks: no further 23 tasks"; N23="No 124 M candidate beats R-c' on the 19 tasks, so no further 23 tasks were measured."; fi
if ! grep -q "^### Added 18.11" "$B2/notes_p2.md" 2>/dev/null; then
    {
        echo; echo "### Added 18.11: the soups of queue b2 were merged without the bias"; echo
        echo "A model saved by the trainer keeps its MLM head bias under \`lm_head.bias\` only, and the fp32 loader of \`scripts/merge_splade_weights.py\` then reads it as zero. For the 82 M arms and R-a the bias is about zero; the models of the R-c' lineage carry -1.27 there (the four shifts of queue b1c). So \`soup-b1-ra-rcd\` (p1, 3 344 dims/doc) and \`soup-b1-rcd-fact20\` (p2, 11 874 dims/doc) are averages with that bias lost, not evidence against averaging; the reading in p1 that R-a must stay out of the soups rests on such a merge."
        echo "\`$S2\` = R-c' + fact20 merged from copies with both bias keys (\`-bias000\`): $([ -n "$SP" ] && echo "\`$SP\` $(lbs "$SP") / trec-covid $(tc "$SP"), mean of the 19 tasks $(m19 "$SP") at $(dims "$SP") dims/doc, $(pq "$FORMS/$SP.json" "$SP") postings/query (R-c' $(m19 "$RC"), fact20 arm $(m19 "$(measured_form "$FA")"), frozen model $(m19 "$FULL"))" || echo "not measured (logs/soup_$S2.log)"). $N23"
    } >> "$B2/notes_p2.md"
fi
EXTRA_LABELS="${BL:-}" write_summary "Add the b2 results: the soup R-c' + fact20 merged with its bias kept"
log "=== e s3 done ($(date '+%d.%m %H:%M')); soup ${SP:-none} $(m19 "${SP:-none}"); best 124 M $BL $(m19 "${BL:-none}") ==="
