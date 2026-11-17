#!/usr/bin/env bash
# Queue b2, stage p3 (about 10.8 h of training + 1.4 h of measurement + 1.5 h per soup): the arm faq20.
# The recipe of R-a unchanged on n5-plus-kj-mined-dose3-faq20.jsonl (80 385 rows), from the start of the lineage chosen
# before p2 (pick_lineage in b2_common.sh: kd = from the KD model, anchor R-a; rc = from the start of R-c', anchor R-c').
# Measured in its first form within both limits, 19 tasks. Then soups (uniform averages of the raw weights), each with
# the 19 tasks:
#   soup-b1-<ra|rcd>-faq20-fact20 = the anchor + faq20 + fact20: the 124 M counterpart of the frozen model (the anchor +
#                                   faq20 when there is no fact20 arm)
#   soup-b1-ra-rcd-fact20-faq20   = the best soup of p2 with faq20 added, when that soup was the one with R-a and R-c'
#   and, when the interpolation with the KD model helped in p1, the best soup of this stage interpolated the same way.
# Run by queue_b2.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b2p3 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-79200}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
ROWS=datasets/arms/n5-plus-kj-mined-dose3-faq20.jsonl
base_cands
pick_lineage
CFG=$(arm_cfg faq20) || { log "  the config of the faq20 arm FAILED: p3 stops"; exit 1; }
for f in "$ROWS" "$CFG" "models/$START/model.safetensors" "models/$ANCHOR/model.safetensors"; do [ -s "$f" ] || { log "  $f missing: p3 stops"; exit 1; }; done
log "=== b2 p3 start: lineage $LINEAGE, arm $FQ from $START on $ROWS ($(grep -c . "$ROWS") rows), mode $RMODE; $(free_gb) GB free on /, $(df -BG --output=avail "$DD" | tail -1 | tr -dc '0-9') GB on the data drive ==="
train_rank "${FQ#arm-}" "$CFG"
pause_point
arm_cand p3 "$FQ" "arm faq20 ($START + dose3 + FAQ rows)"
log "  faq20 arm: ${ARM_PICK:-no form} $(lbs "${ARM_PICK:-none}") / trec-covid $(tc "${ARM_PICK:-none}"), 19 tasks $(m19 "${ARM_PICK:-none}") (R-a $(m19 "$RA15"), R-c' $(m19 "$(measured_form "$RCD")"), frozen $(m19 "$FULL"))"
pause_point
SOUPS=""
if [ -d "models/$FQ" ]; then
    if [ -d "models/$FA" ]; then soup_cand p3 "soup-b1-$AS-faq20-fact20" uniform "soup $AN + faq20 + fact20" measure19 "$ANCHOR" "$FQ" "$FA" && SOUPS="$SOUPS $SOUP_PICK"
    else soup_cand p3 "soup-b1-$AS-faq20" uniform "soup $AN + faq20" measure19 "$ANCHOR" "$FQ" && SOUPS="$SOUPS $SOUP_PICK"; fi
    pause_point
    B2S=$(best_cand --stages p2 --metric m19 --name-prefix soup-b1- | cut -f2)
    if [ "$B2S" = soup-b1-ra-rcd-fact20 ] && [ -d "models/$RCD" ] && [ -d "models/$RA" ] && [ -d "models/$FA" ]; then
        soup_cand p3 soup-b1-ra-rcd-fact20-faq20 uniform "soup R-a + R-c' + fact20 + faq20" measure19 "$RA" "$RCD" "$FA" "$FQ" && SOUPS="$SOUPS $SOUP_PICK"
        pause_point
    else log "  the best soup of p2 is ${B2S:-none}: with faq20 added it is the soup above (or there is none), nothing further"; fi
    WA=$(cat "$B2/wise_alpha.txt" 2>/dev/null)
    if [ -n "$WA" ]; then
        BS=$(best_cand --stages p3 --metric m19 --name-prefix soup-b1-)
        if [ -n "$BS" ]; then
            BRAW=$(echo "$BS" | cut -f2)
            soup_cand p3 "wise-${BRAW#soup-}-a$WA" "0.$WA,0.$((100 - WA))" "0.$WA x ($(echo "$BS" | cut -f3)) + 0.$((100 - WA)) x KD model" measure19 "$BRAW" "$KD" && SOUPS="$SOUPS $SOUP_PICK"
        fi
    fi
else log "  no faq20 arm: no soups"; fi
BEST=$(best_cand --metric m19 --exclude "$FULL")
{
    echo "## p3: the arm faq20 and its soups (lineage $LINEAGE)"; echo
    echo "\`$FQ\` from \`$START\`: the recipe of R-a on \`$ROWS\` ($(grep -c . "$ROWS") rows), training time $(tsec "$FQ") s, mode $RMODE; $(dims "$FQ") dims/doc without a shift. Measured form \`${ARM_PICK:-none}\`: $(lbs "${ARM_PICK:-none}") / trec-covid $(tc "${ARM_PICK:-none}"), mean of the 19 tasks $(m19 "${ARM_PICK:-none}") (R-a bias015 $(m19 "$RA15"), R-c' $(m19 "$(measured_form "$RCD")"), frozen model $(m19 "$FULL"))."; echo
    echo "Soups of this stage (uniform averages of the raw weights):${SOUPS:- none}."
    for l in $SOUPS; do echo "- \`$l\`: $(lbs "$l") / trec-covid $(tc "$l"), 19 tasks $(m19 "$l"), $(dims "$l") dims/doc, $(pq "$FORMS/$l.json" "$l") postings/query"; done
    echo; echo "Best 124 M candidate on the 19 tasks so far: \`$(echo "$BEST" | cut -f1)\` ($(echo "$BEST" | cut -f3)) $(echo "$BEST" | cut -f4); the frozen model has $(m19 "$FULL"), the rule asks for $MARGIN more."
} > "$B2/notes_p3.md"
write_summary "Add the b2 results: the arm faq20 of the 124 M model and its soups"
log "=== b2 p3 done ($(date '+%d.%m %H:%M')); faq20 ${ARM_PICK:-none} $(m19 "${ARM_PICK:-none}"); best so far $(echo "$BEST" | cut -f1,4 | tr '\t' ' ') (frozen $(m19 "$FULL")) ==="
