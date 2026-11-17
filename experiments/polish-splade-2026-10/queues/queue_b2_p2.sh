#!/usr/bin/env bash
# Queue b2, stage p2 (about 10.5 h of training + 1.4 h of measurement + 1.5 h per soup): the arm fact20.
# The recipe of R-a unchanged (gc, 2 x 16, 512 tokens, 2 epochs, lr 2e-5) on n5-plus-kj-mined-dose3-fact20.jsonl
# (77 787 rows). In the 82 M family the fact20 arm had the best trec-covid (73.75) and the soup of the arms from one
# start became the frozen model. The start follows the user's decision of 17.11 ("start from the better checkpoint",
# pick_lineage in b2_common.sh): the lineage of the better of R-a and R-c' on the 19 tasks.
#   lineage kd: arm-b1-ra-fact20 from the KD model; the dose3 arm of the lineage (the anchor of the soups) is R-a
#   lineage rc: arm-b1-rc-fact20 from the start of R-c' (the end of the MS MARCO-pl stage); the anchor is R-c'
# Measured in its first form within both limits, 19 tasks. Then soups (uniform averages of the raw weights; every model
# descends from the KD model), each with the 19 tasks:
#   soup-b1-<ra|rcd>-fact20 = the anchor + fact20
#   soup-b1-ra-rcd-fact20   = R-a + R-c' + fact20: for kd when the gate of p1 step 4 passed (R-c' within 0.2 of R-a), for
#                             rc when the soup R-a + R-c' of p1 was not below R-c' alone on the 11 tasks
#   and, when the interpolation with the KD model helped in p1, the best soup of this stage interpolated the same way.
# Run by queue_b2.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b2p2 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-79200}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
ROWS=datasets/arms/n5-plus-kj-mined-dose3-fact20.jsonl
base_cands
pick_lineage
CFG=$(arm_cfg fact20) || { log "  the config of the fact20 arm FAILED: p2 stops"; exit 1; }
for f in "$ROWS" "$CFG" "models/$START/model.safetensors" "models/$ANCHOR/model.safetensors"; do [ -s "$f" ] || { log "  $f missing: p2 stops"; exit 1; }; done
log "=== b2 p2 start: lineage $LINEAGE, arm $FA from $START on $ROWS ($(grep -c . "$ROWS") rows), mode $RMODE; $(free_gb) GB free on /, $(df -BG --output=avail "$DD" | tail -1 | tr -dc '0-9') GB on the data drive ==="
train_rank "${FA#arm-}" "$CFG"
pause_point
arm_cand p2 "$FA" "arm fact20 ($START + dose3 + fact20 rows)"
log "  fact20 arm: ${ARM_PICK:-no form} $(lbs "${ARM_PICK:-none}") / trec-covid $(tc "${ARM_PICK:-none}"), 19 tasks $(m19 "${ARM_PICK:-none}") (R-a $(m19 "$RA15"), R-c' $(m19 "$(measured_form "$RCD")"), frozen $(m19 "$FULL"))"
pause_point
SOUPS=""
if [ -d "models/$FA" ]; then
    soup_cand p2 "soup-b1-$AS-fact20" uniform "soup $AN + fact20" measure19 "$ANCHOR" "$FA" && SOUPS="$SOUPS $SOUP_PICK"
    pause_point
    if [ "$LINEAGE" = rc ]; then XG=$(cat "$B2/cross_gate.txt" 2>/dev/null); else XG=$(cat "$B2/p14_gate.txt" 2>/dev/null); fi
    if [ "$XG" = 1 ] && [ -d "models/$RCD" ] && [ -d "models/$RA" ]; then
        soup_cand p2 soup-b1-ra-rcd-fact20 uniform "soup R-a + R-c' + fact20" measure19 "$RA" "$RCD" "$FA" && SOUPS="$SOUPS $SOUP_PICK"
        pause_point
    else log "  the gate of the soup with both R-a and R-c' did not pass (lineage $LINEAGE): only the soup with $AN"; fi
    WA=$(cat "$B2/wise_alpha.txt" 2>/dev/null)
    if [ -n "$WA" ]; then
        BS=$(best_cand --stages p2 --metric m19 --name-prefix soup-b1-)
        if [ -n "$BS" ]; then
            BRAW=$(echo "$BS" | cut -f2)
            soup_cand p2 "wise-${BRAW#soup-}-a$WA" "0.$WA,0.$((100 - WA))" "0.$WA x ($(echo "$BS" | cut -f3)) + 0.$((100 - WA)) x KD model" measure19 "$BRAW" "$KD" && SOUPS="$SOUPS $SOUP_PICK"
        fi
    else log "  the interpolation with the KD model did not help in p1: none here"; fi
else log "  no fact20 arm: no soups"; fi
BEST=$(best_cand --metric m19 --exclude "$FULL")
{
    echo "## p2: the arm fact20 and its soups (lineage $LINEAGE)"; echo
    echo "\`$FA\` from \`$START\`: the recipe of R-a on \`$ROWS\` ($(grep -c . "$ROWS") rows), training time $(tsec "$FA") s, mode $RMODE; $(dims "$FA") dims/doc without a shift. Measured form \`${ARM_PICK:-none}\`: $(lbs "${ARM_PICK:-none}") / trec-covid $(tc "${ARM_PICK:-none}"), mean of the 19 tasks $(m19 "${ARM_PICK:-none}") (R-a bias015 $(m19 "$RA15"), R-c' $(m19 "$(measured_form "$RCD")"), frozen model $(m19 "$FULL"))."; echo
    echo "Soups of this stage (uniform averages of the raw weights):${SOUPS:- none}."
    for l in $SOUPS; do echo "- \`$l\`: $(lbs "$l") / trec-covid $(tc "$l"), 19 tasks $(m19 "$l"), $(dims "$l") dims/doc, $(pq "$FORMS/$l.json" "$l") postings/query"; done
    echo; echo "Best 124 M candidate on the 19 tasks so far: \`$(echo "$BEST" | cut -f1)\` ($(echo "$BEST" | cut -f3)) $(echo "$BEST" | cut -f4); the frozen model has $(m19 "$FULL"), the rule asks for $MARGIN more."
} > "$B2/notes_p2.md"
write_summary "Add the b2 results: the arm fact20 of the 124 M model and its soups"
log "=== b2 p2 done ($(date '+%d.%m %H:%M')); fact20 ${ARM_PICK:-none} $(m19 "${ARM_PICK:-none}"); best so far $(echo "$BEST" | cut -f1,4 | tr '\t' ' ') (frozen $(m19 "$FULL")) ==="
