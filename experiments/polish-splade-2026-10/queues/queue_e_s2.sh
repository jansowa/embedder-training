#!/usr/bin/env bash
# Queue e, stage s2 = E2 (about 5 h): the ALBERT (model B) with the teacher of E1.
# From the raw released ALBERT (arm-hl-dqf-full-s3n5kj-r2d3) one more round on the dose3 rows scored by the ensemble
# R-c' + PolDense-400M; the recipe of round r2d3 unchanged (2 epochs, lr 2e-5). Measured: the new round and its soup
# with r2d3 (its own start), each in the first form within both limits (grid 0.05), tiny10 and trec-covid; the better
# of the two on those 11 tasks gets heldout8 and tiny10 as the query encoder over the index of model A.
# Gate for B: >= +0.2 over the released ALBERT on the mean of the 19 tasks within the limits, and no loss as the query
# encoder over the index of A. Results live in the small series (experiments/polish-splade-small-2026-11), the notes and
# the summary with the rest of queue e. Run by queue_e.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=es2 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-21600}
source scripts/queue_lib.sh
MQ=experiments/polish-splade-2026-10/queues
source "$MQ/arm_common.sh"
source "$MQ/release_common.sh"
source "$MQ/b2_common.sh"
source "$MQ/e_common.sh"
guard_alive
CANDS="$EDIR/candidates_b.tsv"
AR=${ALB%-bias[0-9][0-9][0-9]}; NB=$AR-r3rcd; SB=soup-hl-dqf-r2d3-r3rcd
CFG0="$EXP_ROOT/configs/3060/stage3_hl_dqf_full_r2_dose3.yaml"; CFG="$EXP_ROOT/configs/3060/stage3_hl_dqf_full_r3_dose3_ensrcd.yaml"
for f in "$NEWROWS" "$CFG0" "models/$AR/model.safetensors" "models/$ALB/model.safetensors" "models/$FULL/model.safetensors"; do [ -s "$f" ] || { log "  $f missing: s2 stops"; exit 1; }; done
A19=$(m19 "$ALB"); AASYM=$(lbv "asym-$FULL-q-$ALB" tiny10)
log "=== e s2 start: E2, one more round of the ALBERT with R-c' in the teacher; released $ALB $(lbs "$ALB") / $(tc "$ALB") (19 tasks $A19), over the index of A tiny10 ${AASYM:-?}; $(left_min) min before the deadline ==="
cand ref "released ALBERT B (50 M)" "$ALB" "$AR" "round r2d3, teacher ens400"
if [ ! -s "$CFG" ]; then
    "$PY" - "$CFG0" "$CFG" "$NEWROWS" "$NB" "$AR" <<'PYEOF' || { log "  the config of the round FAILED: s2 stops"; exit 1; }
import re, sys
base, out, rows, name, start = sys.argv[1:6]
s = open(base, encoding="utf-8").read()
def sub(pattern, repl):
    global s
    s, n = re.subn(pattern, repl, s, flags=re.M)
    assert n == 1, pattern
sub(r"^runs_dir: runs/arm-hl-dqf-full-s3n5kj-r2d3$", "runs_dir: runs/%s" % name)
sub(r"^train_data: datasets/arms/n5-plus-kj-mined-dose3\.jsonl$", "train_data: %s" % rows)
sub(r"^  - models/arm-hl-dqf-full-s3n5kj$", "  - models/%s" % start)
s = s.replace("backend: sentence-transformers\n", "backend: sentence-transformers\n# Queue e (E2), derived from stage3_hl_dqf_full_r2_dose3.yaml: one more round of the released ALBERT (from its raw weights, models/%s)\n"
              "# on the rows of the dose arm scored by the ensemble of the raw 124 M model R-c' and PolDense-400M instead of soup-v9b and PolDense-400M; the recipe is unchanged.\n" % start, 1)
open(out, "w", encoding="utf-8").write(s)
PYEOF
fi
train "${NB#arm-}" "$CFG"
[ -d "models/$NB" ] || { log "  the round was not trained (logs/arm_${NB#arm-}.log): s2 stops"; echo "## s2 (E2): the ALBERT"$'\n\n'"The round with the new teacher was not trained (logs/arm_${NB#arm-}.log)." > "$EDIR/notes_s2.md"; esummary "Add the e results: E2 failed in training"; exit 1; }
pause_point
cand5 e2 "new round: dose3 rows, teacher R-c' + PolDense" "$NB" "one more round of the released ALBERT with R-c' in the teacher"
pause_point
mksoup "$SB" uniform "$AR" "$NB" && { to_dd "$SB"; cand5 e2 "soup r2d3 + new round" "$SB" "the released round and the new round, uniform"; }
pause_point
BEST=$(best_cand --stages e2 --metric m11 --any-cost --cost "$RESULTS_ROOT/index_cost_a1.json" | cut -f1)
BASYM=""
if [ -n "$BEST" ]; then
    h8 "$BEST"
    PIRB_QUERY_MODEL="models/$BEST" regime quantized "asym-$FULL-q-$BEST" "models/$FULL" tiny10 16; BASYM=$(lbv "asym-$FULL-q-$BEST" tiny10)
fi
B19=$(m19 "${BEST:-none}")
P19=0; [ -n "$B19" ] && [ "$(gte "$B19" "$A19" "$MARGIN")" = 1 ] && P19=1
PAS=0; [ -n "$BASYM" ] && [ -n "$AASYM" ] && [ "$(gte "$BASYM" "$AASYM" -0.05)" = 1 ] && PAS=1
PASS=0; [ "$P19" = 1 ] && [ "$PAS" = 1 ] && PASS=1
[ "$PASS" = 1 ] && echo "$BEST" > "$EDIR/b_winner.txt"
log "  E2: best on the 11 tasks ${BEST:-none} $(lbs "${BEST:-none}") / trec-covid $(tc "${BEST:-none}"), 19 tasks ${B19:-not measured} against $A19; over the index of A tiny10 ${BASYM:-not measured} against ${AASYM:-?} -> the gate for B $([ "$PASS" = 1 ] && echo PASSES || echo "does not pass")"
{
    echo "## s2 (E2): the ALBERT with R-c' in the teacher"; echo
    echo "\`$NB\`: from the raw released ALBERT (\`$AR\`) one more round on the rows of the dose arm with the teacher of E1 (\`$NEWROWS\`), the recipe of round r2d3 unchanged (2 epochs, lr 2e-5); training time $(tsec "$NB") s. \`$SB\` = the uniform average of the raw weights of r2d3 and the new round$([ -d "models/$SB" ] || echo " (NOT made: logs/soup_$SB.log)")."; echo
    "$PY" scripts/b2_candidates.py table $(cargs) --cost "$RESULTS_ROOT/index_cost_a1.json" --ref "$ALB" --margin "$MARGIN" 2>> "$LOGS/queue_summary.log" | sed 's/vs the frozen model/vs the released ALBERT/g'
    echo; echo "The released ALBERT is in the form it was released in (bias020); the candidates are in their first form within the limits on the same grid of 0.05. heldout8 went to the better candidate on the 11 tasks only."; echo
    echo "As the query encoder over the index of model A (\`$FULL\`), tiny10: \`${BEST:-no candidate}\` ${BASYM:-not measured}, the released ALBERT ${AASYM:-not measured}."; echo
    echo "**Gate for B** (>= +$MARGIN over the released ALBERT on the 19 tasks within the limits, and not more than 0.05 below it over the index of A): 19 tasks ${B19:-not measured} against $A19 ($([ "$P19" = 1 ] && echo yes || echo no)), over the index of A ($([ "$PAS" = 1 ] && echo yes || echo no)) -> **$([ "$PASS" = 1 ] && echo "passes" || echo "does not pass")**."
} > "$EDIR/notes_s2.md"
EXTRA_LABELS="${BEST:+asym-$FULL-q-$BEST}" esummary "Add the e results: one more round of the ALBERT with the 124 M model in the teacher (E2)"
log "=== e s2 done ($(date '+%d.%m %H:%M')); best ${BEST:-none} 19 tasks ${B19:-?} (released $A19); gate for B: $PASS ==="
