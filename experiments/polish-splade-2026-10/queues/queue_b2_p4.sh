#!/usr/bin/env bash
# Queue b2, stage p4 (optional; about 4.2 h of training + 1.4 h of measurement): round 2.
# From the raw weights of the best 124 M candidate on the 19 tasks (a soup, or an arm; R-a when nothing is better) one
# epoch on the dose3 rows with lr 1e-5, the rest of the recipe of R-a. In the 82 M family a round added from a soup
# gained (soup-v12 -> r2v12). It is the last training step and must END before Thursday 19.11 12:00: it is started only
# when the estimate (1 epoch = half of R-a's training time, x 1.15, + 1.5 h for the form and the 19 tasks) fits before
# that hour; otherwise it is skipped. Run by queue_b2.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b2p4 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-43200}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
DEADLINE=${P4_DEADLINE:-2026-11-19 12:00}
base_cands
BEST=$(best_cand --metric m19 --exclude "$FULL")
START=$(echo "$BEST" | cut -f2); SLABEL=$(echo "$BEST" | cut -f1); SNAME=$(echo "$BEST" | cut -f3)
[ -n "$START" ] && [ -f "models/$START/model.safetensors" ] || { START=$RA; SLABEL=$RA15; SNAME="R-a, bias015"; }
RAT=$(tsec "$RA"); RAT=${RAT:-29012}
NEED=$(( RAT / 2 * 115 / 100 + 5400 )); LEFT=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) ))
log "=== b2 p4 start: round 2 from $START ($SNAME, 19 tasks $(m19 "$SLABEL")); needs about $((NEED / 60)) min, $((LEFT / 60)) min are left before $DEADLINE ==="
note() { { echo "## p4: round 2 (one epoch on dose3, lr 1e-5)"; echo; echo "$1"; } > "$B2/notes_p4.md"; }
if [ -d "models/$R2" ]; then log "  models/$R2 is present: not trained again"
elif [ "$LEFT" -lt "$NEED" ]; then
    log "  not enough time before $DEADLINE: round 2 is skipped"
    note "Skipped: on $(date '+%d.%m %H:%M') the step needed about $((NEED / 60)) min and $((LEFT / 60)) min were left before $DEADLINE."
    write_summary "Add the b2 results: round 2 skipped for lack of time"
    exit 0
else
    CFG="$C/b2_r2_d3.yaml"
    "$PY" - "$C/b1_ra_dose3.yaml" "$CFG" "$START" "$SNAME" <<'PYEOF' || { log "  the config of round 2 FAILED: p4 stops"; exit 1; }
import re, sys
base, out, start, name = sys.argv[1:5]
s = open(base, encoding="utf-8").read()
def sub(pattern, repl):
    global s
    s, n = re.subn(pattern, repl, s, flags=re.M)
    assert n == 1, pattern
head = s[s.index("# Queue b1, version R-a"):s.index("runs_dir:")]
s = s.replace(head, "# Queue b2, round 2: from the raw weights of the best 124 M candidate of the queue (models/%s: %s) one epoch on the rows of\n"
                    "# the dose arm with lr 1e-5; everything else as in R-a (b1_ra_dose3.yaml). Written by queue_b2_p4.sh; the queue derives a _gc copy.\n" % (start, name))
sub(r"^runs_dir: runs/arm-b1-ra$", "runs_dir: runs/arm-b1-r2-d3")
sub(r"^  - models/kd-rbase-v2-dqf-full$", "  - models/%s" % start)
sub(r"^  - learning_rate: 2e-5$", "  - learning_rate: 1e-5")
sub(r"^    num_train_epochs: 2$", "    num_train_epochs: 1")
open(out, "w", encoding="utf-8").write(s)
PYEOF
    echo "$START" > "$B2/p4_start.txt"
    train_rank "${R2#arm-}" "$CFG"
    pause_point
fi
arm_cand p4 "$R2" "round 2: ($SNAME) + one epoch on dose3, lr 1e-5"
log "  round 2: ${ARM_PICK:-no form} $(lbs "${ARM_PICK:-none}") / trec-covid $(tc "${ARM_PICK:-none}"), 19 tasks $(m19 "${ARM_PICK:-none}") (its start $SLABEL $(m19 "$SLABEL"), frozen $(m19 "$FULL"))"
note "\`$R2\` from \`$(cat "$B2/p4_start.txt" 2>/dev/null || echo "$START")\` ($SNAME; 19 tasks $(m19 "$SLABEL")): one epoch on the dose3 rows with lr 1e-5, training time $(tsec "$R2") s. Measured form \`${ARM_PICK:-none}\`: $(lbs "${ARM_PICK:-none}") / trec-covid $(tc "${ARM_PICK:-none}"), mean of the 19 tasks $(m19 "${ARM_PICK:-none}") (frozen model $(m19 "$FULL")). The plan estimated 8 h for this step; one epoch of dose3 is half of R-a's training."
write_summary "Add the b2 results: round 2 of the 124 M model from its best soup"
log "=== b2 p4 done ($(date '+%d.%m %H:%M')); round 2 ${ARM_PICK:-none} $(m19 "${ARM_PICK:-none}") ==="
