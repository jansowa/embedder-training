#!/usr/bin/env bash
# Queue b2 (3060 Ti, 17-19.11, after queue b1c): the package swaps the user approved, then the search for a 124 M
# replacement of the frozen full model. The release of 26.11 is frozen (full model soup-v20-dqf-bias015, the ALBERT, the
# static table); a stronger 124 M model found by Thursday 19.11 evening would replace the full model. R-a
# (arm-b1-ra-bias015) has 65.34 / 43.76 but trec-covid 67.39 against 72.48, so on the mean of the 19 tasks it is 56.36
# against 56.435: the repair of trec-covid is the condition of a replacement.
# This file is only the driver: every stage is its own script (queue_b2_<stage>.sh, sharing b2_common.sh), so a later
# stage can still be corrected while an earlier one runs. Stages:
#   p0   the new package datasets/transfer/release-20261117 next to the old one (repaired ALBERT int8, the cheaper static
#        table, int8 variants with the MLM head in int8, clean SHA256SUMS) and a tar with the difference for the 5070; CPU only
#   p1   cheap diagnosis and repairs of trec-covid without training: trec-covid and fidelity of the KD model, a denser
#        bias form of R-a, weight interpolation of R-a with the KD model, the soup R-a + R-c' after the dose3 round
#   p1b  the stage-1 models of R-c' with their bias restored (queue b1c shifted it in place, see b2_common.sh): the true
#        points of the dose curve at 50 000 and 200 000 rows
#   p2   the arm fact20 from the KD model and its soups
#   p3   the arm faq20 from the KD model and its soups
#   p4   optional: one more round on dose3 from the best soup, only when it ends before Thursday 19.11 12:00
#   p5   the rule of the replacement, the 23 tasks and the GPU encoding speed of the best candidate, the report
# A stage is skipped when logs/b2.skip.<stage> exists and not repeated when logs/b2.done.<stage> exists. Between stages
# (and between the steps of a stage) the queue waits while logs/b1.pause exists. No CPU timing. No package for a 124 M
# model is made here. Nothing is deleted, the release backup is not touched.
#   bash logs/relaunch-20261113/launch_queue.sh b2 experiments/polish-splade-2026-10/queues/queue_b2.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b2 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
echo $$ > "$LOGS/queue_b2.pid"
guard_alive
wait_for_queues ${WAIT_FOR:-b1c}
Q="$EXP_ROOT/queues"
DD="$(dirname "$(readlink -f runs)")/FlagEmbedding-models-b1"
pause_point() { local said=0; while [ -e "$LOGS/b1.pause" ]; do [ "$said" = 0 ] && { log "  paused by logs/b1.pause (remove it to continue)"; said=1; }; sleep 60; done; }
log "=== b2 start ($(date '+%d.%m %H:%M')): package swaps, then the search for a 124 M replacement of the frozen full model; $(free_gb) GB free on /, $(df -BG --output=avail "$DD" | tail -1 | tr -dc '0-9') GB on the data drive ==="

# The empty directory that kept queue b1c from moving arm-b1-rc-d3 to the data drive before it wrote the bias forms
# (b2_common.sh): its job is done.
[ -d "$DD/arm-b1-rc-d3" ] && rmdir "$DD/arm-b1-rc-d3" 2>/dev/null && log "  the empty marker $DD/arm-b1-rc-d3 is removed"
# pack_sums listed its own temporary file (the stray line ./SHA256SUMS.tmp in the package of 13.11). No queue that
# sources release_common.sh is alive at this point.
if grep -q "find \. -type f ! -name SHA256SUMS -print0" "$Q/release_common.sh"; then
    sed -i "s/find \. -type f ! -name SHA256SUMS -print0/find . -type f ! -name 'SHA256SUMS*' -print0/" "$Q/release_common.sh" && log "  release_common.sh: pack_sums no longer lists SHA256SUMS.tmp"
fi

for st in ${STAGES:-p0 p1 p1b p2 p3 p4 p5}; do
    pause_point
    [ -e "$LOGS/b2.skip.$st" ] && { log "=== b2 stage $st: skipped (logs/b2.skip.$st) ==="; continue; }
    [ -e "$LOGS/b2.done.$st" ] && { log "=== b2 stage $st: done earlier (logs/b2.done.$st) ==="; continue; }
    [ -f "$Q/queue_b2_$st.sh" ] || { log "=== b2 stage $st: no script ==="; continue; }
    log "=== b2 stage $st starts ($(date '+%d.%m %H:%M')) ==="; s=$SECONDS
    bash "$Q/queue_b2_$st.sh" >> "$LOGS/queue_b2_$st.out" 2>&1; rc=$?
    log "=== b2 stage $st: exit $rc in $(( (SECONDS - s) / 60 )) min ($(date '+%d.%m %H:%M')) ==="
    [ $rc = 0 ] && touch "$LOGS/b2.done.$st"
done
log "=== b2 done ($(date '+%d.%m %H:%M')) ==="
