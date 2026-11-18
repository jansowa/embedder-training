#!/usr/bin/env bash
# Queue e (3060 Ti, 17-18.11, after queue b2): a new ensemble teacher for the released models, with the 124 M model
# R-c' in the place of soup-v9b. The presentation freezes its numbers on Thursday 19.11, so every candidate for a swap
# of model A (the full 82 M model) or B (the ALBERT) has to be measured and reported by Wednesday 18.11 about 19:00.
# The 124 M model itself is a possible fourth model, not a replacement of A (the static table and the asymmetric
# ALBERT work over the index of A). This file is only the driver; every stage is its own script:
#   s0  scores: the dose3 rows scored by the raw R-c' (a copy with both bias keys, see below) and by soup-v9b
#   s1  E1: the ensemble with R-c' instead of soup-v9b, the dose3 arm of the frozen model with that teacher, its soups
#       with the arms of the frozen model, the gate for A (+0.2 on the 19 tasks) and, when it passes, the static table
#       and the ALBERT over the new index
#   s2  E2: one more round of the ALBERT with the same teacher, its soup with the released round, the gate for B
#   s3  what is left of the 124 M line for now: the soup R-c' + fact20 merged with its bias kept, the 23 tasks of the
#       best 124 M candidate when it beats R-c' on the 19 tasks, the 23 tasks of a new A when there was no time before
# Bias keys: a model saved by the trainer has its MLM head bias under lm_head.bias only. Loaders that read fp32
# (scripts/merge_splade_weights.py, the trainer, SparseEncoder(path)) then take the bias as zero; the fp16 loaders of
# the sparsity and fidelity tools keep it. For the 82 M models and R-a the bias is about zero, so nothing changed; the
# models of the R-c' lineage carry -1.27 there (the four shifts of queue b1c), so their raw weights must be read from
# a copy with both keys (biased2 with threshold 0: <model>-bias000). The soups R-a + R-c' (p1) and R-c' + fact20 (p2)
# of queue b2 were merged with that bias lost; that, not the averaging, made them unusable.
# A stage is skipped when logs/e.skip.<stage> exists and not repeated when logs/e.done.<stage> exists; between stages
# the queue waits while logs/b1.pause exists. No CPU timing. No package is made here. Nothing is deleted.
#   bash logs/relaunch-20261113/launch_queue.sh e experiments/polish-splade-2026-10/queues/queue_e.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=e MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
echo $$ > "$LOGS/queue_e.pid"
guard_alive
wait_for_queues ${WAIT_FOR:-b1c b2}
Q="$EXP_ROOT/queues"
pause_point() { local said=0; while [ -e "$LOGS/b1.pause" ]; do [ "$said" = 0 ] && { log "  paused by logs/b1.pause (remove it to continue)"; said=1; }; sleep 60; done; }
log "=== e start ($(date '+%d.%m %H:%M')): a teacher with R-c' for the released models; $(free_gb) GB free on / ==="
for st in ${STAGES:-s0 s1 s2 s3}; do
    pause_point
    [ -e "$LOGS/e.skip.$st" ] && { log "=== e stage $st: skipped (logs/e.skip.$st) ==="; continue; }
    [ -e "$LOGS/e.done.$st" ] && { log "=== e stage $st: done earlier (logs/e.done.$st) ==="; continue; }
    until [ -f "$Q/queue_e_$st.sh" ] && [ ! -e "$LOGS/e.hold.$st" ]; do sleep 30; done      # a stage script still being written is held by logs/e.hold.<stage>
    log "=== e stage $st starts ($(date '+%d.%m %H:%M')) ==="; s=$SECONDS
    bash "$Q/queue_e_$st.sh" >> "$LOGS/queue_e_$st.out" 2>&1; rc=$?
    log "=== e stage $st: exit $rc in $(( (SECONDS - s) / 60 )) min ($(date '+%d.%m %H:%M')) ==="
    [ $rc = 0 ] && touch "$LOGS/e.done.$st"
done
log "=== e done ($(date '+%d.%m %H:%M')) ==="
