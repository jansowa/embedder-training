#!/usr/bin/env bash
# Queue e, stage s0 (about 1.5 h): the scores the new teacher is built from.
#   1  models/arm-b1-rc-d3-bias000 = the raw R-c' with both bias keys (no shift), so that every loader reads its bias
#   2  smoke: 300 rows scored by it and by soup-v9b; the stage stops when the two disagree wildly (mean within-row
#      Pearson below 0.5), which would mean a broken teacher, before an hour of scoring and four of training
#   3  the 60 385 rows of the dose arm scored by the raw R-c' and by soup-v9b with the tool and the settings of the
#      ens400 teacher (kitchen bridge/splade_teacher_scores.py, 512 tokens, batch 16, fp16). The soup-v9b scores are
#      needed because the rows keep only the ensemble: the PolDense-400M part of a pair is 2 x ensemble - soup-v9b.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=es0 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
A=datasets/arms; ROWS=$A/n5-plus-kj-mined-dose3.jsonl; T=$RCD-bias000
TR=$A/n5-plus-kj-mined-dose3-trcd.jsonl; TV=$A/n5-plus-kj-mined-dose3-tv9b.jsonl
for f in "$ROWS" "$KITCHEN/bridge/splade_teacher_scores.py" "models/$RCD/model.safetensors" models/soup-v9b/model.safetensors; do [ -s "$f" ] || { log "  $f missing: s0 stops"; exit 1; }; done
log "=== e s0 start: scores of $(grep -c . "$ROWS") rows by the raw R-c' and by soup-v9b; $(free_gb) GB free on / ==="
biased2 "$RCD" "$T" 0.00 || { log "  no copy of R-c' with both bias keys: s0 stops"; exit 1; }
to_dd "$T"
score() {  # score TAG TEACHER OUT TIMEOUT [args]: splade_teacher_scores.py with the settings of the ens400 teacher
    local tag="$1" teacher="$2" out="$3" to="$4" s rc; shift 4
    [ -s "$out" ] && return 0
    need_space || return 1; wait_gpu_free; wait_ram; s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$PY" "$KITCHEN/bridge/splade_teacher_scores.py" --teacher "$teacher" --rows "$ROWS" --output "$out.tmp" --max-seq-length 512 --batch-size 16 "$@" > "$LOGS/e_score_$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out.tmp" ] && mv "$out.tmp" "$out" || rc=1
    log "  scores $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/e_score_$tag.log" | cut -c1-240)"
    record "chain:e_score_$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
SM="runs/e-smoke"; mkdir -p "$SM"
score smoke_rcd "models/$T" "$SM/trcd.jsonl" 900 --smoke 300 && score smoke_v9b models/soup-v9b "$SM/tv9b.jsonl" 900 --smoke 300 || { log "  the smoke scoring FAILED: s0 stops"; exit 1; }
SMK=$("$PY" scripts/swap_splade_teacher.py --ens "$ROWS" --old-splade "$SM/tv9b.jsonl" --new-splade "$SM/trcd.jsonl" --stats-only 2>> "$LOGS/queue_summary.log" | tail -n 1)
log "  smoke, 300 rows: $SMK"
OK=$(echo "$SMK" | sed -n 's/^.*pearson_new_vs_old_splade=\([-0-9.]*\).*$/\1/p')
[ -n "$OK" ] && [ "$(gte "$OK" 0.5)" = 1 ] || { log "  R-c' and soup-v9b disagree on the smoke rows (within-row Pearson ${OK:-not computed}, the bar is 0.5): the teacher looks broken, s0 stops"; exit 1; }
pause_point
score rcd "models/$T" "$TR" 10800 || exit 1
pause_point
score v9b models/soup-v9b "$TV" 7200 || exit 1
log "=== e s0 done ($(date '+%d.%m %H:%M')): $TR, $TV ==="
