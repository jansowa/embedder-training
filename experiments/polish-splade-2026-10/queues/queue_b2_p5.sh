#!/usr/bin/env bash
# Queue b2, stage p5 (about 2.5 h): the rule of the replacement and the report.
#   1  the rule over every candidate with the 19 tasks: the highest mean of the 19 tasks within <= 410 dims/doc and
#      <= 16 807 postings/query; the frozen model stays unless a candidate leads it by >= 0.2 (scripts/freeze_choice.py,
#      metric tasks); one index-cost pass over all candidates for the table
#   2  the 23 tasks of the winner (it must not be below the frozen model's 58.80); when no candidate wins, the 23 tasks
#      of the best 124 M candidate are measured only if it is not below the frozen model on the 19 tasks
#   3  GPU encoding speed of the best 124 M candidate next to the frozen 82 M model (release.gpu_bench, fp16 eager; GPU
#      only, no CPU timing)
#   4  the report in summary_b2.md with a recommendation; the decision is the user's
# logs/final_full_model.txt is NOT changed and no package for a 124 M model is made. Run by queue_b2.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b2p5 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
CHOICE="$B2/b2_choice.json"; COST="$B2/index_cost_b2.json"
base_cands
log "=== b2 p5 start: the rule of the replacement over $(grep -c . "$CANDS") candidates ==="

# ---------------------------------------------------------------- 1. the rule
SPECS=("frozen full model (82 M)=$FULL=$(dims "$FULL")=dose3 + faq20 + fact20 (82 M)"); LABELS=""
while IFS=$'\t' read -r stage name label raw desc; do
    [ "$label" = "$FULL" ] && continue
    [ -n "$(m19 "$label")" ] || continue
    SPECS+=("$(echo "$name" | tr '=' ':')=$label=$(dims "$label")=$(echo "$desc" | tr '=' ':')"); LABELS="$LABELS $label"
done < "$CANDS"
[ -f "$COST" ] && mv "$COST" "$LOGS/b1-aside/index_cost_b2.$(date +%H%M%S).json"
index_cost "$COST" "$FULL" $LABELS
"$PY" scripts/freeze_choice.py --regimes "$QREG" --cost "$COST" --out "$CHOICE" --max-dims "$MAX_DIMS" --max-postings "$MAX_POSTINGS" --margin "$MARGIN" --metric tasks \
    "${SPECS[@]}" 2>&1 | while read -r l; do log "  $l"; done
WIN=$(jget "$CHOICE" winner); OTHER=$(jget "$CHOICE" best_other); WHY=$(jget "$CHOICE" why)
[ -n "$WIN" ] || { log "  no decision of the rule ($CHOICE)"; WIN="$FULL"; }
BEST124=$OTHER; [ "$WIN" != "$FULL" ] && BEST124=$WIN
[ -n "$BEST124" ] || BEST124=$(best_cand --metric m19 --exclude "$FULL" | cut -f1)
log "  rule: ${WHY:-no decision}; best 124 M candidate: ${BEST124:-none} $(m19 "${BEST124:-none}") (frozen $(m19 "$FULL"))"
pause_point

# ---------------------------------------------------------------- 2. the 23 tasks
F23=$(m23 "$FULL"); B23=""; N23="not measured"
if [ -n "$BEST124" ] && [ -d "models/$BEST124" ] && { [ "$WIN" != "$FULL" ] || [ "$(gte "$(m19 "$BEST124")" "$(m19 "$FULL")")" = 1 ]; }; then
    log "=== b2 p5 step 2: the 23 tasks of $BEST124 ==="
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$BEST124" "$ts"; log "  $BEST124 $ts $(lbv "$BEST124" "$ts")"; done
    B23=$(m23 "$BEST124")
    if [ -n "$B23" ]; then N23="$B23 against $F23 of the frozen model: $([ "$(gte "$B23" "$F23")" = 1 ] && echo "not below it" || echo "BELOW it")"; else N23="incomplete (a task set failed)"; fi
    log "  23 tasks of $BEST124: $N23"
else
    log "=== b2 p5 step 2: no candidate is at or above the frozen model on the 19 tasks: the 23 tasks are not measured ==="
    N23="not measured (no 124 M candidate reaches the frozen model on the 19 tasks)"
fi
pause_point

# ---------------------------------------------------------------- 3. GPU encoding speed
V="$DRIVE/FlagEmbedding-venvs/gpuq/bin/python"; [ -x "$V" ] || V="$PY"
gbench() {  # gbench LABEL: release.gpu_bench fp16 eager -> $B2/gpu_bench_LABEL.json
    local out="$B2/gpu_bench_$1.json" s
    [ -s "$out" ] && return 0
    wait_gpu_free; wait_ram; s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$V" -m release.gpu_bench --model "models/$1" --variants fp16:eager --out "$out" > "$LOGS/b2_gpu_bench_$1.log" 2>&1 \
        || { [ "$V" != "$PY" ] && ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$PY" -m release.gpu_bench --model "models/$1" --variants fp16:eager --out "$out" > "$LOGS/b2_gpu_bench_$1.log" 2>&1; }
    log "  gpu_bench $1: $([ -s "$out" ] && echo ok || echo FAILED) in $((SECONDS - s))s: $(grep -E "^\| *fp16" "$LOGS/b2_gpu_bench_$1.log" | tail -n 1 | cut -c1-300)"
    record "gpu-bench:$1" "$([ -s "$out" ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
}
SPEED=""
if [ -n "$BEST124" ] && [ -d "models/$BEST124" ]; then
    log "=== b2 p5 step 3: GPU encoding speed of $BEST124 next to $FULL (fp16 eager) ==="
    gbench "$FULL"; gbench "$BEST124"
    SPEED=$(for l in "$FULL" "$BEST124"; do echo "\`$l\`:"; echo; grep -E "^\|" "$LOGS/b2_gpu_bench_$l.log" 2>/dev/null | cut -c1-400; echo; done)
fi

# ---------------------------------------------------------------- 4. the report
LEAD=$("$PY" -c "import sys;print('%+.3f' % (float(sys.argv[1]) - float(sys.argv[2])))" "$(m19 "${BEST124:-none}")" "$(m19 "$FULL")" 2>/dev/null)
if [ "$WIN" != "$FULL" ] && [ -n "$B23" ] && [ "$(gte "$B23" "$F23")" = 1 ]; then
    REC="**Recommendation: replace the full model with \`$WIN\`.** It leads the frozen model by $LEAD on the 19 tasks (the rule asks for $MARGIN) within both limits and its 23 tasks are not below the frozen model's ($N23)."
elif [ "$WIN" != "$FULL" ]; then
    REC="**Recommendation: keep the frozen model.** \`$WIN\` passes the rule of the 19 tasks ($LEAD) but not the condition of the 23 tasks: $N23."
else
    REC="**Recommendation: keep the frozen model \`$FULL\`.** No 124 M candidate leads it by $MARGIN on the 19 tasks within the limits; the best, \`${BEST124:-none}\`, is at ${LEAD:-?}."
fi
{
    echo "## p5: the rule of the replacement and the recommendation"; echo
    echo "$REC The decision is the user's; \`logs/final_full_model.txt\` was not changed and no package for a 124 M model was made."; echo
    echo "Rule ($(date '+%d.%m %H:%M')): ${WHY:-no decision}. 23 tasks of the best 124 M candidate: $N23."; echo
    "$PY" scripts/freeze_choice.py --table "$CHOICE" 2>> "$LOGS/queue_summary.log"
    if [ -n "$SPEED" ]; then echo; echo "GPU encoding speed (RTX 3060 Ti, \`release.gpu_bench\`, fp16 eager, 1 024 queries and 4 096 passages of kd_v1; one run after the other in this stage):"; echo; echo "$SPEED"; fi
    echo; echo "Not done: CPU timing (by rule), ONNX exports and a package of the 124 M model (they follow a decision to replace)."
} > "$B2/notes_p5.md"
EXTRA_LABELS="${BEST124:-}" write_summary "Add the b2 results: the replacement rule over the 124 M candidates and the recommendation"
log "=== b2 p5 done ($(date '+%d.%m %H:%M')); rule: ${WHY:-no decision}; 23 tasks: $N23 ==="
