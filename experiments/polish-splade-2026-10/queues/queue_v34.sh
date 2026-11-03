#!/usr/bin/env bash
# Queue v34 (3060 Ti, after v33). English teacher: an English SPLADE scores the ENGLISH originals of the n1v2 rows
# (fiqa-2073 from speakleash/fiqa_pl train_en.jsonl, squadv2-2073 from the better-translation file), the Polish
# student (sdadas/polish-splade) trains on the Polish texts with those scores (plan §12, user's idea of 3.11).
#   0  needs datasets/arms/fiqa-plus-squadv2v2-en.jsonl (kitchen cut/build_english_twin.py; the fiqa English file is
#      private on HF and is downloaded by the user with HF_TOKEN). Without it the queue only writes a note and exits.
#   1  teacher scores: opensearch-project/opensearch-neural-sparse-encoding-v2-distill (Apache-2.0, BEIR 0.528) on the
#      twin, written into the Polish rows (bridge/splade_teacher_scores.py --texts-from, suffix -ten).
#   2  arms at 2 x 16: distil-en-kl-n1v2 (score_scale auto), distil-en-kl-n1v2-raw (teacher units); sparsity, tiny10,
#      heldout8 when tiny10 >= 62.0. Controls: arm-distil-v9b-kl-n1v2 (Polish teacher), arm-distil-v9b-kl-n1 63.23 / 42.52.
#   3  soup-v11-en: the soup-v11-v2 slots with the better English arm in the n1 slot, when it beats the n1 slot's heldout8.
#   4  summary_v34.md, commit, push.
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v34.sh > logs/queue_v34.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v34 MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-7200}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v34.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN=${KITCHEN:-$HOME/work/speakleash/retrieval-data-kitchen}
TEACHER=${TEACHER:-opensearch-project/opensearch-neural-sparse-encoding-v2-distill}
V34_END=${V34_END:-2026-11-07 08:00}
END=$(date -d "$V34_END" +%s)
B_SCORE=2400; B_TRAIN=7200; B_TINY=900; B_H8=2700
V9B_H8=42.56; H8_MIN=62.0
LONG="models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"
TWIN=datasets/arms/fiqa-plus-squadv2v2-en.jsonl
ROWS=datasets/arms/fiqa-plus-squadv2v2.jsonl
TEN=datasets/arms/fiqa-plus-squadv2v2-ten.jsonl

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() { local need=$1; shift; fits "$need" || { log "no time before $V34_END for a ${need}s step; skipping: $*"; return 1; }; "$@"; }
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in v33 v32; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
wait_gpu_free
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }

log "=== v34 start: end $V34_END; $(free_gb) GB free on / ==="
if [ ! -s "$TWIN" ]; then
    log "v34: $TWIN missing. Build it first (user downloads speakleash/fiqa_pl train_en.jsonl with HF_TOKEN; then"
    log "  python3 $KITCHEN/cut/build_english_twin.py --rows $ROWS --en <fiqa train_en.jsonl> --en datasets/better_translated_squadv2/train_pl_v2.jsonl --out $TWIN)"
    log "=== v34 done without work ($(date '+%d.%m %H:%M')) ==="
    exit 0
fi
[ "$(wc -l < "$TWIN")" = "$(wc -l < "$ROWS")" ] || { log "v34: $TWIN has a different row count than $ROWS; stopping"; exit 1; }

log "=== v34 step 1: English teacher scores ($TEACHER) on the twin ==="
if [ -s "$TEN" ] && [ "$(wc -l < "$TEN")" = "$(wc -l < "$ROWS")" ]; then log "teacher scores: $TEN already present"
else
    need_space; wait_gpu_free; wait_ram
    log "teacher scores $TEACHER -> $TEN"; s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$B_SCORE" "$PY" "$KITCHEN/bridge/splade_teacher_scores.py" --teacher "$TEACHER" \
        --rows "$ROWS" --texts-from "$TWIN" --output "$TEN" --max-seq-length 512 --batch-size 16 > "$LOGS/tscore_$(basename "$TEN" .jsonl).log" 2>&1
    e=$((SECONDS - s))
    if [ -s "$TEN" ] && [ "$(wc -l < "$TEN")" = "$(wc -l < "$ROWS")" ]; then
        log "teacher scores done in ${e}s: $(tail -1 "$LOGS/tscore_$(basename "$TEN" .jsonl).log")"; record "tscore:$(basename "$TEN")" ok "-" "${e}s"
    else log "teacher scores FAILED after ${e}s (logs/tscore_$(basename "$TEN" .jsonl).log)"; record "tscore:$(basename "$TEN")" FAILED "-" "${e}s"; rm -f "$TEN"; exit 1; fi
fi

arm() {  # arm TAG CONFIG CONTROL-TEXT
    local tag="$1" cfg="$2" ctrl="$3" v
    step $B_TRAIN train "$tag" "$cfg" || return 1
    step 120 sparsity "arm-$tag" "models/arm-$tag"
    step $B_TINY regime quantized "arm-$tag" "models/arm-$tag" tiny10 16
    v=$(lbv "arm-$tag" tiny10)
    if [ -n "$v" ] && [ "$(ge "$v" "$H8_MIN")" = 1 ]; then step $B_H8 regime quantized "arm-$tag" "models/arm-$tag" heldout8 16; fi
    log "  arm-$tag: $(lbs "arm-$tag") ($ctrl), $(dims "arm-$tag") dims/doc"
}
best_of() { local best="" bv="" m h; for m in "$@"; do [ -d "models/$m" ] || continue; h=$(lbv "$m" heldout8); [ -n "$h" ] || continue; if [ -z "$bv" ] || [ "$(ge "$h" "$bv")" = 1 ]; then best="$m"; bv="$h"; fi; done; echo "$best"; }

log "=== v34 step 2: arms ==="
arm distil-en-kl-n1v2 "$C/distil_en_kl_n1v2.yaml" "Polish teacher kl-n1v2 $(lbs arm-distil-v9b-kl-n1v2), kl-n1 $(lbs arm-distil-v9b-kl-n1)"
arm distil-en-kl-n1v2-raw "$C/distil_en_kl_n1v2_raw.yaml" "scale auto $(lbs arm-distil-en-kl-n1v2)"

log "=== v34 step 3: soup-v11-en ==="
EN=$(best_of arm-distil-en-kl-n1v2 arm-distil-en-kl-n1v2-raw)
S1=$(best_of n1-5070 arm-distil-v9b-kl-n1 arm-distil-v9b-kl-n1v2)
S3=$(best_of n3-5070 arm-n3v2-squad8k)
S4=$(best_of n4-5070 arm-distil-v9b-kl-n4 arm-distil-v9b-kl-n4v2)
if [ -n "$EN" ] && [ -n "$S1" ] && [ -n "$S3" ] && [ -n "$S4" ] && [ "$(ge "$(lbv "$EN" heldout8)" "$(lbv "$S1" heldout8)")" = 1 ]; then
    if [ ! -d models/soup-v11-en ]; then
        need_space; wait_gpu_free; wait_ram
        # shellcheck disable=SC2086
        timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "models/$EN" "models/$S3" "models/$S4" $LONG --output models/soup-v11-en > "$LOGS/merge_soup-v11-en.log" 2>&1 \
            && log "soup soup-v11-en = $EN $S3 $S4 $LONG" || log "soup soup-v11-en FAILED"
    fi
    if [ -d models/soup-v11-en ]; then
        step 120 sparsity soup-v11-en models/soup-v11-en
        step $B_TINY regime quantized soup-v11-en models/soup-v11-en tiny10 16
        step $B_H8 regime quantized soup-v11-en models/soup-v11-en heldout8 16
        log "  soup-v11-en: $(lbs soup-v11-en) (soup-v11-v2 $(lbs soup-v11-v2), soup-v9b $(lbs soup-v9b))"
        h=$(lbv soup-v11-en heldout8)
        if [ -n "$h" ] && [ "$(ge "$h" "$V9B_H8")" = 1 ]; then
            for t in czywiesz gpt35wiki poquad quora treccovid; do step $B_H8 regime quantized soup-v11-en models/soup-v11-en "heldout2_$t" 16; done
        fi
    fi
else log "  English arm ${EN:-none} does not beat the n1 slot ${S1:-?} on heldout8; no soup-v11-en"; fi

log "=== v34 step 4: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v34.md"
{
    echo "# v34 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): English SPLADE teacher on the English originals, Polish student on the Polish texts"
    echo
    echo "Teacher: $TEACHER; rows: n1v2 (fiqa-2073 + squadv2-2073, better SQuAD translation); twin: $TWIN."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped n1-5070 arm-distil-v9b-kl-n1 arm-distil-v9b-kl-n1v2 arm-distil-en-kl-n1v2 arm-distil-en-kl-n1v2-raw soup-v11-v2 soup-v11-en soup-v9b \
        --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo; echo "## Held-out-2"; echo
    "$PY" scripts/regime_task_table.py --models soup-v11-v2 soup-v11-en soup-v9b \
        --tasksets heldout2_czywiesz heldout2_gpt35wiki heldout2_poquad heldout2_quora heldout2_treccovid 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-distil-v9b-kl-n1v2 --model arm-distil-en-kl-n1v2 --model arm-distil-en-kl-n1v2-raw --model soup-v11-en 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v34)"; echo; echo '```'
    grep -E "^(tscore|train|sparsity|regime:quantized):(.*-ten|.*distil-en|soup-v11-en)" "$STATE" | tail -30
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v34.out"
log "summary written to $SUM"
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM" "$EXP_ROOT/queues/queue_v34.sh" "$C"/distil_en_*.yaml; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v34.out"
done
if git commit -q -m "Add the v34 results: English SPLADE teacher on the English originals" 2>> "$LOGS/queue_v34.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v34.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v34.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v34.log)"
else log "nothing to commit or commit failed (see logs/queue_v34.out)"; fi
log "=== v34 done ($(date '+%d.%m %H:%M')) ==="
