#!/usr/bin/env bash
# Queue v33 (3060 Ti, after v32; the machine stays available). Two questions on the polish-splade student:
#   A  the better SQuAD-v2 translation (datasets/better_translated_squadv2, kitchen cut/replace_squadv2_translation.py:
#      paired files, same rows / order / ids / scores, only the Polish differs): n3v2-squad8k vs n3-5070 (62.72 / 41.94),
#      n5v2-squad16k vs arm-n5-squad16k (60.99 / 40.50), distil-v9b-kl-n1v2 vs arm-distil-v9b-kl-n1 (63.23 / 42.52).
#   B  SPLADE-to-SPLADE distillation (v32: kl-n1 63.23 / 42.52, kl-n4 63.32 tiny10) on more rows: kl-n4v2, kl-n4xnq
#      (nq-pl in the mix; MNRL control arm-n4x-nq 62.32 / 41.49, nq-pl 43.79), kl-n5v2 (16 k SQuAD; does KD keep paying
#      with the dose where MNRL did not?).
#   0  fiqa-pl of arm-distil-v9b-mmse-n1 heldout8 failed in v32 (disk full); the 7-task file is set aside and re-run.
#   1  teacher scores (soup-v9b dot products, kitchen bridge/splade_teacher_scores.py, suffix -tv9b) for the four KD files.
#   2  arms in priority order; each: sparsity, leaderboard tiny10; heldout8 when tiny10 >= 62.0.
#   3  soups of the v9b slots (n1, n3, n4, long-easy, lenfar): soup-v11-kl (v32 KD arms), soup-v11-v2 (best of each slot by
#      heldout8 among old / v2 / KD), soup-v11-nq (+ kl-n4xnq), soup-v11-n5 (16 k SQuAD arm in the n3 slot); held-out-2
#      when heldout8 >= soup-v9b (42.56).
#   4  nq-pl (4.5 h each, needs ~30 GB in the cache) for kl-n4xnq and the best soup, when there is room.
#   5  summary_v33.md, commit, push.
# Start: setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v33.sh > logs/queue_v33.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v33 MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-16000}   # 16 k rows x 2 epochs at 2 x 16 took 11 884 s in v27
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v33.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN=${KITCHEN:-$HOME/work/speakleash/retrieval-data-kitchen}
V33_END=${V33_END:-2026-11-06 08:00}
END=$(date -d "$V33_END" +%s)
B_SCORE=2400; B_TRAIN=16000; B_TINY=900; B_H8=2700; B_NQ=16200; NQ_MIN_FREE_GB=35
V9B_H8=42.56; H8_MIN=62.0
LONG="models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() { local need=$1; shift; fits "$need" || { log "no time before $V33_END for a ${need}s step; skipping: $*"; return 1; }; "$@"; }
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in v32 v31; do
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

teacher_scores() {  # teacher_scores ROWS OUT: soup-v9b dot products into pos_scores / neg_scores
    local rows="$1" out="$2"
    [ -s "$out" ] && [ "$(wc -l < "$out")" = "$(wc -l < "$rows")" ] && { log "teacher scores: $out already present"; return 0; }
    [ -f "$rows" ] || { log "teacher scores: $rows missing"; record "tscore:$(basename "$out")" SKIPPED "no rows"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "teacher scores soup-v9b -> $out"; local s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$B_SCORE" "$PY" "$KITCHEN/bridge/splade_teacher_scores.py" --teacher models/soup-v9b \
        --rows "$rows" --output "$out" --max-seq-length 512 --batch-size 16 > "$LOGS/tscore_$(basename "$out" .jsonl).log" 2>&1
    local e=$((SECONDS - s))
    if [ -s "$out" ] && [ "$(wc -l < "$out")" = "$(wc -l < "$rows")" ]; then
        log "teacher scores done in ${e}s: $(tail -1 "$LOGS/tscore_$(basename "$out" .jsonl).log")"; record "tscore:$(basename "$out")" ok "-" "${e}s"
    else log "teacher scores FAILED after ${e}s"; record "tscore:$(basename "$out")" FAILED "-" "${e}s"; rm -f "$out"; return 1; fi
}

arm() {  # arm TAG CONFIG CONTROL-TEXT: train, sparsity, tiny10; heldout8 when tiny10 >= H8_MIN
    local tag="$1" cfg="$2" ctrl="$3" v
    step $B_TRAIN train "$tag" "$cfg" || return 1
    step 120 sparsity "arm-$tag" "models/arm-$tag"
    step $B_TINY regime quantized "arm-$tag" "models/arm-$tag" tiny10 16
    v=$(lbv "arm-$tag" tiny10)
    if [ -n "$v" ] && [ "$(ge "$v" "$H8_MIN")" = 1 ]; then step $B_H8 regime quantized "arm-$tag" "models/arm-$tag" heldout8 16; fi
    log "  arm-$tag: $(lbs "arm-$tag") ($ctrl), $(dims "arm-$tag") dims/doc"
}

heldout2() {  # heldout2 NAME: the five held-out-2 tasks in the leaderboard regime
    local t; for t in czywiesz gpt35wiki poquad quora treccovid; do step $B_H8 regime quantized "$1" "models/$1" "heldout2_$t" 16; done
    log "  $1 held-out-2: $("$PY" -c "
import json, sys
v = []
for t in 'czywiesz gpt35wiki poquad quora treccovid'.split():
    try: v.append(json.load(open(f'$REGIMES_ROOT/quantized/$1/heldout2_{t}.json'))['mean_ndcg@10'])
    except Exception: pass
print(f'{sum(v)/len(v):.2f} over {len(v)} tasks' if v else 'missing')
" 2>/dev/null) (soup-v9b 69.42)"
}

soup() {  # soup NAME MODEL...: uniform soup, sparsity, tiny10, heldout8; held-out-2 when heldout8 >= soup-v9b
    local name="$1"; shift; local h
    if [ ! -d "models/$name" ]; then
        for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
        need_space || return 1; wait_gpu_free; wait_ram
        timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    step 120 sparsity "$name" "models/$name"
    step $B_TINY regime quantized "$name" "models/$name" tiny10 16
    step $B_H8 regime quantized "$name" "models/$name" heldout8 16
    log "  $name: $(lbs "$name") (soup-v9b 63.65 / 42.56), $(dims "$name") dims/doc"
    h=$(lbv "$name" heldout8)
    if [ -n "$h" ] && [ "$(ge "$h" "$V9B_H8")" = 1 ]; then heldout2 "$name"; fi
}

best_of() {  # best_of MODEL...: the model with the highest heldout8 (tiny10 breaks ties); models without heldout8 lose
    local best="" bv="" m h
    for m in "$@"; do
        [ -d "models/$m" ] || continue
        h=$(lbv "$m" heldout8); [ -n "$h" ] || continue
        if [ -z "$bv" ] || [ "$(ge "$h" "$bv")" = 1 ]; then best="$m"; bv="$h"; fi
    done
    echo "$best"
}

log "=== v33 start: end $V33_END; $(free_gb) GB free on /; soup-v9b $(lbs soup-v9b) ==="
log "=== v33 step 0: heldout8 of arm-distil-v9b-mmse-n1 again (fiqa-pl failed on a full disk in v32) ==="
H8="$REGIMES_ROOT/quantized/arm-distil-v9b-mmse-n1/heldout8.json"
if [ -f "$H8" ] && [ "$(regime_mean "$H8" | awk '{print $2}')" != "8" ]; then
    mv "$H8" "${H8%.json}.7tasks.json"; log "set aside the 7-task heldout8 of arm-distil-v9b-mmse-n1"
fi
step $B_H8 regime quantized arm-distil-v9b-mmse-n1 models/arm-distil-v9b-mmse-n1 heldout8 16
log "  arm-distil-v9b-mmse-n1: $(lbs arm-distil-v9b-mmse-n1) (kl-n1 $(lbs arm-distil-v9b-kl-n1))"

log "=== v33 step 1: teacher scores (soup-v9b) for the KD files ==="
for f in fiqa-plus-squadv2v2 mix3v2-6219 fiqa-squadv2v2-nq-6219 fiqa-plus-squadv2v2-18k; do
    step $B_SCORE teacher_scores "datasets/arms/$f.jsonl" "datasets/arms/$f-tv9b.jsonl"
done

log "=== v33 step 2: arms ==="
arm distil-v9b-kl-n1v2 "$C/distil_v9b_kl_n1v2.yaml" "kl-n1 old translation $(lbs arm-distil-v9b-kl-n1)"
arm n3v2-squad8k "$C/n3v2_squad8k.yaml" "n3-5070 old translation $(lbs n3-5070)"
arm distil-v9b-kl-n4xnq "$C/distil_v9b_kl_n4xnq.yaml" "arm-n4x-nq MNRL $(lbs arm-n4x-nq), kl-n4 $(lbs arm-distil-v9b-kl-n4)"
arm distil-v9b-kl-n4v2 "$C/distil_v9b_kl_n4v2.yaml" "kl-n4 old translation $(lbs arm-distil-v9b-kl-n4)"
arm n5v2-squad16k "$C/n5v2_squad16k.yaml" "arm-n5-squad16k old translation $(lbs arm-n5-squad16k)"
arm distil-v9b-kl-n5v2 "$C/distil_v9b_kl_n5v2.yaml" "n5v2-squad16k MNRL $(lbs arm-n5v2-squad16k)"

log "=== v33 step 3: soups ==="
# shellcheck disable=SC2086
soup soup-v11-kl models/arm-distil-v9b-kl-n1 models/n3-5070 models/arm-distil-v9b-kl-n4 $LONG
S1=$(best_of n1-5070 arm-distil-v9b-kl-n1 arm-distil-v9b-kl-n1v2)
S3=$(best_of n3-5070 arm-n3v2-squad8k)
S4=$(best_of n4-5070 arm-distil-v9b-kl-n4 arm-distil-v9b-kl-n4v2)
log "  slots: n1 -> ${S1:-?}, n3 -> ${S3:-?}, n4 -> ${S4:-?}"
if [ -n "$S1" ] && [ -n "$S3" ] && [ -n "$S4" ]; then
    # shellcheck disable=SC2086
    soup soup-v11-v2 "models/$S1" "models/$S3" "models/$S4" $LONG
    # shellcheck disable=SC2086
    [ -d models/arm-distil-v9b-kl-n4xnq ] && soup soup-v11-nq "models/$S1" "models/$S3" "models/$S4" $LONG models/arm-distil-v9b-kl-n4xnq
    S5=$(best_of arm-n5v2-squad16k arm-distil-v9b-kl-n5v2)
    if [ -n "$S5" ] && [ "$(ge "$(lbv "$S5" heldout8)" "$(lbv "$S3" heldout8)")" = 1 ]; then
        # shellcheck disable=SC2086
        soup soup-v11-n5 "models/$S1" "models/$S5" "models/$S4" $LONG
    else log "  no 16 k SQuAD arm beats the n3 slot (${S5:-none}); no soup-v11-n5"; fi
fi
BEST=$(best_of soup-v11-kl soup-v11-v2 soup-v11-nq soup-v11-n5)
log "  best soup: ${BEST:-none} $(lbs "${BEST:-soup-v9b}") vs soup-v9b $(lbs soup-v9b)"

log "=== v33 step 4: nq-pl (optional) ==="
nqpl() {  # nqpl LABEL: nq-pl in the leaderboard regime with the long timeout, when the cache has room
    [ -d "models/$1" ] || return 1
    [ -f "$REGIMES_ROOT/quantized/$1/nqpl.json" ] && { log "  $1 nq-pl already present: $(lbv "$1" nqpl)"; return 0; }
    local free; free=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    if [ "${free:-0}" -lt "$NQ_MIN_FREE_GB" ]; then log "  only ${free} GB free; skipping nq-pl for $1"; return 1; fi
    fits $B_NQ || { log "  no time for nq-pl of $1"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "regime quantized/$1/nqpl (batch 8, timeout ${B_NQ}s)"; local s=$SECONDS
    CUDA_VISIBLE_DEVICES=0 timeout -k 60 "$B_NQ" bash scripts/pirb_regime_eval.sh quantized "$1" "models/$1" nqpl 8 > "$LOGS/regime_quantized_$1_nqpl.out" 2>&1
    local e=$((SECONDS - s)) m; m=$(regime_mean "$REGIMES_ROOT/quantized/$1/nqpl.json")
    if [ -n "$m" ] && [ "$(echo "$m" | awk '{print $2}')" != "0" ]; then log "regime quantized/$1/nqpl -> mean=$m in ${e}s"; record "regime:quantized:$1:nqpl" ok "$m" "${e}s"
    else log "regime quantized/$1/nqpl FAILED after ${e}s"; record "regime:quantized:$1:nqpl" FAILED "-" "${e}s"; rm -rf "cache/regimes/quantized_$1_nqpl"; return 1; fi
}
nqpl arm-distil-v9b-kl-n4xnq
[ -n "$BEST" ] && nqpl "$BEST"
log "  nq-pl: base $(lbv base-splade-wrapped nqpl), soup-v9b $(lbv soup-v9b nqpl), arm-n4x-nq $(lbv arm-n4x-nq nqpl), kl-n4xnq $(lbv arm-distil-v9b-kl-n4xnq nqpl), ${BEST:-none} $(lbv "${BEST:-x}" nqpl)"

log "=== v33 step 5: summary, commit, push ==="
SUM="$RESULTS_ROOT/summary_v33.md"
{
    echo "# v33 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): better SQuAD-v2 translation (A/B) and SPLADE-to-SPLADE distillation on more rows"
    echo
    echo "Paired files: *v2* = the same rows, order, ids and scores as the old arm, only the SQuAD texts come from the better translation."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped n3-5070 arm-n3v2-squad8k arm-n5-squad16k arm-n5v2-squad16k arm-distil-v9b-kl-n1 arm-distil-v9b-kl-n1v2 arm-distil-v9b-mmse-n1 arm-distil-v9b-kl-n4 arm-distil-v9b-kl-n4v2 arm-n4x-nq arm-distil-v9b-kl-n4xnq arm-distil-v9b-kl-n5v2 soup-v11-distil soup-v11-kl soup-v11-v2 soup-v11-nq soup-v11-n5 soup-v9b \
        --tasksets tiny10 heldout8 nqpl 2>> "$LOGS/queue_summary.log"
    echo; echo "## Held-out-2 (czywiesz, gpt35wiki, poquad, quora, treccovid)"; echo
    "$PY" scripts/regime_task_table.py --models soup-v11-kl soup-v11-v2 soup-v11-nq soup-v11-n5 soup-v9b \
        --tasksets heldout2_czywiesz heldout2_gpt35wiki heldout2_poquad heldout2_quora heldout2_treccovid 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436)"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model arm-n3v2-squad8k --model arm-n5v2-squad16k --model arm-distil-v9b-kl-n1v2 --model arm-distil-v9b-kl-n4v2 --model arm-distil-v9b-kl-n4xnq --model arm-distil-v9b-kl-n5v2 --model soup-v11-kl --model soup-v11-v2 --model soup-v11-nq --model soup-v11-n5 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v33)"; echo; echo '```'
    grep -E "^(tscore|train|sparsity|regime:quantized):(.*v2|.*n4xnq|.*mmse-n1:heldout8|soup-v11)" "$STATE" | tail -40
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v33.out"
log "summary written to $SUM"
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM" "$EXP_ROOT/queues/queue_v33.sh" "$C"/*v2*.yaml "$C/distil_v9b_kl_n4xnq.yaml"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v33.out"
done
if git commit -q -m "Add the v33 results: better SQuAD-v2 translation and SPLADE distillation on more rows" 2>> "$LOGS/queue_v33.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v33.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v33.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v33.log)"
else log "nothing to commit or commit failed (see logs/queue_v33.out)"; fi
log "=== v33 done ($(date '+%d.%m %H:%M')) ==="
