#!/usr/bin/env bash
# v35 (4/5.11): takes over from v33 after its soups and replaces the two nq-pl evaluations (2 x 3.8 h for one task each)
# with better-value work:
#   1. v33's summary, commit and push (v33 is killed right after its "best soup:" line, before nq-pl starts);
#   2. the index-cost sample (document dims, query dims, postings per query) of the v11 soups next to soup-v9b;
#   3. the KD dose question at equal compute: KL from soup-v9b on fresh rows (18 k fiqa+SQuAD-v2 + msmarco-clean-20000,
#      one epoch, ~38 k examples) vs arm-distil-v9b-kl-n5v2 (18 k rows x 2 epochs, 63.39 / 42.80);
#   4. soup-v11-n6 with that arm in the n3 slot when its heldout8 beats the slot; held-out-2 when it reaches soup-v9b.
# Takeover pattern as v28 -> v29: SIGKILL on v33's process group (setsid session), then the nq-pl cache is removed.
#
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v35.sh > logs/queue_v35.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v35 MIN_RAM_GB=6
export WANDB_MODE=offline
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-20000}   # 18 k rows x 2 epochs took 11 571 s; 38 k x 1 epoch is the same step count
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v35.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN=${KITCHEN:-$HOME/work/speakleash/retrieval-data-kitchen}
V35_END=${V35_END:-2026-11-06 08:00}
END=$(date -d "$V35_END" +%s)
B_SCORE=2400; B_TRAIN=20000; B_TINY=900; B_H8=2700; B_COST=3600
V9B_H8=42.56; H8_MIN=62.0
LONG="models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"
ROWS_MS=datasets/judged/msmarco-clean-20000.jsonl
TMS=datasets/arms/msmarco-clean-20000-tv9b.jsonl
ARM=distil-v9b-kl-n6-fresh38k

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() { local need=$1; shift; fits "$need" || { log "no time before $V35_END for a ${need}s step; skipping: $*"; return 1; }; "$@"; }
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }

teacher_scores() {  # teacher_scores ROWS OUT: soup-v9b dot products into pos_scores / neg_scores (gate scores kept)
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
    else log "teacher scores FAILED after ${e}s (logs/tscore_$(basename "$out" .jsonl).log)"; record "tscore:$(basename "$out")" FAILED "-" "${e}s"; rm -f "$out"; return 1; fi
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
import json
v = []
for t in ['czywiesz', 'gpt35wiki', 'poquad', 'quora', 'treccovid']:
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

best_of() {  # best_of MODEL...: the model with the highest heldout8; models without heldout8 lose
    local best="" bv="" m h
    for m in "$@"; do
        [ -d "models/$m" ] || continue
        h=$(lbv "$m" heldout8); [ -n "$h" ] || continue
        if [ -z "$bv" ] || [ "$(ge "$h" "$bv")" = 1 ]; then best="$m"; bv="$h"; fi
    done
    echo "$best"
}

log "=== v35 start: end $V35_END; $(free_gb) GB free on /; takes over from v33 before its nq-pl step ==="

# ---- 1. take over from v33 right after its soups --------------------------------------------------------------------
if queue_alive v33; then
    V33=$(cat "$LOGS/queue_v33.pid")
    log "waiting for v33 (pid $V33) to log its best soup (end of step 3)"
    until grep -q "\[v33\]   best soup:" "$LOGS/queue.log" || ! queue_alive v33; do sleep 2; done
    if queue_alive v33; then
        # `timeout` puts each step into its own process group: kill the children's groups first, then the bash
        for c in $(pgrep -P "$V33"); do
            pg=$(ps -o pgid= -p "$c" 2>/dev/null | tr -d ' ')
            [ -n "$pg" ] && [ "$pg" != "$V33" ] && kill -9 -- "-$pg" 2>/dev/null
        done
        kill -9 "$V33" 2>/dev/null
        sleep 3
        pkill -9 -f "pirb_regime_eval.sh quantized .* nqpl" 2>/dev/null
        log "v33 (pid $V33, process group) killed before nq-pl: $(grep '\[v33\]   best soup:' "$LOGS/queue.log" | tail -1 | cut -d']' -f3-)"
        record "takeover:v33" ok "-" "killed before nq-pl by v35"
        sleep 20
        for d in cache/regimes/quantized_*_nqpl; do
            [ -d "$d" ] || continue
            n=${d#cache/regimes/quantized_}; n=${n%_nqpl}
            [ -f "$REGIMES_ROOT/quantized/$n/nqpl.json" ] || { rm -rf "$d"; log "removed the unfinished nq-pl cache $d"; }
        done
    else log "v33 exited on its own"; fi
else log "v33 not running"; fi
wait_gpu_free

# ---- 2. v33's summary, commit and push (its step 5) -------------------------------------------------------------------
SUM33="$RESULTS_ROOT/summary_v33.md"
if [ ! -f "$SUM33" ]; then
    {
        echo "# v33 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): better SQuAD-v2 translation (A/B) and SPLADE-to-SPLADE distillation on more rows"
        echo
        echo "Paired files: *v2* = the same rows, order, ids and scores as the old arm, only the SQuAD texts come from the better translation. Written by queue v35, which took over before v33's nq-pl step (2 x 3.8 h for one task each)."
        echo
        "$PY" scripts/regime_task_table.py --models base-splade-wrapped n3-5070 arm-n3v2-squad8k arm-n5-squad16k arm-n5v2-squad16k arm-distil-v9b-kl-n1 arm-distil-v9b-kl-n1v2 arm-distil-en-kl-n1v2 arm-distil-en-kl-n1v2-raw arm-distil-v9b-mmse-n1 arm-distil-v9b-kl-n4 arm-distil-v9b-kl-n4v2 arm-n4x-nq arm-distil-v9b-kl-n4xnq arm-distil-v9b-kl-n5v2 soup-v11-distil soup-v11-kl soup-v11-v2 soup-v11-nq soup-v11-n5 soup-v9b \
            --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
        echo; echo "## Held-out-2 (czywiesz, gpt35wiki, poquad, quora, treccovid)"; echo
        "$PY" scripts/regime_task_table.py --models soup-v11-kl soup-v11-v2 soup-v11-nq soup-v11-n5 soup-v9b \
            --tasksets heldout2_czywiesz heldout2_gpt35wiki heldout2_poquad heldout2_quora heldout2_treccovid 2>> "$LOGS/queue_summary.log"
        echo; echo "## Active document dimensions (base 436)"; echo
        "$PY" scripts/sparsity_table.py --model base=base --model arm-n3v2-squad8k --model arm-n5v2-squad16k --model arm-distil-v9b-kl-n1v2 --model arm-distil-en-kl-n1v2 --model arm-distil-en-kl-n1v2-raw --model arm-distil-v9b-kl-n4v2 --model arm-distil-v9b-kl-n4xnq --model arm-distil-v9b-kl-n5v2 --model soup-v9b --model soup-v11-kl --model soup-v11-v2 --model soup-v11-nq --model soup-v11-n5 2>> "$LOGS/queue_summary.log"
        echo; echo "## Queue state (v33)"; echo; echo '```'
        grep -E "^(tscore|train|sparsity|regime:quantized):(.*v2|.*n4xnq|.*mmse-n1:heldout8|soup-v11)" "$STATE" | tail -60
        echo '```'
    } > "$SUM33" 2>> "$LOGS/queue_v35.out"
    log "v33 summary written to $SUM33"
    for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM33" "$EXP_ROOT/queues/queue_v33.sh" "$C"/*v2*.yaml "$C/distil_v9b_kl_n4xnq.yaml" "$C"/distil_en_*.yaml; do
        [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v35.out"
    done
    git commit -q -m "Add the v33 results: better SQuAD-v2 translation and SPLADE distillation on more rows" 2>> "$LOGS/queue_v35.out" \
        && log "v33 results committed ($(git log --oneline -1))" || log "v33 results: nothing to commit or commit failed"
fi

# ---- 3. index-cost sample of the v11 soups ---------------------------------------------------------------------------
COST="$RESULTS_ROOT/index_cost_v11.json"
if [ ! -f "$COST" ] && fits $B_COST; then
    args=(--model base=models/base-splade-wrapped --model soup-v9b=models/soup-v9b)
    for m in soup-v11-v2 soup-v11-kl soup-v11-nq soup-v11-n5; do [ -d "models/$m" ] && args+=(--model "$m=models/$m"); done
    need_space; wait_gpu_free; wait_ram
    log "index cost: ${args[*]}"; s=$SECONDS
    timeout -k 60 "$B_COST" "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" \
        --output-json "$COST" > "$LOGS/index_cost_v11.log" 2>&1 \
        && { log "index-cost profile written ($COST) in $((SECONDS - s))s"; record "cost:v11" ok "-" "$((SECONDS - s))s"; } \
        || { log "index-cost profile FAILED (logs/index_cost_v11.log)"; record "cost:v11" FAILED "-" "$((SECONDS - s))s"; }
fi

# ---- 4. fresh-data distillation at equal compute -----------------------------------------------------------------------
log "=== v35 step 4: KL from soup-v9b on fresh rows (18 k + msmarco-clean-20000, one epoch) ==="
teacher_scores "$ROWS_MS" "$TMS"
if [ -s "$TMS" ]; then
    arm "$ARM" "$C/distil_v9b_kl_n6_fresh38k.yaml" "kl-n5v2 two epochs on 18 k $(lbs arm-distil-v9b-kl-n5v2), kl-n4v2 $(lbs arm-distil-v9b-kl-n4v2)"
else log "no teacher scores for $ROWS_MS; the fresh-data arm is skipped"; fi

# ---- 5. soup with the fresh-data arm in the n3 slot ----------------------------------------------------------------------
S1=$(best_of n1-5070 arm-distil-v9b-kl-n1 arm-distil-v9b-kl-n1v2)
S4=$(best_of n4-5070 arm-distil-v9b-kl-n4 arm-distil-v9b-kl-n4v2)
S3=$(best_of n3-5070 arm-n3v2-squad8k arm-distil-v9b-kl-n5v2 "arm-$ARM")
log "  slots: n1 -> ${S1:-?}, n3 -> ${S3:-?}, n4 -> ${S4:-?}"
if [ "$S3" = "arm-$ARM" ] && [ -n "$S1" ] && [ -n "$S4" ]; then
    # shellcheck disable=SC2086
    soup soup-v11-n6 "models/$S1" "models/$S3" "models/$S4" $LONG
else log "  arm-$ARM does not take the n3 slot (${S3:-none}); no soup-v11-n6"; fi
BEST=$(best_of soup-v11-kl soup-v11-v2 soup-v11-nq soup-v11-n5 soup-v11-n6)
log "  best soup: ${BEST:-none} $(lbs "${BEST:-soup-v9b}") vs soup-v9b $(lbs soup-v9b)"

# ---- 6. summary, commit, push ----------------------------------------------------------------------------------------------
SUM="$RESULTS_ROOT/summary_v35.md"
{
    echo "# v35 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): fresh rows instead of a second epoch in SPLADE distillation; index cost of the v11 soups"
    echo
    echo "arm-$ARM = KL from soup-v9b on the kl-n5v2 rows (18 k) plus msmarco-clean-20000, one epoch: the same number of steps as kl-n5v2 (18 k x 2 epochs), every example seen once."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-distil-v9b-kl-n4v2 arm-distil-v9b-kl-n5v2 "arm-$ARM" soup-v11-v2 soup-v11-n5 soup-v11-n6 soup-v9b \
        --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo; echo "## Held-out-2 (czywiesz, gpt35wiki, poquad, quora, treccovid)"; echo
    "$PY" scripts/regime_task_table.py --models soup-v11-v2 soup-v11-nq soup-v11-n5 soup-v11-n6 soup-v9b \
        --tasksets heldout2_czywiesz heldout2_gpt35wiki heldout2_poquad heldout2_quora heldout2_treccovid 2>> "$LOGS/queue_summary.log"
    echo; echo "## Active document dimensions (base 436) and index cost"; echo
    "$PY" scripts/sparsity_table.py --model base=base --model soup-v9b --model soup-v11-kl --model soup-v11-v2 --model soup-v11-nq --model soup-v11-n5 --model soup-v11-n6 --model arm-distil-v9b-kl-n5v2 --model "arm-$ARM" --cost-json "$COST" 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v35)"; echo; echo '```'
    grep -E "^(takeover|cost|tscore|train|sparsity|regime:quantized):(.*n6|v33|v11|msmarco-clean-20000|soup-v11-n6)" "$STATE" | tail -40
    echo '```'
} > "$SUM" 2>> "$LOGS/queue_v35.out"
log "summary written to $SUM"
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$SUM" "$COST" "$EXP_ROOT/queues/queue_v35.sh" "$C/distil_v9b_kl_n6_fresh38k.yaml"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v35.out"
done
if git commit -q -m "Add the v35 results: fresh-row distillation at equal compute and the index cost of the v11 soups" 2>> "$LOGS/queue_v35.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v35.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v35.log"; break; }; sleep 600; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v35.log)"
else log "nothing to commit or commit failed (see logs/queue_v35.out)"; fi
log "=== v35 done ($(date '+%d.%m %H:%M')) ==="
