#!/usr/bin/env bash
# Queue v60 (3060 Ti, after the diagnostics of v59, ~7.5 h): do the native FAQ rows add on top of the SQuAD-v2 dose?
# v56 (dose3: + 40 093 gated SQuAD-v2 rows) gave +0.32 / +0.23 and trec-covid +3.1 over the 20 k arm at equal cost; v58
# (faq20: + 20 000 native Polish FAQ rows) gave +0.20 / +0.14 with the gain on other tasks (Allegro FAQ A/B, mfaq against
# arguana, scifact, fiqa, legal questions). The arm of this queue is the candidate for the final model:
#   1  ROWS: datasets/arms/n5-plus-kj-mined-dose3.jsonl (60 385, verbatim) + the same 20 000 FAQ rows that went into v58
#      (cut out of n5-plus-kj-mined-faq20.jsonl by the group webfaq-v1, not drawn again). No mining.
#   2  ARM arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12: the recipe unchanged (from soup-v12, 2 epochs, LR 2e-5, 2x16,
#      512 tokens, KL from the scores in the rows: soup-v9b + PolDense-400M with the fixed dense scale). About 5.5 h.
#   3  MEASURE at the cost of the reference dose3-bias015: bias forms in steps of 0.01 until dims/doc <= 1.02 x the
#      reference's; tiny10, heldout8, trec-covid, query-side index cost.
#   4  READING against dose3-bias015: >= +0.2 on both sets = the new best. Task groups show whether the FAQ gain and the
#      SQuAD gain survive together.
#   5  THE WINNER of the pair (dose3-faq20 when it wins by the rule, otherwise dose3-bias015): the four missing held-out-2
#      tasks and the 23-task mean, next to the 20 k arm bias013 (58.37). results/v60_winner.json names the winner and its
#      row file for queue v59b (the hard MS MARCO rows on top of the winner).
#   6  summary_v60.md, commit, push loop. No further arm starts here.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v60.sh > logs/queue_v60.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v60 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-28800}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
echo $$ > "$LOGS/queue_v60.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
wait_for_queues ${WAIT_FOR:-v59}

M=arm-dense-kl-n5kj-mined-ens400-r2v12
D3=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12; FQ=arm-dense-kl-n5kj-mined-faq20-ens400-r2v12
F=arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12
A=datasets/arms
D3ROWS=$A/n5-plus-kj-mined-dose3.jsonl; FQROWS=$A/n5-plus-kj-mined-faq20.jsonl
DROWS=$A/n5-plus-kj-mined-dose3-faq20.jsonl
BASECFG="$C/distil_dense_kl_n5kj_mined_dose3_ens400_r2v12.yaml"
CFG="$C/distil_dense_kl_n5kj_mined_dose3_faq20_ens400_r2v12.yaml"
COST="$RESULTS_ROOT/index_cost_v60.json"; VERDICT="$RESULTS_ROOT/v60_reading.json"; WINNER="$RESULTS_ROOT/v60_winner.json"
D3_PICK=$(measured_form "$D3"); FQ_PICK=$(measured_form "$FQ")
log "=== v60 start: dose3 + FAQ; dose arm ${D3_PICK:-none} $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK") dims/doc, FAQ arm ${FQ_PICK:-none} $(lbs "$FQ_PICK") / $(tc "$FQ_PICK") at $(dims "$FQ_PICK"), 20 k arm bias013 $(lbs $M-bias013) / $(tc $M-bias013) at $(dims $M-bias013); $(free_gb) GB free on / ==="

# 0. checks
for f in "$D3ROWS" "$FQROWS" "$BASECFG" scripts/arm_reading.py; do [ -s "$f" ] || { log "  $f missing: v60 stops"; exit 1; }; done
[ -n "$D3_PICK" ] && [ -d "models/$D3_PICK" ] || { log "  the measured form of the dose arm is missing: v60 stops"; exit 1; }
[ -d models/soup-v12 ] || { log "  models/soup-v12 missing: v60 stops"; exit 1; }
need_space || { log "  not enough space on /: v60 stops"; exit 1; }

# 1. rows
log "=== v60 step 1: rows = $D3ROWS (unchanged) + the FAQ rows of v58 ($FQROWS, group webfaq-v1) ==="
merge_rows "$D3ROWS" "$FQROWS" "$DROWS" webfaq-v1 || { log "  row file $DROWS not written: v60 stops"; exit 1; }
NROWS=$(jget "$DROWS.rows.json" rows); NNEW=$(jget "$DROWS.rows.json" new_rows); NPAIRS=$(jget "$DROWS.rows.json" training_pairs)
log "  $DROWS: $NROWS rows, $NNEW FAQ rows ($(jget "$DROWS.rows.json" new_rows_left_out_duplicate_query) left out as duplicate queries), $NPAIRS training pairs"
[ "${NNEW:-0}" -ge 19900 ] && [ "${NNEW:-0}" -le 20000 ] || { log "  expected the 20 000 FAQ rows of v58, got ${NNEW:-0}: v60 stops"; exit 1; }

# 2. the arm
log "=== v60 step 2: arm $F ($NROWS rows, 2 epochs, the recipe of the mined arm) ==="
sed -e "4s|.*|# The dose + FAQ arm (queue v60): the rows of the dose arm of v56 ($D3ROWS, unchanged: the 20 k rows of the student-mined arm plus 40 093 gated SQuAD-v2 rows) plus the $NNEW native Polish FAQ rows of the FAQ arm of v58 (WebFAQ v2 + clips/mqa after the kitchen's selection funnel), all with negatives mined by r2v12 (5 best + 5 random of its top-60, length window 0.5-2.0) and scored by the ensemble teacher (soup-v9b + PolDense-400M) with the fixed dense scale 84.19441654567078. The recipe of the mined arm from soup-v12, only the row file differs.|" \
    -e "s|^runs_dir: .*|runs_dir: runs/$F|" -e "s|^train_data: .*|train_data: $DROWS|" "$BASECFG" > "$CFG"
train "${F#arm-}" "$CFG"

# 3. measure at the cost of the reference
log "=== v60 step 3: measure at the cost of $D3_PICK ($(dims "$D3_PICK") dims/doc) ==="
REF="dose arm $D3_PICK $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK"); FAQ arm ${FQ_PICK:-none} $(lbs "$FQ_PICK"); 20 k arm bias013 $(lbs $M-bias013) / $(tc $M-bias013) at $(dims $M-bias013)"
at_cost "$F" "$REF" "$D3_PICK"; F_PICK=${PICK:-}
[ -d "models/$F" ] || F_PICK=""
index_cost "$COST" "$M-bias013" "$D3_PICK" "$FQ_PICK" "$F_PICK"

reading() {  # reading [more arguments of scripts/arm_reading.py]
    "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" \
        --column "20 k arm (v49)=$M-bias013" --column "dose3 (v56)=$D3_PICK" --column "faq20 (v58)=${FQ_PICK:-$FQ}" --column "dose3 + faq20 (v60)=${F_PICK:-$F}" \
        --pick "${F_PICK:-$F}" --ref "$D3_PICK" \
        --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
        --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
        --group "scifact-pl=scifact-pl" --group "maupqa-mkqa=maupqa-mkqa" --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
        --win "THE NEW BEST (>= +0.2 on both sets over the dose arm at its cost): the FAQ rows add on top of the SQuAD-v2 dose" \
        --noise "within the seed noise of the dose arm: by the rule dose3 stays the best arm; the task groups below say whether the FAQ gain was kept at no cost elsewhere" \
        --loss "a LOSS on both sets: together the two sources are worse than the dose alone" "$@"
}

# 4. the winner of the pair
log "=== v60 step 4: reading against $D3_PICK and the winner of the pair ==="
reading --json "$VERDICT" > /dev/null
KIND=$(jget "$VERDICT" verdict)
if [ "$KIND" = win ] && [ -n "$F_PICK" ]; then WIN="$F_PICK"; WARM="$F"; WROWS="$DROWS"; WCFG="$CFG"; WHY="dose3 + faq20 wins by the rule (>= +0.2 on tiny10 and heldout8 over $D3_PICK)"
else WIN="$D3_PICK"; WARM="$D3"; WROWS="$D3ROWS"; WCFG="$BASECFG"; WHY="dose3 + faq20 does not win by the rule (reading: ${KIND:-none}), so the dose arm stays"; fi
"$PY" -c "import json,sys;json.dump(dict(zip(('winner','arm','rows','config','why','reading'),sys.argv[2:8])),open(sys.argv[1],'w'),indent=2)" "$WINNER" "$WIN" "$WARM" "$WROWS" "$WCFG" "$WHY" "${KIND:-none}"
log "  delta $(jget "$VERDICT" delta_tiny10) / $(jget "$VERDICT" delta_heldout8) -> $KIND; winner: $WIN ($WHY)"

# 5. held-out-2 and the 23-task mean of the winner
log "=== v60 step 5: held-out-2 of $WIN ==="
for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
    h2task "$WIN" "$ts"; log "  $WIN $ts $(lbv "$WIN" "$ts")"
done

# 6. summary
log "=== v60 step 6: summary ==="
SUM="$RESULTS_ROOT/summary_v60.md"
{
    echo "# v60 (quantized regime, nDCG@10): the SQuAD-v2 dose and the native FAQ rows together"; echo
    echo "dose3 + faq20 = the rows of the dose arm of v56 (\`$D3ROWS\`, unchanged) plus the same $NNEW native Polish FAQ rows that went into the FAQ arm of v58, cut out of its row file by their group; no new mining. The recipe is unchanged (from soup-v12, 2 epochs, LR 2e-5, 2x16, 512 tokens, KL from the ensemble teacher's scores in the rows). The new arm is measured in the first bias form (steps of 0.01) with at most 1.02 x the dims/doc of the reference \`$D3_PICK\`."; echo
    reading --m23 "$M-bias013" --m23 "$D3_PICK" --m23 "${F_PICK:-$F}" --json "$VERDICT"
    echo; echo "## Winner of the pair"; echo
    echo "**$WIN**: $WHY. Its held-out-2 tasks and 23-task mean are in the table above; the row without them belongs to the arm that was not chosen. Queue v59b trains the hard MS MARCO rows (when H passed its threshold) on the rows of this winner."; echo
    echo "Nothing further was started by this queue. All three arms with new rows (dose3, faq20, dose3 + faq20) also train for more steps than the 20 k arm; these numbers do not separate that from the data."
    echo; echo "## Rows"; echo; echo '```'; cat "$DROWS.rows.json"; echo; echo '```'
    echo; echo "## Queue state (v60)"; echo; echo '```'; grep -E "mined-dose3-faq20|regime:quantized:$WIN:heldout2_" "$STATE" | tail -n 40; echo '```'
} > "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-900 | while read -r l; do log "  READING: $l"; done
log "  23 tasks: $WIN $("$PY" -c "import json,sys;v=json.load(open(sys.argv[1])).get('m23',{}).get(sys.argv[2]);print('%.2f' % v if v is not None else '?')" "$VERDICT" "$WIN") (20 k arm bias013 58.37)"
git add "$SUM" "$EXP_ROOT/queues/queue_v60.sh" "$EXP_ROOT/queues/arm_common.sh" scripts/arm_reading.py "$CFG" "$VERDICT" "$WINNER" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null   # the rows (datasets/ is not tracked) are described in the summary
for m in ${F_PICK:-} $F $WIN; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$F-bias0??.json 2>/dev/null
git commit -q -m "Add the v60 results: the SQuAD-v2 dose and the FAQ rows together" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop v60
log "=== v60 done ($(date '+%d.%m %H:%M')) ==="
