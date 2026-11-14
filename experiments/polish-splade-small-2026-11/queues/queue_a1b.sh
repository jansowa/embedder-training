#!/usr/bin/env bash
# Queue a1b (3060 Ti, 14.11 evening, ~5 h): the end of a1 with two corrections. a1 trained everything (student, KD,
# stage 3, round 2) and was stopped by hand at 19:10 during its measurements, because
#   - its form rule looked at dims/doc only: the first forms with <= 410 dims/doc of the new lineage are above the
#     postings limit (a preview: round 2 bias015 17 487, stage 3 bias010 17 573 postings/query; limit 16 807), so the
#     choice would have rejected the whole new lineage. The user's call: raise the bias instead. Here a form must fit
#     both limits (the first bias form, steps of 0.05, with <= 410 dims/doc and <= 16 807 postings/query);
#   - the heldout8 of round 2 bias015 came back with 4 of 8 tasks: a cost measurement started by hand next to the queue
#     took the GPU memory (CUDA OOM in the evaluation). The partial file is moved aside, never used.
# Also new: stage 3 of the new lineage is measured in full and is a candidate (it ties round 2 on tiny10 at fewer
# dims), the fidelity before KD that a1 missed (a dtype bug, fixed), and the stock-fp16 fidelity of the new student.
#   1  the fidelity files that are missing
#   2  the forms of round 2 and stage 3 of the new lineage within both limits
#   3  tiny10, heldout8, trec-covid of both; trec-covid of the best arm of r1; the query-side cost of all forms
#   4  the choice (the highest mean of tiny10 and heldout8 within the limits, no margin, the best arm of r1 as the
#      reference), held-out-2, the asymmetric runs of the best arm of r1 and of both new models, the package
#   5  summary_a1.md, commit, queue x1 again for the ONNX export
#   bash logs/relaunch-20261113/launch_queue.sh a1b experiments/polish-splade-small-2026-11/queues/queue_a1b.sh ""
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=a1b MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
MAIN=experiments/polish-splade-2026-10
source "$MAIN/queues/arm_common.sh"
source "$MAIN/queues/release_common.sh"
echo $$ > "$LOGS/queue_a1b.pid"
guard_alive
C="$EXP_ROOT/configs/3060"

mlbv() { regime_mean "$MAIN/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }   # the full models live in the main series
CORPUS=datasets/kd/kd_v1.jsonl
FULL=$(final_full); [ -n "$FULL" ] && [ -d "models/$FULL" ] || { log "  no frozen full model in $FINAL_FULL_FILE: a1b stops"; exit 1; }
RAW=${FULL%-bias[0-9][0-9][0-9]}
TEACHER="models/$RAW"                                    # read by fidelity() of queue_lib.sh
FTAG=$(echo "$FULL" | sed -e 's/^arm-dense-kl-n5kj-mined-//' -e 's/-ens400-r2v12//')
OLD_ST=b4-l0125-f1536-hl-r256-w; ST=$OLD_ST-dqf
OLD_KD=kd-$OLD_ST-full; KD=kd-$ST-full
REF=arm-hl-full-s3n5kj; REF_PICK=arm-hl-full-s3n5kj-bias010               # stage 3 of the old lineage
OLD_PICK=arm-hl-full-s3n5kj-r2d3-bias020; OTHER_PICK=arm-hl-full-s3n5kj-r2d3faq-bias020   # round 2 of the old lineage (r1)
S3=arm-hl-dqf-full-s3n5kj; NEW=$S3-r2d3; ROWS="dose3"                    # the new lineage
S3_FIRST=$S3-bias010; NEW_FIRST=$NEW-bias015                              # the forms a1 measured (dims/doc rule only)
COST="$RESULTS_ROOT/index_cost_a1.json"; FORMS="$RESULTS_ROOT/index_cost_a1_forms"; FREEZE="$RESULTS_ROOT/a1_freeze.json"; VERDICT="$RESULTS_ROOT/a1_reading.json"
MAX_POSTINGS=${MAX_POSTINGS:-16807}; MAX_DIMS=410
ASIDE="$LOGS/chain-20261114"; mkdir -p "$ASIDE" "$FORMS" "$RESULTS_ROOT/fidelity"
log "=== a1b start: the end of a1 with forms inside both limits ($MAX_DIMS dims/doc, $MAX_POSTINGS postings/query); frozen full model $FULL; $(free_gb) GB free on / ==="
for d in "models/$S3" "models/$NEW" "models/$KD" "models/students/$ST" "models/$OLD_PICK" "models/$REF_PICK"; do [ -d "$d" ] || { log "  $d missing: a1b stops"; exit 1; }; done
need_space || { log "  not enough space on /: a1b stops"; exit 1; }

# 0. what a1 left behind
if pgrep -f "pirb_regime_eval.sh quantized $NEW_FIRST" > /dev/null; then
    log "  waiting for the evaluation of $NEW_FIRST that a1 left running"
    while pgrep -f "pirb_regime_eval.sh quantized $NEW_FIRST" > /dev/null; do sleep 20; done; sleep 20
fi
TCF=$(regime_mean "$REGIMES_ROOT/quantized/$NEW_FIRST/heldout2_treccovid.json" 2>/dev/null)
[ -n "$TCF" ] && { log "  $NEW_FIRST trec-covid $(echo "$TCF" | awk '{print $1}') (finished after the stop of a1)"; record "regime:quantized:$NEW_FIRST:heldout2_treccovid" ok "$TCF" "finished after the stop of a1"; }
partial_aside() {  # partial_aside LABEL: a heldout8 file with fewer than 7 tasks is moved out of the results
    local f="$REGIMES_ROOT/quantized/$1/heldout8.json" n
    [ -f "$f" ] || return 0
    n=$(regime_mean "$f" | awk '{print $2}')
    if [ "${n:-0}" -lt 7 ]; then mv "$f" "$ASIDE/heldout8-partial-$1-$(date +%H%M%S).json"; log "  $1: heldout8 with ${n:-0} of 8 tasks moved to $ASIDE (not a result)"; fi
}
partial_aside "$NEW_FIRST"

fid() {  # fid FILE KIND: one line of a distill.fidelity JSON
    "$PY" -c "import json,sys;k=json.load(open(sys.argv[1]))['kinds'][sys.argv[2]];m=lambda n:k[n]['mean'];print('cosine %.4f, recall of the teacher dims %.3f, Jaccard %.3f, L1 rel. error %.3f, dims teacher / student %.0f / %.0f (n=%d)' % (m('cosine'), m('teacher_active_recall'), m('jaccard_active'), m('l1_relative_error'), m('teacher_active_dims'), m('student_active_dims'), k['n']))" "$1" "$2" 2>/dev/null || echo "not measured"
}
fidrow() {  # fidrow NAME FILE KIND: a table row
    "$PY" -c "import json,sys;k=json.load(open(sys.argv[2]))['kinds'][sys.argv[3]];m=lambda n:k[n]['mean'];print('| %s | %d | %.4f | %.3f | %.3f | %.3f | %.0f | %.0f |' % (sys.argv[1], k['n'], m('cosine'), m('teacher_active_recall'), m('jaccard_active'), m('l1_relative_error'), m('teacher_active_dims'), m('student_active_dims')))" "$1" "$2" "$3" 2>/dev/null || echo "| $1 | - | - | - | - | - | - | - |"
}
fix8() {  # fix8 LABEL: when heldout8 came back with 7 tasks (fiqa-pl failed), run fiqa-pl alone and fold it in
    local label="$1" n; n=$(regime_mean "$REGIMES_ROOT/quantized/$label/heldout8.json" 2>/dev/null | awk '{print $2}')
    [ "${n:-8}" = 7 ] || return 0
    log "  $label: heldout8 has 7 tasks; fiqa-pl alone"
    regime quantized "$label" "models/$label" heldout8_fiqa 16 && "$PY" scripts/merge_regime_task.py --into "$REGIMES_ROOT/quantized/$label/heldout8.json" \
        --from "$REGIMES_ROOT/quantized/$label/heldout8_fiqa.json" --n-tasks 8 | while read -r l; do log "  $l"; done
}
h8() {  # h8 LABEL: heldout8 with all 8 tasks; one repeat when the file is partial
    regime quantized "$1" "models/$1" heldout8 16; fix8 "$1"
    if [ "$(regime_mean "$REGIMES_ROOT/quantized/$1/heldout8.json" 2>/dev/null | awk '{print $2}')" != 8 ]; then
        partial_aside "$1"; regime quantized "$1" "models/$1" heldout8 16; fix8 "$1"
    fi
    [ "$(regime_mean "$REGIMES_ROOT/quantized/$1/heldout8.json" 2>/dev/null | awk '{print $2}')" = 8 ] || { log "  $1: heldout8 is not complete after a repeat"; partial_aside "$1"; return 1; }
}
pq() { "$PY" -c "import json,sys;print('%.0f' % json.load(open(sys.argv[1]))['models'][sys.argv[2]]['postings_per_query'])" "$1" "$2" 2>/dev/null; }
BIASES="005 010 015 020 025 030"
form_fit() {  # form_fit MODEL START: PICK = the first bias form from START on with <= MAX_DIMS dims/doc and <= MAX_POSTINGS postings/query
    local m="$1" start="$2" t p go=0; PICK=""
    for t in $BIASES; do
        [ "$t" = "$start" ] && go=1; [ "$go" = 1 ] || continue
        biased "models/$m" "$m-bias$t" "0.${t#0}" || return 1
        sparsity "$m-bias$t" "models/$m-bias$t"
        [ "$(gte "$MAX_DIMS" "$(dims "$m-bias$t")")" = 1 ] || { log "  $m-bias$t: $(dims "$m-bias$t") dims/doc, above $MAX_DIMS"; continue; }
        index_cost "$FORMS/$m-bias$t.json" "$m-bias$t"
        p=$(pq "$FORMS/$m-bias$t.json" "$m-bias$t")
        log "  $m-bias$t: $(dims "$m-bias$t") dims/doc, ${p:-?} postings/query (limits $MAX_DIMS and $MAX_POSTINGS)"
        if [ -n "$p" ] && [ "$(gte "$MAX_POSTINGS" "$p")" = 1 ]; then PICK="$m-bias$t"; return 0; fi
    done
    log "  $m: no bias form within both limits"; return 1
}

# 0b. the students after KD alone, before any ranking stage (asked for on 14.11: what the ranking training adds)
log "=== a1b step 0b: tiny10 after KD alone ==="
for m in "$KD" "$OLD_KD"; do
    sparsity "$m" "models/$m"; regime quantized "$m" "models/$m" tiny10 16
    log "  $m (KD alone, no bias shift): tiny10 $(lbv "$m" tiny10) at $(dims "$m") dims/doc"
done

# 1. fidelity
log "=== a1b step 1: the fidelity files a1 missed; the stock fp16 path of the new student ==="
F0="$RESULTS_ROOT/fidelity/$ST.cuda-fp16-fp32head.json"
if [ ! -f "$F0" ]; then
    wait_gpu_free; wait_ram
    timeout -k 60 1800 "$PY" -m distill.fidelity --teacher "$TEACHER" --student "models/students/$ST" --texts "$CORPUS" --n 256 --device cuda --dtype fp16 --fp32-head --output "$F0" > "$LOGS/fidelity_$ST.fp16-fp32head.log" 2>&1 \
        && record "fidelity:$ST.cuda-fp16-fp32head" ok "-" "-" || log "  fidelity of the new student before KD FAILED (logs/fidelity_$ST.fp16-fp32head.log)"
fi
F0S="$RESULTS_ROOT/fidelity/$ST.cuda-fp16.json"
if [ ! -f "$F0S" ]; then
    wait_gpu_free; wait_ram
    timeout -k 60 1800 "$PY" -m distill.fidelity --teacher "$TEACHER" --student "models/students/$ST" --texts "$CORPUS" --n 256 --device cuda --dtype fp16 --output "$F0S" > "$LOGS/fidelity_$ST.fp16.log" 2>&1 \
        && record "fidelity:$ST.cuda-fp16" ok "-" "-" || log "  stock-path fidelity of the new student before KD FAILED (logs/fidelity_$ST.fp16.log)"
fi
fidelity "student-$ST.queries" "models/students/$ST" "$CORPUS" --kinds query --n 512 --fp32-head
fidelity "student-$ST.queries-stock" "models/students/$ST" "$CORPUS" --kinds query --n 512
fidelity "$KD.queries-stock" "models/$KD" "$CORPUS" --kinds query --n 512
log "  before KD, 256 corpus texts: old (stock path) $(fid "$RESULTS_ROOT/fidelity/$OLD_ST.cuda-fp16.json" all)"
log "  before KD, 256 corpus texts: new, stock path $(fid "$F0S" all)"
log "  before KD, 256 corpus texts: new, fp32 head $(fid "$F0" all)"
log "  before KD, 512 queries: old (stock path) $(fid "$RESULTS_ROOT/fidelity/student-$OLD_ST.queries.json" query)"
log "  before KD, 512 queries: new, stock path $(fid "$RESULTS_ROOT/fidelity/student-$ST.queries-stock.json" query)"
log "  before KD, 512 queries: new, fp32 head $(fid "$RESULTS_ROOT/fidelity/student-$ST.queries.json" query)"
log "  after KD, 512 queries: old (stock path) $(fid "$RESULTS_ROOT/fidelity/$OLD_KD.queries.json" query)"
log "  after KD, 512 queries: new, stock path $(fid "$RESULTS_ROOT/fidelity/$KD.queries-stock.json" query)"
log "  after KD, 512 queries: new, fp32 head $(fid "$RESULTS_ROOT/fidelity/$KD.queries.json" query)"

# 2. forms within both limits
log "=== a1b step 2: bias forms within both limits ==="
form_fit "$NEW" 015; NEW_PICK=${PICK:-}
form_fit "$S3" 010; S3_PICK=${PICK:-}
log "  forms: round 2 ${NEW_PICK:-none} (a1 measured $NEW_FIRST), stage 3 ${S3_PICK:-none} (a1 measured $S3_FIRST)"
[ -n "$NEW_PICK" ] || [ -n "$S3_PICK" ] || { log "  the new lineage has no form within the limits: a1b stops without a choice"; exit 1; }

# 3. measure
log "=== a1b step 3: measure ==="
measure() {  # measure LABEL
    [ -n "$1" ] || return 0
    regime quantized "$1" "models/$1" tiny10 16; h8 "$1"
    regime quantized "$1" "models/$1" heldout2_treccovid 8
    log "  $1: $(lbs "$1") / trec-covid $(tc "$1") at $(dims "$1") dims/doc (the best arm of r1 $OLD_PICK $(lbs "$OLD_PICK") at $(dims "$OLD_PICK"))"
}
measure "$NEW_PICK"
measure "$S3_PICK"
regime quantized "$OLD_PICK" "models/$OLD_PICK" heldout2_treccovid 8
log "  the best arm of r1 $OLD_PICK: $(lbs "$OLD_PICK") / trec-covid $(tc "$OLD_PICK") at $(dims "$OLD_PICK") dims/doc"
COSTL=("$REF_PICK" "$OLD_PICK" "$OTHER_PICK" "$S3_FIRST" "$NEW_FIRST")
[ -n "$S3_PICK" ] && [ "$S3_PICK" != "$S3_FIRST" ] && COSTL+=("$S3_PICK")
[ -n "$NEW_PICK" ] && [ "$NEW_PICK" != "$NEW_FIRST" ] && COSTL+=("$NEW_PICK")
index_cost "$COST" "${COSTL[@]}"

# 4. the choice
log "=== a1b step 4: the ALBERT to release ==="
CANDS=("old lineage, round 2 on $ROWS (r1)=$OLD_PICK=$(dims "$OLD_PICK")=soup-v9b lineage")
[ -n "$NEW_PICK" ] && [ -n "$(lbv "$NEW_PICK" heldout8)" ] && CANDS+=("new lineage, round 2 on $ROWS (a1)=$NEW_PICK=$(dims "$NEW_PICK")=$RAW lineage")
[ -n "$S3_PICK" ] && [ -n "$(lbv "$S3_PICK" heldout8)" ] && CANDS+=("new lineage, stage 3 on n5 + kj (a1)=$S3_PICK=$(dims "$S3_PICK")=$RAW lineage")
CANDS+=("old lineage, stage 3 on n5 + kj (v45)=$REF_PICK=$(dims "$REF_PICK")=soup-v9b lineage")
"$PY" scripts/freeze_choice.py --regimes "$REGIMES_ROOT/quantized" --cost "$COST" --out "$FREEZE" --max-dims "$MAX_DIMS" --max-postings "$MAX_POSTINGS" --margin 0 --metric sets2 \
    "${CANDS[@]}" 2>&1 | while read -r l; do log "  $l"; done
WIN=$(jget "$FREEZE" winner)
[ -n "$WIN" ] && [ -d "models/$WIN" ] || { log "  no choice ($FREEZE): a1b stops without an ALBERT label"; exit 1; }
BEST_NEW=$("$PY" -c "
import sys
best = None
for label, t, h in zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3]):
    if label and t and h and (best is None or float(t) + float(h) > best[0]):
        best = (float(t) + float(h), label)
print(best[1] if best else '')" "$NEW_PICK" "$(lbv "$NEW_PICK" tiny10)" "$(lbv "$NEW_PICK" heldout8)" "$S3_PICK" "$(lbv "$S3_PICK" tiny10)" "$(lbv "$S3_PICK" heldout8)" 2>/dev/null)
{ echo "$WIN"; echo "# first line = the label of the ALBERT to release (models/<label>); written by queue a1b on $(date '+%d.%m %H:%M') by its rule: $(jget "$FREEZE" why)"; echo "# overwrite the first line to choose another model before queue x1 runs again"; } > "$FINAL_ALBERT_FILE"
log "  ALBERT to release: $WIN -> $FINAL_ALBERT_FILE (the best model of the new lineage: ${BEST_NEW:-none})"
case "$WIN" in $S3*) ;; *) log "  NOTE: THE RULE CHOSE A MODEL OF THE OLD LINEAGE ($WIN); the user decided the released ALBERT descends from the frozen model, so the best new model ${BEST_NEW:-none} is measured and packed too and the first line of $FINAL_ALBERT_FILE is to be confirmed";; esac
H2LIST="$WIN"; [ -n "$BEST_NEW" ] && [ "$BEST_NEW" != "$WIN" ] && case "$WIN" in $S3*) ;; *) H2LIST="$WIN $BEST_NEW";; esac
for m in $H2LIST; do for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do h2task "$m" "$ts"; log "  $m $ts $(lbv "$m" "$ts")"; done; done
asym() {  # asym LABEL: LABEL as the query encoder over the index of the frozen full model
    [ -n "$1" ] && [ -d "models/$1" ] || return 0
    PIRB_QUERY_MODEL="models/$1" regime quantized "asym-$FTAG-q-$1" "models/$FULL" tiny10 16
    PIRB_QUERY_MODEL="models/$1" regime quantized "asym-$FTAG-q-$1" "models/$FULL" heldout8 16
    log "  $1 as the query encoder over the index of $FULL: $(lbs "asym-$FTAG-q-$1") (symmetric: ALBERT $(lbs "$1"), full model $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8))"
}
asym "$WIN"
for m in "$BEST_NEW" "$NEW_PICK" "$S3_PICK" "$OLD_PICK"; do [ "$m" != "$WIN" ] && asym "$m"; done
for m in $H2LIST; do
    pack_add "albert/$m" "models/$m" "ALBERT student (50 M), release form with the bias baked in; $(lbs "$m") / trec-covid $(tc "$m"), $(dims "$m") dims/doc; as the query encoder over $FULL: $(lbs "asym-$FTAG-q-$m")"
done

# 5. summary
log "=== a1b step 5: summary ==="
SUM="$RESULTS_ROOT/summary_a1.md"
TA=$(lbv "$OLD_PICK" tiny10); TB=$(lbv "$OTHER_PICK" tiny10)
v() { local x; x=$(lbv "$1" "$2"); echo "${x:--}"; }
cget() { "$PY" -c "import json,sys;m=json.load(open(sys.argv[1]))['models'].get(sys.argv[2],{});v=m.get(sys.argv[3]);print(sys.argv[4] % v if v is not None else '-')" "$COST" "$1" "$2" "$3" 2>/dev/null || echo "-"; }
MCOST="$MAIN/results/index_cost_v62.json"
fcget() { "$PY" -c "import json,sys;m=json.load(open(sys.argv[1]))['models'].get(sys.argv[2],{});v=m.get(sys.argv[3]);print(sys.argv[4] % v if v is not None else '-')" "$MCOST" "$1" "$2" "$3" 2>/dev/null || echo "-"; }
row() {  # row NAME LABEL
    [ -n "$2" ] || return 0
    local b=""; [ "$2" = "$WIN" ] && b="**"
    echo "| $b$1$b | \`$2\` | $(v "$2" tiny10) | $(v "$2" heldout8) | $(v "$2" heldout2_treccovid) | $(v "asym-$FTAG-q-$2" tiny10) | $(v "asym-$FTAG-q-$2" heldout8) | $(dims "$2") | $(cget "$2" query_active_mean %.1f) | $(cget "$2" postings_per_query %.0f) |"
}
COLS=(--column "old lineage, stage 3 (v45)=$REF_PICK" --column "old lineage, round 2 (r1)=$OLD_PICK")
[ -n "$S3_PICK" ] && COLS+=(--column "new lineage, stage 3 (a1)=$S3_PICK")
[ -n "$NEW_PICK" ] && COLS+=(--column "new lineage, round 2 (a1)=$NEW_PICK")
{
    echo "# a1 (quantized regime, nDCG@10): the ALBERT student descending from the frozen full model"; echo
    echo "The old lineage: student \`$OLD_ST\` cut from soup-v9b -> KD from soup-v9b on kd_v1 -> stage 3 on n5 + kj (\`$REF\`) -> round 2 in r1. The new lineage repeats it with one intended change: the source model and the KD teacher are the raw weights of the frozen full model (\`$TEACHER\`; the frozen form \`$FULL\` is the same model with the bias shift). Student: layers 0,1,2,5, FFN 1536, the teacher's head transform as an extra layer, rank-256 vocabulary weighted by the kd_v1 token counts. KD: token KL x 0.3 + pooled MSE on all of kd_v1, one epoch, lr 5e-5, linear, batch 16, seed 42. Stage 3 and round 2: the configs of the old lineage with the start model replaced; the ranking teacher is unchanged (the ensemble scores stored in the rows). Round 2 ran on the rows of the r1 arm that won tiny10: **$ROWS** (r1: dose3 ${TA:--}, dose3 + faq20 ${TB:--})."; echo
    echo "One unintended difference: the old student was built before \`distill.student\` wrote the \`head_layer\` marker into config.json, so its KD ran the extra layer and the head transform on the stock fp16 path; the new student has the marker, so its KD ran them in fp32 with that layer's dropout off (\`distill.splade.prepare_head_layer\`). Evaluation (PIRB) uses the stock fp16 path for both."; echo
    echo "## Forms"; echo
    echo "Every model of the new lineage is measured in its first bias form (steps of 0.05) with at most $MAX_DIMS dims/doc **and** at most $MAX_POSTINGS postings/query. Queue a1 looked at dims/doc only and measured \`$S3_FIRST\` and \`$NEW_FIRST\`; both are above the postings limit ($(pq "$FORMS/$S3_FIRST.json" "$S3_FIRST") and $(pq "$FORMS/$NEW_FIRST.json" "$NEW_FIRST")), so a1 was stopped by hand on 14.11 at 19:10 and queue a1b took the next forms: stage 3 \`${S3_PICK:-none}\`, round 2 \`${NEW_PICK:-none}\`. The heldout8 of \`$NEW_FIRST\` is missing: its run lost 4 of 8 tasks to a CUDA out-of-memory error caused by a cost measurement started by hand next to the queue, and the partial file was moved out of the results."; echo
    echo "## Choice"; echo
    echo "Rule: the highest mean of tiny10 and heldout8 within the two limits, no margin; the best arm of r1 is the reference, both models of the new lineage and the stage-3 model of the old lineage are candidates."; echo
    echo "**ALBERT to release: \`$WIN\`** — $(jget "$FREEZE" why). The label is the first line of \`logs/final_albert_model.txt\`; the model is in \`$PACK/albert/$WIN\`."; echo
    case "$WIN" in $S3*) ;; *) echo "**The rule chose a model of the old lineage. The user decided on 14.11 that the released ALBERT descends from the frozen full model: the best model of the new lineage is \`${BEST_NEW:-none}\` (also in the package); the first line of \`logs/final_albert_model.txt\` is to be confirmed by the user.**"; echo;; esac
    echo "## Old lineage against the new one"; echo
    echo "asym = the model as the query encoder over the index of the frozen full model (\`$FULL\`, symmetric $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8))."; echo
    echo "| model | label | tiny10 | heldout8 | trec-covid | asym tiny10 | asym heldout8 | dims/doc | cost: dims/query | cost: postings/query |"
    echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    row "old lineage, stage 3 on n5 + kj (v45); above the postings limit" "$REF_PICK"
    row "old lineage, round 2 on $ROWS (r1)" "$OLD_PICK"
    row "old lineage, round 2, the other arm of r1" "$OTHER_PICK"
    [ "$S3_PICK" != "$S3_FIRST" ] && row "new lineage, stage 3, the form of a1 (above the postings limit)" "$S3_FIRST"
    row "new lineage, stage 3 on n5 + kj (a1)" "$S3_PICK"
    [ "$NEW_PICK" != "$NEW_FIRST" ] && row "new lineage, round 2, the form of a1 (above the postings limit)" "$NEW_FIRST"
    row "new lineage, round 2 on $ROWS (a1)" "$NEW_PICK"
    echo "| frozen full model (82 M), for scale | \`$FULL\` | $(mlbv "$FULL" tiny10) | $(mlbv "$FULL" heldout8) | $(mlbv "$FULL" heldout2_treccovid) | - | - | $(sparsity_mean "$MAIN/results/sparsity/$FULL.json" "$FULL" 2>/dev/null) | $(fcget "$FULL" query_active_mean %.1f) | $(fcget "$FULL" postings_per_query %.0f) |"
    echo
    echo "Unbiased dims/doc: old stage 3 $(dims "$REF"), old round 2 $(dims "${OLD_PICK%-bias[0-9][0-9][0-9]}"), new stage 3 $(dims "$S3"), new round 2 $(dims "$NEW"). One seed each; the seed noise of the ranking recipe is 0.13-0.2."; echo
    echo "The change of the source model and KD teacher alone, in the forms of a1 (stage 3 against stage 3, tiny10; both forms are above the postings limit): $([ -n "$(lbv "$S3_FIRST" tiny10)" ] && fsub "$(lbv "$S3_FIRST" tiny10)" "$(lbv "$REF_PICK" tiny10)" || echo -). Within the limits, against the best arm of r1: new stage 3 tiny10 $([ -n "$(lbv "$S3_PICK" tiny10)" ] && fsub "$(lbv "$S3_PICK" tiny10)" "$(lbv "$OLD_PICK" tiny10)" || echo -), heldout8 $([ -n "$(lbv "$S3_PICK" heldout8)" ] && fsub "$(lbv "$S3_PICK" heldout8)" "$(lbv "$OLD_PICK" heldout8)" || echo -); new round 2 tiny10 $([ -n "$(lbv "$NEW_PICK" tiny10)" ] && fsub "$(lbv "$NEW_PICK" tiny10)" "$(lbv "$OLD_PICK" tiny10)" || echo -), heldout8 $([ -n "$(lbv "$NEW_PICK" heldout8)" ] && fsub "$(lbv "$NEW_PICK" heldout8)" "$(lbv "$OLD_PICK" heldout8)" || echo -)."; echo
    echo "After KD alone, before any ranking stage (no bias shift): new \`$KD\` tiny10 $(v "$KD" tiny10) at $(dims "$KD") dims/doc, old \`$OLD_KD\` tiny10 $(v "$OLD_KD" tiny10) at $(dims "$OLD_KD") dims/doc; after stage 3 without a bias shift the old lineage has $(v "$REF" tiny10) at $(dims "$REF")."; echo
    echo "Training time of the new lineage: KD 14 359 s (3 h 59 min), stage 3 3 175 s, round 2 9 305 s."; echo
    echo "## Fidelity to the own teacher (old: soup-v9b; new: $RAW)"; echo
    echo "Stock path = the whole model in fp16, as inference runs it; fp32 head = the extra layer's feed-forward and the head transform in fp32, as the KD of the new student ran them. The old student has no \`head_layer\` marker, so its files named fp32head are the stock path as well."; echo
    echo "| student | texts | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|"
    fidrow "old, before KD, corpus texts (stock path)" "$RESULTS_ROOT/fidelity/$OLD_ST.cuda-fp16.json" all
    fidrow "new, before KD, corpus texts, stock path" "$F0S" all
    fidrow "new, before KD, corpus texts, fp32 head" "$F0" all
    fidrow "old, before KD, queries (stock path)" "$RESULTS_ROOT/fidelity/student-$OLD_ST.queries.json" query
    fidrow "new, before KD, queries, stock path" "$RESULTS_ROOT/fidelity/student-$ST.queries-stock.json" query
    fidrow "new, before KD, queries, fp32 head" "$RESULTS_ROOT/fidelity/student-$ST.queries.json" query
    fidrow "old, after KD, queries (stock path)" "$RESULTS_ROOT/fidelity/$OLD_KD.queries.json" query
    fidrow "new, after KD, queries, stock path" "$RESULTS_ROOT/fidelity/$KD.queries-stock.json" query
    fidrow "new, after KD, queries, fp32 head" "$RESULTS_ROOT/fidelity/$KD.queries.json" query
    echo; echo "Fidelity curve of the new KD (256 corpus texts per checkpoint):"; echo; echo '```'
    [ -f "runs/$KD/log.jsonl" ] && "$PY" scripts/kd_fidelity_curve.py "runs/$KD/log.jsonl" 2>> "$LOGS/queue_summary.log"
    echo '```'; echo
    PICKN="${BEST_NEW:-$NEW_PICK}"
    if [ -n "$PICKN" ] && [ -n "$(lbv "$PICKN" heldout8)" ] && [ -n "$(lbv "$OLD_PICK" heldout8)" ]; then
        "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" \
            "${COLS[@]}" --pick "$PICKN" --ref "$OLD_PICK" \
            --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
            --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
            --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
            --win "THE NEW LINEAGE WINS (>= +0.2 on both sets over the best arm of r1 within the limits)" \
            --noise "within the seed noise of the best arm of r1 set by set (the choice itself follows the rule above, on the mean of the two sets)" \
            --loss "a LOSS on both sets against the best arm of r1" --m23 "$WIN" --m23 "$PICKN" --json "$VERDICT" 2>> "$LOGS/queue_summary.log"
    fi
    echo; echo "Nothing further was started by this queue except queue x1 for the ONNX export of the chosen ALBERT."
    echo; echo "## Queue state (a1, a1b)"; echo; echo '```'; grep -E "$ST|hl-dqf-full|hl-full-s3n5kj-r2d3|asym-$FTAG-q-arm-hl" "$STATE" | tail -n 90; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_a1.sh" "$EXP_ROOT/queues/queue_a1b.sh" "$C/stage3_hl_dqf_full_n5kj_ens400.yaml" "$C/stage3_hl_dqf_full_r2_dose3.yaml" "$C/stage3_hl_dqf_full_r2_dose3_faq20.yaml" "$FREEZE" 2>/dev/null
for f in "$COST" "$FORMS" "$VERDICT" "$F0" "$F0S" "$RESULTS_ROOT/fidelity/student-$ST.queries.json" "$RESULTS_ROOT/fidelity/student-$ST.queries-stock.json" "$RESULTS_ROOT/fidelity/$KD.queries.json" "$RESULTS_ROOT/fidelity/$KD.queries-stock.json"; do [ -e "$f" ] && git add "$f" 2>/dev/null; done
for m in $KD $OLD_KD $REF_PICK $OLD_PICK $OTHER_PICK $S3 $S3_FIRST $S3_PICK $NEW $NEW_FIRST $NEW_PICK "asym-$FTAG-q-$OLD_PICK" "asym-$FTAG-q-$S3_PICK" "asym-$FTAG-q-$NEW_PICK"; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/arm-hl-dqf-full-s3n5kj*.json 2>/dev/null
git commit -q -m "Add the a1 results: the ALBERT student descending from the frozen full model" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop a1b
log "  queue x1 again for the ONNX export of $WIN and the README of the package"
env -u EXP_ROOT -u RESULTS_ROOT -u REGIMES_ROOT -u HOLDOUT_ROOT -u HOLDOUT2_ROOT -u QUEUE_TAG -u TRAIN_TIMEOUT -u KD_TIMEOUT -u WAIT_FOR -u MIN_RAM_GB bash "$MAIN/queues/queue_x1.sh" >> "$LOGS/queue_x1.out" 2>&1
log "  x1 finished: $(tail -n 1 "$LOGS/queue_x1.out" | cut -c1-200)"
log "=== a1b done ($(date '+%d.%m %H:%M')); ALBERT to release $WIN ==="
