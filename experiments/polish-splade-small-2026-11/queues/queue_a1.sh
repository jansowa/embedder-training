#!/usr/bin/env bash
# Queue a1 (3060 Ti, 14.11 after x1; ~12.5 h): the ALBERT student (50 M) once more, descending from the frozen full
# model. The lineage so far (student cut from soup-v9b -> KD from soup-v9b on kd_v1 -> stage 3 on n5 + kj -> round 2 in
# r1) starts from a model about 1.3 below today's on tiny10. The user's decision of 14.11: repeat exactly the proven
# recipe and change only the source model and the KD teacher, both = the raw weights of the frozen full model
# (models/soup-v20-dqf; the bias form serves measurement only, as soup-v9b was used without a bias). No new architecture
# (no live head attention): a clean comparison "soup-v9b -> the frozen model".
#   1  STUDENT  distill.student from the frozen model, the arguments of b4-l0125-f1536-hl-r256-w in queue_s1.sh (CPU, a
#               minute); fidelity before KD next to the old student's
#   2  KD       the recipe of queue_s3.sh (token KL x 0.3 + pooled MSE, all of kd_v1, one epoch, lr 5e-5, warmup 300,
#               linear, fp16, 256 tokens, batch 16, seed 42), teacher = the frozen model; ~3.9 h; query fidelity next to
#               the old -full (cosine 0.9623, recall 0.775)
#   3  STAGE 3  as stage3_hl_full_n5kj_ens400.yaml from the new KD (~1 h); tiny10 at <= 410 dims/doc (what the change of
#               the teacher alone gives)
#   4  ROUND 2  as r1, on the rows of the r1 arm that won tiny10 (dose3 or dose3 + faq20); 2.7-3.4 h. The ranking
#               teacher is unchanged: the ensemble scores stored in the rows.
#   5  MEASURE  the first bias form with <= 410 dims/doc: tiny10, heldout8, trec-covid, the query-side cost; for the
#               best arm of r1 what is missing (heldout8 if not there, trec-covid); both as the query encoder over the
#               index of the frozen full model
#   6  CHOICE   the highest mean of tiny10 and heldout8 within <= 410 dims/doc and <= 16 807 postings/query, no margin:
#               the new model wins when it is above the best arm of r1 (and above the stage-3 model of the old lineage,
#               which stays a candidate). The winner: held-out-2, 23 tasks, logs/final_albert_model.txt, the package,
#               queue x1 again for its ONNX export.
#   WAIT_FOR="r1b st1 x1" setsid nohup bash experiments/polish-splade-small-2026-11/queues/queue_a1.sh > logs/queue_a1.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=a1 MIN_RAM_GB=6
export EXP_ROOT=experiments/polish-splade-small-2026-11
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export KD_TIMEOUT=${KD_TIMEOUT:-25200} TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-21600}
source scripts/queue_lib.sh
MAIN=experiments/polish-splade-2026-10
source "$MAIN/queues/arm_common.sh"
source "$MAIN/queues/release_common.sh"
echo $$ > "$LOGS/queue_a1.pid"
guard_alive
C="$EXP_ROOT/configs/3060"
wait_for_queues ${WAIT_FOR:-r1b st1 x1}

mlbv() { regime_mean "$MAIN/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }   # the full models live in the main series
CORPUS=datasets/kd/kd_v1.jsonl
FULL=$(final_full); [ -n "$FULL" ] && [ -d "models/$FULL" ] || { log "  no frozen full model in $FINAL_FULL_FILE: a1 stops"; exit 1; }
RAW=${A1_TEACHER:-${FULL%-bias[0-9][0-9][0-9]}}          # the frozen model without the bias shift
TEACHER="models/$RAW"                                    # read by fidelity() of queue_lib.sh
FTAG=$(echo "$FULL" | sed -e 's/^arm-dense-kl-n5kj-mined-//' -e 's/-ens400-r2v12//')
OLD_ST=b4-l0125-f1536-hl-r256-w; ST=$OLD_ST-dqf          # students: models/students/<name>
OLD_KD=kd-$OLD_ST-full; KD=kd-$ST-full
REF=arm-hl-full-s3n5kj; REF_PICK=arm-hl-full-s3n5kj-bias010               # stage 3 of the old lineage
R1A=arm-hl-full-s3n5kj-r2d3; R1B=arm-hl-full-s3n5kj-r2d3faq             # round 2 of the old lineage (r1)
S3=arm-hl-dqf-full-s3n5kj                                                # stage 3 of the new lineage
COST="$RESULTS_ROOT/index_cost_a1.json"; FREEZE="$RESULTS_ROOT/a1_freeze.json"; VERDICT="$RESULTS_ROOT/a1_reading.json"
MAX_POSTINGS=${MAX_POSTINGS:-16807}; MAX_DIMS=410
log "=== a1 start: the ALBERT student from the frozen full model $FULL ($(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8)); source and KD teacher $TEACHER; old lineage: stage 3 $REF_PICK $(lbs $REF_PICK), round 2 on dose3 $(lbs ${R1A}-bias020); $(free_gb) GB free on /, $(df -BG --output=avail runs/ | tail -1 | tr -dc '0-9') GB under runs/ ==="
for f in "$TEACHER/model.safetensors" "$CORPUS" "$EXP_ROOT/results/token_counts_kd_v1_sample.json" datasets/synth/chunks_pool.jsonl datasets/arms/n5-plus-kj-tens400.jsonl datasets/arms/n5-plus-kj-mined-dose3.jsonl \
         "$C/stage3_hl_dqf_full_n5kj_ens400.yaml" "$C/stage3_hl_dqf_full_r2_dose3.yaml" "$C/stage3_hl_dqf_full_r2_dose3_faq20.yaml" scripts/freeze_choice.py scripts/arm_reading.py; do
    [ -s "$f" ] || { log "  $f missing: a1 stops"; exit 1; }
done
need_space || { log "  not enough space on /: a1 stops"; exit 1; }
[ "$(df -BG --output=avail runs/ | tail -1 | tr -dc '0-9')" -ge 10 ] || { log "  less than 10 GB under runs/ (KD checkpoints): a1 stops"; exit 1; }

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
BIASES="005 010 015 020 025 030"
form410() {  # form410 MODEL: PICK = the first bias form (steps of 0.05) with <= 410 dims/doc; the other forms are removed
    local m="$1" t; PICK="$m"
    [ -d "models/$m" ] || { log "$m missing: not measured"; PICK=""; return 1; }
    sparsity "$m" "models/$m"
    if [ "$(gte "$MAX_DIMS" "$(dims "$m")")" != 1 ]; then
        for t in $BIASES; do
            biased "models/$m" "$m-bias$t" "0.${t#0}" || return 1
            sparsity "$m-bias$t" "models/$m-bias$t"; PICK="$m-bias$t"
            [ "$(gte "$MAX_DIMS" "$(dims "$PICK")")" = 1 ] && break
        done
        for t in $BIASES; do [ "$m-bias$t" != "$PICK" ] && rm -rf "models/$m-bias$t"; done
    fi
    log "  $m: $(dims "$m") dims/doc unbiased; measured form $PICK at $(dims "$PICK") (cap $MAX_DIMS)"
}
tiny_form() {  # tiny_form MODEL: the bias form of MODEL that has a tiny10 result (the last one), empty when none
    local p="" t; for t in "" -bias005 -bias010 -bias015 -bias020 -bias025 -bias030; do [ -n "$(lbv "$1$t" tiny10)" ] && p="$1$t"; done; echo "$p"
}

# 1. the student
log "=== a1 step 1: the student from $TEACHER ==="
mkdir -p "$RESULTS_ROOT/fidelity"
if [ ! -d "models/students/$ST" ]; then
    wait_ram; s=$SECONDS
    log "student $ST (layers 0,1,2,5, FFN 1536, head layer, rank-256 vocabulary weighted by the kd_v1 token counts)"
    "$PY" -m distill.student --teacher "$TEACHER" --out "models/students/$ST" --keep-layers 0,1,2,5 --intermediate-size 1536 --embedding-rank 256 --head-layer \
        --token-counts "$EXP_ROOT/results/token_counts_kd_v1_sample.json" --calibration-texts datasets/synth/chunks_pool.jsonl --calibration-sample 96 --threads 4 > "$LOGS/student_$ST.log" 2>&1 \
        || { log "student $ST FAILED (logs/student_$ST.log): a1 stops"; record "student:$ST" FAILED "-" "-"; rm -rf "models/students/$ST"; exit 1; }
    record "student:$ST" ok "-" "$((SECONDS - s))s"
    log "student $ST built in $((SECONDS - s)) s: $("$PY" -c "import json;d=json.load(open('models/students/$ST/student.json'));c=d['calibration'];print('%.1f M params, %.2f of the teacher MACs; embedding subspace energy %.4f, head subspace energy %.4f' % (d['params']['total']/1e6, d['macs_per_token']['ratio'], c['embedding_subspace_energy'], c['head_subspace_energy']))" 2>/dev/null) (old student: 49.8 M, 0.45; 0.7863, 0.9960)"
fi
F0="$RESULTS_ROOT/fidelity/$ST.cuda-fp16-fp32head.json"
if [ ! -f "$F0" ]; then
    wait_gpu_free; wait_ram
    timeout -k 60 1800 "$PY" -m distill.fidelity --teacher "$TEACHER" --student "models/students/$ST" --texts "$CORPUS" --n 256 --device cuda --dtype fp16 --fp32-head --output "$F0" > "$LOGS/fidelity_$ST.fp16-fp32head.log" 2>&1 \
        && record "fidelity:$ST.cuda-fp16-fp32head" ok "-" "-" || log "  fidelity of the new student before KD FAILED (logs/fidelity_$ST.fp16-fp32head.log)"
fi
fidelity "student-$ST.queries" "models/students/$ST" "$CORPUS" --kinds query --n 512 --fp32-head
log "  before KD, 256 corpus texts: new $(fid "$F0" all)"
log "  before KD, 256 corpus texts: old $(fid "$RESULTS_ROOT/fidelity/$OLD_ST.cuda-fp16-fp32head.json" all)"
log "  before KD, 512 queries: new $(fid "$RESULTS_ROOT/fidelity/student-$ST.queries.json" query)"
log "  before KD, 512 queries: old $(fid "$RESULTS_ROOT/fidelity/student-$OLD_ST.queries.json" query)"

# 2. KD
log "=== a1 step 2: KD of $ST on the whole corpus, teacher $TEACHER ==="
KD_COMMON=(--teacher "$TEACHER" --corpus "$CORPUS" --max-seq-length 256 --batch-size 16
           --lr 5e-5 --warmup-steps 300 --schedule linear --epochs 1 --precision fp16 --head-chunk 64
           --save-every 3000 --keep-last 1 --fidelity-texts "$CORPUS" --fidelity-n 256 --log-every 50)
KD_LOSS=(--token-loss kl --token-weight 0.3 --temperature 2.0 --top-k 128 --pooled-loss mse --pooled-weight 1.0)
kd "${KD#kd-}" "models/students/$ST" "${KD_COMMON[@]}" "${KD_LOSS[@]}" --seed 42
[ -d "models/$KD" ] || { log "  no KD student (logs/kd_${KD#kd-}.log): a1 stops; the ALBERT of the old lineage is not chosen here"; exit 1; }
fidelity "$KD.queries" "models/$KD" "$CORPUS" --kinds query --n 512 --fp32-head
log "  after KD, 512 queries: new $(fid "$RESULTS_ROOT/fidelity/$KD.queries.json" query)"
log "  after KD, 512 queries: old $(fid "$RESULTS_ROOT/fidelity/$OLD_KD.queries.json" query)"

# 3. stage 3
log "=== a1 step 3: stage 3 on n5 + kj ==="
train "${S3#arm-}" "$C/stage3_hl_dqf_full_n5kj_ens400.yaml"
[ -d "models/$S3" ] || { log "  no stage-3 model: a1 stops"; exit 1; }
form410 "$S3"; S3_PICK=${PICK:-}
[ -n "$S3_PICK" ] && regime quantized "$S3_PICK" "models/$S3_PICK" tiny10 16
log "  stage 3, new lineage: $S3_PICK tiny10 $(lbv "$S3_PICK" tiny10) at $(dims "$S3_PICK") dims/doc ($(dims "$S3") unbiased); old lineage: $REF_PICK tiny10 $(lbv "$REF_PICK" tiny10) at $(dims "$REF_PICK") ($(dims "$REF") unbiased)"

# 4. round 2 on the rows of the r1 arm that won tiny10
log "=== a1 step 4: round 2 ==="
R1A_PICK=$(tiny_form "$R1A"); R1B_PICK=$(tiny_form "$R1B")
TA=$(lbv "$R1A_PICK" tiny10); TB=""; [ -n "$R1B_PICK" ] && TB=$(lbv "$R1B_PICK" tiny10)
if [ -n "$TB" ] && [ -n "$TA" ] && [ "$TA" != "$TB" ] && [ "$(gte "$TB" "$TA")" = 1 ] && [ -s datasets/arms/n5-plus-kj-mined-dose3-faq20.jsonl ]; then
    OLD_PICK="$R1B_PICK"; OTHER_PICK="$R1A_PICK"; NEW=$S3-r2d3faq; CFG="$C/stage3_hl_dqf_full_r2_dose3_faq20.yaml"; ROWS="dose3 + faq20"
else
    OLD_PICK="$R1A_PICK"; OTHER_PICK="$R1B_PICK"; NEW=$S3-r2d3; CFG="$C/stage3_hl_dqf_full_r2_dose3.yaml"; ROWS="dose3"
fi
[ -n "${A1_ROWS:-}" ] && { case "$A1_ROWS" in faq) OLD_PICK="$R1B_PICK"; OTHER_PICK="$R1A_PICK"; NEW=$S3-r2d3faq; CFG="$C/stage3_hl_dqf_full_r2_dose3_faq20.yaml"; ROWS="dose3 + faq20";; dose3) OLD_PICK="$R1A_PICK"; OTHER_PICK="$R1B_PICK"; NEW=$S3-r2d3; CFG="$C/stage3_hl_dqf_full_r2_dose3.yaml"; ROWS="dose3";; esac; }
log "  r1 on tiny10: dose3 ${TA:--} ($R1A_PICK), dose3 + faq20 ${TB:--} (${R1B_PICK:-none}); rows of round 2: $ROWS; the best arm of r1: $OLD_PICK"
train "${NEW#arm-}" "$CFG"

# 5. measure
log "=== a1 step 5: measure at <= $MAX_DIMS dims/doc ==="
NEW_PICK=""
if [ -d "models/$NEW" ]; then
    form410 "$NEW"; NEW_PICK=${PICK:-}
    if [ -n "$NEW_PICK" ]; then
        regime quantized "$NEW_PICK" "models/$NEW_PICK" tiny10 16; regime quantized "$NEW_PICK" "models/$NEW_PICK" heldout8 16; fix8 "$NEW_PICK"
        log "  $NEW_PICK: $(lbs "$NEW_PICK") at $(dims "$NEW_PICK") dims/doc (the best arm of r1 $OLD_PICK $(lbs "$OLD_PICK") at $(dims "$OLD_PICK"); stage 3 of the old lineage $(lbs "$REF_PICK"))"
        regime quantized "$NEW_PICK" "models/$NEW_PICK" heldout2_treccovid 8; log "  $NEW_PICK trec-covid $(tc "$NEW_PICK")"
    fi
else log "  round 2 gave no model: the new lineage has no candidate"; fi
if [ -n "$OLD_PICK" ] && [ -d "models/$OLD_PICK" ]; then
    regime quantized "$OLD_PICK" "models/$OLD_PICK" heldout8 16; fix8 "$OLD_PICK"
    regime quantized "$OLD_PICK" "models/$OLD_PICK" heldout2_treccovid 8
    log "  the best arm of r1 $OLD_PICK: $(lbs "$OLD_PICK") / trec-covid $(tc "$OLD_PICK") at $(dims "$OLD_PICK") dims/doc"
fi
index_cost "$COST" "$REF_PICK" "$OLD_PICK" "$OTHER_PICK" "$S3_PICK" "$NEW_PICK"

# 6. the choice
log "=== a1 step 6: the ALBERT to release ==="
CANDS=()
[ -n "$OLD_PICK" ] && [ -n "$(lbv "$OLD_PICK" heldout8)" ] && CANDS+=("old lineage, round 2 on $ROWS (r1)=$OLD_PICK=$(dims "$OLD_PICK")=soup-v9b lineage")
[ -n "$NEW_PICK" ] && [ -n "$(lbv "$NEW_PICK" heldout8)" ] && CANDS+=("new lineage, round 2 on $ROWS (a1)=$NEW_PICK=$(dims "$NEW_PICK")=$RAW lineage")
CANDS+=("old lineage, stage 3 on n5 + kj (v45)=$REF_PICK=$(dims "$REF_PICK")=soup-v9b lineage")
"$PY" scripts/freeze_choice.py --regimes "$REGIMES_ROOT/quantized" --cost "$COST" --out "$FREEZE" --max-dims "$MAX_DIMS" --max-postings "$MAX_POSTINGS" --margin 0 --metric sets2 \
    "${CANDS[@]}" 2>&1 | while read -r l; do log "  $l"; done
WIN=$(jget "$FREEZE" winner)
[ -n "$WIN" ] && [ -d "models/$WIN" ] || { log "  no choice ($FREEZE): a1 stops without an ALBERT label"; exit 1; }
{ echo "$WIN"; echo "# first line = the label of the ALBERT to release (models/<label>); written by queue a1 on $(date '+%d.%m %H:%M') by its rule: $(jget "$FREEZE" why)"; echo "# overwrite the first line to choose another model before queue x1 runs again"; } > "$FINAL_ALBERT_FILE"
log "  ALBERT to release: $WIN -> $FINAL_ALBERT_FILE"
for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do h2task "$WIN" "$ts"; log "  $WIN $ts $(lbv "$WIN" "$ts")"; done
asym() {  # asym LABEL: LABEL as the query encoder over the index of the frozen full model
    [ -n "$1" ] && [ -d "models/$1" ] || return 0
    PIRB_QUERY_MODEL="models/$1" regime quantized "asym-$FTAG-q-$1" "models/$FULL" tiny10 16
    PIRB_QUERY_MODEL="models/$1" regime quantized "asym-$FTAG-q-$1" "models/$FULL" heldout8 16
    log "  $1 as the query encoder over the index of $FULL: $(lbs "asym-$FTAG-q-$1") (symmetric: ALBERT $(lbs "$1"), full model $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8))"
}
asym "$WIN"
LOSER="$OLD_PICK"; [ "$WIN" = "$OLD_PICK" ] && LOSER="$NEW_PICK"
asym "$LOSER"
pack_add "albert/$WIN" "models/$WIN" "ALBERT student (50 M), release form with the bias baked in; $(lbs "$WIN") / trec-covid $(tc "$WIN"), $(dims "$WIN") dims/doc; as the query encoder over $FULL: $(lbs "asym-$FTAG-q-$WIN")"

# 7. summary
log "=== a1 step 7: summary ==="
SUM="$RESULTS_ROOT/summary_a1.md"
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
[ -n "$NEW_PICK" ] && COLS+=(--column "new lineage, round 2 (a1)=$NEW_PICK")
{
    echo "# a1 (quantized regime, nDCG@10): the ALBERT student descending from the frozen full model"; echo
    echo "The old lineage: student \`$OLD_ST\` cut from soup-v9b -> KD from soup-v9b on kd_v1 -> stage 3 on n5 + kj (\`$REF\`) -> round 2 in r1. The new lineage repeats it with one change: the source model and the KD teacher are the raw weights of the frozen full model (\`$TEACHER\`; the frozen form \`$FULL\` is the same model with the bias shift). Student: layers 0,1,2,5, FFN 1536, the teacher's head transform as an extra layer, rank-256 vocabulary weighted by the kd_v1 token counts. KD: token KL x 0.3 + pooled MSE on all of kd_v1, one epoch, lr 5e-5, linear, batch 16, seed 42. Stage 3 and round 2: the configs of the old lineage with the start model replaced; the ranking teacher is unchanged (the ensemble scores stored in the rows). Round 2 ran on the rows of the r1 arm that won tiny10: **$ROWS** (r1: dose3 ${TA:--}, dose3 + faq20 ${TB:--}). Every model is measured in its first bias form with at most $MAX_DIMS dims/doc (steps of 0.05)."; echo
    echo "## Choice"; echo
    echo "Rule: the highest mean of tiny10 and heldout8 with at most $MAX_DIMS dims/doc and at most $MAX_POSTINGS postings/query, no margin; the new model wins when it is above the best arm of r1 (the stage-3 model of the old lineage stays a candidate)."; echo
    echo "**ALBERT to release: \`$WIN\`** — $(jget "$FREEZE" why). The label is the first line of \`logs/final_albert_model.txt\`; the model is in \`$PACK/albert/$WIN\`."; echo
    echo "## Old lineage against the new one"; echo
    echo "asym = the model as the query encoder over the index of the frozen full model (\`$FULL\`, symmetric $(mlbv "$FULL" tiny10) / $(mlbv "$FULL" heldout8)); measured for the two round-2 models. Stage 3 of the new lineage and the other arm of r1 have tiny10 only."; echo
    echo "| model | label | tiny10 | heldout8 | trec-covid | asym tiny10 | asym heldout8 | dims/doc | cost: dims/query | cost: postings/query |"
    echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    row "old lineage, stage 3 on n5 + kj (v45)" "$REF_PICK"
    row "old lineage, round 2 on $ROWS (r1)" "$OLD_PICK"
    row "old lineage, round 2, the other arm of r1" "$OTHER_PICK"
    row "new lineage, stage 3 on n5 + kj (a1)" "$S3_PICK"
    row "new lineage, round 2 on $ROWS (a1)" "$NEW_PICK"
    echo "| frozen full model (82 M), for scale | \`$FULL\` | $(mlbv "$FULL" tiny10) | $(mlbv "$FULL" heldout8) | $(mlbv "$FULL" heldout2_treccovid) | - | - | $(sparsity_mean "$MAIN/results/sparsity/$FULL.json" "$FULL" 2>/dev/null) | $(fcget "$FULL" query_active_mean %.1f) | $(fcget "$FULL" postings_per_query %.0f) |"
    echo
    echo "Unbiased dims/doc: old stage 3 $(dims "$REF"), old round 2 $(dims "${OLD_PICK%-bias[0-9][0-9][0-9]}"), new stage 3 $(dims "$S3"), new round 2 $(dims "$NEW"). One seed each; the seed noise of the ranking recipe is 0.13-0.2."; echo
    echo "The change of the source model and KD teacher alone (stage 3 against stage 3, tiny10): $([ -n "$(lbv "$S3_PICK" tiny10)" ] && fsub "$(lbv "$S3_PICK" tiny10)" "$(lbv "$REF_PICK" tiny10)" || echo -). After round 2 (new against the best arm of r1): tiny10 $([ -n "$(lbv "$NEW_PICK" tiny10)" ] && fsub "$(lbv "$NEW_PICK" tiny10)" "$(lbv "$OLD_PICK" tiny10)" || echo -), heldout8 $([ -n "$(lbv "$NEW_PICK" heldout8)" ] && [ -n "$(lbv "$OLD_PICK" heldout8)" ] && fsub "$(lbv "$NEW_PICK" heldout8)" "$(lbv "$OLD_PICK" heldout8)" || echo -)."; echo
    echo "## Fidelity to the own teacher (old: soup-v9b; new: $RAW)"; echo
    echo "| student | texts | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|"
    fidrow "old, before KD, corpus texts" "$RESULTS_ROOT/fidelity/$OLD_ST.cuda-fp16-fp32head.json" all
    fidrow "new, before KD, corpus texts" "$F0" all
    fidrow "old, before KD, queries" "$RESULTS_ROOT/fidelity/student-$OLD_ST.queries.json" query
    fidrow "new, before KD, queries" "$RESULTS_ROOT/fidelity/student-$ST.queries.json" query
    fidrow "old, after KD, queries" "$RESULTS_ROOT/fidelity/$OLD_KD.queries.json" query
    fidrow "new, after KD, queries" "$RESULTS_ROOT/fidelity/$KD.queries.json" query
    echo; echo "Fidelity curve of the new KD (256 corpus texts per checkpoint):"; echo; echo '```'
    [ -f "runs/$KD/log.jsonl" ] && "$PY" scripts/kd_fidelity_curve.py "runs/$KD/log.jsonl" 2>> "$LOGS/queue_summary.log"
    echo '```'; echo
    if [ -n "$NEW_PICK" ] && [ -n "$(lbv "$NEW_PICK" heldout8)" ] && [ -n "$(lbv "$OLD_PICK" heldout8)" ]; then
        "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" \
            "${COLS[@]}" --pick "$NEW_PICK" --ref "$OLD_PICK" \
            --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
            --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
            --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
            --win "THE NEW LINEAGE WINS (>= +0.2 on both sets over the best arm of r1 at <= $MAX_DIMS dims/doc)" \
            --noise "within the seed noise of the best arm of r1 set by set (the choice itself follows the rule above, on the mean of the two sets)" \
            --loss "a LOSS on both sets against the best arm of r1" --m23 "$WIN" --json "$VERDICT" 2>> "$LOGS/queue_summary.log"
    fi
    echo; echo "Nothing further was started by this queue except queue x1 for the ONNX export of the chosen ALBERT."
    echo; echo "## Queue state (a1)"; echo; echo '```'; grep -E "$ST|hl-dqf-full|hl-full-s3n5kj-r2d3|asym-$FTAG-q-arm-hl" "$STATE" | tail -n 70; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_a1.sh" "$C/stage3_hl_dqf_full_n5kj_ens400.yaml" "$C/stage3_hl_dqf_full_r2_dose3.yaml" "$C/stage3_hl_dqf_full_r2_dose3_faq20.yaml" "$FREEZE" 2>/dev/null
for f in "$COST" "$VERDICT" "$F0" "$RESULTS_ROOT/fidelity/student-$ST.queries.json" "$RESULTS_ROOT/fidelity/$KD.queries.json"; do [ -e "$f" ] && git add "$f" 2>/dev/null; done
for m in $REF_PICK $OLD_PICK $OTHER_PICK $S3 $S3_PICK $NEW $NEW_PICK "asym-$FTAG-q-$WIN" "asym-$FTAG-q-$LOSER"; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/arm-hl-dqf-full-s3n5kj*.json 2>/dev/null
git commit -q -m "Add the a1 results: the ALBERT student descending from the frozen full model" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop a1
log "  queue x1 again for the ONNX export of $WIN and the README of the package"
env -u EXP_ROOT -u RESULTS_ROOT -u REGIMES_ROOT -u HOLDOUT_ROOT -u HOLDOUT2_ROOT -u QUEUE_TAG -u TRAIN_TIMEOUT -u KD_TIMEOUT -u WAIT_FOR -u MIN_RAM_GB bash "$MAIN/queues/queue_x1.sh" >> "$LOGS/queue_x1.out" 2>&1
log "  x1 finished: $(tail -n 1 "$LOGS/queue_x1.out" | cut -c1-200)"
log "=== a1 done ($(date '+%d.%m %H:%M')); ALBERT to release $WIN ==="
