# Shared by the stages of queue b2 (source after scripts/queue_lib.sh, arm_common.sh and release_common.sh): the search
# for a 124 M replacement of the frozen full model (17-19.11). Several helpers are copies of queue_b1c.sh (pause_point,
# h8, mkcfg, train_rank, ...), which could not be moved into a shared file while b1c was running.
#
# Bias forms are written by biased2, NOT by biased of arm_common.sh: queue b1c moved a raw model to the data drive
# (to_dd leaves a link under models/) and then called biased, whose `cp -r` copied the LINK; the bias shift went into
# the raw model itself, once per form. Every stage-1 model of R-c' (arm-b1-rc-s1p1..4) carries the shifts of its three
# "forms" (0.05 + 0.10 + 0.15 as thresholds, 0.3183 on the bias in all), its "bias015" numbers are those of that sum,
# and each next segment (and the dose3 round arm-b1-rc-d3) started from such a model. biased2 follows links and refuses
# a copy that is a link.
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
FULL=$(final_full); RAW=${FULL%-bias[0-9][0-9][0-9]}; export TEACHER="models/$RAW"
ALB=$(final_albert)
KD=kd-rbase-v2-dqf-full; KD15=$KD-bias015; RA=arm-b1-ra; RA15=$RA-bias015; RCD=arm-b1-rc-d3
FA=arm-b1-ra-fact20; FQ=arm-b1-ra-faq20; R2=arm-b1-r2-d3; S1=arm-b1-rc-s1p4
MAX_POSTINGS=16807; MAX_DIMS=410; MARGIN=0.2
B2="$RESULTS_ROOT/b2"; FORMS="$RESULTS_ROOT/index_cost_b1_forms"; CANDS="$B2/candidates.tsv"; SUM="$RESULTS_ROOT/summary_b2.md"; PROF="$RESULTS_ROOT/b1_profile.tsv"
DRIVE="$(dirname "$(readlink -f runs)")"; DD="$DRIVE/FlagEmbedding-models-b1"
QREG="$REGIMES_ROOT/quantized"
mkdir -p "$B2" "$FORMS" "$DD" "$LOGS/b1-aside"

pause_point() { local said=0; while [ -e "$LOGS/b1.pause" ]; do [ "$said" = 0 ] && { log "  paused by logs/b1.pause (remove it to continue)"; said=1; }; sleep 60; done; }
pq() { "$PY" -c "import json,sys;print('%.0f' % json.load(open(sys.argv[1]))['models'][sys.argv[2]]['postings_per_query'])" "$1" "$2" 2>/dev/null; }
qd() { "$PY" -c "import json,sys;print('%.1f' % json.load(open(sys.argv[1]))['models'][sys.argv[2]]['query_active_mean'])" "$1" "$2" 2>/dev/null; }
ntasks() { regime_mean "$QREG/$1/heldout8.json" 2>/dev/null | awk '{print $2}'; }
fix8() {
    [ "$(ntasks "$1")" = 7 ] || return 0
    log "  $1: heldout8 has 7 tasks; fiqa-pl alone"
    regime quantized "$1" "models/$1" heldout8_fiqa 8 && "$PY" scripts/merge_regime_task.py --into "$QREG/$1/heldout8.json" \
        --from "$QREG/$1/heldout8_fiqa.json" --n-tasks 8 | while read -r l; do log "  $l"; done
}
partial_aside() {
    local f="$QREG/$1/heldout8.json" n; [ -f "$f" ] || return 0
    n=$(ntasks "$1"); [ "${n:-0}" -lt 7 ] || return 0
    mv "$f" "$LOGS/b1-aside/heldout8-partial-$1-$(date +%H%M%S).json"; log "  $1: heldout8 with ${n:-0} of 8 tasks moved to logs/b1-aside (not a result)"
}
h8() {
    regime quantized "$1" "models/$1" heldout8 16; fix8 "$1"
    if [ "$(ntasks "$1")" != 8 ]; then partial_aside "$1"; regime quantized "$1" "models/$1" heldout8 8; fix8 "$1"; fi
    [ "$(ntasks "$1")" = 8 ] || { log "  $1: heldout8 is not complete after a repeat"; partial_aside "$1"; return 1; }
}
to_dd() {  # to_dd LABEL: move models/LABEL to the data drive and link it back (space on /). Copies of a moved model: only with cp -rL (biased2, unshift).
    [ -d "models/$1" ] && [ ! -L "models/$1" ] || return 0
    [ -e "$DD/$1" ] && { log "  $DD/$1 exists already: models/$1 stays on /"; return 0; }
    mv "models/$1" "$DD/$1" && ln -s "$DD/$1" "models/$1" || log "  moving models/$1 to the data drive FAILED"
}
biased2() {  # biased2 MODEL OUT THRESHOLD: models/OUT = a real copy of models/MODEL (links followed) with the MLM head bias (and its decoder copy) shifted by -(e^THRESHOLD - 1)
    local src="models/$1" out="$2" thr="$3"
    [ -d "models/$out" ] && return 0
    [ -d "$src" ] || { log "$out: $src missing"; return 1; }
    need_space || return 1
    if cp -rL "$src" "models/$out" && [ ! -L "models/$out" ] && "$PY" - "models/$out" "$thr" <<'EOF'
import math, sys
from safetensors.torch import load_file, save_file
d = sys.argv[1] + "/model.safetensors"; thr = float(sys.argv[2])
t = load_file(d); b = math.exp(thr) - 1.0; n = 0
for base in ("lm_head.bias", "predictions.bias"):
    if base in t:
        shifted = t[base] - b
        t[base] = shifted; t[base.replace(".bias", ".decoder.bias")] = shifted.clone(); n += 1
        print(base, "and its decoder copy shifted by", -b)
assert n, "no MLM head bias tensor found: " + ", ".join(k for k in t if "bias" in k and ("head" in k or "pred" in k or "decoder" in k))
save_file(t, d, metadata={"format": "pt"})
EOF
    then log "$out written (lm_head.bias = -(e^$thr - 1); a real copy of $src)"
    else
        log "$out FAILED"
        if [ -L "models/$out" ]; then rm -f "models/$out"; elif [ -d "models/$out" ]; then rm -rf "models/$out"; fi    # only the partial copy just made
        return 1
    fi
}
m11() { "$PY" scripts/b2_candidates.py mean --regimes "$QREG" "$1" m11 2>/dev/null; }
m19() { "$PY" scripts/b2_candidates.py mean --regimes "$QREG" "$1" m19 2>/dev/null; }
m23() { "$PY" scripts/b2_candidates.py mean --regimes "$QREG" "$1" m23 2>/dev/null; }
form_of() { echo "$1${2:+-bias$2}"; }
FINE=("" 005 010 011 012 013 014 015 016 017 018 019 020 022 025 030)
ffit() {  # ffit MODEL INDEX: build (when needed) and measure form FINE[INDEX]; true when it has at most MAX_DIMS dims/doc
    local t="${FINE[$2]}" f; f=$(form_of "$1" "$t")
    if [ -n "$t" ]; then biased2 "$1" "$f" "0.${t#0}" || return 1; to_dd "$f"; fi
    sparsity "$f" "models/$f" || return 1
    [ -n "$(dims "$f")" ] && [ "$(gte "$MAX_DIMS" "$(dims "$f")")" = 1 ]
}
form_fit2() {  # form_fit2 MODEL: PICK = the first form of FINE (no shift, 0.05, 0.10, then steps of 0.01 as for the 82 M soups) with <= MAX_DIMS dims/doc
               # and <= MAX_POSTINGS postings/query. Forms between 0.10 and the first fit are not all built: the search starts at the
               # threshold a straight line through the no-shift and the 0.10 form predicts, steps up until a form fits and then down
               # while the form below fits too (dims/doc fall with the threshold), so the result is the first fit of the list.
    local m="$1" n=${#FINE[@]} i f p; PICK=""
    [ -d "models/$m" ] || { log "  $m missing: no form"; return 1; }
    if ffit "$m" 0; then i=0
    elif ffit "$m" 2; then if ffit "$m" 1; then i=1; else i=2; fi
    else
        i=$("$PY" -c "
import math, sys
d0, d10, cap = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); fine = [int(x) for x in sys.argv[4:]]
t = math.ceil(10.0 * (d0 - cap) / max(d0 - d10, 1e-6))
print(next((k for k, v in enumerate(fine) if k >= 3 and v >= t), len(fine) - 1))" "$(dims "$m")" "$(dims "$m-bias010")" "$MAX_DIMS" 0 "${FINE[@]:1}" 2>/dev/null)
        [ -n "$i" ] || i=3
        log "  $m: $(dims "$m") dims/doc without a shift, $(dims "$m-bias010") at 0.10; the search starts at bias${FINE[$i]}"
        while ! ffit "$m" "$i"; do i=$((i + 1)); [ "$i" -lt "$n" ] || { log "  $m: no form with at most $MAX_DIMS dims/doc up to bias${FINE[$((n - 1))]}"; return 1; }; done
        while [ "$i" -gt 3 ] && ffit "$m" $((i - 1)); do i=$((i - 1)); done
    fi
    while :; do
        f=$(form_of "$m" "${FINE[$i]}")
        index_cost "$FORMS/$f.json" "$f"; p=$(pq "$FORMS/$f.json" "$f")
        log "  $f: $(dims "$f") dims/doc, $(qd "$FORMS/$f.json" "$f") dims/query, ${p:-?} postings/query (limits $MAX_DIMS and $MAX_POSTINGS)"
        if [ -n "$p" ] && [ "$(gte "$MAX_POSTINGS" "$p")" = 1 ]; then PICK="$f"; return 0; fi
        i=$((i + 1)); [ "$i" -lt "$n" ] || break
        ffit "$m" "$i" || true
    done
    log "  $m: no form within both limits up to bias${FINE[$((n - 1))]}"; return 1
}
measure11() {  # measure11 LABEL: tiny10 and trec-covid (11 tasks)
    [ -n "$1" ] && [ -d "models/$1" ] || return 1
    regime quantized "$1" "models/$1" tiny10 16
    regime quantized "$1" "models/$1" heldout2_treccovid 8
    log "  $1: tiny10 $(lbv "$1" tiny10), trec-covid $(tc "$1"), mean of the 11 tasks $(m11 "$1") (frozen $(m11 "$FULL"), R-a bias015 $(m11 "$RA15")) at $(dims "$1") dims/doc, $(pq "$FORMS/$1.json" "$1") postings/query"
}
measure19() {  # measure19 LABEL: tiny10, trec-covid, heldout8 (complete): the 19 tasks of the rule
    measure11 "$1" || return 1
    h8 "$1"
    log "  $1: $(lbs "$1") / trec-covid $(tc "$1"), mean of the 19 tasks $(m19 "$1") (frozen $(m19 "$FULL"); the rule asks for a lead of $MARGIN)"
}
cand() {  # cand STAGE NAME LABEL RAW DESCRIPTION: register a candidate once (by its label)
    [ -n "$3" ] || return 0
    cut -f3 "$CANDS" 2>/dev/null | grep -q -x -F -- "$3" || printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$CANDS"
}
cargs() { echo --cands "$CANDS" --regimes "$QREG" --sparsity "$RESULTS_ROOT/sparsity" --cost-dir "$FORMS" --cost "$RESULTS_ROOT/index_cost_b1.json" --max-dims "$MAX_DIMS" --max-postings "$MAX_POSTINGS"; }
cand_table() { "$PY" scripts/b2_candidates.py table $(cargs) --ref "$FULL" --margin "$MARGIN" "$@" 2>> "$LOGS/queue_summary.log"; }
best_cand() { "$PY" scripts/b2_candidates.py best $(cargs) "$@" 2>> "$LOGS/queue_summary.log"; }     # prints LABEL<TAB>RAW<TAB>NAME<TAB>SCORE
base_cands() {
    cand ref "frozen full model (82 M)" "$FULL" "$RAW" "dose3 + faq20 + fact20, soup of the 82 M arms"
    cand b1 "KD alone (124 M)" "$KD15" "$KD" "roberta-base-v2 after KD from the frozen model"
    cand b1 "R-a, bias015" "$RA15" "$RA" "KD model + the dose3 round"
    local rc; rc=$(measured_form "$RCD"); [ -n "$rc" ] && cand b1 "R-c' after the dose3 round" "$rc" "$RCD" "KD model + 200 000 MS MARCO-pl rows (bias shifted at every segment, see the erratum) + the dose3 round"
}
pick_lineage() {  # The start of the data arms (the user, 17.11: "start from the better checkpoint"), decided once and kept in $B2/arm_lineage.txt
                  # (write kd or rc there to overrule it before stage p2 starts):
                  #   rc  when R-c' after the dose3 round leads every measured form of R-a on the 19 tasks within the limits: the arms start
                  #       where R-c' started (models/arm-b1-rc-s1p4 as it is on the disk, i.e. the end of stage 1 with the bias shift of b1c,
                  #       the start that is proven by R-c'), so R-c', fact20 and faq20 are arms from one start, as the 82 M arms of the frozen soup
                  #   kd  otherwise: the arms start from the KD model, as R-a did
                  # Sets LINEAGE, START, FA, FQ, ANCHOR (the dose3 arm of the lineage), AN (its name), AS (its short name in soup names).
    local f="$B2/arm_lineage.txt" rc ra
    if [ ! -s "$f" ]; then
        rc=$(best_cand --stages b1 --metric m19 --name-prefix "$RCD-bias" | cut -f4); ra=$(best_cand --stages b1,p1 --metric m19 --name-prefix "$RA-bias" | cut -f4)
        if [ -n "$rc" ] && [ -f "models/$S1/model.safetensors" ] && [ -d "models/$RCD" ] && { [ -z "$ra" ] || [ "$(gte "$rc" "$ra" 0.0001)" = 1 ]; }; then echo rc > "$f"; else echo kd > "$f"; fi
        log "  start of the data arms: $(cat "$f") (R-c' ${rc:-not measured} against the best form of R-a ${ra:-not measured} on the 19 tasks)"
    fi
    LINEAGE=$(first_line "$f")
    if [ "$LINEAGE" = rc ]; then START=$S1; FA=arm-b1-rc-fact20; FQ=arm-b1-rc-faq20; ANCHOR=$RCD; AN="R-c'"; AS=rcd
    else LINEAGE=kd; START=$KD; FA=arm-b1-ra-fact20; FQ=arm-b1-ra-faq20; ANCHOR=$RA; AN="R-a"; AS=ra; fi
}
arm_cfg() {  # arm_cfg ROWS_NAME: echo the config of the data arm of the chosen lineage (b2_ra_NAME.yaml for kd; b2_rc_NAME.yaml, derived from it, for rc)
    local base="$C/b2_ra_$1.yaml" out="$C/b2_rc_$1.yaml"
    [ "$LINEAGE" = rc ] || { echo "$base"; return 0; }
    "$PY" - "$base" "$out" "$1" "$START" <<'PYEOF' || return 1
import re, sys
base, out, name, start = sys.argv[1:5]
s = open(base, encoding="utf-8").read()
def sub(pattern, repl):
    global s
    s, n = re.subn(pattern, repl, s, flags=re.M)
    assert n == 1, pattern
sub(r"^runs_dir: runs/arm-b1-ra-%s$" % name, "runs_dir: runs/arm-b1-rc-%s" % name)
sub(r"^  - models/kd-rbase-v2-dqf-full$", "  - models/%s" % start)
s = s.replace("backend: sentence-transformers\n", "backend: sentence-transformers\n# derived by queue b2 from b2_ra_%s.yaml: the arm starts where R-c' started (models/%s, the end of the MS MARCO-pl stage as it is on the disk,\n"
              "# with the bias shift of queue b1c), not from the KD model; rows and recipe unchanged\n" % (name, start), 1)
open(out, "w", encoding="utf-8").write(s)
PYEOF
    echo "$out"
}
mkcfg() {  # mkcfg BASE_YAML MODE: echo the config to use (BASE for plain, else a derived copy next to it)
    local base="$1" mode="$2" out
    out="${base%.yaml}"; [ "$mode" != plain ] && out="${out}_$mode"; out="$out.yaml"
    [ "$out" = "$base" ] && { echo "$base"; return 0; }
    "$PY" - "$base" "$out" "$mode" <<'PYEOF' || return 1
import re, sys
base, out, mode = sys.argv[1:4]
s = open(base, encoding="utf-8").read()
def sub(pattern, repl):
    global s
    s, n = re.subn(pattern, repl, s, flags=re.M)
    assert n == 1, pattern
if mode in ("gc", "b1gc"):
    sub(r"^  gradient_checkpointing: false$", "  gradient_checkpointing: true")
if mode == "b1gc":
    sub(r"^  train_batch_size: 2$", "  train_batch_size: 1")
    sub(r"^  gradient_accumulation_steps: 16$", "  gradient_accumulation_steps: 32")
note = {"gc": "gradient checkpointing on", "b1gc": "physical batch 1 x 32 accumulation and gradient checkpointing (the recipe did not fit 8 GB)"}[mode]
s = s.replace("backend: sentence-transformers\n", "backend: sentence-transformers\n# derived by the b1 queues from %s: %s\n" % (base.split("/")[-1], note), 1)
open(out, "w", encoding="utf-8").write(s)
PYEOF
    echo "$out"
}
prof_get() { awk -F'\t' -v k="$1" '$1==k {v=$2} END {print v}' "$PROF" 2>/dev/null; }
RMODE=$(prof_get rank_mode); [ -s "$LOGS/b1_rank_mode.txt" ] && RMODE=$(first_line "$LOGS/b1_rank_mode.txt"); RMODE=${RMODE:-gc}
train_rank() {  # train_rank NAME BASE_YAML: train in mode $RMODE; after an out-of-memory failure go down the ladder (plain, gc, b1gc)
    local tag="$1" base="$2" cfg m
    for m in plain gc b1gc; do
        case "$RMODE:$m" in gc:plain|b1gc:plain|b1gc:gc) continue;; esac
        cfg=$(mkcfg "$base" "$m") || { log "  config of arm-$tag ($m) FAILED"; return 1; }
        train "$tag" "$cfg"
        if [ -d "models/arm-$tag" ]; then
            [ "$m" != "$RMODE" ] && { RMODE=$m; log "  ranking mode is now $m (arm-$tag ran out of memory in the mode before)"; }
            return 0
        fi
        grep -q -i "out of memory" "$LOGS/arm_$tag.log" 2>/dev/null || return 1
        log "  arm-$tag ran out of memory in mode $m"; cp "$LOGS/arm_$tag.log" "$LOGS/b1-aside/arm_$tag.$m.oom.log"
    done
    return 1
}
tsec() { awk -F'\t' -v k="train:$1" '$1==k && $2=="ok" {v=$4} END {print v}' "$STATE" | tr -dc '0-9'; }
mksoup() {  # mksoup NAME WEIGHTS MODEL...: models/NAME = the weighted average of the raw weights (WEIGHTS: "uniform" or comma-separated, summing to 1)
    local name="$1" w="$2" m dirs=(); shift 2
    [ -d "models/$name" ] && return 0
    for m in "$@"; do [ -d "models/$m" ] || { log "soup $name: models/$m missing; skipped"; return 1; }; dirs+=("models/$m"); done
    need_space || return 1; wait_gpu_free; wait_ram
    local wargs=(); [ "$w" != uniform ] && wargs=(--weights ${w//,/ })
    if ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1800 "$PY" scripts/merge_splade_weights.py --models "${dirs[@]}" "${wargs[@]}" --output "models/$name" > "$LOGS/soup_$name.log" 2>&1; then
        log "soup $name written ($* with weights $w)"
    else
        log "soup $name FAILED (logs/soup_$name.log): $(tail -n 1 "$LOGS/soup_$name.log" | cut -c1-240)"
        [ -d "models/$name" ] && [ ! -L "models/$name" ] && rm -rf "models/$name"      # only the partial output just made
        return 1
    fi
}
soup_cand() {  # soup_cand STAGE NAME WEIGHTS DESCRIPTION MEASURE MODEL...: soup, data drive, first form within the limits, MEASURE (measure11 | measure19), candidate line; SOUP_PICK
    local stage="$1" name="$2" w="$3" desc="$4" how="$5"; shift 5; SOUP_PICK=""
    mksoup "$name" "$w" "$@" || return 1
    to_dd "$name"
    form_fit2 "$name" || return 1
    SOUP_PICK="$PICK"
    "$how" "$SOUP_PICK"
    cand "$stage" "$desc" "$SOUP_PICK" "$name" "$desc"
}
arm_cand() {  # arm_cand STAGE ARM DESCRIPTION: a trained arm on the data drive, first form within the limits, the 19 tasks, candidate line; ARM_PICK
    ARM_PICK=""
    [ -d "models/$2" ] || { log "  $2 was not trained"; return 1; }
    to_dd "$2"
    form_fit2 "$2" || return 1
    ARM_PICK="$PICK"
    measure19 "$ARM_PICK"
    cand "$1" "$3" "$ARM_PICK" "$2" "$3"
}
write_summary() {  # write_summary COMMIT_MESSAGE: summary_b2.md from the candidates and the notes of the stages; commit
    local s
    {
        echo "# b2 (quantized regime, nDCG@10): a 124 M replacement of the frozen full model"; echo
        echo "The release of 26.11 has frozen versions; a stronger 124 M model (sdadas/polish-roberta-base-v2 after KD from the frozen model, \`summary_b1.md\`) found by Thursday 19.11 would replace the full model. Rule of the replacement: the highest mean over the 19 tasks of tiny10, heldout8 and trec-covid (every task with the same weight) with at most $MAX_DIMS dims/doc and at most $MAX_POSTINGS postings/query, and a lead of at least $MARGIN over \`$FULL\` ($(m19 "$FULL")): the architecture changes and the encoding is about twice as slow. The winner then needs its 23 tasks not below the frozen model's ($(m23 "$FULL")). The decision is the user's. Every candidate is measured in its first form within both limits (no shift, 0.05, 0.10, then steps of 0.01). One seed each; the seed noise of the ranking recipe is 0.13-0.2 on tiny10."; echo
        echo "## Candidates"; echo
        cand_table
        echo; echo "Bold = within the limits and at least $MARGIN above the frozen model on the 19 tasks. A dash in the 19-task column = no heldout8 yet (the candidate was read on tiny10 and trec-covid only)."; echo
        for s in p0 p1 p1b p2 p3 p4 p5; do [ -s "$B2/notes_$s.md" ] && { cat "$B2/notes_$s.md"; echo; }; done
        echo "## Queue state (b2)"; echo; echo '```'; grep -a -E "b1-ra-fact20|b1-ra-faq20|b1-rc-fact20|b1-rc-faq20|b1-r2-d3|soup-b1-|wise-|arm-b1-ra-bias01[1-4]|unshifted|fidelity:b2-|$KD15:heldout2" "$STATE" | tail -n 150; echo '```'
    } > "$SUM"
    log "summary written to $SUM"
    git add "$SUM" "$B2" "$EXP_ROOT/queues/queue_b2.sh" "$EXP_ROOT"/queues/queue_b2_p*.sh "$EXP_ROOT/queues/b2_common.sh" "$EXP_ROOT/queues/release_common.sh" scripts/b2_candidates.py "$C"/b2_ra_fact20*.yaml "$C"/b2_ra_faq20*.yaml 2>/dev/null
    ls "$C"/b2_rc_*.yaml > /dev/null 2>&1 && git add "$C"/b2_rc_*.yaml 2>/dev/null
    ls "$C"/b2_r2_d3*.yaml > /dev/null 2>&1 && git add "$C"/b2_r2_d3*.yaml 2>/dev/null
    local l
    for l in $(cut -f3 "$CANDS" 2>/dev/null) ${EXTRA_LABELS:-}; do
        [ -e "$QREG/$l" ] && git add "$QREG/$l" 2>/dev/null
        [ -e "$RESULTS_ROOT/sparsity/$l.json" ] && git add "$RESULTS_ROOT/sparsity/$l.json" 2>/dev/null
        [ -e "$FORMS/$l.json" ] && git add "$FORMS/$l.json" 2>/dev/null
    done
    git add "$RESULTS_ROOT"/sparsity/arm-b1-ra-*.json "$RESULTS_ROOT"/sparsity/arm-b1-rc-f*.json "$RESULTS_ROOT"/sparsity/soup-b1-*.json "$RESULTS_ROOT"/sparsity/wise-*.json "$RESULTS_ROOT"/sparsity/arm-b1-r2-*.json "$RESULTS_ROOT"/sparsity/arm-b1-rc-*unshifted*.json "$RESULTS_ROOT"/fidelity/b2-*.json 2>/dev/null
    git commit -q -m "$1" 2>/dev/null && log "committed $(git log --oneline -1)"
    push_loop "$QUEUE_TAG"
}
