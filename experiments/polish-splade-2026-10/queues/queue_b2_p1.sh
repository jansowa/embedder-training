#!/usr/bin/env bash
# Queue b2, stage p1 (about 4.5 h, no training): cheap diagnosis and cheap repairs of the trec-covid loss of R-a
# (67.39 against 72.48 of the frozen model, at tiny10 65.34 against 65.06).
#   1  the KD model alone: trec-covid of kd-rbase-v2-dqf-full-bias015; fidelity of the KD student to its teacher (the raw
#      frozen model) on texts of trec-covid-pl (512 passages, all queries) next to kd_v1, nfcorpus-pl and scifact-pl,
#      at 512 tokens; the same for the two measured forms (R-a bias015 against the frozen model) on the passages, which
#      shows the document density of the two side by side. Benchmark texts are read for measurement only.
#   2  a denser form of R-a: bias011-bias014 are built (density of each); the first with <= 410 dims/doc and <= 16 807
#      postings/query gets tiny10 and trec-covid
#   3  only when the KD model alone has trec-covid >= 70: weight interpolation a * R-a + (1 - a) * KD for a = 0.7 and 0.5
#      (the two share the start, as in WiSE-FT), first form within the limits, tiny10 and trec-covid
#   4  only when R-c' after the dose3 round is at most 0.2 below R-a on tiny10 and on heldout8: the soup R-a + R-c',
#      tiny10 and trec-covid
#   5  heldout8 for the denser form of R-a (it decides, against R-c', where the data arms of p2 / p3 start), for the best
#      candidate of this stage on the 11 tasks (tiny10 + trec-covid) and for every candidate not below the frozen model on them
# Table of the candidates in summary_b2.md; commit. Run by queue_b2.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b2p1 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
guard_alive
TX="runs/b2-texts"; mkdir -p "$TX" "$RESULTS_ROOT/fidelity"
for m in "$KD" "$KD15" "$RA" "$RA15" "$RAW" "$FULL"; do [ -d "models/$m" ] || { log "  models/$m missing: p1 stops"; exit 1; }; done
log "=== b2 p1 start: frozen $FULL $(lbs "$FULL") / $(tc "$FULL") (19 tasks $(m19 "$FULL")); R-a $RA15 $(lbs "$RA15") / $(tc "$RA15") (19 tasks $(m19 "$RA15")); $(free_gb) GB free on / ==="
[ -d "models/$RCD" ] && to_dd "$RCD"
base_cands

# ---------------------------------------------------------------- 1. the KD model alone
log "=== b2 p1 step 1: trec-covid and fidelity of the KD model ==="
regime quantized "$KD15" "models/$KD15" heldout2_treccovid 8
KD_TC=$(tc "$KD15")
log "  KD alone ($KD15): tiny10 $(lbv "$KD15" tiny10), trec-covid ${KD_TC:-not measured} (R-a $(tc "$RA15"), frozen $(tc "$FULL"))"
pause_point
if [ ! -s "$TX/trec-covid-pl.jsonl" ] || [ ! -s "$TX/nfcorpus-pl.jsonl" ] || [ ! -s "$TX/scifact-pl.jsonl" ]; then
    "$PY" - third_party/pirb/data "$TX" trec-covid-pl nfcorpus-pl scifact-pl > "$LOGS/b2_p1_texts.log" 2>&1 <<'PYEOF'
import json, random, sys
from pathlib import Path
data, out = Path(sys.argv[1]), Path(sys.argv[2])
def texts(path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                t = r.get("contents") or r.get("text") or r.get("query")
                if t:
                    rows.append(str(t))
    return rows
for task in sys.argv[3:]:
    passages = texts(data / task / "passages" / "passages.jsonl"); queries = texts(data / task / "queries" / "queries.jsonl")
    random.Random(42).shuffle(passages)
    with (out / f"{task}.jsonl").open("w", encoding="utf-8") as fh:
        for t in queries:
            fh.write(json.dumps({"text": t, "kind": "query", "source": task}, ensure_ascii=False) + "\n")
        for t in passages[:512]:
            fh.write(json.dumps({"text": t, "kind": "passage", "source": task}, ensure_ascii=False) + "\n")
    print(task, len(queries), "queries,", min(512, len(passages)), "of", len(passages), "passages")
PYEOF
    cut -c1-200 "$LOGS/b2_p1_texts.log" | while read -r l; do log "  texts: $l"; done
fi
fset() { case "$1" in kd_v1) echo datasets/kd/kd_v1.jsonl;; *) echo "$TX/$1.jsonl";; esac; }
for s in kd_v1 trec-covid-pl nfcorpus-pl scifact-pl; do
    [ -s "$(fset "$s")" ] || { log "  no texts for $s"; continue; }
    for k in passage query; do
        TEACHER="models/$RAW" fidelity "b2-$KD.$s.$k" "models/$KD" "$(fset "$s")" --kinds "$k" --n 512 --max-seq-length 512 --batch-size 8
    done
    TEACHER="models/$FULL" fidelity "b2-$RA15-vs-frozen.$s.passage" "models/$RA15" "$(fset "$s")" --kinds passage --n 512 --max-seq-length 512 --batch-size 8
done
FID=$("$PY" - "$RESULTS_ROOT/fidelity" "$KD" "$RA15" <<'PYEOF' 2>> "$LOGS/queue_summary.log"
import json, sys
from pathlib import Path
root, kd, ra = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sets = ["kd_v1", "trec-covid-pl", "nfcorpus-pl", "scifact-pl"]
def kind(path, k):
    try:
        d = json.loads(path.read_text())["kinds"]
    except Exception:
        return None
    return d.get(k) or d.get("all") or (next(iter(d.values())) if d else None)
def row(name, path, k):
    x = kind(path, k)
    if not x:
        return "| %s | - | - | - | - | - | - | - |" % name, None
    m = lambda n: x[n]["mean"]
    return "| %s | %d | %.4f | %.3f | %.3f | %.3f | %.0f | %.0f |" % (name, x["n"], m("cosine"), m("teacher_active_recall"), m("jaccard_active"), m("l1_relative_error"), m("teacher_active_dims"), m("student_active_dims")), m("cosine")
head = "| texts | n | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |\n|---|---:|---:|---:|---:|---:|---:|---:|"
print("Fidelity of the KD student (`%s`, no shift) to its teacher (the raw frozen model), 512 tokens:\n\n%s" % (kd, head))
cos = {}
for k in ("passage", "query"):
    for s in sets:
        line, c = row("%s, %s" % (s, "passages" if k == "passage" else "queries"), root / ("b2-%s.%s.%s.json" % (kd, s, k)), k)
        print(line); cos[(s, k)] = c
print("\nThe two measured forms on the same passages (`%s` as the student, the frozen model as the teacher), i.e. the document density side by side:\n\n%s" % (ra, head))
for s in sets:
    print(row("%s, passages" % s, root / ("b2-%s-vs-frozen.%s.passage.json" % (ra, s)), "passage")[0])
gaps = []
for k in ("passage", "query"):
    t = cos.get(("trec-covid-pl", k)); others = [cos.get((s, k)) for s in sets if s != "trec-covid-pl"]
    if t is not None and all(o is not None for o in others):
        gaps.append((k, min(others) - t, max(others) - t))
if gaps:
    worst = max(g[1] for g in gaps)
    print("\nCosine on trec-covid-pl against the lowest of kd_v1 and the two biomedical controls: " + "; ".join("%s %+.4f" % (k, -lo) for k, lo, _ in gaps) +
          (". **A fidelity gap on trec-covid (>= 0.03 below every control).**" if worst >= 0.03 else ". No fidelity gap on trec-covid by the bar of 0.03 below every control."))
    print("GAP=%d" % int(worst >= 0.03))
PYEOF
)
FGAP=$(echo "$FID" | sed -n 's/^GAP=//p' | tail -n 1); FID=$(echo "$FID" | grep -v '^GAP=')
log "  fidelity gap on trec-covid (bar 0.03 below every control): ${FGAP:-not measured}"
pause_point

# ---------------------------------------------------------------- 2. a denser form of R-a
log "=== b2 p1 step 2: bias011-bias014 of R-a ==="
RAF=""; DENS="| form | dims/doc |\n|---|---:|\n| \`$RA-bias010\` | $(dims "$RA-bias010") |"
for t in 011 012 013 014; do
    f="$RA-bias$t"
    biased2 "$RA" "$f" "0.${t#0}" || continue
    to_dd "$f"; sparsity "$f" "models/$f"
    DENS="$DENS\n| \`$f\` | $(dims "$f") |"
    if [ -z "$RAF" ] && [ -n "$(dims "$f")" ] && [ "$(gte "$MAX_DIMS" "$(dims "$f")")" = 1 ]; then
        index_cost "$FORMS/$f.json" "$f"; p=$(pq "$FORMS/$f.json" "$f")
        log "  $f: $(dims "$f") dims/doc, $(qd "$FORMS/$f.json" "$f") dims/query, ${p:-?} postings/query (limits $MAX_DIMS and $MAX_POSTINGS)"
        [ -n "$p" ] && [ "$(gte "$MAX_POSTINGS" "$p")" = 1 ] && RAF="$f"
    fi
done
DENS="$DENS\n| \`$RA15\` | $(dims "$RA15") |"
if [ -n "$RAF" ]; then
    measure11 "$RAF"; cand p1 "R-a, denser form (${RAF##*-})" "$RAF" "$RA" "KD model + the dose3 round, the first form within the limits in steps of 0.01"
else log "  no form of R-a between bias011 and bias014 is within both limits"; fi
pause_point

# ---------------------------------------------------------------- 3. interpolation with the KD model
WISE=""
if [ -n "$KD_TC" ] && [ "$(gte "$KD_TC" 70)" = 1 ]; then
    log "=== b2 p1 step 3: the KD model alone has trec-covid $KD_TC (>= 70): interpolation of R-a with it ==="
    for a in 70 50; do
        soup_cand p1 "wise-b1-ra-a$a" "0.$a,0.$((100 - a))" "0.$a x R-a + 0.$((100 - a)) x KD model" measure11 "$RA" "$KD" && WISE="$WISE $SOUP_PICK"
        pause_point
    done
else
    log "=== b2 p1 step 3: the KD model alone has trec-covid ${KD_TC:-not measured} (< 70): no interpolation ==="
fi

# ---------------------------------------------------------------- 4. the soup R-a + R-c'
RC=$(measured_form "$RCD"); GATE=0; RS=""
if [ -n "$RC" ] && [ -d "models/$RCD" ] && [ "$(gte "$(lbv "$RC" tiny10)" "$(lbv "$RA15" tiny10)" -0.2)" = 1 ] && [ "$(gte "$(lbv "$RC" heldout8)" "$(lbv "$RA15" heldout8)" -0.2)" = 1 ]; then GATE=1; fi
echo "$GATE" > "$B2/p14_gate.txt"
if [ "$GATE" = 1 ]; then
    log "=== b2 p1 step 4: R-c' after the dose3 round ($RC $(lbs "$RC")) is within 0.2 of R-a ($(lbs "$RA15")) on both sets: the soup R-a + R-c' ==="
    soup_cand p1 soup-b1-ra-rcd uniform "soup R-a + R-c'" measure11 "$RA" "$RCD" && RS="$SOUP_PICK"
else
    log "=== b2 p1 step 4: R-c' after the dose3 round (${RC:-not measured} $(lbs "${RC:-none}")) is not within 0.2 of R-a ($(lbs "$RA15")) on both sets: no soup ==="
fi
pause_point

# ---------------------------------------------------------------- 5. heldout8 of the contenders
log "=== b2 p1 step 5: heldout8 of the best candidate on the 11 tasks and of every candidate not below the frozen model on them ==="
BEST11=$(best_cand --stages p1 --metric m11 | cut -f1)
F11=$(m11 "$FULL"); H8=""
for l in $(awk -F'\t' '$1=="p1" {print $3}' "$CANDS"); do
    v=$(m11 "$l"); [ -n "$v" ] || continue
    if [ "$l" = "$BEST11" ] || [ "$l" = "$RAF" ] || [ "$(gte "$v" "$F11")" = 1 ]; then h8 "$l"; H8="$H8 $l"; log "  $l: mean of the 19 tasks $(m19 "$l") (frozen $(m19 "$FULL"))"; fi
done

# wise helped = the best interpolation beats every form of R-a itself on the 11 tasks; its weight is kept for stage p2
WA=""
if [ -n "$WISE" ]; then
    BW=$(best_cand --stages p1 --metric m11 --name-prefix wise-b1-ra- | cut -f1,4); BR=$(best_cand --stages b1,p1 --metric m11 --name-prefix "$RA-bias" | cut -f4)
    if [ -n "$BW" ] && [ -n "$BR" ] && [ "$(gte "$(echo "$BW" | cut -f2)" "$BR" 0.05)" = 1 ]; then WA=$(echo "$BW" | cut -f1 | sed -n 's/^wise-b1-ra-a\([0-9]*\).*$/\1/p'); fi
    log "  interpolation: best $(echo "$BW" | tr '\t' ' ') against the best form of R-a $BR on the 11 tasks -> ${WA:+it helped, weight 0.$WA is kept for the soups}${WA:-it did not help}"
fi
echo "$WA" > "$B2/wise_alpha.txt"
# the soup across the two lineages works = R-a + R-c' is not below R-c' alone on the 11 tasks; then R-a may join the soups of an R-c' lineage
XG=0; [ -n "$RS" ] && [ -n "$RC" ] && [ "$(gte "$(m11 "$RS")" "$(m11 "$RC")")" = 1 ] && XG=1
echo "$XG" > "$B2/cross_gate.txt"
[ -n "$RS" ] && log "  soup R-a + R-c' $RS: 11 tasks $(m11 "$RS") against R-c' alone $(m11 "$RC") -> R-a $([ "$XG" = 1 ] && echo "may join" || echo "stays out of") the soups of an R-c' lineage"
pick_lineage

{
    echo "## p1: cheap diagnosis and cheap repairs of trec-covid (no training)"; echo
    echo "**The KD model alone** (\`$KD15\`): tiny10 $(lbv "$KD15" tiny10), trec-covid **${KD_TC:-not measured}** (R-a $(tc "$RA15"), frozen model $(tc "$FULL"))."; echo
    echo "$FID"; echo
    echo "**Density of the forms of R-a** (limit $MAX_DIMS dims/doc):"; echo; echo -e "$DENS"; echo
    echo "Measured: \`${RAF:-none of bias011-bias014 is within both limits}\`$([ -n "$RAF" ] && echo ": tiny10 $(lbv "$RAF" tiny10), trec-covid $(tc "$RAF") (bias015: $(lbv "$RA15" tiny10), $(tc "$RA15"))")."; echo
    if [ -n "$WISE" ]; then echo "**Interpolation with the KD model** (trec-covid of the KD model >= 70):$(for l in $WISE; do echo -n " \`$l\` tiny10 $(lbv "$l" tiny10), trec-covid $(tc "$l");"; done) ${WA:+it beats every form of R-a on the 11 tasks by at least 0.05: weight 0.$WA is used for the best soup of p2}${WA:-it does not beat the forms of R-a on the 11 tasks by 0.05}."
    else echo "**Interpolation with the KD model**: not made (trec-covid of the KD model alone ${KD_TC:-not measured}, the bar was 70)."; fi
    echo
    echo "**Soup R-a + R-c' after the dose3 round**: $([ "$GATE" = 1 ] && echo "made (R-c' \`$RC\` $(lbs "$RC") / trec-covid $(tc "$RC") is within 0.2 of R-a on both sets)" || echo "not made (R-c' \`${RC:-not measured}\` $(lbs "${RC:-none}") against R-a $(lbs "$RA15"); the bar: at most 0.2 below on both sets)"). R-c' started from a stage-1 model whose bias had been shifted in place (see p1b)."; echo
    echo "heldout8 was measured for:${H8:- no candidate of this stage}."; echo
    echo "**Start of the data arms of p2 / p3** (the user, 17.11: start from the better checkpoint): **$LINEAGE** = \`$START\`. Rule: the lineage of R-c' when R-c' after the dose3 round (\`${RC:-not measured}\`, 19 tasks $(m19 "${RC:-none}")) leads every measured form of R-a (bias015 $(m19 "$RA15")$([ -n "$RAF" ] && echo ", ${RAF##*-} $(m19 "$RAF")")) on the 19 tasks; then the arms start where R-c' started (the end of the MS MARCO-pl stage as it is on the disk, with the bias shift of b1c: the start R-c' proves), and R-c', fact20 and faq20 are arms from one start. R-a $([ "$XG" = 1 ] && echo "joins a second soup (the soup R-a + R-c' is not below R-c' alone on the 11 tasks)" || echo "stays out of the soups of that lineage (no soup R-a + R-c', or it is below R-c' alone on the 11 tasks)")."; echo
    echo "New biomedical data: $([ "${FGAP:-0}" = 1 ] && [ -n "$KD_TC" ] && [ "$(gte "$(tc "$FULL")" "$KD_TC" 2)" = 1 ] && echo "**the bar is met** (a fidelity gap on trec-covid and the KD model alone loses trec-covid): a trial (finishing the KD on a mix of texts outside PIRB, then trec-covid alone) is proposed; the decision is the user's" || echo "not proposed (the bar: a fidelity gap of 0.03 on trec-covid AND the KD model alone losing trec-covid)")."
} > "$B2/notes_p1.md"
EXTRA_LABELS="$RA-bias011 $RA-bias012 $RA-bias013 $RA-bias014" write_summary "Add the b2 results: trec-covid of the KD model, a denser form of R-a, cheap repairs without training"
log "=== b2 p1 done ($(date '+%d.%m %H:%M')); KD alone trec-covid ${KD_TC:-?}; R-a form ${RAF:-none} $(lbv "${RAF:-none}" tiny10) / $(tc "${RAF:-none}"); best on the 11 tasks ${BEST11:-none} $(m11 "${BEST11:-none}") (frozen $F11) ==="
