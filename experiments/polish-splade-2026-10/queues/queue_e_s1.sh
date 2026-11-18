#!/usr/bin/env bash
# Queue e, stage s1 = E1 (about 8 h): model A with a teacher that has the 124 M model in it.
#   1  the teacher: the ensemble of the dose3 rows with the scores of the raw R-c' in the place of soup-v9b's
#      (scripts/swap_splade_teacher.py; combining, dense scale 84.19 and weights unchanged); its statistics next to ens400
#   2  the arm: the recipe of the dose3 arm of the frozen model unchanged (from soup-v12, the same rows, 2 epochs), only
#      the teacher scores differ; first form within the limits (grid 0.05), tiny10 and trec-covid
#   3  soups (uniform averages of the raw weights, all from soup-v12 with one recipe):
#        soup-v21-nqf  = new arm + faq20 + fact20        (the new arm in the place of the dose3 arm of the frozen model)
#        soup-v21-dqfn = dose3 + faq20 + fact20 + new arm
#      each tiny10 and trec-covid; the best of the arm and the two soups on those 11 tasks gets heldout8
#   4  gate for A: >= +0.2 over the frozen model on the mean of the 19 tasks within 410 dims/doc and 16 807
#      postings/query. When it passes: the static table mw020 and the released ALBERT as query encoders over the NEW
#      index (tiny10; the table must not lose more than 0.2 against 60.67), and the 23 tasks when more than 7 h are left
#      before the deadline (otherwise stage s3 measures them)
# Run by queue_e.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=es1 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-28800}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
source "$EXP_ROOT/queues/b2_common.sh"
source "$EXP_ROOT/queues/e_common.sh"
guard_alive
CANDS="$EDIR/candidates_a.tsv"
D3=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12; FQ82=arm-dense-kl-n5kj-mined-faq20-ens400-r2v12; FA82=arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12
NA=arm-dense-kl-n5kj-mined-dose3-ensrcd-r2v12
CFG0="$C/distil_dense_kl_n5kj_mined_dose3_ens400_r2v12.yaml"; CFG="$C/distil_dense_kl_n5kj_mined_dose3_ensrcd_r2v12.yaml"
TABLE="models/static-$FULL/phr2-k8-kd-r2-mw020"
for f in "$DROWS" "$TR" "$TV" "$CFG0" models/soup-v12/model.safetensors "models/$D3/model.safetensors" "models/$FQ82/model.safetensors" "models/$FA82/model.safetensors"; do [ -s "$f" ] || { log "  $f missing: s1 stops"; exit 1; }; done
log "=== e s1 start: E1, the dose3 arm of the frozen model with R-c' in the teacher; frozen $FULL $(lbs "$FULL") / $(tc "$FULL") (19 tasks $(m19 "$FULL")); $(left_min) min before the deadline ==="
cand ref "frozen full model A (82 M)" "$FULL" "$RAW" "soup dose3 + faq20 + fact20, teacher ens400"
cand ref "dose3 arm of the frozen model (teacher ens400)" "$(measured_form "$D3")" "$D3" "the arm the new one repeats with another teacher"

# ---------------------------------------------------------------- 1. the teacher
if [ ! -s "$NEWROWS" ] || [ ! -s "$NEWROWS.teacher.json" ]; then
    wait_ram
    timeout -k 60 3600 "$PY" scripts/swap_splade_teacher.py --ens "$DROWS" --old-splade "$TV" --new-splade "$TR" --out "$NEWROWS" --new-name "models/arm-b1-rc-d3 (raw)" \
        --check-old "$A/n5-plus-kj-mined-tv9b.jsonl" > "$LOGS/e_teacher.log" 2>&1 || { log "  building the teacher FAILED (logs/e_teacher.log): $(tail -n 1 "$LOGS/e_teacher.log" | cut -c1-300); s1 stops"; exit 1; }
fi
cp "$NEWROWS.teacher.json" "$EDIR/teacher_ensrcd.json"
log "  teacher: $(tail -n 1 "$LOGS/e_teacher.log" | cut -c1-600)"
[ "$(grep -c . "$NEWROWS")" = "$(grep -c . "$DROWS")" ] || { log "  $NEWROWS has another number of rows than $DROWS: s1 stops"; exit 1; }
pause_point

# ---------------------------------------------------------------- 2. the arm
if [ ! -s "$CFG" ]; then
    "$PY" - "$CFG0" "$CFG" "$NEWROWS" "$NA" <<'PYEOF' || { log "  the config of the arm FAILED: s1 stops"; exit 1; }
import re, sys
base, out, rows, name = sys.argv[1:5]
s = open(base, encoding="utf-8").read()
def sub(pattern, repl):
    global s
    s, n = re.subn(pattern, repl, s, flags=re.M)
    assert n == 1, pattern
sub(r"^runs_dir: runs/arm-dense-kl-n5kj-mined-dose3-ens400-r2v12$", "runs_dir: runs/%s" % name)
sub(r"^train_data: datasets/arms/n5-plus-kj-mined-dose3\.jsonl$", "train_data: %s" % rows)
s = s.replace("backend: sentence-transformers\n", "backend: sentence-transformers\n# Queue e (E1), derived from distil_dense_kl_n5kj_mined_dose3_ens400_r2v12.yaml: the dose arm of the frozen full model with one change, the teacher.\n"
              "# The rows are the same; their scores are the ensemble of the raw 124 M model R-c' (models/arm-b1-rc-d3) and PolDense-400M instead of soup-v9b and PolDense-400M\n"
              "# (scripts/swap_splade_teacher.py; combining, dense scale and weights unchanged).\n", 1)
open(out, "w", encoding="utf-8").write(s)
PYEOF
fi
train "${NA#arm-}" "$CFG"
[ -d "models/$NA" ] || { log "  the arm was not trained (logs/arm_${NA#arm-}.log): s1 stops"; echo "## s1 (E1): model A"$'\n\n'"The arm with the new teacher was not trained (logs/arm_${NA#arm-}.log)." > "$EDIR/notes_s1.md"; esummary "Add the e results: the teacher with R-c' (the arm of E1 failed)"; exit 1; }
pause_point
cand5 e1 "new arm: dose3 rows, teacher R-c' + PolDense" "$NA" "the dose3 arm of the frozen model with R-c' in the teacher"; ARM_P=$C5_PICK
pause_point

# ---------------------------------------------------------------- 3. the soups
mksoup soup-v21-nqf uniform "$NA" "$FQ82" "$FA82" && { to_dd soup-v21-nqf; cand5 e1 "soup new arm + faq20 + fact20" soup-v21-nqf "the frozen soup with the new arm in the place of the dose3 arm"; }
pause_point
mksoup soup-v21-dqfn uniform "$D3" "$FQ82" "$FA82" "$NA" && { to_dd soup-v21-dqfn; cand5 e1 "soup dose3 + faq20 + fact20 + new arm" soup-v21-dqfn "the frozen soup with the new arm added"; }
pause_point
BEST=$(best_cand --stages e1 --metric m11 --any-cost | cut -f1)
[ -n "$BEST" ] && h8 "$BEST"
B19=$(m19 "${BEST:-none}"); F19=$(m19 "$FULL")
WITHIN=$(best_cand --stages e1 --metric m19 | cut -f1)
PASS=0; [ -n "$B19" ] && [ "$WITHIN" = "$BEST" ] && [ "$(gte "$B19" "$F19" "$MARGIN")" = 1 ] && PASS=1
log "  E1: best on the 11 tasks ${BEST:-none} $(lbs "${BEST:-none}") / trec-covid $(tc "${BEST:-none}"), mean of the 19 tasks ${B19:-not measured} against $F19 of the frozen model -> the gate for A $([ "$PASS" = 1 ] && echo PASSES || echo "does not pass")"

# ---------------------------------------------------------------- 4. after the gate
CT=""; AT=""; N23="not measured"; EXTRA=""
if [ "$PASS" = 1 ]; then
    echo "$BEST" > "$EDIR/a_winner.txt"
    PIRB_QUERY_MODEL="$TABLE" regime quantized "asym-$BEST-q-static-phr2-k8-kd-r2-mw020" "models/$BEST" tiny10 16; CT=$(lbv "asym-$BEST-q-static-phr2-k8-kd-r2-mw020" tiny10)
    PIRB_QUERY_MODEL="models/$ALB" regime quantized "asym-$BEST-q-$ALB" "models/$BEST" tiny10 16; AT=$(lbv "asym-$BEST-q-$ALB" tiny10)
    EXTRA="asym-$BEST-q-static-phr2-k8-kd-r2-mw020 asym-$BEST-q-$ALB"
    log "  over the index of $BEST: static table mw020 tiny10 ${CT:-not measured} (over A: 60.67; allowed loss 0.2), released ALBERT ${AT:-not measured} (over A: 64.90)"
    if [ "$(left_min)" -gt 420 ]; then
        for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$BEST" "$ts"; done
        N23="$(m23 "$BEST") against $(m23 "$FULL") of the frozen model"
    else N23="left for stage s3 ($(left_min) min before the deadline at this point)"; fi
fi
{
    echo "## s1 (E1): model A with R-c' in the teacher"; echo
    echo "**Teacher.** The 60 385 rows of the dose arm were scored by the raw R-c' (\`models/arm-b1-rc-d3-bias000\`: the raw weights with both bias keys, so that the fp32 loader of the scoring tool reads the bias) and again by soup-v9b, with the tool and the settings of ens400 (kitchen \`bridge/splade_teacher_scores.py\`, 512 tokens, batch 16, fp16). The rows keep only the ensemble, so the PolDense-400M part of a pair is 2 x ensemble - soup-v9b and the new score is ensemble + (R-c' - soup-v9b) / 2 (\`scripts/swap_splade_teacher.py\`); combining, the dense scale 84.19 and the weights are unchanged. Rows: \`$NEWROWS\`."; echo
    "$PY" - "$EDIR/teacher_ensrcd.json" <<'PYEOF' 2>> "$LOGS/queue_summary.log"
import json, sys
d = json.load(open(sys.argv[1])); m, w = d["members"], d["within_row"]
print("| scores | median margin min(pos) - max(neg) | rows with a negative above a positive | P(pos) at T = 2, median | share above 0.95 |\n|---|---:|---:|---:|---:|")
for key, name in (("old_ensemble", "ens400: soup-v9b + PolDense-400M (the teacher of the frozen model)"), ("new_ensemble", "**R-c' + PolDense-400M (new)**"), ("old_splade", "soup-v9b alone"), ("new_splade", "R-c' alone (raw)"), ("dense", "PolDense-400M alone (x 84.19)")):
    x = m[key]
    print("| %s | %.2f | %d (%.1f %%) | %.3f | %.3f |" % (name, x["median_margin"], x["rows_with_negative_above_positive"], 100 * x["share_with_negative_above_positive"], x["positive_mass_median"], x["positive_mass_above_095"]))
g = lambda k: w.get(k, float("nan"))
print("\nWithin-row agreement (%d rows): new ensemble against ens400 Pearson %.4f, Kendall %.4f; R-c' against soup-v9b Pearson %.4f, Kendall %.4f; against PolDense-400M: R-c' Pearson %.4f, soup-v9b %.4f. Within-row spread of the R-c' scores against soup-v9b's (median ratio of the standard deviations): %.3f, so with unchanged weights R-c' weighs a little %s in the mean than soup-v9b did." % (
    d["rows"], g("pearson_new_ensemble_vs_old_ensemble"), g("kendall_new_ensemble_vs_old_ensemble"), g("pearson_new_splade_vs_old_splade"), g("kendall_new_splade_vs_old_splade"),
    g("pearson_new_splade_vs_dense"), g("pearson_old_splade_vs_dense"), d["within_row_std_ratio_new_to_old_splade_median"], "less" if d["within_row_std_ratio_new_to_old_splade_median"] < 1 else "more"))
c = d.get("old_splade_against_its_earlier_scoring")
if c:
    print("\nCheck of the reconstruction: soup-v9b scored again against its scores of the base rows from queue v49 (%d pairs): median |difference| %.4f, 99th percentile %.4f, max %.4f." % (c["pairs"], c["median_abs_diff"], c["p99_abs_diff"], c["max_abs_diff"]))
PYEOF
    echo; echo "**Arm and soups.** \`$NA\`: the recipe of the dose3 arm unchanged (from soup-v12, 2 epochs, lr 2e-5), training time $(tsec "$NA") s. Soups are uniform averages of the raw weights of arms from soup-v12."; echo
    cand_table
    echo; echo "The frozen model and its dose3 arm are in the forms they were measured in (bias015); the candidates are in their first form within the limits on the same grid of 0.05. heldout8 went to the best candidate on the 11 tasks only."; echo
    echo "**Gate for A** (>= +$MARGIN over the frozen model on the 19 tasks, within the limits): \`${BEST:-no candidate}\` ${B19:-not measured} against $F19 -> **$([ "$PASS" = 1 ] && echo "passes" || echo "does not pass")**."
    [ "$PASS" = 1 ] && echo "Over the index of the candidate (tiny10, asymmetric): static table \`phr2-k8-kd-r2-mw020\` ${CT:-not measured} (over the index of A: 60.67; allowed loss 0.2), released ALBERT ${AT:-not measured} (over the index of A: 64.90). 23 tasks: $N23."
} > "$EDIR/notes_s1.md"
EXTRA_LABELS="$EXTRA" esummary "Add the e results: the dose3 arm of the frozen model with the 124 M model in the teacher (E1)"
log "=== e s1 done ($(date '+%d.%m %H:%M')); best ${BEST:-none} 19 tasks ${B19:-?} (frozen $F19); gate for A: $PASS ==="
