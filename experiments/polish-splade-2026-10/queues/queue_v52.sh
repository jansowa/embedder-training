#!/usr/bin/env bash
# Queue v52 (3060 Ti, night 10/11.11 after v50 and v51, ~9 h): follow-ups of the student-mined arm
# (v49/v50: arm-dense-kl-n5kj-mined-ens400-r2v12 = 64.48 / 43.60 at 515 dims/doc, 64.63 tiny10 at 405 with bias 0.10;
# r2v12 on the old negatives 64.20 / 43.15 at 404). Every arm is read at equal index cost: the first bias form
# (0, 0.05, 0.10, 0.15, 0.20) with at most 410 dims/doc gets tiny10, heldout8 and trec-covid.
#   1  SMOKE of the refresh tool (kitchen bridge/splade_mine_and_score.py) on 300 rows; a failure skips steps 2-3.
#   2  REFRESHED NEGATIVES (ANCE-style, coarse): the mined arm used negatives chosen once by r2v12. Here the same
#      2 epochs from soup-v12 run as 4 segments of half an epoch; before every segment the current model mines its
#      own negatives from the rows' own passages and the ensemble teacher (soup-v9b + PolDense-400M) scores them.
#      The dense teacher's pool embeddings are cached, so a refresh costs about 6 minutes. Arm
#      arm-dense-kl-n5kj-iter4-r2v12-s4. Reading against the mined arm at equal cost: >= +0.2 on both sets = the
#      negatives should follow the student; equal = one mining pass is enough.
#      (Each segment restarts the learning-rate warm-up and ST's regulariser ramp; dims/doc may differ, hence the
#      equal-cost reading.)
#   3  HARD POSITIVES: n5 took 16 000 of the 56 k gated SQuAD-v2 rows at random. The refresh tool scores the other
#      gated rows with r2v12 and the ensemble; 16 000 rows where the teacher ranks the positive first and the
#      student does not replace the random 16 000 (fiqa and kj rows unchanged, negatives mined by r2v12 as in the
#      mined arm). Arm arm-dense-kl-n5hpkj-mined-ens400-r2v12, 2 epochs from soup-v12.
#   4  SEED 43 of the mined arm (is +0.45 heldout8 one seed's luck), then soup-v19 = uniform(seed 42, seed 43).
#   5  summary_v52.md, commit, push loop.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v52.sh > logs/queue_v52.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v52 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v52.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v50 v51}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
slbv() { regime_mean "$SREG/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
slbs() { echo "$(slbv "$1" tiny10) / $(slbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
small() { EXP_ROOT=$SEXP RESULTS_ROOT=$SRES REGIMES_ROOT=$SREG "$@"; }
h2task() { local bs=16; case "$2" in heldout2_poquad|heldout2_treccovid) bs=8;; esac; [ -d "models/$1" ] && regime quantized "$1" "models/$1" "$2" "$bs"; }
tc() { lbv "$1" heldout2_treccovid; }
measure() {  # measure LABEL MODEL REFLINE (main series)
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    sparsity "$label" "$model"; regime quantized "$label" "$model" tiny10 16; regime quantized "$label" "$model" heldout8 16
    log "  $label: $(lbs "$label") ($ref), $(dims "$label") dims/doc"
}
biased() {  # biased SRC_DIR OUT_NAME THRESHOLD: models/OUT_NAME = SRC with lm_head.bias (and its decoder copy) shifted by -(e^THRESHOLD - 1)
    local src="$1" out="$2" thr="$3"
    [ -d "models/$out" ] && return 0
    [ -d "$src" ] || { log "$out: $src missing"; return 1; }
    cp -r "$src" "models/$out" && "$PY" - "models/$out" "$thr" <<'EOF' && log "$out written (lm_head.bias = -(e^$thr - 1))" || { log "$out FAILED"; rm -rf "models/$out"; return 1; }
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
}
kscript() {  # kscript TAG TIMEOUT OUTPUT COMMAND...: a GPU helper script; OUTPUT is removed when it fails
    local tag="$1" to="$2" out="$3"; shift 3
    wait_gpu_free; wait_ram; local s=$SECONDS rc
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { rm -f "$out"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/$tag.log" | cut -c1-300)"
    record "chain:$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
soup() {
    local name="$1"; shift
    [ -d "models/$name" ] && return 0
    for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
    wait_gpu_free
    timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/soup_$name.log" 2>&1 \
        && log "soup $name written ($*, uniform)" || { log "soup $name FAILED"; rm -rf "models/$name"; return 1; }
}
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
A=datasets/arms
ROWS=$A/n5-plus-kj-tens400.jsonl
PD=OPI-PIB/PolDense-400M
DCACHE=cache/dense_pool/pd400
BASECFG="$C/distil_dense_kl_n5kj_mined_ens400_r2v12.yaml"
log "=== v52 start: mined arm $(lbs $M) at $(dims $M) dims/doc, bias010 $(lbs $M-bias010) at $(dims $M-bias010), bias015 $(lbs $M-bias015) at $(dims $M-bias015); $(free_gb) GB free ==="

refresh() {  # refresh TAG STUDENT ROWS OUT [tool args]: the student mines, the ensemble teacher scores
    local tag="$1" student="$2" rows="$3" out="$4"; shift 4
    [ -s "$out" ] && return 0
    kscript "refresh_$tag" 7200 "$out" "$PY" "$KITCHEN/bridge/splade_mine_and_score.py" --student "$student" --splade-teacher models/soup-v9b \
        --dense-teacher "$PD" --dense-cache "$DCACHE" --rows "$rows" --output "$out" "$@"
}
side() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]+'.mined.json'));print(d[sys.argv[2]])" "$1" "$2" 2>/dev/null; }
at_cost() {  # at_cost MODEL REFLINE: first bias form with <= 410 dims/doc -> tiny10, heldout8, trec-covid; PICK holds its label
    local m="$1" ref="$2" t; PICK="$m"
    [ -d "models/$m" ] || { log "$m missing: not measured"; return 1; }
    sparsity "$m" "models/$m"
    if [ "$(gte 410 "$(dims "$m")")" != 1 ]; then
        for t in 005 010 015 020; do
            biased "models/$m" "$m-bias$t" "0.${t#0}" || return 1
            sparsity "$m-bias$t" "models/$m-bias$t"; PICK="$m-bias$t"
            [ "$(gte 410 "$(dims "$PICK")")" = 1 ] && break
        done
        for t in 005 010 015 020; do [ "$m-bias$t" != "$PICK" ] && [ "$m" != "$M" ] && rm -rf "models/$m-bias$t"; done
    fi
    log "  $m: $(dims "$m") dims/doc unbiased; measured form $PICK at $(dims "$PICK")"
    measure "$PICK" "models/$PICK" "$ref"
    h2task "$PICK" heldout2_treccovid; log "  $PICK trec-covid $(tc "$PICK") (mined arm full 71.48, r2v12 67.02), $(dims "$PICK") dims/doc"
}
REF="mined arm bias010 $(lbs $M-bias010) at $(dims $M-bias010); bias015 $(lbs $M-bias015) at $(dims $M-bias015); r2v12 64.20 / 43.15 at 404"

# 1. smoke
log "=== v52 step 1: smoke of the refresh tool (300 rows) ==="
rm -rf cache/smoke_v52; mkdir -p cache/smoke_v52
if DCACHE=cache/smoke_v52/dense refresh smoke "models/$R2" "$ROWS" cache/smoke_v52/rows.jsonl --limit 300 \
    && DCACHE=cache/smoke_v52/dense refresh smoke2 "models/$R2" "$ROWS" cache/smoke_v52/rows2.jsonl --limit 300 --pool-rows "$ROWS" --dense-scale 80; then
    TOOL_OK=1; log "  smoke passed (second call: $(side cache/smoke_v52/rows2.jsonl dense_texts_encoded_now) new dense texts, expected 0)"
else TOOL_OK=0; log "  smoke FAILED: steps 2 and 3 skipped"; fi
rm -rf cache/smoke_v52

# 2. refreshed negatives: 4 segments of half an epoch, the current model mines before each
log "=== v52 step 2: refreshed negatives (4 x 0.5 epoch from soup-v12) ==="
ITER=dense-kl-n5kj-iter4-r2v12
if [ "$TOOL_OK" = 1 ]; then
    prev=models/soup-v12; miner="models/$R2"
    for k in 1 2 3 4; do
        rk="$A/n5-plus-kj-iter$k.jsonl"; cfg="$C/distil_dense_kl_n5kj_iter4_r2v12_s$k.yaml"
        refresh "iter$k" "$miner" "$ROWS" "$rk" || { log "  refresh $k failed: the loop stops"; break; }
        log "  round $k (miner $miner): dense scale $(side "$rk" dense_scale), teacher P(pos) median $(side "$rk" teacher_positive_mass_median), negative above positive in $(side "$rk" rows_with_negative_above_positive) rows, student ranks a positive first in $(side "$rk" student_ranks_a_positive_first)"
        sed -e "4s|.*|# Segment $k of 4 of the refreshed-negatives arm (queue v52): half an epoch from $prev on rows whose negatives $miner mined from the rows' own passages (5 best + 5 random of its top-60, length window 0.5-2.0), scored by the ensemble teacher (soup-v9b + PolDense-400M). Otherwise the r2v12 recipe.|" \
            -e "s|^runs_dir: .*|runs_dir: runs/arm-$ITER-s$k|" -e "s|^train_data: .*|train_data: $rk|" -e "s|^  - models/soup-v12|  - $prev|" \
            -e "s|num_train_epochs: 2|num_train_epochs: 0.5|" "$BASECFG" > "$cfg"
        train "$ITER-s$k" "$cfg"
        [ -d "models/arm-$ITER-s$k" ] || { log "  segment $k failed: the loop stops"; break; }
        prev="models/arm-$ITER-s$k"; miner="$prev"
    done
    at_cost "arm-$ITER-s4" "$REF"; ITER_PICK=$PICK
    [ -d "models/arm-$ITER-s4" ] && rm -rf "models/arm-$ITER-s1" "models/arm-$ITER-s2"
fi

# 3. hard positives among the gated SQuAD-v2 rows
log "=== v52 step 3: hard positives (teacher right, student wrong) among the unused gated SQuAD-v2 rows ==="
HP=arm-dense-kl-n5hpkj-mined-ens400-r2v12
REST="$A/squadv2-v2-gated-rest.jsonl"; RESTM="$A/squadv2-v2-gated-rest-mined.jsonl"; HPROWS="$A/n5hp-plus-kj-mined.jsonl"
if [ "$TOOL_OK" = 1 ] && [ -s "$A/n5-plus-kj-iter1.jsonl" ]; then
    [ -s "$REST" ] || "$PY" - "$A/squadv2-v2-all.jsonl" "$ROWS" "$REST" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, sys
src, n5, out = sys.argv[1:4]
used = {str(json.loads(l)["query_id"]) for l in open(n5, encoding="utf-8") if '"squadv2-v2"' in l}
n = kept = 0
with open(out, "w", encoding="utf-8") as f:
    for line in open(src, encoding="utf-8"):
        r = json.loads(line); n += 1
        gate = r.get("pos_scores_stronger_reranker") or []
        if str(r["query_id"]) in used or not gate or max(gate) <= 23.5:
            continue
        f.write(line if line.endswith("\n") else line + "\n"); kept += 1
print("gated SQuAD-v2 rows outside n5: %d of %d (n5 holds %d)" % (kept, n, len(used)))
PYEOF
    refresh hp-candidates "models/$R2" "$REST" "$RESTM" --pool-rows "$ROWS" --dense-scale "$(side "$A/n5-plus-kj-iter1.jsonl" dense_scale)"
    if [ -s "$RESTM" ] && [ ! -s "$HPROWS" ]; then
        "$PY" - "$RESTM" "$A/n5-plus-kj-iter1.jsonl" "$HPROWS" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, random, sys
cand, base, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(cand, encoding="utf-8")]
hard = [r for r in rows if min(r["pos_teacher_rank"]) == 1 and min(r["pos_student_rank"]) >= 2]
easy = [r for r in rows if min(r["pos_teacher_rank"]) == 1 and min(r["pos_student_rank"]) == 1]
rng = random.Random(42); rng.shuffle(hard); rng.shuffle(easy)
pick = (hard + easy)[:16000]
keep = [json.loads(l) for l in open(base, encoding="utf-8")]
keep = [r for r in keep if r.get("translation") != "squadv2-v2"]
allrows = keep + pick; rng.shuffle(allrows)
with open(out, "w", encoding="utf-8") as f:
    for r in allrows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
med = sorted(min(r["pos_student_rank"]) for r in pick)[len(pick) // 2] if pick else 0
print("candidates %d: teacher right + student wrong %d, teacher right + student right %d, teacher wrong %d" % (len(rows), len(hard), len(easy), len(rows) - len(hard) - len(easy)))
print("hard-positive file: %d rows = %d fiqa/kj + %d SQuAD-v2 (%d of them hard, median student rank of the positive %d)" % (len(allrows), len(keep), len(pick), min(len(hard), len(pick)), med))
PYEOF
    fi
    if [ -s "$HPROWS" ]; then
        sed -e "4s|.*|# The mined arm with hard positives (queue v52): the 16 000 random gated SQuAD-v2 rows of n5 are replaced by 16 000 other gated rows where the ensemble teacher ranks the positive first and r2v12 does not; fiqa and kj rows unchanged; negatives mined by r2v12 from the rows' own passages, scored by the ensemble (soup-v9b + PolDense-400M). The r2v12 recipe from soup-v12.|" \
            -e "s|^runs_dir: .*|runs_dir: runs/$HP|" -e "s|^train_data: .*|train_data: $HPROWS|" "$BASECFG" > "$C/distil_dense_kl_n5hpkj_mined_ens400_r2v12.yaml"
        train "${HP#arm-}" "$C/distil_dense_kl_n5hpkj_mined_ens400_r2v12.yaml"
        at_cost "$HP" "$REF"; HP_PICK=$PICK
    else log "  hard-positive rows missing: arm skipped"; fi
fi

# 4. seed 43 of the mined arm and the seed soup
log "=== v52 step 4: seed 43 of the mined arm, soup-v19 ==="
sed -e "s|^runs_dir: .*|runs_dir: runs/$M-s43|" -e '/^    loss: sparse_distill_kl_div/a\    seed: 43' "$BASECFG" > "$C/distil_dense_kl_n5kj_mined_ens400_r2v12_s43.yaml"
train "${M#arm-}-s43" "$C/distil_dense_kl_n5kj_mined_ens400_r2v12_s43.yaml"
at_cost "$M-s43" "seed 42: $REF"; S43_PICK=$PICK
soup soup-v19 "models/$M" "models/$M-s43" && { at_cost soup-v19 "seeds: $REF; s43 $(lbs "$S43_PICK")"; V19_PICK=$PICK; }

# 5. summary
log "=== v52 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_v52.md"
{
    echo "# v52 (night 10/11.11, quantized regime, nDCG@10): refreshed negatives, hard positives, the second seed of the mined arm"; echo
    echo "Every new arm is measured in its first bias form with at most 410 dims/doc (biasT = lm_head.bias shifted by -(e^T - 1)). iter4 = 4 x 0.5 epoch from soup-v12, the current model mines its negatives before each segment; n5hpkj = the SQuAD-v2 part of n5 replaced by rows the teacher gets right and r2v12 does not; mined = v49 (negatives mined once by r2v12)."; echo
    echo "| model | tiny10 | heldout8 | trec-covid | dims/doc |"; echo "|---|---:|---:|---:|---:|"
    for m in $R2 $M $M-bias010 $M-bias015 ${ITER_PICK:-} ${HP_PICK:-} ${S43_PICK:-} ${V19_PICK:-}; do
        echo "| $m | $(lbv $m tiny10) | $(lbv $m heldout8) | $(tc $m) | $(dims $m) |"
    done
    echo; echo "## Refresh rounds (student = miner of the round)"; echo
    echo "| round | miner | dense scale | teacher P(pos) median | rows with a negative above a positive | student ranks a positive first |"; echo "|---|---|---:|---:|---:|---:|"
    for k in 1 2 3 4; do rk="$A/n5-plus-kj-iter$k.jsonl"; [ -s "$rk.mined.json" ] && echo "| $k | $(side "$rk" student) | $(side "$rk" dense_scale) | $(side "$rk" teacher_positive_mass_median) | $(side "$rk" rows_with_negative_above_positive) | $(side "$rk" student_ranks_a_positive_first) |"; done
    [ -s "$RESTM.mined.json" ] && { echo; echo "## Hard-positive candidates (gated SQuAD-v2 rows outside n5, r2v12 as the student)"; echo; echo '```'; cat "$RESTM.mined.json"; echo '```'; }
    echo; echo "## Queue state (v52)"; echo; echo '```'; grep -E "iter4|n5hpkj|mined-ens400-r2v12-s43|soup-v19|chain:refresh" "$STATE"; echo '```'
} > "$SUM"
log "summary written to $SUM"
git add "$SUM" "$EXP_ROOT/queues/queue_v52.sh" "$C"/distil_dense_kl_n5kj_iter4_r2v12_s?.yaml "$C/distil_dense_kl_n5hpkj_mined_ens400_r2v12.yaml" "$C/distil_dense_kl_n5kj_mined_ens400_r2v12_s43.yaml" 2>/dev/null
for m in ${ITER_PICK:-} ${HP_PICK:-} ${S43_PICK:-} ${V19_PICK:-} "arm-$ITER-s4" $HP $M-s43 soup-v19; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/*-bias0??.json 2>/dev/null
git commit -q -m "Add the v52 results: refreshed negatives, hard positives and the second seed of the student-mined arm" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v52.log" 2>&1 && { echo pushed >> "$LOGS/push_v52.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v52 done ($(date '+%d.%m %H:%M')) ==="
