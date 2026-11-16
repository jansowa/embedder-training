#!/usr/bin/env bash
# Queue b1c (3060 Ti, from 15.11; takes over from queue b1 at one of its pause points): the larger backbone
# sdadas/polish-roberta-base-v2 (124 M) after KD from the frozen full model, now with a LARGE ranking stage. The user's
# decision of 15.11: the 82 M family started from sdadas/polish-splade, which had seen all of MS MARCO-pl, and dose3 was
# only a fine-tuning for it; the 124 M student has seen no ranking data at all. So a weak R-a does not stop anything
# (R-a is the "little data" reference), R-b is dropped, and the dose curve of the large stage decides about a larger
# dose on the 5070. The release of 26.11 does not depend on any of this.
#   1  (b1's steps, done only when b1 had not done them) tiny10 of the KD model, R-a = the dose3 rows and recipe from
#      the KD model, MEASURE R-a (tiny10, heldout8, trec-covid, cost, per-task table; 23 tasks when >= +0.2 on both sets)
#   A  MS MARCO-pl ROWS: the rows after the positive gate 23.5 (the cached output of the trainer's filter), one positive
#      per query, without benchmark queries (pirb-queries) and without the queries of dose3, shuffled once and cut into
#      units of 25 000 rows. Negatives as in dose3 (queue_v56.sh): miner r2v12, 5 best + 5 random of its top-60, length
#      window 0.5-2.0; the pool of a unit = the positives of its own rows (about 25 000 passages, the size of the pool of
#      dose3; the large shared pool of v53 made the negatives too hard). Scores of the same ensemble as dose3 (soup-v9b +
#      PolDense-400M, bf16, the FIXED dense scale 84.194), by the same tool as dose3 (splade_mine_and_score.py, one pass
#      over a closed pool). A pause point after every unit. Statistics next to dose3.
#   B  STAGE 1: from the KD model, the recipe of R-a, one epoch over 200 000 MS MARCO-pl rows (no dose3 rows) in four
#      segments of 50 000 rows, each with its own warm-up (10 %) and linear decay, so every point of the DOSE CURVE is an
#      annealed model; after each segment: the first form within both limits, tiny10, density. A pause point after every
#      training and every measurement. The end of stage 1 is measured in full.
#   C  THE DOSE3 ROUND from the end of stage 1, the recipe of R-a unchanged; measured as R-a.
#   summary_b1.md after R-a, after the rows, after stage 1 and at the end; commit, push loop.
# The package for the 5070 is NOT created here (it needs the user's consent); the queue only reports its size.
# Between steps the queue waits while logs/b1.pause exists. No deletions: models and bias forms go to the data drive.
#   bash logs/relaunch-20261113/launch_queue.sh b1c experiments/polish-splade-2026-10/queues/queue_b1c.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b1c MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-79200}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
echo $$ > "$LOGS/queue_b1c.pid"
guard_alive
queue_alive b1 && { log "queue b1 is still running: b1c does not start"; exit 1; }
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"

CORPUS=datasets/kd/kd_v1.jsonl
FULL=$(final_full); [ -n "$FULL" ] && [ -d "models/$FULL" ] || { log "  no frozen full model in $FINAL_FULL_FILE: b1c stops"; exit 1; }
RAW=${FULL%-bias[0-9][0-9][0-9]}
TEACHER="models/$RAW"
D3_PICK=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015
ALB=$(final_albert)
ST=rbase-v2; KD=kd-$ST-dqf-full
AKD=kd-b4-l0125-f1536-hl-r256-w-dqf-full
RA=arm-b1-ra; RCD=arm-b1-rc-d3; NSEG=4; SEG_UNITS=2; UNIT=25000
seg() { echo "arm-b1-rc-s1p$1"; }
MAX_POSTINGS=${MAX_POSTINGS:-16807}; MAX_DIMS=410
FORMS="$RESULTS_ROOT/index_cost_b1_forms"; COST="$RESULTS_ROOT/index_cost_b1.json"; VERDICT="$RESULTS_ROOT/b1_reading.json"
PROF="$RESULTS_ROOT/b1_profile.tsv"; SUM="$RESULTS_ROOT/summary_b1.md"; CURVE="$RESULTS_ROOT/b1c_curve.tsv"
ROWREP="$RESULTS_ROOT/b1c_msmarco_rows.json"; ROWSTATS="$RESULTS_ROOT/b1c_rows_stats.json"
DRIVE="$(dirname "$(readlink -f runs)")"; DD="$DRIVE/FlagEmbedding-models-b1"
MS="$DRIVE/work/kitchen-data/msmarco-rc"; UNITS="$MS/units"; MINED="$MS/mined"; SEGS="$MS/segments"
REFQ="$DRIVE/work/kitchen-data/reference/pirb-queries.jsonl"
MSSRC=cache/huggingface_datasets/speakleash-msmarco_pl-main-train_pl-5e6d3de402769748/dataset.jsonl
PROFILE=configs/dataset_filters/splade_positive_score_gt_23_50_no_metricx.yaml
D3ROWS=datasets/arms/n5-plus-kj-mined-dose3.jsonl
R2=arm-dense-kl-n5kj-ens400-r2v12; PD=OPI-PIB/PolDense-400M; SCALE=84.19441654567078
mkdir -p "$FORMS" "$RESULTS_ROOT/fidelity" "$DD" "$UNITS" "$MINED" "$SEGS" || { log "  cannot create the working directories on the data drive: b1c stops"; exit 1; }
log "=== b1c start: $KD, R-a, then the large MS MARCO-pl stage (R-c'); dose3 $(lbs "$D3_PICK") / trec-covid $(tc "$D3_PICK"); frozen model $FULL; ALBERT ${ALB:-none}; $(free_gb) GB free on /, $(df -BG --output=avail "$DD" | tail -1 | tr -dc '0-9') GB on the data drive ==="
for f in "models/$KD/model.safetensors" "$TEACHER/model.safetensors" "$CORPUS" "$D3ROWS" "$MSSRC" "$PROFILE" "$REFQ" "$PROF" \
         "$C/b1_ra_dose3.yaml" "$C/b1_rc_s1p1.yaml" "$C/b1_rc_s1p2.yaml" "$C/b1_rc_s1p3.yaml" "$C/b1_rc_s1p4.yaml" "$C/b1_rc_d3.yaml" \
         "$KITCHEN/bridge/msmarco_dose_rows.py" "$KITCHEN/bridge/splade_mine_and_score.py" scripts/arm_reading.py scripts/merge_regime_task.py; do
    [ -s "$f" ] || { log "  $f missing: b1c stops"; exit 1; }
done
for m in "$D3_PICK" "$R2" soup-v9b; do [ -d "models/$m" ] || { log "  models/$m missing: b1c stops"; exit 1; }; done
need_space || { log "  not enough space on /: b1c stops"; exit 1; }
[ -e datasets/arms/msmarco-rc ] || ln -s "$SEGS" datasets/arms/msmarco-rc

pause_point() { local said=0; while [ -e "$LOGS/b1.pause" ]; do [ "$said" = 0 ] && { log "  paused by logs/b1.pause (remove it to continue)"; said=1; }; sleep 60; done; }
slbv() { regime_mean "$SEXP/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
sdims() { sparsity_mean "$SEXP/results/sparsity/$1.json" "$1" 2>/dev/null; }
fid() {
    "$PY" -c "import json,sys;k=json.load(open(sys.argv[1]))['kinds'][sys.argv[2]];m=lambda n:k[n]['mean'];print('cosine %.4f, recall of the teacher dims %.3f, Jaccard %.3f, L1 rel. error %.3f, dims teacher / student %.0f / %.0f (n=%d)' % (m('cosine'), m('teacher_active_recall'), m('jaccard_active'), m('l1_relative_error'), m('teacher_active_dims'), m('student_active_dims'), k['n']))" "$1" "$2" 2>/dev/null || echo "not measured"
}
fidrow() {
    "$PY" -c "import json,sys;k=json.load(open(sys.argv[2]))['kinds'][sys.argv[3]];m=lambda n:k[n]['mean'];print('| %s | %d | %.4f | %.3f | %.3f | %.3f | %.0f | %.0f |' % (sys.argv[1], k['n'], m('cosine'), m('teacher_active_recall'), m('jaccard_active'), m('l1_relative_error'), m('teacher_active_dims'), m('student_active_dims')))" "$1" "$2" "$3" 2>/dev/null || echo "| $1 | - | - | - | - | - | - | - |"
}
ntasks() { regime_mean "$REGIMES_ROOT/quantized/$1/heldout8.json" 2>/dev/null | awk '{print $2}'; }
fix8() {
    [ "$(ntasks "$1")" = 7 ] || return 0
    log "  $1: heldout8 has 7 tasks; fiqa-pl alone"
    regime quantized "$1" "models/$1" heldout8_fiqa 8 && "$PY" scripts/merge_regime_task.py --into "$REGIMES_ROOT/quantized/$1/heldout8.json" \
        --from "$REGIMES_ROOT/quantized/$1/heldout8_fiqa.json" --n-tasks 8 | while read -r l; do log "  $l"; done
}
partial_aside() {
    local f="$REGIMES_ROOT/quantized/$1/heldout8.json" n; [ -f "$f" ] || return 0
    n=$(ntasks "$1"); [ "${n:-0}" -lt 7 ] || return 0
    mkdir -p "$LOGS/b1-aside"; mv "$f" "$LOGS/b1-aside/heldout8-partial-$1-$(date +%H%M%S).json"; log "  $1: heldout8 with ${n:-0} of 8 tasks moved to logs/b1-aside (not a result)"
}
h8() {
    regime quantized "$1" "models/$1" heldout8 16; fix8 "$1"
    if [ "$(ntasks "$1")" != 8 ]; then partial_aside "$1"; regime quantized "$1" "models/$1" heldout8 8; fix8 "$1"; fi
    [ "$(ntasks "$1")" = 8 ] || { log "  $1: heldout8 is not complete after a repeat"; partial_aside "$1"; return 1; }
}
pq() { "$PY" -c "import json,sys;print('%.0f' % json.load(open(sys.argv[1]))['models'][sys.argv[2]]['postings_per_query'])" "$1" "$2" 2>/dev/null; }
qd() { "$PY" -c "import json,sys;print('%.1f' % json.load(open(sys.argv[1]))['models'][sys.argv[2]]['query_active_mean'])" "$1" "$2" 2>/dev/null; }
to_dd() {  # to_dd LABEL: move models/LABEL to the data drive and link it back (space on /)
    [ -d "models/$1" ] && [ ! -L "models/$1" ] || return 0
    [ -e "$DD/$1" ] && { log "  $DD/$1 exists already: models/$1 stays on /"; return 0; }
    mv "models/$1" "$DD/$1" && ln -s "$DD/$1" "models/$1" || log "  moving models/$1 to the data drive FAILED"
}
form_fit() {  # form_fit MODEL: PICK = the first form (no shift, then bias steps of 0.05) with <= MAX_DIMS dims/doc and <= MAX_POSTINGS postings/query
    local m="$1" t f p; PICK=""
    [ -d "models/$m" ] || { log "  $m missing: no form"; return 1; }
    for t in "" 005 010 015 020 025 030 035 040 045 050; do
        f="$m${t:+-bias$t}"
        if [ -n "$t" ]; then biased "models/$m" "$f" "0.${t#0}" || return 1; to_dd "$f"; fi
        sparsity "$f" "models/$f"
        [ "$(gte "$MAX_DIMS" "$(dims "$f")")" = 1 ] || { log "  $f: $(dims "$f") dims/doc, above $MAX_DIMS"; continue; }
        index_cost "$FORMS/$f.json" "$f"
        p=$(pq "$FORMS/$f.json" "$f")
        log "  $f: $(dims "$f") dims/doc, $(qd "$FORMS/$f.json" "$f") dims/query, ${p:-?} postings/query (limits $MAX_DIMS and $MAX_POSTINGS)"
        if [ -n "$p" ] && [ "$(gte "$MAX_POSTINGS" "$p")" = 1 ]; then PICK="$f"; return 0; fi
    done
    log "  $m: no form within both limits up to bias050"; return 1
}
measure() {  # measure LABEL: tiny10, heldout8 (complete), trec-covid
    [ -n "$1" ] || return 1
    regime quantized "$1" "models/$1" tiny10 16; h8 "$1"
    regime quantized "$1" "models/$1" heldout2_treccovid 8
    log "  $1: $(lbs "$1") / trec-covid $(tc "$1") at $(dims "$1") dims/doc, $(qd "$FORMS/$1.json" "$1") dims/query, $(pq "$FORMS/$1.json" "$1") postings/query (dose3 $(lbs "$D3_PICK") / $(tc "$D3_PICK"); frozen $(lbs "$FULL") / $(tc "$FULL"))"
}
delta() { [ -n "$1" ] && [ -n "$2" ] && fsub "$1" "$2" || echo "-"; }
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
train_rank() {  # train_rank NAME BASE_YAML: train in mode $RMODE; after an out-of-memory failure go down the ladder (plain, gc, b1gc)
    local tag="$1" base="$2" cfg m
    for m in plain gc b1gc; do
        case "$RMODE:$m" in gc:plain|b1gc:plain|b1gc:gc) continue;; esac
        cfg=$(mkcfg "$base" "$m") || { log "  config of arm-$tag ($m) FAILED"; return 1; }
        train "$tag" "$cfg"
        if [ -d "models/arm-$tag" ]; then
            [ "$m" != "$RMODE" ] && { RMODE=$m; printf 'rank_mode\t%s\n' "$m" >> "$PROF"; log "  ranking mode is now $m (arm-$tag ran out of memory in the mode before)"; }
            return 0
        fi
        grep -q -i "out of memory" "$LOGS/arm_$tag.log" 2>/dev/null || return 1
        log "  arm-$tag ran out of memory in mode $m"; mkdir -p "$LOGS/b1-aside"; cp "$LOGS/arm_$tag.log" "$LOGS/b1-aside/arm_$tag.$m.oom.log"
    done
    return 1
}
prof_get() { awk -F'\t' -v k="$1" '$1==k {v=$2} END {print v}' "$PROF" 2>/dev/null; }
tsec() { awk -F'\t' -v k="train:$1" '$1==k && $2=="ok" {v=$4} END {print v}' "$STATE" | tr -dc '0-9'; }      # seconds of a finished training
kscript() {  # kscript TAG TIMEOUT OUTPUT COMMAND...: a helper script of the kitchen; a failed run leaves no OUTPUT
    local tag="$1" to="$2" out="$3"; shift 3
    wait_gpu_free; wait_ram; local s=$SECONDS rc
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { [ -e "$out" ] && mv "$out" "$out.failed-$(date +%H%M%S)"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/$tag.log" | cut -c1-300)"
    record "chain:$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
side() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]+'.mined.json'));v=d[sys.argv[2]];print('%.3f' % v if isinstance(v,float) else v)" "$1" "$2" 2>/dev/null; }
rget() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));v=d.get(sys.argv[2]);print(v if v is not None else '')" "$ROWREP" "$1" 2>/dev/null; }
ptable() {  # ptable NAME=LABEL...: nDCG@10 per task of tiny10, heldout8 and trec-covid; a label "small:LABEL" is read from the small series
    "$PY" - "$REGIMES_ROOT/quantized" "$SEXP/results-regimes/quantized" "$@" <<'PYEOF'
import json, sys
main, small, cols = sys.argv[1], sys.argv[2], [c.split("=", 1) for c in sys.argv[3:]]
def tasks(label, ts):
    root, lab = (small, label[6:]) if label.startswith("small:") else (main, label)
    try:
        return json.load(open(f"{root}/{lab}/{ts}.json"))["tasks"]
    except Exception:
        return {}
print("| task | " + " | ".join(n for n, _ in cols) + " |"); print("|---|" + "---:|" * len(cols))
for ts in ("tiny10", "heldout8", "heldout2_treccovid"):
    data = [tasks(l, ts) for _, l in cols]
    names = list(dict.fromkeys(t for d in data for t in d))
    for t in names:
        print(f"| {t} | " + " | ".join(("%.2f" % d[t]) if t in d and d[t] is not None else "-" for d in data) + " |")
    if ts != "heldout2_treccovid" and names:
        means = [sum(d.values()) / len(d) if d and len(d) == len(names) else None for d in data]
        print(f"| **{ts} mean** | " + " | ".join(("**%.2f**" % m) if m is not None else "-" for m in means) + " |")
PYEOF
}

TOK=$(grep -h "\[b1\]   tokenizer of" "$LOGS/queue.log" 2>/dev/null | tail -n 1 | sed 's/^.*: //'); TOK=${TOK:-see logs/queue.log}
KD_PICK=""; RA_PICK=""; S1=""; S1_PICK=""; RC_PICK=""; RA_NOTE=""; RC_NOTE=""; M23=""; CURVE_NOTE=""
write_summary() {  # write_summary COMMIT_MESSAGE: the state so far
    local rm="${RMODE:-$(prof_get rank_mode)}" k
    row() {  # row NAME LABEL COSTFILE
        [ -n "$2" ] || return 0
        echo "| $1 | \`$2\` | $(lbv "$2" tiny10) | $(lbv "$2" heldout8) | $(tc "$2") | $(delta "$(lbv "$2" tiny10)" "$(lbv "$D3_PICK" tiny10)") | $(delta "$(lbv "$2" heldout8)" "$(lbv "$D3_PICK" heldout8)") | $(dims "$2") | $(qd "$3" "$2") | $(pq "$3" "$2") |"
    }
    {
        echo "# b1 (quantized regime, nDCG@10): a larger backbone after KD from the frozen full model"; echo
        echo "Student: \`sdadas/polish-roberta-base-v2\` (12 layers, 124 M) as a SparseEncoder (MLMTransformer with the model's own MLM head + SpladePooling max). KD: the recipe of the ALBERT student (token KL x 0.3 + pooled MSE on all of kd_v1, one epoch, lr 5e-5, warmup 300, linear, fp16, 256 tokens, batch 16, seed 42), teacher = the raw weights of the frozen full model (\`$TEACHER\`). Ranking versions start from the KD model: **R-a** = the rows and the recipe of dose3 unchanged (only the start model differs from dose3; the \"little data\" reference); **R-c'** = stage 1 on 200 000 MS MARCO-pl rows alone (four segments of 50 000, the recipe of R-a with one epoch), then a round on the dose3 rows with the recipe of R-a. R-b (stage 3 on n5 + kj) was dropped on the user's decision of 15.11: the 82 M family started from a model trained on all of MS MARCO-pl, the 124 M student has seen no ranking data, so the amount of ranking data is the open question. Every model is measured in its first form (no shift, then bias steps of 0.05) with at most $MAX_DIMS dims/doc and at most $MAX_POSTINGS postings/query."; echo
        echo "## Precondition"; echo
        echo "- Tokenizer against the frozen model: $TOK."
        echo "- Memory profile on the 8 GB card (peak = memory used on the whole card):"; echo; echo '```'; cat "$PROF" 2>/dev/null; echo '```'; echo
        echo "- The first profile attempt (15.11 01:11) ran out of memory in both KD modes: the student before KD activates about 46 800 dimensions per text, so the backward of the sparse decoder asked for entries x hidden floats in one piece (2.3 GiB). It now goes in slices of 32 768 entries (the same sums, \`training/backends/splade_fused_head.py\`); the logs of that attempt are in \`logs/b1-aside/\`."
        echo "- KD mode: $(prof_get kd_mode). Ranking mode: ${rm:-none fits} ($([ "$rm" = plain ] && echo "the dose3 recipe unchanged: 2 x 16, 512 tokens")$([ "$rm" = gc ] && echo "the dose3 recipe with gradient checkpointing; batch 2 x 16 and 512 tokens unchanged")$([ "$rm" = b1gc ] && echo "physical batch 1 x 32 accumulation with gradient checkpointing: NOT the dose3 recipe one to one")). Training time: $(grep -E "^(kd:${KD#kd-}|train:arm-b1-)" "$STATE" | awk -F'\t' '{printf "%s %s; ", $1, $4}')"; echo
        echo "## Fidelity to the teacher (\`$RAW\`)"; echo
        echo "| student | texts | cosine | recall of the teacher dims | Jaccard | L1 rel. error | teacher dims | student dims |"
        echo "|---|---:|---:|---:|---:|---:|---:|---:|"
        fidrow "roberta-base-v2 before KD, corpus texts" "$RESULTS_ROOT/fidelity/student-$ST.corpus.json" all
        fidrow "roberta-base-v2 before KD, queries" "$RESULTS_ROOT/fidelity/student-$ST.queries.json" query
        fidrow "roberta-base-v2 after KD, queries" "$RESULTS_ROOT/fidelity/$KD.queries.json" query
        fidrow "new ALBERT (50 M) before KD, queries" "$SEXP/results/fidelity/student-b4-l0125-f1536-hl-r256-w-dqf.queries-stock.json" query
        fidrow "new ALBERT (50 M) after KD, queries" "$SEXP/results/fidelity/$AKD.queries.json" query
        echo; echo "Fidelity curve of the KD (256 corpus texts per checkpoint):"; echo; echo '```'
        [ -f "runs/$KD/log.jsonl" ] && "$PY" scripts/kd_fidelity_curve.py "runs/$KD/log.jsonl" 2>> "$LOGS/queue_summary.log"
        echo '```'; echo
        echo "After KD alone: \`${KD_PICK:-no form within the limits}\` tiny10 $(lbv "${KD_PICK:-none}" tiny10) at $(dims "${KD_PICK:-none}") dims/doc, $(qd "$FORMS/${KD_PICK:-none}.json" "${KD_PICK:-none}") dims/query, $(pq "$FORMS/${KD_PICK:-none}.json" "${KD_PICK:-none}") postings/query ($(dims "$KD") dims/doc without a shift). The new ALBERT after KD alone: 64.03 at 554.8 dims/doc without a shift (not within the limits)."; echo
        echo "## Ranking versions against dose3"; echo
        echo "| model | label | tiny10 | heldout8 | trec-covid | tiny10 vs dose3 | heldout8 vs dose3 | dims/doc | cost: dims/query | cost: postings/query |"
        echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        row "dose3 (82 M, v56), the reference" "$D3_PICK" "$COST"
        row "frozen full model (82 M)" "$FULL" "$COST"
        row "R-a: roberta-base-v2 after KD, dose3 recipe (124 M)" "${RA_PICK:-}" "$COST"
        row "R-c' after stage 1: 200 000 MS MARCO-pl rows (124 M)" "${S1_PICK:-}" "$COST"
        row "R-c' after the dose3 round (124 M)" "${RC_PICK:-}" "$COST"
        [ -n "$ALB" ] && echo "| ALBERT (50 M, a1) | \`$ALB\` | $(slbv "$ALB" tiny10) | $(slbv "$ALB" heldout8) | $(slbv "$ALB" heldout2_treccovid) | $(delta "$(slbv "$ALB" tiny10)" "$(lbv "$D3_PICK" tiny10)") | $(delta "$(slbv "$ALB" heldout8)" "$(lbv "$D3_PICK" heldout8)") | $(sdims "$ALB") | $(qd "$SEXP/results/index_cost_a1.json" "$ALB") | $(pq "$SEXP/results/index_cost_a1.json" "$ALB") |"
        echo
        echo "Unbiased dims/doc: KD model $(dims "$KD"), R-a $(dims "$RA"), R-c' after stage 1 $(dims "${S1:-none}"), R-c' after the dose3 round $(dims "$RCD"). One seed each; the seed noise of the ranking recipe is 0.13-0.2."; echo
        [ -n "$RA_NOTE" ] && { echo "$RA_NOTE"; echo; }
        [ -n "$RC_NOTE" ] && { echo "$RC_NOTE"; echo; }
        if [ -s "$ROWREP" ]; then
            echo "## MS MARCO-pl rows of R-c'"; echo
            echo "Source: the MS MARCO-pl train rows (\`speakleash/msmarco_pl\`; the pairs of \`datasets/msmarco-full-judged-qwen/pairs_msmarco_full.unique.jsonl\` are the positives of the same rows) through the trainer's filter \`$(basename "$PROFILE")\`. Rows before the filter: $(rget rows_before_filter) ($(rget pairs_before_filter) query-positive pairs); after the filter: $(rget rows_after_filter) ($(rget pairs_after_filter) pairs); after the exclusions: **$(rget eligible_rows)** (benchmark queries of pirb-queries, by normalised text or sorted tokens: $(rget dropped_benchmark_query); queries of the dose3 rows: $(rget dropped_query_of_training_rows); repeated query texts: $(rget dropped_duplicate_query)). One positive per query (the highest gate score; $(rget extra_positives_left_out) further positives left out, $(rget chosen_positive_synthetic) of the kept ones are positives added by the two-stage mining of the source). Shuffled once (seed 42) and cut into units of $UNIT rows; stage 1 uses the first $((NSEG * SEG_UNITS)) units, the others are for the 5070."; echo
            echo "Negatives and scores as in dose3 (\`queue_v56.sh\`): miner \`$R2\`, 5 best + 5 random of its top-60 inside a length window of 0.5-2.0 of the positive; the pool of a unit is the positives of its own rows (about $UNIT passages; dose3: 19 008 passages of its own rows); ensemble teacher soup-v9b + PolDense-400M (bf16) with the fixed dense scale $SCALE, computed by \`bridge/splade_mine_and_score.py\` in one pass over the closed pool (the tool of dose3; the same mean of the two teachers as \`scripts/reranker_rows.py ensemble\`)."; echo
            [ -s "$ROWSTATS" ] && "$PY" - "$ROWSTATS" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("| rows | n | teacher P(pos) at T = 2, median | share above 0.95 | rows with a negative above the positive | negative / positive length, median | p90 | query chars, median | positive chars, median | student ranks the positive first |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, s in d["sets"].items():
    print("| %s | %d | %.3f | %.3f | %d (%.1f %%) | %.2f | %.2f | %.0f | %.0f | %s |" % (name, s["rows"], s["p_pos_median"], s["p_pos_above_095"], s["negative_above_positive"],
          100.0 * s["negative_above_positive"] / max(1, s["rows"]), s["len_ratio_median"], s["len_ratio_p90"], s["query_chars_median"], s["positive_chars_median"],
          ("%.3f" % s["student_first"]) if s.get("student_first") is not None else "-"))
PYEOF
            echo
        fi
        if [ -s "$CURVE" ]; then
            echo "## Dose curve of stage 1 (MS MARCO-pl rows alone)"; echo
            echo "Every segment is one epoch over its own 50 000 rows from the model of the segment before, with its own warm-up (10 %) and linear decay to zero (lr 2e-5), so each point is an annealed model and the points are comparable; in one run with a single schedule the intermediate points would be read at a high learning rate. tiny10 of the first form within both limits."; echo
            echo "| examples | model | measured form | tiny10 | step | dims/doc | dims/query | postings/query | dims/doc without a shift | training time |"; echo "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"
            LC_ALL=C awk -F'\t' 'BEGIN {prev=""} {d = (prev != "" && $4 != "") ? sprintf("%+.2f", $4 - prev) : "-"; printf "| %s | `%s` | `%s` | %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, d, $5, $6, $7, $8, $9; if ($4 != "") prev = $4}' "$CURVE"
            echo; [ -n "$CURVE_NOTE" ] && { echo "$CURVE_NOTE"; echo; }
        fi
        local pick="${RC_PICK:-${S1_PICK:-${RA_PICK:-}}}" cols=(--column "dose3 (v56)=$D3_PICK" --column "frozen full model=$FULL")
        [ -n "${RA_PICK:-}" ] && cols+=(--column "R-a=$RA_PICK"); [ -n "${S1_PICK:-}" ] && cols+=(--column "R-c' stage 1=$S1_PICK"); [ -n "${RC_PICK:-}" ] && cols+=(--column "R-c' + dose3 round=$RC_PICK")
        local m23=(--m23 "$D3_PICK" --m23 "$FULL"); local x; for x in ${M23:-}; do m23+=(--m23 "$x"); done
        if [ -n "$pick" ] && [ -n "$(lbv "$pick" heldout8)" ] && [ -s "$COST" ]; then
            "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" "${cols[@]}" --pick "$pick" --ref "$D3_PICK" \
                --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
                --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
                --group "scifact-pl=scifact-pl" --group "maupqa-mkqa=maupqa-mkqa" --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
                --win "THE LARGER MODEL WINS ($pick >= +0.2 on both sets over dose3 within the limits)" \
                --noise "$pick is within the seed noise of dose3 set by set" \
                --loss "$pick is a LOSS on both sets against dose3" "${m23[@]}" --json "$VERDICT" 2>> "$LOGS/queue_summary.log"
            echo; echo "## Per task, with the ALBERT"; echo
            local pt=("dose3=$D3_PICK" "frozen full model=$FULL")
            [ -n "${RA_PICK:-}" ] && pt+=("R-a=$RA_PICK"); [ -n "${S1_PICK:-}" ] && pt+=("R-c' stage 1=$S1_PICK"); [ -n "${RC_PICK:-}" ] && pt+=("R-c' + dose3 round=$RC_PICK"); [ -n "$ALB" ] && pt+=("ALBERT (50 M)=small:$ALB")
            ptable "${pt[@]}" 2>> "$LOGS/queue_summary.log"
        fi
        echo; echo "## Queue state (b1, b1c)"; echo; echo '```'; grep -E "b1:|$ST|$KD|arm-b1-|chain:rc_" "$STATE" | tail -n 120; echo '```'
    } > "$SUM"
    log "summary written to $SUM"
    git add "$SUM" "$EXP_ROOT/queues/queue_b1.sh" "$EXP_ROOT/queues/queue_b1c.sh" "$C"/b1_*.yaml "$PROF" "$FORMS" 2>/dev/null
    local f m p
    for f in "$COST" "$VERDICT" "$CURVE" "$ROWREP" "$ROWSTATS" "$RESULTS_ROOT/fidelity/student-$ST.corpus.json" "$RESULTS_ROOT/fidelity/student-$ST.queries.json" "$RESULTS_ROOT/fidelity/$KD.queries.json"; do [ -e "$f" ] && git add "$f" 2>/dev/null; done
    for m in "$KD" ${KD_PICK:-} "$RA" ${RA_PICK:-} "$RCD" ${RC_PICK:-} $(for k in $(seq 1 $NSEG); do seg "$k"; done) $(awk -F'\t' '{print $3}' "$CURVE" 2>/dev/null); do
        for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done
    done
    git add "$RESULTS_ROOT"/sparsity/kd-$ST-*.json "$RESULTS_ROOT"/sparsity/arm-b1-*.json 2>/dev/null
    git commit -q -m "$1" 2>/dev/null && log "committed $(git log --oneline -1)"
    push_loop b1c
}
RMODE=$(prof_get rank_mode)
[ -s "$LOGS/b1_rank_mode.txt" ] && { RMODE=$(first_line "$LOGS/b1_rank_mode.txt"); log "  ranking mode from logs/b1_rank_mode.txt: $RMODE"; }
[ -n "$RMODE" ] || { log "  no ranking mode in $PROF: b1c stops"; exit 1; }

# ---------------------------------------------------------------- 1. the steps of b1 (skipped where b1 did them)
log "=== b1c step 1: KD model alone, R-a (mode $RMODE), measure R-a ==="
form_fit "$KD"; KD_PICK=${PICK:-}
[ -n "$KD_PICK" ] && regime quantized "$KD_PICK" "models/$KD_PICK" tiny10 16
log "  KD alone: ${KD_PICK:-no form} tiny10 $(lbv "${KD_PICK:-none}" tiny10) at $(dims "${KD_PICK:-none}") dims/doc ($(dims "$KD") without a shift); the new ALBERT after KD alone: 64.03 at 554.8 dims/doc without a shift"
[ -s "$CURVE" ] || printf '0\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$KD" "${KD_PICK:-}" "$(lbv "${KD_PICK:-none}" tiny10)" "$(dims "${KD_PICK:-none}")" "$(qd "$FORMS/${KD_PICK:-none}.json" "${KD_PICK:-none}")" "$(pq "$FORMS/${KD_PICK:-none}.json" "${KD_PICK:-none}")" "$(dims "$KD")" "KD alone" > "$CURVE"
pause_point
# The int8 diagnosis of the released ALBERT, second run (a release matter, so before the long training). The first run
# (15.11 14:21, in the pause) showed the cause (the extra head layer, group.4: alone in int8 cosine 0.73, back in fp32
# 0.993) and then stopped on a quantizer assertion in the single-node trials; the tool now survives that and repairs such
# a group as a whole. CPU only; then tiny10 with the repaired int8 queries over the torch index. The package is not touched.
OD="runs/onnx-int8-diag/albert-$ALB"; OLABEL="onnx-albert-$ALB-q-mixed-int8-repaired"
if [ -n "$ALB" ] && [ -f "datasets/transfer/release-20261113/onnx/albert-$ALB/fp32/model.onnx" ]; then
    if [ ! -f "$OD/mixed-int8-repaired/model.onnx" ]; then
        wait_gpu_free; wait_ram; s=$SECONDS
        CUDA_VISIBLE_DEVICES= ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2700 "$PY" -m release.int8_sensitivity --fp32-dir "datasets/transfer/release-20261113/onnx/albert-$ALB/fp32" \
            --texts datasets/kd/kd_v1.jsonl --out "$OD" --weight-type u8 --target-cosine 0.995 --threads 6 > "$LOGS/onnx_albert_int8_diag2.log" 2>&1
        log "  int8 diagnosis of the ALBERT, second run: exit $? in $((SECONDS - s))s (logs/onnx_albert_int8_diag2.log, report $OD/int8_sensitivity.json)"
        grep -E "release form|repair [0-9]+:|repaired variant|not reached|quantizer failed" "$LOGS/onnx_albert_int8_diag2.log" | cut -c1-260 | while read -r l; do log "    $l"; done
    fi
    if [ -f "$OD/mixed-int8-repaired/model.onnx" ]; then
        PIRB_QUERY_MODEL="$OD/mixed-int8-repaired" regime quantized "$OLABEL" "models/$ALB" tiny10 16
        log "  tiny10 with the repaired int8 queries of the ALBERT over the torch index: $(lbv "$OLABEL" tiny10) (released int8: 62.10, torch: 64.34)"
    else log "  no repaired int8 variant of the ALBERT was written: tiny10 skipped"; fi
fi
pause_point
train_rank "${RA#arm-}" "$C/b1_ra_dose3.yaml"
pause_point
if [ -d "models/$RA" ]; then
    form_fit "$RA"; RA_PICK=${PICK:-}
    [ -n "$RA_PICK" ] && measure "$RA_PICK"
fi
DT=$(delta "$(lbv "${RA_PICK:-none}" tiny10)" "$(lbv "$D3_PICK" tiny10)"); DH=$(delta "$(lbv "${RA_PICK:-none}" heldout8)" "$(lbv "$D3_PICK" heldout8)")
if [ -z "$RA_PICK" ] || [ "$DT" = "-" ] || [ "$DH" = "-" ]; then
    RA_NOTE="**R-a has no complete measurement (no model, no form within the limits, or a failed evaluation); the large stage runs regardless.**"
elif [ "$(gte "$DT" 0.2)" = 1 ] && [ "$(gte "$DH" 0.2)" = 1 ]; then
    RA_NOTE="**R-a leads dose3 by $DT on tiny10 and $DH on heldout8 (>= +0.2 on both): the 23 tasks were measured.**"
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$RA_PICK" "$ts"; done; M23="$RA_PICK"
elif [ "$(gte -0.3 "$DT")" = 1 ] && [ "$(gte -0.3 "$DH")" = 1 ]; then
    RA_NOTE="**R-a is clearly below dose3 ($DT on tiny10, $DH on heldout8; <= -0.3 on both). This does not stop the large stage (the user's decision of 15.11: R-a is the little-data reference). Diagnosis from the numbers above: query fidelity after KD $(fid "$RESULTS_ROOT/fidelity/$KD.queries.json" query) against the new ALBERT's $(fid "$SEXP/results/fidelity/$AKD.queries.json" query); density without a shift: KD model $(dims "$KD") dims/doc, R-a $(dims "$RA"); the KD model alone has tiny10 $(lbv "${KD_PICK:-none}" tiny10) (ALBERT after KD alone: 64.03); the fidelity curve of the KD is above.**"
else
    RA_NOTE="R-a against dose3: $DT on tiny10, $DH on heldout8 (neither >= +0.2 on both nor <= -0.3 on both)."
fi
log "  R-a ${RA_PICK:-none}: tiny10 $DT, heldout8 $DH against dose3; the large stage follows in any case"
index_cost "$COST" "$D3_PICK" "$FULL" ${RA_PICK:-}
write_summary "Add the b1 results: polish-roberta-base-v2 after KD from the frozen model, ranking version R-a"
pause_point

# ---------------------------------------------------------------- A. MS MARCO-pl rows
log "=== b1c step A: MS MARCO-pl rows (filter 23.5, exclusions, units of $UNIT rows, mining and ensemble scores as in dose3) ==="
if [ ! -s "$MS/rows.json" ]; then
    wait_ram; s=$SECONDS
    FILT=cache/filtered_datasets/splade_positive_score_gt_23_50_no_metricx-d80474a3a8f41216/dataset.jsonl     # the filter's cached output of 24.10 (388 699 -> 317 668 rows)
    [ -s "$FILT" ] || FILT=$(timeout -k 60 3600 "$PY" -c "
from training.dataset_filters.core import materialize_filtered_dataset as m
print(m('$MSSRC', '$PROFILE').output_path)" 2> "$LOGS/b1c_filter.log" | tail -n 1)
    [ -s "$FILT" ] || { log "  the filtered MS MARCO-pl rows are missing (logs/b1c_filter.log): b1c stops before step A"; write_summary "Add the b1 results: R-a (the MS MARCO-pl rows could not be built)"; exit 1; }
    timeout -k 60 3600 nice -n 5 "$PY" "$KITCHEN/bridge/msmarco_dose_rows.py" --rows "$FILT" --filter-report "$(dirname "$FILT")/filter_report.json" --benchmark-queries "$REFQ" \
        --exclude-rows "$D3ROWS" --unit "$UNIT" --seed 42 --out-dir "$UNITS" --report "$MS/rows.json" > "$LOGS/b1c_rows.log" 2>&1
    log "  rows in $((SECONDS - s))s: $(tail -n 1 "$LOGS/b1c_rows.log" | cut -c1-500)"
fi
[ -s "$MS/rows.json" ] && cp "$MS/rows.json" "$ROWREP"
NU=$(ls "$UNITS"/unit-*.jsonl 2>/dev/null | wc -l)
[ "$NU" -ge $((NSEG * SEG_UNITS)) ] || { log "  only $NU units of rows (logs/b1c_rows.log): fewer than the $((NSEG * SEG_UNITS)) that stage 1 needs, b1c stops"; write_summary "Add the b1 results: R-a (the MS MARCO-pl rows could not be built)"; exit 1; }
log "  MS MARCO-pl: $(rget rows_before_filter) rows before the filter, $(rget rows_after_filter) after it, $(rget eligible_rows) after the exclusions; $NU units of $UNIT rows"
mine() {  # mine TAG ROWS OUT CACHE [args]: r2v12 mines inside the positives of ROWS, the ensemble teacher scores with the fixed scale
    local tag="$1" rows="$2" out="$3" cache="$4"; shift 4
    [ -s "$out" ] && [ -s "$out.mined.json" ] && return 0
    kscript "rc_$tag" 7200 "$out" "$PY" "$KITCHEN/bridge/splade_mine_and_score.py" --student "models/$R2" --splade-teacher models/soup-v9b \
        --dense-teacher "$PD" --rows "$rows" --dense-scale "$SCALE" --dense-cache "$cache" --output "$out" "$@"
}
if [ ! -s "$MINED/unit-01.jsonl" ]; then
    mkdir -p "$MS/smoke"
    if mine smoke "$UNITS/unit-01.jsonl" "$MS/smoke/rows-$(date +%H%M%S).jsonl" "$MS/smoke/dense" --limit 300; then log "  smoke of the mining passed (300 rows)"
    else log "  smoke of the mining FAILED (logs/rc_smoke.log): b1c stops before the long steps"; write_summary "Add the b1 results: R-a (the mining of the MS MARCO-pl rows failed)"; exit 1; fi
fi
for u in $(ls "$UNITS"/unit-*.jsonl); do
    n=$(basename "$u" .jsonl); out="$MINED/$n.jsonl"
    if mine "$n" "$u" "$out" "$MS/dense-pd400/$n"; then
        log "  $n: $(side "$out" rows) rows, pool $(side "$out" pools); teacher P(pos) median $(side "$out" teacher_positive_mass_median) (dose3 new rows 0.856), above 0.95: $(side "$out" teacher_positive_mass_above_095) (0.296); negative above positive in $(side "$out" rows_with_negative_above_positive) rows (dose3 new: 4168 of 40093); student ranks the positive first in $(side "$out" student_ranks_a_positive_first) (0.867); negatives $(side "$out" by_origin)"
    else log "  $n: mining FAILED (logs/rc_$n.log)"; fi
    pause_point
done
NM=$(ls "$MINED"/unit-*.jsonl 2>/dev/null | grep -c -v failed)
for k in $(seq 1 $NSEG); do
    [ -s "$SEGS/seg-$k.jsonl" ] && continue
    parts=(); for j in $(seq $(( (k - 1) * SEG_UNITS + 1 )) $(( k * SEG_UNITS ))); do parts+=("$MINED/unit-$(printf '%02d' "$j").jsonl"); done
    ok=1; for p in "${parts[@]}"; do [ -s "$p" ] || ok=0; done
    [ "$ok" = 1 ] && cat "${parts[@]}" > "$SEGS/seg-$k.jsonl.tmp" && mv "$SEGS/seg-$k.jsonl.tmp" "$SEGS/seg-$k.jsonl" && log "  segment $k: $(grep -c . "$SEGS/seg-$k.jsonl") rows ($SEGS/seg-$k.jsonl)" || log "  segment $k: a unit is missing, no row file"
done
if [ ! -s "$ROWSTATS" ]; then
    wait_ram
    timeout -k 60 3600 nice -n 5 "$PY" - "$ROWSTATS" "$D3ROWS" "$MINED" $((NSEG * SEG_UNITS)) > "$LOGS/b1c_rows_stats.log" 2>&1 <<'PYEOF'
import glob, json, math, statistics as st, sys
out, d3, mined, n_stage1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def new():
    return {"p": [], "flip": 0, "ratio": [], "q": [], "pos": [], "first": 0, "ranked": 0, "rows": 0}
def add(acc, r):
    pos, neg = r["pos_scores"], r["neg_scores"]
    s = [max(pos)] + list(neg); m = max(s); e = [math.exp((x - m) / 2.0) for x in s]
    acc["p"].append(e[0] / sum(e)); acc["flip"] += max(neg) > min(pos); acc["rows"] += 1
    pl = st.median(len(t) for t in r["pos"]); acc["pos"].append(pl); acc["q"].append(len(r["query"]))
    acc["ratio"].extend(len(t) / max(1.0, pl) for t in r["neg"])
    if r.get("pos_student_rank"):
        acc["ranked"] += 1; acc["first"] += min(r["pos_student_rank"]) == 1
def done(acc):
    ratio = sorted(acc["ratio"])
    return {"rows": acc["rows"], "p_pos_median": st.median(acc["p"]), "p_pos_above_095": sum(x > 0.95 for x in acc["p"]) / len(acc["p"]),
            "negative_above_positive": acc["flip"], "len_ratio_median": st.median(ratio), "len_ratio_p90": ratio[int(0.9 * (len(ratio) - 1))],
            "query_chars_median": st.median(acc["q"]), "positive_chars_median": st.median(acc["pos"]),
            "student_first": acc["first"] / acc["ranked"] if acc["ranked"] else None}
sets = {}
acc = new()
for line in open(d3, encoding="utf-8"):
    if line.strip():
        add(acc, json.loads(line))
sets["dose3 rows (60 385, the rows of R-a)"] = done(acc)
files = sorted(f for f in glob.glob(mined + "/unit-*.jsonl"))
s1, rest = new(), new()
for i, f in enumerate(files):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            add(s1 if i < n_stage1 else rest, json.loads(line))
if s1["rows"]:
    sets["MS MARCO-pl, stage 1 (units 1-%d)" % n_stage1] = done(s1)
if rest["rows"]:
    sets["MS MARCO-pl, the other units (for the 5070)"] = done(rest)
json.dump({"sets": sets, "units": len(files)}, open(out, "w"), indent=2)
for k, v in sets.items():
    print("%s: %d rows, P(pos) median %.3f, above 0.95 %.3f, negative above positive %d, length ratio median %.2f / p90 %.2f" % (
        k, v["rows"], v["p_pos_median"], v["p_pos_above_095"], v["negative_above_positive"], v["len_ratio_median"], v["len_ratio_p90"]))
PYEOF
    grep -v "^$" "$LOGS/b1c_rows_stats.log" | tail -n 4 | cut -c1-300 | while read -r l; do log "  $l"; done
fi
log "  package for the 5070 (NOT created; it needs the user's consent): KD model $(du -sh "models/$KD" | cut -f1), MS MARCO-pl rows with scores $(du -shc "$MINED"/unit-*.jsonl 2>/dev/null | tail -n 1 | cut -f1) in $NM units, dose3 rows $(du -sh "$D3ROWS" | cut -f1), configs and README; free: $(free_gb) GB on /, $(df -BG --output=avail "$DD" | tail -1 | tr -dc '0-9') GB on the data drive"
write_summary "Add the b1 results: the MS MARCO-pl rows of the large stage"
pause_point

# ---------------------------------------------------------------- B. stage 1 and the dose curve
log "=== b1c step B: stage 1, $NSEG segments of $((SEG_UNITS * UNIT)) MS MARCO-pl rows from $KD (mode $RMODE) ==="
for k in $(seq 1 $NSEG); do
    m=$(seg "$k")
    [ -s "$SEGS/seg-$k.jsonl" ] || { log "  no rows for segment $k: stage 1 ends after segment $((k - 1))"; break; }
    train_rank "${m#arm-}" "$C/b1_rc_s1p$k.yaml"
    [ -d "models/$m" ] || { log "  segment $k was not trained (logs/arm_${m#arm-}.log): stage 1 ends after segment $((k - 1))"; break; }
    to_dd "$m"; S1=$m
    T=$(tsec "$m"); [ -n "$T" ] && log "  segment $k trained in $T s ($("$PY" -c "print('%.2f s per optimiser step, %.1f h per segment' % ($T / ($(grep -c . "$SEGS/seg-$k.jsonl") / 32.0), $T / 3600.0))")); the remaining $((NSEG - k)) segment(s) need about $("$PY" -c "print('%.1f h' % ($T * ($NSEG - $k) / 3600.0))") of training"
    pause_point
    form_fit "$m"; P=${PICK:-}
    [ -n "$P" ] && regime quantized "$P" "models/$P" tiny10 16
    grep -q -P "^\d+\t$m\t" "$CURVE" 2>/dev/null || printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$((k * SEG_UNITS * UNIT))" "$m" "${P:-}" "$(lbv "${P:-none}" tiny10)" "$(dims "${P:-none}")" "$(qd "$FORMS/${P:-none}.json" "${P:-none}")" "$(pq "$FORMS/${P:-none}.json" "${P:-none}")" "$(dims "$m")" "${T:-?} s" >> "$CURVE"
    log "  DOSE CURVE, $((k * SEG_UNITS * UNIT)) examples: ${P:-no form} tiny10 $(lbv "${P:-none}" tiny10) at $(dims "${P:-none}") dims/doc, $(pq "$FORMS/${P:-none}.json" "${P:-none}") postings/query ($(dims "$m") dims/doc without a shift); KD alone $(lbv "${KD_PICK:-none}" tiny10), R-a $(lbv "${RA_PICK:-none}" tiny10), dose3 $(lbv "$D3_PICK" tiny10)"
    S1_PICK=${P:-}
    pause_point
done
CURVE_NOTE=$("$PY" - "$CURVE" <<'PYEOF' 2>> "$LOGS/queue_summary.log"
import sys
pts = []
for line in open(sys.argv[1], encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if f[0] != "0" and len(f) > 3 and f[3]:
        pts.append((int(f[0]), float(f[3])))
if len(pts) < 3:
    print("Reading of the curve: fewer than three measured points, no reading.")
else:
    d = [(pts[i][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))]
    last, prev = d[-1][1], d[-2][1]
    steps = ", ".join("%+.2f up to %d thousand" % (x, n // 1000) for n, x in d)
    if last >= 0.15:
        word = "tiny10 is STILL RISING on the last segment (>= +0.15): a large dose on the 5070 is proposed (the decision is the user's)"
    elif last < 0.1 and prev < 0.1:
        word = "the curve has FLATTENED (below +0.1 on the last two segments): a larger dose is unlikely to add anything"
    else:
        word = "neither clearly rising (>= +0.15 on the last segment) nor flat (below +0.1 on the last two): one more segment or the 5070 dose is the user's call"
    print("Reading of the curve (steps of tiny10: %s; one seed, seed noise 0.13-0.2): %s." % (steps, word))
PYEOF
)
log "  ${CURVE_NOTE:-no curve reading}"
if [ -n "$S1_PICK" ]; then measure "$S1_PICK"; fi
ST1T=$(delta "$(lbv "${S1_PICK:-none}" tiny10)" "$(lbv "$D3_PICK" tiny10)"); ST1H=$(delta "$(lbv "${S1_PICK:-none}" heldout8)" "$(lbv "$D3_PICK" heldout8)")
RC_NOTE="R-c' after stage 1 (${S1:-no model}) against dose3: $ST1T on tiny10, $ST1H on heldout8."
mkdir -p "$LOGS/b1-aside"; [ -f "$COST" ] && mv "$COST" "$LOGS/b1-aside/index_cost_b1.after-ra.$(date +%H%M%S).json"      # measured again with the new forms in it
index_cost "$COST" "$D3_PICK" "$FULL" ${RA_PICK:-} ${S1_PICK:-}
write_summary "Add the b1 results: stage 1 of R-c' on MS MARCO-pl rows and its dose curve"
pause_point

# ---------------------------------------------------------------- C. the dose3 round
log "=== b1c step C: the dose3 round from the end of stage 1 (mode $RMODE) ==="
if [ "$S1" = "$(seg "$NSEG")" ]; then
    train_rank "${RCD#arm-}" "$C/b1_rc_d3.yaml"
    pause_point
    if [ -d "models/$RCD" ]; then
        to_dd "$RCD"
        form_fit "$RCD"; RC_PICK=${PICK:-}
        [ -n "$RC_PICK" ] && measure "$RC_PICK"
        CT=$(delta "$(lbv "${RC_PICK:-none}" tiny10)" "$(lbv "$D3_PICK" tiny10)"); CH=$(delta "$(lbv "${RC_PICK:-none}" heldout8)" "$(lbv "$D3_PICK" heldout8)")
        if [ -n "$RC_PICK" ] && [ "$CT" != "-" ] && [ "$CH" != "-" ] && [ "$(gte "$CT" 0.2)" = 1 ] && [ "$(gte "$CH" 0.2)" = 1 ]; then
            for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$RC_PICK" "$ts"; done; M23="$M23 $RC_PICK"
            RC_NOTE="$RC_NOTE **R-c' after the dose3 round leads dose3 by $CT on tiny10 and $CH on heldout8 (>= +0.2 on both): the 23 tasks were measured.**"
        else RC_NOTE="$RC_NOTE R-c' after the dose3 round against dose3: $CT on tiny10, $CH on heldout8 (threshold: >= +0.2 on both)."; fi
        log "  R-c' ${RC_PICK:-none}: tiny10 $CT, heldout8 $CH against dose3"
    else log "  the dose3 round was not trained (logs/arm_${RCD#arm-}.log)"; RC_NOTE="$RC_NOTE The dose3 round was not trained."; fi
else
    log "  stage 1 did not reach segment $NSEG (last model: ${S1:-none}): the dose3 round is not started"; RC_NOTE="$RC_NOTE Stage 1 did not finish; the dose3 round was not started."
fi
[ -f "$COST" ] && mv "$COST" "$LOGS/b1-aside/index_cost_b1.after-stage1.$(date +%H%M%S).json"
index_cost "$COST" "$D3_PICK" "$FULL" ${RA_PICK:-} ${S1_PICK:-} ${RC_PICK:-}
write_summary "Add the b1 results: R-c', the large MS MARCO-pl stage and the dose3 round of polish-roberta-base-v2"
log "=== b1c done ($(date '+%d.%m %H:%M')); R-a ${RA_PICK:-none} $(lbs "${RA_PICK:-none}"), R-c' stage 1 ${S1_PICK:-none} $(lbs "${S1_PICK:-none}"), R-c' ${RC_PICK:-none} $(lbs "${RC_PICK:-none}"); dose3 $(lbs "$D3_PICK") ==="
