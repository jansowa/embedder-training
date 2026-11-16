#!/usr/bin/env bash
# Queue b1 (3060 Ti, from the night of 14/15.11; ~2 days): a larger backbone. sdadas/polish-roberta-base-v2 (12 layers,
# 124 M; the same tokenizer as our 6-layer family) gets the SPLADE representation of the frozen full model by KD (the
# ALBERT recipe: token KL + pooled MSE on kd_v1) and is then trained for ranking on the same rows and with the same
# teacher scores as the full model. Question (the user's, 14.11): with the same data and the same teacher, does the
# larger model beat our best (dose3 64.96 / 43.63 / 72.82; frozen soup-v20-dqf-bias015)? The release of 26.11 does not
# depend on the answer.
#   0  PRECONDITION  tokenizer identity (files and token ids of 5 000 kd_v1 texts; the queue stops when it differs), the
#                    student as a SparseEncoder (MLMTransformer with the model's own MLM head + SpladePooling max),
#                    fidelity before KD, memory profile of KD (256 tokens, batch 16) and of the ranking recipe (512
#                    tokens, 2 x 16, fused head) on the 8 GB card: plain, else gradient checkpointing, else 1 x 32
#   1  KD            the recipe of queue_s3.sh / a1 (token KL x 0.3 + pooled MSE, all of kd_v1, one epoch, lr 5e-5,
#                    warmup 300, linear, fp16, 256 tokens, batch 16, seed 42), teacher = the raw weights of the frozen
#                    model; query fidelity next to the new ALBERT; tiny10 of the KD model in its first form within both
#                    limits (<= 410 dims/doc, <= 16 807 postings/query)
#   2  R-a           from the KD model, the rows and the recipe of dose3 unchanged
#   3  MEASURE R-a   the first form within both limits: tiny10, heldout8, trec-covid, cost, the per-task table against
#                    dose3; 23 tasks when it leads dose3 by >= 0.2 on both sets
#   4  R-b           (not when R-a is <= -0.3 below dose3 on both sets) stage 3 on n5 + kj with the recipe of r2v12,
#                    then round 2 on the dose3 rows; the same measurement
# R-c (more MS MARCO rows) is not started here: it needs numbers and the user's decision first.
# Between steps the queue waits while logs/b1.pause exists (the card is the user's for fixes of the release package).
# No deletions: bias forms are moved to the data drive and linked back, profile runs stay under runs/.
#   bash logs/relaunch-20261113/launch_queue.sh b1 experiments/polish-splade-2026-10/queues/queue_b1.sh "a1b"
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=b1 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export KD_TIMEOUT=${KD_TIMEOUT:-64800} TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-79200}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
source "$EXP_ROOT/queues/release_common.sh"
echo $$ > "$LOGS/queue_b1.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
wait_for_queues ${WAIT_FOR:-a1b}

CORPUS=datasets/kd/kd_v1.jsonl
BASE=models/polish-roberta-base-v2
FULL=$(final_full); [ -n "$FULL" ] && [ -d "models/$FULL" ] || { log "  no frozen full model in $FINAL_FULL_FILE: b1 stops"; exit 1; }
RAW=${FULL%-bias[0-9][0-9][0-9]}
TEACHER="models/$RAW"                                    # read by fidelity() of queue_lib.sh
D3_PICK=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12-bias015
ALB=$(final_albert)
ST=rbase-v2; KD=kd-$ST-dqf-full                          # models/students/$ST, models/$KD
AKD=kd-b4-l0125-f1536-hl-r256-w-dqf-full                 # the KD model of the new ALBERT (small series)
RA=arm-b1-ra; RBS=arm-b1-rb-s3; RB=arm-b1-rb
MAX_POSTINGS=${MAX_POSTINGS:-16807}; MAX_DIMS=410; VRAM_OK=7300
FORMS="$RESULTS_ROOT/index_cost_b1_forms"; COST="$RESULTS_ROOT/index_cost_b1.json"; VERDICT="$RESULTS_ROOT/b1_reading.json"
PROF="$RESULTS_ROOT/b1_profile.tsv"; SUM="$RESULTS_ROOT/summary_b1.md"
DD="$(dirname "$(readlink -f runs)")/FlagEmbedding-models-b1"      # bias forms live on the data drive
mkdir -p "$FORMS" "$RESULTS_ROOT/fidelity" "$DD" || { log "  cannot create $DD: b1 stops"; exit 1; }
log "=== b1 start: $BASE after KD from $TEACHER, then ranking versions against dose3 ($(lbs "$D3_PICK") / trec-covid $(tc "$D3_PICK")); frozen model $FULL; ALBERT ${ALB:-none}; $(free_gb) GB free on /, $(df -BG --output=avail "$DD" | tail -1 | tr -dc '0-9') GB on the data drive ==="
for f in "$BASE/model.safetensors" "$BASE/config.json" "$BASE/tokenizer.json" "$TEACHER/model.safetensors" "$CORPUS" datasets/arms/n5-plus-kj-mined-dose3.jsonl datasets/arms/n5-plus-kj-tens400.jsonl \
         "$C/b1_ra_dose3.yaml" "$C/b1_rb_s3_n5kj.yaml" "$C/b1_rb_r2_dose3.yaml" scripts/arm_reading.py scripts/merge_regime_task.py; do
    [ -s "$f" ] || { log "  $f missing: b1 stops"; exit 1; }
done
[ -d "models/$D3_PICK" ] || { log "  models/$D3_PICK missing: b1 stops"; exit 1; }
need_space || { log "  not enough space on /: b1 stops"; exit 1; }

pause_point() { local said=0; while [ -e "$LOGS/b1.pause" ]; do [ "$said" = 0 ] && { log "  paused by logs/b1.pause (remove the file to continue)"; said=1; }; sleep 60; done; }
slbv() { regime_mean "$SEXP/results-regimes/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }      # ALBERT results live in the small series
sdims() { sparsity_mean "$SEXP/results/sparsity/$1.json" "$1" 2>/dev/null; }
fid() {  # fid FILE KIND: one line of a distill.fidelity JSON
    "$PY" -c "import json,sys;k=json.load(open(sys.argv[1]))['kinds'][sys.argv[2]];m=lambda n:k[n]['mean'];print('cosine %.4f, recall of the teacher dims %.3f, Jaccard %.3f, L1 rel. error %.3f, dims teacher / student %.0f / %.0f (n=%d)' % (m('cosine'), m('teacher_active_recall'), m('jaccard_active'), m('l1_relative_error'), m('teacher_active_dims'), m('student_active_dims'), k['n']))" "$1" "$2" 2>/dev/null || echo "not measured"
}
fidrow() {  # fidrow NAME FILE KIND: a table row
    "$PY" -c "import json,sys;k=json.load(open(sys.argv[2]))['kinds'][sys.argv[3]];m=lambda n:k[n]['mean'];print('| %s | %d | %.4f | %.3f | %.3f | %.3f | %.0f | %.0f |' % (sys.argv[1], k['n'], m('cosine'), m('teacher_active_recall'), m('jaccard_active'), m('l1_relative_error'), m('teacher_active_dims'), m('student_active_dims')))" "$1" "$2" "$3" 2>/dev/null || echo "| $1 | - | - | - | - | - | - | - |"
}
ntasks() { regime_mean "$REGIMES_ROOT/quantized/$1/heldout8.json" 2>/dev/null | awk '{print $2}'; }
fix8() {  # fix8 LABEL: when heldout8 came back with 7 tasks (fiqa-pl failed), run fiqa-pl alone and fold it in
    [ "$(ntasks "$1")" = 7 ] || return 0
    log "  $1: heldout8 has 7 tasks; fiqa-pl alone"
    regime quantized "$1" "models/$1" heldout8_fiqa 8 && "$PY" scripts/merge_regime_task.py --into "$REGIMES_ROOT/quantized/$1/heldout8.json" \
        --from "$REGIMES_ROOT/quantized/$1/heldout8_fiqa.json" --n-tasks 8 | while read -r l; do log "  $l"; done
}
partial_aside() {  # partial_aside LABEL: a heldout8 file with fewer than 7 tasks is moved out of the results
    local f="$REGIMES_ROOT/quantized/$1/heldout8.json" n; [ -f "$f" ] || return 0
    n=$(ntasks "$1"); [ "${n:-0}" -lt 7 ] || return 0
    mkdir -p "$LOGS/b1-aside"; mv "$f" "$LOGS/b1-aside/heldout8-partial-$1-$(date +%H%M%S).json"; log "  $1: heldout8 with ${n:-0} of 8 tasks moved to logs/b1-aside (not a result)"
}
h8() {  # h8 LABEL: heldout8 with all 8 tasks; one repeat when the file is partial
    regime quantized "$1" "models/$1" heldout8 16; fix8 "$1"
    if [ "$(ntasks "$1")" != 8 ]; then partial_aside "$1"; regime quantized "$1" "models/$1" heldout8 8; fix8 "$1"; fi
    [ "$(ntasks "$1")" = 8 ] || { log "  $1: heldout8 is not complete after a repeat"; partial_aside "$1"; return 1; }
}
pq() { "$PY" -c "import json,sys;print('%.0f' % json.load(open(sys.argv[1]))['models'][sys.argv[2]]['postings_per_query'])" "$1" "$2" 2>/dev/null; }
qd() { "$PY" -c "import json,sys;print('%.1f' % json.load(open(sys.argv[1]))['models'][sys.argv[2]]['query_active_mean'])" "$1" "$2" 2>/dev/null; }
to_dd() {  # to_dd LABEL: move the bias form models/LABEL to the data drive and link it back (space on /)
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
vram_watch() { local max=0 v; echo 0 > "$1"; while :; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'); [ "${v:-0}" -gt "$max" ] && { max=$v; echo "$max" > "$1"; }; sleep 1; done; }
mkcfg() {  # mkcfg BASE_YAML MODE [MAX_STEPS RUNS_DIR START]: echo the config to use (BASE for plain, else a derived copy next to it)
    local base="$1" mode="$2" steps="${3:-}" runs="${4:-}" start="${5:-}" out
    out="${base%.yaml}"; [ "$mode" != plain ] && out="${out}_$mode"; [ -n "$steps" ] && out="${out}_profile"; out="$out.yaml"
    [ "$out" = "$base" ] && { echo "$base"; return 0; }
    "$PY" - "$base" "$out" "$mode" "$steps" "$runs" "$start" <<'PYEOF' || return 1
import re, sys
base, out, mode, steps, runs, start = sys.argv[1:7]
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
if steps:
    sub(r"^    num_train_epochs: 2$", "    num_train_epochs: 2\n    max_steps: %d" % int(steps))
    sub(r"^runs_dir: .*$", "runs_dir: " + runs)
    sub(r"^  - models/.*$", "  - " + start)
note = {"gc": "gradient checkpointing on", "b1gc": "physical batch 1 x 32 accumulation and gradient checkpointing (the recipe did not fit 8 GB)", "plain": "the recipe unchanged"}[mode]
s = s.replace("backend: sentence-transformers\n", "backend: sentence-transformers\n# derived by queue_b1.sh from %s: %s%s\n" % (base.split("/")[-1], note, "; memory profile, %s optimiser steps" % steps if steps else ""), 1)
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

# ---------------------------------------------------------------- 0. precondition
log "=== b1 step 0: tokenizer, student, fidelity before KD, memory profile ==="
TOK=$("$PY" - "$BASE" "$TEACHER" "$CORPUS" <<'PYEOF' 2> "$LOGS/b1_tokenizer.log" | tail -n 1
import json, sys
from tokenizers import Tokenizer
base, teacher, corpus = sys.argv[1:4]
a, b = json.load(open(base + "/tokenizer.json")), json.load(open(teacher + "/tokenizer.json"))
same_vocab = a["model"]["vocab"] == b["model"]["vocab"] and a["model"]["type"] == b["model"]["type"]
same_added = {t["content"]: t["id"] for t in a["added_tokens"]} == {t["content"]: t["id"] for t in b["added_tokens"]}
ta, tb = Tokenizer.from_file(base + "/tokenizer.json"), Tokenizer.from_file(teacher + "/tokenizer.json")
for t in (ta, tb):
    t.no_truncation(); t.no_padding()
texts = []
with open(corpus, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i % 450 == 0:
            texts.append(json.loads(line)["text"])
        if len(texts) >= 5000:
            break
diff = sum(1 for x, y in zip(ta.encode_batch(texts), tb.encode_batch(texts)) if x.ids != y.ids)
ok = same_vocab and same_added and diff == 0
print("%s vocab %d, mapping identical %s, special tokens identical %s, texts with different ids %d of %d" % ("OK" if ok else "DIFFERENT", len(a["model"]["vocab"]), same_vocab, same_added, diff, len(texts)))
PYEOF
)
log "  tokenizer of $BASE against $TEACHER: $TOK"
case "$TOK" in OK*) ;; *) log "  THE TOKENIZER DIFFERS (or the check failed): KD on vectors is not possible, b1 stops"; record "b1:tokenizer" FAILED "-" "-"; exit 1;; esac
record "b1:tokenizer" ok "-" "-"
if [ ! -d "models/students/$ST" ]; then
    mkdir -p "models/students/$ST.tmp" && cp "$BASE/config.json" "$BASE/model.safetensors" "models/students/$ST.tmp/" \
        && cp -r "$TEACHER/tokenizer.json" "$TEACHER/tokenizer_config.json" "$TEACHER/special_tokens_map.json" "$TEACHER/modules.json" "$TEACHER/sentence_bert_config.json" "$TEACHER/config_sentence_transformers.json" "$TEACHER/1_SpladePooling" "models/students/$ST.tmp/" \
        && mv "models/students/$ST.tmp" "models/students/$ST" || { log "  assembling models/students/$ST FAILED: b1 stops"; exit 1; }
    log "  student models/students/$ST: weights and config of $BASE, tokenizer and SparseEncoder files of $TEACHER (MLMTransformer + SpladePooling max)"
fi
"$PY" -c "
import json,sys
a=json.load(open(sys.argv[1]+'/config.json')); b=json.load(open(sys.argv[2]+'/config.json'))
d={k:(a.get(k),b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k) and k not in ('_name_or_path','transformers_version','torch_dtype','dtype')}
print('config differences student | teacher: '+json.dumps(d))" "models/students/$ST" "$TEACHER" 2>&1 | while read -r l; do log "  $l"; done
fidelity "student-$ST.corpus" "models/students/$ST" "$CORPUS" --n 256
fidelity "student-$ST.queries" "models/students/$ST" "$CORPUS" --kinds query --n 512
log "  before KD, 256 corpus texts: $(fid "$RESULTS_ROOT/fidelity/student-$ST.corpus.json" all)"
log "  before KD, 512 queries: $(fid "$RESULTS_ROOT/fidelity/student-$ST.queries.json" query) (new ALBERT before KD: cosine 0.5481)"
[ -f "$RESULTS_ROOT/fidelity/student-$ST.queries.json" ] || { log "  the student does not load as an MLM model (logs/fidelity_student-$ST.queries.log): b1 stops"; exit 1; }

KD_COMMON=(--teacher "$TEACHER" --corpus "$CORPUS" --max-seq-length 256 --batch-size 16 --lr 5e-5 --warmup-steps 300 --schedule linear --precision fp16 --head-chunk 64 --log-every 50)
KD_LOSS=(--token-loss kl --token-weight 0.3 --temperature 2.0 --top-k 128 --pooled-loss mse --pooled-weight 1.0)
if [ -z "$(prof_get kd_mode)" ] && [ ! -d "models/$KD" ]; then
    for mode in plain gc; do
        extra=(); [ "$mode" = gc ] && extra=(--gradient-checkpointing)
        wait_gpu_free; wait_ram; s=$SECONDS
        vram_watch "$LOGS/b1_vram.txt" & W=$!
        timeout -k 60 3600 "$PY" -m distill.train_kd --student "models/students/$ST" --out "runs/kd-b1-profile-$mode" "${KD_COMMON[@]}" "${KD_LOSS[@]}" --epochs 1 --max-items 40000 --max-steps 200 --seed 42 "${extra[@]}" > "$LOGS/kd_b1-profile-$mode.log" 2>&1
        rc=$?; kill "$W" 2>/dev/null; wait "$W" 2>/dev/null
        peak=$(cat "$LOGS/b1_vram.txt" 2>/dev/null); eps=$(grep -o "examples_per_s=[0-9.]*" "$LOGS/kd_b1-profile-$mode.log" | tail -n 1 | cut -d= -f2)
        oom=$(grep -c -i "out of memory" "$LOGS/kd_b1-profile-$mode.log")
        log "  KD profile ($mode; 256 tokens, batch 16, 200 steps): exit $rc, peak ${peak:-?} MiB on the card, ${eps:-?} examples/s, out-of-memory lines $oom, $((SECONDS - s)) s"
        printf 'kd_%s\texit %s, peak %s MiB, %s examples/s, OOM lines %s\n' "$mode" "$rc" "${peak:-?}" "${eps:-?}" "$oom" >> "$PROF"
        if [ "$rc" = 0 ] && [ "$oom" = 0 ] && [ -n "$eps" ] && [ "${peak:-99999}" -le "$VRAM_OK" ]; then
            printf 'kd_mode\t%s\nkd_eps\t%s\nkd_peak\t%s\n' "$mode" "$eps" "$peak" >> "$PROF"; break
        fi
    done
    [ -n "$(prof_get kd_mode)" ] || { log "  KD does not fit the card in either mode (logs/kd_b1-profile-*.log): b1 stops"; exit 1; }
    log "  KD mode: $(prof_get kd_mode); estimate for the whole corpus: $("$PY" -c "print('%.1f h' % (2264676 / float('$(prof_get kd_eps)') / 3600))" 2>/dev/null)"
fi
if [ -z "$(prof_get rank_mode)" ]; then
    for mode in plain gc b1gc; do
        cfg=$(mkcfg "$C/b1_ra_dose3.yaml" "$mode" 40 "runs/b1-profile-rank-$mode" "models/students/$ST") || { log "  profile config ($mode) FAILED"; continue; }
        wait_gpu_free; wait_ram; s=$SECONDS
        vram_watch "$LOGS/b1_vram.txt" & W=$!
        timeout -k 60 5400 "$PY" run_experiments.py --grid-configuration-file "$cfg" --backend sentence-transformers --training-type splade > "$LOGS/b1_profile_rank_$mode.log" 2>&1
        rc=$?; kill "$W" 2>/dev/null; wait "$W" 2>/dev/null
        peak=$(cat "$LOGS/b1_vram.txt" 2>/dev/null); oom=$(grep -c -i "out of memory" "$LOGS/b1_profile_rank_$mode.log")
        ck=$(ls -1dt "runs/b1-profile-rank-$mode/sentence-transformers/splade"/*/final 2>/dev/null | head -1)
        e=$((SECONDS - s))
        log "  ranking profile ($mode; the dose3 recipe, 512 tokens, 40 optimiser steps from the student before KD): exit $rc, final checkpoint ${ck:+yes}${ck:-no}, peak ${peak:-?} MiB on the card, out-of-memory lines $oom, $e s"
        printf 'rank_%s\texit %s, checkpoint %s, peak %s MiB, OOM lines %s, %s s for 40 steps\n' "$mode" "$rc" "${ck:+yes}" "${peak:-?}" "$oom" "$e" >> "$PROF"
        if [ "$rc" = 0 ] && [ -n "$ck" ] && [ "$oom" = 0 ] && [ "${peak:-99999}" -le "$VRAM_OK" ]; then
            printf 'rank_mode\t%s\nrank_peak\t%s\nrank_s40\t%s\n' "$mode" "$peak" "$e" >> "$PROF"; break
        fi
    done
    [ -n "$(prof_get rank_mode)" ] && log "  ranking mode: $(prof_get rank_mode) (40 steps took $(prof_get rank_s40) s with loading; dose3 has 3 774 optimiser steps)" \
        || log "  NO RANKING MODE FITS THE CARD (logs/b1_profile_rank_*.log): KD runs, the ranking steps will be skipped unless logs/b1_rank_mode.txt names a mode"
fi
pause_point

# ---------------------------------------------------------------- 1. KD
log "=== b1 step 1: KD of $ST on the whole corpus, teacher $TEACHER (mode $(prof_get kd_mode)) ==="
extra=(); [ "$(prof_get kd_mode)" = gc ] && extra=(--gradient-checkpointing)
kd "${KD#kd-}" "models/students/$ST" "${KD_COMMON[@]}" "${KD_LOSS[@]}" --epochs 1 --save-every 6000 --keep-last 2 --fidelity-texts "$CORPUS" --fidelity-n 256 --seed 42 "${extra[@]}"
[ -d "models/$KD" ] || { log "  no KD model (logs/kd_${KD#kd-}.log): b1 stops"; exit 1; }
fidelity "$KD.queries" "models/$KD" "$CORPUS" --kinds query --n 512
log "  after KD, 512 queries: $(fid "$RESULTS_ROOT/fidelity/$KD.queries.json" query)"
log "  after KD, 512 queries, the new ALBERT: $(fid "$SEXP/results/fidelity/$AKD.queries.json" query)"
pause_point
form_fit "$KD"; KD_PICK=${PICK:-}
[ -n "$KD_PICK" ] && regime quantized "$KD_PICK" "models/$KD_PICK" tiny10 16
log "  KD alone: ${KD_PICK:-no form} tiny10 $(lbv "${KD_PICK:-none}" tiny10) at $(dims "${KD_PICK:-none}") dims/doc ($(dims "$KD") without a shift); the new ALBERT after KD alone: 64.03 at 554.8 dims/doc without a shift"

write_summary() {  # write_summary: the state so far (called after R-a and after R-b)
    local rm="${RMODE:-$(prof_get rank_mode)}"
    row() {  # row NAME LABEL COSTFILE
        [ -n "$2" ] || return 0
        echo "| $1 | \`$2\` | $(lbv "$2" tiny10) | $(lbv "$2" heldout8) | $(tc "$2") | $(delta "$(lbv "$2" tiny10)" "$(lbv "$D3_PICK" tiny10)") | $(delta "$(lbv "$2" heldout8)" "$(lbv "$D3_PICK" heldout8)") | $(dims "$2") | $(qd "$3" "$2") | $(pq "$3" "$2") |"
    }
    {
        echo "# b1 (quantized regime, nDCG@10): a larger backbone after KD from the frozen full model"; echo
        echo "Student: \`sdadas/polish-roberta-base-v2\` (12 layers, 124 M) as a SparseEncoder (MLMTransformer with the model's own MLM head + SpladePooling max). KD: the recipe of the ALBERT student (token KL x 0.3 + pooled MSE on all of kd_v1, one epoch, lr 5e-5, warmup 300, linear, fp16, 256 tokens, batch 16, seed 42), teacher = the raw weights of the frozen full model (\`$TEACHER\`). Ranking versions start from the KD model: **R-a** = the rows and the recipe of dose3 unchanged (only the start model differs from dose3); **R-b** = stage 3 on n5 + kj with the recipe of r2v12, then round 2 on the dose3 rows. Every model is measured in its first form (no shift, then bias steps of 0.05) with at most $MAX_DIMS dims/doc and at most $MAX_POSTINGS postings/query."; echo
        echo "## Precondition"; echo
        echo "- Tokenizer against the frozen model: $TOK."
        echo "- Memory profile on the 8 GB card (peak = memory used on the whole card; a mode is accepted up to $VRAM_OK MiB without an out-of-memory line):"; echo; echo '```'; cat "$PROF" 2>/dev/null; echo '```'; echo
        echo "- The first profile attempt (15.11 01:11) ran out of memory in both KD modes: the student before KD activates about 46 800 dimensions per text, so the backward of the sparse decoder asked for entries x hidden floats in one piece (2.3 GiB). It now goes in slices of 32 768 entries (the same sums, \`training/backends/splade_fused_head.py\`); the logs of that attempt are in \`logs/b1-aside/\`."
        echo "- KD mode: $(prof_get kd_mode). Ranking mode: ${rm:-none fits} ($([ "$rm" = plain ] && echo "the dose3 recipe unchanged: 2 x 16, 512 tokens")$([ "$rm" = gc ] && echo "the dose3 recipe with gradient checkpointing; batch 2 x 16 and 512 tokens unchanged")$([ "$rm" = b1gc ] && echo "physical batch 1 x 32 accumulation with gradient checkpointing: NOT the dose3 recipe one to one, the regulariser weights act on a physical batch of 1")). Training time: $(grep -E "^(kd:${KD#kd-}|train:$RA|train:$RBS|train:$RB)[[:space:]]" "$STATE" | awk -F'\t' '{printf "%s %s; ", $1, $4}')"; echo
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
        row "R-b: stage 3 on n5 + kj, then round 2 on dose3 (124 M)" "${RB_PICK:-}" "$COST"
        [ -n "$ALB" ] && echo "| ALBERT (50 M, a1) | \`$ALB\` | $(slbv "$ALB" tiny10) | $(slbv "$ALB" heldout8) | $(slbv "$ALB" heldout2_treccovid) | $(delta "$(slbv "$ALB" tiny10)" "$(lbv "$D3_PICK" tiny10)") | $(delta "$(slbv "$ALB" heldout8)" "$(lbv "$D3_PICK" heldout8)") | $(sdims "$ALB") | $(qd "$SEXP/results/index_cost_a1.json" "$ALB") | $(pq "$SEXP/results/index_cost_a1.json" "$ALB") |"
        echo
        echo "Unbiased dims/doc: KD model $(dims "$KD"), R-a $(dims "$RA"), R-b $(dims "$RB"). One seed each; the seed noise of the ranking recipe is 0.13-0.2."; echo
        [ -n "${RA_NOTE:-}" ] && { echo "$RA_NOTE"; echo; }
        local pick="${RB_PICK:-${RA_PICK:-}}" cols=(--column "dose3 (v56)=$D3_PICK" --column "frozen full model=$FULL")
        [ -n "${RA_PICK:-}" ] && cols+=(--column "R-a=$RA_PICK"); [ -n "${RB_PICK:-}" ] && cols+=(--column "R-b=$RB_PICK")
        local m23=(--m23 "$D3_PICK" --m23 "$FULL"); local x; for x in ${M23:-}; do m23+=(--m23 "$x"); done
        if [ -n "${RA_PICK:-}" ] && [ -n "$(lbv "$RA_PICK" heldout8)" ] && [ -s "$COST" ]; then
            "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" "${cols[@]}" --pick "$RA_PICK" --ref "$D3_PICK" \
                --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
                --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
                --group "scifact-pl=scifact-pl" --group "maupqa-mkqa=maupqa-mkqa" --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
                --win "THE LARGER MODEL WINS (R-a >= +0.2 on both sets over dose3 within the limits)" \
                --noise "R-a is within the seed noise of dose3 set by set" \
                --loss "R-a is a LOSS on both sets against dose3" "${m23[@]}" --json "$VERDICT" 2>> "$LOGS/queue_summary.log"
        fi
        echo; echo "R-c (more MS MARCO-pl rows) was not started: it needs proposed numbers and the user's decision."
        echo; echo "## Queue state (b1)"; echo; echo '```'; grep -E "b1:|$ST|$KD|arm-b1-" "$STATE" | tail -n 80; echo '```'
    } > "$SUM"
    log "summary written to $SUM"
    git add "$SUM" "$EXP_ROOT/queues/queue_b1.sh" "$C"/b1_*.yaml "$PROF" "$FORMS" 2>/dev/null
    local f m p
    for f in "$COST" "$VERDICT" "$RESULTS_ROOT/fidelity/student-$ST.corpus.json" "$RESULTS_ROOT/fidelity/student-$ST.queries.json" "$RESULTS_ROOT/fidelity/$KD.queries.json"; do [ -e "$f" ] && git add "$f" 2>/dev/null; done
    for m in "$KD" ${KD_PICK:-} "$RA" ${RA_PICK:-} "$RBS" "$RB" ${RB_PICK:-}; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
    git add "$RESULTS_ROOT"/sparsity/kd-$ST-*.json "$RESULTS_ROOT"/sparsity/arm-b1-*.json 2>/dev/null
    git commit -q -m "$1" 2>/dev/null && log "committed $(git log --oneline -1)"
    push_loop b1
}
RA_PICK=""; RB_PICK=""; RA_NOTE=""; M23=""
[ -s "$LOGS/b1_rank_mode.txt" ] && { printf 'rank_mode\t%s\n' "$(first_line "$LOGS/b1_rank_mode.txt")" >> "$PROF"; log "  ranking mode from logs/b1_rank_mode.txt: $(prof_get rank_mode)"; }
RMODE=$(prof_get rank_mode)
if [ -z "$RMODE" ]; then
    RA_NOTE="**No ranking recipe fitted the 8 GB card in the profile (plain, gradient checkpointing, 1 x 32 with gradient checkpointing): the ranking versions were not trained.**"
    index_cost "$COST" "$D3_PICK" "$FULL"; write_summary "Add the b1 results: KD of polish-roberta-base-v2 from the frozen full model"
    log "=== b1 done ($(date '+%d.%m %H:%M')) without ranking versions ==="; exit 0
fi
pause_point

# ---------------------------------------------------------------- 2-3. R-a
log "=== b1 step 2: R-a, the dose3 recipe from the KD model (mode $RMODE) ==="
train_rank "${RA#arm-}" "$C/b1_ra_dose3.yaml"
pause_point
log "=== b1 step 3: measure R-a ==="
if [ -d "models/$RA" ]; then
    form_fit "$RA"; RA_PICK=${PICK:-}
    [ -n "$RA_PICK" ] && measure "$RA_PICK"
fi
DT=$(delta "$(lbv "${RA_PICK:-none}" tiny10)" "$(lbv "$D3_PICK" tiny10)"); DH=$(delta "$(lbv "${RA_PICK:-none}" heldout8)" "$(lbv "$D3_PICK" heldout8)")
RB_GO=1
if [ -z "$RA_PICK" ] || [ "$DT" = "-" ] || [ "$DH" = "-" ]; then
    RA_NOTE="**R-a has no complete measurement (no model, no form within the limits, or a failed evaluation); R-b was still trained.**"
elif [ "$(gte "$DT" 0.2)" = 1 ] && [ "$(gte "$DH" 0.2)" = 1 ]; then
    RA_NOTE="**R-a leads dose3 by $DT on tiny10 and $DH on heldout8 (>= +0.2 on both): the 23 tasks were measured.**"
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$RA_PICK" "$ts"; done; M23="$RA_PICK"
elif [ "$(gte -0.3 "$DT")" = 1 ] && [ "$(gte -0.3 "$DH")" = 1 ]; then
    RA_NOTE="**R-a is clearly below dose3 ($DT on tiny10, $DH on heldout8; <= -0.3 on both): R-b was not started. Proposed diagnosis: the fidelity after KD against the ALBERT's, the density of the KD model and of R-a without a shift, the KD curve (above).**"; RB_GO=0
else
    RA_NOTE="R-a against dose3: $DT on tiny10, $DH on heldout8 (neither >= +0.2 on both nor <= -0.3 on both); R-b follows."
fi
log "  R-a ${RA_PICK:-none}: tiny10 $DT, heldout8 $DH against dose3; R-b $([ "$RB_GO" = 1 ] && echo follows || echo "is not started")"
index_cost "$COST" "$D3_PICK" "$FULL" ${RA_PICK:-}
write_summary "Add the b1 results: polish-roberta-base-v2 after KD from the frozen model, ranking version R-a"
[ "$RB_GO" = 1 ] || { log "=== b1 done ($(date '+%d.%m %H:%M')); R-a ${RA_PICK:-none} $DT / $DH against dose3; R-b not started ==="; exit 0; }
pause_point

# ---------------------------------------------------------------- 4. R-b
log "=== b1 step 4: R-b, stage 3 on n5 + kj and round 2 on dose3 (mode $RMODE) ==="
train_rank "${RBS#arm-}" "$C/b1_rb_s3_n5kj.yaml"
pause_point
if [ -d "models/$RBS" ]; then train_rank "${RB#arm-}" "$C/b1_rb_r2_dose3.yaml"; else log "  no stage-3 model of R-b: round 2 skipped"; fi
pause_point
if [ -d "models/$RB" ]; then
    form_fit "$RB"; RB_PICK=${PICK:-}
    [ -n "$RB_PICK" ] && measure "$RB_PICK"
    BT=$(delta "$(lbv "${RB_PICK:-none}" tiny10)" "$(lbv "$D3_PICK" tiny10)"); BH=$(delta "$(lbv "${RB_PICK:-none}" heldout8)" "$(lbv "$D3_PICK" heldout8)")
    if [ -n "$RB_PICK" ] && [ "$BT" != "-" ] && [ "$BH" != "-" ] && [ "$(gte "$BT" 0.2)" = 1 ] && [ "$(gte "$BH" 0.2)" = 1 ]; then
        for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$RB_PICK" "$ts"; done; M23="$M23 $RB_PICK"
    fi
    RA_NOTE="$RA_NOTE R-b against dose3: $BT on tiny10, $BH on heldout8."
    log "  R-b ${RB_PICK:-none}: tiny10 $BT, heldout8 $BH against dose3"
fi
mkdir -p "$LOGS/b1-aside"; [ -f "$COST" ] && mv "$COST" "$LOGS/b1-aside/index_cost_b1.after-ra.$(date +%H%M%S).json"      # measured again with R-b in it
index_cost "$COST" "$D3_PICK" "$FULL" ${RA_PICK:-} ${RB_PICK:-}
write_summary "Add the b1 results: ranking version R-b of polish-roberta-base-v2"
log "=== b1 done ($(date '+%d.%m %H:%M')); R-a ${RA_PICK:-none} $(lbs "${RA_PICK:-none}"), R-b ${RB_PICK:-none} $(lbs "${RB_PICK:-none}"); dose3 $(lbs "$D3_PICK") ==="
