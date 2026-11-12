#!/usr/bin/env bash
# Queue v58 (3060 Ti, 12.11 after v56 and v57, ~8-9 h): NATIVE POLISH FAQ rows in the student-mined arm. The training
# rows are almost all translations about American realities, while more than half of the PIRB tasks are native Polish;
# the list composition at fixed queries is exhausted (v52, v53, v57), the open axes are new queries: the dose (v56) and
# a new domain (this queue). The source is the WebFAQ v2 + clips/mqa pool (5.39 M pairs); the tools live in the data
# kitchen (~/work/speakleash/retrieval-data-kitchen, uncommitted there).
#   0  CHECKS: the drives, the input parquet, the reference queries, free space and RAM. A missing file ends the queue.
#   1  FUNNEL, every step only removes:
#      a  selection/select_pool.py (filters, PIRB decontamination, caps 2 / 2 / 200, MinHash); smoke on a prefix first;
#      b  roberta-v3 on the pairs of (a); a pair stays when gemma AND roberta-v3 are at or above the same percentile
#         inside its lexical-coverage band; the percentile is chosen on the 240 calibration pairs with Qwen labels;
#      c  PolDense-400M question embeddings: generic site-dependent questions (>= 5 neighbours at cosine >= 0.85 from
#         >= 3 domains) go with their neighbours, then greedy dedup at cosine >= 0.90 in priority order;
#      d  100 000 candidates, equally from the five coverage bands -> faq_kd_candidates_v1.jsonl (the LLM judge reads
#         this file later, started by the user; the queue does not wait for it).
#   2  ROWS: 60 000 candidates at random (seed 42) as trainer rows (dataset: webfaq-v1); the mining pool = the answers
#      of every pair that passed a-c; the final leak check against pirb-queries must find 0 queries.
#   3  NEGATIVES AND SCORES as in the mined arm: miner r2v12, 5 best + 5 random of the top-60, length window 0.5-2.0,
#      teacher soup-v9b + PolDense-400M with the fixed dense scale 84.19441654567078. Smoke on 300 rows first. The rows
#      go in three parts of 20 000 through the big-pool tool (shards), which keeps the pool out of the GPU memory.
#   4  ARM arm-dense-kl-n5kj-mined-faq20-ens400-r2v12: the 20 292 rows of the mined arm unchanged + the first 20 000
#      FAQ rows, the recipe unchanged. First bias form with <= 410 dims/doc: tiny10, heldout8, trec-covid, index cost.
#   5  PACKAGE for the 5070: datasets/transfer/transfer-5070-faq-20261112.tar.gz.
#   6  mine-hard-negatives: a 200-query smoke of the step-D chain and a time estimate for 100 000 queries, nothing more.
#   7  summary_v58.md with the funnel counts, the examples for the thresholds, the per-task tables and the reading;
#      commit, push loop. No further arm starts here.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v58.sh > logs/queue_v58.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v58 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-18000}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v58.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v56 v57}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
tc()   { lbv "$1" heldout2_treccovid; }
jget() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get(sys.argv[2]) if d.get(sys.argv[2]) is not None else '')" "$1" "$2" 2>/dev/null; }
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
kscript() {  # kscript TAG TIMEOUT OUTPUT COMMAND...: a helper script, alone on the machine; OUTPUT is removed when it fails
    local tag="$1" to="$2" out="$3"; shift 3
    wait_gpu_free; wait_ram; local s=$SECONDS rc
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/v58_$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { rm -f "$out"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/v58_$tag.log" | cut -c1-320)"
    record "chain:v58_$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
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
        for t in 005 010 015 020; do [ "$m-bias$t" != "$PICK" ] && rm -rf "models/$m-bias$t"; done
    fi
    log "  $m: $(dims "$m") dims/doc unbiased; measured form $PICK at $(dims "$PICK")"
    regime quantized "$PICK" "models/$PICK" tiny10 16; regime quantized "$PICK" "models/$PICK" heldout8 16
    log "  $PICK: $(lbs "$PICK") ($ref), $(dims "$PICK") dims/doc"
    regime quantized "$PICK" "models/$PICK" heldout2_treccovid 8; log "  $PICK trec-covid $(tc "$PICK"), $(dims "$PICK") dims/doc"
}
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
D3=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12
F=arm-dense-kl-n5kj-mined-faq20-ens400-r2v12
A=datasets/arms
MROWS=$A/n5-plus-kj-mined-tens400.jsonl      # the rows of the mined arm (v49), used verbatim
FAQ60=$A/webfaq-kd-v1-60k.jsonl              # the sampled FAQ rows, no negatives yet
FAQM=$A/webfaq-kd-v1-60k-mined.jsonl         # the same rows with negatives and teacher scores
DROWS=$A/n5-plus-kj-mined-faq20.jsonl
FAQ_ARM_ROWS=${FAQ_ARM_ROWS:-20000}; FAQ_SAMPLE=${FAQ_SAMPLE:-60000}; CAND_TARGET=${CAND_TARGET:-100000}
PD=OPI-PIB/PolDense-400M
DCACHE=cache/dense_pool/pd400-webfaq
SCALE=84.19441654567078
RECIPE="$KITCHEN/selection/recipes/webfaq-kd-v1.yaml"
FUNNEL="$KITCHEN/selection/faq_kd_funnel.py"
CALIB="$KITCHEN/data/annotation/faq_calib_v1.jsonl"
CALIBLAB="$KITCHEN/data/annotation/faq_calib_v1.v2.qwen.llm.jsonl"
BASECFG="$C/distil_dense_kl_n5kj_mined_ens400_r2v12.yaml"
CFG="$C/distil_dense_kl_n5kj_mined_faq20_ens400_r2v12.yaml"
PKG=datasets/transfer/transfer-5070-faq-20261112.tar.gz
KEEP="$RESULTS_ROOT/v58_faq"                  # small JSON reports of the funnel, copied here for the commit
recipe_get() { "$PY" -c "import sys; sys.path.insert(0, sys.argv[1] + '/selection'); from recipe import load_recipe; r = load_recipe(sys.argv[2]); v = r
for k in sys.argv[3].split('.'): v = v[int(k)] if isinstance(v, list) else v[k]
print(v)" "$KITCHEN" "$RECIPE" "$1" 2>/dev/null; }
OUT=$(recipe_get output_dir); INPUT=$(recipe_get input); REFQ=$(recipe_get decontaminate.queries.0.path)
CAND="$OUT/faq_kd_candidates_v1.jsonl"; POOL="$OUT/pool-answers.jsonl"
log "=== v58 start: native Polish FAQ rows; mined arm bias010 $(lbs $M-bias010) / $(tc $M-bias010) at $(dims $M-bias010) dims/doc, bias013 $(lbs $M-bias013) at $(dims $M-bias013); $(free_gb) GB free on / ==="

# 0. checks
log "=== v58 step 0: checks ==="
[ -n "$OUT" ] && [ -n "$INPUT" ] && [ -n "$REFQ" ] || { log "  the recipe $RECIPE could not be read: v58 stops"; exit 1; }
[ -s "$INPUT" ] || { log "  input $INPUT missing (is the drive 'My Passport' mounted?): v58 stops"; exit 1; }
[ -s "$REFQ" ] || { log "  reference queries $REFQ missing (is the data drive mounted?): v58 stops"; exit 1; }
mkdir -p "$OUT" "$KEEP" || { log "  cannot create $OUT: v58 stops"; exit 1; }
for f in "$FUNNEL" "$KITCHEN/selection/select_pool.py" "$KITCHEN/bridge/reranker_pair_scores.py" "$KITCHEN/bridge/faq_selected_to_rows.py" \
         "$KITCHEN/bridge/splade_mine_big_pool.py" "$CALIB" "$CALIBLAB" "$MROWS" "$BASECFG"; do
    [ -s "$f" ] || { log "  $f missing: v58 stops"; exit 1; }
done
for m in "$R2" soup-v9b soup-v12; do [ -d "models/$m" ] || { log "  models/$m missing: v58 stops"; exit 1; }; done
DATA_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
[ "${DATA_GB:-0}" -ge 10 ] || { log "  only ${DATA_GB:-0} GB free on the data drive ($OUT): v58 stops"; exit 1; }
need_space || { log "  not enough space on /: v58 stops"; exit 1; }
log "  input $(du -h "$INPUT" | cut -f1), output $OUT (${DATA_GB} GB free there), $(free_gb) GB free on /, RAM available $(free -g | awk '/^Mem/{print $7}') GB"
V57_NOTE=""
if grep -q "THE FILTERED LARGE POOL WINS" "$RESULTS_ROOT/summary_v57.md" 2>/dev/null; then
    V57_NOTE="v57 reports a gain of the false-negative filter (>= +0.2 on both sets). The exception of the plan applies (reranker-filtered lists of mine-hard-negatives), but those lists do not exist (step 6 is a smoke only), so this queue still uses the student-mined lists. A decision for the user."
    log "  NOTE: $V57_NOTE"
fi
steps_of() { "$PY" -c "
import json, sys
m = json.load(open(sys.argv[1]))
print('; '.join('%s %s' % (s['step'], s['rows'] if 'removed' not in s else '-%d -> %d' % (s['removed'], s['rows'])) for s in m['steps']))" "$1" 2>/dev/null; }

# 1a. selection A+B
log "=== v58 step 1a: selection A+B (select_pool.py, recipe webfaq-kd-v1) ==="
if [ ! -s "$OUT/selected.parquet" ] || [ ! -s "$OUT/manifest.json" ]; then
    rm -rf "$OUT/smoke"
    if kscript sel_smoke 3600 "$OUT/smoke/manifest.json" nice -n 10 "$PY" "$KITCHEN/selection/select_pool.py" "$RECIPE" --limit 200000 --output-dir "$OUT/smoke" --force; then
        log "  smoke on the first 200 000 rows: $(steps_of "$OUT/smoke/manifest.json")"
        cp "$OUT/smoke/manifest.json" "$KEEP/selection_smoke_manifest.json"
    else log "  selection smoke FAILED (logs/v58_sel_smoke.log): v58 stops"; exit 1; fi
    rm -rf "$OUT/smoke"
    kscript sel_full 10800 "$OUT/manifest.json" nice -n 10 "$PY" "$KITCHEN/selection/select_pool.py" "$RECIPE" --force \
        || { log "  selection FAILED (logs/v58_sel_full.log): v58 stops"; exit 1; }
fi
[ -s "$OUT/selected.parquet" ] || { log "  $OUT/selected.parquet missing: v58 stops"; exit 1; }
log "  A+B: $(steps_of "$OUT/manifest.json")"
cp "$OUT/manifest.json" "$KEEP/selection_manifest.json"

# 1b. two-reranker agreement
log "=== v58 step 1b: roberta-v3 on the A+B pairs, agreement with gemma inside the coverage band ==="
[ -s "$OUT/scores-roberta-v3.parquet" ] || kscript rerank 14400 "$OUT/scores-roberta-v3.parquet" "$PY" "$KITCHEN/bridge/reranker_pair_scores.py" \
    --input "$OUT/selected.parquet" --output "$OUT/scores-roberta-v3.parquet" || { log "  roberta-v3 scoring FAILED (logs/v58_rerank.log): v58 stops"; exit 1; }
if [ ! -s "$OUT/calib-scores-roberta-v3.parquet" ]; then
    "$PY" "$FUNNEL" calib-pairs --calib "$CALIB" --output "$OUT/calib.parquet" > "$LOGS/v58_calib_pairs.log" 2>&1
    kscript rerank_calib 3600 "$OUT/calib-scores-roberta-v3.parquet" "$PY" "$KITCHEN/bridge/reranker_pair_scores.py" \
        --input "$OUT/calib.parquet" --output "$OUT/calib-scores-roberta-v3.parquet" || { log "  roberta-v3 on the calibration pairs FAILED (logs/v58_rerank_calib.log): v58 stops"; exit 1; }
fi
[ -s "$OUT/agreed.parquet" ] || kscript agree 3600 "$OUT/agreed.parquet" nice -n 10 "$PY" "$FUNNEL" agree --selected "$OUT/selected.parquet" --scores "$OUT/scores-roberta-v3.parquet" \
    --calib "$CALIB" --calib-labels "$CALIBLAB" --calib-scores "$OUT/calib-scores-roberta-v3.parquet" --output "$OUT/agreed.parquet" --report "$OUT/funnel_b.json" \
    || { log "  the agreement step FAILED (logs/v58_agree.log): v58 stops"; exit 1; }
log "  b: percentile $(jget "$OUT/funnel_b.json" percentile) ($(jget "$OUT/funnel_b.json" chosen_by)); precision $(jget "$OUT/funnel_b.json" precision), recall $(jget "$OUT/funnel_b.json" recall); $(jget "$OUT/funnel_b.json" kept_rows) of $(jget "$OUT/funnel_b.json" input_rows) pairs stay"

# 1c. question similarity
log "=== v58 step 1c: question similarity (PolDense-400M): generic clusters and near-duplicates ==="
[ -s "$OUT/deduped.parquet" ] || kscript similarity 7200 "$OUT/deduped.parquet" "$PY" "$FUNNEL" similarity --input "$OUT/agreed.parquet" --calib "$CALIB" --calib-labels "$CALIBLAB" \
    --model "$PD" --output "$OUT/deduped.parquet" --report "$OUT/funnel_c.json" || { log "  the similarity step FAILED (logs/v58_similarity.log): v58 stops"; exit 1; }
log "  c: generic clusters -$(jget "$OUT/funnel_c.json" removed_generic_clusters), near-duplicates -$(jget "$OUT/funnel_c.json" removed_near_duplicates); $(jget "$OUT/funnel_c.json" kept_rows) of $(jget "$OUT/funnel_c.json" input_rows) pairs stay; calibration check $(jget "$OUT/funnel_c.json" calibration_check)"

# 1d. candidates
log "=== v58 step 1d: $CAND_TARGET candidates, equally from the coverage bands ==="
[ -s "$CAND" ] || kscript candidates 1800 "$CAND" nice -n 10 "$PY" "$FUNNEL" candidates --input "$OUT/deduped.parquet" --target "$CAND_TARGET" --output "$CAND" --report "$OUT/funnel_d.json" \
    || { log "  the candidate step FAILED (logs/v58_candidates.log): v58 stops"; exit 1; }
log "  d: $(jget "$OUT/funnel_d.json" candidates) candidates, by band $(jget "$OUT/funnel_d.json" taken_by_band) (available $(jget "$OUT/funnel_d.json" available_by_band)) -> $CAND"
for f in funnel_b funnel_c funnel_d; do cp "$OUT/$f.json" "$KEEP/$f.json" 2>/dev/null; done

# 2. rows and the mining pool, the final leak check
log "=== v58 step 2: $FAQ_SAMPLE rows from the candidates, the pool = the answers of every pair that passed a-c ==="
if [ ! -s "$FAQM" ] && { [ ! -s "$FAQ60" ] || [ ! -s "$POOL" ]; }; then
    wait_gpu_free; wait_ram
    ORPHAN_GUARD_EXEMPT=1 nice -n 10 timeout -k 60 3600 "$PY" "$KITCHEN/bridge/faq_selected_to_rows.py" --selected "$OUT/deduped.parquet" --candidates "$CAND" \
        --benchmark-queries "$REFQ" --rows-output "$FAQ60" --pool-output "$POOL" --sample "$FAQ_SAMPLE" --seed 42 --dataset webfaq-v1 --stats "$OUT/rows.stats.json" > "$LOGS/v58_rows.log" 2>&1
    rc=$?
    while read -r l; do log "  $l"; done < <(tail -n 3 "$LOGS/v58_rows.log" | cut -c1-400)
    if [ $rc = 2 ]; then log "  LEAK CHECK FAILED: selected queries match pirb-queries after normalisation ($OUT/rows.stats.json): v58 stops"; rm -f "$FAQ60"; exit 1; fi
    [ $rc = 0 ] && [ -s "$FAQ60" ] && [ -s "$POOL" ] || { log "  building the rows FAILED (logs/v58_rows.log): v58 stops"; exit 1; }
fi
cp "$OUT/rows.stats.json" "$KEEP/rows.stats.json" 2>/dev/null
if [ ! -s "$FAQM" ]; then
    NFAQ=$(grep -c . "$FAQ60")
    [ "$NFAQ" -ge 5000 ] || { log "  only $NFAQ FAQ rows survived the funnel (fewer than 5000): v58 stops"; exit 1; }
    log "  $NFAQ FAQ rows, pool $(grep -c . "$POOL") answers"
    rm -f $A/webfaq-kd-v1-part?.jsonl
    split -l 20000 -d -a 1 --additional-suffix=.jsonl "$FAQ60" "$A/webfaq-kd-v1-part"
fi

# 3. negatives and teacher scores
mine() {  # mine TAG OUT ROWS [tool args]: r2v12 mines over the rows' own answers + the pool, the ensemble teacher scores with the fixed scale
    local tag="$1" out="$2" rows="$3"; shift 3
    [ -s "$out" ] && return 0
    kscript "mine_$tag" 14400 "$out" "$PY" "$KITCHEN/bridge/splade_mine_big_pool.py" --student "models/$R2" --splade-teacher models/soup-v9b \
        --dense-teacher "$PD" --rows "$rows" --extra-pool "$POOL" --extra-chars 1 1000000 --dense-scale "$SCALE" --output "$out" "$@"
}
log "=== v58 step 3: negatives and teacher scores (miner $R2, fixed dense scale $SCALE) ==="
if [ ! -s "$FAQM" ]; then
    rm -rf cache/smoke_v58; mkdir -p cache/smoke_v58
    if mine smoke cache/smoke_v58/rows.jsonl "$A/webfaq-kd-v1-part0.jsonl" --limit 300 --extra-max 20000 --extra-shard 8000 --dense-cache cache/smoke_v58/dense \
        && [ "$(grep -c . cache/smoke_v58/rows.jsonl)" = 300 ]; then
        log "  smoke passed: $(tail -n 1 "$LOGS/v58_mine_smoke.log" | cut -c1-300)"
    else log "  mining smoke FAILED (logs/v58_mine_smoke.log): v58 stops"; rm -rf cache/smoke_v58; exit 1; fi
    rm -rf cache/smoke_v58
    for part in $A/webfaq-kd-v1-part?.jsonl; do
        p=$(basename "$part" .jsonl); p=${p##*part}
        if mine "part$p" "$A/webfaq-kd-v1-part$p-mined.jsonl" "$part" --extra-max 0 --dense-cache "$DCACHE"; then
            cp "$A/webfaq-kd-v1-part$p-mined.jsonl.mined.json" "$KEEP/mining_part$p.json" 2>/dev/null
        elif [ "$p" = 0 ]; then log "  mining of the first part FAILED (logs/v58_mine_part0.log): v58 stops"; exit 1
        else log "  mining of part $p FAILED (logs/v58_mine_part$p.log): the package gets fewer rows, the arm is not affected"; break; fi
    done
    cat $A/webfaq-kd-v1-part?-mined.jsonl > "$FAQM.tmp" && mv "$FAQM.tmp" "$FAQM"
    [ -s "$FAQM" ] || { log "  no mined FAQ rows: v58 stops"; exit 1; }
    rm -f $A/webfaq-kd-v1-part?.jsonl $A/webfaq-kd-v1-part?-mined.jsonl
fi
"$PY" - "$FAQM" "$SCALE" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, math, statistics as st, sys
from collections import Counter
path, scale = sys.argv[1], float(sys.argv[2])
masses, flip, first, n, pools = [], 0, 0, 0, Counter()
for line in open(path, encoding="utf-8"):
    r = json.loads(line); n += 1
    assert len(r["neg"]) == 10 == len(r["neg_scores"]) and len(r["pos_scores"]) == len(r["pos"]), "row shape: " + str(r["query_id"])
    assert abs(r["dense_scale"] - scale) < 1e-9, "dense scale differs from the fixed scale"
    s = [max(r["pos_scores"])] + list(r["neg_scores"]); m = max(s); e = [math.exp((x - m) / 2.0) for x in s]
    masses.append(e[0] / sum(e)); flip += max(r["neg_scores"]) > min(r["pos_scores"]); first += min(r["pos_student_rank"]) == 1
    pools.update(r.get("neg_pool", []))
stats = {"rows": n, "teacher_positive_mass_median": st.median(masses), "teacher_positive_mass_above_095": sum(x > 0.95 for x in masses) / n,
         "rows_with_negative_above_positive": flip, "student_ranks_the_positive_first": first / n, "negatives_by_pool": dict(pools), "dense_scale": scale,
         "old_rows_of_the_mined_arm": {"rows": 20292, "teacher_positive_mass_median": 0.852, "teacher_positive_mass_above_095": 0.303, "rows_with_negative_above_positive": 2658}}
json.dump(stats, open(path + ".stats.json", "w"), indent=2)
print("FAQ rows %d: teacher P(pos) median %.3f (old rows 0.852), share above 0.95 %.3f (0.303), negative above positive in %d rows = %.1f %% (old rows 2 658 of 20 292 = 13.1 %%), student ranks the positive first in %.3f"
      % (n, st.median(masses), stats["teacher_positive_mass_above_095"], flip, 100 * flip / n, first / n))
PYEOF
[ -s "$FAQM.stats.json" ] || { log "  the mined FAQ rows failed their checks: v58 stops"; exit 1; }
cp "$FAQM.stats.json" "$KEEP/mining_stats.json"

# 4. the arm
log "=== v58 step 4: arm $F = the mined-arm rows unchanged + the first $FAQ_ARM_ROWS FAQ rows ==="
if [ ! -s "$DROWS" ]; then
    "$PY" - "$MROWS" "$FAQM" "$DROWS" "$FAQ_ARM_ROWS" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, os, random, sys
base, faq, out, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
bl = [l if l.endswith("\n") else l + "\n" for l in open(base, encoding="utf-8") if l.strip()]
fl = [l if l.endswith("\n") else l + "\n" for l in open(faq, encoding="utf-8") if l.strip()][:k]
bids = {str(json.loads(l)["query_id"]) for l in bl}; fids = [str(json.loads(l)["query_id"]) for l in fl]
assert len(set(fids)) == len(fids), "duplicate FAQ query ids"
assert not (bids & set(fids)), "FAQ query ids overlap the base rows"
both = bl + fl; order = list(range(len(both))); random.Random(42).shuffle(order)
with open(out + ".tmp", "w", encoding="utf-8") as f:
    for i in order:
        f.write(both[i])
os.replace(out + ".tmp", out)
pairs = sum(len(json.loads(l)["pos"]) for l in both)
json.dump({"rows": len(both), "base_rows": len(bl), "faq_rows": len(fl), "training_pairs": pairs, "base_rows_file": base, "faq_rows_file": faq}, open(out + ".rows.json", "w"), indent=2)
print("rows: %d = %d base (verbatim) + %d FAQ; training pairs %d" % (len(both), len(bl), len(fl), pairs))
PYEOF
    [ -s "$DROWS" ] || { log "  row file $DROWS not written: v58 stops"; exit 1; }
fi
NROWS=$(grep -c . "$DROWS"); NFAQARM=$(jget "$DROWS.rows.json" faq_rows)
sed -e "4s|.*|# The FAQ arm (queue v58): the rows of the student-mined arm ($MROWS, unchanged) plus ${NFAQARM} native Polish FAQ rows (WebFAQ v2 + clips/mqa after the selection funnel of the data kitchen: decontaminated against PIRB, two-reranker agreement, semantic dedup), negatives mined by r2v12 under the same rules (5 best + 5 random of its top-60, length window 0.5-2.0) from the answers of the funnel's pool, scored by the ensemble teacher (soup-v9b + PolDense-400M) with the fixed dense scale $SCALE. The recipe of the mined arm from soup-v12, only the row file differs.|" \
    -e "s|^runs_dir: .*|runs_dir: runs/$F|" -e "s|^train_data: .*|train_data: $DROWS|" "$BASECFG" > "$CFG"
train "${F#arm-}" "$CFG"
D3_PICK=""; for t in "" -bias005 -bias010 -bias015 -bias020; do [ -n "$(lbv "$D3$t" heldout8)" ] && D3_PICK="$D3$t"; done
REF="mined arm bias010 $(lbs $M-bias010) / $(tc $M-bias010) at $(dims $M-bias010); bias013 $(lbs $M-bias013) at $(dims $M-bias013); seed 43 bias015 $(lbs $M-s43-bias015) at $(dims $M-s43-bias015); dose arm ${D3_PICK:-none} $([ -n "$D3_PICK" ] && lbs "$D3_PICK")"
at_cost "$F" "$REF"; F_PICK=${PICK:-}
[ -d "models/$F" ] || F_PICK=""
COST="$RESULTS_ROOT/index_cost_v58.json"
if [ -n "$F_PICK" ] && [ ! -s "$COST" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    args=(--model base=models/base-splade-wrapped)
    for m in $M-bias010 $M-bias013 $D3_PICK "$F_PICK"; do [ -d "models/$m" ] && args+=(--model "$m=models/$m"); done
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" --output-json "$COST" > "$LOGS/index_cost_v58.log" 2>&1 \
        && log "index cost written ($COST) in $((SECONDS - s))s" || { log "index cost FAILED (logs/index_cost_v58.log)"; rm -f "$COST"; }
fi

# 5. the package for the 5070
log "=== v58 step 5: package for the 5070 ($PKG) ==="
if [ ! -s "$PKG" ]; then
    ST=cache/transfer_v58/transfer-5070-faq-20261112; rm -rf cache/transfer_v58; mkdir -p "$ST/rows" "$ST/selection" "$(dirname "$PKG")"
    cp "$FAQM" "$FAQM.stats.json" "$ST/rows/"; cp "$KEEP"/mining_part*.json "$ST/rows/" 2>/dev/null
    cp "$KEEP"/selection_manifest.json "$KEEP"/funnel_?.json "$KEEP"/rows.stats.json "$RECIPE" "$ST/selection/" 2>/dev/null
    cat > "$ST/README.md" <<EOF
# Native Polish FAQ rows with student-mined negatives and ensemble-teacher scores (3060 queue v58, 12.11.2026)

- rows/$(basename "$FAQM"): $(grep -c . "$FAQM") trainer rows (query, pos, neg x 10, pos_scores, neg_scores; dataset: webfaq-v1). The first
  $NFAQARM rows are the FAQ part of the 3060 arm $F; the order is the random draw (seed 42), so any prefix is a random sample.
- Source: WebFAQ v2 + clips/mqa Polish FAQ pairs after the kitchen's funnel (selection/): A+B selection with PIRB decontamination
  (questions, answers, template families, blocked source sites), gemma + roberta-v3 agreement inside the lexical-coverage band,
  semantic dedup of the questions and removal of generic site-dependent questions. Counts and thresholds: selection/*.json.
- Negatives: mined by models/$R2 from the answers of every pair that passed the funnel, 5 best + 5 random of its top-60, length window
  0.5-2.0. Scores: mean(models/soup-v9b SPLADE dot product, $PD cosine x $SCALE): the fixed dense scale of the mined arm.
  rows/*.stats.json and rows/mining_part*.json describe the mining (teacher P(pos), negatives above the positive).
- faq_* fields of a row: source, domain, gemma score of the Polish pair (faq_score_gemma_pl), roberta-v3 score, coverage band.
- No LLM judge verdicts yet: the candidate file stays on the 3060 data drive; verdicts can drop rows later without new mining.
- SHA256SUMS covers every file of this package.
EOF
    (cd "$ST" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
    tar -czf "$PKG.tmp" -C "$ST" . && mv "$PKG.tmp" "$PKG" && log "  package written: $(du -h "$PKG" | cut -f1), sha256 $(sha256sum "$PKG" | cut -c1-16)…" || log "  package FAILED"
    rm -rf cache/transfer_v58
fi

# 6. mine-hard-negatives: smoke and a time estimate only
log "=== v58 step 6: mine-hard-negatives smoke (200 queries) and the time estimate for 100 000 queries ==="
EST="$KEEP/mine_smoke_estimate.json"
if [ ! -s "$EST" ] && [ -s "$KITCHEN/bridge/faq_mine_nowait.sh" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    SMOKE=200 POOL="$OUT/deduped.parquet" MINE_OUT="$OUT/mine" PY="$PY" ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 bash "$KITCHEN/bridge/faq_mine_nowait.sh" > "$LOGS/v58_faq_mine_smoke.log" 2>&1; rc=$?
    log "  smoke rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/v58_faq_mine_smoke.log" | cut -c1-200)"
    "$PY" - "$LOGS/v58_faq_mine_smoke.log" "$OUT/mine-smoke/corpus.parquet" "$POOL" "$EST" "$rc" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, re, sys
log, corpus, pool, out, rc = sys.argv[1:6]
steps = {m.group(1).strip(): int(m.group(2)) for m in re.finditer(r"\] \[faq-mine-smoke\] (?!=)(.+?) done in (\d+)s", open(log, encoding="utf-8", errors="replace").read())}
try:
    import pyarrow.parquet as pq
    n_corpus = pq.ParquetFile(corpus).metadata.num_rows
except Exception:
    n_corpus = 5200
n_pool = sum(1 for _ in open(pool, encoding="utf-8"))
est = {}
for label, sec in steps.items():
    factor = n_pool / n_corpus if label.startswith("ingest") else 100000 / 200
    est[label] = {"smoke_seconds": sec, "factor": round(factor, 1), "estimated_hours": round(sec * factor / 3600, 2)}
total = sum(v["estimated_hours"] for v in est.values())
res = {"smoke_rc": int(rc), "smoke_queries": 200, "smoke_corpus_passages": n_corpus, "pool_passages": n_pool, "queries_for_the_estimate": 100000,
       "steps": est, "estimated_hours_total": round(total, 1), "complete": int(rc) == 0 and len(est) >= 6,
       "note": "linear extrapolation of the smoke's step times (ingest by the corpus size, the other steps by the number of queries); the smoke includes model loading, so it overestimates"}
json.dump(res, open(out, "w"), indent=2)
print("estimate for 100 000 queries over %d passages: %.1f h (%s); steps: %s" % (n_pool, total, "complete smoke" if res["complete"] else "INCOMPLETE smoke, rc %s" % rc,
      "; ".join("%s %.2f h" % (k[:28], v["estimated_hours"]) for k, v in est.items())))
print("the full run is NOT started by this queue" + (" (the estimate is within 3 h: a decision for the user)" if res["complete"] and total <= 3 else ""))
PYEOF
    rm -rf "$OUT/mine-smoke/lancedb"
fi

# 7. summary
log "=== v58 step 7: summary ==="
SUM="$RESULTS_ROOT/summary_v58.md"
"$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$COST" "$KEEP" "$M" "$R2" "${D3_PICK:-none}" "${F_PICK:-none}" "$CAND" "$PKG" "$DROWS.rows.json" "$KITCHEN" "$V57_NOTE" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, spa = Path(sys.argv[1]), Path(sys.argv[2])
cost_path, keep, m, r2, d3, pick, cand, pkg, rows_json, kitchen, v57_note = sys.argv[3:14]
keep = Path(keep)
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}
def res(label, ts):
    return load(reg / label / f"{ts}.json")
def val(label, ts, n=None):
    d = res(label, ts)
    return d.get("mean_ndcg@10") if d and (n is None or d.get("n_tasks") == n) else None
def dims(label):
    try:
        d = json.load(open(spa / f"{label}.json"))[label]; v = [t["mean"] for t in d.values() if t.get("mean") is not None]
        return sum(v) / len(v)
    except Exception:
        return None
def fmt(x, nd=2): return "-" if x is None else f"{x:.{nd}f}"
cost = load(cost_path).get("models", {})
def c(label, key, nd=0):
    s = cost.get(label); return fmt(s[key], nd) if s else "-"
man, fb, fc, fd = load(keep / "selection_manifest.json"), load(keep / "funnel_b.json"), load(keep / "funnel_c.json"), load(keep / "funnel_d.json")
rs, ms, est, arm = load(keep / "rows.stats.json"), load(keep / "mining_stats.json"), load(keep / "mine_smoke_estimate.json"), load(rows_json)
print("# v58 (12.11, quantized regime, nDCG@10): native Polish FAQ rows in the student-mined arm\n")
print("faq20 = the rows of the student-mined arm unchanged plus %s native Polish FAQ rows (WebFAQ v2 + clips/mqa after the selection funnel of the data kitchen), "
      "negatives mined by r2v12 under the same rules from the answers of the funnel's pool, the same ensemble teacher with the fixed dense scale, the same recipe "
      "(from soup-v12, 2 epochs). New arms are measured in their first bias form with at most 410 dims/doc.\n" % arm.get("faq_rows", "-"))
if v57_note:
    print("**Note:** " + v57_note + "\n")

# results
print("## Result\n")
print("| model | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |\n|---|---:|---:|---:|---:|---:|---:|---:|")
labels = [r2, f"{m}-bias010", f"{m}-bias013", f"{m}-bias015", f"{m}-s43-bias015"] + [x for x in (d3, pick) if x != "none"]
for label in labels:
    if val(label, "tiny10") is None:
        continue
    name = f"**{label}**" if label == pick else label
    print(f"| {name} | {fmt(val(label, 'tiny10'))} | {fmt(val(label, 'heldout8'))} | {fmt(val(label, 'heldout2_treccovid'))} | {fmt(dims(label), 1)} | {c(label, 'doc_active_mean', 1)} | {c(label, 'query_active_mean', 1)} | {c(label, 'postings_per_query')} |")
t, h, d = val(pick, "tiny10", 10), val(pick, "heldout8", 8), dims(pick)
own = [x for x in (f"{m}-bias0{s}" for s in ("10", "11", "12", "13", "15")) if None not in (val(x, "tiny10", 10), val(x, "heldout8", 8), dims(x))]
ref = min(own, key=lambda x: abs(dims(x) - d)) if own and d is not None else None
FAQ_TASKS = ("poleval-2022-test-A-allegro-faq", "poleval-2022-test-B-allegro-faq", "mfaq")
WATCH = FAQ_TASKS + ("fiqa-pl",)
verdict = "the FAQ arm has no complete result"
if None not in (t, h, d) and ref:
    dt, dh = t - val(ref, "tiny10"), h - val(ref, "heldout8")
    per = {}
    for ts in ("tiny10", "heldout8"):
        a, b = res(pick, ts).get("tasks", {}), res(ref, ts).get("tasks", {})
        per.update({k: a[k] - b[k] for k in a if k in b})
    faq_d = [per[k] for k in FAQ_TASKS if k in per]; mq_d = [v for k, v in per.items() if k.startswith("maupqa")]
    rest_d = [v for k, v in per.items() if k not in FAQ_TASKS]
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    if dt >= 0.2 and dh >= 0.2:
        word = "THE NEW DOMAIN PAYS (>= +0.2 on both sets): a larger FAQ dose (the rest of the mined rows) is the next candidate; not started here"
    elif dt <= -0.2 and dh <= -0.2:
        word = "FAQ rows in this form HURT the polish-splade family (<= -0.2 on both sets); the package for the 5070 is built anyway (stage 1 from MLM is a different question)"
    elif mean(faq_d) >= 0.3 and mean(rest_d) <= -0.1:
        word = "within the noise on the means, but NOT 'no effect': the FAQ tasks gain and the other tasks pay for it"
    else:
        word = "within the noise on the means and no clear gain on the FAQ tasks either"
    verdict = ("%s %.2f / %.2f at %.1f dims/doc against the mined arm %s %.2f / %.2f at %.1f: delta %+.2f / %+.2f, trec-covid %s against %s -> %s. "
               "Per task: Allegro FAQ A %s, Allegro FAQ B %s, mfaq %s (mean of the three %+.2f); MAUPQA tasks mean %+.2f; fiqa-pl %s; all other tasks mean %+.2f"
               % (pick, t, h, d, ref, val(ref, "tiny10"), val(ref, "heldout8"), dims(ref), dt, dh, fmt(val(pick, "heldout2_treccovid")), fmt(val(ref, "heldout2_treccovid")), word,
                  *("%+.2f" % per[k] if k in per else "-" for k in FAQ_TASKS), mean(faq_d), mean(mq_d), "%+.2f" % per["fiqa-pl"] if "fiqa-pl" in per else "-", mean(rest_d)))
    if d3 != "none" and None not in (val(d3, "tiny10", 10), val(d3, "heldout8", 8)):
        verdict += ". Against the dose arm of v56 %s (%.2f / %.2f at %s): delta %+.2f / %+.2f" % (d3, val(d3, "tiny10"), val(d3, "heldout8"), fmt(dims(d3), 1), t - val(d3, "tiny10"), h - val(d3, "heldout8"))
print("\n## Reading\n\n" + verdict + ".")
print("\nSeed noise of this recipe is 0.13-0.2 (one seed here). Nothing further was started by this queue.")
if ref and pick != "none":
    for ts in ("tiny10", "heldout8"):
        a, b, e = res(pick, ts).get("tasks", {}), res(ref, ts).get("tasks", {}), (res(d3, ts).get("tasks", {}) if d3 != "none" else {})
        if not a:
            continue
        print(f"\n### {ts} per task\n\n| task | mined arm ({ref.split('-')[-1]}) | dose arm (v56) | FAQ arm | delta vs mined arm |\n|---|---:|---:|---:|---:|")
        for k in a:
            mark = " (FAQ)" if k in FAQ_TASKS else ""
            print(f"| {k}{mark} | {fmt(b.get(k))} | {fmt(e.get(k))} | {fmt(a[k])} | {('%+.2f' % (a[k] - b[k])) if k in b else '-'} |")

# the funnel
print("\n## Selection funnel (pairs left after each step)\n\n| step | removed | left |\n|---|---:|---:|")
for s in man.get("steps", []):
    print(f"| a: {s['step']} | {s.get('removed', '')} | {s['rows']} |")
if fb:
    print(f"| b: gemma and roberta-v3 both at or above percentile {fb['percentile']} of their coverage band | {fb['removed_rows']} | {fb['kept_rows']} |")
if fc:
    print(f"| c: generic site-dependent questions (clusters) | {fc['removed_generic_clusters']} | {fc['input_rows'] - fc['removed_generic_clusters']} |")
    print(f"| c: near-duplicate questions (cosine >= 0.90) | {fc['removed_near_duplicates']} | {fc['kept_rows']} |")
if fd:
    print(f"| d: candidates, equally from the coverage bands | {fd['input_rows'] - fd['candidates']} | {fd['candidates']} |")
if rs:
    print(f"| rows drawn from the candidates (seed 42) | | {rs.get('rows')} |")
    lc = rs.get("leak_check", {})
    print("\nFinal leak check against %s benchmark queries (%s): **%s** selected queries match after normalisation (normalised text or sorted tokens), %s more only by the "
          "loose key (letters and digits; left out of the rows). Pool for mining: %s answers. Rows by source: %s; by coverage band: %s; domains: %s; domains with 'allegro' in the "
          "funnel's output: %s (allegro.pl itself is on the blocklist)."
          % (rs.get("benchmark_queries_read"), Path(rs.get("benchmark_queries", "")).name, lc.get("selected_queries_matching_a_benchmark_query_after_normalisation"),
             lc.get("selected_queries_matching_only_by_the_loose_key"), rs.get("pool_passages"), rs.get("rows_by_source"), rs.get("rows_by_coverage_band"), rs.get("rows_domains"),
             rs.get("selected_domains_with_allegro") or "none"))
dec = man.get("decontamination", {}).get("whole_input")
if dec:
    print("\nDecontamination on the whole input: %s" % dec)
if fb:
    print("\n### Step b: the reranker threshold\n\nChosen percentile: **%s** (%s). On the calibration pairs (%s pairs with Qwen labels, %s good = %s): precision **%s**, recall **%s**."
          % (fb["percentile"], fb["chosen_by"], fb["calibration"]["pairs"], fb["calibration"]["good"], fb["calibration"]["good_definition"], fmt(fb.get("precision"), 3), fmt(fb.get("recall"), 3)))
    print("The calibration sample is stratified (source x coverage band x gemma band), not a random sample of the pool, so the precision is a guide, not an estimate for the pool.\n")
    print("| percentile | calibration pairs kept | good among them | precision | recall | pool pairs kept |\n|---:|---:|---:|---:|---:|---:|")
    for r in fb["table"]:
        print(f"| {r['percentile']} | {r['calib_kept']} | {r['calib_good_kept']} | {fmt(r['precision'], 3)} | {fmt(r['recall'], 3)} | {r['pool_kept']} |")
    print("\nScore cut-offs by coverage band (gemma / roberta-v3; pool -> kept): " + "; ".join(
        "band %s: %s / %s (%s -> %s)" % (b, fmt(v["gemma"]), fmt(v["roberta_v3"]), v["pool"], v["kept"]) for b, v in fb["score_cutoffs_by_band"].items()) + ".")
if fc:
    cc = fc.get("calibration_check") or {}
    print("\n### Step c: question similarity (%s)\n\nRules: %s; %s." % (fc["model"], fc["rule_generic"], fc["rule_dedup"]))
    if cc:
        fl, nf = cc["flagged_generic"], cc["not_flagged"]
        print("Calibration check (Qwen `context: site`): among the %d calibration questions the cluster rule would flag, %d have context site (%s); among the %d it would not flag, %d (%s)."
              % (fl, cc["flagged_with_context_site"], "%.0f %%" % (100 * cc["flagged_with_context_site"] / fl) if fl else "-", nf, cc["not_flagged_with_context_site"],
                 "%.0f %%" % (100 * cc["not_flagged_with_context_site"] / nf) if nf else "-"))
if ms:
    print("\n## Mining of the FAQ rows\n\n| rows | teacher P(pos) median | share with P(pos) > 0.95 | rows with a negative above the positive | student ranks the positive first |\n|---|---:|---:|---:|---:|")
    print(f"| FAQ ({ms['rows']}) | {ms['teacher_positive_mass_median']:.3f} | {ms['teacher_positive_mass_above_095']:.3f} | {ms['rows_with_negative_above_positive']} ({100 * ms['rows_with_negative_above_positive'] / ms['rows']:.1f} %) | {ms['student_ranks_the_positive_first']:.3f} |")
    print("| old rows of the mined arm (20292) | 0.852 | 0.303 | 2658 (13.1 %) | 0.844 |")
    print("\nIn FAQ data the same general question has different answers on different sites, so more negatives above the positive are expected; no filter is applied (v57).")
print("\n## LLM judge (not run by this queue)\n\nCandidate file: `%s` (%s pairs). The command for the user's inference server (prompt v3):\n" % (cand, fd.get("candidates", "-")))
base = cand[:-len(".jsonl")] if cand.endswith(".jsonl") else cand
print("```bash\ncd %s\npython annotate/annotate_pairs.py --vendor vllm --base-url \"$JUDGE_BASE_URL\" --model \"$JUDGE_MODEL\" --concurrency 8 \\\n"
      "    --pairs %s \\\n    --system-file annotate/prompts/SYS_FAQ_V3.txt --user-file annotate/prompts/DATA_FAQ_V1.txt \\\n"
      "    --schema-file annotate/prompts/SCHEMA_FAQ_V2.json --prompt-version faq-v3 --max-tokens 200 --passage-chars 1500 \\\n"
      "    --output %s.v3.qwen.llm.jsonl \\\n    --journal %s.v3.qwen.journal.jsonl\n```" % (kitchen, cand, base, base))
print("\nThe verdicts can drop mined rows afterwards (pair_id = `faq:<row>`, the rows carry `faq_row`); no new mining is needed.")
if est:
    print("\n## mine-hard-negatives (step D of the kitchen): smoke only\n\nSmoke on 200 queries: rc %s, %s. Linear estimate for 100 000 queries over %s passages: **%s h** (%s). The full run was not started."
          % (est.get("smoke_rc"), "complete" if est.get("complete") else "INCOMPLETE", est.get("pool_passages"), est.get("estimated_hours_total"),
             "; ".join("%s %s h" % (k, v["estimated_hours"]) for k, v in est.get("steps", {}).items())))
print("\n## Package for the 5070\n\n`%s` (rows with negatives and teacher scores, mining sidecars, the selection manifest and funnel reports, SHA256SUMS)." % pkg)

# examples for a reader
def table(rows, cols):
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "")).replace("|", "/").replace("\n", " ") for k in cols) + " |")
if fc:
    print("\n## Examples for judging the thresholds of step c\n")
    for rng_, s in fc.get("pair_samples", {}).items():
        print(f"\n### Question pairs at cosine {rng_} ({s['pairs_in_the_neighbour_lists']} pairs in the neighbour lists)\n")
        table(s["examples"], ["cos", "a", "b", "domain_a", "domain_b"])
    print("\n### Removed generic clusters (centre question, its neighbours at cosine >= 0.85)\n")
    for cl in fc.get("cluster_examples", []):
        print("- **%s** (%s; %s neighbours, %s domains): %s" % (cl["question"], cl["domain"], cl["neighbours"], cl["domains"],
              "; ".join("%s [%s, %.2f]" % (e["question"], e["domain"], e["cos"]) for e in cl["examples"])))
    print("\n### Random pairs removed in step c\n")
    table(fc.get("rejected_examples", []), ["query", "domain", "reason"])
if fb:
    print("\n## Random pairs removed in step b\n")
    table(fb.get("rejected_examples", []), ["query", "passage", "coverage_band", "gemma", "roberta_v3", "pct_gemma", "pct_roberta_v3"])
EOF
{ echo; echo "## Queue state (v58)"; echo; echo '```'; grep -E "mined-faq20|chain:v58_" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-900 | while read -r l; do log "  READING: $l"; done
git add "$SUM" "$EXP_ROOT/queues/queue_v58.sh" "$CFG" "$KEEP" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null   # the rows and the package (datasets/ is not tracked) are described in the summary
for m in ${F_PICK:-} $F; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$F-bias0??.json 2>/dev/null
git commit -q -m "Add the v58 results: native Polish FAQ rows in the student-mined arm" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v58.log" 2>&1 && { echo pushed >> "$LOGS/push_v58.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v58 done ($(date '+%d.%m %H:%M')) ==="
