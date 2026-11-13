#!/usr/bin/env bash
# Queue v61 (3060 Ti, after v60, ~10 h): translated factual sets (NQ, HotpotQA, FEVER, TriviaQA) under KD.
# These sets were dropped almost only on results under MNRL (v25, v26, v29); under KD only NQ was tried, once, with the
# weakest setup (v33: soup-v9b teacher, old negatives: 63.41 / 42.62 against 63.27 / 42.63 without NQ, where MNRL lost a
# point). Their defects are label errors, which a teacher scores low, and the rule "translated factual data do not help"
# contradicts v56: SQuAD-v2 is translated factual data and gave +0.32 / +0.23. This queue settles it with the current recipe.
#   1  ROWS (kitchen tool bridge/fact_kd_rows.py, about 5 000 per source, seed 42): one positive per query (the best by the
#      source's reranker score, at most 512 tokens), no reranker gate, no judge as a gate; left out are PIRB queries
#      (normalised text / sorted tokens / loose key), queries of the dose3 rows and repeated queries.
#        NQ        without popular culture and sport. The domain is a Qwen label that exists only for the 8 000-row pool, so
#                  the NQ rows come from datasets/arms/pool/nq-pool-8000.jsonl (about 3 600 rows), not from the full file.
#        HotpotQA  only questions one passage answers (Qwen answer_location explicit / inferred on the chosen positive):
#                  that label too exists only for the 8 000-row pool.
#        FEVER     the full file (every row has an evidence passage, so NOT ENOUGH INFO claims are not in it).
#        TriviaQA  the full file of the doc2query project (trivia-mined-negatives/train_pl.jsonl), rows with a translation.
#      Broken Polish by a Qwen label is a veto where a label exists (the pools of NQ, HotpotQA, FEVER).
#   2  NEGATIVES and teacher scores, one pool per source (chosen positives + original negatives of that source, up to
#      120 000 passages): r2v12 mines 5 best + 5 random of its top-60 in the length window 0.5-2.0, the ensemble teacher
#      (soup-v9b + PolDense-400M, fixed dense scale) scores. Smoke on 300 rows first.
#   3  ARM arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12 = the dose3 rows unchanged + the factual rows; the recipe
#      unchanged. Before it a memory smoke: 2 epochs over the 64 longest FEVER rows; on a CUDA out-of-memory the FEVER texts
#      are cut to 256 tokens (the batch is not changed), and when that fails too the FEVER rows are left out.
#   4  MEASURE at the cost of dose3-bias015: tiny10, heldout8, trec-covid, index cost; czywiesz and poquad for this arm and
#      for dose3-bias015. READING against dose3-bias015 by +0.2 on both sets.
#   5  summary_v61.md, commit, push loop. No further arm starts here.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v61.sh > logs/queue_v61.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v61 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-28800}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
echo $$ > "$LOGS/queue_v61.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
wait_for_queues ${WAIT_FOR:-v60}

kscript() {  # kscript TAG TIMEOUT OUTPUT COMMAND...: a helper script, alone on the machine; OUTPUT is removed when it fails
    local tag="$1" to="$2" out="$3"; shift 3
    wait_gpu_free; wait_ram; local s=$SECONDS rc
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/v61_$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { rm -f "$out"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/v61_$tag.log" | cut -c1-600)"
    record "chain:v61_$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
D3=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12; FQ=arm-dense-kl-n5kj-mined-faq20-ens400-r2v12; DF=arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12
F=arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12
A=datasets/arms
D3ROWS=$A/n5-plus-kj-mined-dose3.jsonl
FACTM=$A/fact-kd-v1-mined.jsonl
DROWS=$A/n5-plus-kj-mined-dose3-fact20.jsonl
DRIVE=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work
OUT=${V61_OUT:-$DRIVE/kitchen-data/fact-kd-v1}
REFQ=$DRIVE/kitchen-data/reference/pirb-queries.jsonl
SL=$DRIVE/FlagEmbedding-data/speakleash
TRIVIA=$DRIVE/doc2query/data/raw/trivia-mined-negatives/train_pl.jsonl
ANN="$KITCHEN/data/annotation/pairs_v1.llm.jsonl"
TOOL="$KITCHEN/bridge/fact_kd_rows.py"
PD=OPI-PIB/PolDense-400M
SCALE=84.19441654567078
BASECFG="$C/distil_dense_kl_n5kj_mined_dose3_ens400_r2v12.yaml"
CFG="$C/distil_dense_kl_n5kj_mined_dose3_fact20_ens400_r2v12.yaml"
COST="$RESULTS_ROOT/index_cost_v61.json"; VERDICT="$RESULTS_ROOT/v61_reading.json"
KEEP="$RESULTS_ROOT/v61_fact"
POOLMAX=${V61_POOL_MAX:-120000}
D3_PICK=$(measured_form "$D3"); FQ_PICK=$(measured_form "$FQ"); DF_PICK=$(measured_form "$DF")
log "=== v61 start: translated factual rows under KD; dose arm ${D3_PICK:-none} $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK") dims/doc; $(free_gb) GB free on / ==="

# 0. checks
for f in "$D3ROWS" "$BASECFG" "$REFQ" "$TOOL" "$ANN" "$KITCHEN/bridge/splade_mine_big_pool.py" scripts/arm_reading.py \
         "$SL/nq/train_pl.jsonl" "$SL/hotpotqa/train_pl.jsonl" "$SL/fever/train_pl.jsonl" "$A/pool/nq-pool-8000.jsonl" "$A/pool/hotpotqa-pool-8000.jsonl"; do
    [ -s "$f" ] || { log "  $f missing: v61 stops"; exit 1; }
done
[ -n "$D3_PICK" ] && [ -d "models/$D3_PICK" ] || { log "  the measured form of the dose arm is missing: v61 stops"; exit 1; }
for m in "$R2" soup-v9b soup-v12 base-splade-wrapped; do [ -d "models/$m" ] || { log "  models/$m missing: v61 stops"; exit 1; }; done
mkdir -p "$OUT" "$KEEP" || { log "  cannot create $OUT (is the data drive mounted?): v61 stops"; exit 1; }
DATA_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
[ "${DATA_GB:-0}" -ge 10 ] || { log "  only ${DATA_GB:-0} GB free on the data drive: v61 stops"; exit 1; }
need_space || { log "  not enough space on /: v61 stops"; exit 1; }
SOURCES="nq hotpotqa fever trivia"; PER=${V61_PER_SOURCE:-5000}
[ -s "$TRIVIA" ] || { SOURCES="nq hotpotqa fever"; PER=${V61_PER_SOURCE:-6700}; log "  the full TriviaQA file is missing ($TRIVIA): three sources, up to $PER rows each"; }

# 1. rows
log "=== v61 step 1: rows, up to $PER per source ($SOURCES), one positive each, no gate ==="
sel() {  # sel SOURCE [tool args]: rows and the passage pool of one source
    local src="$1"; shift
    [ -s "$OUT/$src-rows.jsonl" ] && [ -s "$OUT/$src-pool.jsonl" ] && [ -s "$OUT/$src.stats.json" ] && return 0
    kscript "rows_$src" 5400 "$OUT/$src-rows.jsonl" nice -n 10 "$PY" "$TOOL" --source "$src" --benchmark-queries "$REFQ" --exclude-rows "$D3ROWS" \
        --tokenizer models/base-splade-wrapped --max-positive-tokens 512 --sample "$PER" --seed 42 --pool-max "$POOLMAX" \
        --rows-output "$OUT/$src-rows.jsonl" --pool-output "$OUT/$src-pool.jsonl" --stats "$OUT/$src.stats.json" "$@"
}
OK=""
for src in $SOURCES; do
    case "$src" in
        nq)       sel nq --rows-from "$A/pool/nq-pool-8000.jsonl" --pool-from "$SL/nq/train_pl.jsonl" --annotations "$ANN" --annotation-source nq-pool-8000 \
                      --exclude-domain culture_entertainment sport --polish-veto ;;
        hotpotqa) sel hotpotqa --rows-from "$A/pool/hotpotqa-pool-8000.jsonl" --pool-from "$SL/hotpotqa/train_pl.jsonl" --annotations "$ANN" --annotation-source hotpotqa-pool-8000 \
                      --answer-location explicit inferred --polish-veto ;;
        fever)    sel fever --rows-from "$SL/fever/train_pl.jsonl" --annotations "$ANN" --annotation-source fever-pool-8000 --polish-veto ;;
        trivia)   sel trivia --rows-from "$TRIVIA" ;;
    esac
    if [ -s "$OUT/$src-rows.jsonl" ] && [ -s "$OUT/$src-pool.jsonl" ]; then
        OK="$OK $src"; cp "$OUT/$src.stats.json" "$KEEP/rows_$src.json"
        log "  $src: $(grep -c . "$OUT/$src-rows.jsonl") rows, pool $(grep -c . "$OUT/$src-pool.jsonl") passages"
    else log "  $src: the rows were NOT built (logs/v61_rows_$src.log): the source is left out"; fi
done
[ "$(echo $OK | wc -w)" -ge 2 ] || { log "  fewer than two sources have rows: v61 stops"; exit 1; }

# 2. negatives and teacher scores, one pool per source
mine() {  # mine TAG OUT ROWS POOL [tool args]: r2v12 mines over the rows' positives + the source's pool, the ensemble teacher scores with the fixed scale
    local tag="$1" out="$2" rows="$3" pool="$4"; shift 4
    [ -s "$out" ] && return 0
    kscript "mine_$tag" 14400 "$out" "$PY" "$KITCHEN/bridge/splade_mine_big_pool.py" --student "models/$R2" --splade-teacher models/soup-v9b \
        --dense-teacher "$PD" --rows "$rows" --extra-pool "$pool" --extra-chars 1 1000000 --dense-scale "$SCALE" --output "$out" "$@"
}
mstats() {  # mstats FILE: the teacher's view of mined rows -> FILE.stats.json and one log line
    "$PY" - "$1" "$SCALE" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, math, statistics as st, sys
from collections import Counter
path, scale = sys.argv[1], float(sys.argv[2])
masses, flip, first, n, pools, ds = [], 0, 0, 0, Counter(), Counter()
for line in open(path, encoding="utf-8"):
    r = json.loads(line); n += 1; ds[r.get("dataset")] += 1
    assert len(r["neg"]) == 10 == len(r["neg_scores"]) and len(r["pos_scores"]) == len(r["pos"]) == 1, "row shape: " + str(r["query_id"])
    assert abs(r["dense_scale"] - scale) < 1e-9, "dense scale differs from the fixed scale"
    s = [max(r["pos_scores"])] + list(r["neg_scores"]); m = max(s); e = [math.exp((x - m) / 2.0) for x in s]
    masses.append(e[0] / sum(e)); flip += max(r["neg_scores"]) > min(r["pos_scores"]); first += min(r["pos_student_rank"]) == 1
    pools.update(r.get("neg_pool", []))
stats = {"rows": n, "datasets": dict(ds), "teacher_positive_mass_median": st.median(masses), "teacher_positive_mass_above_095": sum(x > 0.95 for x in masses) / n,
         "rows_with_negative_above_positive": flip, "student_ranks_the_positive_first": first / n, "negatives_by_pool": dict(pools), "dense_scale": scale}
json.dump(stats, open(path + ".stats.json", "w"), indent=2)
print("%s rows %d: teacher P(pos) median %.3f, share above 0.95 %.3f, negative above positive in %d rows = %.1f %%, student ranks the positive first in %.3f"
      % ("/".join(str(k) for k in ds), n, st.median(masses), stats["teacher_positive_mass_above_095"], flip, 100 * flip / n, first / n))
PYEOF
    [ -s "$1.stats.json" ]
}
log "=== v61 step 2: negatives and teacher scores per source (miner $R2, fixed dense scale $SCALE; dose3 new rows: P(pos) median 0.856, above 0.95 0.296, negative above positive 10.4 %; FAQ rows: 0.982, 0.655, 6.4 %) ==="
if [ ! -s "$FACTM" ]; then
    first=$(echo $OK | awk '{print $1}')
    rm -rf cache/smoke_v61; mkdir -p cache/smoke_v61
    if mine smoke cache/smoke_v61/rows.jsonl "$OUT/$first-rows.jsonl" "$OUT/$first-pool.jsonl" --limit 300 --extra-max 20000 --extra-shard 8000 --dense-cache cache/smoke_v61/dense \
        && [ "$(grep -c . cache/smoke_v61/rows.jsonl)" = 300 ]; then
        log "  smoke passed ($first): $(tail -n 1 "$LOGS/v61_mine_smoke.log" | cut -c1-300)"
    else log "  mining smoke FAILED (logs/v61_mine_smoke.log): v61 stops"; rm -rf cache/smoke_v61; exit 1; fi
    rm -rf cache/smoke_v61
    MINED=""
    for src in $OK; do
        if mine "$src" "$OUT/$src-mined.jsonl" "$OUT/$src-rows.jsonl" "$OUT/$src-pool.jsonl" --extra-max 0 --dense-cache "$OUT/dense-$src" && mstats "$OUT/$src-mined.jsonl"; then
            MINED="$MINED $src"; cp "$OUT/$src-mined.jsonl.stats.json" "$KEEP/mining_$src.json"; cp "$OUT/$src-mined.jsonl.mined.json" "$KEEP/mining_tool_$src.json" 2>/dev/null
        else log "  $src: mining FAILED (logs/v61_mine_$src.log): the source is left out"; fi
    done
    [ "$(echo $MINED | wc -w)" -ge 2 ] || { log "  fewer than two sources were mined: v61 stops"; exit 1; }
    : > "$FACTM.tmp"; for src in $MINED; do cat "$OUT/$src-mined.jsonl" >> "$FACTM.tmp"; done
    mv "$FACTM.tmp" "$FACTM"
fi
NFACT=$(grep -c . "$FACTM")
[ "$NFACT" -ge 8000 ] || { log "  only $NFACT factual rows: v61 stops"; exit 1; }
log "  $FACTM: $NFACT rows"

# 3. the arm
log "=== v61 step 3: arm $F = the dose3 rows unchanged + the factual rows ==="
smoke_train() {  # smoke_train ROWS: 2 epochs over a few rows with the arm's recipe; 0 = fine, 2 = CUDA out of memory, 1 = another failure
    local rows="$1" rc
    rm -rf runs/smoke-v61-fever cache/smoke_v61; mkdir -p cache/smoke_v61
    sed -e "s|^runs_dir: .*|runs_dir: runs/smoke-v61-fever|" -e "s|^train_data: .*|train_data: $rows|" "$BASECFG" > cache/smoke_v61/train.yaml
    wait_gpu_free; wait_ram; local s=$SECONDS
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 2400 "$PY" run_experiments.py --grid-configuration-file cache/smoke_v61/train.yaml \
        --backend sentence-transformers --training-type splade > "$LOGS/v61_train_smoke.log" 2>&1; rc=$?
    ls -d runs/smoke-v61-fever/sentence-transformers/splade/*/final > /dev/null 2>&1 && rc=0 || { [ $rc = 0 ] && rc=1; }
    rm -rf runs/smoke-v61-fever cache/smoke_v61
    if [ $rc != 0 ] && grep -qi "out of memory" "$LOGS/v61_train_smoke.log"; then rc=2; fi
    log "  training smoke on $(grep -c . "$rows") rows: rc=$rc in $((SECONDS - s))s"
    return $rc
}
longest_fever() {  # longest_fever ROWS OUT N: the N fever-kd rows with the most characters in their positive and negatives
    "$PY" -c "
import json, sys
rows = [l for l in open(sys.argv[1], encoding='utf-8') if json.loads(l).get('dataset') == 'fever-kd']
rows.sort(key=lambda l: -sum(len(t) for t in json.loads(l)['pos'] + json.loads(l)['neg']))
open(sys.argv[2], 'w', encoding='utf-8').writelines(rows[:int(sys.argv[3])])
print(len(rows))" "$1" "$2" "$3"
}
FEVER_NOTE="none of the factual rows were changed for memory"
if [ ! -s "$DROWS" ] && [ "$(longest_fever "$FACTM" "$OUT/smoke-train-rows.jsonl" 64)" -ge 64 ]; then
    smoke_train "$OUT/smoke-train-rows.jsonl"; rc=$?
    if [ $rc = 2 ]; then
        log "  CUDA out of memory on the longest FEVER rows: the FEVER texts are cut to 256 tokens (the batch is not changed)"
        "$PY" - "$FACTM" 256 <<'PYEOF' | while read -r l; do log "  $l"; done
import json, os, sys
from transformers import AutoTokenizer
path, cap = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained("models/base-splade-wrapped")
def cut(text):
    enc = tok(text, add_special_tokens=True, truncation=True, max_length=cap, return_offsets_mapping=True)
    end = max((b for a, b in enc["offset_mapping"]), default=len(text))
    return text if end >= len(text) else text[:end]
n = changed = 0
with open(path, encoding="utf-8") as f, open(path + ".tmp", "w", encoding="utf-8") as g:
    for line in f:
        r = json.loads(line)
        if r.get("dataset") == "fever-kd":
            n += 1; before = (r["pos"], r["neg"])
            r["pos"] = [cut(t) for t in r["pos"]]; r["neg"] = [cut(t) for t in r["neg"]]; r["fact_cut_to_tokens"] = cap
            changed += (r["pos"], r["neg"]) != before
        g.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(path + ".tmp", path)
print("FEVER rows %d, %d with a text cut to %d tokens (the teacher's scores are those of the uncut texts)" % (n, changed, cap))
PYEOF
        FEVER_NOTE="the texts of the FEVER rows were cut to 256 tokens after a CUDA out-of-memory in the training smoke; their teacher scores are those of the uncut texts"
        longest_fever "$FACTM" "$OUT/smoke-train-rows.jsonl" 64 > /dev/null
        smoke_train "$OUT/smoke-train-rows.jsonl"; rc=$?
        if [ $rc = 2 ]; then
            log "  still out of memory at 256 tokens: the FEVER rows are left out of the arm"
            "$PY" -c "
import json, os, sys
p = sys.argv[1]; keep = [l for l in open(p, encoding='utf-8') if json.loads(l).get('dataset') != 'fever-kd']
open(p + '.tmp', 'w', encoding='utf-8').writelines(keep); os.replace(p + '.tmp', p)" "$FACTM"
            FEVER_NOTE="the FEVER rows were left out: the training smoke ran out of memory even with their texts cut to 256 tokens"
        fi
    fi
    [ $rc = 1 ] && log "  the training smoke failed for another reason (logs/v61_train_smoke.log); the arm is tried anyway"
fi
echo "$FEVER_NOTE" > "$KEEP/fever_note.txt"
merge_rows "$D3ROWS" "$FACTM" "$DROWS" || { log "  row file $DROWS not written: v61 stops"; exit 1; }
NROWS=$(jget "$DROWS.rows.json" rows); NNEW=$(jget "$DROWS.rows.json" new_rows); NPAIRS=$(jget "$DROWS.rows.json" training_pairs)
log "  $DROWS: $NROWS rows, $NNEW factual rows ($(jget "$DROWS.rows.json" new_rows_left_out_duplicate_query) left out as duplicate queries), $NPAIRS training pairs; $FEVER_NOTE"
[ "${NNEW:-0}" -ge 8000 ] || { log "  only ${NNEW:-0} factual rows in the arm: v61 stops"; exit 1; }
sed -e "4s|.*|# The factual arm (queue v61): the rows of the dose arm of v56 ($D3ROWS, unchanged) plus $NNEW rows of translated factual sets (NQ without popular culture, HotpotQA questions one passage answers, FEVER, TriviaQA; one positive per query, no reranker gate, PIRB queries removed), negatives mined by r2v12 from a pool of each source's own passages (5 best + 5 random of its top-60, length window 0.5-2.0), scored by the ensemble teacher (soup-v9b + PolDense-400M) with the fixed dense scale $SCALE. The recipe of the mined arm from soup-v12, only the row file differs.|" \
    -e "s|^runs_dir: .*|runs_dir: runs/$F|" -e "s|^train_data: .*|train_data: $DROWS|" "$BASECFG" > "$CFG"
train "${F#arm-}" "$CFG"

# 4. measure at the cost of the reference
log "=== v61 step 4: measure at the cost of $D3_PICK ($(dims "$D3_PICK") dims/doc) ==="
at_cost "$F" "dose arm $D3_PICK $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK"); FAQ arm ${FQ_PICK:-none} $(lbs "$FQ_PICK"); dose3 + faq20 ${DF_PICK:-none} $(lbs "$DF_PICK")" "$D3_PICK"; F_PICK=${PICK:-}
[ -d "models/$F" ] || F_PICK=""
index_cost "$COST" "$M-bias013" "$D3_PICK" "$FQ_PICK" "$DF_PICK" "$F_PICK"
for m in $F_PICK $D3_PICK; do
    for ts in heldout2_czywiesz heldout2_poquad; do h2task "$m" "$ts"; log "  $m $ts $(lbv "$m" "$ts")"; done
done

# 5. summary
log "=== v61 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_v61.md"
{
    echo "# v61 (quantized regime, nDCG@10): translated factual sets (NQ, HotpotQA, FEVER, TriviaQA) under KD"; echo
    echo "fact20 = the rows of the dose arm of v56 (\`$D3ROWS\`, unchanged) plus $NNEW rows of translated factual sets with one positive per query, negatives mined by r2v12 from a pool of each source's own passages and the scores of the ensemble teacher with the fixed dense scale; the recipe unchanged (from soup-v12, 2 epochs). The base is dose3, not dose3 + faq20, so the arm answers its own question and compares with v56, v58 and v60. The new arm is measured in the first bias form (steps of 0.01) with at most 1.02 x the dims/doc of the reference \`$D3_PICK\`."; echo
    "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" \
        --column "20 k arm (v49)=$M-bias013" --column "dose3 (v56)=$D3_PICK" --column "faq20 (v58)=${FQ_PICK:-$FQ}" --column "dose3 + faq20 (v60)=${DF_PICK:-$DF}" \
        --column "dose3 + fact20 (v61)=${F_PICK:-$F}" --pick "${F_PICK:-$F}" --ref "$D3_PICK" --extra-set heldout2_czywiesz --extra-set heldout2_poquad \
        --group "wiki-factual tasks (MAUPQA incl. czywiesz, poquad, 1z10, wiki-def)=maupqa-*" \
        --group "legal questions and mfaq (old TriviaQA pattern)=poleval-2022-test-A-legal-questions,mfaq" \
        --group "Allegro FAQ A, B (old NQ pattern)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq" \
        --group "arguana-pl (old HotpotQA pattern)=arguana-pl" --group "maupqa-czy-wiesz-v2 (old FEVER pattern)=maupqa-czy-wiesz-v2" --group "maupqa-poquad=maupqa-poquad" \
        --group "all tasks=*" \
        --win "TRANSLATED FACTUAL ROWS WORK UNDER KD (>= +0.2 on both sets over the dose arm at its cost). Proposed, not started: the split by source and the combination with the FAQ rows" \
        --noise "within the seed noise of the dose arm: these translated factual sources are neutral also under KD with the current recipe (ensemble teacher, student-mined negatives)" \
        --loss "a LOSS on both sets against the dose arm; the task groups below point at a source by the old patterns (TriviaQA: legal questions and mfaq, NQ: Allegro FAQ, HotpotQA: arguana, FEVER: czywiesz)" \
        --json "$VERDICT"
    echo; echo "The group means include czywiesz and poquad only for the models that have them (the count is in brackets when it differs). Nothing further was started by this queue."
    echo; echo "## Rows per source (left after each filter)"; echo
    "$PY" - "$KEEP" <<'PYEOF'
import json, sys
from pathlib import Path
keep = Path(sys.argv[1])
for p in sorted(keep.glob("rows_*.json")):
    d = json.load(open(p)); r = d["rules"]
    print("### %s: %d rows, pool %d passages\n" % (d["source"], d["rows"], d["pool_passages"]))
    print("Rows from `%s`; pool from %s. Rules: excluded domains %s; answer location %s; broken-Polish veto %s (annotation records: %s); positive at most %s tokens "
          "(median of the chosen positives %s, max %s); positives per row in the source %s.\n"
          % (d["rows_from"], ", ".join("`%s`" % x for x in d["pool_from"]), r["exclude_domain"] or "none", r["answer_location"] or "any", "yes" if r["polish_veto"] else "no",
             r["annotation_records"], r["max_positive_tokens"], d["positive_tokens_median"], d["positive_tokens_max"], d["positives_per_row_in_source"]))
    print("| step | removed | left |\n|---|---:|---:|")
    for s in d["funnel"]:
        print("| %s | %s | %s |" % (s["step"], s["removed"], s["left"]))
    print()
print("## Mining per source (teacher P(pos) at T = 2)\n")
print("| rows | n | teacher P(pos) median | share with P(pos) > 0.95 | rows with a negative above the positive | student ranks the positive first |\n|---|---:|---:|---:|---:|---:|")
for p in sorted(keep.glob("mining_[a-z]*.json")):
    if p.name.startswith("mining_tool_"):
        continue
    m = json.load(open(p))
    print("| %s | %d | %.3f | %.3f | %d (%.1f %%) | %.3f |" % (p.stem.replace("mining_", ""), m["rows"], m["teacher_positive_mass_median"], m["teacher_positive_mass_above_095"],
          m["rows_with_negative_above_positive"], 100 * m["rows_with_negative_above_positive"] / m["rows"], m["student_ranks_the_positive_first"]))
print("| new SQuAD-v2 rows of dose3 (v56) | 40093 | 0.856 | 0.296 | 4168 (10.4 %) | 0.867 |")
print("| FAQ rows (v58) | 60000 | 0.982 | 0.655 | 3864 (6.4 %) | 0.868 |")
print("| rows of the 20 k arm | 20292 | 0.852 | 0.303 | 2658 (13.1 %) | 0.844 |")
PYEOF
    echo; echo "Memory: $FEVER_NOTE."
    echo; echo "## Rows of the arm"; echo; echo '```'; cat "$DROWS.rows.json"; echo; echo '```'
    echo; echo "## Queue state (v61)"; echo; echo '```'; grep -E "mined-dose3-fact20|chain:v61_" "$STATE" | tail -n 60; echo '```'
} > "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-900 | while read -r l; do log "  READING: $l"; done
git add "$SUM" "$EXP_ROOT/queues/queue_v61.sh" "$CFG" "$VERDICT" "$KEEP" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null   # the rows (datasets/ and the data drive are not tracked) are described in the summary
for m in ${F_PICK:-} $F $D3_PICK; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$F-bias0??.json 2>/dev/null
git commit -q -m "Add the v61 results: translated factual sets under KD" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop v61
log "=== v61 done ($(date '+%d.%m %H:%M')) ==="
