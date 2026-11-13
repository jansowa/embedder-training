#!/usr/bin/env bash
# Queue v61b (3060 Ti, after v61, before the soups of v62; ~7.5 h): translated GooAQ pairs under KD.
# datasets/qooaq.jsonl = GooAQ ("people also ask" questions of a search engine + the answer snippet) translated by
# Qwen3.5-27B: new queries in the style of search questions with short answers in natural prose (close to FAQ in form,
# to MS MARCO in domain). The lever that has worked is the number of different queries (v56: +40 k SQuAD-v2 rows =
# +0.32 / +0.23), so the rows go on top of the dose arm the way the factual rows of v61 did.
#   1  ROWS (kitchen bridge/gooaq_kd_rows.py): one row per question; out go untranslated pairs, repeated questions,
#      answers already used by an earlier question, answers with web debris or characters of another script, answers
#      under 80 characters, PIRB queries (the normalisation of v58 / v61) and queries of the dose3 rows. No reranker
#      gate, no judge. The file is read as it is when this step runs; the number of pairs is recorded.
#   2  NEGATIVES AND SCORES as in v61: r2v12 mines (5 best + 5 random of its top-60, length window 0.5-2.0) from the
#      answers + 100 000 MS MARCO-pl passages, the ensemble teacher (soup-v9b + PolDense-400M) scores with the fixed
#      dense scale. Smoke on 300 rows first.
#   3  ARM arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12 = the dose3 rows unchanged + the GooAQ rows, the recipe
#      unchanged.
#   4  MEASURE at the cost of dose3-bias015 (bias steps of 0.01, at most 1.02 x its dims/doc): tiny10, heldout8,
#      trec-covid, query-side index cost. READING against dose3-bias015; the soups of v62 take the arm when it is not
#      more than 0.2 below dose3 on either set.
#   5  summary_v61b.md, commit, push loop.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v61b.sh > logs/queue_v61b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v61b MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-28800}
source scripts/queue_lib.sh
source "$EXP_ROOT/queues/arm_common.sh"
echo $$ > "$LOGS/queue_v61b.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
wait_for_queues ${WAIT_FOR:-v61}

kscript() {  # kscript TAG TIMEOUT OUTPUT COMMAND...: a helper script, alone on the machine; OUTPUT is removed when it fails
    local tag="$1" to="$2" out="$3"; shift 3
    wait_gpu_free; wait_ram; local s=$SECONDS rc
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/v61b_$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { rm -f "$out"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/v61b_$tag.log" | cut -c1-900)"
    record "chain:v61b_$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
D3=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12; FQ=arm-dense-kl-n5kj-mined-faq20-ens400-r2v12; DF=arm-dense-kl-n5kj-mined-dose3-faq20-ens400-r2v12
FA=arm-dense-kl-n5kj-mined-dose3-fact20-ens400-r2v12
G=arm-dense-kl-n5kj-mined-dose3-gooaq-ens400-r2v12
A=datasets/arms
D3ROWS=$A/n5-plus-kj-mined-dose3.jsonl
PAIRS=${V61B_PAIRS:-datasets/qooaq.jsonl}
MSPAIRS=datasets/msmarco-full-judged-qwen/pairs_msmarco_full.unique.jsonl
GM=$A/gooaq-kd-v1-mined.jsonl
DROWS=$A/n5-plus-kj-mined-dose3-gooaq.jsonl
DRIVE=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work
OUT=${V61B_OUT:-$DRIVE/kitchen-data/gooaq-kd-v1}
REFQ=$DRIVE/kitchen-data/reference/pirb-queries.jsonl
TOOL="$KITCHEN/bridge/gooaq_kd_rows.py"
PD=OPI-PIB/PolDense-400M
SCALE=84.19441654567078
BASECFG="$C/distil_dense_kl_n5kj_mined_dose3_ens400_r2v12.yaml"
CFG="$C/distil_dense_kl_n5kj_mined_dose3_gooaq_ens400_r2v12.yaml"
COST="$RESULTS_ROOT/index_cost_v61b.json"; VERDICT="$RESULTS_ROOT/v61b_reading.json"
KEEP="$RESULTS_ROOT/v61b_gooaq"
D3_PICK=$(measured_form "$D3"); FQ_PICK=$(measured_form "$FQ"); DF_PICK=$(measured_form "$DF"); FA_PICK=$(measured_form "$FA")
log "=== v61b start: translated GooAQ rows under KD; dose arm ${D3_PICK:-none} $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK") dims/doc; fact20 ${FA_PICK:-not measured} $(lbs "$FA_PICK"); $(free_gb) GB free on / ==="

# 0. checks
for f in "$D3ROWS" "$BASECFG" "$REFQ" "$TOOL" "$PAIRS" "$MSPAIRS" "$KITCHEN/bridge/splade_mine_big_pool.py" "$KITCHEN/bridge/fact_kd_rows.py" scripts/arm_reading.py; do
    [ -s "$f" ] || { log "  $f missing: v61b stops"; exit 1; }
done
[ -n "$D3_PICK" ] && [ -d "models/$D3_PICK" ] || { log "  the measured form of the dose arm is missing: v61b stops"; exit 1; }
for m in "$R2" soup-v9b soup-v12 base-splade-wrapped; do [ -d "models/$m" ] || { log "  models/$m missing: v61b stops"; exit 1; }; done
mkdir -p "$OUT" "$KEEP" || { log "  cannot create $OUT (is the data drive mounted?): v61b stops"; exit 1; }
DATA_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
[ "${DATA_GB:-0}" -ge 10 ] || { log "  only ${DATA_GB:-0} GB free on the data drive: v61b stops"; exit 1; }
need_space || { log "  not enough space on /: v61b stops"; exit 1; }

# 1. rows
log "=== v61b step 1: rows from $PAIRS ($(grep -c . "$PAIRS") lines now, modified $(date -r "$PAIRS" '+%d.%m %H:%M')), one positive each, no gate ==="
if [ ! -s "$OUT/gooaq-rows.jsonl" ] || [ ! -s "$OUT/gooaq-pool.jsonl" ] || [ ! -s "$OUT/gooaq.stats.json" ]; then
    kscript rows 3600 "$OUT/gooaq-rows.jsonl" nice -n 10 "$PY" "$TOOL" --pairs "$PAIRS" --benchmark-queries "$REFQ" --exclude-rows "$D3ROWS" \
        --tokenizer models/base-splade-wrapped --max-positive-tokens 512 --min-answer-chars 80 --ms-pairs "$MSPAIRS" --ms-passages 100000 --seed 42 \
        --rows-output "$OUT/gooaq-rows.jsonl" --pool-output "$OUT/gooaq-pool.jsonl" --stats "$OUT/gooaq.stats.json"
fi
[ -s "$OUT/gooaq-rows.jsonl" ] && [ -s "$OUT/gooaq-pool.jsonl" ] && [ -s "$OUT/gooaq.stats.json" ] || { log "  the rows were NOT built (logs/v61b_rows.log): v61b stops"; exit 1; }
cp "$OUT/gooaq.stats.json" "$KEEP/rows_gooaq.json"
NG=$(grep -c . "$OUT/gooaq-rows.jsonl")
log "  gooaq: $NG rows of $(jget "$OUT/gooaq.stats.json" pairs_read) pairs, pool $(grep -c . "$OUT/gooaq-pool.jsonl") MS MARCO-pl passages"
[ "$NG" -ge 8000 ] || { log "  only $NG rows: v61b stops"; exit 1; }

# 2. negatives and teacher scores
mine() {  # mine TAG OUT ROWS POOL [tool args]: r2v12 mines over the rows' positives + the pool, the ensemble teacher scores with the fixed scale
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
print("%s rows %d: teacher P(pos) median %.3f, share above 0.95 %.3f, negative above positive in %d rows = %.1f %%, student ranks the positive first in %.3f; negatives by pool %s"
      % ("/".join(str(k) for k in ds), n, st.median(masses), stats["teacher_positive_mass_above_095"], flip, 100 * flip / n, first / n, dict(pools)))
PYEOF
    [ -s "$1.stats.json" ]
}
log "=== v61b step 2: negatives and teacher scores (miner $R2, fixed dense scale $SCALE; dose3 new rows: P(pos) median 0.856, above 0.95 0.296, negative above positive 10.4 %; FAQ rows: 0.982, 0.655, 6.4 %; v61: nq 0.689 / 25.0 %, hotpotqa 0.952 / 13.0 %, fever 0.967 / 9.5 %, trivia 0.664 / 30.2 %) ==="
if [ ! -s "$GM" ]; then
    rm -rf cache/smoke_v61b; mkdir -p cache/smoke_v61b
    if mine smoke cache/smoke_v61b/rows.jsonl "$OUT/gooaq-rows.jsonl" "$OUT/gooaq-pool.jsonl" --limit 300 --extra-max 20000 --extra-shard 8000 --dense-cache cache/smoke_v61b/dense \
        && [ "$(grep -c . cache/smoke_v61b/rows.jsonl)" = 300 ]; then
        log "  smoke passed: $(tail -n 1 "$LOGS/v61b_mine_smoke.log" | cut -c1-300)"
    else log "  mining smoke FAILED (logs/v61b_mine_smoke.log): v61b stops"; rm -rf cache/smoke_v61b; exit 1; fi
    rm -rf cache/smoke_v61b
    mine gooaq "$OUT/gooaq-mined.jsonl" "$OUT/gooaq-rows.jsonl" "$OUT/gooaq-pool.jsonl" --extra-max 0 --dense-cache "$OUT/dense" && mstats "$OUT/gooaq-mined.jsonl" \
        || { log "  mining FAILED (logs/v61b_mine_gooaq.log): v61b stops"; exit 1; }
    cp "$OUT/gooaq-mined.jsonl.stats.json" "$KEEP/mining_gooaq.json"; cp "$OUT/gooaq-mined.jsonl.mined.json" "$KEEP/mining_tool_gooaq.json" 2>/dev/null
    cp "$OUT/gooaq-mined.jsonl" "$GM.tmp" && mv "$GM.tmp" "$GM"
fi
NGM=$(grep -c . "$GM")
[ "$NGM" -ge 8000 ] || { log "  only $NGM mined rows: v61b stops"; exit 1; }
log "  $GM: $NGM rows"

# 3. the arm
log "=== v61b step 3: arm $G = the dose3 rows unchanged + the GooAQ rows ==="
merge_rows "$D3ROWS" "$GM" "$DROWS" || { log "  row file $DROWS not written: v61b stops"; exit 1; }
NROWS=$(jget "$DROWS.rows.json" rows); NNEW=$(jget "$DROWS.rows.json" new_rows); NPAIRS=$(jget "$DROWS.rows.json" training_pairs)
log "  $DROWS: $NROWS rows, $NNEW GooAQ rows ($(jget "$DROWS.rows.json" new_rows_left_out_duplicate_query) left out as duplicate queries), $NPAIRS training pairs"
[ "${NNEW:-0}" -ge 8000 ] || { log "  only ${NNEW:-0} GooAQ rows in the arm: v61b stops"; exit 1; }
sed -e "4s|.*|# The GooAQ arm (queue v61b): the rows of the dose arm of v56 ($D3ROWS, unchanged) plus $NNEW rows of GooAQ translated by Qwen3.5-27B (search-style questions with the answer snippet as the only positive; no reranker gate; PIRB queries removed), negatives mined by r2v12 from the answers plus 100 000 MS MARCO-pl passages (5 best + 5 random of its top-60, length window 0.5-2.0), scored by the ensemble teacher (soup-v9b + PolDense-400M) with the fixed dense scale $SCALE. The recipe of the mined arm from soup-v12, only the row file differs.|" \
    -e "s|^runs_dir: .*|runs_dir: runs/$G|" -e "s|^train_data: .*|train_data: $DROWS|" "$BASECFG" > "$CFG"
train "${G#arm-}" "$CFG"

# 4. measure at the cost of the reference
log "=== v61b step 4: measure at the cost of $D3_PICK ($(dims "$D3_PICK") dims/doc) ==="
at_cost "$G" "dose arm $D3_PICK $(lbs "$D3_PICK") / $(tc "$D3_PICK") at $(dims "$D3_PICK"); fact20 ${FA_PICK:-none} $(lbs "$FA_PICK"); dose3 + faq20 ${DF_PICK:-none} $(lbs "$DF_PICK")" "$D3_PICK"; G_PICK=${PICK:-}
[ -d "models/$G" ] || G_PICK=""
index_cost "$COST" "$M-bias013" "$D3_PICK" "$FQ_PICK" "$DF_PICK" "$FA_PICK" "$G_PICK"

# 5. summary
log "=== v61b step 5: summary ==="
SUM="$RESULTS_ROOT/summary_v61b.md"
FACOL=(); [ -n "$FA_PICK" ] && FACOL=(--column "dose3 + fact20 (v61)=$FA_PICK")
{
    echo "# v61b (quantized regime, nDCG@10): translated GooAQ pairs under KD"; echo
    echo "dose3 + gooaq = the rows of the dose arm of v56 (\`$D3ROWS\`, unchanged) plus $NNEW rows of GooAQ (\`$PAIRS\`: questions of a search engine with the answer snippet, translated by Qwen3.5-27B), one positive per query, negatives mined by r2v12 from the answers plus 100 000 MS MARCO-pl passages and the scores of the ensemble teacher with the fixed dense scale; the recipe unchanged (from soup-v12, 2 epochs). The new arm is measured in the first bias form (steps of 0.01) with at most 1.02 x the dims/doc of the reference \`$D3_PICK\`."; echo
    "$PY" scripts/arm_reading.py --regimes "$REGIMES_ROOT/quantized" --sparsity "$RESULTS_ROOT/sparsity" --cost "$COST" \
        --column "20 k arm (v49)=$M-bias013" --column "dose3 (v56)=$D3_PICK" --column "faq20 (v58)=${FQ_PICK:-$FQ}" --column "dose3 + faq20 (v60)=${DF_PICK:-$DF}" "${FACOL[@]}" \
        --column "dose3 + gooaq (v61b)=${G_PICK:-$G}" --pick "${G_PICK:-$G}" --ref "$D3_PICK" \
        --group "FAQ tasks (Allegro FAQ A, B, mfaq)=poleval-2022-test-A-allegro-faq,poleval-2022-test-B-allegro-faq,mfaq" \
        --group "tasks of the SQuAD-v2 gain (arguana, scifact, fiqa, legal questions)=arguana-pl,scifact-pl,fiqa-pl,poleval-2022-test-A-legal-questions" \
        --group "scifact-pl=scifact-pl" --group "maupqa-mkqa=maupqa-mkqa" --group "MAUPQA tasks=maupqa-*" --group "all tasks=*" \
        --win "THE GooAQ ROWS GAIN (>= +0.2 on both sets over the dose arm at its cost); the arm is a candidate of the freeze and of the soups of v62" \
        --noise "within the seed noise of the dose arm; the soups of v62 take the arm when it is not more than 0.2 below dose3 on either set" \
        --loss "a LOSS on both sets against the dose arm" --json "$VERDICT"
    echo; echo "Nothing further was started by this queue."
    echo; echo "## Rows (left after each filter)"; echo
    "$PY" - "$KEEP" <<'PYEOF'
import json, sys
from pathlib import Path
keep = Path(sys.argv[1])
d = json.load(open(keep / "rows_gooaq.json"))
print("%d rows of %d pairs read (file modified %s; %d unreadable lines). Questions %s words (p10 / p50 / p90), answers %s characters, positives median %s tokens (max %s). Pool: %d of the %d distinct MS MARCO-pl passages of `%s` (seed %d) plus the answers of the rows.\n"
      % (d["rows"], d["pairs_read"], d["pairs_file_modified"], d["unreadable_lines"], " / ".join(map(str, d["question_words_p10_p50_p90"])), " / ".join(map(str, d["answer_chars_p10_p50_p90"])),
         d["positive_tokens_median"], d["positive_tokens_max"], d["pool_passages"], d["pool_distinct_passages_in_source"], d["pool_from"], d["seed"]))
print("| step | removed | left |\n|---|---:|---:|")
for s in d["funnel"]:
    print("| %s | %s | %s |" % (s["step"], s["removed"], s["left"]))
print("\nDebris patterns (case-insensitive regular expressions on the answer): " + "; ".join("%s = `%s`" % kv for kv in d["debris_patterns"].items())
      + ". Other-script characters (question or answer): `%s`. An untranslated side = the question or the answer equal to its English original. No Qwen labels exist for these pairs, so the broken-Polish veto of v61 does not apply; the other-script check is the only cheap detector.\n" % d["other_script_pattern"])
print("## Mining (teacher P(pos) at T = 2)\n")
print("| rows | n | teacher P(pos) median | share with P(pos) > 0.95 | rows with a negative above the positive | student ranks the positive first |\n|---|---:|---:|---:|---:|---:|")
def row(name, m):
    print("| %s | %d | %.3f | %.3f | %d (%.1f %%) | %.3f |" % (name, m["rows"], m["teacher_positive_mass_median"], m["teacher_positive_mass_above_095"],
          m["rows_with_negative_above_positive"], 100 * m["rows_with_negative_above_positive"] / m["rows"], m["student_ranks_the_positive_first"]))
m = json.load(open(keep / "mining_gooaq.json")); row("gooaq (v61b)", m)
for p in sorted((keep.parent / "v61_fact").glob("mining_[a-z]*.json")):
    if not p.name.startswith("mining_tool_"):
        row(p.stem.replace("mining_", "") + " (v61)", json.load(open(p)))
print("| new SQuAD-v2 rows of dose3 (v56) | 40093 | 0.856 | 0.296 | 4168 (10.4 %) | 0.867 |")
print("| FAQ rows (v58) | 60000 | 0.982 | 0.655 | 3864 (6.4 %) | 0.868 |")
print("| rows of the 20 k arm | 20292 | 0.852 | 0.303 | 2658 (13.1 %) | 0.844 |")
print("\nNegatives by pool: %s." % ", ".join("%s %d" % kv for kv in sorted(m["negatives_by_pool"].items())))
PYEOF
    echo; echo "## Rows of the arm"; echo; echo '```'; cat "$DROWS.rows.json"; echo; echo '```'
    echo; echo "## Queue state (v61b)"; echo; echo '```'; grep -E "mined-dose3-gooaq|chain:v61b_" "$STATE" | tail -n 60; echo '```'
} > "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-900 | while read -r l; do log "  READING: $l"; done
git add "$SUM" "$EXP_ROOT/queues/queue_v61b.sh" "$CFG" "$VERDICT" "$KEEP" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null   # the rows (datasets/ and the data drive are not tracked) are described in the summary
for m in ${G_PICK:-} $G; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$G-bias0??.json 2>/dev/null
git commit -q -m "Add the v61b results: translated GooAQ pairs under KD" 2>/dev/null && log "committed $(git log --oneline -1)"
push_loop v61b
log "=== v61b done ($(date '+%d.%m %H:%M')) ==="
