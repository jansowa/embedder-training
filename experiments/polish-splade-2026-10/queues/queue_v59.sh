#!/usr/bin/env bash
# Queue v59 (3060 Ti, after v58, ~1.5 h of diagnostics and ~5 h for the conditional arm): on which MS MARCO queries does the
# sparse model lose to the dense one, and do such rows help? Random MS MARCO rows gave the polish-splade family nothing
# (18 k + 20 k MS: 63.42 / 42.60 against 63.39 / 42.80), but with old easy negatives and a teacher of the student's own
# family. The user's idea: take the queries PolDense-400M gets right and SPLADE does not: there the ensemble teacher has
# something to pass on. Hardness alone would also pick pairs with no lexical anchor, so the choice has two axes (the
# student's rank of the positive, the lexical coverage) and the data decide which cells are reachable: the base
# polish-splade against the current best model on the same pairs.
#   1  DIAGNOSTICS (always): 200 000 queries of the judged full MS MARCO-pl pairs (seed 42, one positive each, no PIRB
#      queries, no queries of the mined arm), pool = their positives; ranks of the positive for base polish-splade, the
#      publication form (mined arm, bias 0.13) and PolDense-400M; coverage bands of the FAQ selection; the tables.
#   2  H = relevant + natural Polish, dense top 3, current model at rank 2-50, coverage below 0.8, in a reachable band
#      (>= 25 % of the base model's rank 2-50 pairs reached rank 1). |H| < 8 000: the queue ends with the diagnostics.
#      |H| > 20 000: 20 000, equally from the kept bands.
#   3  ROWS OF H, no training: negatives mined by r2v12 from the 200 000 positives (5 best + 5 random of the top-60, length
#      window 0.5-2.0) and scored by the ensemble teacher with the fixed dense scale -> datasets/arms/msmarco-hard-mined.jsonl.
#      Changed on 12.11 19:31, while the rank step ran: the dose arm (v56) and the FAQ arm (v58) both gained, so queue v60
#      (dose3 + FAQ) runs first and the arm with H is trained by queue v61 on the rows of v60's winner. The first instance of
#      this queue was killed (SIGKILL, no trap) and this one waits for its rank step (V59_WAIT_PID).
#   4  summary_v59.md (tables, |H|, the mining), commit, push loop. No arm starts here.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v59.sh > logs/queue_v59.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v59 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-18000}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v59.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v56 v57 v58}; do
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
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/v59_$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { rm -f "$out"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/v59_$tag.log" | cut -c1-320)"
    record "chain:v59_$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
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
D3=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12; FQ=arm-dense-kl-n5kj-mined-faq20-ens400-r2v12
F=arm-dense-kl-n5kj-mined-mshard-ens400-r2v12
BASE=base-splade-wrapped; CUR=$M-bias013
A=datasets/arms
MROWS=$A/n5-plus-kj-mined-tens400.jsonl      # the rows of the mined arm (v49), used verbatim
MS=datasets/msmarco-full-judged-qwen
DATA=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/kitchen-data
OUT=${V59_OUT:-$DATA/msmarco-hard}
REFQ=$DATA/reference/pirb-queries.jsonl
TOOL="$KITCHEN/bridge/msmarco_hard_select.py"
POOL="$OUT/pool.jsonl"; HROWS="$OUT/h_rows.jsonl"
DCACHE="$OUT/dense-pd400"                     # one cache for the diagnostics and the mining
HM=$A/msmarco-hard-mined.jsonl
DROWS=$A/n5-plus-kj-mined-mshard.jsonl
PD=OPI-PIB/PolDense-400M
SCALE=84.19441654567078
BASECFG="$C/distil_dense_kl_n5kj_mined_ens400_r2v12.yaml"
CFG="$C/distil_dense_kl_n5kj_mined_mshard_ens400_r2v12.yaml"
KEEP="$RESULTS_ROOT/v59_mshard"
hget() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['H'][sys.argv[2]])" "$OUT/tables.json" "$1" 2>/dev/null; }
log "=== v59 start: targeted MS MARCO rows; mined arm bias010 $(lbs $M-bias010) / $(tc $M-bias010) at $(dims $M-bias010) dims/doc, bias013 $(lbs $M-bias013) at $(dims $M-bias013); $(free_gb) GB free on / ==="

# 0. checks
for f in "$MS/pairs_msmarco_full.unique.jsonl" "$REFQ" "$TOOL" "$KITCHEN/bridge/splade_mine_big_pool.py" "$MROWS" "$BASECFG"; do
    [ -s "$f" ] || { log "  $f missing: v59 stops"; exit 1; }
done
ls $MS/pairs_msmarco_full.llm.*.jsonl.gz > /dev/null 2>&1 || { log "  no verdict files in $MS: v59 stops"; exit 1; }
for m in "$BASE" "$CUR" "$R2" soup-v9b soup-v12; do [ -d "models/$m" ] || { log "  models/$m missing: v59 stops"; exit 1; }; done
mkdir -p "$OUT" "$KEEP" || { log "  cannot create $OUT (is the data drive mounted?): v59 stops"; exit 1; }
DATA_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
[ "${DATA_GB:-0}" -ge 10 ] || { log "  only ${DATA_GB:-0} GB free on the data drive: v59 stops"; exit 1; }
need_space || { log "  not enough space on /: v59 stops"; exit 1; }
# takeover: the rank step of the killed first instance runs on as an orphan; wait for it instead of starting it again
if [ -n "${V59_WAIT_PID:-}" ]; then
    log "  takeover: waiting for the running rank step (pid $V59_WAIT_PID) to exit"
    while kill -0 "$V59_WAIT_PID" 2>/dev/null; do sleep 20; done
    if [ -s "$OUT/ranks.npz" ] && grep -q "^dense: positive at rank 1" "$LOGS/v59_ranks.log"; then
        date > "$OUT/ranks.done"; record "chain:v59_ranks" ok "-" "takeover"
        log "  the rank step finished: $(tail -n 1 "$LOGS/v59_ranks.log" | cut -c1-300)"
    else
        cp "$LOGS/v59_ranks.log" "$LOGS/v59_ranks.first.log" 2>/dev/null
        log "  the taken-over rank step did not finish (logs/v59_ranks.first.log): it is run again"
    fi
fi

# 1. diagnostics
log "=== v59 step 1: diagnostics on ${V59_N:-200000} MS MARCO queries (base = $BASE, current = $CUR, dense = $PD) ==="
[ -s "$OUT/sample.jsonl" ] && [ -s "$POOL" ] || kscript sample 3600 "$OUT/sample.jsonl" nice -n 10 "$PY" "$TOOL" sample --pairs "$MS/pairs_msmarco_full.unique.jsonl" \
    --verdicts "$MS/pairs_msmarco_full.llm.*.jsonl.gz" --benchmark-queries "$REFQ" --exclude-rows "$MROWS" --n "${V59_N:-200000}" --seed 42 \
    --output "$OUT/sample.jsonl" --pool-output "$POOL" --report "$OUT/sample.json" || { log "  sampling FAILED (logs/v59_sample.log): v59 stops"; exit 1; }
[ -s "$OUT/ranks.done" ] || { kscript ranks 18000 "$OUT/ranks.npz" "$PY" "$TOOL" ranks --sample "$OUT/sample.jsonl" --pool "$POOL" --splade "base=models/$BASE" \
    --splade "current=models/$CUR" --dense "$PD" --dense-cache "$DCACHE" --output "$OUT/ranks.npz" && date > "$OUT/ranks.done"; } \
    || { log "  the rank step FAILED (logs/v59_ranks.log): v59 stops"; exit 1; }
grep -E "^(base|current|dense):" "$LOGS/v59_ranks.log" 2>/dev/null | cut -c1-200 | while read -r l; do log "  $l"; done
kscript tables 1800 "$OUT/tables.json" nice -n 10 "$PY" "$TOOL" tables --sample "$OUT/sample.jsonl" --ranks "$OUT/ranks.npz" --base-label "base polish-splade" \
    --current-label "mined arm, bias 0.13" --dense-label "PolDense-400M" --reach 0.25 --min-h 8000 --max-h 20000 --rows-output "$HROWS" \
    --report "$OUT/tables.json" --markdown "$OUT/tables.md" || { log "  the table step FAILED (logs/v59_tables.log): v59 stops"; exit 1; }
cp "$OUT/sample.json" "$OUT/tables.json" "$OUT/tables.md" "$KEEP/"
BUILD=$(hget build_arm)
log "  |H| = $(hget pairs), kept bands $(hget kept_bands), by band $(hget available_by_band); skipped $(hget skipped_bands); arm: $BUILD ($(hget rows) rows)"

F_PICK=""; COST="$RESULTS_ROOT/index_cost_v59.json"
D3_PICK=""; for t in "" -bias005 -bias010 -bias015 -bias020; do [ -n "$(lbv "$D3$t" heldout8)" ] && D3_PICK="$D3$t"; done
FQ_PICK=""; for t in "" -bias005 -bias010 -bias015 -bias020; do [ -n "$(lbv "$FQ$t" heldout8)" ] && FQ_PICK="$FQ$t"; done
if [ "$BUILD" = True ]; then
    # 2. negatives and teacher scores for H
    mine() {  # mine TAG OUT [tool args]: r2v12 mines for the rows of H over the pool of positives, the ensemble teacher scores with the fixed scale
        local tag="$1" out="$2"; shift 2
        [ -s "$out" ] && return 0
        kscript "mine_$tag" 14400 "$out" "$PY" "$KITCHEN/bridge/splade_mine_big_pool.py" --student "models/$R2" --splade-teacher models/soup-v9b \
            --dense-teacher "$PD" --rows "$HROWS" --extra-pool "$POOL" --extra-chars 1 1000000 --dense-scale "$SCALE" --output "$out" "$@"
    }
    log "=== v59 step 2: negatives and teacher scores for $(grep -c . "$HROWS") rows of H (miner $R2, fixed dense scale $SCALE) ==="
    if [ ! -s "$HM" ]; then
        rm -rf cache/smoke_v59; mkdir -p cache/smoke_v59
        if mine smoke cache/smoke_v59/rows.jsonl --limit 300 --extra-max 20000 --extra-shard 8000 --dense-cache cache/smoke_v59/dense && [ "$(grep -c . cache/smoke_v59/rows.jsonl)" = 300 ]; then
            log "  smoke passed: $(tail -n 1 "$LOGS/v59_mine_smoke.log" | cut -c1-300)"
        else log "  mining smoke FAILED (logs/v59_mine_smoke.log): no arm, the summary keeps the diagnostics"; BUILD=failed; fi
        rm -rf cache/smoke_v59
        [ "$BUILD" = True ] && { mine full "$HM" --extra-max 0 --dense-cache "$DCACHE" || { log "  mining FAILED (logs/v59_mine_full.log): no arm"; BUILD=failed; }; }
    fi
fi
if [ "$BUILD" = True ]; then
    "$PY" - "$HM" "$SCALE" <<'PYEOF' | while read -r l; do log "  $l"; done
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
print("H rows %d: teacher P(pos) median %.3f (old rows 0.852), share above 0.95 %.3f (0.303), negative above positive in %d rows = %.1f %% (old rows 2 658 of 20 292 = 13.1 %%), student ranks the positive first in %.3f"
      % (n, st.median(masses), stats["teacher_positive_mass_above_095"], flip, 100 * flip / n, first / n))
PYEOF
    [ -s "$HM.stats.json" ] && cp "$HM.stats.json" "$KEEP/mining_stats.json" && cp "$HM.mined.json" "$KEEP/mining.json" || { log "  the mined rows failed their checks: no arm"; BUILD=failed; }
fi
if [ "$BUILD" = True ]; then
    # 3. no training here: queue v60 (dose3 + FAQ) runs first, queue v61 trains the rows of its winner + the rows of H
    log "=== v59 step 3: no training in this queue; the $(grep -c . "$HM") mined rows of H ($HM) wait for queue v61 (after v60) ==="
fi

# 4. summary
log "=== v59 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_v59.md"
"$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$COST" "$KEEP" "$M" "$R2" "${D3_PICK:-none}" "${FQ_PICK:-none}" "${F_PICK:-none}" "$BUILD" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, spa = Path(sys.argv[1]), Path(sys.argv[2])
cost_path, keep, m, r2, d3, fq, pick, build = sys.argv[3:11]
keep = Path(keep)
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}
def res(label, ts): return load(reg / label / f"{ts}.json")
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
tab, smp, ms = load(keep / "tables.json"), load(keep / "sample.json"), load(keep / "mining_stats.json")
H = tab.get("H", {})
print("# v59 (quantized regime, nDCG@10): MS MARCO queries where the dense model wins and SPLADE does not\n")
print("Diagnostics on %s queries of the judged full MS MARCO-pl pairs (%s pairs, %s queries; left out: %s), then, when it is large enough, the hard-reachable set H "
      "with negatives mined by r2v12 from the pool of the sample's positives and the scores of the same ensemble teacher with the fixed dense scale. No arm is "
      "trained here: queue v60 (SQuAD-v2 dose + FAQ) runs first and queue v61 trains the rows of its winner plus H.\n" % (smp.get("sampled_queries", "-"), smp.get("pairs", "-"), smp.get("queries", "-"), smp.get("dropped", {})))
print("**|H| = %s; kept coverage bands: %s; the arm: %s.**\n" % (H.get("pairs", "-"), ", ".join(H.get("kept_bands", [])) or "none",
      ("%s rows mined, to be trained by queue v61 after v60" % H.get("rows")) if build == "True" else ("NOT built (|H| below %s)" % H.get("min_for_the_arm") if build == "False" else "NOT built: a step failed, see the queue state")))
if pick != "none":
    print("## Result\n\n| model | new rows | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, what in [(r2, "old negatives"), (f"{m}-bias010", "-"), (f"{m}-bias013", "-"), (f"{m}-bias015", "-"), (f"{m}-s43-bias015", "seed 43"), (d3, "SQuAD-v2 dose (v56)"),
                        (fq, "native FAQ (v58)"), (pick, "targeted MS MARCO")]:
        if label == "none" or val(label, "tiny10") is None:
            continue
        name = f"**{label}**" if label == pick else label
        print(f"| {name} | {what} | {fmt(val(label, 'tiny10'))} | {fmt(val(label, 'heldout8'))} | {fmt(val(label, 'heldout2_treccovid'))} | {fmt(dims(label), 1)} | {c(label, 'doc_active_mean', 1)} | {c(label, 'query_active_mean', 1)} | {c(label, 'postings_per_query')} |")
    t, h, d = val(pick, "tiny10", 10), val(pick, "heldout8", 8), dims(pick)
    own = [x for x in (f"{m}-bias0{s}" for s in ("10", "11", "12", "13", "15")) if None not in (val(x, "tiny10", 10), val(x, "heldout8", 8), dims(x))]
    ref = min(own, key=lambda x: abs(dims(x) - d)) if own and d is not None else None
    verdict = "the targeted arm has no complete result"
    if None not in (t, h, d) and ref:
        dt, dh = t - val(ref, "tiny10"), h - val(ref, "heldout8")
        per = {}
        for ts in ("tiny10", "heldout8"):
            a, b = res(pick, ts).get("tasks", {}), res(ref, ts).get("tasks", {})
            per.update({k: a[k] - b[k] for k in a if k in b})
        is_short = lambda k: k.startswith("maupqa") or "allegro-faq" in k or k == "mfaq"
        sq = [v for k, v in per.items() if is_short(k)]; rest = [v for k, v in per.items() if not is_short(k)]
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        if dt >= 0.2 and dh >= 0.2:
            word = ("TARGETED SELECTION WORKS (>= +0.2 on both sets). Proposed, not started: the control arm with the same number of RANDOM MS MARCO rows, the same "
                    "mined negatives and the same teacher; without it the gain cannot be told apart from new lists plus the ensemble teacher")
        elif dt <= -0.2 and dh <= -0.2:
            word = "a LOSS on both sets; see the per-task tables for whether a gain on the short-question tasks was paid for elsewhere"
        else:
            word = "within the seed noise: MS MARCO is exhausted for this family also under targeted selection"
        verdict = ("%s %.2f / %.2f at %.1f dims/doc against the mined arm %s %.2f / %.2f at %.1f: delta %+.2f / %+.2f, trec-covid %s against %s -> %s. "
                   "Short-question tasks (MAUPQA, Allegro FAQ, mfaq) mean delta %+.2f, the other tasks %+.2f"
                   % (pick, t, h, d, ref, val(ref, "tiny10"), val(ref, "heldout8"), dims(ref), dt, dh, fmt(val(pick, "heldout2_treccovid")), fmt(val(ref, "heldout2_treccovid")), word, mean(sq), mean(rest)))
        for other, what in ((d3, "the dose arm of v56"), (fq, "the FAQ arm of v58")):
            if other != "none" and None not in (val(other, "tiny10", 10), val(other, "heldout8", 8)):
                verdict += ". Against %s %s (%.2f / %.2f at %s): delta %+.2f / %+.2f" % (what, other, val(other, "tiny10"), val(other, "heldout8"), fmt(dims(other), 1), t - val(other, "tiny10"), h - val(other, "heldout8"))
    print("\n## Reading\n\n" + verdict + ".")
    print("\nSeed noise of this recipe is 0.13-0.2 (one seed here). Nothing further was started by this queue.")
    if ref:
        for ts in ("tiny10", "heldout8"):
            a, b = res(pick, ts).get("tasks", {}), res(ref, ts).get("tasks", {})
            e, g = (res(d3, ts).get("tasks", {}) if d3 != "none" else {}), (res(fq, ts).get("tasks", {}) if fq != "none" else {})
            if not a:
                continue
            print(f"\n### {ts} per task\n\n| task | mined arm ({ref.split('-')[-1]}) | dose (v56) | FAQ (v58) | targeted MS | delta vs mined arm |\n|---|---:|---:|---:|---:|---:|")
            for k in a:
                print(f"| {k} | {fmt(b.get(k))} | {fmt(e.get(k))} | {fmt(g.get(k))} | {fmt(a[k])} | {('%+.2f' % (a[k] - b[k])) if k in b else '-'} |")
else:
    print("## Reading\n\nNo arm was trained in this queue: %s. The diagnostics below stand on their own." % (
        "H passed the threshold and its rows are mined; queue v61 trains them after v60, on the rows of v60's winner, against that winner" if build == "True"
        else "|H| is below the threshold" if build == "False" else "a step of the mining failed"))
if ms:
    print("\n## Mining of the rows of H\n\n| rows | teacher P(pos) median | share with P(pos) > 0.95 | rows with a negative above the positive | student ranks the positive first |\n|---|---:|---:|---:|---:|")
    print(f"| H ({ms['rows']}) | {ms['teacher_positive_mass_median']:.3f} | {ms['teacher_positive_mass_above_095']:.3f} | {ms['rows_with_negative_above_positive']} ({100 * ms['rows_with_negative_above_positive'] / ms['rows']:.1f} %) | {ms['student_ranks_the_positive_first']:.3f} |")
    print("| old rows of the mined arm (20292) | 0.852 | 0.303 | 2658 (13.1 %) | 0.844 |")
print("\n## Diagnostics\n")
try:
    print(open(keep / "tables.md", encoding="utf-8").read())
except Exception:
    print("(tables missing)")
EOF
{ echo; echo "## Queue state (v59)"; echo; echo '```'; grep -E "mined-mshard|chain:v59_" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-900 | while read -r l; do log "  READING: $l"; done
git add "$SUM" "$EXP_ROOT/queues/queue_v59.sh" "$KEEP" 2>/dev/null
[ -e "$CFG" ] && git add "$CFG" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null   # the rows (datasets/ and the data drive are not tracked) are described in the summary
for m in ${F_PICK:-} $F; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$F-bias0??.json 2>/dev/null
git commit -q -m "Add the v59 results: MS MARCO queries where the dense model wins and SPLADE does not" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v59.log" 2>&1 && { echo pushed >> "$LOGS/push_v59.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v59 done ($(date '+%d.%m %H:%M')) ==="
