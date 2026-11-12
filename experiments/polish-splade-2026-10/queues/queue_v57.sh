#!/usr/bin/env bash
# Queue v57 (3060 Ti, 12.11 after v56, ~2.7 h): the large-pool arm with a FALSE-NEGATIVE FILTER. The large pool of v53
# (the rows' own passages + 1 M passages of kd_v1, 5 best + 5 random of the student's top-60) gave 64.58 / 43.22 at 363
# dims/doc against 64.67 / 43.35 for the own pools. Its rows show why it may have lost by the rules and not by the
# pool: the teacher scores 18 % of the negatives from the student's ranks 1-5 at or above the best positive (6.6 % with
# the own pools), 32 % of the SQuAD-v2 rows carry such a negative, and the teacher's P(pos) median fell from 0.85 to 0.52.
#   1  ROWS, no new mining: from the v53 rows drop every negative whose teacher score is >= 0.95 x the score of the row's
#      best positive; refill from the same row's own-pool negatives (the v49 rows, same teacher) under the same threshold;
#      rows still short take the lowest-scored remaining candidates (counted). Queries and positives stay identical.
#   2  ARM arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12: the recipe of the mined arms unchanged (from soup-v12, 2 epochs,
#      LR 2e-5, 2x16, 512 tokens), only the row file differs. Measured in the bias form 0.15 (the form of the v53 arm):
#      tiny10, heldout8, trec-covid, query-side index cost.
#   3  READING against the own-pool arm and the unfiltered large-pool arm in the same bias form. summary_v57.md,
#      commit, push loop. No further arm starts here.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v57.sh > logs/queue_v57.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v57 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-14400}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v57.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v54 v56}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
tc()   { lbv "$1" heldout2_treccovid; }
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
at_bias() {  # at_bias MODEL REFLINE: the bias form 0.15 (0.20 when that is still above 410 dims/doc) -> tiny10, heldout8, trec-covid; PICK holds its label
    local m="$1" ref="$2" t; PICK=""
    [ -d "models/$m" ] || { log "$m missing: not measured"; return 1; }
    sparsity "$m" "models/$m"
    for t in 015 020; do
        biased "models/$m" "$m-bias$t" "0.${t#0}" || return 1
        sparsity "$m-bias$t" "models/$m-bias$t"; PICK="$m-bias$t"
        [ "$(gte 410 "$(dims "$PICK")")" = 1 ] && break
    done
    [ "$PICK" != "$m-bias015" ] && rm -rf "models/$m-bias015"
    log "  $m: $(dims "$m") dims/doc unbiased; measured form $PICK at $(dims "$PICK")"
    regime quantized "$PICK" "models/$PICK" tiny10 16; regime quantized "$PICK" "models/$PICK" heldout8 16
    log "  $PICK: $(lbs "$PICK") ($ref), $(dims "$PICK") dims/doc"
    regime quantized "$PICK" "models/$PICK" heldout2_treccovid 8; log "  $PICK trec-covid $(tc "$PICK"), $(dims "$PICK") dims/doc"
}
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
XL=arm-dense-kl-n5kj-minedxl-ens400-r2v12
F=arm-dense-kl-n5kj-minedxl-f95-ens400-r2v12
A=datasets/arms
MROWS=$A/n5-plus-kj-mined-tens400.jsonl     # the rows of the mined arm (v49): own-pool negatives, the refill
XLROWS=$A/n5-plus-kj-minedxl.jsonl          # the rows of the large-pool arm (v53)
FROWS=$A/n5-plus-kj-minedxl-f95.jsonl
FRAC=${FRAC:-0.95}
BASECFG="$C/distil_dense_kl_n5kj_minedxl_ens400_r2v12.yaml"
CFG="$C/distil_dense_kl_n5kj_minedxl_f95_ens400_r2v12.yaml"
log "=== v57 start: the large pool with a false-negative filter; own pools bias015 $(lbs $M-bias015) / $(tc $M-bias015) at $(dims $M-bias015) dims/doc, large pool bias015 $(lbs $XL-bias015) / $(tc $XL-bias015) at $(dims $XL-bias015); $(free_gb) GB free on / ==="
for f in "$MROWS" "$XLROWS" "$BASECFG"; do [ -s "$f" ] || { log "  $f missing: v57 stops"; exit 1; }; done
[ -d models/soup-v12 ] || { log "  models/soup-v12 missing: v57 stops"; exit 1; }

# 1. the row file
log "=== v57 step 1: rows = the v53 rows without negatives at or above $FRAC x the best positive, refilled from the own-pool negatives ==="
if [ ! -s "$FROWS" ]; then
    "$PY" - "$XLROWS" "$MROWS" "$FROWS" "$FRAC" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, math, os, statistics as st, sys
from collections import Counter
xl_path, own_path, out, frac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
def key(r): return (str(r["query_id"]), r["query"])
def mass(best, negs):
    s = [best] + list(negs); m = max(s); e = [math.exp((x - m) / 2.0) for x in s]
    return e[0] / sum(e)
xl = rows(xl_path); own = {key(r): r for r in rows(own_path)}
assert len(own) == len(xl) == len({key(r) for r in xl}), "the two row files differ in their queries"
n_xl_kept = n_refill = n_last = untouched = rows_last = 0
dropped_by_pool, total_by_pool, after_by_pool = Counter(), Counter(), Counter()
m_before, m_after, m_own, pos_diff = [], [], [], []
flip_before = flip_after = flip_own = 0
with open(out + ".tmp", "w", encoding="utf-8") as f:
    for r in xl:
        o = own[key(r)]
        assert o["pos"] == r["pos"], "positives differ: " + str(r["query_id"])
        assert len(r["neg"]) == len(r["neg_scores"]) == len(r["neg_pool"]) == len(r["neg_mined_rank"]) == 10, "row shape: " + str(r["query_id"])
        assert len(o["neg"]) == len(o["neg_scores"]) == len(o["neg_mined_rank"]) == 10, "own row shape: " + str(r["query_id"])
        best = max(r["pos_scores"]); assert best > 0, "non-positive teacher score of a positive: " + str(r["query_id"])
        thr = frac * best
        pos_diff.extend(abs(a - b) for a, b in zip(r["pos_scores"], o["pos_scores"]))
        cand = [(t, s, p, k, "xl") for t, s, p, k in zip(r["neg"], r["neg_scores"], r["neg_pool"], r["neg_mined_rank"])]
        for c in cand:
            total_by_pool[c[2]] += 1
        keep = [c for c in cand if c[1] < thr]
        spare = [(t, s, p, k, "above-threshold") for t, s, p, k, _ in cand if s >= thr]
        for c in spare:
            dropped_by_pool[c[2]] += 1
        n_xl_kept += len(keep); untouched += len(keep) == 10
        have = set(r["pos"]) | {c[0] for c in cand}
        for t, s, k in zip(o["neg"], o["neg_scores"], o["neg_mined_rank"]):
            if t in have:
                continue
            have.add(t)
            if s >= thr:
                spare.append((t, s, "own", k, "above-threshold"))
            elif len(keep) < 10:
                keep.append((t, s, "own", k, "own-refill")); n_refill += 1
        if len(keep) < 10:  # the lowest-scored of what is left, still at or above the threshold
            need = 10 - len(keep); spare.sort(key=lambda c: c[1])
            assert len(spare) >= need, "not enough candidates: " + str(r["query_id"])
            keep += spare[:need]; n_last += need; rows_last += 1
        texts = [c[0] for c in keep]
        assert len(texts) == 10 == len(set(texts)) and not (set(texts) & set(r["pos"])), "bad negatives: " + str(r["query_id"])
        new = dict(r)
        new["neg"] = texts; new["neg_scores"] = [c[1] for c in keep]; new["neg_pool"] = [c[2] for c in keep]
        new["neg_mined_rank"] = [c[3] for c in keep]; new["neg_origin"] = [c[4] for c in keep]
        new["neg_filter"] = "teacher score < %.2f x best positive" % frac
        f.write(json.dumps(new, ensure_ascii=False) + "\n")
        for c in keep:
            after_by_pool[c[2]] += 1
        m_before.append(mass(best, r["neg_scores"])); m_after.append(mass(best, new["neg_scores"])); m_own.append(mass(max(o["pos_scores"]), o["neg_scores"]))
        flip_before += max(r["neg_scores"]) > min(r["pos_scores"]); flip_after += max(new["neg_scores"]) > min(r["pos_scores"]); flip_own += max(o["neg_scores"]) > min(o["pos_scores"])
os.replace(out + ".tmp", out)
n = len(xl); total = 10 * n
stats = {"rows": n, "rule": "drop negatives with teacher score >= %.2f x the best positive of the row" % frac, "source_rows": xl_path, "refill_rows": own_path,
         "negatives_kept_from_large_pool_rows": n_xl_kept, "negatives_dropped": total - n_xl_kept, "negatives_refilled_from_own_pool_rows": n_refill,
         "negatives_above_threshold_kept_as_last_resort": n_last, "rows_with_a_last_resort_negative": rows_last, "rows_untouched": untouched,
         "negatives_from_shared_pool_before": 1 - total_by_pool["own"] / total, "negatives_from_shared_pool_after": 1 - after_by_pool["own"] / total,
         "dropped_share_by_pool": {p: round(dropped_by_pool[p] / c, 4) for p, c in total_by_pool.most_common()},
         "teacher_positive_mass_median": {"large pool": st.median(m_before), "filtered": st.median(m_after), "own pools": st.median(m_own)},
         "rows_with_negative_above_positive": {"large pool": flip_before, "filtered": flip_after, "own pools": flip_own},
         "positive_score_abs_diff_between_the_two_scoring_passes": {"median": st.median(pos_diff), "max": max(pos_diff)}}
json.dump(stats, open(out + ".filter.json", "w"), indent=2)
print("rows %d; negatives dropped %d of %d (%.3f), refilled from the own-pool rows %d, above the threshold kept as the last resort %d in %d rows; rows untouched %d"
      % (n, total - n_xl_kept, total, (total - n_xl_kept) / total, n_refill, n_last, rows_last, untouched))
print("negatives from the shared pool %.3f -> %.3f; teacher P(pos) at T=2, median: large pool %.3f, filtered %.3f, own pools %.3f; rows with a negative above a positive: %d -> %d (own pools %d)"
      % (stats["negatives_from_shared_pool_before"], stats["negatives_from_shared_pool_after"], st.median(m_before), st.median(m_after), st.median(m_own), flip_before, flip_after, flip_own))
print("dropped share by pool: %s" % stats["dropped_share_by_pool"])
PYEOF
    [ -s "$FROWS" ] || { log "  row file $FROWS not written: v57 stops"; exit 1; }
fi
NROWS=$(grep -c . "$FROWS")
[ "$NROWS" = "$(grep -c . "$XLROWS")" ] || { log "  $FROWS has $NROWS rows, not those of $XLROWS: v57 stops"; exit 1; }
log "  $FROWS: $NROWS rows"

# 2. the arm
log "=== v57 step 2: arm $F ($NROWS rows, 2 epochs, the recipe of the mined arms) ==="
sed -e "4s|.*|# The large-pool arm with a false-negative filter (queue v57): the rows of the v53 arm ($XLROWS) without the negatives the ensemble teacher (soup-v9b + PolDense-400M) scores at or above $FRAC x the best positive of the row, refilled from the same row's own-pool negatives ($MROWS) under the same threshold. Queries, positives and the r2v12 recipe from soup-v12 are unchanged.|" \
    -e "s|^runs_dir: .*|runs_dir: runs/$F|" -e "s|^train_data: .*|train_data: $FROWS|" "$BASECFG" > "$CFG"
train "${F#arm-}" "$CFG"
REF="own pools bias015 $(lbs $M-bias015) / $(tc $M-bias015) at $(dims $M-bias015); large pool bias015 $(lbs $XL-bias015) / $(tc $XL-bias015) at $(dims $XL-bias015); own pools seed 43 bias015 $(lbs $M-s43-bias015) at $(dims $M-s43-bias015)"
at_bias "$F" "$REF"; F_PICK=${PICK:-}
[ -d "models/$F" ] || F_PICK=""

# query-side index cost
COST="$RESULTS_ROOT/index_cost_v57.json"
if [ -n "$F_PICK" ] && [ ! -s "$COST" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    args=(--model base=models/base-splade-wrapped)
    for m in $M-bias013 $M-bias015 $XL-bias015 "$F_PICK"; do [ -d "models/$m" ] && args+=(--model "$m=models/$m"); done
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" --output-json "$COST" > "$LOGS/index_cost_v57.log" 2>&1 \
        && log "index cost written ($COST) in $((SECONDS - s))s" || { log "index cost FAILED (logs/index_cost_v57.log)"; rm -f "$COST"; }
fi

# 3. reading and summary
log "=== v57 step 3: reading and summary ==="
SUM="$RESULTS_ROOT/summary_v57.md"
"$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$COST" "$FROWS.filter.json" "$M" "$XL" "$R2" "${F_PICK:-none}" "$FRAC" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, spa = Path(sys.argv[1]), Path(sys.argv[2])
cost_path, filter_path, m, xl, r2, pick, frac = sys.argv[3:10]
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}
def val(label, ts, n=None):
    try:
        d = json.load(open(reg / label / f"{ts}.json"))
        return d["mean_ndcg@10"] if n is None or d["n_tasks"] == n else None
    except Exception:
        return None
def dims(label):
    try:
        d = json.load(open(spa / f"{label}.json"))[label]; v = [t["mean"] for t in d.values() if t.get("mean") is not None]
        return sum(v) / len(v)
    except Exception:
        return None
def fmt(x, nd=2): return "-" if x is None else f"{x:.{nd}f}"
cost = load(cost_path).get("models", {}); flt = load(filter_path)
def c(label, key, nd=0):
    s = cost.get(label); return fmt(s[key], nd) if s else "-"
print("# v57 (12.11, quantized regime, nDCG@10): the large-pool arm with a false-negative filter\n")
print("f95 = the rows of the large-pool arm (v53: own passages + 1 M passages of kd_v1, 5 best + 5 random of the student's top-60) without the negatives "
      "the ensemble teacher scores at or above %s x the best positive of the row, refilled from the same row's own-pool negatives (the rows of the mined arm) "
      "under the same threshold. Queries, positives and the recipe (from soup-v12, 2 epochs) are unchanged; no new mining. The cost columns come from the "
      "cost sample (1 350 passages + 1 350 queries).\n" % frac)
if flt:
    pm, fl = flt["teacher_positive_mass_median"], flt["rows_with_negative_above_positive"]
    print("Rows: %d. Negatives dropped: %d of %d; refilled from the own-pool rows: %d; kept above the threshold as the last resort: %d (in %d rows); rows untouched: %d. "
          "Negatives from the shared pool: %.3f -> %.3f. Teacher P(pos) at T = 2, median: large pool %.3f, filtered %.3f, own pools %.3f. Rows with a negative above "
          "a positive: large pool %d, filtered %d, own pools %d.\n"
          % (flt["rows"], flt["negatives_dropped"], 10 * flt["rows"], flt["negatives_refilled_from_own_pool_rows"], flt["negatives_above_threshold_kept_as_last_resort"],
             flt["rows_with_a_last_resort_negative"], flt["rows_untouched"], flt["negatives_from_shared_pool_before"], flt["negatives_from_shared_pool_after"],
             pm["large pool"], pm["filtered"], pm["own pools"], fl["large pool"], fl["filtered"], fl["own pools"]))
print("| model | negatives | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
table = [(r2, "old (round 2)"), (f"{m}-bias013", "own pools"), (f"{m}-bias015", "own pools"), (f"{m}-s43-bias015", "own pools, seed 43"),
         (f"{xl}-bias015", "large pool"), (pick, "large pool, filtered")]
for label, what in table:
    if label == "none" or val(label, "tiny10") is None:
        continue
    name = f"**{label}**" if label == pick else label
    print(f"| {name} | {what} | {fmt(val(label, 'tiny10'))} | {fmt(val(label, 'heldout8'))} | {fmt(val(label, 'heldout2_treccovid'))} | {fmt(dims(label), 1)} | {c(label, 'doc_active_mean', 1)} | {c(label, 'query_active_mean', 1)} | {c(label, 'postings_per_query')} |")
verdict = "the filtered arm has no complete result"
t, h, d = val(pick, "tiny10", 10), val(pick, "heldout8", 8), dims(pick)
own = [x for x in (f"{m}-bias0{s}" for s in ("10", "11", "12", "13", "15")) if None not in (val(x, "tiny10", 10), val(x, "heldout8", 8), dims(x))]
xlf = f"{xl}-bias015"
if None not in (t, h, d) and own:
    ref = min(own, key=lambda x: abs(dims(x) - d))
    dt, dh = t - val(ref, "tiny10"), h - val(ref, "heldout8")
    if dt >= 0.2 and dh >= 0.2:
        word = "THE FILTERED LARGE POOL WINS (>= +0.2 on both sets over the own pools)"
    elif dt <= -0.2 and dh <= -0.2:
        word = "the filtered large pool LOSES to the own pools (<= -0.2 on both sets)"
    else:
        word = ("within the seed noise of the own pools: the large pool adds nothing under this filter. With this, the composition of the negative lists at "
                "fixed queries is exhausted: own pools (teacher P(pos) median 0.85), the large pool (0.52), this filter (0.74), refreshed negatives and hard "
                "positives (v52) all land within the noise of each other, so no further variant of negative selection is proposed; the open axes are new queries")
    verdict = "%s %.2f / %.2f at %.1f dims/doc against the own pools %s %.2f / %.2f at %.1f: delta %+.2f / %+.2f, trec-covid %s against %s -> %s" % (
        pick, t, h, d, ref, val(ref, "tiny10"), val(ref, "heldout8"), dims(ref), dt, dh, fmt(val(pick, "heldout2_treccovid")), fmt(val(ref, "heldout2_treccovid")), word)
    if None not in (val(xlf, "tiny10", 10), val(xlf, "heldout8", 8)):
        verdict += ". Against the unfiltered large pool %s (%.2f / %.2f at %.1f): delta %+.2f / %+.2f" % (
            xlf, val(xlf, "tiny10"), val(xlf, "heldout8"), dims(xlf) or 0.0, t - val(xlf, "tiny10"), h - val(xlf, "heldout8"))
print("\n## Reading\n\n" + verdict + ".")
print("\nSeed noise of this recipe is 0.13-0.2 (one seed here). The refill negatives were scored in another pass of the same teacher (dense factor 84.194 "
      "against 84.200; the positives of the two passes differ by the amounts in the block below). Nothing further was started by this queue.")
if flt:
    print("\n## The filter\n\n```\n" + json.dumps(flt, indent=2, ensure_ascii=False) + "\n```")
EOF
{ echo; echo "## Queue state (v57)"; echo; echo '```'; grep -E "minedxl-f95" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-700 | while read -r l; do log "  READING: $l"; done
git add "$SUM" "$EXP_ROOT/queues/queue_v57.sh" "$CFG" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null   # the row sidecar (datasets/ is not tracked) is quoted in the summary
for m in ${F_PICK:-} $F; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$F-bias0??.json 2>/dev/null
git commit -q -m "Add the v57 results: the large-pool arm with a false-negative filter" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v57.log" 2>&1 && { echo pushed >> "$LOGS/push_v57.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v57 done ($(date '+%d.%m %H:%M')) ==="
