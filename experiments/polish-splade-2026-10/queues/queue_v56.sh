#!/usr/bin/env bash
# Queue v56 (3060 Ti, 12.11 after v52, v53, v55 and v54, ~6 h; ~9 h on the large-pool branch): the query DOSE of the
# student-mined arm. The mined arm (64.63 / 43.45 / trec-covid 70.15 at 405 dims/doc with bias 0.10) uses 20 292 rows,
# 16 000 of them from the 56 093 SQuAD-v2 rows that pass the gate 23.5. KL distillation scaled with the dose on the old,
# easy negatives (6 k -> 18 k rows: 63.25 -> 63.39 / 42.62 -> 42.80) while their target was close to one-hot; with
# student-mined negatives every row carries more, so the dose is asked again: do three times as many non-MS-MARCO
# queries give more than 20 k?
#   0  BRANCH from the v53 result files: when the large-pool arm won by its own rule (>= +0.2 on both sets over the
#      mined arm at equal cost), the new rows are mined with the v53 tool and pool and the base rows / the reference are
#      those of v53; otherwise the rows' own passages, as in the mined arm.
#   1  SMOKE of the mining tool on 300 rows; a failure ends the queue.
#   2  MINING for the gated SQuAD-v2 rows outside n5 (40 093): the same miner (r2v12) and rules (5 best + 5 random of the
#      top-60, length window 0.5-2.0); pool = the passages of all gated SQuAD-v2 rows; the ensemble teacher soup-v9b +
#      PolDense-400M with the FIXED dense scale of the base rows (84.194 for the mined arm), not refitted.
#   3  ROWS: the base file unchanged (every line verbatim) + the new rows, shuffled once (seed 42): a strict superset.
#   4  ARM arm-dense-kl-n5kj-mined-dose3-ens400-r2v12: the recipe of the mined arm unchanged (from soup-v12, 2 epochs,
#      LR 2e-5, 2x16, 512 tokens), only the row file differs. Measured in its first bias form with at most 410 dims/doc:
#      tiny10, heldout8, trec-covid, query-side index cost.
#   5  READING against the mined arm at the nearest cost (and seed 43): >= +0.2 on both sets = the dose pays; within the
#      seed noise (0.13-0.2) = the 20 k recipe is final. summary_v56.md, commit, push loop. No further arm starts here.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v56.sh > logs/queue_v56.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v56 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-21600}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v56.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v52 v53 v55 v54}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
h2task() { local bs=16; case "$2" in heldout2_poquad|heldout2_treccovid) bs=8;; esac; [ -d "models/$1" ] && regime quantized "$1" "models/$1" "$2" "$bs"; }
tc() { lbv "$1" heldout2_treccovid; }
measure() {  # measure LABEL MODEL REFLINE
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
side() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]+'.mined.json'));print(d[sys.argv[2]])" "$1" "$2" 2>/dev/null; }
jget() { "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get(sys.argv[2]) if d.get(sys.argv[2]) is not None else '')" "$1" "$2" 2>/dev/null; }
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
    h2task "$PICK" heldout2_treccovid; log "  $PICK trec-covid $(tc "$PICK"), $(dims "$PICK") dims/doc"
}
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
XL=arm-dense-kl-n5kj-minedxl-ens400-r2v12
D=arm-dense-kl-n5kj-mined-dose3-ens400-r2v12
A=datasets/arms
ROWS=$A/n5-plus-kj-tens400.jsonl            # n5 + kj with the old negatives: its SQuAD-v2 rows complete the pool
MROWS=$A/n5-plus-kj-mined-tens400.jsonl     # the rows of the mined arm (v49)
XLROWS=$A/n5-plus-kj-minedxl.jsonl          # the rows of the large-pool arm (v53)
ALL=$A/squadv2-v2-all.jsonl
REST=$A/squadv2-v2-gated-rest.jsonl         # gated SQuAD-v2 rows outside n5 (written by v52 step 3)
NEW=$A/squadv2-v2-gated-rest-mined-v56.jsonl
DROWS=$A/n5-plus-kj-mined-dose3.jsonl
PD=OPI-PIB/PolDense-400M
DCACHE=cache/dense_pool/pd400
EXTRA=datasets/kd/kd_v1.jsonl
EXTRA_MAX=${EXTRA_MAX:-1000000}
BASECFG="$C/distil_dense_kl_n5kj_mined_ens400_r2v12.yaml"
CFG="$C/distil_dense_kl_n5kj_mined_dose3_ens400_r2v12.yaml"
PLAN="$RESULTS_ROOT/v56_plan.json"
log "=== v56 start: the dose of the mined arm; mined arm bias010 $(lbs $M-bias010) / $(tc $M-bias010) at $(dims $M-bias010) dims/doc; $(free_gb) GB free on / ==="
for f in "$MROWS" "$ROWS" "$ALL" "$BASECFG"; do [ -s "$f" ] || { log "  $f missing: v56 stops"; exit 1; }; done
for m in "$R2" soup-v9b soup-v12; do [ -d "models/$m" ] || { log "  models/$m missing: v56 stops"; exit 1; }; done

# 0. the branch: did the large pool of v53 win by its own rule?
log "=== v56 step 0: branch from the v53 results ==="
"$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$M" "$XL" "$XLROWS" "$PLAN" <<'EOF' | while read -r l; do log "  $l"; done
import json, os, sys
from pathlib import Path
reg, spa, m, xl, xlrows, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
def val(label, ts, n):
    try:
        d = json.load(open(reg / label / f"{ts}.json"))
        return d["mean_ndcg@10"] if d["n_tasks"] == n else None
    except Exception:
        return None
def dims(label):
    try:
        d = json.load(open(spa / f"{label}.json"))[label]; v = [t["mean"] for t in d.values() if t.get("mean") is not None]
        return sum(v) / len(v)
    except Exception:
        return None
def forms(arm, suffixes):
    res = []
    for s in suffixes:
        label = arm + s; a, b, d = val(label, "tiny10", 10), val(label, "heldout8", 8), dims(label)
        if None not in (a, b, d):
            res.append({"label": label, "tiny10": a, "heldout8": b, "dims_per_doc": d})
    return res
mined = forms(m, ["-bias010", "-bias011", "-bias012", "-bias013", "-bias015"])
xlf = [f for f in forms(xl, ["", "-bias005", "-bias010", "-bias015", "-bias020"]) if f["dims_per_doc"] <= 410]
plan = {"branch": "own", "mined_forms": mined, "xl_forms": xlf, "rule": "large pool wins when its form is >= +0.2 on tiny10 and on heldout8 over the mined-arm form at the nearest dims/doc"}
if xlf and mined:
    x = max(xlf, key=lambda f: f["dims_per_doc"])  # the first form at or under 410 = the one v53 measured
    ref = min(mined, key=lambda f: abs(f["dims_per_doc"] - x["dims_per_doc"]))
    dt, dh = x["tiny10"] - ref["tiny10"], x["heldout8"] - ref["heldout8"]
    plan.update({"xl_pick": x["label"], "xl_reference": ref["label"], "xl_delta_tiny10": dt, "xl_delta_heldout8": dh})
    print("large pool %s: %.2f / %.2f at %.1f dims/doc; mined arm at the nearest cost %s: %.2f / %.2f at %.1f; delta %+.2f / %+.2f"
          % (x["label"], x["tiny10"], x["heldout8"], x["dims_per_doc"], ref["label"], ref["tiny10"], ref["heldout8"], ref["dims_per_doc"], dt, dh))
    if dt >= 0.2 and dh >= 0.2:
        if os.path.getsize(xlrows) if os.path.exists(xlrows) else 0:
            plan["branch"] = "xl"
        else:
            print("the large pool won but %s is missing: the own-pool branch is used" % xlrows)
else:
    print("no complete result of the large-pool arm (v53): the own-pool branch is used")
json.dump(plan, open(out, "w"), indent=2)
print("branch: " + plan["branch"])
EOF
BRANCH=$(jget "$PLAN" branch); BRANCH=${BRANCH:-own}
if [ "$BRANCH" = xl ]; then
    BASEROWS=$XLROWS; SCALE=$(side "$XLROWS" dense_scale); BASEARM=$(jget "$PLAN" xl_pick)
    [ -s "$EXTRA" ] || { log "  $EXTRA missing (is the data drive mounted?): v56 stops"; exit 1; }
else
    BASEROWS=$MROWS; BASEARM=$M-bias010
    SCALE=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['scale'])" "$A/n5-plus-kj-mined-tpd400.jsonl.teacher.json" 2>/dev/null); SCALE=${SCALE:-84.19441654567078}
fi
[ -n "$SCALE" ] || { log "  no dense scale for $BASEROWS: v56 stops"; exit 1; }
log "  branch $BRANCH: base rows $BASEROWS, fixed dense scale $SCALE, reference arm $BASEARM"

# the gated rows outside n5 (v52 wrote them; rebuilt the same way when missing)
[ -s "$REST" ] || "$PY" - "$ALL" "$ROWS" "$REST" <<'PYEOF' | while read -r l; do log "  $l"; done
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
NREST=$(grep -c . "$REST" 2>/dev/null); NREST=${NREST:-0}
[ "$NREST" -gt 0 ] || { log "  $REST empty: v56 stops"; exit 1; }
log "  new rows to mine: $NREST gated SQuAD-v2 rows outside n5 ($REST)"

mine() {  # mine TAG OUT [tool args]: r2v12 mines for the rows of $REST, the ensemble teacher scores with the fixed scale
    local tag="$1" out="$2"; shift 2
    [ -s "$out" ] && return 0
    if [ "$BRANCH" = xl ]; then
        kscript "dose_$tag" 21600 "$out" "$PY" "$KITCHEN/bridge/splade_mine_big_pool.py" --student "models/$R2" --splade-teacher models/soup-v9b \
            --dense-teacher "$PD" --rows "$REST" --pool-rows "$ROWS" --extra-pool "$EXTRA" --dense-scale "$SCALE" --output "$out" "$@"
    else
        kscript "dose_$tag" 7200 "$out" "$PY" "$KITCHEN/bridge/splade_mine_and_score.py" --student "models/$R2" --splade-teacher models/soup-v9b \
            --dense-teacher "$PD" --rows "$REST" --pool-rows "$ROWS" --dense-scale "$SCALE" --output "$out" "$@"
    fi
}

# 1. smoke
log "=== v56 step 1: smoke of the mining tool (300 rows, branch $BRANCH) ==="
if [ ! -s "$NEW" ] && [ ! -s "$DROWS" ]; then
    rm -rf cache/smoke_v56; mkdir -p cache/smoke_v56
    if [ "$BRANCH" = xl ]; then SMOKE_ARGS=(--limit 300 --extra-max 20000 --extra-shard 8000 --dense-cache cache/smoke_v56/dense)
    else SMOKE_ARGS=(--limit 300 --dense-cache cache/smoke_v56/dense); fi
    if mine smoke cache/smoke_v56/rows.jsonl "${SMOKE_ARGS[@]}" && [ "$(grep -c . cache/smoke_v56/rows.jsonl)" = 300 ]; then
        log "  smoke passed: dense scale $(side cache/smoke_v56/rows.jsonl dense_scale), teacher P(pos) median $(side cache/smoke_v56/rows.jsonl teacher_positive_mass_median), pools $(side cache/smoke_v56/rows.jsonl pools)"
    else log "  smoke FAILED (logs/dose_smoke.log): v56 stops"; rm -rf cache/smoke_v56; exit 1; fi
    rm -rf cache/smoke_v56
else log "  mined rows already present: smoke skipped"; fi

# 2. mining for the new rows
log "=== v56 step 2: mining for $NREST new rows (miner $R2, fixed dense scale $SCALE) ==="
if [ ! -s "$DROWS" ]; then
    if [ "$BRANCH" = xl ]; then mine full "$NEW" --extra-max "$EXTRA_MAX" --dense-cache "$DCACHE"; else mine full "$NEW" --dense-cache "$DCACHE"; fi
    [ -s "$NEW" ] || { log "  mining FAILED (logs/dose_full.log): v56 stops"; exit 1; }
    log "  mined: pools $(side "$NEW" pools); teacher P(pos) median $(side "$NEW" teacher_positive_mass_median) (rows of the mined arm 0.852); negative above positive in $(side "$NEW" rows_with_negative_above_positive) of $(side "$NEW" rows) rows (mined arm 2663 of 20 292); student ranks a positive first in $(side "$NEW" student_ranks_a_positive_first) (0.844); negatives $(side "$NEW" by_origin)"
    [ "$BRANCH" = xl ] && log "  negatives from the shared pool $(side "$NEW" negatives_from_shared_pool) ($(side "$NEW" negatives_by_pool))"
fi

# 3. the row file: the base rows verbatim + the new rows
log "=== v56 step 3: rows = $BASEROWS (unchanged) + the new rows ==="
if [ ! -s "$DROWS" ]; then
    "$PY" - "$BASEROWS" "$NEW" "$DROWS" "$SCALE" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, math, random, statistics as st, sys
base, new, out, scale = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
def mass(r):
    s = [max(r["pos_scores"])] + list(r["neg_scores"]); m = max(s); e = [math.exp((x - m) / 2.0) for x in s]
    return e[0] / sum(e)
def load(path):
    lines, ids, p, flip, npos = [], set(), [], 0, 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        assert len(r["neg"]) == 10 and len(r["neg_scores"]) == 10 and len(r["pos_scores"]) == len(r["pos"]), "row shape: " + str(r["query_id"])
        lines.append(line if line.endswith("\n") else line + "\n"); p.append(mass(r)); flip += max(r["neg_scores"]) > min(r["pos_scores"]); npos += len(r["pos"])
        if r.get("translation") == "squadv2-v2":
            ids.add(str(r["query_id"]))
        if path == new:
            assert abs(r["dense_scale"] - scale) < 1e-9, "dense scale of a new row differs from the fixed scale"
    return lines, ids, p, flip, npos
bl, bids, bp, bflip, bpos = load(base)
nl, nids, np_, nflip, npos = load(new)
assert len(nids) == len(nl), "new rows: duplicate or non-SQuAD query ids"
assert not (bids & nids), "new rows overlap the base rows: %d query ids" % len(bids & nids)
order = list(range(len(bl) + len(nl))); random.Random(42).shuffle(order)
both = bl + nl
with open(out + ".tmp", "w", encoding="utf-8") as f:
    for i in order:
        f.write(both[i])
import os
os.replace(out + ".tmp", out)
stats = {"base_rows_file": base, "base_rows": len(bl), "new_rows": len(nl), "rows": len(both), "training_pairs": bpos + npos, "base_training_pairs": bpos,
         "squadv2_rows": len(bids) + len(nids), "dense_scale": scale,
         "teacher_positive_mass_median_base": st.median(bp), "teacher_positive_mass_median_new": st.median(np_),
         "teacher_positive_mass_above_095_base": sum(x > 0.95 for x in bp) / len(bp), "teacher_positive_mass_above_095_new": sum(x > 0.95 for x in np_) / len(np_),
         "rows_with_negative_above_positive_base": bflip, "rows_with_negative_above_positive_new": nflip}
json.dump(stats, open(out + ".dose.json", "w"), indent=2)
print("rows: %d = %d base (verbatim) + %d new; SQuAD-v2 rows %d; training pairs %d (base %d)" % (len(both), len(bl), len(nl), len(bids) + len(nids), bpos + npos, bpos))
print("teacher P(pos) at T=2, median: new rows %.3f, base rows %.3f; share above 0.95: new %.3f, base %.3f; negative above positive: new %d of %d, base %d of %d"
      % (st.median(np_), st.median(bp), stats["teacher_positive_mass_above_095_new"], stats["teacher_positive_mass_above_095_base"], nflip, len(nl), bflip, len(bl)))
PYEOF
    [ -s "$DROWS" ] || { log "  row file $DROWS not written: v56 stops"; exit 1; }
    rm -f "$NEW"   # every new row is in $DROWS; the sidecar $NEW.mined.json stays
fi
NROWS=$(grep -c . "$DROWS")
log "  $DROWS: $NROWS rows"

# 4. the arm
log "=== v56 step 4: arm $D ($NROWS rows, 2 epochs, the recipe of the mined arm) ==="
sed -e "4s|.*|# The DOSE arm (queue v56): the rows of the reference arm ($BASEROWS, unchanged) plus the $NREST gated SQuAD-v2 rows outside n5 with negatives mined by r2v12 under the same rules (5 best + 5 random of its top-60, length window 0.5-2.0; branch $BRANCH), scored by the ensemble teacher (soup-v9b + PolDense-400M) with the fixed dense scale $SCALE. The recipe of the mined arm from soup-v12, only the row file differs.|" \
    -e "s|^runs_dir: .*|runs_dir: runs/$D|" -e "s|^train_data: .*|train_data: $DROWS|" "$BASECFG" > "$CFG"
train "${D#arm-}" "$CFG"
REF="mined arm bias010 $(lbs $M-bias010) / $(tc $M-bias010) at $(dims $M-bias010); bias015 $(lbs $M-bias015) / $(tc $M-bias015) at $(dims $M-bias015); seed 43 bias015 $(lbs $M-s43-bias015) at $(dims $M-s43-bias015); reference arm $BASEARM $(lbs "$BASEARM")"
at_cost "$D" "$REF"; D_PICK=${PICK:-}
[ -d "models/$D" ] || D_PICK=""

# query-side index cost
COST="$RESULTS_ROOT/index_cost_v56.json"
if [ -n "$D_PICK" ] && [ ! -s "$COST" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    args=(--model base=models/base-splade-wrapped); seen=" "
    for m in $M-bias010 $M-bias011 $M-bias012 $M-bias013 $M-bias015 $M-s43-bias015 "$BASEARM" "$D_PICK"; do
        case "$seen" in *" $m "*) continue;; esac
        [ -d "models/$m" ] && { args+=(--model "$m=models/$m"); seen="$seen$m "; }
    done
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" --output-json "$COST" > "$LOGS/index_cost_v56.log" 2>&1 \
        && log "index cost written ($COST) in $((SECONDS - s))s" || { log "index cost FAILED (logs/index_cost_v56.log)"; rm -f "$COST"; }
fi

# 5. reading and summary
log "=== v56 step 5: reading and summary ==="
SUM="$RESULTS_ROOT/summary_v56.md"
"$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$COST" "$PLAN" "$DROWS.dose.json" "$NEW.mined.json" "$M" "$R2" "${D_PICK:-none}" "$BASEARM" "$BRANCH" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, spa = Path(sys.argv[1]), Path(sys.argv[2])
cost_path, plan_path, dose_path, mined_path, m, r2, pick, basearm, branch = sys.argv[3:12]
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
cost = load(cost_path).get("models", {}); plan = load(plan_path); dose = load(dose_path); mined = load(mined_path)
def c(label, key, nd=0):
    s = cost.get(label); return fmt(s[key], nd) if s else "-"
print("# v56 (12.11, quantized regime, nDCG@10): the query dose of the student-mined arm\n")
print("dose3 = the rows of the reference arm unchanged plus the gated SQuAD-v2 rows outside n5, negatives mined by r2v12 under the same rules, the same "
      "ensemble teacher with the fixed dense scale, the same recipe (from soup-v12, 2 epochs). Branch: **%s** (%s). New arms are measured in their first bias "
      "form with at most 410 dims/doc. The cost columns come from the cost sample (1 350 passages + 1 350 queries).\n"
      % (branch, "rows' own passages, as in the mined arm" if branch == "own" else "the large pool of v53; base rows and reference = the v53 arm"))
if dose:
    print("Rows: %d = %d base + %d new (training pairs %d against %d). Teacher P(pos) at T = 2, median: new rows %.3f, base rows %.3f; share above 0.95: %.3f / %.3f; "
          "rows with a negative above a positive: %d of %d new, %d of %d base. Pools of the new rows: %s.\n"
          % (dose["rows"], dose["base_rows"], dose["new_rows"], dose["training_pairs"], dose["base_training_pairs"], dose["teacher_positive_mass_median_new"],
             dose["teacher_positive_mass_median_base"], dose["teacher_positive_mass_above_095_new"], dose["teacher_positive_mass_above_095_base"],
             dose["rows_with_negative_above_positive_new"], dose["new_rows"], dose["rows_with_negative_above_positive_base"], dose["base_rows"], mined.get("pools", "-")))
print("| model | tiny10 | heldout8 | trec-covid | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |\n|---|---:|---:|---:|---:|---:|---:|---:|")
labels = [r2] + [f"{m}-bias0{t}" for t in ("10", "11", "12", "13", "15")] + [f"{m}-s43-bias015"]
for extra in (basearm, pick):
    if extra != "none" and extra not in labels:
        labels.append(extra)
for label in labels:
    if val(label, "tiny10") is None:
        continue
    name = f"**{label}**" if label == pick else label
    print(f"| {name} | {fmt(val(label, 'tiny10'))} | {fmt(val(label, 'heldout8'))} | {fmt(val(label, 'heldout2_treccovid'))} | {fmt(dims(label), 1)} | {c(label, 'doc_active_mean', 1)} | {c(label, 'query_active_mean', 1)} | {c(label, 'postings_per_query')} |")
verdict = "the dose arm has no complete result"
t, h, d = val(pick, "tiny10", 10), val(pick, "heldout8", 8), dims(pick)
if None not in (t, h, d):
    cands =[x for x in ([basearm] if branch == "xl" else []) + [f"{m}-bias0{s}" for s in ("10", "11", "12", "13", "15")]
             if None not in (val(x, "tiny10", 10), val(x, "heldout8", 8), dims(x))]
    if branch == "xl" and basearm in cands:
        ref = basearm
    else:
        ref = min(cands, key=lambda x: abs(dims(x) - d)) if cands else None
    if ref:
        dt, dh = t - val(ref, "tiny10"), h - val(ref, "heldout8")
        tc_new, tc_ref = val(pick, "heldout2_treccovid"), val(ref, "heldout2_treccovid")
        if dt >= 0.2 and dh >= 0.2:
            word = "THE DOSE PAYS (>= +0.2 on both sets)"
        elif dt <= -0.2 and dh <= -0.2:
            word = "the dose HURTS (<= -0.2 on both sets)"
        else:
            word = "within the seed noise: the 20 k recipe stays"
        verdict = "%s %.2f / %.2f at %.1f dims/doc against %s %.2f / %.2f at %.1f: delta %+.2f / %+.2f, trec-covid %s against %s -> %s" % (
            pick, t, h, d, ref, val(ref, "tiny10"), val(ref, "heldout8"), dims(ref), dt, dh, fmt(tc_new), fmt(tc_ref), word)
print("\n## Reading\n\n" + verdict + ".")
print("\nSeed noise of this recipe is 0.13-0.2 (one seed here). Nothing further was started by this queue.")
if mined:
    print("\n## Mining of the new rows\n\n```\n" + json.dumps(mined, indent=2, ensure_ascii=False) + "\n```")
if plan:
    print("\n## Branch decision (v53 result files)\n\n```\n" + json.dumps({k: v for k, v in plan.items() if k not in ("mined_forms",)}, indent=2) + "\n```")
EOF
{ echo; echo "## Queue state (v56)"; echo; echo '```'; grep -E "mined-dose3|chain:dose_" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
grep -A 2 '^## Reading' "$SUM" | tail -n 1 | cut -c1-600 | while read -r l; do log "  READING: $l"; done
git add "$SUM" "$PLAN" "$EXP_ROOT/queues/queue_v56.sh" "$CFG" 2>/dev/null
[ -e "$COST" ] && git add "$COST" 2>/dev/null   # the row sidecars (datasets/ is not tracked) are quoted in the summary
for m in ${D_PICK:-} $D; do for p in "$REGIMES_ROOT/quantized/$m" "$RESULTS_ROOT/sparsity/$m.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done; done
git add "$RESULTS_ROOT"/sparsity/$D-bias0??.json 2>/dev/null
git commit -q -m "Add the v56 results: the query dose of the student-mined arm" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v56.log" 2>&1 && { echo pushed >> "$LOGS/push_v56.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v56 done ($(date '+%d.%m %H:%M')) ==="
