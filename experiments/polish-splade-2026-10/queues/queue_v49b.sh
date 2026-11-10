#!/usr/bin/env bash
# Queue v49b (3060 Ti, day 10.11 after v49a, ~9 h to the last training start; optional steps start only before
# 10.11 17:00, DEADLINE_AT overrides): two levers for the full model that were never tried, then the static-table
# steps of v47 that v49a left out.
#   1  SMOKE of the new mining chain on 300 rows (miner, both teachers, ensemble); a failure skips step 2 only.
#   2  STUDENT-MINED NEGATIVES under the ensemble teacher. On n5 + kj the two-way teacher puts 0.997 of the softmax
#      mass on the positive (median, T = 2; 86 % of the rows above 0.95), so the KL target is close to one-hot. Here
#      r2v12 ranks the passages of each row group (the rows' own texts only: fiqa, squadv2-v2, kj apart), every
#      query takes 5 best + 5 random of its top-60 non-positive passages inside a 0.5-2.0 length window of its
#      positive (kitchen bridge/splade_mine_row_negatives.py), soup-v9b and PolDense-400M score the new rows, the
#      ensemble is their per-pair mean. Arm arm-dense-kl-n5kj-mined-ens400-r2v12 = the r2v12 recipe on these rows.
#      Reading against r2v12 64.20 / 43.15 / 67.02 (seed noise 0.13-0.2): heldout8 >= 43.45 = harder candidates
#      carry signal (next: the same over 100-200 k kd_v1 queries); within noise = the candidates are not the limit.
#      (28.10: student-mined negatives hurt under MNRL with one teacher and no length match.)
#   3  FLOPS WITH AN L0 THRESHOLD (document_regularizer_threshold 256): the r2v12 recipe, documents with at most
#      256 active dimensions are not regularised. Reading: nDCG against dims/doc (r2v12 404).
#   4  v47 step 1, first arm: joint doc-side training with the frozen static table (phr2-frozen).
#   5  the reranker-only arm (v47 step 2, trr), the ALBERT student on the ens3 rows (v47 step 3), joint phr2-both.
#   6  summary_v49.md, commit (with the v45 / v47 leftovers), push loop.
# queue_v48.sh (unchanged) can run behind this queue with WAIT_FOR="v49b".
#   WAIT_FOR="v49a" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v49b.sh > logs/queue_v49b.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v49b MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v49b.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
DEADLINE=$(date -d "${DEADLINE_AT:-2026-11-10 17:00}" +%s)
late() { [ "$(date +%s)" -ge "$DEADLINE" ] && { log "past the deadline: $1 skipped"; return 0; }; return 1; }

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v49a}; do
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
smeasure() {  # smeasure LABEL MODEL REFLINE (small series)
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    small sparsity "$label" "$model"; small regime quantized "$label" "$model" tiny10 16; small regime quantized "$label" "$model" heldout8 16
    log "  $label: $(slbs "$label") ($ref), $(sparsity_mean "$SRES/sparsity/$label.json" "$label" 2>/dev/null) dims/doc"
}
R2=arm-dense-kl-n5kj-ens400-r2v12
R2B="models/$R2-bias005"
ENS3=arm-dense-kl-n5kj-ens3-r2v12; TRR=arm-dense-kl-n5kj-trr-r2v12
MINED=arm-dense-kl-n5kj-mined-ens400-r2v12; L0=arm-dense-kl-n5kj-ens400-r2v12-l0t256
A=datasets/arms
ROWS=$A/n5-plus-kj-tens400.jsonl
TABLE=models/static-r2v12b/phr2-k8-kd-r2
PD=OPI-PIB/PolDense-400M
log "=== v49b start: $(free_gb) GB free on /; ens3 $(lbs $ENS3) trec-covid $(tc $ENS3) (r2v12 64.20 / 43.15 / 67.02) ==="

kscript() {  # kscript TAG TIMEOUT OUTPUT COMMAND...: a GPU helper script; OUTPUT is removed when it fails
    local tag="$1" to="$2" out="$3"; shift 3
    wait_gpu_free; wait_ram; local s=$SECONDS rc
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$to" "$@" > "$LOGS/$tag.log" 2>&1; rc=$?
    [ $rc = 0 ] && [ -s "$out" ] || { rm -f "$out"; rc=1; }
    log "  $tag rc=$rc in $((SECONDS - s))s: $(tail -n 1 "$LOGS/$tag.log" | cut -c1-300)"
    record "chain:$tag" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    return $rc
}
mine_and_score() {  # mine_and_score PREFIX TAG [miner args]: PREFIX.jsonl (mined rows), -tv9b, -tpd400, -tens400
    local pre="$1" tag="$2"; shift 2
    [ -s "$pre.jsonl" ] || kscript "mine_$tag" 5400 "$pre.jsonl" "$PY" "$KITCHEN/bridge/splade_mine_row_negatives.py" --student "models/$R2" --rows "$ROWS" --output "$pre.jsonl" "$@" || return 1
    [ -s "$pre-tv9b.jsonl" ] || kscript "tscore_v9b_$tag" 5400 "$pre-tv9b.jsonl" "$PY" "$KITCHEN/bridge/splade_teacher_scores.py" --teacher models/soup-v9b \
        --rows "$pre.jsonl" --output "$pre-tv9b.jsonl" --max-seq-length 512 --batch-size 16 || return 1
    [ -s "$pre-tpd400.jsonl" ] || kscript "tscore_pd400_$tag" 5400 "$pre-tpd400.jsonl" "$PY" "$KITCHEN/bridge/dense_teacher_scores.py" --teacher "$PD" --preset poldense --dtype bf16 \
        --rows "$pre.jsonl" --reference "$pre-tv9b.jsonl" --output "$pre-tpd400.jsonl" --max-seq-length 512 --batch-size 32 --device cuda || return 1
    [ -s "$pre-tens400.jsonl" ] || "$PY" scripts/reranker_rows.py ensemble --inputs "$pre-tv9b.jsonl" "$pre-tpd400.jsonl" --out "$pre-tens400.jsonl" | while read -r l; do log "  ensemble $l"; done
    [ -s "$pre-tens400.jsonl" ] || { log "  ensemble file $pre-tens400.jsonl missing"; return 1; }
    "$PY" - "$pre-tens400.jsonl" "$ROWS" <<'PYEOF' | while read -r l; do log "  $l"; done
import json, math, statistics as st, sys
def stats(path, limit=0):
    p, flip, n, L = [], 0, 0, []
    for line in open(path, encoding="utf-8"):
        if limit and n >= limit: break
        r = json.loads(line); n += 1
        s = [max(r["pos_scores"])] + list(r["neg_scores"]); m = max(s); e = [math.exp((x - m) / 2.0) for x in s]
        p.append(e[0] / sum(e)); flip += max(r["neg_scores"]) > min(r["pos_scores"]); L += [len(t) for t in r["neg"]]
    return n, st.median(p), sum(x > 0.95 for x in p) / n, flip, st.median(L)
n, med, share, flip, ln = stats(sys.argv[1])
print("mined rows: %d, teacher P(pos) at T=2 median %.3f, share > 0.95 %.3f, rows with a negative above a positive %d, median negative length %d chars" % (n, med, share, flip, ln))
n, med, share, flip, ln = stats(sys.argv[2], n)
print("old rows (same %d): teacher P(pos) median %.3f, share > 0.95 %.3f, negative above positive %d, median negative length %d chars" % (n, med, share, flip, ln))
PYEOF
}

# 1. smoke
log "=== v49b step 1: smoke of the mining chain (300 rows) ==="
SMOKE=cache/smoke_v49/n5kj-mined; rm -rf cache/smoke_v49; mkdir -p cache/smoke_v49
if mine_and_score "$SMOKE" smoke --limit 300; then CHAIN_OK=1; log "  smoke passed"; else CHAIN_OK=0; log "  smoke FAILED: step 2 skipped"; fi
rm -rf cache/smoke_v49

# 2. student-mined negatives under the ensemble teacher
log "=== v49b step 2: student-mined, length-matched negatives under the ensemble teacher ==="
if [ "$CHAIN_OK" = 1 ] && mine_and_score "$A/n5-plus-kj-mined" full; then
    log "  miner: $(tr -d '\n' < "$A/n5-plus-kj-mined.jsonl.mined.json" | tr -s ' ' | cut -c1-600)"
    train dense-kl-n5kj-mined-ens400-r2v12 "$C/distil_dense_kl_n5kj_mined_ens400_r2v12.yaml"
    measure $MINED models/$MINED "r2v12 (old negatives) 64.20 / 43.15; ens3 $(lbs $ENS3)"
    h2task $MINED heldout2_treccovid; log "  $MINED trec-covid $(tc $MINED) (r2v12 67.02, soup-v14s 70.00)"
else log "  mined rows missing: arm skipped"; fi

# 3. FLOPS with an L0 threshold
log "=== v49b step 3: FLOPS with an L0 threshold (documents above 256 active dimensions only) ==="
if ! late "L0 threshold arm"; then
    train dense-kl-n5kj-ens400-r2v12-l0t256 "$C/distil_dense_kl_n5kj_ens400_r2v12_l0t256.yaml"
    measure $L0 models/$L0 "r2v12 (no threshold) 64.20 / 43.15, 404 dims/doc"
    h2task $L0 heldout2_treccovid; log "  $L0 trec-covid $(tc $L0) (r2v12 67.02)"
fi

# 4. joint doc-side training with the static table (v47 step 1)
log "=== v49b step 4: joint training of the document encoder with the frozen static query table ==="
joint() {  # joint NAME [extra args]: train, copy to models/joint-r2v12b-NAME/{doc,query}, asym tiny10 (+ heldout8 when >= 61), symmetric tiny10 of the doc model
    local name="$1"; shift
    local dir="models/joint-r2v12b-$name"
    if [ ! -f "$dir/doc/model.safetensors" ]; then
        need_space || return 1; wait_gpu_free; wait_ram; local s=$SECONDS; log "joint $name ($*)"
        timeout -k 60 "$TRAIN_TIMEOUT" "$PY" -m distill.train_joint_static --rows "$ROWS" --doc-model "$R2B" --table "$TABLE" --out "runs/joint-r2v12b-$name" \
            --epochs 2 --rows-per-step 2 --grad-accum 16 --lr 2e-5 --table-lr 5e-3 --temperature 2.0 --doc-flops 1e-3 --query-flops 5e-5 \
            --val-rows 400 --eval-every 200 --log-every 20 --seed 42 "$@" > "$LOGS/joint_$name.log" 2>&1
        if [ -f "runs/joint-r2v12b-$name/final/doc/model.safetensors" ]; then
            mkdir -p "$dir"; cp -r "runs/joint-r2v12b-$name/final/doc" "$dir/doc"; cp -r "runs/joint-r2v12b-$name/final/query" "$dir/query"
            cp "runs/joint-r2v12b-$name/joint_summary.json" "$SRES/joint/$name.json"
            log "joint $name done in $((SECONDS - s))s: $("$PY" -c "import json;d=json.load(open('$SRES/joint/$name.json'));v=d['val_final'];print('val top1 %.3f pos-first %.3f dims/doc %.1f dims/q %.1f' % (v['top1_agreement'], v['positive_first'], v['dims_per_doc'], v['dims_per_query']))")"
            record "joint:r2v12b:$name" ok "-" "$((SECONDS - s))s"
        else log "joint $name FAILED (logs/joint_$name.log)"; record "joint:r2v12b:$name" FAILED "-" "$((SECONDS - s))s"; return 1; fi
    fi
    export PIRB_QUERY_MODEL="$dir/query"
    small regime quantized "asym-joint-$name" "$dir/doc" tiny10 16
    local v; v=$(slbv "asym-joint-$name" tiny10)
    log "  static table over the jointly trained index ($name): tiny10 $v (frozen index: phr2-k8-kd-r2 60.20; symmetric r2v12-bias005 64.21)"
    if [ "$(gte "$v" 61.0)" = 1 ]; then small regime quantized "asym-joint-$name" "$dir/doc" heldout8 16
        log "  asym-joint-$name heldout8 $(slbv "asym-joint-$name" heldout8) (frozen index 38.6; symmetric 43.08)"
    else log "  asym-joint-$name tiny10 $v < 61.0: no heldout8"; fi
    unset PIRB_QUERY_MODEL
    small sparsity "joint-$name-doc" "$dir/doc"
    small regime quantized "joint-$name-doc-sym" "$dir/doc" tiny10 16
    log "  joint-$name doc model symmetric tiny10 $(slbv "joint-$name-doc-sym" tiny10) (r2v12-bias005 64.21), $(sparsity_mean "$SRES/sparsity/joint-$name-doc.json" "joint-$name-doc" 2>/dev/null) dims/doc"
}
late "joint phr2-frozen" || joint phr2-frozen --freeze-table

# 5. the reranker-only arm, the ALBERT student on the ens3 rows, the second joint arm
log "=== v49b step 5: reranker-only arm, ALBERT stage 3 on the ens3 rows, joint phr2-both ==="
if [ -s "$A/n5-plus-kj-trr.jsonl" ] && ! late "reranker-only arm"; then
    train dense-kl-n5kj-trr-r2v12 "$C/distil_dense_kl_n5kj_trr_r2v12.yaml"
    measure $TRR models/$TRR "ens3 $(lbs $ENS3); r2v12 64.20 / 43.15"
    h2task $TRR heldout2_treccovid; log "  $TRR trec-covid $(tc $TRR) (ens3 $(tc $ENS3), r2v12 67.02)"
fi
if [ -s "$A/n5-plus-kj-tens3.jsonl" ] && [ "$(gte "$(lbv $ENS3 heldout8)" 43.10)" = 1 ] && ! late "student on ens3"; then
    small train hl-r2-s3n5kj-ens3 "$SC/stage3_hl_r2_n5kj_ens3.yaml"
    smeasure arm-hl-r2-s3n5kj-ens3 models/arm-hl-r2-s3n5kj-ens3 "two-way teacher arm-hl-r2-s3n5kj $(slbs arm-hl-r2-s3n5kj); ens3 full model $(lbs $ENS3)"
else log "  ens3 heldout8 $(lbv $ENS3 heldout8) (< 43.10 or missing): the student stays on the two-way rows"; fi
late "joint phr2-both" || joint phr2-both

# 6. summary
log "=== v49b step 6: summary ==="
SUM="$RESULTS_ROOT/summary_v49.md"
"$PY" - "$REGIMES_ROOT/quantized" "$SREG/quantized" "$RESULTS_ROOT/sparsity" "$SRES/sparsity" "$SRES/joint" "$A" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, sreg, spa, sspa, joint, arms = map(Path, sys.argv[1:7])
H2 = ["czywiesz", "gpt35wiki", "quora", "poquad", "treccovid"]
def val(root, m, ts):
    try:
        return json.load(open(root / m / f"{ts}.json"))["mean_ndcg@10"]
    except Exception:
        return None
def ntasks(root, m, ts):
    try:
        return json.load(open(root / m / f"{ts}.json"))["n_tasks"]
    except Exception:
        return 0
def fmt(x): return "-" if x is None else f"{x:.2f}"
def dims(root, m):
    try:
        d = json.load(open(root / f"{m}.json"))[m]; v = [t["mean"] for t in d.values() if t.get("mean") is not None]; return f"{sum(v)/len(v):.0f}"
    except Exception:
        return "-"
print("# v49 (day 10.11, quantized regime, nDCG@10): the cross-encoder teacher, student-mined negatives under the ensemble teacher, FLOPS with an L0 threshold, joint doc-side training with the static table, the repaired heldout8 runs\n")
print("## Full models (heldout8 = 8/8 unless marked)\n")
print("| model | tiny10 | heldout8 | dims/doc | " + " | ".join(H2) + " | held-out-2 | 23 tasks |\n|---|---:|---:|---:|" + "---:|" * (len(H2) + 2))
for m in ["soup-v9b", "soup-v12-bias005", "arm-dense-kl-n5kj-ens400", "soup-v14s", "soup-v14s-bias005", "soup-v15", "soup-v17", "arm-dense-kl-n5kj-ens400-r2v12",
          "arm-dense-kl-n5kj-ens400-r2v12-s43", "soup-v16", "arm-dense-kl-n5kj-ens400-r2v12-bias005", "arm-dense-kl-n5kj-ens3-r2v12", "arm-dense-kl-n5kj-trr-r2v12", "arm-dense-kl-n5kj-mined-ens400-r2v12", "arm-dense-kl-n5kj-ens400-r2v12-l0t256"]:
    t, h = val(reg, m, "tiny10"), val(reg, m, "heldout8"); h2 = [val(reg, m, f"heldout2_{x}") for x in H2]
    if t is None and all(x is None for x in h2):
        continue
    hs = fmt(h) + ("" if ntasks(reg, m, "heldout8") in (0, 8) else f" ({ntasks(reg, m, 'heldout8')}/8)")
    full = all(x is not None for x in h2)
    mean = f"{sum(h2)/5:.2f}" if full else "-"
    m23 = f"{(10*t + 8*h + sum(h2))/23:.2f}" if full and t is not None and h is not None and ntasks(reg, m, "heldout8") == 8 else "-"
    print(f"| {m} | {fmt(t)} | {hs} | {dims(spa, m)} | " + " | ".join(fmt(x) for x in h2) + f" | {mean} | {m23} |")
print("\n## ALBERT student (50 M)\n")
print("| model | tiny10 | heldout8 | dims/doc |\n|---|---:|---:|---:|")
for m in ["arm-b4-l0125-f1536-hl-r256-w-full-s3", "hl-full-s3-bias010", "arm-hl-full-s3n5kj", "arm-hl-r2-s3n5kj", "arm-hl-r2-s3n5kj-bias010", "asym-r2v12b-q-arm-hl-r2-s3n5kj", "arm-hl-r2-s3n5kj-ens3"]:
    if val(sreg, m, "tiny10") is None:
        continue
    hs = fmt(val(sreg, m, "heldout8")) + ("" if ntasks(sreg, m, "heldout8") in (0, 8) else f" ({ntasks(sreg, m, 'heldout8')}/8)")
    print(f"| {m} | {fmt(val(sreg, m, 'tiny10'))} | {hs} | {dims(sspa, m)} |")
print("\n## Static query table phr2-k8-kd-r2 over a jointly trained index (frozen index of r2v12-bias005: 60.20 / 38.6; symmetric 64.21 / 43.08)\n")
print("| arm | asym tiny10 | asym heldout8 | doc model symmetric tiny10 | dims/doc | val top1 | val dims/doc | val dims/q |\n|---|---:|---:|---:|---:|---:|---:|---:|")
for name in ["phr2-frozen", "phr2-both"]:
    f = joint / f"{name}.json"
    v = json.load(open(f))["val_final"] if f.exists() else {}
    print(f"| {name} | {fmt(val(sreg, f'asym-joint-{name}', 'tiny10'))} | {fmt(val(sreg, f'asym-joint-{name}', 'heldout8'))} | {fmt(val(sreg, f'joint-{name}-doc-sym', 'tiny10'))} | {dims(sspa, f'joint-{name}-doc')} | {v.get('top1_agreement', 0):.3f} | {v.get('dims_per_doc', 0):.0f} | {v.get('dims_per_query', 0):.1f} |")
for extra, title in [("n5-plus-kj-mined.jsonl.mined.json", "Student-mined negatives (r2v12 as the miner)"), ("n5-plus-kj-mined-tpd400.jsonl.teacher.json", "PolDense-400M on the mined rows")]:
    if (arms / extra).exists():
        print(f"\n## {title}\n\n```\n" + (arms / extra).read_text() + "```")
side = arms / "n5-plus-kj-trr.jsonl.teacher.json"
if side.exists():
    print("\n## Cross-encoder teacher (sdadas/polish-reranker-roberta-v3) on n5 + kj\n\n```\n" + side.read_text() + "```")
EOF
{ echo; echo "## Queue state (v49)"; echo; echo '```'; grep -E "heldout8_fiqa|heldout8_1z10|joint|rerank|ens3|trr-r2v12|soup-v17|mined|l0t256|chain:" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
# the v45 leftovers go in too (its own git add failed on an unmatched path)
git add "$SUM" "$RESULTS_ROOT/summary_v45.md" "$EXP_ROOT/queues/queue_v45.sh" "$EXP_ROOT/queues/queue_v47.sh" "$EXP_ROOT/queues/queue_v48.sh" "$EXP_ROOT/queues/queue_v49a.sh" "$EXP_ROOT/queues/queue_v49b.sh" "$C/distil_dense_kl_n5kj_mined_ens400_r2v12.yaml" "$C/distil_dense_kl_n5kj_ens400_r2v12_l0t256.yaml" \
    "$C/distil_dense_kl_n5_ens400_r2v12.yaml" "$C/distil_dense_kl_n5kj_ens3_r2v12.yaml" "$C/distil_dense_kl_n5kj_trr_r2v12.yaml" \
    "$SC/stage3_hl_full_n5kj_ens400.yaml" "$SC/stage3_hl_r2_n5kj_ens400.yaml" "$SC/stage3_hl_r2_n5kj_ens3.yaml" \
    distill/train_joint_static.py scripts/reranker_rows.py scripts/merge_regime_task.py configs/pirb_tasks/heldout8_fiqa.json configs/pirb_tasks/heldout8_1z10.json \
    "$REGIMES_ROOT"/quantized/soup-v15 "$REGIMES_ROOT"/quantized/soup-v14/heldout2_treccovid.json "$RESULTS_ROOT"/sparsity/soup-v15.json 2>/dev/null
for p in "$REGIMES_ROOT"/quantized/soup-v14s-bias005 "$REGIMES_ROOT"/quantized/$R2-s43 "$REGIMES_ROOT"/quantized/soup-v16 "$REGIMES_ROOT"/quantized/$ENS3 "$REGIMES_ROOT"/quantized/$TRR "$REGIMES_ROOT"/quantized/soup-v17 \
    "$RESULTS_ROOT"/sparsity/$ENS3.json "$RESULTS_ROOT"/sparsity/$TRR.json "$RESULTS_ROOT"/sparsity/soup-v17.json \
    "$SREG"/quantized/arm-hl-full-s3n5kj "$SREG"/quantized/arm-hl-r2-s3n5kj "$SREG"/quantized/arm-hl-r2-s3n5kj-bias010 "$SREG"/quantized/asym-r2v12b-q-arm-hl-r2-s3n5kj \
    "$SREG"/quantized/asym-r2v12b-q-static-phr2-k8-kd-r2-q300k "$SREG"/quantized/arm-hl-r2-s3n5kj-ens3 "$SREG"/quantized/asym-joint-phr2-frozen "$SREG"/quantized/asym-joint-phr2-both \
    "$SREG"/quantized/joint-phr2-frozen-doc-sym "$SREG"/quantized/joint-phr2-both-doc-sym "$SRES"/sparsity "$SRES"/fidelity/kd-hl-r2.queries.json "$SRES"/static-r2v12b "$SRES"/joint "$SRES"/index_cost_v45.json \
    "$A"/n5-plus-kj-trr.jsonl.teacher.json "$REGIMES_ROOT"/quantized/$MINED "$REGIMES_ROOT"/quantized/$L0 "$RESULTS_ROOT"/sparsity/$MINED.json "$RESULTS_ROOT"/sparsity/$L0.json; do [ -e "$p" ] && git add "$p" 2>/dev/null; done
git commit -q -m "Add the v45 and v49 results: the cross-encoder teacher, student-mined negatives, FLOPS with an L0 threshold and joint doc-side training with the static table" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v49.log" 2>&1 && { echo pushed >> "$LOGS/push_v49.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v49b done ($(date '+%d.%m %H:%M')) ==="
