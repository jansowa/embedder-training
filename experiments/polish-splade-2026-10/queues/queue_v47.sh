#!/usr/bin/env bash
# Queue v47 (3060 Ti, night 9/10.11 + day 10.11, ~14 h; optional steps start only before 10.11 17:00): two techniques
# not used so far, plus the repair of the heldout8 runs that lost fiqa-pl to a full disk on 9.11.
#   0  fiqa-pl (and maupqa-1z10 for soup-v16) for the five heldout8 runs with 7/8 or 6/8 tasks; folded into heldout8.json.
#   1  JOINT doc-side training with the static query table (distill.train_joint_static): the static line kept the index
#      of the full model frozen and stalled 4 points below it (phr2-k8-kd-r2 60.20 / 38.6 vs r2v12-bias005 64.21 / 43.08).
#      Here the document encoder (start r2v12-bias005) is trained on the n5 + kj rows (KL to the ensemble scores,
#      FLOPS kept) so the documents take over the expansion work a static query cannot do. Two arms:
#      (a) table frozen = phr2-k8-kd-r2 (the deployed query side keeps working, only the index changes);
#      (b) table trained too. Each: asym tiny10 (table over the new index); heldout8 when tiny10 >= 61.0; the new doc
#      model's own symmetric tiny10 (is it still a full model); dims/doc.
#      Reading: >= 62 = the joint route is the static line's way forward; 60-61 = the gap is mostly the query side.
#   2  CROSS-ENCODER TEACHER: every teacher so far was a bi-encoder (soup-v9b, PolDense-400M). The n5 + kj rows get
#      sdadas/polish-reranker-roberta-v3 logits (data kitchen bridge/reranker_pair_scores.py, ~225 k pairs, ~50 min),
#      scaled to soup-v9b units by within-row least squares (scripts/reranker_rows.py). Arms, round 2 from soup-v12 as
#      r2v12: ens3 = mean(v9b, pd400, reranker); trr = the reranker alone. sparsity, tiny10, heldout8, trec-covid each.
#      Reading against r2v12 64.20 / 43.15 / 67.02 and soup-v14s 64.18 / 43.11 / 70.00.
#   3  the ALBERT student (kd-hl-r2) stage 3 on the ens3 rows when the ens3 full model holds heldout8 >= 43.10:
#      sparsity, tiny10, heldout8 (against arm-hl-r2-s3n5kj 63.72 / 44.48 on 7 tasks).
#   4  task-vector extrapolation: soup-v17 = soup-v12 + 1.5 (r2v12 - soup-v12): tiny10, trec-covid (soup-v15 = 0.5 gave
#      64.20 / 43.10 / 68.91).
#   5  summary_v47.md, commit (with the v45 leftovers: its git add had failed on an unmatched path), push loop.
#   WAIT_FOR="" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v47.sh > logs/queue_v47.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v47 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v47.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/joint" "$SREG/quantized"
KITCHEN="$HOME/work/speakleash/retrieval-data-kitchen"
DEADLINE=$(date -d '2026-11-10 17:00' +%s)
late() { [ "$(date +%s)" -ge "$DEADLINE" ] && { log "past the 10.11 17:00 deadline: $1 skipped"; return 0; }; return 1; }

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-}; do
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
log "=== v47 start: $(free_gb) GB free on / ==="

# 0. the lost fiqa-pl / 1z10 tasks of 9.11
log "=== v47 step 0: fiqa-pl (and 1z10) for the heldout8 runs cut by the full disk ==="
refill() {  # refill ROOT LABEL MODEL TASK...: run each single task set and fold it into heldout8.json
    local root="$1" label="$2" model="$3"; shift 3
    local ts srcs=()
    for ts in "$@"; do REGIMES_ROOT=$root regime quantized "$label" "$model" "$ts" 16 && srcs+=("$root/quantized/$label/$ts.json"); done
    [ ${#srcs[@]} -gt 0 ] && "$PY" scripts/merge_regime_task.py --into "$root/quantized/$label/heldout8.json" --from "${srcs[@]}" --n-tasks 8 | while read -r l; do log "  $l"; done
}
refill "$REGIMES_ROOT" soup-v14s-bias005 models/soup-v14s-bias005 heldout8_fiqa
refill "$REGIMES_ROOT" $R2-s43 models/$R2-s43 heldout8_fiqa
refill "$REGIMES_ROOT" soup-v16 models/soup-v16 heldout8_fiqa heldout8_1z10
refill "$SREG" arm-hl-full-s3n5kj models/arm-hl-full-s3n5kj heldout8_fiqa
refill "$SREG" arm-hl-r2-s3n5kj models/arm-hl-r2-s3n5kj heldout8_fiqa
log "  heldout8 (8/8): soup-v14s-bias005 $(lbv soup-v14s-bias005 heldout8), r2v12-s43 $(lbv $R2-s43 heldout8), soup-v16 $(lbv soup-v16 heldout8), hl-full-s3n5kj $(slbv arm-hl-full-s3n5kj heldout8), hl-r2-s3n5kj $(slbv arm-hl-r2-s3n5kj heldout8) (soup-v14s 43.11, r2v12 43.15, hl-full-s3 42.10)"

# 1. joint doc-side training with the static table
log "=== v47 step 1: joint training of the document encoder with the static query table ==="
ROWS=datasets/arms/n5-plus-kj-tens400.jsonl
TABLE=models/static-r2v12b/phr2-k8-kd-r2
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
joint phr2-frozen --freeze-table
joint phr2-both

# 2. the cross-encoder teacher
log "=== v47 step 2: cross-encoder teacher scores (sdadas/polish-reranker-roberta-v3) on n5 + kj ==="
A=datasets/arms
[ -s "$A/n5-plus-kj-tv9b.jsonl" ] || cat "$A/fiqa-plus-squadv2v2-18k-tv9b.jsonl" "$A/karoljez-2219-tv9b.jsonl" > "$A/n5-plus-kj-tv9b.jsonl"
[ -s "$A/n5-plus-kj-tpd400.jsonl" ] || cat "$A/fiqa-plus-squadv2v2-18k-tpd400.jsonl" "$A/karoljez-2219-tpd400.jsonl" > "$A/n5-plus-kj-tpd400.jsonl"
[ -s "$A/n5-plus-kj.pairs.parquet" ] || "$PY" scripts/reranker_rows.py pairs --rows "$ROWS" --out "$A/n5-plus-kj.pairs.parquet" | while read -r l; do log "  $l"; done
if [ ! -s "$A/n5-plus-kj.rr.parquet" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS; log "reranker scoring of n5 + kj pairs"
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 14400 "$PY" "$KITCHEN/bridge/reranker_pair_scores.py" --input "$A/n5-plus-kj.pairs.parquet" --output "$A/n5-plus-kj.rr.parquet" --batch-size 32 > "$LOGS/rerank_n5kj.log" 2>&1
    [ -s "$A/n5-plus-kj.rr.parquet" ] && { log "reranker scoring done in $((SECONDS - s))s"; record "rerank:n5kj" ok "-" "$((SECONDS - s))s"; } \
        || { log "reranker scoring FAILED (logs/rerank_n5kj.log)"; record "rerank:n5kj" FAILED "-" "$((SECONDS - s))s"; }
fi
if [ -s "$A/n5-plus-kj.rr.parquet" ]; then
    [ -s "$A/n5-plus-kj-trr.jsonl" ] || "$PY" scripts/reranker_rows.py merge --rows "$ROWS" --scores "$A/n5-plus-kj.rr.parquet" --reference "$A/n5-plus-kj-tv9b.jsonl" --out "$A/n5-plus-kj-trr.jsonl" > "$LOGS/rerank_merge.log" 2>&1 \
        && log "  reranker rows: $(tr -d '\n' < "$A/n5-plus-kj-trr.jsonl.teacher.json" | tr -s ' ')"
    [ -s "$A/n5-plus-kj-tens3.jsonl" ] || "$PY" scripts/reranker_rows.py ensemble --inputs "$A/n5-plus-kj-tv9b.jsonl" "$A/n5-plus-kj-tpd400.jsonl" "$A/n5-plus-kj-trr.jsonl" --out "$A/n5-plus-kj-tens3.jsonl" | while read -r l; do log "  ens3 $l"; done
fi
ENS3=arm-dense-kl-n5kj-ens3-r2v12; TRR=arm-dense-kl-n5kj-trr-r2v12
if [ -s "$A/n5-plus-kj-tens3.jsonl" ]; then
    train dense-kl-n5kj-ens3-r2v12 "$C/distil_dense_kl_n5kj_ens3_r2v12.yaml"
    measure $ENS3 models/$ENS3 "r2v12 (two-way teacher) 64.20 / 43.15; soup-v14s 64.18 / 43.11"
    h2task $ENS3 heldout2_treccovid; log "  $ENS3 trec-covid $(tc $ENS3) (r2v12 67.02, soup-v14s 70.00)"
fi
if [ -s "$A/n5-plus-kj-trr.jsonl" ] && ! late "reranker-only arm"; then
    train dense-kl-n5kj-trr-r2v12 "$C/distil_dense_kl_n5kj_trr_r2v12.yaml"
    measure $TRR models/$TRR "ens3 $(lbs $ENS3); r2v12 64.20 / 43.15"
    h2task $TRR heldout2_treccovid; log "  $TRR trec-covid $(tc $TRR) (ens3 $(tc $ENS3), r2v12 67.02)"
fi

# 3. the ALBERT student on the ens3 rows
log "=== v47 step 3: ALBERT stage 3 on the ens3 rows ==="
if [ -s "$A/n5-plus-kj-tens3.jsonl" ] && [ "$(gte "$(lbv $ENS3 heldout8)" 43.10)" = 1 ] && ! late "student on ens3"; then
    small train hl-r2-s3n5kj-ens3 "$SC/stage3_hl_r2_n5kj_ens3.yaml"
    smeasure arm-hl-r2-s3n5kj-ens3 models/arm-hl-r2-s3n5kj-ens3 "two-way teacher arm-hl-r2-s3n5kj $(slbs arm-hl-r2-s3n5kj); ens3 full model $(lbs $ENS3)"
else log "  ens3 heldout8 $(lbv $ENS3 heldout8) (< 43.10 or missing): the student stays on the two-way rows"; fi

# 4. task-vector extrapolation
log "=== v47 step 4: soup-v17 = soup-v12 + 1.5 (r2v12 - soup-v12) ==="
if [ ! -d models/soup-v17 ] && ! late "soup-v17"; then
    cp -r models/soup-v12 models/soup-v17 && "$PY" - models/soup-v12 "models/$R2" models/soup-v17 1.5 <<'EOF' && log "soup-v17 written" || { log "soup-v17 FAILED"; rm -rf models/soup-v17; }
import sys, torch
from safetensors.torch import load_file, save_file
base, arm, out, alpha = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
b, a = load_file(base + "/model.safetensors"), load_file(arm + "/model.safetensors")
t = {k: (b[k].float() + alpha * (a[k].float() - b[k].float())).to(b[k].dtype) if k in a and b[k].dtype.is_floating_point else b[k] for k in b}
save_file(t, out + "/model.safetensors", metadata={"format": "pt"})
print("extrapolated", len(t), "tensors, alpha", alpha)
EOF
fi
if [ -d models/soup-v17 ]; then
    sparsity soup-v17 models/soup-v17; regime quantized soup-v17 models/soup-v17 tiny10 16; h2task soup-v17 heldout2_treccovid
    log "  soup-v17: tiny10 $(lbv soup-v17 tiny10), trec-covid $(tc soup-v17) (soup-v15 at 0.5: 64.20 / 68.91; r2v12 64.20 / 67.02), $(dims soup-v17) dims/doc"
fi

# 5. summary
log "=== v47 step 5: summary ==="
SUM="$RESULTS_ROOT/summary_v47.md"
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
print("# v47 (night 9/10.11 + day 10.11, quantized regime, nDCG@10): joint doc-side training with the static table, the cross-encoder teacher, the repaired heldout8 runs\n")
print("## Full models (heldout8 = 8/8 unless marked)\n")
print("| model | tiny10 | heldout8 | dims/doc | " + " | ".join(H2) + " | held-out-2 | 23 tasks |\n|---|---:|---:|---:|" + "---:|" * (len(H2) + 2))
for m in ["soup-v9b", "soup-v12-bias005", "arm-dense-kl-n5kj-ens400", "soup-v14s", "soup-v14s-bias005", "soup-v15", "soup-v17", "arm-dense-kl-n5kj-ens400-r2v12",
          "arm-dense-kl-n5kj-ens400-r2v12-s43", "soup-v16", "arm-dense-kl-n5kj-ens400-r2v12-bias005", "arm-dense-kl-n5kj-ens3-r2v12", "arm-dense-kl-n5kj-trr-r2v12"]:
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
side = arms / "n5-plus-kj-trr.jsonl.teacher.json"
if side.exists():
    print("\n## Cross-encoder teacher (sdadas/polish-reranker-roberta-v3) on n5 + kj\n\n```\n" + side.read_text() + "```")
EOF
{ echo; echo "## Queue state (v47)"; echo; echo '```'; grep -E "heldout8_fiqa|heldout8_1z10|joint|rerank|ens3|trr-r2v12|soup-v17" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
# the v45 leftovers go in too (its own git add failed on an unmatched path)
git add "$SUM" "$RESULTS_ROOT/summary_v45.md" "$EXP_ROOT/queues/queue_v45.sh" "$EXP_ROOT/queues/queue_v47.sh" \
    "$C/distil_dense_kl_n5_ens400_r2v12.yaml" "$C/distil_dense_kl_n5kj_ens3_r2v12.yaml" "$C/distil_dense_kl_n5kj_trr_r2v12.yaml" \
    "$SC/stage3_hl_full_n5kj_ens400.yaml" "$SC/stage3_hl_r2_n5kj_ens400.yaml" "$SC/stage3_hl_r2_n5kj_ens3.yaml" \
    distill/train_joint_static.py scripts/reranker_rows.py scripts/merge_regime_task.py configs/pirb_tasks/heldout8_fiqa.json configs/pirb_tasks/heldout8_1z10.json \
    "$REGIMES_ROOT"/quantized/soup-v15 "$REGIMES_ROOT"/quantized/soup-v14/heldout2_treccovid.json "$RESULTS_ROOT"/sparsity/soup-v15.json 2>/dev/null
for p in "$REGIMES_ROOT"/quantized/soup-v14s-bias005 "$REGIMES_ROOT"/quantized/$R2-s43 "$REGIMES_ROOT"/quantized/soup-v16 "$REGIMES_ROOT"/quantized/$ENS3 "$REGIMES_ROOT"/quantized/$TRR "$REGIMES_ROOT"/quantized/soup-v17 \
    "$RESULTS_ROOT"/sparsity/$ENS3.json "$RESULTS_ROOT"/sparsity/$TRR.json "$RESULTS_ROOT"/sparsity/soup-v17.json \
    "$SREG"/quantized/arm-hl-full-s3n5kj "$SREG"/quantized/arm-hl-r2-s3n5kj "$SREG"/quantized/arm-hl-r2-s3n5kj-bias010 "$SREG"/quantized/asym-r2v12b-q-arm-hl-r2-s3n5kj \
    "$SREG"/quantized/asym-r2v12b-q-static-phr2-k8-kd-r2-q300k "$SREG"/quantized/arm-hl-r2-s3n5kj-ens3 "$SREG"/quantized/asym-joint-phr2-frozen "$SREG"/quantized/asym-joint-phr2-both \
    "$SREG"/quantized/joint-phr2-frozen-doc-sym "$SREG"/quantized/joint-phr2-both-doc-sym "$SRES"/sparsity "$SRES"/fidelity/kd-hl-r2.queries.json "$SRES"/static-r2v12b "$SRES"/joint "$SRES"/index_cost_v45.json \
    "$A"/n5-plus-kj-trr.jsonl.teacher.json; do [ -e "$p" ] && git add "$p" 2>/dev/null; done
git commit -q -m "Add the v45 and v47 results: the ALBERT student from the r2v12 teacher, joint doc-side training with the static table and the cross-encoder teacher" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v47.log" 2>&1 && { echo pushed >> "$LOGS/push_v47.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v47 done ($(date '+%d.%m %H:%M')) ==="
