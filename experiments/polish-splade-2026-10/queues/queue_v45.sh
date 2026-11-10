#!/usr/bin/env bash
# Queue v45 (3060 Ti, day 9.11, after s8 and v44, ~10-11 h; optional steps start only before 17:30): where each of the
# three model lines can still gain with the data on this machine.
#   Full SPLADE: r2v12 / r2v12-bias005 (64.20 / 43.15, 64.21 / 43.08) lost trec-covid on held-out-2 (66.02 against
#   69.4-70.2), which costs as much on the 23-task mean as every tiny10 gain since soup-v12. v44 says which step lost it.
#   ALBERT (hl-full-s3 62.97 / 42.10, bias010 63.17 / 42.01) was distilled from soup-v9b (63.65 / 42.56) and ranked on the
#   n4 rows; the teacher is now r2v12 (+0.55 / +0.59) and the rows are n5 + kj with the ensemble scores.
#   Static tables: s8 writes its own results tonight; one more pool step here at the end when time allows.
#   1  trec-covid for soup-v15 = uniform(r2v12, soup-v12) and soup-v14 (22 min each); soup-v15 tiny10 + heldout8 when it
#      holds trec-covid >= 68.5
#   2  ALBERT, two arms: (a) stage 3 of the s3 KD student on the n5 + kj rows with the ensemble scores (the r2v12 recipe);
#      (b) KD continuation of the same student from r2v12 on 600 k kd_v1 texts (lr 2e-5), then the same stage 3.
#      Each: sparsity, tiny10, heldout8. Reading against hl-full-s3 62.97 / 42.10 and b4-full-s3 63.77 / 42.54.
#   3  r2 from soup-v12 on the n5 rows alone (no colleague rows): when v44 shows n5kj's trec-covid >= 1.0 below ens-n5v2's
#      (the colleague rows are the cause) or v44 left no reading. tiny10, heldout8, trec-covid; when it keeps trec-covid
#      >= 68.5 at heldout8 >= 43.0: bias005 form -> tiny10, heldout8, the other four held-out-2 tasks.
#   4  the better ALBERT arm by heldout8: as a query encoder over the r2v12-bias005 index (tiny10, heldout8) and its
#      bias010 form (sparsity, tiny10, heldout8).
#   5  static: a third / fourth mined pool of 300 k queries (seed 12) for the best s8 table, 3 epochs, tiny10; heldout8 when
#      it beats the s8 best by >= 0.15.
#   6  summary_v45.md (both series), commit (no third_party paths), push.
#   WAIT_FOR="s8 v44" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v45.sh > logs/queue_v45.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v45 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800} KD_TIMEOUT=${KD_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v45.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"
SEXP=experiments/polish-splade-small-2026-11
SC="$SEXP/configs/3060"
SRES="$SEXP/results"; SREG="$SEXP/results-regimes"
mkdir -p "$SRES/sparsity" "$SRES/fidelity" "$SRES/static-r2v12b" "$SREG/quantized"
CORPUS=${CORPUS:-datasets/kd/kd_v1.jsonl}
DEADLINE=$(date -d '2026-11-09 17:30' +%s)
late() { [ "$(date +%s)" -ge "$DEADLINE" ] && { log "past the 17:30 deadline: $1 skipped"; return 0; }; return 1; }

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-s8 v44}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
slbv() { regime_mean "$SREG/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
slbs() { echo "$(slbv "$1" tiny10) / $(slbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
small() { EXP_ROOT=$SEXP RESULTS_ROOT=$SRES REGIMES_ROOT=$SREG "$@"; }   # queue_lib helpers writing into the small series
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
soup() {  # soup NAME MODEL...: uniform soup (merge only)
    local name="$1"; shift
    [ -d "models/$name" ] && return 0
    for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
    wait_gpu_free
    timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/soup_$name.log" 2>&1 \
        && log "soup $name written ($*, uniform)" || { log "soup $name FAILED (logs/soup_$name.log)"; rm -rf "models/$name"; return 1; }
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

R2=arm-dense-kl-n5kj-ens400-r2v12
N5KJ=arm-dense-kl-n5kj-ens400
ENS=arm-dense-kl-n5v2-ens400
HLKD=models/kd-b4-l0125-f1536-hl-r256-w-full
log "=== v45 start: $(free_gb) GB free on /; v44 trec-covid: r2v12 $(tc $R2), n5kj $(tc $N5KJ), ens-n5v2 $(tc $ENS), soup-v12 $(tc soup-v12), soup-v14s $(tc soup-v14s); r2v12-bias005 66.02, soup-v12-bias005 70.24, soup-v9b 69.66 ==="

# 1. trec-covid of the soups around r2v12
log "=== v45 step 1: trec-covid of soup-v15 = uniform(r2v12, soup-v12) and soup-v14 ==="
soup soup-v15 "models/$R2" models/soup-v12
h2task soup-v15 heldout2_treccovid; log "  soup-v15 trec-covid $(tc soup-v15)"
h2task soup-v14 heldout2_treccovid; log "  soup-v14 trec-covid $(tc soup-v14) (tiny10 64.15 / heldout8 43.00)"
if [ "$(gte "$(tc soup-v15)" 68.5)" = 1 ]; then measure soup-v15 models/soup-v15 "r2v12 $(lbs $R2); soup-v14 64.15 / 43.00"
else log "  soup-v15 trec-covid $(tc soup-v15) < 68.5: no tiny10 / heldout8"; fi

# 2. ALBERT: (a) stage 3 only, (b) KD continuation from r2v12 + stage 3
log "=== v45 step 2a: ALBERT stage 3 on n5 + kj with the ensemble scores (the s3 KD student) ==="
small train hl-full-s3n5kj "$SC/stage3_hl_full_n5kj_ens400.yaml"
smeasure arm-hl-full-s3n5kj models/arm-hl-full-s3n5kj "hl-full-s3 62.97 / 42.10 at 512 dims/doc; b4-full-s3 63.77 / 42.54; teacher r2v12 $(lbs $R2)"

log "=== v45 step 2b: KD continuation of the ALBERT from r2v12 (600 k texts) + stage 3 ==="
KD_ARGS=(--teacher "models/$R2" --corpus "$CORPUS" --max-items 600000 --max-seq-length 256 --batch-size 16
         --lr 2e-5 --warmup-steps 200 --schedule linear --epochs 1 --precision fp16 --head-chunk 64
         --save-every 5000 --keep-last 1 --fidelity-texts "$CORPUS" --fidelity-n 256 --log-every 50
         --token-loss kl --token-weight 0.3 --temperature 2.0 --top-k 128 --pooled-loss mse --pooled-weight 1.0 --seed 42)
SMOKE_OK=1
if [ ! -d models/kd-hl-r2 ]; then
    wait_gpu_free; wait_ram; s=$SECONDS; rm -rf runs/kd-hl-r2-smoke
    timeout -k 60 900 "$PY" -m distill.train_kd --student "$HLKD" --out runs/kd-hl-r2-smoke "${KD_ARGS[@]}" --max-items 2000 --max-steps 20 --save-every 20 > "$LOGS/kd_hl-r2-smoke.log" 2>&1
    if [ -f runs/kd-hl-r2-smoke/final/model.safetensors ]; then log "KD smoke from the s3 checkpoint passed in $((SECONDS - s))s"; rm -rf runs/kd-hl-r2-smoke
    else log "KD smoke from the s3 checkpoint FAILED (logs/kd_hl-r2-smoke.log); step 2b skipped"; SMOKE_OK=0; fi
fi
if [ "$SMOKE_OK" = 1 ]; then
    small kd hl-r2 "$HLKD" "${KD_ARGS[@]}"
    TEACHER="models/$R2" small fidelity kd-hl-r2.queries models/kd-hl-r2 "$CORPUS" --kinds query --n 512 --fp32-head
    small train hl-r2-s3n5kj "$SC/stage3_hl_r2_n5kj_ens400.yaml"
    smeasure arm-hl-r2-s3n5kj models/arm-hl-r2-s3n5kj "stage 3 only $(slbs arm-hl-full-s3n5kj); hl-full-s3 62.97 / 42.10"
fi

# 3. round 2 from soup-v12 on the n5 rows alone, when the colleague rows are the trec-covid suspect
log "=== v45 step 3: r2 from soup-v12 on n5 alone ==="
T_N5KJ=$(tc $N5KJ); T_ENS=$(tc $ENS); T_R2=$(tc $R2); RUN=0
if [ -z "$T_N5KJ" ] || [ -z "$T_ENS" ]; then RUN=1; log "  v44 left no trec-covid reading for n5kj / ens-n5v2: the n5-only arm runs as the fallback"
elif [ "$(gte "$T_ENS" "$T_N5KJ" 1.0)" = 1 ]; then RUN=1; log "  n5kj $T_N5KJ vs ens-n5v2 $T_ENS: the colleague rows are the suspect; the n5-only arm runs"
else log "  n5kj $T_N5KJ vs ens-n5v2 $T_ENS (r2v12 $T_R2): not the colleague rows; the n5-only arm skipped"; fi
N5R2=arm-dense-kl-n5-ens400-r2v12
if [ "$RUN" = 1 ] && ! late "n5-only round 2"; then
    train dense-kl-n5-ens400-r2v12 "$C/distil_dense_kl_n5_ens400_r2v12.yaml"
    measure $N5R2 models/$N5R2 "r2v12 $(lbs $R2); ens-n5v2 63.95 / 42.96"
    h2task $N5R2 heldout2_treccovid; log "  $N5R2 trec-covid $(tc $N5R2) (r2v12 $T_R2, ens-n5v2 $T_ENS)"
    if [ "$(gte "$(tc $N5R2)" 68.5)" = 1 ] && [ "$(gte "$(lbv $N5R2 heldout8)" 43.0)" = 1 ] && ! late "bias005 of the n5-only arm"; then
        biased models/$N5R2 $N5R2-bias005 0.05 && measure $N5R2-bias005 models/$N5R2-bias005 "without bias $(lbs $N5R2); r2v12-bias005 64.21 / 43.08"
        for ts in heldout2_treccovid heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do late "held-out-2 $ts" || h2task $N5R2-bias005 "$ts"; done
    else log "  $N5R2: trec-covid $(tc $N5R2) / heldout8 $(lbv $N5R2 heldout8): no bias form, no held-out-2"; fi
fi

# 4. the better ALBERT arm: asymmetric over the r2v12-bias005 index, and its bias010 form
log "=== v45 step 4: the better ALBERT arm asymmetric and at bias 0.10 ==="
HB=""; HB_V=""
for m in arm-hl-full-s3n5kj arm-hl-r2-s3n5kj; do
    v=$(slbv "$m" heldout8); [ -n "$v" ] || continue
    if [ -z "$HB_V" ] || [ "$(gte "$v" "$HB_V")" = 1 ]; then HB="$m"; HB_V="$v"; fi
done
if [ -n "$HB" ]; then
    log "  better ALBERT arm by heldout8: $HB ($(slbs "$HB"); hl-full-s3 62.97 / 42.10)"
    if ! late "asymmetric $HB"; then
        export PIRB_QUERY_MODEL="models/$HB"
        small regime quantized "asym-r2v12b-q-$HB" "models/$R2-bias005" tiny10 16
        small regime quantized "asym-r2v12b-q-$HB" "models/$R2-bias005" heldout8 16
        unset PIRB_QUERY_MODEL
        log "  $HB as the query encoder over the r2v12-bias005 index: $(slbs "asym-r2v12b-q-$HB") (symmetric r2v12-bias005 64.21 / 43.08; asym-hlfull-q-v9b 63.35 / 42.09)"
    fi
    late "bias010 of $HB" || { biased "models/$HB" "$HB-bias010" 0.10 && smeasure "$HB-bias010" "models/$HB-bias010" "without bias $(slbs "$HB"); hl-full-s3-bias010 63.17 / 42.01 at 404 dims/doc"; }
else log "  no ALBERT arm measured; step 4 skipped"; fi

# 5. static: one more mined pool (300 k queries) for the best s8 table
log "=== v45 step 5: static table, a 300 k-query pool for the best s8 table ==="
SM=models/static-r2v12b; DOC="models/$R2-bias005"; POOL5=cache/static_pools/kd-r2v12b-q300k
REF_ST=phr2-k8-kd-r2; REF_V=$(slbv asym-r2v12b-q-static-$REF_ST tiny10)
for cand in phr2-k8-kd-r3 phr2m3-k8-kd-r2; do
    v=$(slbv "asym-r2v12b-q-static-$cand" tiny10)
    [ -f "$SM/$cand/static_query.json" ] && [ "$(gte "$v" "$REF_V" 0.15)" = 1 ] && { REF_ST="$cand"; REF_V="$v"; }
done
log "  best s8 table: $REF_ST (tiny10 $REF_V; heldout8 $(slbv asym-r2v12b-q-static-$REF_ST heldout8))"
if ! late "static pool q300k" && [ -f "$SM/$REF_ST/static_query.json" ]; then
    if [ ! -f "$POOL5/pool.json" ]; then
        need_space && { wait_gpu_free; wait_ram; s=$SECONDS; log "mine $POOL5 (seed 12, 300 k queries, student $REF_ST)"
        timeout -k 60 7200 "$PY" -m distill.static_mine --doc-model "$DOC" --corpus "$CORPUS" --out "$POOL5" --student "$SM/$REF_ST" \
            --device cuda --dtype fp16 --n-docs 250000 --n-queries 300000 --teacher-top 32 --student-top 16 --max-candidates 64 \
            --batch-size 16 --search-batch 128 --seed 12 > "$LOGS/static_mine_kd-r2v12b-q300k.log" 2>&1
        [ -f "$POOL5/pool.json" ] && { log "mine done in $((SECONDS - s))s"; record "static_mine:kd-r2v12b-q300k" ok "-" "$((SECONDS - s))s"; } \
            || { log "mine FAILED"; record "static_mine:kd-r2v12b-q300k" FAILED "-" "$((SECONDS - s))s"; }; }
    fi
    NAME="$REF_ST-q300k"
    if [ -f "$POOL5/pool.json" ] && [ ! -f "$SM/$NAME/static_query.json" ]; then
        wait_gpu_free; wait_ram; s=$SECONDS; log "train $NAME from $REF_ST on q300k, 3 epochs"
        timeout -k 60 5400 "$PY" -m distill.train_static --pool "$POOL5" --init "$SM/$REF_ST" --out "runs/static-r2v12b-$NAME" --lr 5e-3 \
            --target rows --learn-scale --epochs 3 --batch-size 64 --warmup-steps 200 --schedule linear --val-rows 2000 --eval-every 500 \
            --log-every 100 --device cuda --seed 42 > "$LOGS/static_train_r2v12b_$NAME.log" 2>&1
        if [ -f "runs/static-r2v12b-$NAME/final/static_query.json" ]; then
            cp -r "runs/static-r2v12b-$NAME/final" "$SM/$NAME"; cp "runs/static-r2v12b-$NAME/static_summary.json" "$SRES/static-r2v12b/$NAME.json"
            log "train $NAME done in $((SECONDS - s))s"; record "static_train:r2v12b:$NAME" ok "-" "$((SECONDS - s))s"
        else log "train $NAME FAILED"; record "static_train:r2v12b:$NAME" FAILED "-" "$((SECONDS - s))s"; fi
    fi
    if [ -f "$SM/$NAME/static_query.json" ]; then
        export PIRB_QUERY_MODEL="$SM/$NAME"
        small regime quantized "asym-r2v12b-q-static-$NAME" "$DOC" tiny10 16
        v=$(slbv "asym-r2v12b-q-static-$NAME" tiny10); log "  static $NAME over r2v12-bias005: tiny10 $v ($REF_ST $REF_V)"
        if [ "$(gte "$v" "$REF_V" 0.15)" = 1 ] && ! late "heldout8 of $NAME"; then small regime quantized "asym-r2v12b-q-static-$NAME" "$DOC" heldout8 16
            log "  static $NAME heldout8 $(slbv "asym-r2v12b-q-static-$NAME" heldout8) ($REF_ST $(slbv asym-r2v12b-q-static-$REF_ST heldout8))"; fi
        unset PIRB_QUERY_MODEL
    fi
fi

# 6. summary
log "=== v45 step 6: summary ==="
SUM="$RESULTS_ROOT/summary_v45.md"
"$PY" - "$REGIMES_ROOT/quantized" "$SREG/quantized" "$RESULTS_ROOT/sparsity" "$SRES/sparsity" "$SRES/static-r2v12b" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, sreg, spa, sspa, st = map(Path, sys.argv[1:6])
H2 = ["czywiesz", "gpt35wiki", "quora", "poquad", "treccovid"]
def val(root, m, ts):
    f = root / m / f"{ts}.json"
    try:
        return json.load(open(f))["mean_ndcg@10"]
    except Exception:
        return None
def fmt(x): return "-" if x is None else f"{x:.2f}"
def dims(root, m):
    f = root / f"{m}.json"
    try:
        d = json.load(open(f))[m]; v = [t["mean"] for t in d.values() if t.get("mean") is not None]; return f"{sum(v)/len(v):.0f}"
    except Exception:
        return "-"
print("# v45 (day 9.11, quantized regime, nDCG@10): trec-covid around r2v12, the ALBERT student on the n5 + kj rows and from the r2v12 teacher, the n5-only round 2, one more static pool\n")
print("References: soup-v9b 63.65 / 42.56, trec-covid 69.66; soup-v12-bias005 64.05 / 42.83, trec-covid 70.24; r2v12 64.20 / 43.15; r2v12-bias005 64.21 / 43.08, trec-covid 66.02; hl-full-s3 62.97 / 42.10 (bias010 63.17 / 42.01); b4-full-s3 63.77 / 42.54.\n")
print("## Full models\n")
print("| model | tiny10 | heldout8 | dims/doc | " + " | ".join(H2) + " | held-out-2 mean |\n|---|---:|---:|---:|" + "---:|" * (len(H2) + 1))
for m in ["soup-v9b", "soup-v12", "soup-v12-bias005", "arm-dense-kl-n5v2-ens400", "arm-dense-kl-n5kj-ens400", "soup-v14", "soup-v14s", "soup-v15",
          "arm-dense-kl-n5kj-ens400-r2v12", "arm-dense-kl-n5kj-ens400-r2v12-bias005", "arm-dense-kl-n5kj-ens400-r2v12-bias010",
          "arm-dense-kl-n5-ens400-r2v12", "arm-dense-kl-n5-ens400-r2v12-bias005"]:
    h = [val(reg, m, f"heldout2_{t}") for t in H2]
    if all(x is None for x in h) and val(reg, m, "tiny10") is None:
        continue
    mean = f"{sum(h)/5:.2f}" if all(x is not None for x in h) else "-"
    print(f"| {m} | {fmt(val(reg, m, 'tiny10'))} | {fmt(val(reg, m, 'heldout8'))} | {dims(spa, m)} | " + " | ".join(fmt(x) for x in h) + f" | {mean} |")
print("\n## ALBERT student (50 M, 0.45 of the teacher's cost)\n")
print("| model | what | tiny10 | heldout8 | dims/doc |\n|---|---|---:|---:|---:|")
desc = {"arm-b4-l0125-f1536-hl-r256-w-full-s3": "s3: KD from soup-v9b, stage 3 on n4 (v9b scores)", "hl-full-s3-bias010": "s3 at bias 0.10",
        "arm-hl-full-s3n5kj": "s3 KD student, stage 3 on n5 + kj (ensemble scores)", "arm-hl-r2-s3n5kj": "KD continued from r2v12 (600 k), stage 3 on n5 + kj",
        "arm-hl-full-s3n5kj-bias010": "stage-3-only arm at bias 0.10", "arm-hl-r2-s3n5kj-bias010": "KD-continued arm at bias 0.10",
        "asym-r2v12b-q-arm-hl-full-s3n5kj": "query encoder over the r2v12-bias005 index", "asym-r2v12b-q-arm-hl-r2-s3n5kj": "query encoder over the r2v12-bias005 index"}
for m, what in desc.items():
    if val(sreg, m, "tiny10") is None:
        continue
    print(f"| {m} | {what} | {fmt(val(sreg, m, 'tiny10'))} | {fmt(val(sreg, m, 'heldout8'))} | {dims(sspa, m)} |")
print("\n## Static query tables over r2v12-bias005 (symmetric 64.21 / 43.08)\n")
print("| table | tiny10 | heldout8 | val top1 | dims/q | postings fraction |\n|---|---:|---:|---:|---:|---:|")
for f in sorted(st.glob("*.json")):
    name = f.stem; v = json.load(open(f)).get("val_final", {})
    print(f"| {name} | {fmt(val(sreg, f'asym-r2v12b-q-static-{name}', 'tiny10'))} | {fmt(val(sreg, f'asym-r2v12b-q-static-{name}', 'heldout8'))} | {v.get('top1_agreement', 0):.3f} | {v.get('dims_per_query', 0):.1f} | {v.get('postings_fraction', 0):.3f} |")
EOF
{ echo; echo "## Queue state (v45)"; echo; echo '```'; grep -E "soup-v15|soup-v14:heldout2|hl-full-s3n5kj|hl-r2|n5-ens400-r2v12|q300k" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
git add "$SUM" "$C/distil_dense_kl_n5_ens400_r2v12.yaml" "$SC/stage3_hl_full_n5kj_ens400.yaml" "$SC/stage3_hl_r2_n5kj_ens400.yaml" "$EXP_ROOT/queues/queue_v45.sh" \
    "$REGIMES_ROOT"/quantized/soup-v15 "$REGIMES_ROOT"/quantized/soup-v14/heldout2_treccovid.json "$REGIMES_ROOT"/quantized/arm-dense-kl-n5-ens400-r2v12* \
    "$RESULTS_ROOT"/sparsity/soup-v15.json "$RESULTS_ROOT"/sparsity/arm-dense-kl-n5-ens400-r2v12*.json \
    "$SREG"/quantized/arm-hl-* "$SREG"/quantized/asym-r2v12b-q-arm-hl-* "$SREG"/quantized/asym-r2v12b-q-static-*q300k "$SRES"/sparsity/arm-hl-*.json \
    "$SRES"/fidelity/kd-hl-r2.queries.json "$SRES"/static-r2v12b 2>/dev/null
git commit -q -m "Add the v45 results: trec-covid around r2v12, the ALBERT student on the n5 + kj rows and from the r2v12 teacher, the n5-only round 2" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v45.log" 2>&1 && { echo pushed >> "$LOGS/push_v45.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v45 done ($(date '+%d.%m %H:%M')) ==="
