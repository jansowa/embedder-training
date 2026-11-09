#!/usr/bin/env bash
# Queue v46 (3060 Ti, 9.11 evening/night, after v45, ~6 h): the publication candidate after v44.
# v44: the seed soup soup-v14s (uniform of n5kj seeds 42 / 43) has the best held-out-2 (69.73, trec-covid 70.00) at
# 64.18 / 43.11, while r2v12 (64.20 / 43.15) drops trec-covid to 67.02 and its bias forms further (66.02 / 65.35).
#   1  soup-v14s-bias005 (lm_head.bias -0.0513): sparsity, tiny10, heldout8; the five held-out-2 tasks when heldout8 holds
#      within 0.15 of soup-v14s (43.11). Reading: the bias form as the publication recipe over the seed soup.
#   2  round-2 seed 43 (r2v12-s43, from soup-v12, 85 min) -> sparsity, tiny10, heldout8; soup-v16 = uniform(r2v12 s42, s43)
#      -> sparsity, tiny10, heldout8, trec-covid; the other four held-out-2 tasks when trec-covid >= 69.0.
#      Reading: does the seed soup hold trec-covid for round 2 as it did for round 1.
#   3  summary_v46.md with the 23-task mean of every model that has all three sets, commit, push.
#   WAIT_FOR="v45" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v46.sh > logs/queue_v46.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v46 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v46.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v45}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
gte()  { "$PY" -c "import sys;a,b,d=sys.argv[1:4];print(int(bool(a) and bool(b) and float(a)>=float(b)+float(d)))" "$1" "$2" "${3:-0}"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
h2task() { local bs=16; case "$2" in heldout2_poquad|heldout2_treccovid) bs=8;; esac; [ -d "models/$1" ] && regime quantized "$1" "models/$1" "$2" "$bs"; }
heldout2() { for ts in heldout2_treccovid heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad; do h2task "$1" "$ts"; done; }
tc() { lbv "$1" heldout2_treccovid; }
measure() {  # measure LABEL MODEL REFLINE
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    sparsity "$label" "$model"; regime quantized "$label" "$model" tiny10 16; regime quantized "$label" "$model" heldout8 16
    log "  $label: $(lbs "$label") ($ref), $(dims "$label") dims/doc"
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
log "=== v46 start: $(free_gb) GB free on /; soup-v14s $(lbs soup-v14s), trec-covid $(tc soup-v14s); r2v12 $(lbs $R2), trec-covid $(tc $R2) ==="

# 1. the bias form of the seed soup
log "=== v46 step 1: soup-v14s-bias005 ==="
biased models/soup-v14s soup-v14s-bias005 0.05 && measure soup-v14s-bias005 models/soup-v14s-bias005 "soup-v14s $(lbs soup-v14s) at $(dims soup-v14s) dims/doc; r2v12-bias005 64.21 / 43.08 at 365"
if [ "$(gte "$(lbv soup-v14s-bias005 heldout8)" "$(lbv soup-v14s heldout8)" -0.15)" = 1 ]; then heldout2 soup-v14s-bias005
    log "  soup-v14s-bias005 trec-covid $(tc soup-v14s-bias005) (soup-v14s 70.00, soup-v12-bias005 70.24)"
else log "  soup-v14s-bias005 heldout8 $(lbv soup-v14s-bias005 heldout8) more than 0.15 below soup-v14s: no held-out-2"; fi

# 2. round-2 seed 43 and the seed soup of round 2
log "=== v46 step 2: r2v12 seed 43 and soup-v16 = uniform(r2v12 s42, s43) ==="
train dense-kl-n5kj-ens400-r2v12-s43 "$C/distil_dense_kl_n5kj_ens400_r2v12_s43.yaml"
measure $R2-s43 models/$R2-s43 "seed 42: $(lbs $R2)"
soup soup-v16 "models/$R2" "models/$R2-s43" && measure soup-v16 models/soup-v16 "r2v12 $(lbs $R2); soup-v14s $(lbs soup-v14s)"
h2task soup-v16 heldout2_treccovid; log "  soup-v16 trec-covid $(tc soup-v16) (r2v12 $(tc $R2), soup-v14s 70.00)"
if [ "$(gte "$(tc soup-v16)" 69.0)" = 1 ]; then heldout2 soup-v16; else log "  soup-v16 trec-covid $(tc soup-v16) < 69.0: no further held-out-2"; fi

# 3. summary
log "=== v46 step 3: summary ==="
SUM="$RESULTS_ROOT/summary_v46.md"
"$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, spa = Path(sys.argv[1]), Path(sys.argv[2])
H2 = ["czywiesz", "gpt35wiki", "quora", "poquad", "treccovid"]
def val(m, ts):
    try:
        return json.load(open(reg / m / f"{ts}.json"))["mean_ndcg@10"]
    except Exception:
        return None
def fmt(x): return "-" if x is None else f"{x:.2f}"
def dims(m):
    try:
        d = json.load(open(spa / f"{m}.json"))[m]; v = [t["mean"] for t in d.values() if t.get("mean") is not None]; return f"{sum(v)/len(v):.0f}"
    except Exception:
        return "-"
print("# v46 (9.11, quantized regime, nDCG@10): the publication candidate after v44 — the bias form of the seed soup, round-2 seed 43 and its seed soup\n")
print("23-task mean = (10 x tiny10 + 8 x heldout8 + 5 x held-out-2) / 23. polish-splade (leaderboard) 55.77, soup-v9b 57.58.\n")
print("| model | tiny10 | heldout8 | dims/doc | " + " | ".join(H2) + " | held-out-2 | 23 tasks |\n|---|---:|---:|---:|" + "---:|" * (len(H2) + 2))
for m in ["soup-v9b", "soup-v12", "soup-v12-bias005", "arm-dense-kl-n5kj-ens400", "arm-dense-kl-n5kj-ens400-s43", "soup-v14", "soup-v14s", "soup-v14s-bias005", "soup-v15",
          "arm-dense-kl-n5kj-ens400-r2v12", "arm-dense-kl-n5kj-ens400-r2v12-s43", "soup-v16", "arm-dense-kl-n5kj-ens400-r2v12-bias005", "arm-dense-kl-n5kj-ens400-r2v12-bias010",
          "arm-dense-kl-n5-ens400-r2v12", "arm-dense-kl-n5-ens400-r2v12-bias005"]:
    t, h = val(m, "tiny10"), val(m, "heldout8"); h2 = [val(m, f"heldout2_{x}") for x in H2]
    if t is None and all(x is None for x in h2):
        continue
    full = all(x is not None for x in h2)
    mean = f"{sum(h2)/5:.2f}" if full else "-"
    m23 = f"{(10*t + 8*h + sum(h2))/23:.2f}" if full and t is not None and h is not None else "-"
    print(f"| {m} | {fmt(t)} | {fmt(h)} | {dims(m)} | " + " | ".join(fmt(x) for x in h2) + f" | {mean} | {m23} |")
EOF
{ echo; echo "## Queue state (v46)"; echo; echo '```'; grep -E "soup-v14s-bias005|r2v12-s43|soup-v16" "$STATE"; echo '```'; } >> "$SUM"
log "summary written to $SUM"
git add "$SUM" "$C/distil_dense_kl_n5kj_ens400_r2v12_s43.yaml" "$EXP_ROOT/queues/queue_v46.sh" \
    "$REGIMES_ROOT"/quantized/soup-v14s-bias005 "$REGIMES_ROOT"/quantized/$R2-s43 "$REGIMES_ROOT"/quantized/soup-v16 \
    "$RESULTS_ROOT"/sparsity/soup-v14s-bias005.json "$RESULTS_ROOT"/sparsity/$R2-s43.json "$RESULTS_ROOT"/sparsity/soup-v16.json 2>/dev/null
git commit -q -m "Add the v46 results: the bias form of the seed soup and the seed soup of round 2" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v46.log" 2>&1 && { echo pushed >> "$LOGS/push_v46.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v46 done ($(date '+%d.%m %H:%M')) ==="
