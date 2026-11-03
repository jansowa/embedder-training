#!/usr/bin/env bash
# Queue mlm2 (5070 Ti, 4/5.11): probe M = SPLADE from the MLM checkpoint of OPI-PIB/pl-ModernBERT-512-base on the
# data, recipe and step grid of probe A (configs/next/5070/modernbert_probe_m.yaml; plan docs/PLAN-5070-MLM-2026-11-03.md
# §12.3). Self-contained: it waits for queue mlm1 (probe A) to exit, checks the repo and the data, downloads the
# model, benchmarks the training head in bf16 on this card and picks the fastest variant whose gradients match the
# stock path (scripts/splade_head_bench.py; SPEEDUP-SPLADE-3060.md §6), trains 5 000 steps with step checkpoints,
# measures sparsity + tiny10 (quantized regime, PIRB_DTYPE=bf16, 512 tokens) per checkpoint starting with s05000,
# writes the curve next to probe A at equal steps, the GM verdict and results/summary_5070_mlm2.md, commits, pushes.
#
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_5070_mlm2.sh > logs/queue_5070_mlm2.out 2>&1 < /dev/null &
#   tail -f logs/queue.log
#
# Environment: MLM2_END (default tomorrow 10:00), TRAIN_TIMEOUT (16 h), HEAD (auto | stock | exact | exact-nockpt:
# skip the benchmark and force a head), MLM2_WAIT_PID (wait for this pid before the first step).
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=mlm2 MIN_RAM_GB=6
export WANDB_MODE=${WANDB_MODE:-offline}
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-57600}
export TORCHINDUCTOR_COMPILE_THREADS=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_mlm2.pid"
guard_alive
C="$EXP_ROOT/configs/next/5070"
BASE_CFG="$C/modernbert_probe_m.yaml"
NAME=modernbert-probe-m
MODEL_ID=OPI-PIB/pl-ModernBERT-512-base
STEPS_PRIORITY=(5000 2500 1250 625 1875 3750)     # s05000 first: the GM gate needs it even when time runs out
MLM2_END=${MLM2_END:-$(date -d "tomorrow 10:00" '+%Y-%m-%d %H:%M')}
END=$(date -d "$MLM2_END" +%s)
B_BENCH=1800; B_SPARS=900; B_TINY=2400
HEAD=${HEAD:-auto}
GEN="runs/arm-$NAME/configs"; mkdir -p "$GEN" "$RESULTS_ROOT/data" "$RESULTS_ROOT/sparsity"

fits() { [ $(( $(date +%s) + $1 )) -le "$END" ]; }
step() { local need=$1; shift; fits "$need" || { log "no time before $MLM2_END for a ${need}s step; skipping: $*"; return 1; }; "$@"; }
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_.*$1"; }
pad() { printf '%05d' "$1"; }
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }

log "=== mlm2 start: end $MLM2_END; $(free_gb) GB free on /; HEAD=$HEAD ==="
if [ -n "${MLM2_WAIT_PID:-}" ]; then
    log "waiting for pid $MLM2_WAIT_PID"; while kill -0 "$MLM2_WAIT_PID" 2>/dev/null; do sleep 60; done
fi
if queue_alive mlm1; then
    log "waiting for queue mlm1 (pid $(cat "$LOGS/queue_mlm1.pid")) to exit"
    while queue_alive mlm1; do sleep 60; done; sleep 30
fi
# Probe A may still be training under another pid file name: wait for an idle card without the 40-minute cap of
# wait_gpu_free (a second process next to the probe would OOM the 16 GB card or kill the probe).
gpu_idle_strict() { while [ "$(gpu_busy)" != 0 ]; do sleep 60; done; sleep 20; [ "$(gpu_busy)" = 0 ] || gpu_idle_strict; }
log "waiting for an idle GPU (no cap)"; gpu_idle_strict; log "GPU idle"

# ---- 0. checks: nothing here is guessed; a missing piece stops the queue before it costs GPU time ---------------
missing=0
need_file() { [ -e "$1" ] || { log "MISSING: $1 ($2)"; missing=1; }; }
need_grep() { grep -q "$2" "$1" 2>/dev/null || { log "MISSING: '$2' in $1 ($3)"; missing=1; }; }
need_file "$BASE_CFG" "probe M config"
need_file training/splade_regularizer_control.py "density regulator (commit cc79edf)"
need_file training/backends/splade_fused_head.py "sparse-gradient head (commit b7e56bc)"
need_file scripts/splade_head_bench.py "head benchmark (commit b7e56bc)"
need_grep scripts/pirb_regime_eval.sh "PIRB_DTYPE=" "bf16 evaluation (commit 0d015aa)"
need_grep third_party/pirb/search/splade.py "vocab_size" "ModernBERT head has 50 008 logits for 50 002 tokens (submodule 0d015aa)"
need_file models/base-splade-wrapped "density reference of the regulator and of the sparsity script"
for f in datasets/judged/msmarco-clean-20000.jsonl datasets/judged/msmarco-plain-20000.jsonl datasets/arms/squadv2-fresh-16000.jsonl \
         datasets/arms/squadv2-2073.jsonl datasets/arms/fiqa-2073.jsonl; do need_file "$f" "probe data (transfer package)"; done
[ -x "${JAVA_HOME:-/nonexistent}/bin/java" ] || { log "MISSING: JDK 21 for Anserini (JAVA_HOME)"; missing=1; }
[ "$missing" = 0 ] || { log "checks failed; fix the items above and start the queue again"; record "mlm2:checks" FAILED "-"; exit 1; }
[ "$(free_gb)" -ge 40 ] || log "WARNING: only $(free_gb) GB free on /; 6 checkpoints need ~6 GB, the run ~3 GB"
log "checks ok"

# ---- 1. model download (network, no GPU) --------------------------------------------------------------------------
log "model $MODEL_ID (download or cache check)"
if ! timeout -k 30 1800 "$PY" -c "
from sentence_transformers import SparseEncoder
m = SparseEncoder('$MODEL_ID')
print('loaded', type(m[0]).__name__, type(m[1]).__name__, m[0].auto_model.config.num_hidden_layers, 'layers')
" > "$LOGS/mlm2_model_download.log" 2>&1; then log "model download FAILED ($LOGS/mlm2_model_download.log)"; record "mlm2:model" FAILED "-"; exit 1; fi
log "  $(tail -1 "$LOGS/mlm2_model_download.log")"

# ---- 2. head benchmark in bf16 on this card -> training config --------------------------------------------------
# stock8 = the config as written (SpladePooling chunk 8, gradient checkpointing); exact = sparse-gradient head;
# exact-nockpt = the same without checkpointing. Chosen: the fastest variant that ran (no OOM), whose forward is
# identical (max loss rel == 0) and whose gradients stay at rounding level (rel <= 0.02 in bf16; fp16 on the 3060
# gave 9e-4, SPEEDUP-SPLADE-3060.md §6.7), with a VRAM peak <= 13 GB on the benchmark step.
make_cfg() {  # make_cfg VARIANT -> path of a generated config (stock = the base config itself)
    local v="$1" out="$GEN/modernbert_probe_m_$1.yaml"
    [ "$v" = stock ] && { echo "$BASE_CFG"; return 0; }
    "$PY" - "$BASE_CFG" "$out" "$v" <<'PYEOF'
import sys, yaml
src, dst, variant = sys.argv[1:4]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
st = cfg.setdefault("sentence_transformers", {})
st["splade_fused_head"] = "exact"
st["gradient_checkpointing"] = variant != "exact-nockpt"
cfg["runs_dir"] = cfg.get("runs_dir", "runs/arm-modernbert-probe-m")
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(f"# generated by queue_5070_mlm2.sh from {src}: head variant {variant}\n")
    yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
PYEOF
    echo "$out"
}
BENCH="runs/bench-speed/modernbert/head_bench"
if [ "$HEAD" = auto ]; then
    if [ ! -f "$BENCH.json" ] && fits $B_BENCH; then
        need_space; gpu_idle_strict; wait_ram
        log "head benchmark (bf16, 8 x (1+1+6) x 256 tokens)"; s=$SECONDS
        ORPHAN_GUARD_EXEMPT=1 timeout -k 60 "$B_BENCH" "$PY" scripts/splade_head_bench.py --model "$MODEL_ID" --dtype bf16 \
            --rows datasets/judged/msmarco-clean-20000.jsonl --rows-per-step 32 --micro-batch 8 --negatives 6 --max-seq-length 256 \
            --repeats 3 --variants stock8,exact,exact-nockpt --output "$BENCH" > "$LOGS/mlm2_head_bench.log" 2>&1
        log "head benchmark finished in $((SECONDS - s))s (rc=$?)"
    fi
    HEAD=$("$PY" - "$BENCH.json" <<'PYEOF' 2>/dev/null || echo stock
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("stock"); sys.exit()
res = {r["name"]: r for r in d["results"]}
ok = {"stock"}
for name in ("exact", "exact-nockpt"):
    r = res.get(name)
    if not r: continue
    c = r["vs_reference"]
    if c["max_loss_rel_diff"] == 0 and c["grad_rel_diff"] <= 0.02 and r["peak_memory_gb"] <= 13.0:
        ok.add(name)
best = min(ok, key=lambda n: res.get("stock8" if n == "stock" else n, {"step_seconds": 1e9})["step_seconds"])
print(best)
PYEOF
)
    [ -f "$BENCH.md" ] && { log "head benchmark:"; sed -n '/^| variant/,/^$/p' "$BENCH.md" | sed 's/^/    /' | tee -a "$LOGS/queue.log" > /dev/null; }
fi
CFG=$(make_cfg "$HEAD")
SPS=$("$PY" -c "
import json,sys
try:
    d=json.load(open('$BENCH.json')); n={'stock':'stock8'}.get('$HEAD','$HEAD'); r=[x for x in d['results'] if x['name']==n][0]; print(round(r['step_seconds'],2))
except Exception: print('?')
")
log "training head: $HEAD (config $CFG), ~${SPS} s/step on the benchmark step -> 5 000 steps ~ $("$PY" -c "print(round($( [ "$SPS" = "?" ] && echo 0 || echo "$SPS")*5000/3600,1))") h"
if [ "$SPS" != "?" ] && [ "$(ge "$SPS" 11)" = 1 ]; then
    log "  > 11 s/step: physical batch 4 x 8 instead of 8 x 4 (NEXT-SESSION-5070-MLM.md, night 2 step 4)"
    B4="$GEN/$(basename "${CFG%.yaml}")_b4.yaml"
    "$PY" - "$CFG" "$B4" <<'PYEOF'
import sys, yaml
src, dst = sys.argv[1:3]
cfg = yaml.safe_load(open(src, encoding="utf-8")); st = cfg["sentence_transformers"]
st["train_batch_size"] = 4; st["gradient_accumulation_steps"] = 8
with open(dst, "w", encoding="utf-8") as fh:
    fh.write(f"# generated by queue_5070_mlm2.sh from {src}: physical batch 4 x 8\n"); yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
PYEOF
    CFG="$B4"
fi
record "mlm2:head" ok "$HEAD" "${SPS}s/step"

# ---- 3. training with step checkpoints -> models/<NAME>-sNNNNN (no optimizer files) and models/<NAME> ------------
train_probe() {
    local cfg="$1" run ck n
    if [ -d "models/$NAME-s05000" ]; then log "train $NAME: s05000 already in models/; skipping training"; return 0; fi
    need_space || return 1; wait_gpu_free; wait_ram
    log "train $NAME ($cfg, timeout ${TRAIN_TIMEOUT}s)"; local s=$SECONDS
    timeout -k 60 "$TRAIN_TIMEOUT" "$PY" run_experiments.py --grid-configuration-file "$cfg" \
        --backend sentence-transformers --training-type splade > "$LOGS/arm_$NAME.log" 2>&1
    local rc=$? e=$((SECONDS - s))
    run=$(ls -1dt "runs/arm-$NAME/sentence-transformers/splade"/*/ 2>/dev/null | head -1)
    [ -n "$run" ] || { log "train $NAME: no run directory after ${e}s (rc=$rc)"; record "train:$NAME" FAILED "-" "${e}s"; return 1; }
    for ck in "$run"/step-checkpoints/step-*; do
        [ -d "$ck" ] || continue
        n=$(pad "${ck##*/step-}")
        [ -d "models/$NAME-s$n" ] && continue
        cp -r "$ck" "models/$NAME-s$n.tmp" && rm -f "models/$NAME-s$n.tmp"/optimizer.pt "models/$NAME-s$n.tmp"/scheduler.pt "models/$NAME-s$n.tmp"/rng_state*.pth \
            && mv "models/$NAME-s$n.tmp" "models/$NAME-s$n" && log "  checkpoint models/$NAME-s$n"
    done
    [ -d "$run/final" ] && [ ! -d "models/$NAME" ] && cp -r "$run/final" "models/$NAME" && log "  final models/$NAME"
    if [ -d "models/$NAME-s05000" ]; then log "train $NAME done in ${e}s (rc=$rc)"; record "train:$NAME" ok "-" "${e}s"
    else log "train $NAME: no s05000 after ${e}s (rc=$rc; checkpoints: $(ls -d models/$NAME-s* 2>/dev/null | tr '\n' ' '))"; record "train:$NAME" FAILED "-" "${e}s"; return 1; fi
}
train_probe "$CFG"

# ---- 4. measurements per checkpoint, s05000 first ---------------------------------------------------------------
for st in "${STEPS_PRIORITY[@]}"; do
    m="$NAME-s$(pad "$st")"
    [ -d "models/$m" ] || { log "  $m missing; skipped"; continue; }
    step $B_SPARS sparsity "$m" "models/$m"
    PIRB_DTYPE=bf16 step $B_TINY regime quantized "$m" "models/$m" tiny10 16 512
    log "  $m: tiny10 $(lbv "$m" tiny10), $(dims "$m") dims/doc"
done

# ---- 5. curve next to probe A at equal steps, GM verdict, summary --------------------------------------------------
CURVE="$RESULTS_ROOT/data/$NAME-curve.json"
SUM="$RESULTS_ROOT/summary_5070_mlm2.md"
"$PY" - "$NAME" "$CURVE" "$SUM" "$HEAD" "$CFG" <<'PYEOF' 2>> "$LOGS/queue_5070_mlm2.out"
import glob, json, os, sys, time
name, curve_path, summary_path, head, cfg = sys.argv[1:6]
R = os.environ.get("REGIMES_ROOT", "experiments/polish-splade-2026-10/results-regimes")
S = os.environ.get("RESULTS_ROOT", "experiments/polish-splade-2026-10/results")
STEPS = [625, 1250, 1875, 2500, 3750, 5000]

def tiny(label):
    try:
        d = json.load(open(f"{R}/quantized/{label}/tiny10.json"))
        return (round(d["mean_ndcg@10"], 2), d["n_tasks"]) if d["mean_ndcg@10"] is not None else (None, 0)
    except Exception:
        return (None, 0)

def dims(label):
    try:
        d = json.load(open(f"{S}/sparsity/{label}.json"))[label]
        v = [t["mean"] for t in d.values() if t.get("mean") is not None]
        return round(sum(v) / len(v), 1) if v else None
    except Exception:
        return None

def probe_a_label(step):
    for pat in (f"mlm-probe-a-s{step:05d}", f"mlm-probe-a-s{step}", f"*probe-a*s{step:05d}*", f"*probe-a*-{step}"):
        hits = sorted(glob.glob(f"{R}/quantized/{pat}"))
        if hits:
            return os.path.basename(hits[0])
    return None

points = []
for st in STEPS:
    m = f"{name}-s{st:05d}"
    a = probe_a_label(st)
    points.append({"step": st, "examples": st * 32, "label": m, "tiny10": tiny(m)[0], "n_tasks": tiny(m)[1], "dims": dims(m),
                   "probe_a_label": a, "probe_a_tiny10": tiny(a)[0] if a else None, "probe_a_dims": dims(a) if a else None})
m5 = next(p for p in points if p["step"] == 5000)
verdict = "no s05000 measurement"
if m5["tiny10"] is not None and m5["probe_a_tiny10"] is not None:
    adv = round(m5["tiny10"] - m5["probe_a_tiny10"], 2)
    band = m5["dims"] is not None and 300 <= m5["dims"] <= 600
    if adv >= 2.0 and m5["tiny10"] >= 57.0 and band: verdict = f"PASS (M − A = {adv:+.2f}, M = {m5['tiny10']}, dims {m5['dims']})"
    elif 0.8 <= adv < 2.0: verdict = f"MAYBE (M − A = {adv:+.2f}, M = {m5['tiny10']}, dims {m5['dims']})"
    else: verdict = f"FAIL (M − A = {adv:+.2f}, M = {m5['tiny10']}, dims {m5['dims']}, band {'ok' if band else 'violated'})"
elif m5["tiny10"] is not None:
    verdict = f"no probe-A s05000 result found; M@s05000 = {m5['tiny10']} (dims {m5['dims']})"
json.dump({"model": name, "head": head, "config": cfg, "examples_per_step": 32, "points": points, "gate_GM": verdict,
           "written": time.strftime("%Y-%m-%d %H:%M")}, open(curve_path, "w"), indent=2)
lines = [f"# Probe M ({time.strftime('%d.%m.%Y %H:%M')}): SPLADE from pl-ModernBERT-512-base vs probe A (distilroberta) at equal steps",
         "", f"Training head: `{head}` (config `{cfg}`; SPEEDUP-SPLADE-3060.md §6). Leaderboard regime (quantized, 512 tokens), tiny10 nDCG@10; dims = active document dimensions (base 436).",
         "", "| step | examples | M tiny10 | M dims | A tiny10 | A dims | M − A |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for p in points:
    d = (p["tiny10"] - p["probe_a_tiny10"]) if p["tiny10"] is not None and p["probe_a_tiny10"] is not None else None
    f = lambda v: "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines.append(f"| {p['step']} | {p['examples']} | {f(p['tiny10'])} | {f(p['dims'])} | {f(p['probe_a_tiny10'])} | {f(p['probe_a_dims'])} | {('%+.2f' % d) if d is not None else '–'} |")
lines += ["", f"**Gate GM (s05000):** {verdict}", "",
          "GM: PASS when M − A ≥ +2.0 and M ≥ 57.0 and dims in 300–600; MAYBE when the advantage is in [+0.8, +2.0); FAIL otherwise.",
          "Next: PASS → ModernBERT stage 1 on fresh data (NEXT-SESSION-5070-MLM.md, night 3); MAYBE → one day of stage 1 with a hard ≥ 59.5 threshold; FAIL → branch A or B by G1."]
open(summary_path, "w").write("\n".join(lines) + "\n")
print(f"curve -> {curve_path}; gate GM: {verdict}")
PYEOF
log "summary written to $SUM: $(grep -o 'Gate GM.*' "$SUM" | head -1)"

# ---- 6. commit and push results ---------------------------------------------------------------------------------
for p in "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$RESULTS_ROOT/data" "$SUM" "$EXP_ROOT/queues/queue_5070_mlm2.sh"; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_5070_mlm2.out"
done
if git commit -q -m "Add the probe M results: SPLADE from pl-ModernBERT against probe A at equal steps" 2>> "$LOGS/queue_5070_mlm2.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_mlm2.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_mlm2.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_mlm2.log)"
else log "nothing to commit or commit failed (see logs/queue_5070_mlm2.out)"; fi
log "=== mlm2 done ($(date '+%d.%m %H:%M')) ==="
