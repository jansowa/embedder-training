#!/usr/bin/env bash
# Queue v55 (3060 Ti, night 11/12.11 after v52 and v53, ~1.2 h): the 23-task number of the student-mined arm.
# The mined arm (arm-dense-kl-n5kj-mined-ens400-r2v12) has tiny10, heldout8 and trec-covid in several bias forms, but no
# queue measured the other four held-out-2 tasks, so it has no 23-task mean next to r2v12 (57.94), soup-v12-bias005
# (57.88), soup-v14s (58.06) and polish-splade (55.73).
#   1  PICK the publication form among the measured bias forms of the arm (0.10 ... 0.15 from v50, v52p, v52q): the
#      sparsest one whose tiny10 and heldout8 are both within 0.1 of bias010 (unrounded values). The user's choice wins:
#      V55_MODEL=<label> in the environment, or a label in the first line of logs/v55_model.txt (read when v55 starts).
#   2  held-out-2 of that form: czywiesz, gpt35wiki, quora, poquad (trec-covid only when missing), quantized regime as
#      in queue_v44.sh.
#   3  index cost on the cost sample (dims/doc, dims/query, postings/query) of the form next to the reference models.
#   4  summary_v55.md, commit, push loop.
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v55.sh > logs/queue_v55.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v55 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v55.pid"
guard_alive
queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v52 v53}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done

lbv()  { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs()  { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
h2task() { local bs=16; case "$2" in heldout2_poquad|heldout2_treccovid) bs=8;; esac; [ -d "models/$1" ] && regime quantized "$1" "models/$1" "$2" "$bs"; }
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
R2=arm-dense-kl-n5kj-ens400-r2v12; M=arm-dense-kl-n5kj-mined-ens400-r2v12
log "=== v55 start: the 23-task number of the mined arm; $(free_gb) GB free on / ==="

# 1. the publication form
log "=== v55 step 1: the publication form of $M ==="
CHOICE="$RESULTS_ROOT/v55_choice.json"
PICK=${V55_MODEL:-}; WHO="the user's choice (V55_MODEL)"
if [ -z "$PICK" ] && [ -s "$LOGS/v55_model.txt" ]; then PICK=$(head -n 1 "$LOGS/v55_model.txt" | tr -d '[:space:]'); WHO="the user's choice (logs/v55_model.txt)"; fi
AUTO=$("$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$M" "$CHOICE" <<'EOF' 2>> "$LOGS/v55_choice.err"
import json, sys
from pathlib import Path
reg, spa, m, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
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
ref = f"{m}-bias010"
rt, rh = val(ref, "tiny10", 10), val(ref, "heldout8", 8)
rt = 64.63 if rt is None else rt; rh = 43.45 if rh is None else rh
forms = []
for t in ("010", "011", "012", "013", "014", "015", "020"):
    label = f"{m}-bias{t}"; a, b, d = val(label, "tiny10", 10), val(label, "heldout8", 8), dims(label)
    if a is None or b is None or d is None:
        continue
    forms.append({"label": label, "tiny10": a, "heldout8": b, "dims_per_doc": d, "within_0.1": a >= rt - 0.1 and b >= rh - 0.1})
ok = [f for f in forms if f["within_0.1"]]
pick = min(ok, key=lambda f: f["dims_per_doc"])["label"] if ok else ref
json.dump({"rule": "sparsest measured bias form with tiny10 and heldout8 both within 0.1 of bias010 (unrounded)", "reference": ref,
           "reference_tiny10": rt, "reference_heldout8": rh, "forms": forms, "auto_pick": pick}, open(out, "w"), indent=2)
print(pick)
EOF
)
"$PY" -c "import json,sys;[print('%s: %.2f / %.2f at %.1f dims/doc, %s' % (f['label'], f['tiny10'], f['heldout8'], f['dims_per_doc'], 'within 0.1 of bias010' if f['within_0.1'] else 'outside 0.1')) for f in json.load(open(sys.argv[1]))['forms']]" "$CHOICE" 2>/dev/null | while read -r l; do log "  $l"; done
if [ -z "$PICK" ]; then PICK=${AUTO:-$M-bias010}; WHO="the rule (sparsest form within 0.1 of bias010 on both sets)"; fi
if [ ! -d "models/$PICK" ]; then
    case "$PICK" in
        "$M"-bias0[0-9][0-9]) t=${PICK##*-bias}; biased "models/$M" "$PICK" "0.${t#0}" ;;
    esac
fi
if [ ! -d "models/$PICK" ]; then
    log "  models/$PICK missing and not a bias form of $M: falling back to ${AUTO:-$M-bias010}"
    PICK=${AUTO:-$M-bias010}; WHO="the rule (the requested model was missing)"
    t=${PICK##*-bias}; biased "models/$M" "$PICK" "0.${t#0}"
fi
[ -d "models/$PICK" ] || { log "  models/$PICK missing: v55 stops"; exit 1; }
log "  publication form: $PICK, by $WHO"

# 2. the three sets of that form (tiny10 and heldout8 are present for every form the rule can pick)
log "=== v55 step 2: held-out-2 of $PICK ==="
sparsity "$PICK" "models/$PICK"
regime quantized "$PICK" "models/$PICK" tiny10 16; regime quantized "$PICK" "models/$PICK" heldout8 16
for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
    h2task "$PICK" "$ts"; log "  $PICK $ts $(lbv "$PICK" "$ts")"
done

# 3. index cost on the cost sample
log "=== v55 step 3: index cost ==="
COST="$RESULTS_ROOT/index_cost_v55.json"
if [ ! -s "$COST" ]; then
    wait_gpu_free; wait_ram; s=$SECONDS
    args=(--model base=models/base-splade-wrapped)
    for m in soup-v12-bias005 soup-v14s $R2 $M-bias010; do [ -d "models/$m" ] && [ "$m" != "$PICK" ] && args+=(--model "$m=models/$m"); done
    args+=(--model "$PICK=models/$PICK")
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${args[@]}" --output-json "$COST" > "$LOGS/index_cost_v55.log" 2>&1 \
        && log "index cost written ($COST) in $((SECONDS - s))s" || { log "index cost FAILED (logs/index_cost_v55.log)"; rm -f "$COST"; }
fi

# 4. summary
log "=== v55 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_v55.md"
"$PY" - "$REGIMES_ROOT/quantized" "$RESULTS_ROOT/sparsity" "$COST" "$PICK" "$WHO" "$M" "$R2" "$CHOICE" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
reg, spa, cost_path, pick, who, m, r2, choice = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
H2 = ["czywiesz", "gpt35wiki", "quora", "poquad", "treccovid"]
def val(label, ts):
    try:
        return json.load(open(reg / label / f"{ts}.json"))["mean_ndcg@10"]
    except Exception:
        return None
def ntasks(label, ts):
    try:
        return json.load(open(reg / label / f"{ts}.json"))["n_tasks"]
    except Exception:
        return 0
def fmt(x, nd=2): return "-" if x is None else f"{x:.{nd}f}"
def dims(label):
    try:
        d = json.load(open(spa / f"{label}.json"))[label]; v = [t["mean"] for t in d.values() if t.get("mean") is not None]; return f"{sum(v) / len(v):.0f}"
    except Exception:
        return "-"
try:
    cost = json.load(open(cost_path))["models"]
except Exception:
    cost = {}
def c(label, key, nd=0):
    s = cost.get("base" if label == "base-splade-wrapped" else label)
    return fmt(s[key], nd) if s else "-"
print("# v55 (night 11/12.11, quantized regime, nDCG@10): held-out-2 and the 23-task mean of the student-mined arm\n")
print(f"Publication form: **{pick}**, chosen by {who}. 23-task mean = (10 x tiny10 + 8 x heldout8 + 5 x held-out-2) / 23. "
      "dims/doc = the sparsity sample (300 passages of each tiny task); the cost columns = the cost sample (1 350 passages + 1 350 queries, scripts/splade_index_cost.py).\n")
print("| model | tiny10 | heldout8 | " + " | ".join(H2) + " | held-out-2 | 23 tasks | dims/doc | cost: dims/doc | cost: dims/query | cost: postings/query |")
print("|---|---:|---:|" + "---:|" * (len(H2) + 6))
print(f"| polish-splade (base-splade-wrapped), old-pyserini regime* | 61.59 | 41.03 | - | - | - | - | - | 67.53 | 55.73 | 436 | {c('base-splade-wrapped', 'doc_active_mean', 1)} | {c('base-splade-wrapped', 'query_active_mean', 1)} | {c('base-splade-wrapped', 'postings_per_query')} |")
models = ["soup-v12-bias005", "soup-v14s", r2, f"{m}-bias010"]
models = [x for x in models if x != pick] + [pick]
for label in models:
    t, h = val(label, "tiny10"), val(label, "heldout8"); h2 = [val(label, f"heldout2_{x}") for x in H2]
    full = all(x is not None for x in h2)
    hs = fmt(h) + ("" if ntasks(label, "heldout8") in (0, 8) else f" ({ntasks(label, 'heldout8')}/8)")
    mean = fmt(sum(h2) / 5) if full else "-"
    m23 = fmt((10 * t + 8 * h + sum(h2)) / 23) if full and t is not None and h is not None and ntasks(label, "heldout8") == 8 else "-"
    name = f"**{label}**" if label == pick else label
    print(f"| {name} | {fmt(t)} | {hs} | " + " | ".join(fmt(x) for x in h2) + f" | {mean} | {m23} | {dims(label)} | {c(label, 'doc_active_mean', 1)} | {c(label, 'query_active_mean', 1)} | {c(label, 'postings_per_query')} |")
print("\n\\* polish-splade has its three sets only in the old-pyserini regime (summary_v36_leaderboard.md); the other rows are quantized. Its quantized tiny10 is " + fmt(val("base-splade-wrapped", "tiny10")) + ".")
try:
    ch = json.load(open(choice))
    print("\n## Bias forms of the mined arm considered by the rule (reference bias010: %.2f / %.2f)\n" % (ch["reference_tiny10"], ch["reference_heldout8"]))
    print("| form | tiny10 | heldout8 | trec-covid | dims/doc | within 0.1 on both |\n|---|---:|---:|---:|---:|---|")
    for f in ch["forms"]:
        print(f"| {f['label']} | {f['tiny10']:.2f} | {f['heldout8']:.2f} | {fmt(val(f['label'], 'heldout2_treccovid'))} | {f['dims_per_doc']:.1f} | {'yes' if f['within_0.1'] else 'no'} |")
    print(f"\nThe rule's pick: {ch['auto_pick']}.")
except Exception:
    pass
EOF
{ echo; echo "## Queue state (v55)"; echo; echo '```'; grep -F "regime:quantized:$PICK:" "$STATE" | tail -n 8; echo '```'; } >> "$SUM"
log "summary written to $SUM"
M23=$("$PY" -c "import re,sys;t=open(sys.argv[1]).read();r=[l for l in t.splitlines() if l.startswith('| **')];print(r[0].split('|')[10].strip() if r else '')" "$SUM" 2>/dev/null)
log "  $PICK: $(lbs "$PICK"), 23 tasks ${M23:-?} (r2v12 57.94, soup-v12-bias005 57.88, soup-v14s 58.06, polish-splade 55.73), $(dims "$PICK") dims/doc"
git add "$SUM" "$CHOICE" "$EXP_ROOT/queues/queue_v55.sh" 2>/dev/null
for p in "$COST" "$REGIMES_ROOT/quantized/$PICK" "$RESULTS_ROOT/sparsity/$PICK.json"; do [ -e "$p" ] && git add "$p" 2>/dev/null; done
git commit -q -m "Add the v55 results: held-out-2 and the 23-task mean of the student-mined arm" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v55.log" 2>&1 && { echo pushed >> "$LOGS/push_v55.log"; break; }; sleep 1800; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v55 done ($(date '+%d.%m %H:%M')) ==="
