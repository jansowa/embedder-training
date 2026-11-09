#!/usr/bin/env bash
# Queue v44 (3060 Ti, night 8/9.11, ~6 h, after s8): held-out-2 before the publication decision.
# v42: r2v12-bias005 (the candidate) has held-out-2 68.80 vs soup-v12-bias005 69.62 and soup-v9b 69.42; four tasks are
# level or better, trec-covid is 66.02 vs 70.24 / 69.66. Which step lost trec-covid: the bias, round 2, or the n5kj recipe?
#   1  trec-covid alone (~22 min each): r2v12 (no bias), n5kj, ens-n5v2 (n5 rows only, no colleague rows), soup-v12
#   2  full held-out-2 (~80 min each): r2v12, n5kj, r2v12-bias010 (the cheaper-index option)
#   3  soup-v14s full held-out-2 (best tiny10 of the seed soups) if time allows
#   4  summary_v44.md (per-task held-out-2 of every model that has it), commit, push
# Reading: r2v12 trec-covid ~66 = round 2 / n5kj recipe, not the bias; n5kj ~66 = the colleague rows or n5; ens-n5v2 ~70 = the
# colleague rows (legal / FAQ data pulling away from biomedical text).
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v44 MIN_RAM_GB=8
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v44.pid"
guard_alive
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-}; do
    if queue_alive "$prev"; then log "waiting for queue $prev to exit"; while queue_alive "$prev"; do sleep 60; done; sleep 30; fi
done
log "=== v44 start: held-out-2 diagnosis of trec-covid; $(free_gb) GB free on / ==="

h2task() { local bs=16; case "$2" in heldout2_poquad|heldout2_treccovid) bs=8;; esac; [ -d "models/$1" ] && regime quantized "$1" "models/$1" "$2" "$bs"; }
heldout2() { for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do h2task "$1" "$ts"; done; }

N5KJ=arm-dense-kl-n5kj-ens400
log "=== v44 step 1: trec-covid ==="
for m in $N5KJ-r2v12 $N5KJ arm-dense-kl-n5v2-ens400 soup-v12; do
    h2task "$m" heldout2_treccovid; log "  $m trec-covid $(lbv "$m" heldout2_treccovid)"
done
log "=== v44 step 2: full held-out-2 ==="
for m in $N5KJ-r2v12 $N5KJ $N5KJ-r2v12-bias010 soup-v14s; do heldout2 "$m"; done

log "=== v44 step 4: summary ==="
SUM="$RESULTS_ROOT/summary_v44.md"
"$PY" - "$REGIMES_ROOT/quantized" <<'EOF' > "$SUM"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
tasks = ["czywiesz", "gpt35wiki", "quora", "poquad", "treccovid"]
models = ["soup-v9b", "soup-v12", "soup-v12-bias005", "arm-dense-kl-n5v2-ens400", "arm-dense-kl-n5kj-ens400", "soup-v14s",
          "arm-dense-kl-n5kj-ens400-r2v12", "arm-dense-kl-n5kj-ens400-r2v12-bias005", "arm-dense-kl-n5kj-ens400-r2v12-bias010"]
def get(m, t):
    f = root / m / f"heldout2_{t}.json"
    try:
        return json.load(open(f))["mean_ndcg@10"]
    except Exception:
        return None
def tm(m, ts):
    f = root / m / f"{ts}.json"
    try:
        return f"{json.load(open(f))['mean_ndcg@10']:.2f}"
    except Exception:
        return "-"
print("# v44 (night 8/9.11, quantized regime, nDCG@10): held-out-2 per task before the publication decision\n")
print("soup-v9b in old-pyserini: czywiesz 86.88, gpt35wiki 91.97, quora 79.32, poquad 19.25, treccovid 69.66, mean 69.42 (quantized = old-pyserini within 0.3 except trec-covid, -0.83).\n")
print("| model | tiny10 | heldout8 | " + " | ".join(tasks) + " | mean (5/5) |")
print("|---|---:|---:|" + "---:|" * (len(tasks) + 1))
for m in models:
    v = [get(m, t) for t in tasks]
    if all(x is None for x in v):
        continue
    mean = f"{sum(v) / 5:.2f}" if all(x is not None for x in v) else "-"
    print(f"| {m} | {tm(m, 'tiny10')} | {tm(m, 'heldout8')} | " + " | ".join("-" if x is None else f"{x:.2f}" for x in v) + f" | {mean} |")
EOF
log "summary written to $SUM"
git add "$SUM" "$REGIMES_ROOT"/quantized/*/heldout2_* "$EXP_ROOT/queues/queue_v44.sh" 2>/dev/null
git commit -q -m "Add the v44 results: held-out-2 per task for the publication candidates" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v44.log" 2>&1 && { echo pushed >> "$LOGS/push_v44.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v44 done ($(date '+%d.%m %H:%M')) ==="
