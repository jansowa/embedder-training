#!/usr/bin/env bash
# v36 (5.11): the best candidates measured once in the regime that reproduces the PIRB leaderboard (old-pyserini:
# our fork + pyserini 0.21.0, float query weights; docs/benchmark-vs-leaderboard.md §1). Everything since 26.10 was
# measured in the cheaper `quantized` regime, validated against old-pyserini only on the base and only on tiny10.
#   0  sanity: tiny10 of the base again in old-pyserini (expected 61.59 +-0.15; the script has not run since the
#      1.11 repository move) -> abort on a mismatch
#   1  for each model in MODELS (default soup-v9b soup-v11-nq): tiny10, heldout8, the five held-out-2 tasks
#      (batch 8 for poquad and trec-covid, as on 27.10), ~3 h per model
#   2  results/summary_v36_leaderboard.md (scripts/leaderboard_compare.py: published polish-splade, our base,
#      the candidates, deltas, best standalone model per task), commit, push
#   setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v36.sh > logs/queue_v36.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v36 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v36.pid"
guard_alive
MODELS=${MODELS:-soup-v9b soup-v11-nq}
BASE_REF=61.59; BASE_TOL=0.15
REG=old-pyserini

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v34s v35 s1}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
lbv() { regime_mean "$REGIMES_ROOT/$REG/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
h2mean() { "$PY" - "$REGIMES_ROOT/$REG/$1" <<'EOF'
import json, sys
v = []
for t in ["czywiesz", "gpt35wiki", "poquad", "quora", "treccovid"]:
    try: v.append(json.load(open(f"{sys.argv[1]}/heldout2_{t}.json"))["mean_ndcg@10"])
    except Exception: pass
print(f"{sum(v)/len(v):.2f} over {len(v)} tasks" if v else "missing")
EOF
}
[ -x "$HOME/work/venv-pyserini021/bin/python" ] || { log "venv-pyserini021 missing; cannot run the $REG regime"; exit 1; }
for m in $MODELS; do [ -d "models/$m" ] || { log "models/$m missing"; exit 1; }; done

log "=== v36 start: $REG regime for $MODELS; $(free_gb) GB free on / ==="

# 0. sanity on the base
log "=== v36 step 0: base tiny10 in $REG (expected $BASE_REF) ==="
rm -rf "$REGIMES_ROOT/$REG/base-splade-wrapped-recheck"
regime "$REG" base-splade-wrapped-recheck models/base-splade-wrapped tiny10 16
B=$(lbv base-splade-wrapped-recheck tiny10)
if [ -z "$B" ] || [ "$(awk -v a="$B" -v r="$BASE_REF" -v t="$BASE_TOL" 'BEGIN {d = a - r; if (d < 0) d = -d; print (d <= t) ? 1 : 0}')" != 1 ]; then
    log "base recheck ${B:-FAILED} differs from $BASE_REF by more than $BASE_TOL; the $REG regime is not reproducing the leaderboard; stopping"
    record "sanity:$REG:base" FAILED "${B:-?}" "expected $BASE_REF"
    exit 2
fi
log "base recheck ok: $B (reference $BASE_REF)"
record "sanity:$REG:base" ok "$B" "reference $BASE_REF"

# 1. the candidates
for m in $MODELS; do
    log "=== v36: $m in $REG ==="
    regime "$REG" "$m" "models/$m" tiny10 16
    regime "$REG" "$m" "models/$m" heldout8 16
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
        bs=16; case "$ts" in heldout2_poquad|heldout2_treccovid) bs=8;; esac
        regime "$REG" "$m" "models/$m" "$ts" "$bs"
    done
    log "  $m ($REG): tiny10 $(lbv "$m" tiny10) / heldout8 $(lbv "$m" heldout8) / held-out-2 $(h2mean "$m") (quantized: $(regime_mean "$REGIMES_ROOT/quantized/$m/tiny10.json" | awk '{print $1}') / $(regime_mean "$REGIMES_ROOT/quantized/$m/heldout8.json" | awk '{print $1}'))"
done

# 2. summary against the leaderboard
SUM="$RESULTS_ROOT/summary_v36_leaderboard.md"
MARGS=""; for m in $MODELS; do MARGS="$MARGS --model $m"; done
{
    echo "# v36 ($(date '+%d.%m.%Y %H:%M'), old-pyserini regime, nDCG@10): the best candidates against the PIRB leaderboard"
    echo
    echo "old-pyserini = our PIRB fork + pyserini 0.21.0 (float query weights), the regime of the published polish-splade entry (docs/benchmark-vs-leaderboard.md §1). Base sanity re-run: tiny10 $B (reference $BASE_REF). The same models in the quantized regime: results/summary_v35.md."
    echo
    "$PY" scripts/leaderboard_compare.py --regime "$REG" --model base-splade-wrapped $MARGS --taskset tiny10 --taskset heldout8 --taskset heldout2 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Queue state (v36)"; echo; echo '```'
    grep -E "^(sanity|regime):$REG:" "$STATE" | tail -40
    echo '```'
} > "$SUM"
log "summary written to $SUM"
for m in $MODELS; do git add "$REGIMES_ROOT/$REG/$m" 2>/dev/null; done
git add "$SUM" 2>/dev/null
git commit -q -m "Add the v36 results: the best soups in the leaderboard regime" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master > "$LOGS/push_v36.log" 2>&1 && { echo pushed >> "$LOGS/push_v36.log"; break; }; sleep 600; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v36 done ($(date '+%d.%m %H:%M')) ==="
