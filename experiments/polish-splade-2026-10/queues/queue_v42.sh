#!/usr/bin/env bash
# v42 (7/8.11 night, 3060 Ti, after v41, ~12 h): around the new best arm-dense-kl-n5kj-ens400 (n5 + colleague rows,
# tiny10 64.14 at 10:38; heldout8 pending when this was written). Conditional steps read v41's results from disk.
#   1  n5kj seed 43: the noise floor of 64.14 and a second member for a soup of seeds -> sparsity, tiny10, heldout8.
#   2  soups, only when n5kj's heldout8 >= 42.85 (ens-n5v2 42.96 - noise): soup-v14 = uniform(soup-v12, n5kj);
#      soup-v14s = uniform(n5kj seed 42, n5kj seed 43). Reading against soup-v12 63.96 / 42.92.
#   3  round 2: the same n5 + kj recipe with the student starting from soup-v12 (teacher files unchanged: soup-v9b +
#      PolDense). Reading: above n5kj = a better start adds on top of the teacher; at or below = the teacher is the ceiling
#      and the next gain needs a better teacher (soup-v12 member, 5070 pool).
#   3b n7 = n5 + kj + the n4 rows absent from them by query text (hotpotqa 2 073 + 2 069 squadv2-v2 rows of the n4 cut,
#      24 434 rows, gated 23.5): does source diversity at this dose add on top of n5 + kj.
#   4  the best model of the night by heldout8 (among n5kj, n5kj-s43, soup-v14, soup-v14s, r2v12, n7, soup-v12) gets the bias
#      form of the threshold (lm_head.bias -0.0513; v41: soup-v12-bias005 64.05 / 42.83 at 378 dims/doc, i.e. above soup-v9b
#      with a smaller index); the publication candidate = the bias form when it holds heldout8 within 0.15: the five
#      held-out-2 tasks (czywiesz, gpt35wiki, quora, poquad, treccovid -> the 23 leaderboard tasks), tiny10 + heldout8 at
#      index-cost profile (doc + query side, with the 0.05 threshold) for v9b, v12, v12-bias005, the best and its bias form;
#      held-out-2 also for soup-v12-bias005 (the v41 candidate).
#   5  the density regulator arm (n5kj recipe with regularizer_control, band 0.90-1.02 of soup-v9b), unconditional: can the
#      density be trained in (doc + query profile next to n5kj and soup-v9b) instead of cut at indexing (free in v41).
#   6  summary_v42.md, commit (no third_party paths), push.
#   WAIT_FOR="v41" setsid nohup bash experiments/polish-splade-2026-10/queues/queue_v42.sh > logs/queue_v42.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
export QUEUE_TAG=v42 MIN_RAM_GB=6
export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-10800}
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v42.pid"
guard_alive
C="$EXP_ROOT/configs/next/3060"

queue_alive() { local pf="$LOGS/queue_$1.pid"; [ -f "$pf" ] && ps -p "$(cat "$pf")" -o args= 2>/dev/null | grep -q "queue_$1.sh"; }
for prev in ${WAIT_FOR:-v41}; do
    if queue_alive "$prev"; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while queue_alive "$prev"; do sleep 60; done
        sleep 30
    fi
done
lbv() { regime_mean "$REGIMES_ROOT/quantized/$1/$2.json" 2>/dev/null | awk '{print $1}'; }
lbs() { echo "$(lbv "$1" tiny10) / $(lbv "$1" heldout8)"; }
dims() { sparsity_mean "$RESULTS_ROOT/sparsity/$1.json" "$1" 2>/dev/null; }
task() { "$PY" -c "import json;print(round(json.load(open('$REGIMES_ROOT/quantized/$1/$2.json'))['tasks'].get('$3', float('nan')), 2))" 2>/dev/null; }
gte() { "$PY" -c "import sys;a,b=sys.argv[1:3];print(int(bool(a) and bool(b) and float(a)>=float(b)))" "$1" "$2"; }
h2mean() { "$PY" - "$REGIMES_ROOT/quantized/$1" <<'EOF'
import json, sys
v = []
for t in ("czywiesz", "gpt35wiki", "quora", "poquad", "treccovid"):
    try: v.append(json.load(open(f"{sys.argv[1]}/heldout2_{t}.json"))["mean_ndcg@10"])
    except Exception: pass
print(f"{sum(v)/len(v):.2f} ({len(v)}/5)" if v else "-")
EOF
}
N5KJ=arm-dense-kl-n5kj-ens400
V12=models/soup-v12

measure() {  # measure LABEL MODEL REFLINE
    local label="$1" model="$2" ref="$3"
    [ -d "$model" ] || { log "$label: $model missing"; return 1; }
    sparsity "$label" "$model"
    regime quantized "$label" "$model" tiny10 16
    regime quantized "$label" "$model" heldout8 16
    log "  $label: $(lbs "$label") ($ref), $(dims "$label") dims/doc; legal-questions $(task "$label" heldout8 poleval-2022-test-A-legal-questions), arguana $(task "$label" tiny10 arguana-pl)"
}
soup() {  # soup NAME MODEL...: uniform soup, then measure
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
        wait_gpu_free
        timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/soup_$name.log" 2>&1 \
            && log "soup $name written ($*, uniform)" || { log "soup $name FAILED (logs/soup_$name.log)"; rm -rf "models/$name"; return 1; }
    fi
    measure "$name" "models/$name" "soup-v12 63.96 / 42.92, n5kj $(lbs $N5KJ)"
}

log "=== v42 start: n5kj seed 43, soups, round 2 from soup-v12, held-out-2 for the best, regulator; n5kj so far $(lbs $N5KJ); $(free_gb) GB free ==="

# 1. n5 + kj, seed 43
train dense-kl-n5kj-ens400-s43 "$C/distil_dense_kl_n5kj_ens400_s43.yaml"
measure "$N5KJ-s43" "models/$N5KJ-s43" "seed 42: $(lbs $N5KJ); ens-n5v2 63.95 / 42.96"

# 2. soups with n5kj, when its heldout8 holds
H=$(lbv $N5KJ heldout8)
if [ "$(gte "${H:-0}" 42.85)" = 1 ]; then
    soup soup-v14 "$V12" "models/$N5KJ"
    [ -d "models/$N5KJ-s43" ] && soup soup-v14s "models/$N5KJ" "models/$N5KJ-s43"
else
    log "n5kj heldout8 ${H:-?} < 42.85: soups with n5kj skipped"
fi

# 3. round 2 from soup-v12
train dense-kl-n5kj-ens400-r2v12 "$C/distil_dense_kl_n5kj_ens400_r2v12.yaml"
measure "$N5KJ-r2v12" "models/$N5KJ-r2v12" "from polish-splade: $(lbs $N5KJ); soup-v12 (the start) 63.96 / 42.92"

# 3b. n7 = n5 + kj + the n4 rows absent from them (hotpotqa 2 073 + 2 069 squadv2-v2 rows of the n4 cut, by query text)
N7=datasets/arms/n7-tens400.jsonl
if [ ! -s "$N7" ]; then
    "$PY" - datasets/arms/n5-plus-kj-tens400.jsonl datasets/arms/mix3v2-6219-tens400.jsonl "$N7" <<'EOF' | while read -r l; do log "  $l"; done
import json, sys
base, n4, out = sys.argv[1:4]
seen = set(); nb = na = 0
with open(out, "w", encoding="utf-8") as o:
    for line in open(base, encoding="utf-8"):
        seen.add(json.loads(line)["query"].strip()); o.write(line.rstrip("\n") + "\n"); nb += 1
    for line in open(n4, encoding="utf-8"):
        r = json.loads(line)
        if r["query"].strip() in seen: continue
        s = r.get("pos_scores_stronger_reranker") or r.get("pos_scores")
        if s and s[0] is not None and s[0] > 23.5:   # the 23.5 gate, as for the n5 rows
            seen.add(r["query"].strip()); o.write(line.rstrip("\n") + "\n"); na += 1
print(f"n7 file: {nb} n5 + kj rows + {na} extra n4 rows -> {out}")
EOF
fi
if [ -s "$N7" ]; then
    train dense-kl-n7-ens400 "$C/distil_dense_kl_n7_ens400.yaml"
    measure arm-dense-kl-n7-ens400 models/arm-dense-kl-n7-ens400 "n5kj $(lbs $N5KJ); ens-n5v2 63.95 / 42.96"
fi

# 4. the best two of the night by heldout8 -> held-out-2 (both), threshold and index cost (best)
BEST=""; BEST_H=""; SECOND=""; SECOND_H=""
for m in $N5KJ $N5KJ-s43 soup-v14 soup-v14s $N5KJ-r2v12 arm-dense-kl-n7-ens400 soup-v12; do
    h=$(lbv "$m" heldout8); [ -n "$h" ] || continue
    if [ -z "$BEST_H" ] || [ "$(gte "$h" "$BEST_H")" = 1 ]; then SECOND="$BEST"; SECOND_H="$BEST_H"; BEST="$m"; BEST_H="$h"
    elif [ -z "$SECOND_H" ] || [ "$(gte "$h" "$SECOND_H")" = 1 ]; then SECOND="$m"; SECOND_H="$h"; fi
done
log "best of the night by heldout8: ${BEST:-none} (${BEST_H:-?}); second: ${SECOND:-none} (${SECOND_H:-?})"
heldout2() {  # heldout2 MODEL: the five held-out-2 tasks
    for ts in heldout2_czywiesz heldout2_gpt35wiki heldout2_quora heldout2_poquad heldout2_treccovid; do
        bs=16; case "$ts" in heldout2_poquad|heldout2_treccovid) bs=8;; esac
        regime quantized "$1" "models/$1" "$ts" "$bs"
    done
    log "  $1: held-out-2 $(h2mean "$1") (soup-v9b 69.42 old-pyserini: czywiesz 86.88, gpt35wiki 91.97, quora 79.32, poquad 19.25, treccovid 69.66)"
}
bias005() {  # bias005 MODEL: models/MODEL-bias005 = the checkpoint with lm_head.bias shifted by -(e^0.05 - 1) (v41: soup-v12 64.05 / 42.83 at 378 dims/doc)
    [ -d "models/$1-bias005" ] && return 0
    [ -d "models/$1" ] || return 1
    cp -r "models/$1" "models/$1-bias005" && "$PY" - "models/$1-bias005" <<'EOF' && log "$1-bias005 written (lm_head.bias = -0.0513)" || { log "$1-bias005 FAILED"; rm -rf "models/$1-bias005"; return 1; }
import math, sys
from safetensors.torch import load_file, save_file
d = sys.argv[1] + "/model.safetensors"
t = load_file(d); b = math.exp(0.05) - 1.0
for k in ("lm_head.bias", "lm_head.decoder.bias"):
    if k in t: t[k] = t[k] - b; print(k, "shifted by", -b)
save_file(t, d, metadata={"format": "pt"})
EOF
}
PUB=""
if [ -n "$BEST" ] && [ -d "models/$BEST" ]; then
    # the publication candidate = the best model with the bias form of the threshold, when it holds heldout8 within 0.15
    bias005 "$BEST" && measure "$BEST-bias005" "models/$BEST-bias005" "without bias $(lbs "$BEST"); soup-v12-bias005 64.05 / 42.83 at 378 dims/doc"
    HB=$(lbv "$BEST-bias005" heldout8)
    if [ -n "$HB" ] && [ "$(gte "$HB" "$("$PY" -c "print(float('$BEST_H') - 0.15)")")" = 1 ]; then PUB="$BEST-bias005"; else PUB="$BEST"; fi
    log "publication candidate: $PUB ($(lbs "$PUB"))"
    heldout2 "$PUB"
    COST="$RESULTS_ROOT/index_cost_v42.json"
    if [ ! -s "$COST" ]; then
        wait_gpu_free; s=$SECONDS
        MODELS="--model soup-v9b=models/soup-v9b --model soup-v12=$V12 --model soup-v12-bias005=models/soup-v12-bias005 --model $BEST=models/$BEST"
        [ -d "models/$BEST-bias005" ] && MODELS+=" --model $BEST-bias005=models/$BEST-bias005"
        # shellcheck disable=SC2086
        timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 $MODELS \
            --min-weight 0.05 --output-json "$COST" > "$LOGS/index_cost_v42.log" 2>&1 \
            && log "index-cost profile (doc + query side) written ($COST) in $((SECONDS - s))s" || log "index-cost profile FAILED"
        grep -E "doc_active" "$LOGS/index_cost_v42.log" | while read -r l; do log "  $l"; done
    fi
fi
# the second candidate always gets its 23 tasks: soup-v12-bias005 (the v41 candidate), unless it is already PUB
[ -d models/soup-v12-bias005 ] && [ "$PUB" != soup-v12-bias005 ] && heldout2 soup-v12-bias005

# 5. the density regulator (unconditional, the user's call on 7.11 13:50: the threshold at indexing is free on
#    heldout8 (v41: soup-v12 42.93 vs 42.92), the question here is whether the density can be trained in instead)
log "threshold 0.05 on soup-v12 (v41): heldout8 $(lbv soup-v12-mw005 heldout8) vs $(lbv soup-v12 heldout8); the regulator arm runs regardless"
train dense-kl-n5kj-ens400-reg "$C/distil_dense_kl_n5kj_ens400_reg.yaml"
measure "$N5KJ-reg" "models/$N5KJ-reg" "n5kj $(lbs $N5KJ) at $(dims $N5KJ) dims/doc; soup-v9b 409"
if [ -d "models/$N5KJ-reg" ]; then
    COSTR="$RESULTS_ROOT/index_cost_v42_reg.json"
    wait_gpu_free; s=$SECONDS
    timeout -k 60 3600 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 \
        --model soup-v9b=models/soup-v9b --model n5kj="models/$N5KJ" --model n5kj-reg="models/$N5KJ-reg" --min-weight 0.05 \
        --output-json "$COSTR" > "$LOGS/index_cost_v42_reg.log" 2>&1 \
        && log "index-cost profile of the regulator arm written ($COSTR) in $((SECONDS - s))s" || log "index-cost profile (reg) FAILED"
    grep -E "doc_active" "$LOGS/index_cost_v42_reg.log" | while read -r l; do log "  $l"; done
fi

# 6. summary
SUM="$RESULTS_ROOT/summary_v42.md"
{
    echo "# v42 ($(date '+%d.%m.%Y %H:%M'), leaderboard regime, nDCG@10): around n5 + kj — seed, soups, round 2 from soup-v12, held-out-2 for the best, regulator"
    echo
    echo "n5kj = the 23.5-gated n5 rows (fiqa 2 073 + 16 000 squadv2-v2) plus all 2 219 colleague rows, KL from the ensemble teacher (soup-v9b + PolDense-400M), the ens-n5v2 recipe, seeds 42 and 43. soup-v14 = uniform(soup-v12, n5kj); soup-v14s = uniform(n5kj s42, n5kj s43). r2v12 = the same recipe with the student starting from soup-v12. reg = regularizer_control keeping dims/doc inside 0.90-1.02 of soup-v9b. Best of the night: ${BEST:-none}; publication candidate: ${PUB:-none} (bias005 = lm_head.bias shifted by -0.0513, the threshold inside the checkpoint); held-out-2 = the five tasks that complete the 23 leaderboard tasks."
    echo
    "$PY" scripts/regime_task_table.py --models base-splade-wrapped arm-dense-kl-n5v2-ens400 $N5KJ $N5KJ-s43 soup-v14 soup-v14s $N5KJ-r2v12 arm-dense-kl-n7-ens400 $N5KJ-reg soup-v12 ${BEST:+$BEST-bias005} soup-v12-bias005 soup-v9b --tasksets tiny10 heldout8 2>> "$LOGS/queue_summary.log"
    echo
    echo "## Held-out-2 (quantized; soup-v9b in old-pyserini: czywiesz 86.88, gpt35wiki 91.97, quora 79.32, poquad 19.25, treccovid 69.66, mean 69.42)"; echo
    for m in $PUB $BEST soup-v12-bias005; do
        [ -n "$m" ] && [ -f "$REGIMES_ROOT/quantized/$m/heldout2_czywiesz.json" ] || continue
        echo "- $m: czywiesz $(lbv "$m" heldout2_czywiesz), gpt35wiki $(lbv "$m" heldout2_gpt35wiki), quora $(lbv "$m" heldout2_quora), poquad $(lbv "$m" heldout2_poquad), treccovid $(lbv "$m" heldout2_treccovid); mean $(h2mean "$m")"
    done | awk '!seen[$0]++'
    echo
    "$PY" scripts/sparsity_table.py --model base=base --model $N5KJ --model $N5KJ-s43 --model soup-v14 --model soup-v14s --model $N5KJ-r2v12 --model arm-dense-kl-n7-ens400 --model $N5KJ-reg --model soup-v12 --model soup-v12-bias005 ${BEST:+--model $BEST-bias005} 2>> "$LOGS/queue_summary.log"
    echo; echo "## Queue state (v42)"; echo; echo '```'
    grep -E "^(train|sparsity|regime:quantized):(.*(n5kj|soup-v14|n7-ens400|heldout2).*)" "$STATE"
    echo '```'
} > "$SUM"
log "summary: $SUM"
git add "$SUM" "$RESULTS_ROOT"/index_cost_v42.json "$RESULTS_ROOT"/index_cost_v42_reg.json "$RESULTS_ROOT"/sparsity/$N5KJ-s43.json "$RESULTS_ROOT"/sparsity/soup-v14.json "$RESULTS_ROOT"/sparsity/soup-v14s.json "$RESULTS_ROOT"/sparsity/$N5KJ-r2v12.json "$RESULTS_ROOT"/sparsity/$N5KJ-reg.json \
    "$REGIMES_ROOT"/quantized/$N5KJ "$REGIMES_ROOT"/quantized/$N5KJ-s43 "$REGIMES_ROOT"/quantized/soup-v14 "$REGIMES_ROOT"/quantized/soup-v14s "$REGIMES_ROOT"/quantized/$N5KJ-r2v12 "$REGIMES_ROOT"/quantized/$N5KJ-reg ${BEST:+"$REGIMES_ROOT/quantized/$BEST-bias005"} "$REGIMES_ROOT"/quantized/soup-v12-bias005 ${BEST:+"$RESULTS_ROOT/sparsity/$BEST-bias005.json"} \
    "$RESULTS_ROOT"/sparsity/arm-dense-kl-n7-ens400.json "$REGIMES_ROOT"/quantized/arm-dense-kl-n7-ens400 \
    "$C/distil_dense_kl_n5kj_ens400_s43.yaml" "$C/distil_dense_kl_n5kj_ens400_r2v12.yaml" "$C/distil_dense_kl_n5kj_ens400_reg.yaml" "$C/distil_dense_kl_n7_ens400.yaml" "$EXP_ROOT/queues/queue_v42.sh" 2>/dev/null
git commit -q -m "Add the v42 results: n5 plus the colleague rows with a second seed, soups, round 2 from soup-v12 and the regulator" 2>/dev/null && log "committed $(git log --oneline -1)"
(for i in 1 2 3; do timeout -k 10 180 git push origin HEAD > "$LOGS/push_v42.log" 2>&1 && { echo pushed >> "$LOGS/push_v42.log"; break; }; sleep 300; done) > /dev/null 2>&1 < /dev/null &
disown
log "=== v42 done ($(date '+%d.%m %H:%M')) ==="
