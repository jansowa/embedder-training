#!/usr/bin/env bash
# 31.10 -> 1.11 on the 3060 Ti, unattended: can the new speakleash sets (fever, hotpotqa_pl,
# nq-pl-mined-negatives) give a component that lifts soup-v9b (63.65 / 42.56 / 69.42)?
# Data: scripts/build_speakleash_cuts.py (gate 23.5 on the gemma score of the English pair, the
# same gate that made fiqa-2073 / squadv2-2073 / msmarco-clean-2073; 2073 rows per set; positives
# <= 512 tokens, at most 2 negatives > 512, negatives longest first). Configs: build_v25_configs.py.
#   0  diagnostics (GPU, ~65 min): student margins (tail) and roberta-v3 Polish scores on the
#      8000-row pools and on the reference files -> results/data/speakleash-diag.json.
#   1  single-set screen, order nq, hotpotqa, fever: <set>-2073 alone at 2 x 16; tiny + sparsity.
#      Toxicity threshold: truncated tiny >= SINGLE_MIN (base 60.14; squadv2-2073 60.85,
#      msmarco-clean-2073 60.49 were useful additives, wikisynth 59.67 and trivia 55.15 were not).
#   2  n4 pattern for every set that passed: fiqa-2073 + squadv2-2073 + <set>-2073 (arm-n4x-<set>);
#      tiny, sparsity, held-out-8.
#   3  control arm-n4-3060: n4's own rows (mix3-6219) at 2 x 16 here; tiny, sparsity, held-out-8.
#   4  leaderboard regime (tiny10 + heldout8) for every n4x arm with truncated tiny >= control - 0.3,
#      then for the control itself.
#   5  soups: soup-v10-<set> = the five v9b ingredients + arm-n4x-<set>, uniform; sparsity,
#      leaderboard regime (v9b 63.65 / 42.56).
#   6  held-out-2 (5 tasks) for the best new soup when its leaderboard heldout8 >= v9b.
#   7  filler when time remains and INCLUDE_PROBES=1: the GradCache speed probes of queue v24.
#   8  results/summary_v25.md (per-task tables, diagnostics, queue state); commit; push.
# A 2 x 16 training that fails (CUDA OOM) is retried once with its _cb32 config. Every step is
# skipped when it already has a result. No step starts after DEADLINE.
# Start: setsid nohup bash scripts/queue_v25.sh > logs/queue_v25.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export QUEUE_TAG=v25 MIN_RAM_GB=6
export WANDB_MODE=offline
source scripts/queue_lib.sh
echo $$ > "$LOGS/queue_v25.pid"
guard_alive
DEADLINE=$(date -d "${V25_DEADLINE:-2026-11-01 10:00}" +%s)
SINGLE_MIN=${SINGLE_MIN:-60.0}
INCLUDE_PROBES=${INCLUDE_PROBES:-1}
ORDER=(${V25_ORDER:-nq hotpotqa fever})
before_deadline() { [ "$(date +%s)" -lt "$DEADLINE" ]; }
step() { before_deadline || { log "deadline reached; skipping: $*"; return 1; }; "$@"; }
C=configs/next/3060
V9B="models/n1-5070 models/n3-5070 models/n4-5070 models/long-easy-5070 models/arm-mnrl-fiqa-lenmatch-far"

# an earlier queue owns the GPU and the git index until it exits
for prev in v24 v23; do
    if [ -f "$LOGS/queue_$prev.pid" ] && kill -0 "$(cat "$LOGS/queue_$prev.pid")" 2>/dev/null; then
        log "waiting for queue $prev (pid $(cat "$LOGS/queue_$prev.pid")) to exit"
        while kill -0 "$(cat "$LOGS/queue_$prev.pid")" 2>/dev/null; do sleep 60; done
        sleep 30
    fi
done

lb() { "$PY" -c "import json; print(round(json.load(open('results-regimes/quantized/$1/$2.json'))['mean_ndcg@10'], 2))" 2>/dev/null; }
dims() { sparsity_mean "results/sparsity/$1.json" "$1"; }
leaderboard() { regime quantized "$1" "models/$1" tiny10 16; regime quantized "$1" "models/$1" heldout8 16; }
train2() {  # train2 TAG CFG [FALLBACK]: train; when the 2 x 16 run fails, retry once with the cached-loss config
    local tag="$1" cfg="$2" fb="${3:-}"
    [ -d "models/arm-$tag" ] && return 0
    train "$tag" "$cfg" && return 0
    [ -d "models/arm-$tag" ] && return 0
    { [ -n "$fb" ] && [ -f "$fb" ]; } || return 1
    mv -f "$LOGS/arm_$tag.log" "$LOGS/arm_$tag.first.log" 2>/dev/null
    log "train arm-$tag: first attempt failed ($(grep -m1 -oE 'OutOfMemoryError|CUDA out of memory' "$LOGS/arm_$tag.first.log" 2>/dev/null || echo 'see logs/arm_'"$tag"'.first.log')); retry with $fb"
    train "$tag" "$fb"
}
arm() {  # arm TAG CFG [FALLBACK]: train, tiny, sparsity, held-out-8
    local tag="$1"
    step train2 "$tag" "$2" "${3:-}" || return 1
    step bench "arm-$tag" "models/arm-$tag"; step sparsity "arm-$tag" "models/arm-$tag"; step holdout "arm-$tag" "models/arm-$tag"
    log "  arm-$tag: tiny $(tiny "arm-$tag") held-out $(hold "arm-$tag") (n4-5070 $(tiny n4-5070) / $(hold n4-5070), n1-ldose2 $(tiny arm-n1-ldose2) / $(hold arm-n1-ldose2)), $(dims "arm-$tag") dims/doc"
}
merge_lb() {  # merge_lb NAME MODEL...: uniform soup into models/NAME, sparsity, leaderboard regime
    local name="$1"; shift
    if [ ! -d "models/$name" ]; then
        need_space || return 1; wait_gpu_free; wait_ram
        "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/merge_$name.log" 2>&1 \
            && log "soup $name = $*" || { log "soup $name FAILED (logs/merge_$name.log)"; return 1; }
    fi
    step sparsity "$name" "models/$name"; step leaderboard "$name"
    log "  $name leaderboard: $(lb "$name" tiny10) / $(lb "$name" heldout8), $(dims "$name") dims/doc (soup-v9b $(lb soup-v9b tiny10) / $(lb soup-v9b heldout8), 409)"
}
heldout2() {  # heldout2 NAME: the five held-out-2 tasks in the leaderboard regime
    local t; for t in czywiesz gpt35wiki poquad quora treccovid; do before_deadline || return 0; regime quantized "$1" "models/$1" "heldout2_$t" 16; done
    log "  $1 held-out-2: $("$PY" -c "
import json, glob
v = [json.load(open(f))['mean_ndcg@10'] for f in sorted(glob.glob('results-regimes/quantized/$1/heldout2_*.json'))]
print(f'{sum(v)/len(v):.2f} over {len(v)} tasks' if v else 'no result')") (soup-v9b 69.42, soup-v7a 69.16)"
}
best_by_lb() {  # best_by_lb NAME...: the name with the highest leaderboard heldout8 among those that have one
    local best="" bv="" m v
    for m in "$@"; do
        v=$(lb "$m" heldout8); [ -n "$v" ] || continue
        if [ -z "$bv" ] || [ "$(ge "$v" "$bv")" = 1 ]; then best="$m"; bv="$v"; fi
    done
    echo "$best"
}
diag() {  # diag: student tail + Polish teacher on the pools and the reference files (skipped when complete)
    local out=results/data/speakleash-diag.json
    if [ -f "$out" ] && [ "$("$PY" -c "import json; print(len(json.load(open('$out'))['files']))" 2>/dev/null)" = 6 ]; then return 0; fi
    need_space || return 1; wait_gpu_free; wait_ram
    local s=$SECONDS
    "$PY" scripts/speakleash_pool_diag.py \
        --input datasets/arms/pool/nq-pool-8000.jsonl --input datasets/arms/pool/hotpotqa-pool-8000.jsonl --input datasets/arms/pool/fever-pool-8000.jsonl \
        --input datasets/arms/fiqa-2073.jsonl --input datasets/arms/squadv2-2073.jsonl --input datasets/arms/msmarco-clean-2073.jsonl \
        --out "$out" --per-row-dir results/data/speakleash-diag > "$LOGS/speakleash_diag.log" 2>&1
    local rc=$?
    record "diag:speakleash" "$([ $rc = 0 ] && echo ok || echo FAILED)" "-" "$((SECONDS - s))s"
    [ $rc = 0 ] && log "diagnostics written ($out, $((SECONDS - s)) s)" || log "diagnostics FAILED (logs/speakleash_diag.log)"
    return $rc
}
screen_set() {  # screen_set SET: the single-set arm, tiny + sparsity
    local s="$1" tag="mnrl-$1-2073"
    step train2 "$tag" "$C/mnrl_${s}_2073.yaml" "$C/mnrl_${s}_2073_cb32.yaml" || return 1
    step bench "arm-$tag" "models/arm-$tag"; step sparsity "arm-$tag" "models/arm-$tag"
    log "  arm-$tag: tiny $(tiny "arm-$tag") (base $BASE, squadv2-2073 $(tiny arm-squadv2-2073), msmarco-clean-2073 $(tiny arm-msmarco-clean-2073), threshold $SINGLE_MIN), $(dims "arm-$tag") dims/doc"
}
passed() { local v; v=$(tiny "arm-mnrl-$1-2073"); [ -n "$v" ] && [ "$(ge "$v" "$SINGLE_MIN")" = 1 ]; }
probe() {  # probe TAG CFG: one GradCache speed probe (3 steps); appends a line to the probe log
    local tag="$1" cfg="$2" out mem s e peak t
    out="$LOGS/smoke_$tag.log"; mem="$LOGS/smoke_$tag.mem"
    grep -q "^$tag " "$PROBES" 2>/dev/null && return 0
    wait_gpu_free; wait_ram
    rm -rf "runs/smoke-$tag"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$mem" 2>/dev/null & local mempid=$!
    s=$SECONDS
    timeout -k 60 900 "$PY" run_experiments.py --grid-configuration-file "$cfg" --backend sentence-transformers --training-type splade > "$out" 2>&1
    e=$((SECONDS - s))
    kill "$mempid" 2>/dev/null; wait "$mempid" 2>/dev/null
    peak=$(sort -n "$mem" 2>/dev/null | tail -1); peak=${peak:-0}
    t=$(grep -oE "3/3 \[[0-9:]+<[0-9:]+, *[0-9.]+s/it" "$out" | tail -1 | grep -oE "[0-9.]+s/it" | tr -d 's/it')
    if grep -qE "OutOfMemoryError|CUDA out of memory" "$out"; then
        echo "$tag OOM ${e} ${peak} -" >> "$PROBES"; log "probe $tag: CUDA OOM after ${e}s (peak ${peak} MiB)"
    elif [ -n "$t" ]; then
        echo "$tag ok ${e} ${peak} ${t}" >> "$PROBES"; log "probe $tag: ${t} s/step, peak ${peak} MiB, ${e}s total"
    else
        echo "$tag FAILED ${e} ${peak} -" >> "$PROBES"; log "probe $tag FAILED (logs/smoke_$tag.log)"
    fi
    rm -rf "runs/smoke-$tag"
}
PROBES=results/data/gradcache-smoke.txt
mkdir -p results/data

BASE=$(tiny base-splade-wrapped)
log "=== v25 start: base $BASE, n4-5070 $(tiny n4-5070) / $(hold n4-5070), soup-v9b $(lb soup-v9b tiny10) / $(lb soup-v9b heldout8), order ${ORDER[*]}, deadline $(date -d "@$DEADLINE" '+%d.%m %H:%M'), $(free_gb) GB free ==="

log "=== v25 step 0: diagnostics (student tail, Polish teacher) on the pools and the reference files ==="
step diag

log "=== v25 step 1: single-set screen (${ORDER[*]}) ==="
for s in "${ORDER[@]}"; do step screen_set "$s"; done

log "=== v25 step 2: n4 pattern for the sets that passed the screen ==="
for s in "${ORDER[@]}"; do
    if passed "$s"; then step arm "n4x-$s" "$C/n4x_$s.yaml" "$C/n4x_${s}_cb32.yaml"
    else log "  $s: single arm $(tiny "arm-mnrl-$s-2073") below $SINGLE_MIN (or missing); no n4x arm"; fi
done

log "=== v25 step 3: control arm-n4-3060 (n4's rows at 2 x 16) ==="
step arm n4-3060 "$C/n4_3060.yaml" "$C/n4_3060_cb32.yaml"

log "=== v25 step 4: leaderboard regime for the n4x arms that reach the control, then the control ==="
CTRL=$(tiny arm-n4-3060); CTRL=${CTRL:-$(tiny n4-5070)}
LB_ARMS=()
for s in "${ORDER[@]}"; do
    m="arm-n4x-$s"; v=$(tiny "$m"); [ -n "$v" ] || continue
    if [ "$(ge "$v" "$(fsub "$CTRL" 0.3)")" = 1 ]; then
        step leaderboard "$m" && LB_ARMS+=("$m")
        log "  $m leaderboard: $(lb "$m" tiny10) / $(lb "$m" heldout8) (n4-5070 $(lb n4-5070 tiny10) / $(lb n4-5070 heldout8), n1-5070 $(lb n1-5070 tiny10) / $(lb n1-5070 heldout8))"
    else log "  $m: truncated tiny $v below control $CTRL - 0.3; no leaderboard regime"; fi
done
[ -d models/arm-n4-3060 ] && { step leaderboard arm-n4-3060; log "  arm-n4-3060 leaderboard: $(lb arm-n4-3060 tiny10) / $(lb arm-n4-3060 heldout8) (n4-5070 $(lb n4-5070 tiny10) / $(lb n4-5070 heldout8))"; }

log "=== v25 step 5: soups v9b + arm ==="
SOUPS=()
for m in "${LB_ARMS[@]}"; do
    s=${m#arm-n4x-}
    # shellcheck disable=SC2086
    step merge_lb "soup-v10-$s" $V9B "models/$m" && SOUPS+=("soup-v10-$s")
done

log "=== v25 step 6: held-out-2 for the best new soup ==="
if [ "${#SOUPS[@]}" -gt 0 ]; then
    BEST=$(best_by_lb "${SOUPS[@]}")
    a=$(lb "${BEST:-none}" heldout8); b=$(lb soup-v9b heldout8)
    if [ -n "$BEST" ] && [ -n "$a" ] && [ "$(ge "$a" "$b")" = 1 ]; then
        log "  best new soup by leaderboard heldout8: $BEST $a (soup-v9b $b)"; step heldout2 "$BEST"
    else
        log "  no new soup reaches soup-v9b on the leaderboard heldout8 (best ${BEST:-none} ${a:-}); no held-out-2"
    fi
else
    log "  no soups built"
fi

if [ "$INCLUDE_PROBES" = 1 ]; then
    log "=== v25 step 7: filler, GradCache speed probes (true batch 32, 3 steps) ==="
    for mb in 4 8 16; do for gc in gc nogc; do step probe "cb32-mb$mb-$gc" "$C/smoke_cb32_mb${mb}_$gc.yaml"; done; done
    [ -f "$PROBES" ] && "$PY" - <<'EOF' 2>> "$LOGS/queue_summary.log"
import json
rows = []
for line in open("results/data/gradcache-smoke.txt"):
    tag, status, total, peak, step = line.split()
    rows.append({"probe": tag, "status": status, "total_s": int(total), "peak_mib": int(peak), "s_per_step": None if step == "-" else float(step)})
json.dump({"recipe": "cluster-like, true batch 32, 512 tokens, 10 negatives, fp16, pooling chunk 8, 3 optimizer steps", "probes": rows},
          open("results/data/gradcache-smoke.json", "w"), indent=2)
EOF
fi

log "=== v25 step 8: summary, commit, push ==="
cp -f datasets/arms/manifest_speakleash.json results/data/speakleash-manifest.json 2>/dev/null
"$PY" scripts/queue_summary.py > results/summary_v25.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v25.md)"
{
    echo; echo "## Per-task, leaderboard regime (queue v25)"; echo
    "$PY" scripts/regime_task_table.py --tasksets tiny10 heldout8 heldout2_treccovid \
        --models base-splade-wrapped n1-5070 n4-5070 soup-v9b arm-n4-3060 arm-n4x-nq arm-n4x-hotpotqa arm-n4x-fever soup-v10-nq soup-v10-hotpotqa soup-v10-fever 2>> "$LOGS/queue_summary.log"
    echo; echo "## Per-task, truncated regime: single-set screen (vs the additive components used so far)"; echo
    "$PY" scripts/seed_noise_table.py --models arm-squadv2-2073 arm-msmarco-clean-2073 arm-mnrl-nq-2073 arm-mnrl-hotpotqa-2073 arm-mnrl-fever-2073 2>> "$LOGS/queue_summary.log"
    echo; echo "## Per-task, truncated regime: n4 pattern arms vs the control"; echo
    "$PY" scripts/seed_noise_table.py --models arm-n4-3060 arm-n4x-nq arm-n4x-hotpotqa arm-n4x-fever n4-5070 arm-n1-ldose2 2>> "$LOGS/queue_summary.log"
    echo; echo "## Data: cuts (results/data/speakleash-manifest.json) and diagnostics (results/data/speakleash-diag.json)"; echo
    "$PY" - <<'EOF' 2>> "$LOGS/queue_summary.log"
import json, os
m = json.load(open("results/data/speakleash-manifest.json")) if os.path.exists("results/data/speakleash-manifest.json") else {}
print("| set | rows gated | positives pass | cut examples | pos tokens p50 | neg tokens p50 | median neg/max pos | gate p50 (cut) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for k, v in m.items():
    print(f"| {k} | {v['rows_gated_share']:.3f} | {v['positives_pass_share']:.3f} | {v['cut_training_examples']} | {v['cut_tokens_pos'].get('p50')} | {v['cut_tokens_neg'].get('p50')} | {v['cut_median_neg_len_over_max_pos_len'].get('p50')} | {v['cut_gate_score_percentiles'].get('p50')} |")
if os.path.exists("results/data/speakleash-diag.json"):
    d = json.load(open("results/data/speakleash-diag.json"))
    print(); print("| file | tail negs/row | rows with tail | negs above pos | v3 pos p50 | below fiqa-v3 p10 | gate~v3 r | v3 false negs | v3 pos-max neg p50 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in d["files"].items():
        print(f"| {k} | {v['tail_negatives_per_row_mean']} | {v['rows_with_tail_share']} | {v['negatives_above_positive_share']} | {v.get('v3_pos', {}).get('p50')} | {v.get('v3_pos_below_reference', {}).get('p10')} | {v.get('gate_vs_v3_pearson')} | {v.get('v3_neg_at_or_above_weakest_pos_share')} | {v.get('v3_pos_minus_max_v3_neg', {}).get('p50')} |")
EOF
    echo; echo "## Queue state"; echo; echo '```'; grep -E "^(train|bench|holdout|sparsity|regime|diag):" "$STATE" | tail -60; echo '```'
} >> results/summary_v25.md
for p in results/arm-mnrl-nq-2073 results/arm-mnrl-hotpotqa-2073 results/arm-mnrl-fever-2073 results/arm-n4x-* results/arm-n4-3060 \
         results-holdout/arm-n4x-* results-holdout/arm-n4-3060 results-regimes/quantized results/sparsity \
         results/data/speakleash-diag.json results/data/speakleash-diag results/data/speakleash-manifest.json \
         results/data/gradcache-smoke.json results/data/gradcache-smoke.txt results/summary_v25.md; do
    [ -e "$p" ] && git add "$p" 2>> "$LOGS/queue_v25.out"
done
if git commit -q -m "Add the v25 results: the new speakleash sets as components, the n4 control at 2 x 16 and the soups with them" 2>> "$LOGS/queue_v25.out"; then
    log "results committed ($(git log --oneline -1))"
    ( for i in 1 2 3 4 5 6; do timeout -k 10 180 git push origin master >> "$LOGS/push_v25.log" 2>&1 && { echo "pushed (attempt $i)" >> "$LOGS/push_v25.log"; break; }; sleep 120; done ) > /dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
    log "push started in the background (logs/push_v25.log)"
else
    log "nothing to commit or commit failed (see logs/queue_v25.out)"
fi
log "=== v25 done ($(date '+%d.%m %H:%M')) ==="
