#!/usr/bin/env bash
# Evening/overnight queue (Sunday 2026-10-25). Starts when scripts/queue_v3.sh
# (pid in PREV_QUEUE_PID, default 206133) has exited and the GPU is idle.
#
# Blocks, in order of how much each can change a decision:
#   A. per-row base-model difficulty for every arm file (GPU, ~10 min) and the
#      half-dose split arms built from it + the lexical metrics
#   B. soup v2 candidates (soup-of-ft + the SQuAD arms), tiny, held-out if close
#   C. FiQA splits: random / hard / easy (base loss) / low / high coverage
#   D. trivia splits: random / easy / hard (answerability hypothesis)
#   E. second held-out set (5 PIRB tasks never used so far) for base + soup-of-ft
#   F. SQuAD-v2 splits, G. wikisynth splits
#   H. second held-out set for ck-warmup-s, arm-squadv2-2073, soup v2 winner
#   I. index-cost profile of everything new, HF metric files, summary table
#
# Safety: every GPU job runs serially under `timeout`; benchmarks wait for the
# CPU lemma job (which runs under a 5 GB cgroup cap) and for >= 6 GB free RAM;
# held-out runs use encoding batch 16 (8 on the two long-document corpora);
# the disk guard moves finished old run dirs to the media drive instead of
# aborting; a failed step is logged and the queue moves on.
#
# Stop it: kill $(cat logs/queue_v4.pid); then kill the python child it lists.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export JAVA_HOME=/home/jsowa/opt/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
PY="$(pwd)/.venv/bin/python"
LOGS="$(pwd)/logs"
STATE="$LOGS/queue_state.tsv"
HOLDOUT_DIR="$(pwd)/results-holdout"
HOLDOUT2_DIR="$(pwd)/results-holdout2"
HOLDOUT_TASKS="maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022-test-A-legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl"
HOLDOUT2_TASKS=(maupqa-czy-wiesz-v2 maupqa-gpt3.5-wiki quora-pl maupqa-poquad trec-covid-pl)
MEDIA=/media/jsowa/6aaaf1ac-2d6f-44c6-9a39-a59761abdadf/work/FlagEmbedding-runs
PREV_PID="${PREV_QUEUE_PID:-206133}"
mkdir -p "$LOGS" results/base_loss "$HOLDOUT2_DIR"
echo $$ > "$LOGS/queue_v4.pid"

log()    { echo "[$(date +%H:%M:%S)] [v4] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }
avg_of() { "$PY" -c "import json;print(round(json.load(open('$1'))['epoch0/pirb_average_ndcg@10'],2))" 2>/dev/null; }
tiny()   { avg_of "results/$1/final/metrics.json"; }
hold()   { avg_of "$HOLDOUT_DIR/$1/final/metrics.json"; }
# ge A B: 1 if A >= B (floats; empty counts as -inf)
ge()     { "$PY" -c "import sys;a,b=sys.argv[1:3];print(int(bool(a) and bool(b) and float(a)>=float(b)))" "$1" "$2"; }
fsub()   { "$PY" -c "import sys;print(round(float(sys.argv[1])-float(sys.argv[2]),2))" "$1" "$2"; }

# ---------------------------------------------------------------- guards ----
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }
need_space() {
    local gb; gb=$(free_gb)
    if [ "${gb:-0}" -lt 15 ]; then
        log "only ${gb} GB free: moving finished run dirs to the media drive"
        mkdir -p "$MEDIA/moved-runs"
        for d in runs/sweep-* runs/weekend-3060 runs/weekend-3060-long runs/judged-* runs/smoke; do
            [ -d "$d" ] && [ ! -L "$d" ] || continue
            mv "$d" "$MEDIA/moved-runs/" && ln -s "$MEDIA/moved-runs/$(basename "$d")" "$d" && log "moved $d"
            [ "$(free_gb)" -ge 20 ] && break
        done
        gb=$(free_gb)
    fi
    local waited=0
    while [ "${gb:-0}" -lt 12 ] && [ "$waited" -lt 3600 ]; do
        log "still only ${gb} GB free; waiting"; sleep 600; waited=$((waited + 600)); gb=$(free_gb)
    done
    if [ "${gb:-0}" -lt 12 ]; then log "disk still full (${gb} GB); skipping this step"; return 1; fi
    return 0
}
gpu_busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
wait_gpu_free() {
    local waited=0 n
    while :; do
        n=$(gpu_busy); [ "$n" -eq 0 ] && sleep 10 && [ "$(gpu_busy)" -eq 0 ] && return 0
        [ "$waited" -ge 2400 ] && { log "GPU still busy after 40 min ($n procs); going ahead anyway"; return 0; }
        sleep 30; waited=$((waited + 30))
    done
}
CPU_JOB_PID=""
wait_ram() {
    # benchmarks need ~5 GB on top of the JVM; never start one next to the Stanza job
    local waited=0 avail
    while :; do
        if [ -n "$CPU_JOB_PID" ] && kill -0 "$CPU_JOB_PID" 2>/dev/null; then
            [ "$waited" -ge 7200 ] && { log "CPU job $CPU_JOB_PID still running after 2 h; not waiting longer"; break; }
            sleep 60; waited=$((waited + 60)); continue
        fi
        avail=$(free -g | awk '/^Mem:/ {print $7}')
        [ "${avail:-0}" -ge 6 ] && return 0
        [ "$waited" -ge 7200 ] && { log "RAM available ${avail} GB after 2 h; going ahead"; return 0; }
        sleep 60; waited=$((waited + 60))
    done
}

# ------------------------------------------------------------- GPU steps ----
# bench NAME MODEL [extra args]: tiny (10 tasks)
bench() {
    local name="$1" model="$2"; shift 2
    [ -f "results/$name/final/metrics.json" ] && { log "bench $name: already present ($(tiny "$name"))"; return 0; }
    [ -e "$model" ] || { log "bench $name: $model missing"; record "bench:$name" SKIPPED "missing"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "bench $name"; local s=$SECONDS
    timeout -k 60 5400 "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-max-seq-length 512 --remove-pirb-indexes "$@" \
        --benchmark-target "${model}::" --benchmark-output-dir "results/$name" > "$LOGS/bench_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "results/$name/final/metrics.json" ]; then
        log "bench $name -> ndcg=$(tiny "$name") in ${e}s"; record "bench:$name" ok "$(tiny "$name")" "${e}s"
    else
        log "bench $name FAILED after ${e}s"; record "bench:$name" FAILED "-" "${e}s"; return 1
    fi
}
# holdout NAME MODEL: the 8 held-out tasks, encoding batch 16
holdout() {
    local name="$1" model="$2"
    [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ] && { log "holdout $name: already present ($(hold "$name"))"; return 0; }
    [ -e "$model" ] || { log "holdout $name: $model missing"; record "holdout:$name" SKIPPED "missing"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "holdout $name"; local s=$SECONDS
    timeout -k 60 9000 "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
        --pirb-tasks "$HOLDOUT_TASKS" --pirb-max-seq-length 512 --pirb-batch-size 16 --remove-pirb-indexes \
        --benchmark-target "${model}::" --benchmark-output-dir "$HOLDOUT_DIR/$name" > "$LOGS/holdout_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "$HOLDOUT_DIR/$name/final/metrics.json" ]; then
        log "holdout $name -> ndcg=$(hold "$name") in ${e}s"; record "holdout:$name" ok "$(hold "$name")" "${e}s"
    else
        log "holdout $name FAILED after ${e}s"; record "holdout:$name" FAILED "-" "${e}s"; return 1
    fi
}
# holdout2 NAME MODEL: the second never-used task set, one process per task so
# one OOM cannot sink the rest; batch 8 on the two long-document corpora
holdout2() {
    local name="$1" model="$2" t bs
    [ -e "$model" ] || { log "holdout2 $name: $model missing"; record "holdout2:$name" SKIPPED "missing"; return 1; }
    for t in "${HOLDOUT2_TASKS[@]}"; do
        [ -f "$HOLDOUT2_DIR/$name/$t/final/metrics.json" ] && continue
        need_space || return 1; wait_gpu_free; wait_ram
        bs=16; case "$t" in trec-covid-pl|maupqa-poquad) bs=8;; esac
        log "holdout2 $name / $t (batch $bs)"; local s=$SECONDS
        timeout -k 60 10800 "$PY" run_experiments.py --mode benchmark --run_pirb --pirb_scope tiny \
            --pirb-tasks "$t" --pirb-max-seq-length 512 --pirb-batch-size "$bs" --remove-pirb-indexes \
            --benchmark-target "${model}::" --benchmark-output-dir "$HOLDOUT2_DIR/$name/$t" > "$LOGS/holdout2_${name}_$t.log" 2>&1
        local e=$((SECONDS - s))
        if [ -f "$HOLDOUT2_DIR/$name/$t/final/metrics.json" ]; then
            log "holdout2 $name / $t -> ndcg=$(avg_of "$HOLDOUT2_DIR/$name/$t/final/metrics.json") in ${e}s"
            record "holdout2:$name:$t" ok "$(avg_of "$HOLDOUT2_DIR/$name/$t/final/metrics.json")" "${e}s"
        else
            log "holdout2 $name / $t FAILED after ${e}s"; record "holdout2:$name:$t" FAILED "-" "${e}s"
        fi
    done
    local base_arg=()
    [ "$name" != base ] && [ -d "$HOLDOUT2_DIR/base" ] && base_arg=(--base "$HOLDOUT2_DIR/base")
    "$PY" scripts/aggregate_task_runs.py "$HOLDOUT2_DIR/$name" --expect "${#HOLDOUT2_TASKS[@]}" "${base_arg[@]}" >> "$LOGS/holdout2_summary.log" 2>&1 \
        && log "holdout2 $name summary: $("$PY" -c "import json;d=json.load(open('$HOLDOUT2_DIR/$name/summary.json'));print(d['n_tasks'],'tasks mean',round(d['mean_ndcg@10'] or 0,2),'delta',d.get('mean_delta_common'))")"
}
# train NAME CONFIG: train and copy final/ to models/arm-NAME
train() {
    local tag="$1" cfg="$2"
    [ -d "models/arm-$tag" ] && return 0
    [ -f "$cfg" ] || { log "train arm-$tag: config $cfg missing"; record "train:arm-$tag" SKIPPED "no config"; return 1; }
    need_space || return 1; wait_gpu_free
    log "train arm-$tag"; local s=$SECONDS
    timeout -k 60 7200 "$PY" run_experiments.py --grid-configuration-file "$cfg" \
        --backend sentence-transformers --training-type splade > "$LOGS/arm_$tag.log" 2>&1
    local e=$((SECONDS - s)) ck
    ck=$(ls -1dt "runs/arm-$tag/sentence-transformers/splade"/*/final 2>/dev/null | head -1)
    [ -z "$ck" ] && { log "train arm-$tag: no checkpoint after ${e}s"; record "train:arm-$tag" FAILED "-" "${e}s"; return 1; }
    cp -r "$ck" "models/arm-$tag" || return 1
    log "train arm-$tag done in ${e}s"; record "train:arm-$tag" ok "-" "${e}s"
}
# split_block SRC TAG...: train every split first (GPU-only, low RAM, so it can
# overlap the CPU lemma job), then tiny for each, then held-out for the control
# and the best split when the best beats the control by >= 0.4
split_block() {
    local src="$1"; shift
    local tag name best="" best_v="" v
    for tag in "$@"; do
        name="$src-$tag"
        [ -f "configs/weekend/splits/arm_${src}_${tag}.yaml" ] || { log "split $name: not built (metric file missing?)"; continue; }
        train "$name" "configs/weekend/splits/arm_${src}_${tag}.yaml"
    done
    for tag in "$@"; do
        name="$src-$tag"
        [ -d "models/arm-$name" ] || continue
        bench "arm-$name" "models/arm-$name"
        v=$(tiny "arm-$name")
        if [ "$tag" != rand ] && [ -n "$v" ] && { [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; }; then best="$name"; best_v="$v"; fi
    done
    local ctrl; ctrl=$(tiny "arm-$src-rand")
    log "split block $src: control=$ctrl best=$best ($best_v)"
    if [ -n "$best" ] && [ -n "$ctrl" ] && [ "$(ge "$(fsub "$best_v" "$ctrl")" 0.4)" = 1 ]; then
        log "split block $src: $best beats the random half by $(fsub "$best_v" "$ctrl"); held-out for both"
        holdout "arm-$src-rand" "models/arm-$src-rand"
        holdout "arm-$best" "models/arm-$best"
    else
        log "split block $src: no split beats the random half by 0.4 on tiny; no held-out"
    fi
}

# ============================================================== take over ==
if kill -0 "$PREV_PID" 2>/dev/null; then
    log "waiting for queue_v3 (pid $PREV_PID) to finish"
    while kill -0 "$PREV_PID" 2>/dev/null; do sleep 60; done
fi
log "queue_v3 gone; waiting for the GPU"
wait_gpu_free
log "=== v4 start: $(free_gb) GB free on /, $(free -g | awk '/^Mem:/ {print $7}') GB RAM available ==="

# CPU lemma job for trivia + wikisynth, capped at 5 GB so it can never push a
# GPU job into swap; benchmarks wait for it (wait_ram)
if [ ! -s results/lexical_difficulty/full/wikisynth-2073.jsonl ] || [ ! -s results/lexical_difficulty/full/trivia-2073.jsonl ]; then
    systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 --quiet \
        bash scripts/lexical_cpu_job2.sh > "$LOGS/lexical_cpu_job2.out" 2>&1 &
    CPU_JOB_PID=$!
    log "started lexical CPU job 2 (pid $CPU_JOB_PID, 5 GB cap)"
fi

# ---------------------------------------------------------------- block A --
log "=== v4 block A: per-row base difficulty + split arms ==="
if [ ! -s results/base_loss/wikisynth-2073.jsonl ]; then
    wait_gpu_free
    timeout -k 60 3600 "$PY" scripts/splade_pair_loss.py --model models/base-splade-wrapped --n 0 \
        --input datasets/arms/fiqa-2073.jsonl --input datasets/arms/squadv2-2073.jsonl \
        --input datasets/arms/trivia-2073.jsonl --input datasets/arms/wikisynth-2073.jsonl \
        --input datasets/arms/msmarco-clean-2073.jsonl \
        --per-row-dir results/base_loss --output-json results/base_loss/summary.json > "$LOGS/base_loss.log" 2>&1 \
        && log "per-row base loss written" || log "per-row base loss FAILED"
fi
"$PY" scripts/build_split_arms.py 2>&1 | tee -a "$LOGS/build_split_arms.log" | while read -r l; do log "  split arms: $l"; done

# ---------------------------------------------------------------- block B --
log "=== v4 block B: soup v2 candidates ==="
if [ -d models/arm-fiqa-plus-squadv2 ] && [ -d models/arm-squadv2-2073 ]; then
    [ -d models/soup-v2a ] || "$PY" scripts/merge_splade_weights.py --models models/soup-of-ft models/arm-fiqa-plus-squadv2 \
        --alpha 0.5 --output models/soup-v2a > "$LOGS/merge_soup-v2a.log" 2>&1 || log "merge soup-v2a FAILED"
    [ -d models/soup-v2b ] || "$PY" scripts/merge_splade_weights.py --models models/soup-of-ft models/arm-squadv2-2073 models/arm-fiqa-plus-squadv2 \
        --weights 0.5 0.25 0.25 --output models/soup-v2b > "$LOGS/merge_soup-v2b.log" 2>&1 || log "merge soup-v2b FAILED"
    bench soup-v2a models/soup-v2a
    bench soup-v2b models/soup-v2b
    SOUP_REF=$(tiny soup-of-ft)
    SOUP_V2=""; SOUP_V2_V=""
    for n in soup-v2a soup-v2b; do
        v=$(tiny "$n"); [ -n "$v" ] || continue
        if [ -z "$SOUP_V2_V" ] || [ "$(ge "$v" "$SOUP_V2_V")" = 1 ]; then SOUP_V2="$n"; SOUP_V2_V="$v"; fi
    done
    if [ -n "$SOUP_V2" ] && [ "$(ge "$SOUP_V2_V" "$(fsub "$SOUP_REF" 0.2)")" = 1 ]; then
        log "soup v2 winner $SOUP_V2 ($SOUP_V2_V vs soup-of-ft $SOUP_REF): held-out"
        holdout "$SOUP_V2" "models/$SOUP_V2"
    else
        log "soup v2: best $SOUP_V2 ($SOUP_V2_V) is not within 0.2 of soup-of-ft ($SOUP_REF); no held-out"
        SOUP_V2=""
    fi
else
    log "soup v2 skipped: SQuAD arms from queue_v3 missing"; SOUP_V2=""
fi

# ---------------------------------------------------------------- block C --
log "=== v4 block C: FiQA half-dose splits ==="
split_block fiqa rand hard easy lowcov highcov

# ---------------------------------------------------------------- block D --
log "=== v4 block D: trivia half-dose splits (answerability hypothesis) ==="
"$PY" scripts/build_split_arms.py --sources trivia wikisynth >> "$LOGS/build_split_arms.log" 2>&1
split_block trivia rand easy hard lowcov highcov

# ---------------------------------------------------------------- block E --
log "=== v4 block E: second held-out set (5 unused PIRB tasks) for base and soup-of-ft ==="
holdout2 base models/base-splade-wrapped
holdout2 soup-of-ft models/soup-of-ft

# ---------------------------------------------------------------- block F --
log "=== v4 block F: SQuAD-v2 half-dose splits ==="
split_block squadv2 rand hard easy lowcov highcov

# ---------------------------------------------------------------- block G --
log "=== v4 block G: wikisynth half-dose splits ==="
"$PY" scripts/build_split_arms.py --sources wikisynth >> "$LOGS/build_split_arms.log" 2>&1
split_block wikisynth rand hard easy lowcov highcov

# ---------------------------------------------------------------- block H --
log "=== v4 block H: second held-out set for the runners-up ==="
holdout2 ck-warmup-s models/ck-warmup-s
holdout2 arm-squadv2-2073 models/arm-squadv2-2073
[ -n "$SOUP_V2" ] && holdout2 "$SOUP_V2" "models/$SOUP_V2"
[ -d models/arm-fiqa-plus-squadv2 ] && [ "$(ge "$(hold arm-fiqa-plus-squadv2)" "$(hold arm-squadv2-2073)")" = 1 ] && holdout2 arm-fiqa-plus-squadv2 models/arm-fiqa-plus-squadv2

# ---------------------------------------------------------------- block I --
log "=== v4 block I: index-cost profile, HF metric files, summary ==="
wait_gpu_free; wait_ram
models_args=(--model base=models/base-splade-wrapped --model soup-of-ft=models/soup-of-ft)
for d in models/ck-warmup-s models/soup-v2a models/soup-v2b models/arm-*; do
    [ -d "$d" ] || continue
    models_args+=(--model "$(basename "$d")=$d")
done
timeout -k 60 7200 "$PY" scripts/splade_index_cost.py --device cuda --per-task 150 --batch-size 4 "${models_args[@]}" \
    --output-json results/index_cost_v4.json > "$LOGS/index_cost_v4.log" 2>&1 && log "index-cost profile written (results/index_cost_v4.json)" || log "index-cost profile FAILED"
"$PY" scripts/build_hf_metric_files.py >> "$LOGS/build_hf_metric_files.log" 2>&1 && log "HF metric files written (results/lexical_difficulty/hf_upload)"
"$PY" scripts/queue_summary.py > results/summary_v4.md 2>> "$LOGS/queue_summary.log" && log "summary written (results/summary_v4.md)"
log "=== v4 queue finished ==="
