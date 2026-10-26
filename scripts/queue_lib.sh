#!/usr/bin/env bash
# Shared helpers for the GPU queues (sourced by scripts/queue_v5.sh and later
# queues). Set QUEUE_TAG before sourcing. Everything runs serially; every GPU
# job waits for an idle card and >= 6 GB free RAM, runs under `timeout`, and a
# failed step is logged and skipped, never fatal.
# JDK 21 for Anserini: a JAVA_HOME already pointing at a JDK wins, otherwise the
# first local JDK 21 install (~/opt/jdk-21*, /usr/lib/jvm/*21*).
if [ -z "${JAVA_HOME:-}" ] || [ ! -x "${JAVA_HOME}/bin/java" ]; then
    JAVA_HOME=$(ls -d "$HOME"/opt/jdk-21* /usr/lib/jvm/*21* 2>/dev/null | head -1)
fi
[ -x "${JAVA_HOME:-/nonexistent}/bin/java" ] || echo "queue_lib: no JDK 21 found; set JAVA_HOME before launching (PIRB needs it)" >&2
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"
PY="$(pwd)/.venv/bin/python"
LOGS="$(pwd)/logs"
STATE="$LOGS/queue_state.tsv"
HOLDOUT_DIR="$(pwd)/results-holdout"
HOLDOUT2_DIR="$(pwd)/results-holdout2"
HOLDOUT_TASKS="maupqa-gpt3.5-cc,maupqa-multilingual-NLI,mfaq,poleval-2022-test-A-legal-questions,maupqa-wiki-def,scidocs-pl,maupqa-1z10,fiqa-pl"
HOLDOUT2_TASKS=(maupqa-czy-wiesz-v2 maupqa-gpt3.5-wiki quora-pl maupqa-poquad trec-covid-pl)
QUEUE_TAG="${QUEUE_TAG:-q}"
MIN_RAM_GB="${MIN_RAM_GB:-6}"
mkdir -p "$LOGS" results/base_loss "$HOLDOUT2_DIR" results-regimes

log()    { echo "[$(date +%H:%M:%S)] [$QUEUE_TAG] $*" | tee -a "$LOGS/queue.log"; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${4:-}" >> "$STATE"; }
avg_of() { "$PY" -c "import json;print(round(json.load(open('$1'))['epoch0/pirb_average_ndcg@10'],2))" 2>/dev/null; }
tiny()   { avg_of "results/$1/final/metrics.json"; }
hold()   { avg_of "$HOLDOUT_DIR/$1/final/metrics.json"; }
ge()     { "$PY" -c "import sys;a,b=sys.argv[1:3];print(int(bool(a) and bool(b) and float(a)>=float(b)))" "$1" "$2"; }
fsub()   { "$PY" -c "import sys;print(round(float(sys.argv[1])-float(sys.argv[2]),2))" "$1" "$2"; }

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }
need_space() {
    local gb waited=0; gb=$(free_gb)
    while [ "${gb:-0}" -lt 10 ] && [ "$waited" -lt 3600 ]; do
        log "only ${gb} GB free on /; waiting (clean runs/ or cache/ to continue)"; sleep 600; waited=$((waited + 600)); gb=$(free_gb)
    done
    [ "${gb:-0}" -lt 10 ] && { log "disk still full (${gb} GB); skipping this step"; return 1; }
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
wait_ram() {
    local waited=0 avail
    while :; do
        avail=$(free -g | awk '/^Mem:/ {print $7}')
        [ "${avail:-0}" -ge "$MIN_RAM_GB" ] && return 0
        [ "$waited" -ge 7200 ] && { log "RAM available ${avail} GB after 2 h; going ahead"; return 0; }
        sleep 60; waited=$((waited + 60))
    done
}
guard_alive() {
    ps -eo args | grep -q "[o]rphan_guard.py" || { setsid nohup python3 scripts/orphan_guard.py >> "$LOGS/orphan_guard.out" 2>&1 < /dev/null & log "restarted orphan guard"; }
}

sparsity_mean() {  # sparsity_mean FILE LABEL: mean over tasks of the mean active document dims
    "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]))[sys.argv[2]];v=[t['mean'] for t in d.values() if t.get('mean') is not None];print(round(sum(v)/len(v),1))" "$1" "$2" 2>/dev/null
}
sparsity() {  # sparsity NAME MODEL: active document dims (round(w*100) > 0) on 300 passages of each tiny task, next to base
    local name="$1" model="$2"
    [ -f "results/sparsity/$name.json" ] && return 0
    [ -e "$model" ] || { log "sparsity $name: $model missing"; return 1; }
    wait_gpu_free; wait_ram
    log "sparsity $name"; local s=$SECONDS
    timeout -k 60 1800 "$PY" scripts/compare_splade_sparsity.py --model "base=models/base-splade-wrapped" --model "$name=$model" \
        --scope tiny --max-docs-per-task 300 --device cuda --batch-size 16 \
        --output-json "results/sparsity/$name.json" > "$LOGS/sparsity_$name.log" 2>&1
    local e=$((SECONDS - s))
    if [ -f "results/sparsity/$name.json" ]; then
        log "sparsity $name -> $(sparsity_mean "results/sparsity/$name.json" "$name") active dims/doc (base $(sparsity_mean "results/sparsity/$name.json" base)) in ${e}s"
        record "sparsity:$name" ok "$(sparsity_mean "results/sparsity/$name.json" "$name")" "${e}s"
    else log "sparsity $name FAILED after ${e}s"; record "sparsity:$name" FAILED "-" "${e}s"; return 1; fi
}
sparsity_missing() {  # sparsity for every model with a tiny result and no sparsity file (base itself excluded)
    local f name
    for f in results/*/final/metrics.json; do
        name=$(basename "$(dirname "$(dirname "$f")")")
        [ "$name" = base-splade-wrapped ] && continue
        [ -d "models/$name" ] && [ ! -f "results/sparsity/$name.json" ] && sparsity "$name" "models/$name"
    done
}

drop_indexes() {  # drop_indexes MODEL: remove the PIRB index dirs of a model whose benchmark failed (no saved result would ever prune them; 26.10 morning: 12 GB leaked this way)
    local slug n
    slug=$("$PY" -c "import os,sys;print(os.path.abspath(sys.argv[1]).replace('/','_').replace('.','_'))" "$1")
    n=$(ls -d third_party/pirb/cache/*/"$slug" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && { rm -rf third_party/pirb/cache/*/"$slug"; log "dropped $n index dir(s) of $1 after the failure"; }
    return 0
}

bench() {  # bench NAME MODEL [extra args]: tiny, truncated-query regime (as every number so far)
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
    else log "bench $name FAILED after ${e}s"; record "bench:$name" FAILED "-" "${e}s"; drop_indexes "$model"; return 1; fi
}
holdout() {  # holdout NAME MODEL: the 8 held-out tasks, truncated-query regime, encoding batch 16
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
    else log "holdout $name FAILED after ${e}s"; record "holdout:$name" FAILED "-" "${e}s"; drop_indexes "$model"; return 1; fi
}
train() {  # train NAME CONFIG: train and copy final/ to models/arm-NAME
    local tag="$1" cfg="$2"
    [ -d "models/arm-$tag" ] && return 0
    [ -f "$cfg" ] || { log "train arm-$tag: config $cfg missing"; record "train:arm-$tag" SKIPPED "no config"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "train arm-$tag"; local s=$SECONDS
    timeout -k 60 7200 "$PY" run_experiments.py --grid-configuration-file "$cfg" \
        --backend sentence-transformers --training-type splade > "$LOGS/arm_$tag.log" 2>&1
    local e=$((SECONDS - s)) ck
    ck=$(ls -1dt "runs/arm-$tag/sentence-transformers/splade"/*/final 2>/dev/null | head -1)
    [ -z "$ck" ] && { log "train arm-$tag: no checkpoint after ${e}s"; record "train:arm-$tag" FAILED "-" "${e}s"; return 1; }
    cp -r "$ck" "models/arm-$tag" || return 1
    log "train arm-$tag done in ${e}s"; record "train:arm-$tag" ok "-" "${e}s"
}
split_block() {  # split_block SRC TAG...: train all, tiny all, held-out for control + best when best >= control + 0.4
    local src="$1"; shift
    local tag name best="" best_v="" v
    for tag in "$@"; do
        name="$src-$tag"
        [ -f "configs/weekend/splits/arm_${src}_${tag}.yaml" ] || { log "split $name: not built"; continue; }
        train "$name" "configs/weekend/splits/arm_${src}_${tag}.yaml"
    done
    for tag in "$@"; do
        name="$src-$tag"; [ -d "models/arm-$name" ] || continue
        bench "arm-$name" "models/arm-$name"
        v=$(tiny "arm-$name")
        if [ "$tag" != rand ] && [ -n "$v" ] && { [ -z "$best_v" ] || [ "$(ge "$v" "$best_v")" = 1 ]; }; then best="$name"; best_v="$v"; fi
    done
    local ctrl; ctrl=$(tiny "arm-$src-rand")
    log "split block $src: control=$ctrl best=$best ($best_v)"
    if [ -n "$best" ] && [ -n "$ctrl" ] && [ "$(ge "$(fsub "$best_v" "$ctrl")" 0.4)" = 1 ]; then
        log "split block $src: $best beats the random half by $(fsub "$best_v" "$ctrl"); held-out for both"
        holdout "arm-$src-rand" "models/arm-$src-rand"; holdout "arm-$best" "models/arm-$best"
    else log "split block $src: no split beats the random half by 0.4 on tiny; no held-out"; fi
}
regime_mean() { "$PY" -c "import json;d=json.load(open('$1'));print(round(d['mean_ndcg@10'],2) if d['mean_ndcg@10'] is not None else '', d['n_tasks'])" 2>/dev/null; }
regime() {  # regime REGIME LABEL MODEL TASKSET [BATCH] [MAXLEN]: scripts/pirb_regime_eval.sh with the usual guards
    local reg="$1" label="$2" model="$3" ts="$4" bs="${5:-16}" maxlen="${6:-}" out="results-regimes/$1/$2/$4.json"
    [ -f "$out" ] && [ "$(regime_mean "$out" | awk '{print $2}')" != "0" ] && { log "regime $reg/$label/$ts: already present ($(regime_mean "$out"))"; return 0; }
    [ -e "$model" ] || { log "regime $reg/$label/$ts: $model missing"; return 1; }
    need_space || return 1; wait_gpu_free; wait_ram
    log "regime $reg/$label/$ts (batch $bs)"; local s=$SECONDS
    CUDA_VISIBLE_DEVICES=0 timeout -k 60 10800 bash scripts/pirb_regime_eval.sh "$reg" "$label" "$model" "$ts" "$bs" $maxlen > "$LOGS/regime_${reg}_${label}_${ts}.out" 2>&1
    local e=$((SECONDS - s)) m
    m=$(regime_mean "$out")
    if [ -n "$m" ] && [ "$(echo "$m" | awk '{print $2}')" != "0" ]; then
        log "regime $reg/$label/$ts -> mean=$m in ${e}s"; record "regime:$reg:$label:$ts" ok "$m" "${e}s"
    else log "regime $reg/$label/$ts FAILED after ${e}s"; record "regime:$reg:$label:$ts" FAILED "-" "${e}s"; rm -rf "cache/regimes/${reg}_${label}_${ts}"; return 1; fi
}
