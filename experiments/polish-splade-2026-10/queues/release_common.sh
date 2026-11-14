# Shared by the release queues (v62, r1, st1, x1; source after scripts/queue_lib.sh and arm_common.sh): the frozen
# models (label files under logs/, which the user may overwrite), uniform soups, the transfer package for the 5070.
# Do not edit while a queue that sources this file is running.
PACK=${PACK:-datasets/transfer/release-20261113}
FINAL_FULL_FILE="$LOGS/final_full_model.txt"
FINAL_ALBERT_FILE="$LOGS/final_albert_model.txt"
FINAL_STATIC_FILE="$LOGS/final_static_table.txt"
first_line() { [ -s "$1" ] && sed -n '1p' "$1" | awk '{print $1}'; }
final_full()   { echo "${FINAL_FULL_MODEL:-$(first_line "$FINAL_FULL_FILE")}"; }     # label of the frozen full model (a directory under models/)
final_albert() { echo "${FINAL_ALBERT_MODEL:-$(first_line "$FINAL_ALBERT_FILE")}"; }
soup() {  # soup NAME MODEL_DIR...: uniform average of the raw weights (merge only)
    local name="$1" m; shift
    [ -d "models/$name" ] && return 0
    for m in "$@"; do [ -d "$m" ] || { log "soup $name: $m missing; skipped"; return 1; }; done
    wait_gpu_free; wait_ram
    ORPHAN_GUARD_EXEMPT=1 timeout -k 60 1200 "$PY" scripts/merge_splade_weights.py --models "$@" --output "models/$name" > "$LOGS/soup_$name.log" 2>&1 \
        && log "soup $name written ($*, uniform)" || { log "soup $name FAILED (logs/soup_$name.log)"; rm -rf "models/$name"; return 1; }
}
pack_sums() {  # SHA256SUMS over every file of the package
    ( cd "$PACK" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS )
}
pack_add() {  # pack_add DEST SRC NOTE: copy a directory or file into the package (replacing DEST), one line in CONTENTS.tsv, new sums
    local dest="$1" src="$2" note="$3"
    [ -e "$src" ] || { log "  package: $src missing, $dest not added"; return 1; }
    mkdir -p "$PACK/$(dirname "$dest")" && rm -rf "$PACK/$dest" && cp -r "$src" "$PACK/$dest" || { log "  package: copying $src FAILED"; return 1; }
    touch "$PACK/CONTENTS.tsv"; grep -v -P "^\Q$dest\E\t" "$PACK/CONTENTS.tsv" > "$PACK/CONTENTS.tsv.tmp"; mv "$PACK/CONTENTS.tsv.tmp" "$PACK/CONTENTS.tsv"
    printf '%s\t%s\t%s\t%s\n' "$dest" "$(du -sh "$PACK/$dest" | cut -f1)" "$src" "$note" >> "$PACK/CONTENTS.tsv"
    pack_sums; log "  package: $PACK/$dest <- $src ($note)"
}
