#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 RUN_DIR REPO TAG ASSET_PREFIX [PROCESS_PATTERN] [INTERVAL_SECONDS]" >&2
  exit 1
fi

RUN_DIR="$1"
REPO="$2"
TAG="$3"
ASSET_PREFIX="$4"
PROCESS_PATTERN="${5:-}"
INTERVAL_SECONDS="${6:-600}"
STATE_FILE="$RUN_DIR/.last_uploaded_ckpt"
LOG_FILE="$RUN_DIR/upload_watch.log"

mkdir -p "$RUN_DIR"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG_FILE"
}

latest_ckpt() {
  find "$RUN_DIR" -maxdepth 1 -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null \
    | grep -E '/(last|epoch=[0-9]+-step=[0-9]+(-v[0-9]+)?)\.ckpt$' \
    | sort -n | tail -n1 | cut -d' ' -f2-
}

release_exists() {
  gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1
}

ensure_release() {
  if release_exists; then
    return
  fi

  gh release create "$TAG" --repo "$REPO" --title "$TAG" --notes 'CDVAE MP-20 model trained on A100.' >/dev/null
  log "created release $TAG"
}

upload_checkpoint() {
  local checkpoint_path="$1"
  local checkpoint_name
  checkpoint_name="$(basename "$checkpoint_path" .ckpt)"
  local unique_asset="$RUN_DIR/${ASSET_PREFIX}-${checkpoint_name}.tar.gz"
  local latest_asset="$RUN_DIR/${ASSET_PREFIX}-latest.tar.gz"

  python scripts/package_and_upload_github_release.py "$RUN_DIR" --repo "$REPO" --tag "$TAG" --release-name "$TAG" --asset-name "$(basename "$unique_asset")" --checkpoint "$checkpoint_path" --notes 'CDVAE MP-20 model trained on A100.' >>"$LOG_FILE" 2>&1
  cp -f "$unique_asset" "$latest_asset"

  ensure_release
  gh release upload "$TAG" "$unique_asset" --repo "$REPO" --clobber >>"$LOG_FILE" 2>&1
  gh release upload "$TAG" "$latest_asset" --repo "$REPO" --clobber >>"$LOG_FILE" 2>&1

  printf '%s\n' "$checkpoint_path" >"$STATE_FILE"
  log "uploaded $(basename "$unique_asset") and $(basename "$latest_asset")"
}

last_uploaded=''
if [[ -f "$STATE_FILE" ]]; then
  last_uploaded="$(cat "$STATE_FILE")"
fi

log "watcher started interval=${INTERVAL_SECONDS}s"

while true; do
  current_ckpt="$(latest_ckpt || true)"
  if [[ -n "$current_ckpt" && "$current_ckpt" != "$last_uploaded" ]]; then
    upload_checkpoint "$current_ckpt"
    last_uploaded="$current_ckpt"
  fi

  if [[ -n "$PROCESS_PATTERN" ]] && ! pgrep -f "$PROCESS_PATTERN" >/dev/null 2>&1; then
    current_ckpt="$(latest_ckpt || true)"
    if [[ -z "$current_ckpt" || "$current_ckpt" == "$last_uploaded" ]]; then
      log "watcher exiting: process ended and latest checkpoint is uploaded"
      exit 0
    fi
  fi

  sleep "$INTERVAL_SECONDS"
done