#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export HYDRA_JOBS="${HYDRA_JOBS:-$ROOT_DIR/hydra_runs}"
export WABDB_DIR="${WABDB_DIR:-$ROOT_DIR/wandb_runs}"

mkdir -p "$HYDRA_JOBS" "$WABDB_DIR"

cd "$ROOT_DIR"

python -m cdvae.run \
  data=mp_20 \
  expname=local_mp20 \
  logging.wandb.mode=offline \
  "$@"
