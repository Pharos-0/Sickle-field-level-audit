#!/usr/bin/env bash
# Train one ladder rung at one seed.
#   bash scripts/train.sh v1 0 [extra args...]
set -euo pipefail
CFG="${1:?usage: train.sh <config> [seed]}"
SEED="${2:-0}"
shift 2 2>/dev/null || shift 1
PY="${SICKLE_PY:-/d/Final Project M.Tech/.venv/Scripts/python.exe}"
cd "$(dirname "$0")/.."
PYTHONIOENCODING=utf-8 "$PY" -m src.train --config "$CFG" --seed "$SEED" "$@"
