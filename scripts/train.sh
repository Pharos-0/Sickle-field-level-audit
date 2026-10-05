#!/usr/bin/env bash
# Train one ladder rung at one seed.
#   bash scripts/train.sh v1 0 [extra args...]
set -euo pipefail
CFG="${1:?usage: train.sh <config> [seed]}"
SEED="${2:-0}"
shift 2 2>/dev/null || shift 1
# Interpreter resolution, in order: $SICKLE_PY, the project's own .venv,
# then whatever `python` is on PATH. The project is self-contained; it must not
# depend on an environment belonging to a different project.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SICKLE_PY:-$ROOT/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
cd "$(dirname "$0")/.."
PYTHONIOENCODING=utf-8 "$PY" -m src.train --config "$CFG" --seed "$SEED" "$@"
