#!/usr/bin/env bash
# Plot-level evaluation across all twelve released cells.
# Batch size explicit and constant: evaluation is batch-size dependent
# (docs/REPRODUCTION_GATE.md section 2).
set -u
# Interpreter resolution, in order: $SICKLE_PY, the project's own .venv,
# then whatever `python` is on PATH. The project is self-contained; it must not
# depend on an environment belonging to a different project.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SICKLE_PY:-$ROOT/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
BS=32
for r in "[S2]_unet3d" "[S1]_unet3d" "[L8]_unet3d" "[S2,L8,S1]_unet3d" \
         "[S2]_utae" "[S1]_utae" "[L8]_utae" "[S2,L8,S1]_utae" \
         "[S2]_convlstm" "[S1]_convlstm" "[L8]_convlstm" "[S2,L8,S1]_convlstm"; do
  printf "%-24s " "$r"
  PYTHONIOENCODING=utf-8 "$PY" scripts/plot_level_eval.py --run "$r" --batch-size $BS \
    2>&1 | grep -E "^   (macro_iou|plot_macro_iou) " | tr -s ' ' | tr '\n' ' '
  echo
done
echo "PLOT-LEVEL SWEEP COMPLETE (batch_size=$BS)"
