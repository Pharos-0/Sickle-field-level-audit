#!/usr/bin/env bash
# Progress of the running v1 job, at a glance.
cd "$(dirname "$0")/.."
L=outputs/logs/v1_seed0.log
echo "epochs done : $(grep -cE '^ep ' $L)"
echo "NaN batches : $(grep -c 'non-finite' $L)"
echo "last epochs :"; grep -E '^ep ' $L | tail -5
echo "best so far :"; [ -f outputs/crop_type/v1_seed0/best.json ] && \
  python -c "import json;d=json.load(open('outputs/crop_type/v1_seed0/best.json'));print('  epoch',d.get('best_epoch'),'macro_iou',round(d.get('macro_iou',0),4),'paddy',round(d.get('paddy_iou',0),4))" 2>/dev/null || echo "  (best.json not written yet)"
