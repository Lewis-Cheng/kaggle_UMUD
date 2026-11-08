#!/bin/bash
set -e
cd ~/kaggle_UMUD
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONPATH=$HOME/kaggle_UMUD/src
echo "PY=$PY"
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
$PY scripts/baselines/fasc_fv_stages.py --check-aux1 \
  --ckpt reports/phase3/seg_ckpts/seg_fasc_v5_best.pt \
  --eval-json reports/phase3/eval_seg_fasc_v5_best_native.json \
  --out reports/phase3/fasc_fv_status.json
echo HARNESS_EXIT:$?