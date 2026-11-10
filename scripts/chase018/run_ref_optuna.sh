#!/bin/bash
# New apo and fasc reference. Four GPUs search, then one GPU trains the winner.
# MultExp is held out inside the trainer. No Kaggle submit.
set -euo pipefail
cd /home/chocheng9/kaggle_UMUD
export PYTHONPATH=src
PY=.venv/bin/python
SCRIPT=scripts/chase018/train_ref_optuna.py
LOG=reports/phase3/chase018/ref_optuna
mkdir -p "$LOG" reports/phase3/chase018/seg_ckpts

echo "[ref] warming ImageNet encoder weights"
$PY -c "import segmentation_models_pytorch as smp; smp.Unet('resnet34', encoder_weights='imagenet', in_channels=1, classes=1); print('encoder_ready')"

for task in fasc apo; do
  echo "[ref] study $task"
  pids=()
  for gpu in 0 1 2 3; do
    $PY "$SCRIPT" --task "$task" --mode study --device "cuda:${gpu}" \
      --trials 2 --trial-epochs 5 --study-name "ref_${task}" \
      > "$LOG/${task}_gpu${gpu}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[ref] study failed for $task" >&2
    exit 1
  fi
  echo "[ref] final $task"
  $PY "$SCRIPT" --task "$task" --mode final --device cuda:0 \
    --final-epochs 20 --study-name "ref_${task}" \
    > "$LOG/${task}_final.log" 2>&1
done
echo "[ref] done"
