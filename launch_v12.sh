#!/bin/bash
set -euo pipefail
cd ~/kaggle_UMUD
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONPATH=$HOME/kaggle_UMUD/src
DATA=$HOME/kaggle_UMUD/data/raw/umud
SPLIT=$HOME/kaggle_UMUD/reports/phase2/split_assignments.jsonl
OUT=$HOME/kaggle_UMUD/reports/phase3/seg_ckpts
LOG=$HOME/kaggle_UMUD/reports/phase3/run_fasc_v12.log
CKPT_V5=$OUT/seg_fasc_v5_best.pt
ls -la "$DATA" >/dev/null
ls -la "$SPLIT" "$CKPT_V5"
# free GPU check again
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# use cuda:1 — leave GPU0 free; GPU2/3 busy with other users
DEVICE=cuda:1
# kill old session if any
tmux has-session -t umud_fasc_v12 2>/dev/null && tmux kill-session -t umud_fasc_v12 || true
tmux new-session -d -s umud_fasc_v12 "cd ~/kaggle_UMUD && PYTHONPATH=$HOME/kaggle_UMUD/src $PY scripts/baselines/train_seg_fasc_v12.py \
  --data-root $DATA \
  --split-assignments $SPLIT \
  --out-dir $OUT \
  --epochs 40 \
  --batch-size 8 \
  --size 512 \
  --base-ch 48 \
  --edt-radius 2.5 \
  --cldice-w 0.5 \
  --device $DEVICE \
  --warm-start $CKPT_V5 \
  --num-workers 4 \
  2>&1 | tee $LOG"
sleep 8
tmux ls | grep v12 || true
echo "==== log head ===="
head -40 "$LOG" || true
echo "==== nvidia ===="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv