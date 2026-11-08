#!/bin/bash
set -uo pipefail
ROOT=$HOME/kaggle_UMUD; OUT=$ROOT/reports/phase3; CKPT=$OUT/seg_ckpts
PY=/home/chocheng9/images_reduced/.venv/bin/python
cd "$ROOT"; export PYTHONPATH=$ROOT/src
DATA=data/raw/umud; SPLIT=reports/phase2/split_assignments.jsonl
LOG=$OUT/watch_fasc_v8_resume.log
echo "[$(date -Iseconds)] v8 watchdog start" | tee -a "$LOG"
while true; do
  if [[ -f $CKPT/seg_fasc_v8_train.json ]]; then
    EP=$(python3 -c "import json;d=json.load(open('$CKPT/seg_fasc_v8_train.json'));h=d.get('history') or [];print(h[-1].get('epoch',0) if h else 0)")
    if [[ "${EP:-0}" -ge 35 ]]; then echo "[$(date -Iseconds)] v8 done ep=$EP" | tee -a "$LOG"; exit 0; fi
  fi
  if pgrep -f "train_seg_fasc_v8\.py" >/dev/null; then sleep 90; continue; fi
  if [[ -f $CKPT/seg_fasc_v8_last.pt ]]; then
    echo "[$(date -Iseconds)] v8 DEAD — resume" | tee -a "$LOG"
    nohup $PY scripts/baselines/train_seg_fasc_v8.py \
      --data-root $DATA --split-assignments $SPLIT --out-dir $CKPT \
      --epochs 35 --batch-size 16 --size 448 --encoder resnet34 --base-ch 32 \
      --edt-radius 3.5 --pos-weight 45 --cldice-w 1.0 --device cuda:2 --num-workers 6 --amp \
      --resume $CKPT/seg_fasc_v8_last.pt >> $OUT/run_fasc_v8.log 2>&1 &
  else
    echo "[$(date -Iseconds)] v8 missing last.pt — wait" | tee -a "$LOG"
  fi
  sleep 90
done
