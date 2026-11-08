#!/bin/bash
set -euo pipefail
ROOT=$HOME/kaggle_UMUD
cd $ROOT
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONPATH=$ROOT/src
# sync already done via scp into place by caller
mkdir -p reports/phase3/seg_ckpts

# smoke one batch
$PY - <<'PY'
import torch
from umud.seg.ms_unet import MSTinyUNet
from umud.seg.timm_unet import TimmUNet
m=MSTinyUNet(1,48)
x=torch.randn(2,1,512,512)
y,a3,a2=m(x,return_aux=True)
assert y.shape==a3.shape==a2.shape==(2,1,512,512)
m2=TimmUNet('resnet18',32,pretrained=True)
y2=m2(torch.randn(1,1,384,384))
print('smoke ok', y.shape, y2.shape)
PY

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=$ROOT/src $PY scripts/baselines/train_seg_fasc_v5.py \
  --data-root data/raw/umud --split-assignments reports/phase2/split_assignments.jsonl \
  --out-dir reports/phase3/seg_ckpts --epochs 40 --batch-size 8 --size 512 --base-ch 48 \
  --edt-radius 2.5 --pos-weight 35 --cldice-w 0.5 --device cuda:3 --num-workers 4 \
  > reports/phase3/run_fasc_v5.log 2>&1 &
echo V5_PID=$!

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=$ROOT/src $PY scripts/baselines/train_seg_fasc_v5b.py \
  --data-root data/raw/umud --split-assignments reports/phase2/split_assignments.jsonl \
  --out-dir reports/phase3/seg_ckpts --epochs 35 --batch-size 10 --size 384 \
  --encoder resnet18 --base-ch 32 --edt-radius 2.5 --pos-weight 35 --cldice-w 0.5 \
  --device cuda:2 --num-workers 4 \
  > reports/phase3/run_fasc_v5b.log 2>&1 &
echo V5B_PID=$!

sleep 8
tail -20 reports/phase3/run_fasc_v5.log || true
echo '---'
tail -20 reports/phase3/run_fasc_v5b.log || true
pgrep -af 'train_seg_fasc_v5' || true