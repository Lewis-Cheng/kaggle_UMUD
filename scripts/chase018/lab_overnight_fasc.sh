#!/bin/bash
set -euo pipefail
cd ~/kaggle_UMUD
mkdir -p reports/phase3/chase018/seg_ckpts
PY=.venv/bin/python

tmux kill-session -t umud_fasc_overnight 2>/dev/null || true
tmux new-session -d -s umud_fasc_overnight bash -lc 'cd ~/kaggle_UMUD && .venv/bin/python -u scripts/chase018/train_fasc_smp_ft.py --img-dir data/raw/umud/fasc_imgs_v1 --mask-dir data/raw/umud/fasc_masks_v1 --init reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_scratch_best.pt --epochs 40 --batch 8 --lr 1e-4 --cldice-w 0.5 --device cuda:0 --out-dir reports/phase3/chase018/seg_ckpts 2>&1 | tee reports/phase3/chase018/fasc_overnight_ft_cldice.log'

tmux kill-session -t umud_fasc_scratch60 2>/dev/null || true
tmux new-session -d -s umud_fasc_scratch60 bash -lc 'cd ~/kaggle_UMUD && .venv/bin/python -u scripts/chase018/train_fasc_frundles_recipe.py --img-dir data/raw/umud/fasc_imgs_v1 --mask-dir data/raw/umud/fasc_masks_v1 --epochs 60 --bs 8 --lr 3e-4 --pos-weight 6.0 --seed 7 --device cuda:1 --tag fasc_resnet34_384x576_scratch60_s7 2>&1 | tee reports/phase3/chase018/fasc_overnight_scratch60.log'

sleep 3
tmux ls
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo FT_LOG:
tail -n 8 reports/phase3/chase018/fasc_overnight_ft_cldice.log || true
echo S60_LOG:
tail -n 8 reports/phase3/chase018/fasc_overnight_scratch60.log || true
