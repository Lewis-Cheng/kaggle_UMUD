#!/bin/bash
set -e
cd /home/chocheng9/kaggle_UMUD
cp reports/phase3/chase018/multexp_expdrawn_frundles.json reports/phase3/chase018/multexp_expdrawn_frundles_mix309.json
cp reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_expdrawn_frundles_best.pt reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_expdrawn_frundles_mix309_best.pt
exec /home/chocheng9/kaggle_UMUD/.venv/bin/python -u scripts/chase018/train_fasc_expdrawn_frundles.py --device cuda:1 --epochs 10 --batch 8 --lr 5e-5 --mix-labtrain 0 --dilate-r 1 --skip-extract --cache-dir reports/phase3/chase018/cache/maps_multexp_expdrawn_only
