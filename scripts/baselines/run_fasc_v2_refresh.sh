#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/kaggle_UMUD"
DATA="$ROOT/data/raw/umud"
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONPATH="$ROOT/src"
SPLIT="$ROOT/reports/phase2/split_assignments.jsonl"
SCALE="$ROOT/reports/phase3/ambrosm_scale_v2_rows.csv"
OUT="$ROOT/reports/phase3"
CKPT="$OUT/seg_ckpts"
PRED="$OUT/seg_preds"
DEVICE=cuda:1
cd "$ROOT"

echo "== fasc v2 train =="
"$PY" scripts/baselines/train_seg_fasc_v2.py \
  --data-root "$DATA" --split-assignments "$SPLIT" \
  --out-dir "$CKPT" --epochs 20 --batch-size 12 --size 384 \
  --device "$DEVICE"

echo "== re-infer =="
"$PY" scripts/baselines/infer_seg.py \
  --data-root "$DATA" \
  --apo-ckpt "$CKPT/seg_apo_best.pt" \
  --fasc-ckpt "$CKPT/seg_fasc_best.pt" \
  --out-dir "$PRED" \
  --splits val test \
  --split-assignments "$SPLIT" \
  --device "$DEVICE" --thr 0.35

echo "== tier B seg GH refresh =="
"$PY" scripts/baselines/tier_b_seg_gh.py \
  --data-root "$DATA" \
  --split-assignments "$SPLIT" \
  --ambrosm-scale-csv "$SCALE" \
  --pred-mask-dir "$PRED" \
  --out-dir "$OUT" \
  --artifact-prefix tier_b_seg_gh \
  --scale-provenance ambrosm_per_family_mixed_v2_hflip

echo DONE fasc_v2_refresh
