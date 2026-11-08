#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/kaggle_UMUD}"
DATA="${DATA:-$ROOT/data/raw/umud}"
PY="${PY:-/home/chocheng9/images_reduced/.venv/bin/python}"
export PYTHONPATH="$ROOT/src"
SPLIT="$ROOT/reports/phase2/split_assignments.jsonl"
SCALE="$ROOT/reports/phase3/ambrosm_scale_v2_rows.csv"
OUT="$ROOT/reports/phase3"
CKPT="$ROOT/reports/phase3/seg_ckpts"
PRED="$ROOT/reports/phase3/seg_preds"
DEVICE="${DEVICE:-cuda:1}"

mkdir -p "$CKPT" "$PRED" "$OUT"
cd "$ROOT"

echo "== residual scale audit =="
"$PY" scripts/audits/residual_scale_audit.py \
  --scale-csv "$SCALE" --data-root "$DATA" \
  --out-json "$OUT/residual_scale_audit.json" \
  --out-md "$OUT/residual_scale_audit.md"

echo "== expert GT hunt =="
"$PY" scripts/audits/hunt_expert_gt.py \
  --data-root "$DATA" --repo-root "$ROOT" \
  --out-json "$OUT/expert_gt_hunt.json" \
  --out-md "$OUT/expert_gt_hunt.md"

echo "== train apo seg =="
"$PY" scripts/baselines/train_seg.py \
  --data-root "$DATA" --split-assignments "$SPLIT" \
  --task apo --out-dir "$CKPT" --epochs 8 --batch-size 16 \
  --device "$DEVICE" --num-workers 4

echo "== train fasc seg =="
"$PY" scripts/baselines/train_seg.py \
  --data-root "$DATA" --split-assignments "$SPLIT" \
  --task fasc --out-dir "$CKPT" --epochs 8 --batch-size 16 \
  --device "$DEVICE" --num-workers 4

echo "== infer val+test masks =="
"$PY" scripts/baselines/infer_seg.py \
  --data-root "$DATA" \
  --apo-ckpt "$CKPT/seg_apo_best.pt" \
  --fasc-ckpt "$CKPT/seg_fasc_best.pt" \
  --out-dir "$PRED" \
  --splits val test \
  --split-assignments "$SPLIT" \
  --device "$DEVICE"

echo "== tier B seg GH =="
"$PY" scripts/baselines/tier_b_seg_gh.py \
  --data-root "$DATA" \
  --split-assignments "$SPLIT" \
  --ambrosm-scale-csv "$SCALE" \
  --pred-mask-dir "$PRED" \
  --out-dir "$OUT" \
  --artifact-prefix tier_b_seg_gh \
  --scale-provenance ambrosm_per_family_mixed_v2_hflip

echo "DONE pipeline"
