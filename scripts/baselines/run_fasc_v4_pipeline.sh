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
PRED="$OUT/seg_preds_v4"
DEVICE="${DEVICE:-cuda:2}"
cd "$ROOT"
mkdir -p "$CKPT" "$PRED"

echo "== diagnose shapes =="
"$PY" scripts/audits/audit_fasc_shapes.py | tee "$OUT/audit_fasc_shapes.txt" || true

echo "== re-eval prior fasc best (native macro, resize-align) =="
"$PY" scripts/audits/eval_fasc_ckpt_native.py \
  --data-root "$DATA" --split-assignments "$SPLIT" \
  --ckpt "$CKPT/seg_fasc_best.pt" --device "$DEVICE" \
  --out "$OUT/eval_fasc_best_native.json" || true

if [[ -f "$CKPT/seg_fasc_v2_recover_best.pt" ]]; then
  "$PY" scripts/audits/eval_fasc_ckpt_native.py \
    --data-root "$DATA" --split-assignments "$SPLIT" \
    --ckpt "$CKPT/seg_fasc_v2_recover_best.pt" --device "$DEVICE" \
    --out "$OUT/eval_fasc_recover_native.json" || true
fi

echo "== train fasc v4 (align mask→img then letterbox) =="
"$PY" scripts/baselines/train_seg_fasc_v4.py \
  --data-root "$DATA" --split-assignments "$SPLIT" \
  --out-dir "$CKPT" --epochs 40 --batch-size 10 --size 512 \
  --base-ch 32 --dilate 2 --pos-weight 45 --device "$DEVICE" --num-workers 4

echo "== eval v4 native =="
"$PY" scripts/audits/eval_fasc_ckpt_native.py \
  --data-root "$DATA" --split-assignments "$SPLIT" \
  --ckpt "$CKPT/seg_fasc_v4_best.pt" --device "$DEVICE" \
  --out "$OUT/eval_fasc_v4_native.json"

echo "== infer v4 =="
"$PY" scripts/baselines/infer_seg_letterbox.py \
  --data-root "$DATA" \
  --apo-ckpt "$CKPT/seg_apo_best.pt" \
  --fasc-ckpt "$CKPT/seg_fasc_v4_best.pt" \
  --out-dir "$PRED" \
  --split-assignments "$SPLIT" \
  --splits val test \
  --device "$DEVICE"

echo "== tier B seg GH v4 =="
"$PY" scripts/baselines/tier_b_seg_gh.py \
  --data-root "$DATA" \
  --split-assignments "$SPLIT" \
  --ambrosm-scale-csv "$SCALE" \
  --pred-mask-dir "$PRED" \
  --out-dir "$OUT" \
  --artifact-prefix tier_b_seg_gh_v4 \
  --scale-provenance ambrosm_per_family_mixed_v2_hflip

echo DONE fasc_v4
