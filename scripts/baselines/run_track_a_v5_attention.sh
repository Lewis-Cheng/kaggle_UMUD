#!/usr/bin/env bash
# Track A: wait for v5 train-mask infer, train attention residual, infer test, write CSV.
set -euo pipefail
ROOT="${ROOT:-$HOME/kaggle_UMUD}"
cd "$ROOT"
PY="${PY:-/home/chocheng9/images_reduced/.venv/bin/python}"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
OUT="$ROOT/reports/phase3"
LOG="$OUT/run_track_a_v5_attention.log"
exec > >(tee -a "$LOG") 2>&1

DATA="$ROOT/data/raw/umud"
SPLIT="$ROOT/reports/phase2/split_assignments.jsonl"
SCALE="$OUT/ambrosm_scale_v2_rows.csv"
PRED_TRAIN="$OUT/seg_preds_v5_train"
PRED_VALTEST="$OUT/seg_preds_best_fasc_v5"
CKPT_DIR="$OUT/attn_ckpts"
DEVICE="${DEVICE:-cuda:1}"

echo "[$(date -Iseconds)] Track A start device=$DEVICE"

# Wait for train infer to finish (pid file or process)
echo "[$(date -Iseconds)] waiting for train mask infer under $PRED_TRAIN"
for i in $(seq 1 180); do
  if ! pgrep -f 'infer_seg_letterbox.py.*seg_preds_v5_train' >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] infer process gone"
    break
  fi
  n=$(ls "$PRED_TRAIN/pred_fasc_masks" 2>/dev/null | wc -l || echo 0)
  echo "[$(date -Iseconds)] waiting... fasc_masks=$n"
  sleep 30
done

# Ensure val+test preds exist; if missing, re-infer quickly
if [[ ! -d "$PRED_VALTEST/pred_fasc_masks" ]]; then
  echo "missing val/test preds — inferring"
  $PY scripts/baselines/infer_seg_letterbox.py \
    --data-root "$DATA" --split-assignments "$SPLIT" \
    --apo-ckpt "$OUT/seg_ckpts/seg_apo_best.pt" \
    --fasc-ckpt "$OUT/seg_ckpts/seg_fasc_v5_best.pt" \
    --out-dir "$PRED_VALTEST" --splits val test --device "$DEVICE"
fi

n_train=$(ls "$PRED_TRAIN/pred_fasc_masks" 2>/dev/null | wc -l || echo 0)
echo "[$(date -Iseconds)] train fasc preds=$n_train"
if [[ "$n_train" -lt 500 ]]; then
  echo "ERROR: train preds incomplete ($n_train)"; exit 1
fi

mkdir -p "$CKPT_DIR"
echo "[$(date -Iseconds)] train attention"
$PY scripts/baselines/train_attention_v5.py \
  --data-root "$DATA" \
  --split-assignments "$SPLIT" \
  --ambrosm-scale-csv "$SCALE" \
  --pred-train-dir "$PRED_TRAIN" \
  --pred-val-dir "$PRED_VALTEST" \
  --out-dir "$CKPT_DIR" \
  --device "$DEVICE" \
  --epochs 25 \
  --batch-size 16 \
  --size 256 \
  --num-workers 4

echo "[$(date -Iseconds)] infer test"
SUB="$OUT/submission_trackA_v5_attention.csv"
$PY scripts/baselines/infer_attention_v5.py \
  --data-root "$DATA" \
  --ambrosm-scale-csv "$SCALE" \
  --pred-mask-dir "$PRED_VALTEST" \
  --ckpt "$CKPT_DIR/attn_v5_best.pt" \
  --out-csv "$SUB" \
  --device "$DEVICE" \
  --delimiter ';'

# Validate ID set vs sample
$PY scripts/validate_submission.py "$SUB" \
  --expected-ids <($PY - <<'PY'
from pathlib import Path
import csv
p=Path("data/raw/umud/sample_submission.csv")
raw=p.read_text(encoding="utf-8-sig")
lines=[ln for ln in raw.splitlines() if ln.strip()]
delim=";" if lines[0].count(";")>=lines[0].count(",") else ","
r=csv.DictReader(lines, delimiter=delim)
ids=[row.get("image_id") or row.get("id") or next(iter(row.values())) for row in r]
Path("/tmp/umud_expected_ids.txt").write_text("\n".join(ids)+"\n")
print("/tmp/umud_expected_ids.txt")
PY
) || true

# rewrite validate with explicit file
$PY - <<'PY'
from pathlib import Path
import csv, sys
sys.path.insert(0,"src")
from umud.metric.submission import validate_submission, load_expected_ids
import pandas as pd
p=Path("data/raw/umud/sample_submission.csv")
raw=p.read_text(encoding="utf-8-sig")
lines=[ln for ln in raw.splitlines() if ln.strip()]
delim=";" if lines[0].count(";")>=lines[0].count(",") else ","
r=csv.DictReader(lines, delimiter=delim)
ids=[(row.get("image_id") or row.get("id") or next(iter(row.values()))).strip() for row in r]
Path("/tmp/umud_expected_ids.txt").write_text("\n".join(ids)+"\n")
df=pd.read_csv("reports/phase3/submission_trackA_v5_attention.csv", sep=";")
# validator expects id col
try:
    from umud.metric.submission import SubmissionValidationErro
    validate_submission(df, ids)
    print("OK validate", len(ids))
except Exception as e:
    # try rename
    d2=df.rename(columns={"image_id":"id"})
    try:
        validate_submission(d2, ids)
        print("OK validate via id rename", len(ids))
    except Exception as e2:
        print("VALIDATE_FAIL", e2)
        raise
PY

echo "[$(date -Iseconds)] Track A artifacts ready: $SUB"
ls -la "$SUB" "$CKPT_DIR/attn_v5_best.pt"
