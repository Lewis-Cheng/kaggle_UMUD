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
mkdir -p "$CKPT" "$PRED"

echo "== diag fasc =="
"$PY" scripts/audits/diag_fasc_seg.py \
  --data-root "$DATA" --split "$SPLIT" --scale-csv "$SCALE" \
  --out "$OUT/diag_fasc_seg.json" --n 120 || true

echo "== re-hunt expert GT =="
"$PY" scripts/audits/hunt_expert_gt.py \
  --data-root "$DATA" --repo-root "$ROOT" \
  --out-json "$OUT/expert_gt_hunt.json" \
  --out-md "$OUT/expert_gt_hunt.md"

# broader search outside data-root
echo "== broader GT search =="
"$PY" - <<'PY'
import json
from pathlib import Path
hits=[]
roots=[Path.home()/"kaggle_UMUD", Path.home()/"Downloads", Path.home()/"data"]
for root in roots:
  if not root.exists():
    continue
  for p in root.rglob("*"):
    if not p.is_file():
      continue
    if p.suffix.lower() not in {".csv",".xlsx",".tsv"}: continue
    if p.stat().st_size>80_000_000: continue
    try:
      txt=p.read_text(encoding="utf-8", errors="ignore")[:30000]
    except Exception:
      continue
    if "pa_deg" in txt and "fl_mm" in txt and "mt_mm" in txt:
      hits.append(str(p))
print(json.dumps({"broader_hits": hits[:50], "n": len(hits)}, indent=2))
Path("/home/chocheng9/kaggle_UMUD/reports/phase3/expert_gt_broader.json").write_text(
  json.dumps({"broader_hits": hits, "n": len(hits)}, indent=2)
)
PY

echo "== train fasc v3 =="
"$PY" scripts/baselines/train_seg_fasc_v3.py \
  --data-root "$DATA" --split-assignments "$SPLIT" --scale-csv "$SCALE" \
  --out-dir "$CKPT" --epochs 30 --batch-size 8 --size 512 --base-ch 48 \
  --dilate 3 --device "$DEVICE" --num-workers 4

echo "== infer v3 =="
"$PY" scripts/baselines/infer_seg_v3.py \
  --data-root "$DATA" \
  --apo-ckpt "$CKPT/seg_apo_best.pt" \
  --fasc-ckpt "$CKPT/seg_fasc_best.pt" \
  --out-dir "$PRED" \
  --split-assignments "$SPLIT" \
  --scale-csv "$SCALE" \
  --splits val test \
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

echo DONE fasc_v3
