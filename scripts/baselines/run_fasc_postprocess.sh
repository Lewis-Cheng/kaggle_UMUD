#!/bin/bash
set -euo pipefail
ROOT="$HOME/kaggle_UMUD"
cd "$ROOT"
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONPATH="$ROOT/src"
OUT="$ROOT/reports/phase3"
CKPT="$OUT/seg_ckpts"
DATA="$ROOT/data/raw/umud"
SPLIT="$ROOT/reports/phase2/split_assignments.jsonl"
SCALE="$OUT/ambrosm_scale_v2_rows.csv"
DEVICE="${DEVICE:-cuda:0}"
LOG="$OUT/run_fasc_postprocess.log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Iseconds)] postprocess resume (skip failing evals)"

# optional: try v4c if present; do not abort on failure
if [[ -f "$CKPT/seg_fasc_v4c_best.pt" && ! -f "$OUT/eval_seg_fasc_v4c_best_native.json" ]]; then
  "$PY" scripts/audits/eval_fasc_ckpt_native.py \
    --data-root "$DATA" --split-assignments "$SPLIT" \
    --ckpt "$CKPT/seg_fasc_v4c_best.pt" --device "$DEVICE" \
    --out "$OUT/eval_seg_fasc_v4c_best_native.json" || echo "v4c eval failed (ignored)"
fi

"$PY" - <<'PY'
import json
from pathlib import Path
out = Path("reports/phase3")
ranked = []
for p in sorted(out.glob("eval_seg_fasc*_native.json")):
    d = json.loads(p.read_text())
    best = -1.0
    best_mode = None
    for mode in ("square_resize_native", "letterbox_native"):
        block = d.get(mode) or {}
        for thr, rec in block.items():
            sc = float(rec.get("macro_dice", -1))
            if sc > best:
                best = sc
                best_mode = {"mode": mode, "thr": float(thr), **rec}
    ranked.append({
        "eval_path": str(p),
        "ckpt": d.get("ckpt"),
        "variant": d.get("variant"),
        "stored_val_dice": d.get("stored_val_dice"),
        "best_native_macro_dice": best,
        "best_detail": best_mode,
        "size": d.get("size"),
        "arch": d.get("arch"),
    })
ranked.sort(key=lambda x: x["best_native_macro_dice"], reverse=True)
payload = {"baseline_prior_best": 0.2633, "ranked": ranked, "winner": ranked[0] if ranked else None}
(out / "fasc_native_rank.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
if not ranked:
    raise SystemExit("no evals")
print("WINNER_CKPT=" + ranked[0]["ckpt"])
print("WINNER_DICE=%.6f" % ranked[0]["best_native_macro_dice"])
PY

WINNER_CKPT=$("$PY" -c 'import json;print(json.load(open("reports/phase3/fasc_native_rank.json"))["winner"]["ckpt"])')
WINNER_DICE=$("$PY" -c 'import json;print(json.load(open("reports/phase3/fasc_native_rank.json"))["winner"]["best_native_macro_dice"])')
echo "Selected $WINNER_CKPT dice=$WINNER_DICE"
PRED="$OUT/seg_preds_best_fasc"
rm -rf "$PRED"; mkdir -p "$PRED"
"$PY" scripts/baselines/infer_seg_letterbox.py \
  --data-root "$DATA" --apo-ckpt "$CKPT/seg_apo_best.pt" --fasc-ckpt "$WINNER_CKPT" \
  --out-dir "$PRED" --split-assignments "$SPLIT" --splits val test --device "$DEVICE"
"$PY" scripts/baselines/tier_b_seg_gh.py \
  --data-root "$DATA" --split-assignments "$SPLIT" --ambrosm-scale-csv "$SCALE" \
  --pred-mask-dir "$PRED" --out-dir "$OUT" --artifact-prefix tier_b_seg_gh_v4best \
  --scale-provenance ambrosm_per_family_mixed_v2_hflip
echo "[$(date -Iseconds)] postprocess DONE"