#!/bin/bash
set -euo pipefail
ROOT=$HOME/kaggle_UMUD
OUT=$ROOT/reports/phase3
CKPT=$OUT/seg_ckpts
PY=/home/chocheng9/images_reduced/.venv/bin/python
cd "$ROOT"
export PYTHONPATH=$ROOT/src
DEVICE=cuda:3
DATA=data/raw/umud
SPLIT=reports/phase2/split_assignments.jsonl

echo "[$(date -Iseconds)] start v6 HR MSTinyUNet"
$PY scripts/baselines/train_seg_fasc_v6.py \
  --data-root $DATA --split-assignments $SPLIT --out-dir $CKPT \
  --epochs 30 --batch-size 4 --size 640 --base-ch 48 \
  --edt-radius 3.0 --pos-weight 40 --cldice-w 0.75 \
  --device $DEVICE --num-workers 4 \
  --init-from $CKPT/seg_fasc_v5_best.pt \
  2>&1 | tee -a $OUT/run_fasc_v6.log

if pgrep -f "train_seg_fasc_v6b\.py" >/dev/null; then
  echo "[$(date -Iseconds)] v6b already running — wait"
  while pgrep -f "train_seg_fasc_v6b\.py" >/dev/null; do sleep 60; done
elif [[ -f $CKPT/seg_fasc_v6b_best.pt && -f $CKPT/seg_fasc_v6b_train.json ]]; then
  echo "[$(date -Iseconds)] v6b already finished — skip retrain"
else
  echo "[$(date -Iseconds)] start v6b hrnet"
  $PY scripts/baselines/train_seg_fasc_v6b.py     --data-root $DATA --split-assignments $SPLIT --out-dir $CKPT     --epochs 30 --batch-size 6 --size 512 --encoder hrnet_w18     --edt-radius 2.5 --pos-weight 35 --cldice-w 0.5     --device $DEVICE --num-workers 4     2>&1 | tee -a $OUT/run_fasc_v6b.log
fi

echo "[$(date -Iseconds)] v6 chain train done — eval + GH + verdict"
for ck in seg_fasc_v4b_best.pt seg_fasc_v5_best.pt seg_fasc_v5b_best.pt seg_fasc_v6_best.pt seg_fasc_v6b_best.pt; do
  if [[ -f $CKPT/$ck ]]; then
    base=${ck%.pt}
    $PY scripts/audits/eval_fasc_ckpt_native.py \
      --data-root $DATA --split-assignments $SPLIT \
      --ckpt $CKPT/$ck --device $DEVICE \
      --out $OUT/eval_${base}_native.json || true
  fi
done

$PY - <<'PY'
import json
from pathlib import Path
out=Path("reports/phase3")
ranked=[]
for p in sorted(out.glob("eval_seg_fasc*_native.json")):
    d=json.loads(p.read_text())
    best=-1.0; detail=None
    for mode in ("square_resize_native","letterbox_native"):
        for thr,rec in (d.get(mode) or {}).items():
            sc=float(rec.get("macro_dice",-1))
            if sc>best:
                best=sc; detail={"mode":mode,"thr":float(thr),**rec}
    ranked.append({"eval_path":str(p),"ckpt":d.get("ckpt"),"variant":d.get("variant"),
                   "best_native_macro_dice":best,"best_detail":detail,"size":d.get("size"),"arch":d.get("arch")})
ranked.sort(key=lambda x:x["best_native_macro_dice"], reverse=True)
payload={"baseline_v4b":0.2676,"ranked":ranked,"winner":ranked[0] if ranked else None}
(out/"fasc_native_rank.json").write_text(json.dumps(payload,indent=2))
print(json.dumps({"winner":payload["winner"],"top":[(r["variant"], round(r["best_native_macro_dice"],4)) for r in ranked[:10]]}, indent=2))
PY

WIN=$($PY -c 'import json;print(json.load(open("reports/phase3/fasc_native_rank.json"))["winner"]["ckpt"])')
PRED=$OUT/seg_preds_best_fasc_latest
rm -rf $PRED; mkdir -p $PRED
$PY scripts/baselines/infer_seg_letterbox.py \
  --data-root $DATA --apo-ckpt $CKPT/seg_apo_best.pt --fasc-ckpt "$WIN" \
  --out-dir $PRED --split-assignments $SPLIT --splits val test --device $DEVICE
$PY scripts/baselines/tier_b_seg_gh.py \
  --data-root $DATA --split-assignments $SPLIT --ambrosm-scale-csv $OUT/ambrosm_scale_v2_rows.csv \
  --pred-mask-dir $PRED --out-dir $OUT --artifact-prefix tier_b_seg_gh_latest \
  --scale-provenance ambrosm_per_family_mixed_v2_hflip
ls -la $OUT/submission_tier_b_seg_gh_latest.csv $OUT/tier_b_seg_gh_latest_metrics.json 2>/dev/null || true
$PY scripts/baselines/write_attention_gate_verdict.py \
  --rank-json $OUT/fasc_native_rank.json \
  --metrics-json $OUT/tier_b_seg_gh_latest_metrics.json \
  --out $OUT/attention_gate_verdict.md \
  --fasc-winner-ckpt "$WIN"

# Hard-stop note if still plateau
$PY - <<'PY'
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
rank=json.loads(Path("reports/phase3/fasc_native_rank.json").read_text())
dice=float(rank["winner"]["best_native_macro_dice"])
note=Path("reports/phase3/fasc_hard_stop.md")
if dice < 0.40:
    text=f"""# Fasc hard-stop candidacy

**Date:** {datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M HKT')}
**Winner Dice:** {dice:.4f} (bar ~0.40 for usable MT/geometry)
**Official S:** null (no expert GT)
**attention_blocked:** YES

After v5 (MS+clDice), v5b (resnet18), v6 (HR 640), v6b (hrnet): fasc still below usable bar.
Do not open attention training. Optional remaining levers: SAM install, apo-boundary refine, tiny held-out for S — otherwise document plateau + S=null.
"""
    note.write_text(text)
    print(text)
else:
    print("dice above 0.40 — see verdict; still blocked without expert GT")
PY
echo "[$(date -Iseconds)] v6 chain DONE"
