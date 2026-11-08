#!/bin/bash
# Wait for v5 train, eval, seg→GH, update verdict, kick next lever if still blocked.
set -euo pipefail
ROOT=$HOME/kaggle_UMUD
OUT=$ROOT/reports/phase3
LOG=$OUT/run_fasc_v5_finish.log
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"
echo "[$(date -Iseconds)] wait v5 (v5b already done)"
while pgrep -f 'train_seg_fasc_v5\.py' >/dev/null; do
  echo "[$(date -Iseconds)] training..."
  tail -2 $OUT/run_fasc_v5.log 2>/dev/null || true
  sleep 60
done
echo "[$(date -Iseconds)] trains done — postprocess"
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONPATH=$ROOT/src
CKPT=$OUT/seg_ckpts
DATA=$ROOT/data/raw/umud
SPLIT=$ROOT/reports/phase2/split_assignments.jsonl
SCALE=$OUT/ambrosm_scale_v2_rows.csv
DEVICE=cuda:3

for ck in seg_fasc_v4b_best.pt seg_fasc_v5_best.pt seg_fasc_v5b_best.pt; do
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
(out/"fasc_native_rank_v5.json").write_text(json.dumps(payload,indent=2))
print(json.dumps({"winner":payload["winner"],"top":[(r["variant"],r["best_native_macro_dice"]) for r in ranked[:8]]},indent=2))
PY

WIN=$($PY -c 'import json;print(json.load(open("reports/phase3/fasc_native_rank.json"))["winner"]["ckpt"])')
DICE=$($PY -c 'import json;print(json.load(open("reports/phase3/fasc_native_rank.json"))["winner"]["best_native_macro_dice"])')
echo "WINNER $WIN dice=$DICE"
PRED=$OUT/seg_preds_best_fasc_v5
rm -rf $PRED; mkdir -p $PRED
$PY scripts/baselines/infer_seg_letterbox.py \
  --data-root $DATA --apo-ckpt $CKPT/seg_apo_best.pt --fasc-ckpt "$WIN" \
  --out-dir $PRED --split-assignments $SPLIT --splits val test --device $DEVICE
$PY scripts/baselines/tier_b_seg_gh.py \
  --data-root $DATA --split-assignments $SPLIT --ambrosm-scale-csv $SCALE \
  --pred-mask-dir $PRED --out-dir $OUT --artifact-prefix tier_b_seg_gh_v5best \
  --scale-provenance ambrosm_per_family_mixed_v2_hflip

# Prefer submission name for current winner
ls -la $OUT/submission_tier_b_seg_gh_v5best.csv $OUT/tier_b_seg_gh_v5best_metrics.json 2>/dev/null || true

$PY scripts/baselines/write_attention_gate_verdict.py \
  --rank-json $OUT/fasc_native_rank.json \
  --metrics-json $OUT/tier_b_seg_gh_v5best_metrics.json \
  --out $OUT/attention_gate_verdict.md \
  --fasc-winner-ckpt "$WIN" || true

echo "[$(date -Iseconds)] postprocess DONE dice=$DICE"

# Kick next lever if still blocked (honest: fasc < 0.40 or ET still dominates)
NEED_NEXT=$($PY - <<'PY'
import json
from pathlib import Path
rank=json.loads(Path("reports/phase3/fasc_native_rank.json").read_text())
dice=float(rank["winner"]["best_native_macro_dice"])
mpath=Path("reports/phase3/tier_b_seg_gh_v5best_metrics.json")
mae={}
if mpath.exists():
    m=json.loads(mpath.read_text())
    # tolerate nested shapes
    mae=m.get("val_metrics") or m.get("val_mae") or m.get("mae") or m
    def g(*keys):
        for k in keys:
            if isinstance(mae, dict) and k in mae: return float(mae[k])
            if isinstance(m, dict) and k in m: return float(m[k])
        return None
    pa=g("pa_mae","pa_deg","PA","pa"); fl=g("fl_mae","fl_mm","FL","fl"); mt=g("mt_mae","mt_mm","MT","mt")
else:
    pa=fl=mt=None
print(f"dice={dice} pa={pa} fl={fl} mt={mt}")
# ExtraTrees floor: PA 2.23 FL 9.86 MT 1.00
et_better = (pa is None) or (pa > 2.5) or (mt is None) or (mt > 1.5)
fasc_weak = dice < 0.40
print("NEED_NEXT" if (fasc_weak or et_better) else "STOP_OK")
PY
)
echo "$NEED_NEXT"
if echo "$NEED_NEXT" | grep -q NEED_NEXT; then
  echo "[$(date -Iseconds)] attention still blocked — start v6 HR + v6b hrnet (sequential on cuda:3)"
  nohup bash scripts/baselines/run_fasc_v6_chain.sh > $OUT/run_fasc_v6_chain.log 2>&1 &
  echo "started v6 chain pid=$!"
else
  echo "[$(date -Iseconds)] gates allow considering unblock — verdict already written"
fi
echo "[$(date -Iseconds)] DONE"
