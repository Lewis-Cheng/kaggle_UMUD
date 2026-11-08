#!/bin/bash
# Run when v6/v7/v8 done (or force with FORCE=1). Native eval + GH + honest verdict.
set -euo pipefail
cd ~/kaggle_UMUD
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONPATH=$HOME/kaggle_UMUD/src
OUT=reports/phase3; CKPT=$OUT/seg_ckpts
DATA=data/raw/umud; SPLIT=reports/phase2/split_assignments.jsonl
DEVICE=${DEVICE:-cuda:2}
FORCE=${FORCE:-0}

need_wait() {
  local json=$1 target=$2
  $PY - "$json" "$target" <<'PY' || return 0
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); t=int(sys.argv[2])
if not p.exists(): raise SystemExit(1)
d=json.loads(p.read_text()); h=d.get("history") or []
ep=int(h[-1].get("epoch",0)) if h else 0
raise SystemExit(0 if ep>=t else 1)
PY
}

if [[ "$FORCE" != "1" ]]; then
  for pair in "seg_fasc_v6_train.json:30" "seg_fasc_v7_train.json:35" "seg_fasc_v8_train.json:35"; do
    j=${pair%%:*}; t=${pair##*:}
    if ! need_wait "$CKPT/$j" "$t"; then
      echo "still training: $j"; exit 2
    fi
  done
fi

echo "[$(date -Iseconds)] eval all fasc ckpts on $DEVICE"
for ck in seg_fasc_v4b_best.pt seg_fasc_v5_best.pt seg_fasc_v5b_best.pt \
          seg_fasc_v6_best.pt seg_fasc_v6b_best.pt seg_fasc_v7_best.pt seg_fasc_v8_best.pt; do
  [[ -f $CKPT/$ck ]] || continue
  base=${ck%.pt}
  $PY scripts/audits/eval_fasc_ckpt_native.py --data-root $DATA --split-assignments $SPLIT \
    --ckpt $CKPT/$ck --device $DEVICE --out $OUT/eval_${base}_native.json || true
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
print(json.dumps({"winner":payload["winner"]["variant"] if payload["winner"] else None,
                  "dice": payload["winner"]["best_native_macro_dice"] if payload["winner"] else None,
                  "top":[(r["variant"], round(r["best_native_macro_dice"],4)) for r in ranked[:10]]}, indent=2))
PY

WIN=$($PY -c 'import json;print(json.load(open("reports/phase3/fasc_native_rank.json"))["winner"]["ckpt"])')
PRED=$OUT/seg_preds_best_fasc_latest
rm -rf $PRED; mkdir -p $PRED
# Prefer flexible/square NEAREST path (ckpt letterbox=False → square+NEAREST mask)
INF=scripts/baselines/infer_seg_letterbox.py
$PY $INF --data-root $DATA --apo-ckpt $CKPT/seg_apo_best.pt --fasc-ckpt "$WIN" \
  --out-dir $PRED --split-assignments $SPLIT --splits val test --device $DEVICE
$PY scripts/baselines/tier_b_seg_gh.py \
  --data-root $DATA --split-assignments $SPLIT --ambrosm-scale-csv $OUT/ambrosm_scale_v2_rows.csv \
  --pred-mask-dir $PRED --out-dir $OUT --artifact-prefix tier_b_seg_gh_latest \
  --scale-provenance ambrosm_per_family_mixed_v2_hflip
$PY scripts/baselines/write_attention_gate_verdict.py \
  --rank-json $OUT/fasc_native_rank.json \
  --metrics-json $OUT/tier_b_seg_gh_latest_metrics.json \
  --out $OUT/attention_gate_verdict.md \
  --fasc-winner-ckpt "$WIN"
date -Iseconds > $OUT/v678_eval_done.stamp
echo "=== metrics ==="; cat $OUT/tier_b_seg_gh_latest_metrics.json
echo "=== verdict head ==="; head -60 $OUT/attention_gate_verdict.md
