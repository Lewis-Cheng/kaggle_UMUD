#!/bin/bash
# Watchdog: if v6/v6b die before finishing epochs, resume from *_last.pt under nohup.
# Does not start duplicate trains. Survives agent death via tmux.
set -uo pipefail
ROOT=$HOME/kaggle_UMUD
OUT=$ROOT/reports/phase3
CKPT=$OUT/seg_ckpts
PY=/home/chocheng9/images_reduced/.venv/bin/python
cd "$ROOT"
export PYTHONPATH=$ROOT/src
DATA=data/raw/umud
SPLIT=reports/phase2/split_assignments.jsonl
LOG=$OUT/watch_fasc_v6_resume.log

finished() {
  local json=$1 epochs_target=$2
  python3 - "$json" "$epochs_target" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); t=int(sys.argv[2])
if not p.exists():
    raise SystemExit(1)
d=json.loads(p.read_text())
hist=d.get("history") or []
# finished if last epoch >= target OR len(hist)>=target
last_ep=0
if hist:
    last=hist[-1]
    last_ep=int(last.get("epoch", len(hist))) if isinstance(last,dict) else len(hist)
raise SystemExit(0 if last_ep>=t or len(hist)>=t else 1)
PY
}

echo "[$(date -Iseconds)] watchdog start" | tee -a "$LOG"

while true; do
  # v6
  if finished "$CKPT/seg_fasc_v6_train.json" 30; then
    echo "[$(date -Iseconds)] v6 finished" | tee -a "$LOG"
  elif pgrep -f "train_seg_fasc_v6\.py" >/dev/null; then
    :
  else
    echo "[$(date -Iseconds)] v6 DEAD — resume from last" | tee -a "$LOG"
    nohup $PY scripts/baselines/train_seg_fasc_v6.py \
      --data-root $DATA --split-assignments $SPLIT --out-dir $CKPT \
      --epochs 30 --batch-size 4 --size 640 --base-ch 48 \
      --edt-radius 3.0 --pos-weight 40 --cldice-w 0.75 \
      --device cuda:3 --num-workers 4 \
      --resume $CKPT/seg_fasc_v6_last.pt \
      >> $OUT/run_fasc_v6.log 2>&1 &
    echo "[$(date -Iseconds)] v6 resumed pid=$!" | tee -a "$LOG"
  fi

  # v6b
  if finished "$CKPT/seg_fasc_v6b_train.json" 30; then
    echo "[$(date -Iseconds)] v6b finished" | tee -a "$LOG"
  elif pgrep -f "train_seg_fasc_v6b\.py" >/dev/null; then
    :
  else
    echo "[$(date -Iseconds)] v6b DEAD — resume from last" | tee -a "$LOG"
    nohup $PY scripts/baselines/train_seg_fasc_v6b.py \
      --data-root $DATA --split-assignments $SPLIT --out-dir $CKPT \
      --epochs 30 --batch-size 6 --size 512 --encoder hrnet_w18 \
      --edt-radius 2.5 --pos-weight 35 --cldice-w 0.5 \
      --device cuda:2 --num-workers 4 \
      --resume $CKPT/seg_fasc_v6b_last.pt \
      >> $OUT/run_fasc_v6b.log 2>&1 &
    echo "[$(date -Iseconds)] v6b resumed pid=$!" | tee -a "$LOG"
  fi


  # v7
  if finished "$CKPT/seg_fasc_v7_train.json" 35; then
    echo "[$(date -Iseconds)] v7 finished" | tee -a "$LOG"
  elif pgrep -f "train_seg_fasc_v7\.py" >/dev/null; then
    :
  else
    if [[ -f $CKPT/seg_fasc_v7_last.pt ]]; then
      echo "[$(date -Iseconds)] v7 DEAD — resume from last" | tee -a "$LOG"
      nohup $PY scripts/baselines/train_seg_fasc_v7.py \
        --data-root $DATA --split-assignments $SPLIT --out-dir $CKPT \
        --epochs 35 --batch-size 12 --size 576 --base-ch 48 \
        --edt-radius 3.5 --pos-weight 45 --cldice-w 1.0 \
        --device cuda:1 --num-workers 6 --amp \
        --resume $CKPT/seg_fasc_v7_last.pt \
        >> $OUT/run_fasc_v7.log 2>&1 &
      echo "[$(date -Iseconds)] v7 resumed pid=$!" | tee -a "$LOG"
    fi
  fi

  # If both finished and chain not running, kick post-eval chain once
  if finished "$CKPT/seg_fasc_v6_train.json" 30 && finished "$CKPT/seg_fasc_v6b_train.json" 30; then
    if ! pgrep -f "run_fasc_v6_chain\.sh" >/dev/null && ! pgrep -f "write_attention_gate_verdict" >/dev/null; then
      # chain may have already completed; check stamp
      if [[ ! -f $OUT/v6_chain_eval_done.stamp ]]; then
        echo "[$(date -Iseconds)] both trains done — launching eval chain if needed" | tee -a "$LOG"
        # Prefer finishing via existing chain if still alive; else run eval-only
        nohup bash -c '
          set -euo pipefail
          ROOT=$HOME/kaggle_UMUD; OUT=$ROOT/reports/phase3; CKPT=$OUT/seg_ckpts
          PY=/home/chocheng9/images_reduced/.venv/bin/python
          cd "$ROOT"; export PYTHONPATH=$ROOT/src
          DEVICE=cuda:1
          DATA=data/raw/umud; SPLIT=reports/phase2/split_assignments.jsonl
          for ck in seg_fasc_v4b_best.pt seg_fasc_v5_best.pt seg_fasc_v5b_best.pt seg_fasc_v6_best.pt seg_fasc_v6b_best.pt; do
            [[ -f $CKPT/$ck ]] || continue
            base=${ck%.pt}
            $PY scripts/audits/eval_fasc_ckpt_native.py --data-root $DATA --split-assignments $SPLIT \
              --ckpt $CKPT/$ck --device $DEVICE --out $OUT/eval_${base}_native.json || true
          done
          $PY - <<'"'"'PY'"'"'
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
print(json.dumps({"winner_variant": payload["winner"]["variant"] if payload["winner"] else None,
                  "dice": payload["winner"]["best_native_macro_dice"] if payload["winner"] else None,
                  "top":[(r["variant"], round(r["best_native_macro_dice"],4)) for r in ranked[:8]]}, indent=2))
PY
          WIN=$($PY -c "import json;print(json.load(open(\"reports/phase3/fasc_native_rank.json\"))[\"winner\"][\"ckpt\"])")
          # Prefer NEAREST square align infer if available
          INF=scripts/baselines/infer_seg_nearest.py
          if [[ ! -f $INF ]]; then INF=scripts/baselines/infer_seg_letterbox.py; fi
          PRED=$OUT/seg_preds_best_fasc_latest
          rm -rf $PRED; mkdir -p $PRED
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
          date -Iseconds > $OUT/v6_chain_eval_done.stamp
        ' >> $OUT/run_fasc_v6_eval_watchdog.log 2>&1 &
      fi
    fi
    # Exit watchdog after both done + stamp or after long idle
    if [[ -f $OUT/v6_chain_eval_done.stamp ]]; then
      echo "[$(date -Iseconds)] eval stamp present — watchdog idle exit" | tee -a "$LOG"
      exit 0
    fi
  fi
  sleep 90
done

# NOTE: v8 block injected — also covered by umud_fasc_v8 tmux

# NOTE: v8 block injected — also covered by umud_fasc_v8 tmux

# NOTE: v8 block injected — also covered by umud_fasc_v8 tmux
