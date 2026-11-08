#!/bin/bash
# Pick best fasc ckpt among recover/v3e/v3c/best, infer, tier_b
set -uo pipefail
cd ~/kaggle_UMUD
PY=/home/chocheng9/images_reduced/.venv/bin/python
export PYTHONUNBUFFERED=1

if [[ -f /tmp/umud_infer_flex.tgz ]]; then
  tar -xzf /tmp/umud_infer_flex.tgz -C ~/kaggle_UMUD
fi

$PY - <<'PY'
import torch
from pathlib import Path
ckpts = list(Path('reports/phase3/seg_ckpts').glob('*.pt'))
cands = []
for p in ckpts:
    if 'apo' in p.name: continue
    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
    except Exception as e:
        print('skip', p, e); continue
    cands.append((float(ck.get('val_dice') or -1), p.name, ck.get('variant'), ck.get('arch'), str(p)))
cands.sort(reverse=True)
print('FASC CKPT RANKING:')
for d,n,v,a,p in cands:
    print(f'  dice={d:.4f} name={n} variant={v} arch={a}')
best = cands[0][4] if cands else ''
print('BEST', best)
Path('/tmp/best_fasc_ckpt.txt').write_text(best)
Path('/tmp/best_fasc_dice.txt').write_text(str(cands[0][0]) if cands else 'nan')
PY

BEST=$(cat /tmp/best_fasc_ckpt.txt)
DICE=$(cat /tmp/best_fasc_dice.txt)
if [[ -z "$BEST" || ! -f "$BEST" ]]; then
  echo "No fasc ckpt found"; exit 1
fi
APO=reports/phase3/seg_ckpts/seg_apo_best.pt
OUT=reports/phase3/seg_preds_best
mkdir -p "$OUT" reports/phase3

echo "== infer apo+fasc with BEST=$BEST dice=$DICE =="
$PY scripts/baselines/infer_seg_flexible.py \
  --data-root data/raw/umud \
  --split-assignments reports/phase2/split_assignments.jsonl \
  --apo-ckpt "$APO" \
  --fasc-ckpt "$BEST" \
  --out-dir "$OUT" \
  --device cuda:2 \
  --splits val test

echo "== tier_b seg GH =="
$PY scripts/baselines/tier_b_seg_gh.py \
  --data-root data/raw/umud \
  --split-assignments reports/phase2/split_assignments.jsonl \
  --ambrosm-scale-csv reports/phase3/ambrosm_scale_v2_rows.csv \
  --pred-mask-dir "$OUT" \
  --out-dir reports/phase3 \
  --artifact-prefix tier_b_seg_gh_best \
  --scale-provenance ambrosm_v2_ui_prior \
  2>&1 | tee reports/phase3/run_tier_b_best.log

echo DONE_INFER_TIERB
