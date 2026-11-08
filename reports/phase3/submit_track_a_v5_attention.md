# Track A — v5 + Attention → Kaggle public score

**Date (Asia/Hong_Kong):** 2026-11-05 ~02:00 HKT  
**Competition:** `umud-challenge-muscle-architecture-in-ultrasound-data`  
**Submission message tag:** `trackA_v5_attention`  
**Cheating:** none (no hand labels on test/val, no private-score blending, no LB ensembles)

## Pipeline

1. **Frozen fasc v5** `seg_fasc_v5_best.pt` (MSTinyUNet, native macro Dice ~0.2846) + **apo** `seg_apo_best.pt`
2. Pred masks: train → `reports/phase3/seg_preds_v5_train`; val+test → `reports/phase3/seg_preds_best_fasc_v5`
3. **Attention residual** on top of GeometryHead(v5 priors)+AmbrosM:
   - Model: `src/umud/models/anatomy_query_v5.py` (`V5AttentionRegressor`)
   - Input channels: image + apo prior + fasc prior
   - All attention ON (image SA → anatomy-query CA → query SA)
   - Head predicts residual Δ(PA,FL,MT) added to v5 GH base
4. Train: `scripts/baselines/train_attention_v5.py` on cuda:1, 25 epochs  
   - n_train usable AmbrosM GH targets: **1744**  
   - n_val: **544**  
   - Best proxy score (tol-normalized MAE/3): **0.5461** @ ep21  
   - Val MAE vs GT-mask GH (NOT expert / NOT LB): PA **3.79°** · FL **3.53 mm** · MT **2.14 mm**  
     (v5 base alone was PA 3.93 · FL 7.69 · MT 2.96 — attention mainly helped FL/MT)
5. Infer: `scripts/baselines/infer_attention_v5.py` → 309/309 test rows
6. Validate: exact ID set vs `test_images_v2/test_set_v2` stems (**OK**)

## Artifacts

| Item | Path |
|---|---|
| Best ckpt | `reports/phase3/attn_ckpts/attn_v5_best.pt` |
| Train log JSON | `reports/phase3/attn_ckpts/attn_v5_train.json` |
| Submission CSV | `reports/phase3/submission_trackA_v5_attention.csv` |
| Meta | `reports/phase3/submission_trackA_v5_attention.meta.json` |
| Windows copy | `C:\Users\lewis\Desktop\umud_submissions\submission_trackA_v5_attention.csv` |
| Run log | `reports/phase3/run_track_a_v5_attention.log` |

CSV format: `image_id;pa_deg;fl_mm;mt_mm` (semicolon), 310 lines (header+309).

## Kaggle submit attempt

```
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data \
  -f submission_trackA_v5_attention.csv -m "trackA_v5_attention"
```

**Result: BLOCKED — HTTP 401 Unauthorized**

- Found: `C:\Users\lewis\.kaggle\kaggle.json` (username `lewischeng`, key_len=37, mtime 2026-02-06)
- `kaggle competitions list` / `files` / `submit` all return **401**
- Key is present but **rejected by Kaggle API** (expired/revoked/rotated)

### Exact unblock path

1. Open https://www.kaggle.com/settings → **API** → **Create New Token** (downloads `kaggle.json`)
2. Replace `C:\Users\lewis\.kaggle\kaggle.json` with the new file (same path)
3. Re-run:

```powershell
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle"
kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data `
  -f "$env:USERPROFILE\Desktop\umud_submissions\submission_trackA_v5_attention.csv" `
  -m "trackA_v5_attention"
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data
```

## Public score

**Not available yet** — submit did not upload. After the token refresh above, poll `kaggle competitions submissions` for the public score and append it here.

## Note on sample_submission.csv

Lab `data/raw/umud/sample_submission.csv` is a **2-row stub**. Infer correctly used the 309 stems under `test_images_v2/test_set_v2` (same path as prior Tier-B v5best CSV).
