# Track B — frozen v5 + RL hybrid (Mission B)

**Time (Asia/Hong_Kong):** 2026-11-05 02:03:46 HKT

## Goal
Real Kaggle **public score** for `trackB_v5_rl_hybrid` (not self-eval).

## Pipeline
- Frozen apo + frozen fasc **MSTinyUNet v5** (`seg_fasc_v5_best.pt`) → geometry head (GH) + ExtraTrees prior.
- RL = **post-seg SoftmaxBandit** refiner (8 discrete actions), trained on **UMUD MultExpData** (35 expert images, OSF hb5xj / `data/external/umud_expert_arch_v0.1.0/`).
- Reward label: `S_expert_NOT_LB` (τ 6° / 12 mm / 3 mm). **Not** fasc Dice. **Not** LB as train reward.
- Declared external: `reports/phase3/external_data_declare.md`. No FALLMUD. No hand-label test. v5 encoder not replaced.

## LOOCV on MultExpData (self-check only)
| policy | S_expert_NOT_LB |
|---|---|
| GH only | 2.070 |
| ET only | 2.212 |
| median prior | 1.598 |
| **bandit LOOCV** | **1.642** (gate beat GH: yes) |
| final greedy full-fit | 1.617 (collapsed to blend_gh50_et50 ×35) |

LOOCV action hist: blend_gh50_et50×18, median_prior×14, blend_gh25_et75×3.

## Test inference
- CSV: `reports/phase3/rl_track_b/submission_track_b_v5_rl.csv`
- 309 rows; header `image_id;pa_deg;fl_mm;mt_mm`; ids match GH/v5best/expected_test_ids.
- Soft-validate: **pass**.
- Format rewritten to match official sample: **UTF-8 BOM + CRLF + semicolon**.
- Backup comma variant: `submission_track_b_v5_rl_comma.csv`.
- **Learned / applied test actions:** all 309 → `blend_gh50_et50` (action index 2). Policy collapsed under full-fit greedy; LOOCV had more mix (incl. median_prior).

## Kaggle public score
**Not available yet — submit paused.**

### Attempts (dataworkch Bearer `access_token`, not lewischeng kaggle.json)
1. Upload as `filename.ext` → error: Invalid extension `.ext`.
2. Correct `.csv` upload (ref **57994042**, msg `trackB_v5_rl_hybrid`) → error: **`ID column image_id not found in submission`** despite header/ids matching sample & Track A. File size uploaded = 21395 B (exact).

### Blocker (per parent steering 02:01 HKT)
- lewischeng `kaggle.json` → **401 expired**.
- Do **not** burn more submit cycles until parent confirms token refresh.
- CSV + BOM fix ready for immediate resubmit when auth is green.
- Note: prior POSTs with dataworkch Bearer returned HTTP 200 "Successfully submitted"; failures were **grader format**, not auth. Still holding per steering.

## Artifacts
- Policy: `reports/phase3/rl_track_b/geometry_bandit.npz`
- Train report: `reports/phase3/rl_track_b/train_report.json`
- Rows+actions: `reports/phase3/rl_track_b/submission_track_b_v5_rl_rows.csv`
- Plan: `reports/phase3/rl_external_gt_plan.md`

## Next (parent)
1. Refresh Kaggle token / confirm which account (lewischeng vs dataworkch).
2. Resubmit `submission_track_b_v5_rl.csv` (BOM+CRLF) with message `trackB_v5_rl_hybrid`; if still `image_id not found`, try `submission_track_b_v5_rl_comma.csv`.
3. Record numeric **publicScore** here.
