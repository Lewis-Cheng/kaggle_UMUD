# Tier B GeometryHead refresh — AmbrosM scale v2 (hflip)

**Date (Asia/Hong_Kong):** 2026-11-04  
**Host:** SSH `chocheng9@144.214.209.211:~/kaggle_UMUD`  
**Protocol / run provenance:** `ambrosm_per_family_mixed_v2_hflip`  
**Attention:** still **blocked**. **Official S / LB:** **null** (no independent expert GT).

## Before → after (val vs GH+AmbrosM targets, NOT expert / NOT LB)

| Metric | Tier B v1 (scale v1) | Tier B v2 (scale v2 hflip) | Δ |
|---|---:|---:|---:|
| Train mm targets | 699 / 2376 | **1744 / 2376** | **+1045** |
| Val mm targets | 269 / 738 | **544 / 738** | **+275** |
| Val MAE PA | 2.315° | **2.228°** | −0.087° |
| Val MAE FL | 17.755 mm | **9.858 mm** | **−7.90 mm** |
| Val MAE MT | 1.868 mm | **1.003 mm** | **−0.865 mm** |
| Diagnostic S1 vs GH | 0.829 | **0.509** | −0.320 |
| Official S | null | null | — |

### Scale provenance on GeometryHead stems

| Split | measured | family_prior | unknown |
|---|---:|---:|---:|
| train v1 | 460 | 242 | 1674 |
| train v2 | **1525** | 223 | **628** |
| val v1 | 177 | 93 | 468 |
| val v2 | **461** | 84 | **193** |
| test (AmbrosM) | 149 | 160 | **0** (309/309) |

Main coverage lift = fasc `rgb_800x1200_right_ticks_hflip` from AmbrosM scale v2. `q` (mm/tick) remains AmbrosM UI prior.

## Test path

- CSV: `reports/phase3/submission_tier_b_geometry_v2.csv` (semicolon; 309 rows)
- ID source: `test_stems_with_sample_suffix` — official `sample_submission.csv` is a **2-row format template**; IDs expanded from 309 test images as `IMG_xxxxx.tif` (matches v1)
- Policy: ExtraTrees supervised by GeometryHead(mask, AmbrosM mm) — **not** median stub, **not** direct test-mask GH (no test masks / no seg)
- Failures to median: 0 / 309
- **Not submitted to LB**

## Code / inputs

- `scripts/baselines/tier_b_geometry.py` (`--artifact-prefix tier_b_geometry_v2`)
- Scale: `reports/phase3/ambrosm_scale_v2_rows.csv`
- Split: `reports/phase2/split_assignments.jsonl` (duplicate-aware)
- Outputs: `reports/phase3/tier_b_geometry_v2_{metrics.json,rows.csv,notes.md,run.log}` + `submission_tier_b_geometry_v2.csv`

## Blockers

1. No independent expert PA/FL/MT GT → cannot claim Tier B LB / official S.
2. Residual AmbrosM unknown shapes (`1080×1640`, `500×760`, etc.) still excluded from mm targets.
3. `q` still AmbrosM UI prior even when ticks measured.
4. Test has no masks → ExtraTrees image→mm until seg→direct GH.
5. FL still harder than PA/MT (family multimodality); improved but diagnostic only.

## Next lever

1. **Seg → direct GeometryHead + AmbrosM on test** (replace ExtraTrees).
2. Optionally extend AmbrosM for residual unknowns (**do not invent `q`**).
3. Keep **attention blocked** until held-out evidence vs independent measurements exists.
