# AmbrosM / xiaoleilian per-family scale — freeze notes

**Date (Asia/Hong_Kong):** 2026-11-04  
**Sources:** public Kaggle notebooks (Apache 2.0)  
- `ambrosm/umud-quick-and-dirty` → `collect_metadata`  
- `xiaoleilian/umud-hardcore-eda` → `ambrosm_test_scale` + `family_of`  
Refs checked into `refs/ambrosm_scale/`.

## What was ported

Module: `src/umud/calibration/ambrosm_scale.py`  
Runner: `scripts/audits/run_ambrosm_scale.py`  

Per image: family classify → AmbrosM branch → `px_per_cm` / `px_per_mm` / `mm_per_px` + FOV bbox when available.  
**Never** uses TIFF print-dpi.

### Provenance enum

| Tag | Meaning |
|---|---|
| `measured` | Tick *positions* read from this image; physical pitch `q` (`assumed_mm_per_tick`) is still AmbrosM device-UI prior |
| `family_prior` | Fixed AmbrosM shape/HUD formula (e.g. `(783-42)/5`, `630.5/5`, bottom 5 cm spans) |
| `unknown` | No AmbrosM family match / fingerprint miss |

`q_provenance` is almost always `family_prior` even when overall provenance is `measured` (identifiability: ticks alone do not invent mm).

## Full-data coverage (SSH `~/kaggle_UMUD/data/raw/umud`)

| Pool | N | measured | family_prior | unknown | coverage |
|---|---:|---:|---:|---:|---:|
| test | 309 | 149 | 160 | **0** | **100%** |
| apo | 1048 | 474 | 188 | 386 | 63.2% |
| fasc | 2761 | 0 | 79 | 2682 | 2.9% |
| **all** | **4118** | **623** | **427** | **3068** | **25.5%** (1050 with px/mm) |

Test set matches AmbrosM claim: all 309 get a scale.

## Per-family median scale (images with px/mm)

| Family | N | provenance | median px/mm | median mm/px | q used (mm) |
|---|---:|---|---:|---:|---|
| png_lumify | 58 | measured | 15.00 | 0.0667 | 10 |
| rgb_800x1200_right_ticks | 565 | measured | 14.29 | 0.0700 | 5 or 2 |
| rgb_800x1200_left_ticks | 90 | family_prior | 14.82 | 0.0675 | 10 (5 cm / 5) |
| rgb_644x1088 | 50 | family_prior | 12.61 | 0.0793 | 10 |
| h512_bottom_ticks | 269 | family_prior | 7.78 | 0.1285 | 10 |
| h513_bottom_ticks | 6 | family_prior | 7.78 | 0.1285 | 10 |
| gray_853_bottom_ticks | 12 | family_prior | 16.68 | 0.0600 | 10 |

## Fail modes / unknowns

Largest unknown families (mostly **train**, esp. fasc / DL_Track mix):

- `rgb_800x1200_other` — 1972 (AmbrosM notes train can be **hflipped**; left/right fingerprints miss)
- `other_1080x1640_RGB` — 645
- `other_500x760_RGB` — 216
- `other_652x800_RGB` — 118
- `other_600x800_RGB` — 79
- plus smaller shapes

These have no AmbrosM test-family parser. Do **not** fill with rival `0.07225` or TIFF dpi. Prefer pixel-space geometry or leave FL/MT mm gated until a family parser / UI `q` exists.

## Paths

**SSH lab** (`chocheng9@144.214.209.211`):  
- code: `~/kaggle_UMUD/src/umud/calibration/`  
- runner: `~/kaggle_UMUD/scripts/audits/run_ambrosm_scale.py`  
- report: `~/kaggle_UMUD/reports/phase3/ambrosm_scale_report.json`  
- csv: `~/kaggle_UMUD/reports/phase3/ambrosm_scale_rows.csv`

**Box:**  
- same relative paths under `/workspace/kaggle_UMUD/`  
- slim summary: `reports/phase3/ambrosm_scale_summary.json`

## Recommended next step (Tier B GeometryHead)

1. **Pin per-image `mm_per_px` for all 309 test + covered train** from this report (provenance column required).  
2. Run GeometryHead / mask-proxy **in mm-space** only where provenance ∈ {`measured`,`family_prior`}; keep pixel-space or null for `unknown`.  
3. Optional follow-up: extend AmbrosM for **hflipped** `800x1200` train (mirror left/right fingerprints) to recover fasc coverage — still tag `q` as family_prior.  
4. Attention remains blocked until Tier B with pinned S is green.
