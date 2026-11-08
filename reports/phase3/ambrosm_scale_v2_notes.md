# AmbrosM per-family scale v2 — hflip / uncovered-shape expansion

**Date (Asia/Hong_Kong):** 2026-11-04  
**Version:** v2 (extends v1; provenance rules unchanged)

## What changed vs v1

| Change | Detail |
|---|---|
| `rgb_800x1200_right_ticks_hflip` | AmbrosM right-tick math on mirrored column **x≈49** (train hflip). Tick positions **measured**; `q` still AmbrosM UI prior (5 mm if n≤14, 2 mm if n=15). |
| `rgb_800x1200_left_ticks_hflip` | Mirrored left HUD `a[42,1126:1133]`; fixed span `(783-42)/5` → **family_prior** (0 hits this run). |
| Uncovered shapes | `1080×1640` (645), `500×760` (216), `652×800` (118), `600×800` (79), `556×660` (23), `688×1234` (13), residual `rgb_800x1200_other` (277) stay **unknown** — no AmbrosM `q`. |

**Never** TIFF print-dpi. Attention still blocked. Not an LB claim.

## Before / after coverage (SSH `data/raw/umud`)

| Pool | v1 with px/mm | v2 with px/mm | Δ |
|---|---:|---:|---:|
| test | 309/309 (100%) | 309/309 (100%) | 0 |
| apo | 662/1048 (63.2%) | 662/1048 (63.2%) | 0 |
| fasc | 79/2761 (**2.9%**) | 1774/2761 (**64.3%**) | **+1695** |
| **all** | **1050/4118 (25.5%)** | **2745/4118 (66.7%)** | **+1695** |

### Provenance counts

| | measured | family_prior | unknown |
|---|---:|---:|---:|
| v1 | 623 | 427 | 3068 |
| v2 | **2318** | 427 | **1373** |

All +1695 new scales are `measured` ticks on hflip right bars; `q` remains `family_prior` (5 or 2 mm).

### Hflip family detail

- `rgb_800x1200_right_ticks_hflip`: **1695** (n=14:1269, n=9:271, n=8:133, n=10:15, n=11:7)
- Residual `rgb_800x1200_other`: 277 (false tick columns / irregular bottom — left unknown)
- Fasc remaining unknown: `other_1080x1640_RGB` 645, `other_500x760_RGB` 216, `other_600x800_RGB` 65, `other_652x800_RGB` 42, `rgb_800x1200_other` 19

## Audit notes (honest blockers)

1. **1080×1640**: left chrome bright; spike gaps ~127 px but **no AmbrosM UI `q`** → unknown.
2. **500×760**: right-edge gaps irregular (median ~25, std ~14); only 6 images with stable L gap 58 → unknown.
3. **652×800 / 600×800**: weak/no scale-bar signal → unknown.
4. `q` is still AmbrosM device-UI prior everywhere — tick spacing can be measured, mm-per-tick is not independently observed.

## Paths

**SSH** `chocheng9@144.214.209.211:~/kaggle_UMUD`  
- `reports/phase3/ambrosm_scale_v2_{report.json,rows.csv,summary.json,notes.md,run.log}`  
- code: `src/umud/calibration/ambrosm_scale.py`

**Box** `/workspace/kaggle_UMUD` — same relative paths (v1 files kept for before/after).

## Tier B refresh?

**Worth it next** for train mm coverage (fasc 2.9%→64.3% → many more GeometryHead mm targets), but:
- still GH+AmbrosM targets, **not** expert GT / LB
- test already 309/309 — submission scale unchanged
- Attention remains blocked

Recommended: optional Tier B re-run with v2 CSV as scale source; then seg→direct GH on test. Do **not** invent `q` for remaining unknowns.
