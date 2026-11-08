# Tier B GeometryHead (AmbrosM per-image mm/px)

Ran 2026-11-04 on SSH (`chocheng9@144.214.209.211:~/kaggle_UMUD`).

## Results (NOT leaderboard)

| Metric | Value |
|---|---|
| Train mm targets (GH ok + AmbrosM usable) | **699** / 2376 stems |
| Val mm targets | **269** / 738 stems |
| Val MAE PA (vs GH AmbrosM) | **2.315°** |
| Val MAE FL (vs GH AmbrosM) | **17.755 mm** |
| Val MAE MT (vs GH AmbrosM) | **1.868 mm** |
| Official S | **null** (no independent expert GT CSV) |
| Diagnostic S1 vs GH targets | 0.829 (NOT official / NOT LB) |

Train AmbrosM coverage among GeometryHead stems: measured 460, family_prior 242, unknown 1674.
Val: measured 177, family_prior 93, unknown 468.
Test AmbrosM: **309/309** (149 measured + 160 family_prior).

## Test path

- CSV: `reports/phase3/submission_tier_b_geometry.csv`
- Delimiter: **semicolon** (matches sample_submission)
- IDs: `IMG_xxxxx.tif` (sample format)
- Policy: ExtraTrees supervised by GeometryHead(mask, AmbrosM mm) — **not** train-median stub, **not** direct test-mask geometry (no test masks / no seg)
- Failures to median: 0 / 309

## Code

- `scripts/baselines/tier_b_geometry.py`
- `src/umud/baselines/ambrosm_scale_io.py`
- Scale source: `reports/phase3/ambrosm_scale_rows.csv`
- Split: `reports/phase2/split_assignments.jsonl` (duplicate-aware)

## Blockers

1. **No independent expert PA/FL/MT GT** in data root → cannot claim true Tier B LB / official S.
2. **Fasc majority AmbrosM unknown** (`rgb_800x1200_other` etc.) → mm targets concentrated on apo-overlapping stems with scale.
3. **High FL val MAE** (~17.8 mm): AmbrosM-scaled FL is multimodal across families (train p25≈39 / p75≈108 mm); ExtraTrees image features struggle — diagnostic only.
4. Test has **no masks** → cannot run direct GeometryHead on test until a seg model exists.
5. `q` (mm-per-tick) remains AmbrosM UI prior even when ticks are measured.

## Attention

**Still blocked.**

## Next lever

1. Extend AmbrosM parsers for fasc `unknown` families (esp. `rgb_800x1200_other`) to grow mm coverage.
2. Train/seg model for test masks → direct GeometryHead + AmbrosM on test (replace ExtraTrees).
3. Keep attention blocked until held-out evidence vs independent measurements exists.
