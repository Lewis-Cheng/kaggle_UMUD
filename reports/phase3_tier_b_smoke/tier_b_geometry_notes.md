# Tier B GeometryHead (AmbrosM per-image mm/px)

- Run provenance: `ambrosm_per_family_mixed`
- Train mm targets: **40** / GeometryHead ok+scale
- Val mm targets: **40**
- Val MAE (vs GH AmbrosM, NOT expert/LB): PA=3.9795117234973945 FL=1.9681826368520539 MT=2.2435476145233535
- Official S: **null** (no_independent_expert_gt_csv_in_data_root)
- Test policy: `extratrees_supervised_by_geometry_head_ambrosm_mm_NOT_direct_test_mask_geometry`
- Submission: `/home/chocheng9/kaggle_UMUD/reports/phase3_tier_b_smoke/submission_tier_b_geometry.csv`
- Attention: still blocked

## Blockers

- No independent expert PA/FL/MT CSV in data root → cannot claim Tier B LB S.
- Fasc majority still AmbrosM `unknown` (e.g. rgb_800x1200_other) → mm targets mostly from apo-overlapping stems with scale.
- Test has no masks → ExtraTrees image→mm, not direct GeometryHead on test.

## Next lever

- Seg model for test masks → direct GeometryHead+AmbrosM on test (replace ExtraTrees).
- Extend AmbrosM parsers for fasc `unknown` families.
- Keep attention blocked until true Tier B held-out evidence exists.

