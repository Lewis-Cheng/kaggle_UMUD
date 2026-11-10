# Output-contract formal verification — `output_recipe_identifiability`

**Date (Asia/Hong_Kong):** 2026-11-09  
**Harness:** `scripts/chase018/fv_output_stages.py`  
**Method:** Anthropic-style formal verification (same pattern as `reports/phase3/fasc_fv_protocol.md`). Properties are aborting checks; a soft assert is not a property.

## What the public tries taught (locked in)

| Public file | Score | What changed vs texture_rl |
|---|---|---|
| `texture_rl` (= `cal_pa_mt`) | **0.29895** | PA affine (+0.92° mean abs), MT +0.18 mm; FL identical to texture_pa |
| `scale_probe_flmt_0.97` | **0.35218** | PA same; FL and MT ×0.97 |

Conclusions the harness encodes:

- `texture_rl` is `cal_pa_mt` exactly (PA/MT affine, FL identity). Rebuilt to < 1e-6.
- Shrinking FL+MT (0.97) is publicly worse. A candidate that shrinks both lengths re-opens a losing direction.
- 0.17 needs summed normalized-MAE gain 0.387; FL-only that is 4.64 mm, PA-only 2.32°, MT-only 1.16 mm.

## Stage chain (abort ids)

| Stage | Property |
|---|---|
| `official_ids` | 309 official IMG ids, exact set |
| `fl_identity` | texture_rl keeps texture_pa FL exactly |
| `pa_affine_rebuild` / `mt_affine_rebuild` | texture_rl = clipped PA/MT affine of texture_pa (< 1e-6) |
| `fl_identity_manifest` | manifest declares FL identity, not applied |
| `probe_pa_unchanged` / `probe_factor` | 0.97 probe copied PA, scaled FL/MT by 0.97 |
| `length_shrink_rejected` | 0.97 public 0.35218 > 0.29895 |
| `candidate_no_length_shrink` | candidate median FL and MT ratios ≥ 0.999 vs texture_rl |
| `candidate_not_slight` | candidate changes ≥1 target by > 5% of tolerance on average |

## Breakage witnesses (mutants)

- Identical-to-rl rows → must fail `candidate_not_slight`. PASS.
- FL/MT ×0.97 rows → must fail `candidate_no_length_shrink`. PASS.

## Trusted base

This Python checker, the checked-in CSVs, `src/umud/metric/score.py`, and the three public numbers above. Harness green = the output contract holds. **Not a public-score claim.**

## Command

```bash
export PYTHONPATH=src
.venv/bin/python scripts/chase018/fv_output_stages.py \
  --candidate reports/phase3/chase018/submission_ref_geometry_v2.csv
```

JSON: `reports/phase3/chase018/fv_output_status.json`.
