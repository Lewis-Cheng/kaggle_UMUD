# UMUD contract claims

Number source: [`fv_umud_contracts.json`](fv_umud_contracts.json), written by `python tests/properties/test_umud_contracts.py`.

This file records executable properties. It is not a Rocq or RISC-V proof. `production_readiness` stays `BLOCKED`: the harness did not run real ultrasound images, checkpoints, or a `texture_rl` CSV replay.

## Status

| Claim | Statement | Status | Falsifier | Proof path |
|---|---|---|---|---|
| `coordinate_roundtrip` | For invertible `J`, `u' = J u` and `G' = J^{-T} G J^{-1}` preserve `u^T G u`. | holds on the seed and on 1,000 random affines | drop the inverse (pushforward `J^T G J`) | harness + `inverse_affine_metric` |
| `angle_geometry` | Cosine uses `G = diag(sx^2, sy^2)`. Setting `sy = sx` changes it when `sy != sx`. | holds on the seed | set `sy = sx` | `physical_cosine` |
| `length_units` | 100 px at 0.1 mm/px is 10 mm. | holds on the seed | multiply the scale by 10 | `physical_length` |
| `metric_normalization` | S1 is the mean of MAE/6, MAE/12, MAE/3. The seed errors give S1 = 0.5. | holds; matches `score_components` | replace the mean by the sum (1.5) | `src/umud/metric/score.py` |
| `group_disjoint` | Train and evaluation group ids are disjoint. | holds on the seed | put group 3 on both sides | `assert_groups_disjoint` |
| `finite_output` | A NaN prediction is rejected by `validate_submission`. | holds on the seed | swallow the validation error | `src/umud/metric/submission.py` |
| `metric_anchor` | `family_prior` stored as `measured` is rejected. A registered fallback may still exist under its own label. | holds on the seed | copy `family_prior` into the measured label | `assert_scale_not_relabeled` |
| `promotion_evidence` | Synthetic passes cannot set `production_readiness` to PASS. | holds; this JSON is refused | set PASS while `real_images_ran` is false | `assert_promotable` |
| `coordinate_pullback` | `host_faithful.measure` on one synthetic scene agrees before and after `INTER_AREA` resize to 384×576, within 1.5° / 3 mm / 1 mm. | pass on that scene only | spacing not pulled back by the resize Jacobian, or a delta above the tolerance | integration block in the JSON |

## Witness numbers

Positive checks: 8/8. Mutants rejected: 8/8.

Random affines: n=1000, seed=20261108.

- max abs length-squared residual: 1.382e-14
- max abs cosine residual: 4.441e-16
- max abs point round-trip: 7.772e-15

`coordinate_pullback` on the synthetic scene (`INTER_AREA`, source 600×800, spacing 0.15×0.08 mm/px):

- native PA/FL/MT: 24.958° / 71.097 mm / 27.600 mm
- resized PA/FL/MT: 24.960° / 71.093 mm / 27.500 mm
- absolute delta: 0.00148° / 0.00395 mm / 0.100 mm
- verdict: `pass` inside the preregistered tolerance

That pass does not extend to network probability maps, letterbox padding, or real test images. Those were not inputs.

## Algebra the numbers check

Let `G = diag(sx^2, sy^2)` and `u' = J u` with `J` invertible. The pulled-back metric `G' = J^{-T} G J^{-1}` satisfies `u'^T G' u' = u^T G u`, and the same identity holds for the inner product `u^T G v`, hence for the cosine. Assumptions: `J` invertible, spacing positive, the quantity is computed in that metric. Interpolation and fascicle detection are outside this identity.

## Commands

```bash
python tests/properties/test_umud_contracts.py
python tests/properties/test_umud_contracts.py --prove-mutants
```

The first command exits 2 if `coordinate_pullback` is `fail`. This run exited 0 because the synthetic scene passed. Production remains `BLOCKED`.
