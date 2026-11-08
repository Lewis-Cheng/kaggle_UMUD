# R3 — remove or restrict the frozen calibration

```text
claim: The MultExp PA+MT affine hurts the test domain relative to raw texture_rl outputs.
competing_explanation: The public step from texture_pa 0.31903 to texture_rl 0.29895 is small and positive, and the affine file disagrees with the deploy path about FL, so "remove calibration" is not yet one component.
one_changed_component: drop the PA and MT affine (FL already raw in infer_rl_refiner.py), or restrict the affine to a subset named before the run
fixed_components: maps, scale, measurement aggregator, tex_w, fl_sin_w, fl_factor
evaluation_groups: official test ids, one submit after R0 has matched
primary_metric: public score of this single change against texture_rl 0.29895; offline, paired delta on held-out groups only
non_regression_conditions: no new coefficients are fit; FL is not given a new slope while the JSON/deploy disagreement is open
abort_condition: abort until R0 has reproduced the CSV. Abort while multexp_chase_affine.json says fl_mm applied=true and infer_rl_refiner leaves FL raw, because those are two different calibrations.
```

Example only, not a test image: raw (20°, 70 mm, 20 mm) deploys as (19.397°, 70 mm, 20.179 mm) under the current PA+MT path.

Inner split would have to name any restricted subset before looking at an outer or public number. MultExp is not used to refit the affine.
