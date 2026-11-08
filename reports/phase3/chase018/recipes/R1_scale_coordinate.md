# R1 — scale or coordinate correction only

```text
claim: A confirmed scale or coordinate fault is the dominant public error.
competing_explanation: The synthetic INTER_AREA pullback already agrees, so a coordinate patch would be a change without a confirmed fault.
one_changed_component: none authorized
fixed_components: texture_rl geometry constants, maps, calibration, measurement aggregator
evaluation_groups: not opened
primary_metric: paired PA/FL/MT delta versus R0 on the same images, after a fault is confirmed
non_regression_conditions: family_prior stays labeled family_prior; unknown scale is not invented; fallback rows stay tagged
abort_condition: abort if coordinate_pullback is pass and scale rows are still missing. Do not apply a Jacobian or q change in that state.
```

Current evidence, from `stage_fault_dump.json`:

- `coordinate_pullback`: pass on one synthetic scene (delta 0.00148° / 0.00395 mm / 0.100 mm). This does not cover network maps or letterbox padding.
- `scale_anchor`: inconclusive. Protocol counts are 2318 measured, 427 family_prior, 1373 unknown. The per-image scale CSV is absent.

Inner development, outer group evaluation, and the sealed external benchmark are not opened. MultExp is not used to choose a correction.
