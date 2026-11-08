# R0 — reproduce texture_rl

```text
claim: The frozen texture_rl stack regenerates the recorded 309-row CSV.
competing_explanation: The workspace no longer contains the weights, bandit, or CSV, so the public number cannot be tied to this tree.
one_changed_component: none
fixed_components: tex_w=0.5, fl_sin_w=0.25, fl_factor=0.9, Frundles apo+fasc, PA+MT affine, FL raw in infer_rl_refiner.py
evaluation_groups: official test ids in reports/phase3/official_test_file_names.txt (309 image basenames)
primary_metric: byte-level or numeric identity with the recorded submission CSV
non_regression_conditions: image ids unchanged, including the 58 .png names; no new affine fit
abort_condition: abort while apo/fasc checkpoints, chase018_bandit.npz, or submission_texture_rl.csv are absent. Do not submit a regenerated file that has not matched the recorded CSV.
```

Status: not run. `baseline_freeze/manifest.json` records `replay_pass: false`.

Inner / outer / sealed roles are unused because nothing is fit. MultExp is not a selection metric.
