# R2 — measurement model only

```text
claim: Pennation changes because the aggregator ignores local deep-aponeurosis tangent and multi-position fascicles.
competing_explanation: On a straight synthetic fascicle the weighted median and the unweighted median already agree, and depth-fit is undefined, so the aggregator is not shown to be the fault.
one_changed_component: measurement aggregator on frozen maps (multi-position fascicle orientation and local deep-apo tangent)
fixed_components: maps, scale, calibration, network weights, texture_rl constants
evaluation_groups: inner competition-mask groups when images exist; outer groups score the frozen choice and do not pick the family
primary_metric: paired angle and pixel-length error versus mask-derived geometry on the inner split; official S only after the recipe is frozen
non_regression_conditions: Dice is recorded and is not the selection metric; a Dice rise with an angle rise aborts
abort_condition: abort while apo and fasc maps are missing. Abort if the inner split's depth span cannot define the depth-fit and no alternative was pre-registered.
```

Synthetic contrast already computed, not selected:

- weighted-median PA 24.958°
- unweighted-median PA 24.969°
- depth-fit undefined (`depth_span` 0.091 < 0.2)

No winner. No fit. No submit. MultExp is not the selection set.
