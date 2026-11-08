# Tier B seg → GeometryHead + AmbrosM

- scale protocol: `ambrosm_v2_ui_prior`
- pred masks: `reports/phase3/seg_preds_best`
- val paired MAE (pred GH vs GT-mask GH): PA=6.42750028393872 FL=None MT=6.256838635506608
- test ok mm: 309/309
- submission: `reports/phase3/submission_tier_b_seg_gh_best.csv` (direct seg→GH, not ExtraTrees)
- official_S: `None` (no_independent_expert_gt_csv_in_data_root)
- attention: still blocked pending gate verdict
- NOT LB
