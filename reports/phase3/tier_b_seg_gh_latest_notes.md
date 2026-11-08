# Tier B seg → GeometryHead + AmbrosM

- scale protocol: `ambrosm_per_family_mixed_v2_hflip`
- pred masks: `/home/chocheng9/kaggle_UMUD/reports/phase3/seg_preds_best_fasc_latest`
- val paired MAE (pred GH vs GT-mask GH): PA=3.9311923418711103 FL=None MT=2.964498382558961
- test ok mm: 309/309
- submission: `/home/chocheng9/kaggle_UMUD/reports/phase3/submission_tier_b_seg_gh_latest.csv` (direct seg→GH, not ExtraTrees)
- official_S: `None` (no_independent_expert_gt_csv_in_data_root)
- attention: still blocked pending gate verdict
- NOT LB
