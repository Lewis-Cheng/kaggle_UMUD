# Tier B seg → GeometryHead + AmbrosM

- scale protocol: `ambrosm_per_family_mixed_v2_hflip`
- pred masks: `/home/chocheng9/kaggle_UMUD/reports/phase3/seg_preds`
- val paired MAE (pred GH vs GT-mask GH): PA=5.0784 FL=9.2887 MT=6.1980
- n_paired_mm: 544
- test ok mm: 309/309
- submission: `/home/chocheng9/kaggle_UMUD/reports/phase3/submission_tier_b_seg_gh.csv` (direct seg→GH, not ExtraTrees)
- official_S: `None` (no_independent_expert_gt_csv_in_data_root)
- attention: **blocked** — see `attention_gate_verdict.md`
- NOT LB
