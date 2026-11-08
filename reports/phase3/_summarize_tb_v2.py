import json
from pathlib import Path
r=json.load(open(Path.home()/"kaggle_UMUD/reports/phase3/tier_b_geometry_v2_metrics.json"))
print("n_test", r["n_test"], r.get("test_id_source"), r.get("n_sample_submission_ids"))
print("test_scale", r["test_scale_coverage"])
print("train", r["train_coverage"]["n_mm_targets"], "/", r["train_coverage"]["n_stems"], r["train_coverage"]["provenance_counts"])
print("val", r["val_coverage"]["n_mm_targets"], "/", r["val_coverage"]["n_stems"], r["val_coverage"]["provenance_counts"])
vm=r["val_metrics"]
print("MAE", vm["pa_mae"], vm["fl_mae"], vm["mt_mae"], "S1", vm.get("diagnostic_S1_vs_gh_targets"))
print("official_S", r["official_S"], "prov", r["run_scale_provenance"])
