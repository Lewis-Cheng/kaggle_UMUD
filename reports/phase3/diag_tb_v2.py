import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.home() / "kaggle_UMUD" / "src"))
r=json.load(open(Path.home()/"kaggle_UMUD/reports/phase3/tier_b_geometry_v2_metrics.json"))
print("n_test", r.get("n_test"))
print("test_policy", r.get("test_policy"))
print("n_test_failures", r.get("n_test_failures_to_median"))
print("submission", r.get("submission"))
print("test_scale", r.get("test_scale_coverage"))
from umud.data.layout import discover_layout
from umud.baselines.io import read_sample_ids
layout = discover_layout(Path.home()/"kaggle_UMUD/data/raw/umud")
print("sample", layout.sample_submission)
ids = read_sample_ids(layout.sample_submission)
print("n_sample_ids", len(ids))
print("first3", ids[:3])
print("last3", ids[-3:])
p=Path.home()/"kaggle_UMUD/reports/phase3/submission_tier_b_geometry_v2.csv"
print("sub_bytes", p.stat().st_size, "lines", sum(1 for _ in p.open()))
# also check write_submission_csv path and whether pandas flushed
print("v1_sub_lines", sum(1 for _ in open(Path.home()/"kaggle_UMUD/reports/phase3/submission_tier_b_geometry.csv")))
