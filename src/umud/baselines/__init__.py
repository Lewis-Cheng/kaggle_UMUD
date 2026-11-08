"""Phase 3 floor baselines (median / ExtraTrees / mask GeometryHead).

All scale-dependent outputs must carry ``scale_provenance``. Mask-derived
measurements are Tier A proxies, never Tier B / leaderboard claims.
"""

from umud.baselines.io import (
    load_split_assignments,
    load_test_ids,
    write_submission_csv,
)
from umud.baselines.mask_proxy import (
    proxy_apo_peak_gap,
    proxy_fascicle_pca,
)

__all__ = [
    "load_split_assignments",
    "load_test_ids",
    "write_submission_csv",
    "proxy_apo_peak_gap",
    "proxy_fascicle_pca",
]
