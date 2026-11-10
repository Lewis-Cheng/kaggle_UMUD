"""One FL/MT scale factor for the texture_rl probe.

The direction is the shared sign of the MultExp median ``gt / pred`` ratios.
MultExp is one external set of 35 images, not the reference model. The sign
is only a public probe. The size is a single 3% step, not a grid.
Disagreeing signs refuse a factor.
"""

from __future__ import annotations

import math

PROBE_FRACTION = 0.03


def probe_factor_from_median_ratios(fl_gt_over_pred: float, mt_gt_over_pred: float) -> float:
    """Return ``1 + sign * 0.03`` when FL and MT residual signs agree.

    A ratio below 1 means the prediction is high, so the factor shrinks both
    lengths. A ratio above 1 stretches both. A tie or a disagreement raises
    ``ValueError`` instead of picking a compromise.
    """

    def _sign(ratio: float) -> int:
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("median gt/pred must be finite and positive")
        if ratio < 1.0:
            return -1
        if ratio > 1.0:
            return 1
        return 0

    fl_sign = _sign(fl_gt_over_pred)
    mt_sign = _sign(mt_gt_over_pred)
    if fl_sign == 0 or mt_sign == 0 or fl_sign != mt_sign:
        raise ValueError(
            "FL and MT median residual signs do not agree; refusing a single scale factor"
        )
    return 1.0 + fl_sign * PROBE_FRACTION
