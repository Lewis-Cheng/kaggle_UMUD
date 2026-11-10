"""The coupled PA/FL identities match the pinned scorer and a known sensitivity."""

from __future__ import annotations

import math

from umud.metric.score import score_components
from umud.verification.contracts import ContractAbort
from umud.verification.coupling import (
    first_order_length_change_mm,
    normalized_mean_score,
)


def test_supplied_mae_is_the_mean_not_the_sum() -> None:
    """MAE 1.127 / 2.489 / 0.162 is S1 = 0.14975, and the sum 0.44925 is not the score.

    mutant: drop the outer 1/3 and the check must abort. This function does not
    apply that mutant; the audit script records the sum beside the mean.
    """
    hand = normalized_mean_score(1.127, 2.489, 0.162)
    official = score_components(
        {"pa_deg": [1.127], "fl_mm": [2.489], "mt_mm": [0.162]}
    )["S1"]
    if abs(hand - official) > 1e-12:
        raise ContractAbort(f"hand mean {hand} != score_components {official}")
    if abs(hand - 0.14975) > 1e-9:
        raise ContractAbort(f"supplied MAE mean {hand} != 0.14975")
    summed = hand * 3.0
    if abs(summed - 0.44925) > 1e-9:
        raise ContractAbort(f"normalized sum {summed} != 0.44925")
    if abs(summed - official) < 1e-9:
        raise ContractAbort("score_components returned the sum")


def test_one_degree_pennation_moves_length_by_about_3_mm() -> None:
    """At 50 mm and 15 degrees, +1 degree with fixed thickness moves FL by about -3.26 mm."""
    delta = first_order_length_change_mm(
        50.0,
        15.0,
        1.0,
        50.0 * math.sin(math.radians(15.0)),
        0.0,
    )
    if abs(delta - (-3.26)) > 0.02:
        raise ContractAbort(f"first-order FL change {delta} is not about -3.26 mm")
