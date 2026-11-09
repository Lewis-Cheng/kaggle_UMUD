"""Coupled pennation / fascicle-length identities.

These are diagnostics. They are not a new predictor and they do not use
test labels. ``d_theta`` is in degrees at the call boundary and converted
to radians inside.
"""

from __future__ import annotations

import math


def normalized_mean_score(mae_pa_deg: float, mae_fl_mm: float, mae_mt_mm: float) -> float:
    """S1 = mean(MAE_PA/6, MAE_FL/12, MAE_MT/3). Not the sum."""
    return (
        (mae_pa_deg / 6.0) + (mae_fl_mm / 12.0) + (mae_mt_mm / 3.0)
    ) / 3.0


def first_order_length_change_mm(
    length_mm: float,
    theta_deg: float,
    d_theta_deg: float,
    thickness_mm: float,
    d_thickness_mm: float = 0.0,
) -> float:
    """First-order ``dL = dT/sin(theta) - L cot(theta) d(theta)``.

    Holds for straight fascicles between locally parallel aponeuroses.
    It is not an identity for curved fascicles.
    """
    theta = math.radians(theta_deg)
    d_theta = math.radians(d_theta_deg)
    s = math.sin(theta)
    if abs(s) < 1e-8:
        raise ValueError("pennation too close to 0 or 180 for the sine model")
    return (d_thickness_mm / s) - length_mm * (math.cos(theta) / s) * d_theta


def explained_length_residual_mm(
    e_theta_deg: float,
    e_thickness_mm: float,
    theta_star_deg: float,
    length_star_mm: float,
) -> float:
    """Right-hand side of the signed residual identity. Not a submission feature."""
    return first_order_length_change_mm(
        length_star_mm,
        theta_star_deg,
        e_theta_deg,
        length_star_mm * math.sin(math.radians(theta_star_deg)),
        e_thickness_mm,
    )
