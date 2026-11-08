"""Fixture-style geometry contracts (no real ultrasound images)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from umud.geometry.physical import (
    IntersectionRejected,
    director_from_theta,
    near_parallel_intersection_length,
    physical_angle_to_horizontal_deg,
    theta_from_director,
)


def test_anisotropic_pixel_tangent_angle_to_horizontal():
    """(1,1) px with spacings (1,2) → physical angle ≈ 63.4349488° to horizontal."""
    angle = physical_angle_to_horizontal_deg((1.0, 1.0), sx=1.0, sy=2.0)
    assert angle == pytest.approx(63.4349488, abs=1e-6)
    # Sanity: square pixels still give 45°.
    assert physical_angle_to_horizontal_deg((1.0, 1.0), sx=1.0, sy=1.0) == pytest.approx(
        45.0, abs=1e-9
    )


def test_near_parallel_intersection_rejection():
    """Near-zero perpendicular component must reject, not silently explode."""
    # Fascicle nearly parallel to apo (normal is +y); tiny ε.
    with pytest.raises(IntersectionRejected):
        near_parallel_intersection_length(
            separation_mm=10.0,
            fascicle_unit_phys=(1.0, 1e-6),
            apo_normal_unit_phys=(0.0, 1.0),
            min_perp_component=1e-3,
        )
    # Well-conditioned case: 45° to apo → length = d / sin(45°) = d√2.
    length = near_parallel_intersection_length(
        separation_mm=10.0,
        fascicle_unit_phys=(1.0, 1.0),
        apo_normal_unit_phys=(0.0, 1.0),
        min_perp_component=1e-3,
    )
    assert length == pytest.approx(10.0 * math.sqrt(2.0), rel=1e-9)


def test_orientation_doubled_angle_invariance_for_180_flip():
    """Director (cos 2θ, sin 2θ) is unchanged under θ → θ+π (direction flip)."""
    theta = math.radians(37.5)
    d0 = director_from_theta(theta)
    d1 = director_from_theta(theta + math.pi)
    np.testing.assert_allclose(d0, d1, atol=1e-12)
    # Recovered θ matches modulo π (same undirected line).
    t_rec = theta_from_director(d0)
    # Compare undirected orientations via director equality again.
    np.testing.assert_allclose(director_from_theta(t_rec), d0, atol=1e-12)
