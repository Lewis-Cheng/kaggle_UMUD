"""Geometry head + baseline classical path contracts (no real ultrasound)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from umud.models.baseline import SegGeometryBaseline
from umud.models.geometry_head import GeometryHead


def test_geometry_head_measures_from_tangents_and_separation():
    head = GeometryHead(min_perp_component=1e-3, fail_closed=True)
    # Fascicle (1,1) px, deep apo horizontal (1,0), sx=sy=1 → PA=45°.
    # separation 10 mm, fascicle unit phys ∝ (1,1), apo normal (0,1)
    # → FL = 10 / sin(45°) = 10√2.
    structure = {
        "fascicle_tangent_px": (1.0, 1.0),
        "deep_apo_tangent_px": (1.0, 0.0),
        "separation_mm": 10.0,
        "fascicle_unit_phys": (1.0, 1.0),
        "apo_normal_unit_phys": (0.0, 1.0),
    }
    result = head.measure(structure, sx=1.0, sy=1.0)
    assert result.ok
    assert result.pa_deg == pytest.approx(45.0, abs=1e-6)
    assert result.mt_mm == pytest.approx(10.0, abs=1e-9)
    assert result.fl_mm == pytest.approx(10.0 * math.sqrt(2.0), rel=1e-9)


def test_geometry_head_fail_closed_on_near_parallel():
    head = GeometryHead(min_perp_component=1e-3, fail_closed=True)
    structure = {
        "fascicle_tangent_px": (1.0, 0.0),
        "deep_apo_tangent_px": (1.0, 0.0),
        "separation_mm": 10.0,
        "fascicle_unit_phys": (1.0, 1e-6),
        "apo_normal_unit_phys": (0.0, 1.0),
    }
    result = head.measure(structure, sx=1.0, sy=1.0)
    assert not result.ok
    assert result.pa_deg is None
    assert result.failure_reason is not None


def test_baseline_segment_not_implemented_but_classical_path_works():
    baseline = SegGeometryBaseline({"fail_closed": True})
    with pytest.raises(NotImplementedError, match="calibrated segmentation"):
        baseline.segment(np.zeros((8, 8)))

    structure = {
        "fascicle_tangent_px": (1.0, 1.0),
        "deep_apo_tangent_px": (1.0, 0.0),
        "thickness_vector_px": (0.0, 5.0),
        "fascicle_length_px": (10.0, 10.0),
    }
    # sx=1, sy=2 → thickness physical = 10 mm; FL = ||(10,20)|| = √500
    result = baseline.classical_geometry(structure, sx=1.0, sy=2.0)
    assert result.ok
    assert result.mt_mm == pytest.approx(10.0, abs=1e-9)
    assert result.fl_mm == pytest.approx(math.sqrt(10.0**2 + 20.0**2), rel=1e-9)
    assert result.pa_deg == pytest.approx(
        math.degrees(math.atan2(2.0, 1.0)), abs=1e-6
    )
