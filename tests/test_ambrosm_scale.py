"""Unit tests for AmbrosM family scale parsers (synthetic HUD fingerprints)."""

from __future__ import annotations

import numpy as np
import pytest

from umud.calibration.ambrosm_scale import classify_family, estimate_scale
from umud.calibration.provenance import ScaleProvenance


def test_classify_png_family():
    assert classify_family(height=500, width=400, ndim=3, suffix=".png") == "png_lumify"


def test_644x1088_family_prior():
    a = np.zeros((644, 1088, 3), dtype=np.uint8)
    est = estimate_scale("dummy_644.tif", array=a)
    assert est.family == "rgb_644x1088"
    assert est.provenance == ScaleProvenance.FAMILY_PRIOR.value
    assert est.px_per_cm == pytest.approx(630.5 / 5.0)
    assert est.px_per_mm == pytest.approx((630.5 / 5.0) / 10.0)
    assert est.mm_per_px == pytest.approx(10.0 / (630.5 / 5.0))
    assert est.assumed_mm_per_tick == 10.0
    assert est.q_provenance == ScaleProvenance.FAMILY_PRIOR.value
    assert "do_not_use_tiff_dpi" in est.notes
    assert est.fov_l == 140 and est.fov_r == 947


def test_800x1200_left_family_prior():
    a = np.zeros((800, 1200, 3), dtype=np.uint8)
    a[42, 67:74, 0] = 200
    est = estimate_scale("dummy_left.tif", array=a)
    assert est.family == "rgb_800x1200_left_ticks"
    assert est.provenance == ScaleProvenance.FAMILY_PRIOR.value
    assert est.px_per_cm == pytest.approx((783 - 42) / 5.0)


def test_800x1200_right_measured():
    a = np.zeros((800, 1200, 3), dtype=np.uint8)
    a[87, 1147:1157] = 175
    # ticks every 50 px from y=87 → n≈14 over 700 px span → (787-87)/50=14
    for y in range(87, 800, 50):
        a[y, 1150, :] = 200
    # last bright near bottom
    a[787, 1150, :] = 200
    est = estimate_scale("dummy_right.tif", array=a)
    assert est.family == "rgb_800x1200_right_ticks"
    assert est.provenance == ScaleProvenance.MEASURED.value
    assert est.px_per_mm is not None and est.px_per_mm > 0
    assert est.assumed_mm_per_tick in (2.0, 5.0)
    assert est.q_provenance == ScaleProvenance.FAMILY_PRIOR.value


def test_unknown_shape():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    est = estimate_scale("odd.tif", array=a)
    assert est.provenance == ScaleProvenance.UNKNOWN.value
    assert est.px_per_mm is None
    assert "do_not_use_tiff_dpi" in est.notes


def test_h853_fingerprint():
    a = np.zeros((853, 1000), dtype=np.uint8)
    a[-5, 100] = 170
    a[-5, 934] = 170
    est = estimate_scale("g853.tif", array=a)
    assert est.family == "gray_853_bottom_ticks"
    assert est.provenance == ScaleProvenance.FAMILY_PRIOR.value
    assert est.px_per_cm == pytest.approx((934 - 100) / 5.0)


def test_800x1200_right_hflip_measured():
    """Train-style horizontal flip: right ticks at col 49."""
    a = np.zeros((800, 1200, 3), dtype=np.uint8)
    a[87, 43:53] = 175
    for y in range(87, 800, 50):
        a[y, 49, :] = 200
    a[787, 49, :] = 200
    est = estimate_scale("dummy_hflip.tif", array=a)
    assert est.family == "rgb_800x1200_right_ticks_hflip"
    assert est.provenance == ScaleProvenance.MEASURED.value
    assert est.px_per_mm is not None and est.px_per_mm > 0
    assert est.assumed_mm_per_tick in (2.0, 5.0)
    assert est.q_provenance == ScaleProvenance.FAMILY_PRIOR.value
    assert "hflip_train" in est.notes
    assert est.diagnostics.get("tick_col") == 49


def test_800x1200_left_hflip_family_prior():
    a = np.zeros((800, 1200, 3), dtype=np.uint8)
    a[42, 1126:1133, 0] = 200
    est = estimate_scale("dummy_left_hflip.tif", array=a)
    assert est.family == "rgb_800x1200_left_ticks_hflip"
    assert est.provenance == ScaleProvenance.FAMILY_PRIOR.value
    assert est.px_per_cm == pytest.approx((783 - 42) / 5.0)
    assert "hflip_train" in est.notes


def test_classify_hflip_family():
    a = np.zeros((800, 1200, 3), dtype=np.uint8)
    a[87, 43:53] = 200
    assert classify_family(height=800, width=1200, ndim=3, suffix=".tif", array=a) == (
        "rgb_800x1200_right_ticks_hflip"
    )
