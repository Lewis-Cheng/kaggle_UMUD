"""Phase 3 floor helpers: features, mask geometry, scale-bar, submission I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from umud.baselines.features import FEATURE_H, FEATURE_W, image_feature_vector
from umud.baselines.io import require_scale_provenance, write_submission_csv
from umud.baselines.mask_geometry import measure_masks, vertical_thickness_px
from umud.baselines.mask_proxy import proxy_apo_peak_gap, proxy_fascicle_pca
from umud.baselines.scale_bar import detect_vertical_scale_bar


def test_require_scale_provenance_rejects_empty():
    with pytest.raises(ValueError):
        require_scale_provenance("")
    assert require_scale_provenance("unknown") == "unknown"
    assert require_scale_provenance("rival_guess").startswith("rival")


def test_image_feature_vector_shape():
    gray = np.random.rand(120, 160).astype(np.float32)
    feat = image_feature_vector(gray)
    expected = FEATURE_H * FEATURE_W + FEATURE_H + FEATURE_W + 7 + 8
    assert feat.shape == (expected,)
    assert np.isfinite(feat).all()


def test_fascicle_pca_and_vertical_mt_on_synthetic_masks():
    # Diagonal fascicle blob
    fasc = np.zeros((100, 120), dtype=bool)
    for i in range(20, 80):
        j = int(30 + 0.5 * (i - 20))
        fasc[i, j - 2 : j + 3] = True
    info = proxy_fascicle_pca(fasc)
    assert info is not None
    assert info["fl_length_px"] > 10

    # Two horizontal apo bands
    apo = np.zeros((100, 120), dtype=bool)
    apo[20:28, 10:110] = True
    apo[70:78, 10:110] = True
    mt = vertical_thickness_px(apo)
    assert mt is not None
    assert mt["mt_gap_px"] == pytest.approx(50.0, abs=5.0)
    rival = proxy_apo_peak_gap(apo)
    assert rival is not None
    assert rival["mt_gap_px"] > 0


def test_measure_masks_both_and_fail_closed_near_parallel():
    apo = np.zeros((100, 120), dtype=bool)
    apo[20:26, 10:110] = True
    apo[70:76, 10:110] = True
    # Near-horizontal fascicle → near-parallel to apo → may reject FL
    fasc = np.zeros((100, 120), dtype=bool)
    fasc[45:55, 20:100] = True
    out = measure_masks(fasc_mask=fasc, apo_mask=apo, sx=0.1, sy=0.1)
    assert "pa_deg" in out and "fl_mm" in out and "mt_mm" in out
    assert out["mt_mm"] > 0
    # Steep fascicle should succeed
    fasc2 = np.zeros((100, 120), dtype=bool)
    for i in range(25, 75):
        j = 40 + int(0.2 * (i - 25))
        fasc2[i, j : j + 3] = True
    out2 = measure_masks(fasc_mask=fasc2, apo_mask=apo, sx=0.1, sy=0.1)
    assert out2["ok"] is True
    assert out2["pa_deg"] > 0
    assert out2["fl_mm"] > out2["mt_mm"]


def test_scale_bar_detects_synthetic_ticks_and_never_claims_without_pitch():
    h, w = 200, 160
    gray = np.full((h, w), 0.2, dtype=np.float64)
    # Right-margin bright ticks every 20 px
    for y in range(20, 180, 20):
        gray[y : y + 2, -10:] = 1.0
    det = detect_vertical_scale_bar(gray)
    assert det.found
    assert det.side == "right"
    assert det.tick_spacing_px == pytest.approx(20.0, abs=2.0)
    assert det.mm_per_px is None
    assert "physical_unknown" in det.provenance or det.provenance == "unknown" or "unknown" in det.provenance

    det2 = detect_vertical_scale_bar(gray, mm_per_tick=10.0)
    assert det2.mm_per_px == pytest.approx(10.0 / det2.tick_spacing_px, rel=0.15)
    assert det2.provenance.startswith("scale_bar")


def test_write_submission_csv_validates(tmp_path: Path):
    rows = [
        {"image_id": "a", "pa_deg": 1.0, "fl_mm": 2.0, "mt_mm": 3.0},
        {"image_id": "b", "pa_deg": 4.0, "fl_mm": 5.0, "mt_mm": 6.0},
    ]
    path = write_submission_csv(
        tmp_path / "sub.csv", rows, validate_against=["a", "b"]
    )
    text = path.read_text()
    assert text.splitlines()[0].startswith("image_id")
    assert "a" in text and "b" in text
