"""Output-contract checks for official ids, calibration, the scale probe, and anchors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from umud.baselines.io import load_official_submission_ids
from umud.calibration.executable import CalibrationSpecError, require_executable_calibration
from umud.calibration.gyy_anchor import assess_vertical_anchor
from umud.calibration.scale_probe import probe_factor_from_median_ratios


def test_official_ids_copy_filename_and_reject_collisions(tmp_path: Path):
    path = tmp_path / "names.txt"
    path.write_text(
        "sample_submission.csv\n"
        "test_images_v2/test_set_v2/IMG_00001.tif\n"
        "test_images_v2/test_set_v2/IMG_00002.png\n",
        encoding="utf-8",
    )
    assert load_official_submission_ids(path) == ["IMG_00001.tif", "IMG_00002.png"]

    path.write_text(
        "a/IMG_00001.tif\nb/IMG_00001.tif\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collide"):
        load_official_submission_ids(path)


def test_calibration_spec_keeps_fl_identity_and_aborts_on_fl_affine():
    manifest = {
        "pa_deg": {"function": "affine", "applied": True, "a": 0.5, "b": 1.0},
        "fl_mm": {"function": "identity", "applied": False, "a": 0.9, "b": 3.0},
        "mt_mm": {"function": "affine", "applied": True, "a": 1.1, "b": 0.2},
    }
    coeff = require_executable_calibration(manifest)
    assert coeff["pa_deg"] == (0.5, 1.0)
    assert coeff["fl_mm"] is None
    assert coeff["mt_mm"] == (1.1, 0.2)

    manifest["fl_mm"]["function"] = "affine"
    manifest["fl_mm"]["applied"] = True
    with pytest.raises(CalibrationSpecError, match="fl_mm"):
        require_executable_calibration(manifest)


def test_scale_probe_uses_shared_sign_and_refuses_disagreement():
    assert probe_factor_from_median_ratios(0.997, 0.999) == pytest.approx(0.97)
    assert probe_factor_from_median_ratios(1.01, 1.02) == pytest.approx(1.03)
    with pytest.raises(ValueError, match="do not agree"):
        probe_factor_from_median_ratios(0.99, 1.01)


def test_vertical_anchor_blocks_when_horizontal_length_is_unbounded():
    missing = assess_vertical_anchor(
        image_id="IMG_00252.png",
        y0=None,
        y1=None,
        depth_mm=None,
        ocr="",
        layout="lumify",
        provenance="test",
    )
    assert missing["status"] == "BLOCKED"
    assert "endpoints" in missing["reasons"]
    assert missing["fills_millimetres"] is False
    assert missing["partial_gyy"] is None

    identified_gyy = assess_vertical_anchor(
        image_id="IMG_00252.png",
        y0=10.0,
        y1=210.0,
        depth_mm=40.0,
        ocr="4 cm",
        layout="lumify",
        provenance="test",
    )
    assert identified_gyy["status"] == "BLOCKED"
    assert identified_gyy["H_rank"] == 1
    assert identified_gyy["partial_gyy"] == pytest.approx((40.0 / 200.0) ** 2)
    assert "feasible_interval" in identified_gyy["reasons"]
    assert identified_gyy["fills_millimetres"] is False


def test_checked_in_manifest_matches_executable_spec():
    manifest = json.loads(
        Path("reports/phase3/chase018/multexp_chase_affine.json").read_text(encoding="utf-8")
    )
    coeff = require_executable_calibration(manifest)
    assert coeff["fl_mm"] is None
    assert coeff["pa_deg"] is not None
    assert coeff["mt_mm"] is not None
