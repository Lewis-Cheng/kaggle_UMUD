#!/usr/bin/env python3
"""One real-image vertical anchor attempt. BLOCKED is a valid result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_official_submission_ids  # noqa: E402
from umud.calibration.anisotropic_scale import (  # noqa: E402
    best_ruler,
    classify_layout,
    ocr_text,
    parse_depth_mm,
)
from umud.calibration.gyy_anchor import assess_vertical_anchor  # noqa: E402

TEST_DIR = ROOT / "data/raw/umud/test_images_v2/test_set_v2"
OUT_JSON = ROOT / "reports/phase3/chase018/gyy_anchor_once.json"


def _first_png_id() -> str:
    ids = load_official_submission_ids(ROOT / "reports/phase3/official_test_file_names.txt")
    for image_id in ids:
        if image_id.lower().endswith(".png"):
            return image_id
    raise SystemExit("no png id in the official list")


def _measure(path: Path) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {
            "layout": "unreadable",
            "ocr": "",
            "depth_mm": None,
            "y0": None,
            "y1": None,
            "ocr_error": "imread failed",
        }
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    layout = classify_layout(path, gray)
    h, w = gray.shape
    ocr = ""
    ocr_error = None
    depth = float("nan")
    y0 = float("nan")
    y1 = float("nan")
    try:
        if layout == "lumify":
            ocr = ocr_text(bgr[60:600, w - 120 : w])
            _spacing, _n, y0, y1 = best_ruler(gray, "y", (0, 30), (90, h - 40))
            depth = parse_depth_mm(ocr)
        elif layout == "siemens":
            ocr = ocr_text(bgr[90 : h - 60, 960:1100])
            _spacing, _n, y0, y1 = best_ruler(gray, "y", (w - 80, w), (40, h - 40))
            depth = parse_depth_mm(ocr)
        else:
            ocr_error = f"layout {layout} has no OCR depth path"
    except Exception as exc:  # OCR runtime failure is a missing endpoint, not a metric
        ocr_error = f"{type(exc).__name__}: {exc}"
    return {
        "layout": layout,
        "ocr": ocr,
        "depth_mm": None if depth != depth else float(depth),
        "y0": None if y0 != y0 else float(y0),
        "y1": None if y1 != y1 else float(y1),
        "ocr_error": ocr_error,
    }


def main() -> int:
    image_id = _first_png_id()
    path = TEST_DIR / image_id
    if not path.is_file():
        report = assess_vertical_anchor(
            image_id=image_id,
            y0=None,
            y1=None,
            depth_mm=None,
            ocr="",
            layout="missing",
            provenance="file-missing",
        )
        report["ocr_error"] = f"missing file {path}"
    else:
        measured = _measure(path)
        report = assess_vertical_anchor(
            image_id=image_id,
            y0=measured["y0"],
            y1=measured["y1"],
            depth_mm=measured["depth_mm"],
            ocr=measured["ocr"],
            layout=measured["layout"],
            provenance="anisotropic_scale.ocr_depth_vertical_ruler",
        )
        report["ocr_error"] = measured["ocr_error"]
        report["image_path"] = str(path)
        report["endpoint_note"] = (
            "Depth text and vertical tick span come from the lumify/siemens "
            "path in anisotropic_scale. That pairing is not a second annotation "
            "of the same endpoints."
        )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "image_id", "layout", "reasons", "partial_gyy", "ocr_error")}, indent=2))
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
