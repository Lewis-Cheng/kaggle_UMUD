#!/usr/bin/env python3
"""Fit simple affine offsets on MultExpData expert consensus; write calibrated submission."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.calibration.anisotropic_scale import mm_per_prob_pixel  # noqa: E402
from umud.geometry.host_faithful import measure  # noqa: E402
from umud.seg.frundles_smp import load_pair, predict_prob  # noqa: E402

RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def fit_affine(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Least-squares y ≈ a*x + b; fall back to identity if degenerate."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    m = np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 5:
        return 1.0, 0.0
    a, b = np.polyfit(pred[m], gt[m], 1)
    return float(a), float(b)


def apply_affine(x, a, b):
    return a * x + b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--expert-csv",
        type=Path,
        default=ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv",
    )
    ap.add_argument(
        "--expert-dir",
        type=Path,
        default=ROOT
        / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0",
    )
    ap.add_argument(
        "--floor-csv",
        type=Path,
        default=ROOT / "reports/phase3/chase018/submission_geo_floor.csv",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "reports/phase3/chase018/submission_geo_floor_cal.csv",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "reports/phase3/chase018/multexp_affine.json",
    )
    ap.add_argument("--weights", type=Path, default=ROOT / "data/external/frundles_weights")
    ap.add_argument("--fl-factor", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    expert = pd.read_csv(args.expert_csv)
    device = args.device
    pair = load_pair(args.weights, device=device)
    m_apo, s_apo = pair["apo"]
    m_fasc, s_fasc = pair["fasc"]

    rows = []
    for _, er in expert.iterrows():
        path = args.expert_dir / er["image_file"]
        if not path.exists():
            print("missing", path)
            continue
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print("unreadable", path)
            continue
        H0, W0 = g.shape
        # Expert scale is isotropic px/cm → px/mm
        px_per_mm = float(er["Scale_pixel_per_cm"]) / 10.0
        pa = predict_prob(m_apo, s_apo, g, device)
        pf = predict_prob(m_fasc, s_fasc, g, device)
        h, w = pa.shape
        mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
        res = measure(pa, pf, mx, my)
        rows.append(
            dict(
                image_file=er["image_file"],
                ok=res["ok"],
                pred_pa=res["pa_deg"],
                pred_fl=res["fl_mm"] * args.fl_factor,
                pred_mt=res["mt_mm"],
                gt_pa=er["median_PA"],
                gt_fl=er["median_FL"],
                gt_mt=er["median_MT"],
            )
        )
    cal = pd.DataFrame(rows)
    cal_path = args.out_json.with_name("multexp_pred_vs_gt.csv")
    cal.to_csv(cal_path, index=False)
    print("wrote", cal_path, "n", len(cal), "ok", int(cal.ok.sum()))

    fits = {}
    for name, pk, gk in (
        ("pa_deg", "pred_pa", "gt_pa"),
        ("fl_mm", "pred_fl", "gt_fl"),
        ("mt_mm", "pred_mt", "gt_mt"),
    ):
        a, b = fit_affine(cal[pk].values, cal[gk].values)
        mae0 = float(np.nanmean(np.abs(cal[pk] - cal[gk])))
        mae1 = float(np.nanmean(np.abs(apply_affine(cal[pk], a, b) - cal[gk])))
        fits[name] = {"a": a, "b": b, "mae_before": mae0, "mae_after": mae1}
        print(f"{name}: a={a:.4f} b={b:.4f} MAE {mae0:.3f} → {mae1:.3f}")

    args.out_json.write_text(json.dumps(fits, indent=2))
    print("wrote", args.out_json)

    floor = pd.read_csv(args.floor_csv)
    out = floor.copy()
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        a, b = fits[col]["a"], fits[col]["b"]
        out[col] = apply_affine(out[col].values, a, b)
        lo, hi = RANGES[col]
        out[col] = out[col].clip(lo, hi)
    out.to_csv(args.out_csv, index=False, lineterminator="\n")
    print("wrote", args.out_csv)
    print(out.describe().round(2))


if __name__ == "__main__":
    main()
