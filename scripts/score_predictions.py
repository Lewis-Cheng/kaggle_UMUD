#!/usr/bin/env python3
"""Compute pinned official S/S1/S2/S3 and per-target MAE/MedAE/RMSE.

Inputs are prediction and ground-truth CSVs with columns:
  id, pa_deg, fl_mm, mt_mm

Does not invent competition scores — both CSVs must be real files you provide.
Join policy for pred rows whose id is absent from gt:
  --on-extra reject  (default; local strict)
  --on-extra ignore  (notebook-like drop of extras before scoring)

Empty join after alignment exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umud.metric.score import (  # noqa: E402
    PINNED_NOTEBOOK_SHA256,
    score_prediction_frames,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_csv", type=Path)
    parser.add_argument("gt_csv", type=Path)
    parser.add_argument(
        "--on-extra",
        choices=("reject", "ignore"),
        default="reject",
        help="policy for pred ids absent from gt (default: reject)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit full report as JSON",
    )
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    gt = pd.read_csv(args.gt_csv)
    print(f"pinned_scorer_sha256: {PINNED_NOTEBOOK_SHA256}")

    try:
        report = score_prediction_frames(pred, gt, on_extra=args.on_extra)
    except (ValueError, KeyError) as exc:
        print(f"SCORE_FAILED: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, default=float))
        return 0

    comps = report["components"]
    print(
        f"n_scored={report['n_scored']} ignored_extra_n={report['ignored_extra_n']} "
        f"on_extra={report['on_extra']}"
    )
    print(
        f"S={comps['S']:.12f}  S1={comps['S1']:.8f}  "
        f"S2={comps['S2']:.8f}  S3={comps['S3']:.8f}"
    )
    for key, stats in report["per_target"].items():
        print(
            f"{key}: MAE={stats['MAE']:.6f} MedAE={stats['MedAE']:.6f} "
            f"RMSE={stats['RMSE']:.6f} | "
            f"MAE/tol={stats['MAE_over_tol']:.6f} "
            f"MedAE/tol={stats['MedAE_over_tol']:.6f} "
            f"RMSE/tol={stats['RMSE_over_tol']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
