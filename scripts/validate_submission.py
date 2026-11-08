#!/usr/bin/env python3
"""Validate a submission CSV against an exact expected ID set.

Optionally score against a ground-truth CSV using the pinned official formulas
(S / S1 / S2 / S3). Scoring never invents labels; gt must be supplied by you.

Extra-row policy reminder: this validator REJECTS extras (stronger than the
organizer notebook's inner join). See umud.metric.submission docstring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umud.metric.score import score_prediction_frames  # noqa: E402
from umud.metric.submission import (  # noqa: E402
    SubmissionValidationError,
    load_expected_ids,
    validate_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_csv", type=Path)
    parser.add_argument(
        "--expected-ids",
        type=Path,
        required=True,
        help="text file with one image id per line",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="optional ground-truth CSV (id, pa_deg, fl_mm, mt_mm) to score against",
    )
    parser.add_argument(
        "--on-extra",
        choices=("reject", "ignore"),
        default="reject",
        help="scoring join policy for pred ids absent from gt (default: reject)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print score report as JSON when --gt-csv is set",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.submission_csv)
    expected = load_expected_ids(args.expected_ids)
    try:
        validate_submission(df, expected)
    except SubmissionValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(expected)} ids, finite preds, exact ID set")

    if args.gt_csv is None:
        return 0

    gt = pd.read_csv(args.gt_csv)
    try:
        report = score_prediction_frames(df, gt, on_extra=args.on_extra)
    except (ValueError, KeyError) as exc:
        print(f"SCORE_FAILED: {exc}", file=sys.stderr)
        return 1

    comps = report["components"]
    if args.json:
        print(json.dumps(report, indent=2, default=float))
    else:
        print(
            f"SCORE n={report['n_scored']} "
            f"S={comps['S']:.10f} S1={comps['S1']:.6f} "
            f"S2={comps['S2']:.6f} S3={comps['S3']:.6f} "
            f"on_extra={report['on_extra']} ignored_extra_n={report['ignored_extra_n']}"
        )
        for key, stats in report["per_target"].items():
            print(
                f"  {key}: MAE={stats['MAE']:.4f} MedAE={stats['MedAE']:.4f} "
                f"RMSE={stats['RMSE']:.4f} (tol={stats['tol']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
