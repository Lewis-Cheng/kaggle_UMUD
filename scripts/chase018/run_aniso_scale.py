#!/usr/bin/env python3
"""Compute Frundles anisotropic sx/sy for all test (and optional train) images."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.calibration.anisotropic_scale import estimate_anisotropic_scales  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test-dir",
        type=Path,
        default=ROOT / "data/raw/umud/test_images_v2/test_set_v2",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/phase3/chase018/scale_aniso.csv",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(args.test_dir.glob("IMG_*"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no images in {args.test_dir}")
    print(f"measuring {len(files)} images …", flush=True)
    df = estimate_anisotropic_scales(files, show_progress=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("wrote", args.out)
    print(df.groupby("layout").agg(n=("sx", "size"), sx=("sx", "median"), sy=("sy", "median"), aspect=("aspect", "median")).round(3))
    missing = df[["sx", "sy"]].isna().any(axis=1).sum()
    print("missing sx/sy rows:", int(missing))


if __name__ == "__main__":
    main()
