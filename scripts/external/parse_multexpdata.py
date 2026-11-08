#!/usr/bin/env python3
"""Parse UMUD MultExpData architecture Results xlsx → expert_consensus.csv (median R1–R7)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="Manual_architecture")
    df.columns = [str(c).strip() for c in df.columns]
    raters = [f"R{i}" for i in range(1, 8)]
    metrics = ["MT", "FL", "PA"]
    rows = []
    for _, r in df.iterrows():
        row = {
            "ImageID": r["ImageID"],
            "image_file": f"{r['ImageID']}.tif",
            "Scale_pixel_per_cm": float(r["Scale_pixel_per_cm"]),
            "q_mm_per_px": 10.0 / float(r["Scale_pixel_per_cm"]),
        }
        for m in metrics:
            vals = []
            for ri in raters:
                col = f"{ri}_{m}"
                if col in df.columns and pd.notna(r[col]):
                    vals.append(float(r[col]))
                row[col] = float(r[col]) if col in df.columns and pd.notna(r[col]) else np.nan
            arr = np.asarray(vals, dtype=float)
            row[f"n_raters_{m}"] = int(arr.size)
            row[f"median_{m}"] = float(np.median(arr)) if arr.size else np.nan
            row[f"mean_{m}"] = float(np.mean(arr)) if arr.size else np.nan
            if arr.size:
                q25, q75 = np.percentile(arr, [25, 75])
                row[f"iqr_{m}"] = float(q75 - q25)
            else:
                row[f"iqr_{m}"] = np.nan
        for prefix in ("SMA", "DLTrack"):
            for m in metrics:
                col = f"{prefix}_{m}"
                row[col] = float(r[col]) if col in df.columns and pd.notna(r[col]) else np.nan
        rows.append(row)
    out_df = pd.DataFrame(rows)
    front = [
        "ImageID", "image_file", "Scale_pixel_per_cm", "q_mm_per_px",
        "median_MT", "median_FL", "median_PA",
        "mean_MT", "mean_FL", "mean_PA",
        "iqr_MT", "iqr_FL", "iqr_PA",
        "n_raters_MT", "n_raters_FL", "n_raters_PA",
    ]
    rater_cols = [f"R{i}_{m}" for i in range(1, 8) for m in metrics]
    base_cols = [f"{p}_{m}" for p in ("SMA", "DLTrack") for m in metrics]
    cols = front + [c for c in rater_cols if c in out_df.columns] + [c for c in base_cols if c in out_df.columns]
    return out_df[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xlsx",
        type=Path,
        default=Path("data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0/Results_benchmark_architecture_v0.1.0.xlsx"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"),
    )
    args = ap.parse_args()
    out_df = parse(args.xlsx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {args.out} n_rows={len(out_df)}")


if __name__ == "__main__":
    main()
