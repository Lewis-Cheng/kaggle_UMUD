#!/usr/bin/env python3
"""Phase 3 floor A — train median of mask proxies → constant test submission.

Uses frozen ``split_assignments.jsonl`` train rows when provided. Mask-derived
medians are Tier A proxies (NOT Tier B / leaderboard). Scale-dependent mm
values require ``--scale-provenance`` (e.g. unknown).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import (  # noqa: E402
    load_split_assignments,
    load_test_ids,
    require_scale_provenance,
    write_submission_csv,
)
from umud.baselines.mask_proxy import (  # noqa: E402
    FALLBACK_PRIORS,
    apply_isotropic_mm,
    load_mask,
    proxy_apo_peak_gap,
    proxy_fascicle_pca,
)
from umud.data.layout import discover_layout  # noqa: E402


def _mask_path_for_stem(folder: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    # case-insensitive scan (slow path)
    for p in folder.iterdir():
        if p.is_file() and p.stem == stem:
            return p
    return None


def compute_train_medians(
    *,
    data_root: Path,
    split_path: Path | None,
    mm_per_px: float,
) -> dict[str, float]:
    layout = discover_layout(data_root)
    pa_vals: list[float] = []
    fl_vals: list[float] = []
    mt_vals: list[float] = []

    if split_path is not None and split_path.is_file():
        rows = load_split_assignments(split_path)
        train = [r for r in rows if r.get("split") == "train"]
    else:
        # All train images as rows
        from umud.data.layout import list_images

        train = []
        for p in list_images(layout.fasc_imgs):
            train.append({"task": "fascicle", "image_id": p.stem, "split": "train"})
        for p in list_images(layout.apo_imgs):
            train.append({"task": "aponeurosis", "image_id": p.stem, "split": "train"})

    for row in train:
        stem = str(row["image_id"])
        task = str(row.get("task", ""))
        if task == "fascicle":
            mp = _mask_path_for_stem(layout.fasc_masks, stem)
            if mp is None:
                continue
            info = proxy_fascicle_pca(load_mask(mp))
            if info is None:
                continue
            scaled = apply_isotropic_mm(
                fl_length_px=info["fl_length_px"],
                pa_deg=info["pa_deg_proxy"],
                mt_gap_px=None,
                mm_per_px=mm_per_px,
            )
            if scaled["pa_deg"] is not None:
                pa_vals.append(float(scaled["pa_deg"]))
            if scaled["fl_mm"] is not None:
                fl_vals.append(float(scaled["fl_mm"]))
        elif task == "aponeurosis":
            mp = _mask_path_for_stem(layout.apo_masks, stem)
            if mp is None:
                continue
            info = proxy_apo_peak_gap(load_mask(mp))
            if info is None:
                continue
            scaled = apply_isotropic_mm(
                fl_length_px=None,
                pa_deg=None,
                mt_gap_px=float(info["mt_gap_px"]),
                mm_per_px=mm_per_px,
            )
            if scaled["mt_mm"] is not None:
                mt_vals.append(float(scaled["mt_mm"]))

    medians = {
        "pa_deg": float(np.median(pa_vals)) if pa_vals else float(FALLBACK_PRIORS["pa_deg"]),
        "fl_mm": float(np.median(fl_vals)) if fl_vals else float(FALLBACK_PRIORS["fl_mm"]),
        "mt_mm": float(np.median(mt_vals)) if mt_vals else float(FALLBACK_PRIORS["mt_mm"]),
        "n_pa": len(pa_vals),
        "n_fl": len(fl_vals),
        "n_mt": len(mt_vals),
    }
    return medians


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mm-per-px", type=float, default=0.07225)
    parser.add_argument(
        "--scale-provenance",
        type=str,
        required=True,
        help="required label, e.g. unknown (never treat TIFF dpi as real)",
    )
    parser.add_argument("--delimiter", type=str, default=",")
    args = parser.parse_args()

    provenance = require_scale_provenance(args.scale_provenance)
    if args.mm_per_px <= 0:
        print("ERROR: --mm-per-px must be positive", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    medians = compute_train_medians(
        data_root=args.data_root,
        split_path=args.split_assignments,
        mm_per_px=args.mm_per_px,
    )
    test_ids = load_test_ids(args.data_root)
    rows = [
        {
            "image_id": tid,
            "pa_deg": medians["pa_deg"],
            "fl_mm": medians["fl_mm"],
            "mt_mm": medians["mt_mm"],
        }
        for tid in test_ids
    ]
    sub_path = out_dir / "submission_median_prior.csv"
    write_submission_csv(sub_path, rows, delimiter=args.delimiter, validate_against=test_ids)

    report = {
        "baseline": "median_prior",
        "tier": "A_proxy_not_tier_b",
        "mm_per_px": args.mm_per_px,
        "scale_provenance": provenance,
        "medians": {k: medians[k] for k in ("pa_deg", "fl_mm", "mt_mm")},
        "n_train_proxy_values": {k: medians[k] for k in ("n_pa", "n_fl", "n_mt")},
        "n_test": len(test_ids),
        "submission": str(sub_path),
        "notes": [
            "Constant prior from train-split mask-proxy medians (or FALLBACK_PRIORS).",
            "NOT a leaderboard claim. Scale may be unknown.",
        ],
    }
    (out_dir / "median_prior_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"median_prior wrote {sub_path} "
        f"pa={medians['pa_deg']:.3f} fl={medians['fl_mm']:.3f} "
        f"mt={medians['mt_mm']:.3f} provenance={provenance}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
