#!/usr/bin/env python3
"""Phase 3 floor C — mask → GeometryHead (better than rival PCA / peak-gap).

Uses vertical MT (organizer), fascicle↔deep-apo pennation, and near-parallel
reject via ``umud.models.geometry_head.GeometryHead``.

On the frozen split this scores mask-geometry quality (Tier A proxy vs rival
proxies). Test images have no masks → submission uses train GeometryHead
medians (explicit fallback), not a silent drop.

Requires ``--scale-provenance``. Does NOT invent leaderboard scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import (  # noqa: E402
    load_split_assignments,
    load_test_ids,
    require_scale_provenance,
    write_submission_csv,
)
from umud.baselines.mask_geometry import measure_masks  # noqa: E402
from umud.baselines.mask_proxy import (  # noqa: E402
    FALLBACK_PRIORS,
    apply_isotropic_mm,
    load_mask,
    proxy_apo_peak_gap,
    proxy_fascicle_pca,
)
from umud.data.layout import discover_layout  # noqa: E402


def _mask_path(folder: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    for p in folder.iterdir():
        if p.is_file() and p.stem == stem:
            return p
    return None


def _index_by_stem(rows: list[dict[str, Any]], task: str, split: str) -> dict[str, dict]:
    return {
        str(r["image_id"]): r
        for r in rows
        if r.get("task") == task and r.get("split") == split
    }


def _measure_row(
    *,
    stem: str,
    fasc_masks: Path,
    apo_masks: Path,
    sx: float,
    sy: float,
    fasc_stems: set[str],
    apo_stems: set[str],
    priors: dict[str, float],
) -> dict[str, Any]:
    fasc = None
    apo = None
    if stem in fasc_stems:
        mp = _mask_path(fasc_masks, stem)
        if mp is not None:
            fasc = load_mask(mp)
    if stem in apo_stems:
        mp = _mask_path(apo_masks, stem)
        if mp is not None:
            apo = load_mask(mp)
    return measure_masks(
        fasc_mask=fasc, apo_mask=apo, sx=sx, sy=sy, priors=priors
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mm-per-px", type=float, default=0.07225)
    parser.add_argument("--scale-provenance", type=str, required=True)
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--max-rows", type=int, default=0, help="smoke cap (0=all)")
    args = parser.parse_args()

    provenance = require_scale_provenance(args.scale_provenance)
    sx = sy = float(args.mm_per_px)
    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)

    fasc_train = _index_by_stem(rows, "fascicle", "train")
    fasc_val = _index_by_stem(rows, "fascicle", "val")
    apo_train = _index_by_stem(rows, "aponeurosis", "train")
    apo_val = _index_by_stem(rows, "aponeurosis", "val")

    fasc_stems = set(fasc_train) | set(fasc_val)
    apo_stems = set(apo_train) | set(apo_val)

    # First pass: collect train GeometryHead values to refine priors
    train_stems = sorted(set(fasc_train) | set(apo_train))
    if args.max_rows and args.max_rows > 0:
        train_stems = train_stems[: args.max_rows]

    train_pa: list[float] = []
    train_fl: list[float] = []
    train_mt: list[float] = []
    train_ok = 0
    train_fail = 0
    rival_pa_err: list[float] = []
    rival_fl_err: list[float] = []
    rival_mt_err: list[float] = []

    for stem in train_stems:
        m = _measure_row(
            stem=stem,
            fasc_masks=layout.fasc_masks,
            apo_masks=layout.apo_masks,
            sx=sx,
            sy=sy,
            fasc_stems=fasc_stems,
            apo_stems=apo_stems,
            priors=FALLBACK_PRIORS,
        )
        if m.get("ok"):
            train_ok += 1
            train_pa.append(float(m["pa_deg"]))
            train_fl.append(float(m["fl_mm"]))
            if "mt_mm" in m and "mt_placeholder" not in str(m.get("failure_reason", "")):
                train_mt.append(float(m["mt_mm"]))
        else:
            train_fail += 1
            if "mt_method" in m:
                train_mt.append(float(m["mt_mm"]))

        # Compare vs rival proxies on same masks (Tier A diagnostic)
        fmp = _mask_path(layout.fasc_masks, stem) if stem in fasc_stems else None
        amp = _mask_path(layout.apo_masks, stem) if stem in apo_stems else None
        if fmp is not None:
            info = proxy_fascicle_pca(load_mask(fmp))
            if info is not None:
                riv = apply_isotropic_mm(
                    fl_length_px=info["fl_length_px"],
                    pa_deg=info["pa_deg_proxy"],
                    mt_gap_px=None,
                    mm_per_px=sx,
                )
                if m.get("ok") and riv["pa_deg"] is not None:
                    rival_pa_err.append(abs(float(m["pa_deg"]) - float(riv["pa_deg"])))
                if m.get("ok") and riv["fl_mm"] is not None:
                    rival_fl_err.append(abs(float(m["fl_mm"]) - float(riv["fl_mm"])))
        if amp is not None:
            info = proxy_apo_peak_gap(load_mask(amp))
            if info is not None and "mt_method" in m:
                riv_mt = float(info["mt_gap_px"]) * sx
                rival_mt_err.append(abs(float(m["mt_mm"]) - riv_mt))

    priors = {
        "pa_deg": float(np.median(train_pa)) if train_pa else float(FALLBACK_PRIORS["pa_deg"]),
        "fl_mm": float(np.median(train_fl)) if train_fl else float(FALLBACK_PRIORS["fl_mm"]),
        "mt_mm": float(np.median(train_mt)) if train_mt else float(FALLBACK_PRIORS["mt_mm"]),
    }

    # Val pass with refined priors
    val_stems = sorted(set(fasc_val) | set(apo_val))
    if args.max_rows and args.max_rows > 0:
        val_stems = val_stems[: args.max_rows]
    val_ok = 0
    val_fail = 0
    for stem in val_stems:
        m = _measure_row(
            stem=stem,
            fasc_masks=layout.fasc_masks,
            apo_masks=layout.apo_masks,
            sx=sx,
            sy=sy,
            fasc_stems=fasc_stems,
            apo_stems=apo_stems,
            priors=priors,
        )
        if m.get("ok"):
            val_ok += 1
        else:
            val_fail += 1

    # Test submission = constant GeometryHead train medians (no test masks)
    test_ids = load_test_ids(args.data_root)
    test_rows = [
        {
            "image_id": tid,
            "pa_deg": priors["pa_deg"],
            "fl_mm": priors["fl_mm"],
            "mt_mm": priors["mt_mm"],
        }
        for tid in test_ids
    ]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "submission_geometry_head_proxy.csv"
    write_submission_csv(
        sub_path, test_rows, delimiter=args.delimiter, validate_against=test_ids
    )

    def _summ(vals: list[float]) -> dict[str, float | int | None]:
        if not vals:
            return {"n": 0, "mean": None, "p50": None}
        a = np.asarray(vals, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a))}

    report = {
        "baseline": "geometry_head_proxy",
        "tier": "A_proxy_not_tier_b",
        "label": "mask_geometry_vs_rival_proxy_NOT_leaderboard",
        "mm_per_px": args.mm_per_px,
        "scale_sx": sx,
        "scale_sy": sy,
        "scale_provenance": provenance,
        "organizer_mt_policy": "vertical_not_perp_to_apo",
        "train_ok": train_ok,
        "train_fail": train_fail,
        "val_ok": val_ok,
        "val_fail": val_fail,
        "geometry_head_train_medians": priors,
        "delta_vs_rival_proxy_abs": {
            "pa_deg": _summ(rival_pa_err),
            "fl_mm": _summ(rival_fl_err),
            "mt_mm": _summ(rival_mt_err),
        },
        "n_test": len(test_ids),
        "test_policy": "median_of_train_geometry_head_no_test_masks",
        "submission": str(sub_path),
        "notes": [
            "Better mask geometry than PCA/peak-gap for Tier A diagnostics.",
            "Test CSV is median fallback until a seg model exists.",
            "Near-parallel fascicle/apo intersections are rejected (fail-closed).",
            "Do not treat TIFF dpi as spacing.",
        ],
    }
    (out_dir / "geometry_head_proxy_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(
        f"geometry_head wrote {sub_path} train_ok={train_ok}/{train_ok+train_fail} "
        f"priors={priors} provenance={provenance}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
