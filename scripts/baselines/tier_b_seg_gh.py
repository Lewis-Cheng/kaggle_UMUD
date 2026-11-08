#!/usr/bin/env python3
"""Tier B: predicted (or GT) masks → GeometryHead + AmbrosM mm/px.

Replaces ExtraTrees test path with direct seg→GH when pred masks exist.
Val MAE reported vs GT-mask GeometryHead+AmbrosM targets (NOT expert / NOT LB).
Official S stays null unless --expert-gt-csv is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.ambrosm_scale_io import (  # noqa: E402
    AmbrosmScaleRow,
    load_ambrosm_scale_rows,
    scale_map_by_stem,
)
from umud.baselines.io import (  # noqa: E402
    load_split_assignments,
    load_test_ids,
    require_scale_provenance,
    write_submission_csv,
)
from umud.baselines.mask_geometry import measure_masks  # noqa: E402
from umud.baselines.mask_proxy import FALLBACK_PRIORS, load_mask  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.metric.score import score_prediction_frames  # noqa: E402
import pandas as pd  # noqa: E402


def _mask_path(folder: Path, stem: str) -> Path | None:
    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def _measure(
    *,
    stem: str,
    fasc_dir: Path | None,
    apo_dir: Path | None,
    scale: AmbrosmScaleRow | None,
    priors: dict[str, float],
) -> dict[str, Any]:
    fasc = None
    apo = None
    if fasc_dir is not None:
        mp = _mask_path(fasc_dir, stem)
        if mp is not None:
            fasc = load_mask(mp)
    if apo_dir is not None:
        mp = _mask_path(apo_dir, stem)
        if mp is not None:
            apo = load_mask(mp)
    usable = scale is not None and scale.usable
    sx = sy = float(scale.mm_per_px) if usable else 1.0  # type: ignore[arg-type]
    m = measure_masks(fasc_mask=fasc, apo_mask=apo, sx=sx, sy=sy, priors=priors)
    m["stem"] = stem
    m["usable_for_mm_targets"] = bool(usable and m.get("ok"))
    m["mm_per_px"] = float(scale.mm_per_px) if usable else None
    m["scale_provenance"] = scale.provenance if scale is not None else "unknown"
    m["has_fasc"] = fasc is not None and bool(fasc.any())
    m["has_apo"] = apo is not None and bool(apo.any())
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--ambrosm-scale-csv", type=Path, required=True)
    ap.add_argument("--pred-mask-dir", type=Path, required=True,
                    help="dir with pred_apo_masks/ and pred_fasc_masks/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--artifact-prefix", type=str, default="tier_b_seg_gh")
    ap.add_argument("--scale-provenance", type=str, default="ambrosm_per_family_mixed_v2_hflip")
    ap.add_argument("--expert-gt-csv", type=Path, default=None)
    ap.add_argument("--delimiter", type=str, default=";")
    args = ap.parse_args()

    run_prov = require_scale_provenance(args.scale_provenance)
    layout = discover_layout(args.data_root)
    split_rows = load_split_assignments(args.split_assignments)
    scales = scale_map_by_stem(load_ambrosm_scale_rows(args.ambrosm_scale_csv))

    pred_apo = args.pred_mask_dir / "pred_apo_masks"
    pred_fasc = args.pred_mask_dir / "pred_fasc_masks"
    if not pred_apo.is_dir() or not pred_fasc.is_dir():
        print(f"ERROR: missing pred mask dirs under {args.pred_mask_dir}", file=sys.stderr)
        return 1

    val_stems = sorted(
        {
            str(r["image_id"])
            for r in split_rows
            if r.get("split") == "val"
        }
    )
    train_stems = sorted(
        {
            str(r["image_id"])
            for r in split_rows
            if r.get("split") == "train"
        }
    )

    # GT-mask GeometryHead targets on val (oracle path)
    gt_val = [
        _measure(
            stem=s,
            fasc_dir=layout.fasc_masks,
            apo_dir=layout.apo_masks,
            scale=scales.get(s),
            priors=FALLBACK_PRIORS,
        )
        for s in val_stems
    ]
    mm_gt = [m for m in gt_val if m.get("usable_for_mm_targets")]
    priors = {
        "pa_deg": float(np.median([m["pa_deg"] for m in mm_gt])) if mm_gt else float(FALLBACK_PRIORS["pa_deg"]),
        "fl_mm": float(np.median([m["fl_mm"] for m in mm_gt])) if mm_gt else float(FALLBACK_PRIORS["fl_mm"]),
        "mt_mm": float(np.median([m["mt_mm"] for m in mm_gt])) if mm_gt else float(FALLBACK_PRIORS["mt_mm"]),
    }

    # Recompute GT with refined priors
    gt_val = [
        _measure(
            stem=s,
            fasc_dir=layout.fasc_masks,
            apo_dir=layout.apo_masks,
            scale=scales.get(s),
            priors=priors,
        )
        for s in val_stems
    ]
    gt_by = {m["stem"]: m for m in gt_val}

    # Pred-mask GeometryHead on val
    pred_val = [
        _measure(
            stem=s,
            fasc_dir=pred_fasc,
            apo_dir=pred_apo,
            scale=scales.get(s),
            priors=priors,
        )
        for s in val_stems
    ]

    # Pair MAE where both usable
    paired_pa, paired_fl, paired_mt = [], [], []
    n_pred_mm = 0
    n_gt_mm = 0
    for pm in pred_val:
        gm = gt_by.get(pm["stem"])
        if gm and gm.get("usable_for_mm_targets"):
            n_gt_mm += 1
        if pm.get("usable_for_mm_targets"):
            n_pred_mm += 1
        if (
            pm.get("usable_for_mm_targets")
            and gm
            and gm.get("usable_for_mm_targets")
        ):
            paired_pa.append(abs(float(pm["pa_deg"]) - float(gm["pa_deg"])))
            paired_fl.append(abs(float(pm["fl_mm"]) - float(gm["fl_mm"])))
            paired_mt.append(abs(float(pm["mt_mm"]) - float(gm["mt_mm"])))

    val_metrics = {
        "label": "MAE_predMask_GH_vs_GTMask_GH_AmbrosM_NOT_expert_NOT_LB",
        "n_val_stems": len(val_stems),
        "n_gt_mm": n_gt_mm,
        "n_pred_mm": n_pred_mm,
        "n_paired_mm": len(paired_pa),
        "pa_mae": float(np.mean(paired_pa)) if paired_pa else None,
        "fl_mae": float(np.mean(paired_fl)) if paired_fl else None,
        "mt_mae": float(np.mean(paired_mt)) if paired_mt else None,
        "priors_from_gt_mm": priors,
    }
    if paired_pa:
        tau = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
        s1 = float(
            np.mean(
                [
                    max(paired_pa[i] / tau["pa"], 0)
                    + max(paired_fl[i] / tau["fl"], 0)
                    + max(paired_mt[i] / tau["mt"], 0)
                    for i in range(len(paired_pa))
                ]
            )
            / 3.0
        )
        val_metrics["diagnostic_S1_vs_gt_mask_gh"] = s1

    # Test: direct pred → GH + AmbrosM
    test_stems = load_test_ids(args.data_root)
    from umud.baselines.io import read_sample_ids
    sample_ids = (
        read_sample_ids(layout.sample_submission)
        if layout.sample_submission is not None
        else []
    )
    if sample_ids and len(sample_ids) >= max(len(test_stems), 1):
        test_ids = sample_ids
    elif sample_ids:
        suffix = Path(sample_ids[0]).suffix or ".tif"
        test_ids = [f"{s}{suffix}" for s in test_stems]
    else:
        test_ids = [f"{s}.tif" for s in test_stems]
    id_for_stem = {Path(i).stem: i for i in test_ids}
    test_stems = [Path(i).stem for i in test_ids]

    pred_test = [
        _measure(
            stem=s,
            fasc_dir=pred_fasc,
            apo_dir=pred_apo,
            scale=scales.get(s),
            priors=priors,
        )
        for s in test_stems
    ]

    # Fallback to priors when not ok / no scale
    rows_out = []
    n_fail = 0
    for m in pred_test:
        stem = str(m["stem"])
        ok = bool(m.get("usable_for_mm_targets"))
        if not ok:
            n_fail += 1
        rows_out.append(
            {
                "image_id": id_for_stem.get(stem, f"{stem}.tif"),
                "pa_deg": float(m["pa_deg"]),
                "fl_mm": float(m["fl_mm"]),
                "mt_mm": float(m["mt_mm"]),
                "ok": ok,
                "failure_reason": m.get("failure_reason"),
                "scale_provenance": m.get("scale_provenance"),
                "has_fasc": m.get("has_fasc"),
                "has_apo": m.get("has_apo"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.artifact_prefix
    sub_path = args.out_dir / f"submission_{prefix}.csv"
    write_submission_csv(
        sub_path,
        [
            {
                "image_id": r["image_id"],
                "pa_deg": r["pa_deg"],
                "fl_mm": r["fl_mm"],
                "mt_mm": r["mt_mm"],
            }
            for r in rows_out
        ],
        delimiter=args.delimiter,
    )

    official_S = None
    official_S_reason = "no_independent_expert_gt_csv_in_data_root"
    expert_score = None
    if args.expert_gt_csv is not None and args.expert_gt_csv.is_file():
        pred_df = pd.read_csv(sub_path, sep=args.delimiter)
        gt_df = pd.read_csv(args.expert_gt_csv, sep=None, engine="python")
        id_col = "image_id" if "image_id" in pred_df.columns else "id"
        if id_col not in gt_df.columns and "id" in gt_df.columns:
            gt_df = gt_df.rename(columns={"id": id_col})
        if id_col not in gt_df.columns and "image_id" in gt_df.columns:
            pass
        expert_score = score_prediction_frames(
            pred_df.rename(columns={id_col: "id"}) if id_col != "id" else pred_df,
            gt_df.rename(columns={id_col: "id"}) if id_col != "id" and id_col in gt_df.columns else (
                gt_df.rename(columns={"image_id": "id"}) if "image_id" in gt_df.columns else gt_df
            ),
            id_col="id",
            on_extra="ignore",
        )
        official_S = expert_score["components"]["S"]
        official_S_reason = "computed_vs_expert_gt_csv"

    # Compare to ExtraTrees Tier B v2 metrics if present
    et_path = args.out_dir / "tier_b_geometry_v2_metrics.json"
    et_compare = None
    if et_path.is_file():
        et = json.loads(et_path.read_text(encoding="utf-8"))
        et_compare = {
            "extratrees_val_mae": et.get("val_metrics")
            or et.get("val_mae_vs_geometry_head_ambrosm_NOT_expert_GT_NOT_LB")
            or {
                "pa_mae": et.get("val_metrics", {}).get("pa_mae")
                if isinstance(et.get("val_metrics"), dict)
                else None
            },
            "note": "ET supervised by GH targets; seg-GH compared to GT-mask GH — different reference, both NOT expert",
        }

    report = {
        "audit": "tier_b_seg_gh",
        "scale_provenance_run": run_prov,
        "pred_mask_dir": str(args.pred_mask_dir),
        "val_metrics": val_metrics,
        "test": {
            "n": len(rows_out),
            "n_fail_or_unusable": n_fail,
            "n_ok_mm": len(rows_out) - n_fail,
            "policy": "direct_pred_masks_GeometryHead_AmbrosM_NOT_ExtraTrees",
            "submission": str(sub_path),
        },
        "official_S": official_S,
        "official_S_reason": official_S_reason,
        "expert_score": expert_score,
        "extratrees_v2_compare": et_compare,
        "n_train_stems_split": len(train_stems),
        "attention_still_blocked": True,
        "note": "NOT LB. Val MAE is pred-mask GH vs GT-mask GH under AmbrosM scale.",
    }
    metrics_path = args.out_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # rows csv
    rows_path = args.out_dir / f"{prefix}_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "stem",
                "source",
                "ok",
                "usable_for_mm_targets",
                "pa_deg",
                "fl_mm",
                "mt_mm",
                "scale_provenance",
                "has_fasc",
                "has_apo",
                "failure_reason",
            ],
        )
        w.writeheader()
        for m in pred_val:
            w.writerow(
                {
                    "split": "val",
                    "stem": m["stem"],
                    "source": "pred",
                    "ok": m.get("ok"),
                    "usable_for_mm_targets": m.get("usable_for_mm_targets"),
                    "pa_deg": m.get("pa_deg"),
                    "fl_mm": m.get("fl_mm"),
                    "mt_mm": m.get("mt_mm"),
                    "scale_provenance": m.get("scale_provenance"),
                    "has_fasc": m.get("has_fasc"),
                    "has_apo": m.get("has_apo"),
                    "failure_reason": m.get("failure_reason"),
                }
            )
        for m in pred_test:
            w.writerow(
                {
                    "split": "test",
                    "stem": m["stem"],
                    "source": "pred",
                    "ok": m.get("ok"),
                    "usable_for_mm_targets": m.get("usable_for_mm_targets"),
                    "pa_deg": m.get("pa_deg"),
                    "fl_mm": m.get("fl_mm"),
                    "mt_mm": m.get("mt_mm"),
                    "scale_provenance": m.get("scale_provenance"),
                    "has_fasc": m.get("has_fasc"),
                    "has_apo": m.get("has_apo"),
                    "failure_reason": m.get("failure_reason"),
                }
            )

    notes = args.out_dir / f"{prefix}_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# Tier B seg → GeometryHead + AmbrosM",
                "",
                f"- scale protocol: `{run_prov}`",
                f"- pred masks: `{args.pred_mask_dir}`",
                f"- val paired MAE (pred GH vs GT-mask GH): PA={val_metrics.get('pa_mae')} FL={val_metrics.get('fl_mm') and val_metrics.get('fl_mae')} MT={val_metrics.get('mt_mae')}",
                f"- test ok mm: {len(rows_out)-n_fail}/{len(rows_out)}",
                f"- submission: `{sub_path}` (direct seg→GH, not ExtraTrees)",
                f"- official_S: `{official_S}` ({official_S_reason})",
                "- attention: still blocked pending gate verdict",
                "- NOT LB",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"{prefix} val_MAE_pa={val_metrics.get('pa_mae')} fl={val_metrics.get('fl_mae')} "
        f"mt={val_metrics.get('mt_mae')} test_ok={len(rows_out)-n_fail}/{len(rows_out)} "
        f"submission={sub_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
