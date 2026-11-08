#!/usr/bin/env python3
"""Tier B GeometryHead with AmbrosM per-image mm/px (not global stub).

Uses frozen duplicate-aware split. For stems with AmbrosM provenance in
{measured, family_prior}, runs mask → GeometryHead with sx=sy=mm_per_px.
Unknown scale stays pixel-space / skipped for mm targets (clear flags).

Supervises ExtraTrees on GeometryHead(AmbrosM) targets so test has a real
geometry+scale path (not train-median stub). Val MAE is vs those mask+scale
GeometryHead targets — NOT independent expert labels / NOT leaderboard.

Attention remains blocked. Does not invent LB scores or official S without GT.
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
from umud.baselines.features import image_feature_vector  # noqa: E402
from umud.baselines.io import (  # noqa: E402
    load_split_assignments,
    load_test_ids,
    require_scale_provenance,
    write_submission_csv,
)
from umud.baselines.mask_geometry import measure_masks  # noqa: E402
from umud.baselines.mask_proxy import FALLBACK_PRIORS, load_gray, load_mask  # noqa: E402
from umud.data.layout import discover_layout  # noqa: E402


def _mask_path(folder: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    if folder.is_dir():
        for p in folder.iterdir():
            if p.is_file() and p.stem == stem:
                return p
    return None


def _img_path(folder: Path, stem: str) -> Path | None:
    return _mask_path(folder, stem)


def _index_by_stem(rows: list[dict[str, Any]], task: str, split: str) -> dict[str, dict]:
    return {
        str(r["image_id"]): r
        for r in rows
        if r.get("task") == task and r.get("split") == split
    }


def _clip_to_quantiles(
    arr: np.ndarray, ref: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0
) -> np.ndarray:
    lo = np.percentile(ref, q_lo, axis=0)
    hi = np.percentile(ref, q_hi, axis=0)
    return np.clip(arr, lo, hi)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def _summ(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p25": None, "p75": None}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
    }


def _measure_stem(
    *,
    stem: str,
    fasc_masks: Path,
    apo_masks: Path,
    fasc_stems: set[str],
    apo_stems: set[str],
    scale: AmbrosmScaleRow | None,
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

    usable = scale is not None and scale.usable
    if usable:
        sx = sy = float(scale.mm_per_px)  # type: ignore[arg-type]
        scale_mode = "ambrosm_mm"
        provenance = scale.provenance
        family = scale.family
        q_prov = scale.q_provenance
    else:
        # Pixel-space: sx=sy=1 → FL/MT in px; PA still degrees (isotropic).
        sx = sy = 1.0
        scale_mode = "pixel_space_unknown_scale"
        provenance = scale.provenance if scale is not None else "unknown"
        family = scale.family if scale is not None else ""
        q_prov = scale.q_provenance if scale is not None else "unknown"

    m = measure_masks(
        fasc_mask=fasc, apo_mask=apo, sx=sx, sy=sy, priors=priors
    )
    m["stem"] = stem
    m["scale_mode"] = scale_mode
    m["scale_provenance"] = provenance
    m["scale_family"] = family
    m["q_provenance"] = q_prov
    m["mm_per_px"] = float(scale.mm_per_px) if usable else None
    m["has_fasc_mask"] = fasc is not None
    m["has_apo_mask"] = apo is not None
    m["usable_for_mm_targets"] = bool(usable and m.get("ok"))
    return m


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--ambrosm-scale-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default="tier_b_geometry",
        help="basename prefix for metrics/rows/notes/submission (e.g. tier_b_geometry_v2)",
    )
    parser.add_argument(
        "--scale-provenance",
        type=str,
        default="ambrosm_per_family_mixed",
        help="run-level tag (per-image tags come from AmbrosM CSV)",
    )
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max-features", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--delimiter",
        type=str,
        default=";",
        help="sample_submission uses semicolon",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="smoke cap (0=all)")
    parser.add_argument(
        "--write-submission",
        action="store_true",
        default=True,
        help="write ExtraTrees test CSV when geometry+scale path trained",
    )
    parser.add_argument(
        "--no-write-submission",
        action="store_true",
        help="skip test CSV even if model trains",
    )
    args = parser.parse_args()
    write_sub = bool(args.write_submission) and not bool(args.no_write_submission)

    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as exc:
        print(f"ERROR: scikit-learn required ({exc})", file=sys.stderr)
        return 1

    run_prov = require_scale_provenance(args.scale_provenance)
    layout = discover_layout(args.data_root)
    split_rows = load_split_assignments(args.split_assignments)
    scale_rows = load_ambrosm_scale_rows(args.ambrosm_scale_csv)
    scales = scale_map_by_stem(scale_rows)

    fasc_train = _index_by_stem(split_rows, "fascicle", "train")
    fasc_val = _index_by_stem(split_rows, "fascicle", "val")
    apo_train = _index_by_stem(split_rows, "aponeurosis", "train")
    apo_val = _index_by_stem(split_rows, "aponeurosis", "val")
    fasc_stems = set(fasc_train) | set(fasc_val)
    apo_stems = set(apo_train) | set(apo_val)

    train_stems = sorted(set(fasc_train) | set(apo_train))
    val_stems = sorted(set(fasc_val) | set(apo_val))
    if args.max_rows and args.max_rows > 0:
        train_stems = train_stems[: args.max_rows]
        val_stems = val_stems[: args.max_rows]

    # Pass 1: GeometryHead on train with AmbrosM scale → refine priors
    train_meas: list[dict[str, Any]] = []
    for stem in train_stems:
        train_meas.append(
            _measure_stem(
                stem=stem,
                fasc_masks=layout.fasc_masks,
                apo_masks=layout.apo_masks,
                fasc_stems=fasc_stems,
                apo_stems=apo_stems,
                scale=scales.get(stem),
                priors=FALLBACK_PRIORS,
            )
        )

    mm_ok_train = [m for m in train_meas if m.get("usable_for_mm_targets")]
    priors = {
        "pa_deg": float(np.median([m["pa_deg"] for m in mm_ok_train]))
        if mm_ok_train
        else float(FALLBACK_PRIORS["pa_deg"]),
        "fl_mm": float(np.median([m["fl_mm"] for m in mm_ok_train]))
        if mm_ok_train
        else float(FALLBACK_PRIORS["fl_mm"]),
        "mt_mm": float(np.median([m["mt_mm"] for m in mm_ok_train]))
        if mm_ok_train
        else float(FALLBACK_PRIORS["mt_mm"]),
    }

    # Re-measure train with refined priors; measure val
    train_meas = [
        _measure_stem(
            stem=stem,
            fasc_masks=layout.fasc_masks,
            apo_masks=layout.apo_masks,
            fasc_stems=fasc_stems,
            apo_stems=apo_stems,
            scale=scales.get(stem),
            priors=priors,
        )
        for stem in train_stems
    ]
    val_meas = [
        _measure_stem(
            stem=stem,
            fasc_masks=layout.fasc_masks,
            apo_masks=layout.apo_masks,
            fasc_stems=fasc_stems,
            apo_stems=apo_stems,
            scale=scales.get(stem),
            priors=priors,
        )
        for stem in val_stems
    ]

    def _cov(meas: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(meas)
        by_prov: dict[str, int] = {}
        by_mode: dict[str, int] = {}
        n_ok = 0
        n_mm = 0
        for m in meas:
            by_prov[m["scale_provenance"]] = by_prov.get(m["scale_provenance"], 0) + 1
            by_mode[m["scale_mode"]] = by_mode.get(m["scale_mode"], 0) + 1
            if m.get("ok"):
                n_ok += 1
            if m.get("usable_for_mm_targets"):
                n_mm += 1
        mm = [m for m in meas if m.get("usable_for_mm_targets")]
        return {
            "n_stems": n,
            "n_geometry_ok": n_ok,
            "n_mm_targets": n_mm,
            "provenance_counts": by_prov,
            "scale_mode_counts": by_mode,
            "geometry_head_mm_targets": {
                "pa_deg": _summ([float(m["pa_deg"]) for m in mm]),
                "fl_mm": _summ([float(m["fl_mm"]) for m in mm]),
                "mt_mm": _summ([float(m["mt_mm"]) for m in mm]),
            },
        }

    train_cov = _cov(train_meas)
    val_cov = _cov(val_meas)

    # ExtraTrees supervised by GeometryHead + AmbrosM mm targets
    def _feat_for(stem: str) -> np.ndarray | None:
        # Prefer apo image (overlap), else fasc, for feature extraction.
        for folder in (layout.apo_imgs, layout.fasc_imgs):
            ip = _img_path(folder, stem)
            if ip is not None:
                return image_feature_vector(load_gray(ip))
        return None

    X_tr: list[np.ndarray] = []
    Y_tr: list[list[float]] = []
    id_tr: list[str] = []
    for m in train_meas:
        if not m.get("usable_for_mm_targets"):
            continue
        feat = _feat_for(str(m["stem"]))
        if feat is None:
            continue
        X_tr.append(feat)
        Y_tr.append([float(m["pa_deg"]), float(m["fl_mm"]), float(m["mt_mm"])])
        id_tr.append(str(m["stem"]))

    X_va: list[np.ndarray] = []
    Y_va: list[list[float]] = []
    id_va: list[str] = []
    for m in val_meas:
        if not m.get("usable_for_mm_targets"):
            continue
        feat = _feat_for(str(m["stem"]))
        if feat is None:
            continue
        X_va.append(feat)
        Y_va.append([float(m["pa_deg"]), float(m["fl_mm"]), float(m["mt_mm"])])
        id_va.append(str(m["stem"]))

    model_ok = len(X_tr) >= 10
    val_metrics: dict[str, Any] = {
        "label": "MAE_vs_GeometryHead_AmbrosM_targets_NOT_expert_GT_NOT_LB",
        "n_train_mm": len(X_tr),
        "n_val_mm": len(X_va),
    }
    model = None
    Y_tr_arr = np.zeros((0, 3), dtype=np.float64)
    if model_ok:
        X_tr_arr = np.stack(X_tr)
        Y_tr_arr = np.asarray(Y_tr, dtype=np.float64)
        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X_tr_arr, Y_tr_arr)
        if X_va:
            X_va_arr = np.stack(X_va)
            Y_va_arr = np.asarray(Y_va, dtype=np.float64)
            pred = _clip_to_quantiles(model.predict(X_va_arr), Y_tr_arr)
            val_metrics["pa_mae"] = _mae(pred[:, 0], Y_va_arr[:, 0])
            val_metrics["fl_mae"] = _mae(pred[:, 1], Y_va_arr[:, 1])
            val_metrics["mt_mae"] = _mae(pred[:, 2], Y_va_arr[:, 2])
            # Soft S-like using pinned τ — NOT official LB (targets are GH, not expert)
            tau = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
            e_pa = np.abs(pred[:, 0] - Y_va_arr[:, 0])
            e_fl = np.abs(pred[:, 1] - Y_va_arr[:, 1])
            e_mt = np.abs(pred[:, 2] - Y_va_arr[:, 2])
            s1 = float(
                np.mean(
                    np.maximum(e_pa / tau["pa_deg"], 0)
                    + np.maximum(e_fl / tau["fl_mm"], 0)
                    + np.maximum(e_mt / tau["mt_mm"], 0)
                )
                / 3.0
            )
            # Match pinned composition shape loosely for diagnostics only
            val_metrics["diagnostic_S1_vs_gh_targets"] = s1
            val_metrics["note_S"] = (
                "diagnostic only vs GeometryHead AmbrosM targets; "
                "NOT official S / NOT leaderboard (no independent expert GT locally)"
            )
        else:
            val_metrics["pa_mae"] = None
            val_metrics["fl_mae"] = None
            val_metrics["mt_mae"] = None
            val_metrics["note"] = "no_val_mm_rows"
    else:
        val_metrics["error"] = f"too_few_train_mm_examples n={len(X_tr)}"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-row dumps for audit
    prefix = args.artifact_prefix.strip() or "tier_b_geometry"
    rows_path = out_dir / f"{prefix}_rows.csv"
    fieldnames = [
        "split",
        "stem",
        "ok",
        "usable_for_mm_targets",
        "scale_mode",
        "scale_provenance",
        "scale_family",
        "q_provenance",
        "mm_per_px",
        "pa_deg",
        "fl_mm",
        "mt_mm",
        "has_fasc_mask",
        "has_apo_mask",
        "failure_reason",
        "mt_method",
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for split_name, meas in (("train", train_meas), ("val", val_meas)):
            for m in meas:
                w.writerow(
                    {
                        "split": split_name,
                        "stem": m.get("stem"),
                        "ok": m.get("ok"),
                        "usable_for_mm_targets": m.get("usable_for_mm_targets"),
                        "scale_mode": m.get("scale_mode"),
                        "scale_provenance": m.get("scale_provenance"),
                        "scale_family": m.get("scale_family"),
                        "q_provenance": m.get("q_provenance"),
                        "mm_per_px": m.get("mm_per_px"),
                        "pa_deg": m.get("pa_deg"),
                        "fl_mm": m.get("fl_mm"),
                        "mt_mm": m.get("mt_mm"),
                        "has_fasc_mask": m.get("has_fasc_mask"),
                        "has_apo_mask": m.get("has_apo_mask"),
                        "failure_reason": m.get("failure_reason"),
                        "mt_method": m.get("mt_method"),
                    }
                )

    # Test submission only via ExtraTrees (geometry+scale supervised) — never median stub
    sub_path = None
    test_policy = "skipped_no_median_stub"
    n_test_fail = 0
    test_stems = load_test_ids(args.data_root)
    # sample_submission may be a short format template (official zip has 2 rows).
    # Prefer full test stem list; keep sample suffix convention (usually .tif).
    sample_ids: list[str] | None = None
    if layout.sample_submission is not None:
        from umud.baselines.io import read_sample_ids

        sample_ids = read_sample_ids(layout.sample_submission)
    if sample_ids and len(sample_ids) >= max(len(test_stems), 1):
        test_ids = sample_ids
    elif sample_ids:
        suffix = Path(sample_ids[0]).suffix or ".tif"
        test_ids = [f"{s}{suffix}" for s in test_stems]
    elif test_stems:
        test_ids = list(test_stems)
    else:
        test_ids = []
    stem_of = {Path(i).stem: i for i in test_ids}

    test_scale_cov = {"measured": 0, "family_prior": 0, "unknown": 0, "missing": 0}
    for tid in test_ids:
        stem = Path(tid).stem
        sc = scales.get(stem)
        if sc is None:
            test_scale_cov["missing"] += 1
        else:
            test_scale_cov[sc.provenance] = test_scale_cov.get(sc.provenance, 0) + 1

    if write_sub and model is not None and model_ok:
        test_rows = []
        med = {
            "pa_deg": float(np.median(Y_tr_arr[:, 0])),
            "fl_mm": float(np.median(Y_tr_arr[:, 1])),
            "mt_mm": float(np.median(Y_tr_arr[:, 2])),
        }
        for tid in test_ids:
            stem = Path(tid).stem
            ip = _img_path(layout.test_imgs, stem)
            if ip is None:
                n_test_fail += 1
                test_rows.append({"image_id": tid, **med})
                continue
            try:
                feat = image_feature_vector(load_gray(ip)).reshape(1, -1)
                pred = _clip_to_quantiles(model.predict(feat), Y_tr_arr)[0]
                test_rows.append(
                    {
                        "image_id": tid,
                        "pa_deg": float(pred[0]),
                        "fl_mm": float(pred[1]),
                        "mt_mm": float(pred[2]),
                    }
                )
            except Exception:
                n_test_fail += 1
                test_rows.append({"image_id": tid, **med})
        sub_path = out_dir / f"submission_{prefix}.csv"
        # Validate against stems for local helper; CSV keeps sample id strings.
        write_submission_csv(
            sub_path,
            test_rows,
            delimiter=args.delimiter,
            validate_against=None,
        )
        test_policy = (
            "extratrees_supervised_by_geometry_head_ambrosm_mm_"
            "NOT_direct_test_mask_geometry"
        )
    elif write_sub and not model_ok:
        test_policy = "skipped_insufficient_train_mm_no_median_stub"

    report = {
        "baseline": prefix,
        "tier": "B_proxy_scale_aware_geometry_head",
        "label": (
            "GeometryHead+AmbrosM_per_image_mm_ExtraTrees_"
            "NOT_expert_GT_NOT_leaderboard"
        ),
        "run_scale_provenance": run_prov,
        "ambrosm_scale_csv": str(args.ambrosm_scale_csv),
        "split_assignments": str(args.split_assignments),
        "organizer_mt_policy": "vertical_not_perp_to_apo",
        "forbid_tiff_dpi_25_4": True,
        "attention_still_blocked": True,
        "geometry_head_priors_from_train_mm": priors,
        "train_coverage": train_cov,
        "val_coverage": val_cov,
        "val_metrics": val_metrics,
        "official_S": None,
        "official_S_reason": "no_independent_expert_gt_csv_in_data_root",
        "model": {
            "type": "ExtraTreesRegressor" if model_ok else None,
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "seed": args.seed,
            "n_train": len(X_tr),
            "n_val": len(X_va),
        },
        "n_test": len(test_ids),
        "n_sample_submission_ids": len(sample_ids) if sample_ids else 0,
        "test_id_source": (
            "sample_submission"
            if sample_ids and len(sample_ids) >= max(len(test_stems), 1)
            else (
                "test_stems_with_sample_suffix"
                if sample_ids
                else "test_stems"
            )
        ),
        "test_scale_coverage": test_scale_cov,
        "test_policy": test_policy,
        "n_test_failures_to_median": n_test_fail,
        "submission": str(sub_path) if sub_path else None,
        "rows_csv": str(rows_path),
        "notes": [
            "Per-image mm_per_px from AmbrosM when provenance in {measured, family_prior}.",
            "q (mm-per-tick) remains AmbrosM UI prior — tagged via q_provenance.",
            "Unknown scale → pixel_space; excluded from mm ExtraTrees targets.",
            "Val MAE is ExtraTrees vs GeometryHead(mask, AmbrosM) — NOT expert PA/FL/MT.",
            "No official S without independent GT. Attention still blocked.",
            "Test CSV is ExtraTrees (geometry+scale supervised), not train-median stub.",
            "If sample_submission is a short template, test IDs expand from test image stems with sample suffix.",
            "Do not treat TIFF dpi as ultrasound spacing.",
        ],
    }
    report_path = out_dir / f"{prefix}_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )

    notes_path = out_dir / f"{prefix}_notes.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Tier B GeometryHead (AmbrosM per-image mm/px)",
                "",
                f"- Run provenance: `{run_prov}`",
                f"- Train mm targets: **{val_metrics.get('n_train_mm')}** "
                f"/ GeometryHead ok+scale",
                f"- Val mm targets: **{val_metrics.get('n_val_mm')}**",
                f"- Val MAE (vs GH AmbrosM, NOT expert/LB): "
                f"PA={val_metrics.get('pa_mae')} "
                f"FL={val_metrics.get('fl_mae')} "
                f"MT={val_metrics.get('mt_mae')}",
                f"- Official S: **null** ({report['official_S_reason']})",
                f"- Test policy: `{test_policy}`",
                f"- Submission: `{sub_path}`",
                f"- Attention: still blocked",
                "",
                "## Blockers",
                "",
                "- No independent expert PA/FL/MT CSV in data root → cannot claim Tier B LB S.",
                "- Residual AmbrosM `unknown` shapes (1080x1640, 500x760, etc.) "
                "still excluded from mm targets; q remains UI prior.",
                "- Test has no masks → ExtraTrees image→mm, not direct GeometryHead on test.",
                "",
                "## Next lever",
                "",
                "- Seg model for test masks → direct GeometryHead+AmbrosM on test "
                "(replace ExtraTrees).",
                "- Optionally extend AmbrosM for residual unknown shapes (no invent q).",
                "- Keep attention blocked until true Tier B held-out evidence exists.",
                "",
            ]
        )
        + "\n"
    )

    print(
        f"{prefix} "
        f"train_mm={len(X_tr)} val_mm={len(X_va)} "
        f"val_MAE_pa={val_metrics.get('pa_mae')} "
        f"fl={val_metrics.get('fl_mae')} mt={val_metrics.get('mt_mae')} "
        f"submission={sub_path} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
