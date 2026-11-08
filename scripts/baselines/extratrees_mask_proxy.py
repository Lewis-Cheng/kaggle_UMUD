#!/usr/bin/env python3
"""Phase 3 floor B — rival ExtraTrees image→mask-proxy port.

Trains ExtraTreesRegressor(n_estimators=160, ...) on frozen duplicate-aware
train rows only. Features: 32×24 gray + bins + percentiles + orientation hist.
Heads: PA+FL from fascicle masks; MT from apo masks. Predictions clipped to
train 1–99% quantiles. Failures / missing → train medians.

Local metrics on frozen val are Tier A proxy (NOT Tier B / leaderboard).
Requires ``--scale-provenance`` (default scale 0.07225 is a rival guess only).
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

from umud.baselines.features import image_feature_vector  # noqa: E402
from umud.baselines.io import (  # noqa: E402
    load_split_assignments,
    load_test_ids,
    require_scale_provenance,
    write_submission_csv,
)
from umud.baselines.mask_proxy import (  # noqa: E402
    FALLBACK_PRIORS,
    apply_isotropic_mm,
    load_gray,
    load_mask,
    proxy_apo_peak_gap,
    proxy_fascicle_pca,
)
from umud.data.layout import discover_layout, list_images  # noqa: E402


def _mask_path(folder: Path, stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    for p in folder.iterdir():
        if p.is_file() and p.stem == stem:
            return p
    return None


def _img_path(folder: Path, stem: str) -> Path | None:
    return _mask_path(folder, stem)


def _build_fasc_examples(
    rows: list[dict[str, Any]],
    *,
    img_dir: Path,
    mask_dir: Path,
    mm_per_px: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[list[float]] = []
    ids: list[str] = []
    for row in rows:
        stem = str(row["image_id"])
        ip = _img_path(img_dir, stem)
        mp = _mask_path(mask_dir, stem)
        if ip is None or mp is None:
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
        if scaled["pa_deg"] is None or scaled["fl_mm"] is None:
            continue
        xs.append(image_feature_vector(load_gray(ip)))
        ys.append([float(scaled["pa_deg"]), float(scaled["fl_mm"])])
        ids.append(stem)
    if not xs:
        return np.zeros((0, 1)), np.zeros((0, 2)), []
    return np.stack(xs), np.asarray(ys, dtype=np.float64), ids


def _build_apo_examples(
    rows: list[dict[str, Any]],
    *,
    img_dir: Path,
    mask_dir: Path,
    mm_per_px: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ids: list[str] = []
    for row in rows:
        stem = str(row["image_id"])
        ip = _img_path(img_dir, stem)
        mp = _mask_path(mask_dir, stem)
        if ip is None or mp is None:
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
        if scaled["mt_mm"] is None:
            continue
        xs.append(image_feature_vector(load_gray(ip)))
        ys.append(float(scaled["mt_mm"]))
        ids.append(stem)
    if not xs:
        return np.zeros((0, 1)), np.zeros((0,)), []
    return np.stack(xs), np.asarray(ys, dtype=np.float64), ids


def _clip_to_quantiles(arr: np.ndarray, ref: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0) -> np.ndarray:
    lo = np.percentile(ref, q_lo, axis=0)
    hi = np.percentile(ref, q_hi, axis=0)
    return np.clip(arr, lo, hi)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mm-per-px", type=float, default=0.07225)
    parser.add_argument(
        "--scale-provenance",
        type=str,
        required=True,
        help="required, e.g. unknown (rival 0.07225 is a guess only)",
    )
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max-features", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="optional cap per task for smoke runs (0=all)",
    )
    args = parser.parse_args()

    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as exc:
        print("ERROR: scikit-learn required for ExtraTrees floor", file=sys.stderr)
        print(f"  pip install scikit-learn  ({exc})", file=sys.stderr)
        return 1

    provenance = require_scale_provenance(args.scale_provenance)
    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    fasc_train = [r for r in rows if r.get("task") == "fascicle" and r.get("split") == "train"]
    fasc_val = [r for r in rows if r.get("task") == "fascicle" and r.get("split") == "val"]
    apo_train = [r for r in rows if r.get("task") == "aponeurosis" and r.get("split") == "train"]
    apo_val = [r for r in rows if r.get("task") == "aponeurosis" and r.get("split") == "val"]
    if args.max_train and args.max_train > 0:
        fasc_train = fasc_train[: args.max_train]
        apo_train = apo_train[: args.max_train]

    Xf_tr, Yf_tr, _ = _build_fasc_examples(
        fasc_train,
        img_dir=layout.fasc_imgs,
        mask_dir=layout.fasc_masks,
        mm_per_px=args.mm_per_px,
    )
    Xa_tr, Ya_tr, _ = _build_apo_examples(
        apo_train,
        img_dir=layout.apo_imgs,
        mask_dir=layout.apo_masks,
        mm_per_px=args.mm_per_px,
    )
    if Xf_tr.shape[0] < 5 or Xa_tr.shape[0] < 5:
        print(
            f"ERROR: too few train examples fasc={Xf_tr.shape[0]} apo={Xa_tr.shape[0]}",
            file=sys.stderr,
        )
        return 1

    fasc_model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=-1,
    )
    apo_model = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=-1,
    )
    fasc_model.fit(Xf_tr, Yf_tr)
    apo_model.fit(Xa_tr, Ya_tr)

    # Train medians / quantiles for clip + fallback
    pa_med = float(np.median(Yf_tr[:, 0]))
    fl_med = float(np.median(Yf_tr[:, 1]))
    mt_med = float(np.median(Ya_tr))

    # Val proxy metrics (Tier A)
    val_metrics: dict[str, Any] = {"tier": "A_proxy_not_tier_b"}
    Xf_va, Yf_va, _ = _build_fasc_examples(
        fasc_val,
        img_dir=layout.fasc_imgs,
        mask_dir=layout.fasc_masks,
        mm_per_px=args.mm_per_px,
    )
    Xa_va, Ya_va, _ = _build_apo_examples(
        apo_val,
        img_dir=layout.apo_imgs,
        mask_dir=layout.apo_masks,
        mm_per_px=args.mm_per_px,
    )
    if Xf_va.shape[0]:
        pred = _clip_to_quantiles(fasc_model.predict(Xf_va), Yf_tr)
        val_metrics["fasc_val_n"] = int(Xf_va.shape[0])
        val_metrics["pa_mae_proxy"] = _mae(pred[:, 0], Yf_va[:, 0])
        val_metrics["fl_mae_proxy"] = _mae(pred[:, 1], Yf_va[:, 1])
    if Xa_va.shape[0]:
        pred_m = _clip_to_quantiles(apo_model.predict(Xa_va).reshape(-1), Ya_tr)
        val_metrics["apo_val_n"] = int(Xa_va.shape[0])
        val_metrics["mt_mae_proxy"] = _mae(pred_m, Ya_va)

    # Test predictions: image features from test set
    test_ids = load_test_ids(args.data_root)
    test_rows = []
    n_fail = 0
    for tid in test_ids:
        ip = _img_path(layout.test_imgs, tid)
        if ip is None:
            n_fail += 1
            test_rows.append(
                {
                    "image_id": tid,
                    "pa_deg": pa_med,
                    "fl_mm": fl_med,
                    "mt_mm": mt_med,
                }
            )
            continue
        try:
            feat = image_feature_vector(load_gray(ip)).reshape(1, -1)
            pf = _clip_to_quantiles(fasc_model.predict(feat), Yf_tr)[0]
            pm = float(_clip_to_quantiles(apo_model.predict(feat).reshape(-1), Ya_tr)[0])
            test_rows.append(
                {
                    "image_id": tid,
                    "pa_deg": float(pf[0]),
                    "fl_mm": float(pf[1]),
                    "mt_mm": pm,
                }
            )
        except Exception:
            n_fail += 1
            test_rows.append(
                {
                    "image_id": tid,
                    "pa_deg": pa_med,
                    "fl_mm": fl_med,
                    "mt_mm": mt_med,
                }
            )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "submission_extratrees_mask_proxy.csv"
    write_submission_csv(
        sub_path, test_rows, delimiter=args.delimiter, validate_against=test_ids
    )

    report = {
        "baseline": "extratrees_mask_proxy",
        "tier": "A_proxy_not_tier_b",
        "label": "NOT_Tier_B_NOT_leaderboard",
        "mm_per_px": args.mm_per_px,
        "scale_provenance": provenance,
        "model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "seed": args.seed,
        },
        "n_train_fasc": int(Xf_tr.shape[0]),
        "n_train_apo": int(Xa_tr.shape[0]),
        "train_medians": {"pa_deg": pa_med, "fl_mm": fl_med, "mt_mm": mt_med},
        "fallback_priors_unused_unless_empty": FALLBACK_PRIORS,
        "val_proxy_metrics": val_metrics,
        "n_test": len(test_ids),
        "n_test_failures_to_median": n_fail,
        "submission": str(sub_path),
        "notes": [
            "Supervised by mask-derived proxies, not expert PA/FL/MT.",
            "Frozen duplicate-aware split only — do not use rival (id//5)%5.",
            "Local MAE is Tier A proxy; never promote on it alone.",
        ],
    }
    (out_dir / "extratrees_mask_proxy_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(
        f"extratrees wrote {sub_path} train_fasc={Xf_tr.shape[0]} "
        f"train_apo={Xa_tr.shape[0]} val={val_metrics} provenance={provenance}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
