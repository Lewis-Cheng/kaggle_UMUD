"""Official UMUD score formulas (organizer notebook paulritsche/umud-score v9).

S = S1 + 1e-6*S2 + 1e-9*S3
Each of S1/S2/S3 averages the three targets after normalizing by tolerances
6° PA, 12 mm FL, 3 mm MT. S1 uses MAE, S2 median AE, S3 RMSE.

These equations describe the retrieved scorer, not an inferred approximation.

Verification gate: pin notebook-source SHA256 before tuning objectives or
claiming any competition score. Attention training is blocked until this
scorer is hash-verified against a local notebook file.
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

PINNED_NOTEBOOK_SHA256 = (
    "3873f613d55be3eee82229d661dd2c06c1d3c962ff63a1ec45b0a97b29cf061f"
)

TOLERANCES: Mapping[str, float] = {
    "pa_deg": 6.0,
    "fl_mm": 12.0,
    "mt_mm": 3.0,
}

TARGET_ORDER = ("pa_deg", "fl_mm", "mt_mm")
S2_COEFF = 1e-6
S3_COEFF = 1e-9

# Join policy documentation (see also submission.py):
# - Organizer notebook: inner join on image id; EXTRA unrelated rows can silently
#   disappear. Missing IDs / dups / non-finite preds are rejected.
# - Local exact-set validator: rejects extras (stronger than the notebook).
# - score_aligned / score_prediction_frames:
#     on_extra="ignore"  → drop pred rows whose id is absent from gt (notebook-like)
#     on_extra="reject"  → raise if pred has ids not in gt (local strict policy)
JoinExtraPolicy = Literal["ignore", "reject"]


def _as_array(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("each target error vector must be 1-D")
    return arr


def _normalized_mae(abs_err: np.ndarray, tol: float) -> float:
    return float(np.mean(abs_err) / tol)


def _normalized_median_ae(abs_err: np.ndarray, tol: float) -> float:
    return float(np.median(abs_err) / tol)


def _normalized_rmse(abs_err: np.ndarray, tol: float) -> float:
    return float(np.sqrt(np.mean(np.square(abs_err))) / tol)


def score_components(
    errors: Mapping[str, np.ndarray | list[float]],
) -> dict[str, float]:
    """Compute S1, S2, S3 and S from per-target absolute-error vectors.

    Parameters
    ----------
    errors:
        Mapping with keys pa_deg, fl_mm, mt_mm. Each value is a 1-D array of
        absolute errors for that target (same length across targets).
    """
    for key in TARGET_ORDER:
        if key not in errors:
            raise KeyError(f"missing target errors for {key}")

    abs_errs = {k: np.abs(_as_array(errors[k])) for k in TARGET_ORDER}
    n = {k: abs_errs[k].size for k in TARGET_ORDER}
    if len(set(n.values())) != 1:
        raise ValueError(f"target error lengths must match, got {n}")
    if next(iter(n.values())) == 0:
        raise ValueError("error vectors must be non-empty")

    s1 = float(
        np.mean([_normalized_mae(abs_errs[k], TOLERANCES[k]) for k in TARGET_ORDER])
    )
    s2 = float(
        np.mean(
            [_normalized_median_ae(abs_errs[k], TOLERANCES[k]) for k in TARGET_ORDER]
        )
    )
    s3 = float(
        np.mean([_normalized_rmse(abs_errs[k], TOLERANCES[k]) for k in TARGET_ORDER])
    )
    s = s1 + S2_COEFF * s2 + S3_COEFF * s3
    return {"S1": s1, "S2": s2, "S3": s3, "S": s}


def official_score(errors: Mapping[str, np.ndarray | list[float]]) -> float:
    """Return the composite official score S."""
    return score_components(errors)["S"]


def per_target_error_stats(
    errors: Mapping[str, np.ndarray | list[float]],
) -> dict[str, dict[str, float]]:
    """Return MAE / MedAE / RMSE (raw units) for each target.

    Does not invent competition scores; these are diagnostic summaries of the
    absolute-error vectors already supplied by the caller.
    """
    out: dict[str, dict[str, float]] = {}
    for key in TARGET_ORDER:
        if key not in errors:
            raise KeyError(f"missing target errors for {key}")
        ae = np.abs(_as_array(errors[key]))
        if ae.size == 0:
            raise ValueError("error vectors must be non-empty")
        out[key] = {
            "MAE": float(np.mean(ae)),
            "MedAE": float(np.median(ae)),
            "RMSE": float(np.sqrt(np.mean(np.square(ae)))),
            "tol": float(TOLERANCES[key]),
            "MAE_over_tol": float(np.mean(ae) / TOLERANCES[key]),
            "MedAE_over_tol": float(np.median(ae) / TOLERANCES[key]),
            "RMSE_over_tol": float(np.sqrt(np.mean(np.square(ae))) / TOLERANCES[key]),
        }
    return out


def absolute_errors_from_frames(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    on_extra: JoinExtraPolicy = "reject",
    id_col: str = "id",
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Align pred/gt on ``id`` and return absolute-error vectors + joined frame.

    Policy
    ------
    on_extra="reject"
        Raise if pred contains ids absent from gt (local exact-set / strict).
    on_extra="ignore"
        Drop those extra pred rows before scoring (closer to notebook inner join
        on the pred→gt direction). Gt rows without preds still cause a miss.

    Empty join after alignment raises ``ValueError`` (cannot score zero rows).
    Missing required columns raise ``KeyError``.
    """
    for frame_name, frame in (("pred", pred), ("gt", gt)):
        missing = [c for c in (id_col, *TARGET_ORDER) if c not in frame.columns]
        if missing:
            raise KeyError(f"{frame_name} missing columns: {missing}")

    pred_work = pred.copy()
    gt_work = gt.copy()
    pred_work[id_col] = pred_work[id_col].astype(str)
    gt_work[id_col] = gt_work[id_col].astype(str)

    pred_ids = set(pred_work[id_col])
    gt_ids = set(gt_work[id_col])
    extra = pred_ids - gt_ids
    if extra and on_extra == "reject":
        raise ValueError(
            f"pred has {len(extra)} id(s) absent from gt "
            f"(on_extra='reject'); sample={sorted(extra)[:5]}"
        )

    if gt_work[id_col].duplicated().any():
        raise ValueError("gt contains duplicate ids")
    if pred_work[id_col].duplicated().any():
        raise ValueError("pred contains duplicate ids")

    joined = pred_work.merge(
        gt_work,
        on=id_col,
        how="inner",
        suffixes=("_pred", "_gt"),
    )
    if joined.empty:
        raise ValueError("empty join: no overlapping ids between pred and gt")

    errors: dict[str, np.ndarray] = {}
    for key in TARGET_ORDER:
        p = pd.to_numeric(joined[f"{key}_pred"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        g = pd.to_numeric(joined[f"{key}_gt"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not (np.isfinite(p).all() and np.isfinite(g).all()):
            raise ValueError(f"non-finite values in aligned column {key}")
        errors[key] = np.abs(p - g)
    return errors, joined


def score_prediction_frames(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    on_extra: JoinExtraPolicy = "reject",
    id_col: str = "id",
) -> dict[str, object]:
    """Score pred vs gt DataFrames with pinned S/S1/S2/S3 + per-target stats.

    Returns a dict with keys: components (S/S1/S2/S3), per_target, n_scored,
    on_extra, ignored_extra_n.
    """
    pred_ids = set(pred[id_col].astype(str))
    gt_ids = set(gt[id_col].astype(str))
    ignored_extra_n = len(pred_ids - gt_ids) if on_extra == "ignore" else 0

    errors, joined = absolute_errors_from_frames(
        pred, gt, on_extra=on_extra, id_col=id_col
    )
    comps = score_components(errors)
    stats = per_target_error_stats(errors)
    return {
        "components": comps,
        "per_target": stats,
        "n_scored": int(len(joined)),
        "on_extra": on_extra,
        "ignored_extra_n": ignored_extra_n,
    }


def verify_scorer_hash(
    actual_sha256: str, expected: str = PINNED_NOTEBOOK_SHA256
) -> bool:
    """Return True iff the notebook-source hash matches the pinned protocol."""
    return actual_sha256.lower() == expected.lower()
