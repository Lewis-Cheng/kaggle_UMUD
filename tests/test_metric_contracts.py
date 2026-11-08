"""Official metric + submission validator contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umud.metric.score import (
    TOLERANCES,
    absolute_errors_from_frames,
    official_score,
    per_target_error_stats,
    score_components,
    score_prediction_frames,
)
from umud.metric.submission import SubmissionValidationError, validate_submission


def test_metric_s1_weighting_with_known_mae():
    """Equal-weight S1 = mean of (MAE_t / tol_t) over PA/FL/MT."""
    # Construct constant absolute errors equal to each tolerance → each term = 1 → S1 = 1.
    n = 5
    errors = {
        "pa_deg": np.full(n, TOLERANCES["pa_deg"]),
        "fl_mm": np.full(n, TOLERANCES["fl_mm"]),
        "mt_mm": np.full(n, TOLERANCES["mt_mm"]),
    }
    comps = score_components(errors)
    assert comps["S1"] == pytest.approx(1.0, abs=1e-12)
    # Median AE same as MAE here; RMSE same as MAE for constant errors.
    assert comps["S2"] == pytest.approx(1.0, abs=1e-12)
    assert comps["S3"] == pytest.approx(1.0, abs=1e-12)
    assert comps["S"] == pytest.approx(1.0 + 1e-6 + 1e-9, abs=1e-15)
    assert official_score(errors) == pytest.approx(comps["S"], abs=1e-15)

    # Half-tolerance MAE → S1 = 0.5
    half = {
        "pa_deg": np.full(n, TOLERANCES["pa_deg"] / 2),
        "fl_mm": np.full(n, TOLERANCES["fl_mm"] / 2),
        "mt_mm": np.full(n, TOLERANCES["mt_mm"] / 2),
    }
    assert score_components(half)["S1"] == pytest.approx(0.5, abs=1e-12)


def test_submission_validator_rejects_missing_dup_and_nan():
    expected = ["a", "b", "c"]

    # Happy path
    ok = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "pa_deg": [1.0, 2.0, 3.0],
            "fl_mm": [10.0, 11.0, 12.0],
            "mt_mm": [4.0, 5.0, 6.0],
        }
    )
    validate_submission(ok, expected)

    # Missing ID
    missing = ok.iloc[:2].copy()
    with pytest.raises(SubmissionValidationError, match="ID set mismatch"):
        validate_submission(missing, expected)

    # Duplicate ID
    dup = pd.DataFrame(
        {
            "id": ["a", "a", "b"],
            "pa_deg": [1.0, 1.0, 2.0],
            "fl_mm": [10.0, 10.0, 11.0],
            "mt_mm": [4.0, 4.0, 5.0],
        }
    )
    with pytest.raises(SubmissionValidationError, match="duplicate"):
        validate_submission(dup, ["a", "b"])

    # Extra ID (exact-set contract; stronger than notebook inner join)
    extra = pd.DataFrame(
        {
            "id": ["a", "b", "c", "zzz"],
            "pa_deg": [1.0, 2.0, 3.0, 0.0],
            "fl_mm": [10.0, 11.0, 12.0, 0.0],
            "mt_mm": [4.0, 5.0, 6.0, 0.0],
        }
    )
    with pytest.raises(SubmissionValidationError, match="ID set mismatch"):
        validate_submission(extra, expected)

    # NaN / non-finite
    bad = ok.copy()
    bad.loc[1, "fl_mm"] = np.nan
    with pytest.raises(SubmissionValidationError, match="non-finite"):
        validate_submission(bad, expected)

    inf = ok.copy()
    inf.loc[0, "pa_deg"] = np.inf
    with pytest.raises(SubmissionValidationError, match="non-finite"):
        validate_submission(inf, expected)


def test_score_components_rejects_empty_error_vectors():
    with pytest.raises(ValueError, match="non-empty"):
        score_components(
            {
                "pa_deg": np.array([]),
                "fl_mm": np.array([]),
                "mt_mm": np.array([]),
            }
        )


def test_empty_join_raises():
    """No overlapping ids → cannot score (documented empty-join policy)."""
    pred = pd.DataFrame(
        {"id": ["x"], "pa_deg": [1.0], "fl_mm": [2.0], "mt_mm": [3.0]}
    )
    gt = pd.DataFrame(
        {"id": ["y"], "pa_deg": [1.0], "fl_mm": [2.0], "mt_mm": [3.0]}
    )
    with pytest.raises(ValueError, match="empty join"):
        absolute_errors_from_frames(pred, gt, on_extra="ignore")


def test_extra_rows_reject_vs_ignore_policy():
    """Documented policy: reject raises; ignore drops extras then scores.

    Organizer notebook inner-joins (extras can disappear). Local default is
    reject; ignore approximates the notebook drop behavior for scoring only.
    """
    gt = pd.DataFrame(
        {
            "id": ["a", "b"],
            "pa_deg": [10.0, 20.0],
            "fl_mm": [100.0, 200.0],
            "mt_mm": [5.0, 6.0],
        }
    )
    pred = pd.DataFrame(
        {
            "id": ["a", "b", "zzz"],
            "pa_deg": [10.0, 20.0, 0.0],
            "fl_mm": [100.0, 200.0, 0.0],
            "mt_mm": [5.0, 6.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="absent from gt"):
        absolute_errors_from_frames(pred, gt, on_extra="reject")

    report = score_prediction_frames(pred, gt, on_extra="ignore")
    assert report["n_scored"] == 2
    assert report["ignored_extra_n"] == 1
    assert report["components"]["S1"] == pytest.approx(0.0, abs=1e-12)


def test_score_prediction_frames_known_errors():
    gt = pd.DataFrame(
        {
            "id": ["a", "b"],
            "pa_deg": [0.0, 0.0],
            "fl_mm": [0.0, 0.0],
            "mt_mm": [0.0, 0.0],
        }
    )
    pred = pd.DataFrame(
        {
            "id": ["a", "b"],
            "pa_deg": [TOLERANCES["pa_deg"], TOLERANCES["pa_deg"]],
            "fl_mm": [TOLERANCES["fl_mm"], TOLERANCES["fl_mm"]],
            "mt_mm": [TOLERANCES["mt_mm"], TOLERANCES["mt_mm"]],
        }
    )
    report = score_prediction_frames(pred, gt, on_extra="reject")
    assert report["components"]["S1"] == pytest.approx(1.0, abs=1e-12)
    stats = report["per_target"]
    assert stats["pa_deg"]["MAE"] == pytest.approx(TOLERANCES["pa_deg"])
    assert stats["fl_mm"]["MedAE"] == pytest.approx(TOLERANCES["fl_mm"])
    assert stats["mt_mm"]["RMSE"] == pytest.approx(TOLERANCES["mt_mm"])


def test_per_target_error_stats_keys():
    errors = {
        "pa_deg": np.array([1.0, 3.0]),
        "fl_mm": np.array([2.0, 4.0]),
        "mt_mm": np.array([0.5, 1.5]),
    }
    stats = per_target_error_stats(errors)
    for key in ("pa_deg", "fl_mm", "mt_mm"):
        assert set(stats[key]) >= {
            "MAE",
            "MedAE",
            "RMSE",
            "tol",
            "MAE_over_tol",
            "MedAE_over_tol",
            "RMSE_over_tol",
        }
