"""Submission ID-set validator (stronger than the notebook inner join).

The organizer scorer aligns by image ID and rejects missing IDs, duplicates,
non-numeric predictions, NaNs, and infinities. Extra unrelated rows can
disappear in its inner join. This local validator enforces an exact ID-set
contract: every expected test ID once, nothing else, finite preds.

Join / extra-row policy summary
-------------------------------
| Context                         | Extra pred rows        | Missing ids |
|---------------------------------|------------------------|-------------|
| Organizer notebook scorer       | silently dropped (IJ)  | reject      |
| validate_submission (local)     | reject                 | reject      |
| score_prediction_frames reject  | raise                  | empty join  |
| score_prediction_frames ignore  | drop then score        | empty join  |
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("id", "pa_deg", "fl_mm", "mt_mm")
PRED_COLUMNS = ("pa_deg", "fl_mm", "mt_mm")


class SubmissionValidationError(ValueError):
    """Raised when a submission violates the exact-ID / finite-pred contract."""


def load_expected_ids(path: Path | str) -> list[str]:
    """Load one image id per non-empty line from a text file."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_prediction_csv(path: Path | str) -> pd.DataFrame:
    """Read a submission / prediction CSV (id, pa_deg, fl_mm, mt_mm)."""
    return pd.read_csv(path)


def validate_submission(
    df: pd.DataFrame,
    expected_ids: Sequence[str] | Iterable[str],
) -> None:
    """Validate a submission DataFrame against the expected test ID set.

    Raises
    ------
    SubmissionValidationError
        On missing columns, duplicate IDs, ID-set mismatch, or non-finite preds.
    """
    expected = [str(x) for x in expected_ids]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise SubmissionValidationError("expected_ids itself contains duplicates")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SubmissionValidationError(f"missing columns: {missing_cols}")

    ids = df["id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise SubmissionValidationError("duplicate prediction IDs")

    pred_set = set(ids)
    missing = expected_set - pred_set
    extra = pred_set - expected_set
    if missing or extra:
        raise SubmissionValidationError(
            f"ID set mismatch; missing={sorted(missing)[:5]}... "
            f"extra={sorted(extra)[:5]}... "
            f"(missing_n={len(missing)}, extra_n={len(extra)})"
        )

    for col in PRED_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise SubmissionValidationError(
                f"non-finite predictions in column {col} (NaN/inf/non-numeric)"
            )
