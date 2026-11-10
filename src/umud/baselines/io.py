"""I/O helpers for Phase 3 floor baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from umud.data.layout import discover_layout, list_images
from umud.metric.submission import validate_submission

# Competition sample uses image_id; local validator historically used id.
SUBMISSION_ID_COL = "image_id"
PRED_COLS = ("pa_deg", "fl_mm", "mt_mm")


def load_split_assignments(path: Path | str) -> list[dict[str, Any]]:
    """Load Phase 2 ``split_assignments.jsonl`` rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Index assignments by task → split → rows."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        task = str(row.get("task", "unknown"))
        split = str(row.get("split", "train"))
        out.setdefault(task, {}).setdefault(split, []).append(dict(row))
    return out


def load_test_ids(data_root: Path | str) -> list[str]:
    """Test image stems from layout; fall back to sample_submission ids."""
    layout = discover_layout(data_root)
    stems = [p.stem for p in list_images(layout.test_imgs)]
    if stems:
        return stems
    if layout.sample_submission is None:
        raise FileNotFoundError("no test images and no sample_submission.csv")
    return read_sample_ids(layout.sample_submission)


def load_official_submission_ids(path: Path | str) -> list[str]:
    """Copy literal submission ids from the official test-name list.

    Each image line is a path or a filename. The id is that line's final
    path component exactly as written. ``sample_submission.csv`` is the
    list's header artifact, not an image. This never strips a suffix and
    never appends one. Duplicate filenames abort.
    """
    text = Path(path).read_text(encoding="utf-8")
    ids: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        name = line.replace("\\", "/").rsplit("/", 1)[-1]
        if name == "sample_submission.csv":
            continue
        if not name or name in {".", ".."}:
            raise ValueError(f"official id line has no filename: {line!r}")
        ids.append(name)
    if not ids:
        raise ValueError(f"official submission id list is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError("official submission ids collide")
    return ids


def read_sample_ids(path: Path | str) -> list[str]:
    """Parse sample_submission ids (supports ``;`` or ``,``)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return []
    delim = ";" if lines[0].count(";") >= lines[0].count(",") else ","
    reader = csv.DictReader(lines, delimiter=delim)
    ids: list[str] = []
    for row in reader:
        if SUBMISSION_ID_COL in row:
            ids.append(str(row[SUBMISSION_ID_COL]).strip())
        elif "id" in row:
            ids.append(str(row["id"]).strip())
        else:
            # first column
            first = next(iter(row.values()))
            ids.append(str(first).strip())
    return ids


def write_submission_csv(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    delimiter: str = ",",
    validate_against: Sequence[str] | None = None,
) -> Path:
    """Write ``image_id,pa_deg,fl_mm,mt_mm`` submission CSV.

    When ``validate_against`` is set, also run the local exact-ID validator
    (renaming ``image_id`` → ``id``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        records.append(
            {
                SUBMISSION_ID_COL: str(row[SUBMISSION_ID_COL]),
                "pa_deg": float(row["pa_deg"]),
                "fl_mm": float(row["fl_mm"]),
                "mt_mm": float(row["mt_mm"]),
            }
        )
    df = pd.DataFrame(records, columns=[SUBMISSION_ID_COL, *PRED_COLS])
    df.to_csv(path, index=False, sep=delimiter)
    if validate_against is not None:
        check = df.rename(columns={SUBMISSION_ID_COL: "id"})
        validate_submission(check, list(validate_against))
    return path


def require_scale_provenance(value: str) -> str:
    """Normalize / accept scale provenance labels (never invent real)."""
    v = (value or "").strip().lower()
    if not v:
        raise ValueError("--scale-provenance is required (e.g. unknown)")
    # Allow a small controlled vocabulary; anything else is kept verbatim.
    allowed_prefix = (
        "unknown",
        "rival_guess",
        "scale_bar",
        "organizer",
        "manual",
        "literature",
        "pixel_space",
    )
    if not any(v == p or v.startswith(p + "_") or v.startswith(p + ":") for p in allowed_prefix):
        # Still accept, but callers should prefer the vocabulary above.
        return value.strip()
    return value.strip()
