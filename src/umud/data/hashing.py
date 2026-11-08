"""File hashing and rival-split helpers for Phase 2 audits."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

_NUMERIC_STEM = re.compile(r"(\d+)")


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA-256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_numeric_id(stem: str) -> int | None:
    """Extract the last contiguous digit run from a filename stem, if any.

    Rival notebook used integer ids for ``(id // 5) % 5 == 0`` val assignment.
    Returns None when no digits are present.
    """
    matches = _NUMERIC_STEM.findall(stem)
    if not matches:
        return None
    return int(matches[-1])


def rival_val_mask(numeric_id: int) -> bool:
    """Rival public split: ``(numeric_id // 5) % 5 == 0`` → validation."""
    return (int(numeric_id) // 5) % 5 == 0


def content_group_id_from_hash(file_hash: str, *, prefix: str = "cg_") -> str:
    """Stable content_group_id from an exact file hash (full hex)."""
    return f"{prefix}{file_hash}"


def choose_val_content_groups(
    content_group_ids: Iterable[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> list[str]:
    """Deterministically pick a fraction of unique content groups for validation.

    Groups are sorted then shuffled with a Python ``random.Random(seed)`` so the
    choice is reproducible across platforms without numpy.
    """
    import random

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    unique = sorted({str(g) for g in content_group_ids})
    if not unique:
        return []
    rng = random.Random(seed)
    order = unique[:]
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    n_val = min(n_val, len(order) - 1) if len(order) > 1 else 1
    return sorted(order[:n_val])


def rival_leak_content_groups(
    rows: Iterable[dict],
    *,
    group_key: str = "content_group_id",
    numeric_id_key: str = "numeric_id",
) -> list[str]:
    """Content groups that rival's block-of-5 split would put in both train and val.

    Each row needs ``content_group_id`` and ``numeric_id`` (int). Rows with
    ``numeric_id is None`` are ignored for the rival assignment.
    """
    from collections import defaultdict

    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        nid = row.get(numeric_id_key)
        if nid is None:
            continue
        split = "val" if rival_val_mask(int(nid)) else "train"
        by_group[str(row[group_key])].add(split)
    return sorted(g for g, splits in by_group.items() if splits == {"train", "val"})
