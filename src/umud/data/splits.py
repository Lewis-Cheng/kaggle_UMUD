"""Duplicate-aware split helpers.

Gate: no validation content group (or known subject) may occur in training
for either the aponeurosis or fascicle task/head. Baseline and candidate must
share this split.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


class SplitError(ValueError):
    """Raised when a proposed split violates cross-task exclusion."""


def assign_duplicate_aware_split(
    rows: Sequence[Mapping[str, object]],
    *,
    val_content_groups: Iterable[str],
    group_key: str = "content_group_id",
    split_key: str = "split",
) -> list[dict[str, object]]:
    """Assign train/val by content_group_id.

    Every row whose content_group_id is in val_content_groups becomes ``val``;
    all others become ``train``. Raises if the same content group would be
    forced into both (should not happen with this rule, but guards callers
    who pre-labelled rows inconsistently).
    """
    val_set = {str(g) for g in val_content_groups}
    out: list[dict[str, object]] = []
    group_to_splits: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        item = dict(row)
        gid = str(item[group_key])
        split = "val" if gid in val_set else "train"
        item[split_key] = split
        group_to_splits[gid].add(split)
        out.append(item)

    leaked = {g: s for g, s in group_to_splits.items() if s == {"train", "val"}}
    if leaked:
        raise SplitError(f"content groups present in both splits: {sorted(leaked)[:10]}")
    return out


def assert_no_val_group_in_train(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str = "content_group_id",
    split_key: str = "split",
) -> None:
    """Fail closed if any content_group_id appears in both train and val."""
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[str(row[split_key])].add(str(row[group_key]))
    overlap = by_split.get("train", set()) & by_split.get("val", set())
    if overlap:
        raise SplitError(
            f"validation content groups leaked into train: {sorted(overlap)[:10]}"
        )
