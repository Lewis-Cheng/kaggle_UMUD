"""MultExp holdout for training manifests (dHash Hamming distance)."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

MULTEXP_MAX_DIST = 8


class HoldoutLeak(RuntimeError):
    pass


def near_any(h: np.ndarray, others: Sequence[np.ndarray], max_dist: int = MULTEXP_MAX_DIST) -> bool:
    h = np.asarray(h, dtype=np.uint8)
    return any(int((h != np.asarray(o, dtype=np.uint8)).sum()) <= max_dist for o in others)


def assert_multexp_excluded(
    train_hashes: Iterable[np.ndarray],
    multexp_hashes: Sequence[np.ndarray],
    keys: Sequence[str] | None = None,
    max_dist: int = MULTEXP_MAX_DIST,
) -> None:
    for i, h in enumerate(train_hashes):
        if near_any(h, multexp_hashes, max_dist):
            name = keys[i] if keys is not None else str(i)
            raise HoldoutLeak(f"training image {name} is within dHash {max_dist} of a MultExp image")
