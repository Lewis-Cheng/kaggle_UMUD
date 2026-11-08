"""pull_scorer hash gate without network."""

from __future__ import annotations

import hashlib
from pathlib import Path

from umud.metric.score import PINNED_NOTEBOOK_SHA256, verify_scorer_hash


def test_verify_scorer_hash_pin():
    assert verify_scorer_hash(PINNED_NOTEBOOK_SHA256)
    assert not verify_scorer_hash("0" * 64)


def test_sha256_of_temp_file(tmp_path: Path):
    payload = b"not-the-real-notebook"
    path = tmp_path / "fake.ipynb"
    path.write_bytes(payload)
    actual = hashlib.sha256(payload).hexdigest()
    assert not verify_scorer_hash(actual)  # must not match pinned protocol
