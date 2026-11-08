"""Unit tests for Phase 2 hashing / duplicate-aware split helpers (no real data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from umud.data.hashing import (
    choose_val_content_groups,
    content_group_id_from_hash,
    parse_numeric_id,
    rival_leak_content_groups,
    rival_val_mask,
    sha256_bytes,
    sha256_file,
)
from umud.data.layout import discover_layout, list_images, resolve_leaf_dir
from umud.data.splits import assert_no_val_group_in_train, assign_duplicate_aware_split


def test_sha256_bytes_and_file(tmp_path: Path):
    payload = b"umud-phase2"
    digest = sha256_bytes(payload)
    p = tmp_path / "x.bin"
    p.write_bytes(payload)
    assert sha256_file(p) == digest
    assert len(digest) == 64


def test_parse_numeric_id_and_rival_mask():
    assert parse_numeric_id("img_42") == 42
    assert parse_numeric_id("0042") == 42
    assert parse_numeric_id("no-digits") is None
    # Rival: (id // 5) % 5 == 0 → ids 0..4 false for 1? 0//5%5=0 True; 5//5%5=1 False; 25//5%5=0 True
    assert rival_val_mask(0) is True
    assert rival_val_mask(5) is False   # 5//5 % 5 == 1
    assert rival_val_mask(25) is True   # 25//5 % 5 == 0
    assert rival_val_mask(10) is False  # 10//5 % 5 == 2


def test_content_group_and_choose_val_deterministic():
    g = content_group_id_from_hash("ab" * 32)
    assert g.startswith("cg_")
    groups = [f"cg_{i}" for i in range(10)]
    a = choose_val_content_groups(groups, val_fraction=0.2, seed=0)
    b = choose_val_content_groups(groups, val_fraction=0.2, seed=0)
    c = choose_val_content_groups(groups, val_fraction=0.2, seed=1)
    assert a == b
    assert a != c
    assert 1 <= len(a) < len(groups)


def test_assign_split_zero_leak_and_rival_leak_detection():
    # Two files, same content group, different numeric ids that rival splits apart.
    rows = [
        {"content_group_id": "cg_same", "numeric_id": 0, "image_id": "a"},   # rival val
        {"content_group_id": "cg_same", "numeric_id": 5, "image_id": "b"},   # rival train
        {"content_group_id": "cg_other", "numeric_id": 6, "image_id": "c"},  # rival train
    ]
    # Our split: put cg_same in val only.
    assigned = assign_duplicate_aware_split(rows, val_content_groups=["cg_same"])
    assert_no_val_group_in_train(assigned)
    assert {r["image_id"]: r["split"] for r in assigned} == {
        "a": "val",
        "b": "val",
        "c": "train",
    }
    leaks = rival_leak_content_groups(assigned)
    assert leaks == ["cg_same"]


def test_discover_layout_nested(tmp_path: Path):
    root = tmp_path / "umud"
    # Nested leaf layout matching SSH.
    pairs = [
        ("apo_imgs_v1", "apo_images_new_model_v1"),
        ("apo_masks_v1", "apo_masks_new_model_v1"),
        ("fasc_imgs_v1", "fasc_images_new_model_v1"),
        ("fasc_masks_v1", "fasc_masks_new_model_v1"),
        ("test_images_v2", "test_set_v2"),
    ]
    for top, leaf in pairs:
        d = root / top / leaf
        d.mkdir(parents=True)
        (d / "1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    (root / "sample_submission.csv").write_text(
        "image_id;pa_deg;fl_mm;mt_mm\n1;0;0;0\n", encoding="utf-8"
    )
    layout = discover_layout(root)
    assert layout.apo_imgs.name == "apo_images_new_model_v1"
    assert layout.sample_submission is not None
    assert len(list_images(layout.test_imgs)) == 1
    # resolve_leaf_dir returns top when images are flat
    flat = tmp_path / "flat_top"
    flat.mkdir()
    (flat / "z.png").write_bytes(b"x")
    assert resolve_leaf_dir(flat) == flat
