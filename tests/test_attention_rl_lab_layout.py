"""Run every stage of train_spatial_attention_rl.py on a tiny fake data layout.

Synthetic images stand in for competition files and cached maps stand in for
Frundles output, so no weights or GPU are needed. Checks the file plumbing,
the MultExp holdout, and the three submission CSVs, not model quality.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from umud.data.synthetic_us import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("train_spatial_attention_rl", ROOT / "scripts/chase018/train_spatial_attention_rl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


def _maps(path: Path, s) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, apo=s.p_apo.astype(np.float16), fasc=s.p_fasc.astype(np.float16))


def _layout(root: Path) -> None:
    train = make_dataset(24, seed=11)
    mex = make_dataset(5, seed=12)
    test = make_dataset(6, seed=13)
    raw = root / "data/raw/umud"
    for i, s in enumerate(train):
        stem = f"image_{i:04d}"
        _write_png(raw / "apo_imgs_v1" / f"{stem}.png", s.image)
        _write_png(raw / "fasc_imgs_v1" / f"{stem}.png", s.image)
        _write_png(raw / "apo_masks_v1" / f"{stem}.png", np.maximum(s.sup_mask, s.deep_mask) * 255)
        _write_png(raw / "fasc_masks_v1" / f"{stem}.png", s.fasc_mask * 255)
        _maps(root / "reports/phase3/chase018/cache/maps_train_overlap" / f"{stem}.npz", s)
    leak = "image_9999"
    _write_png(raw / "apo_imgs_v1" / f"{leak}.png", mex[0].image)
    _write_png(raw / "fasc_imgs_v1" / f"{leak}.png", mex[0].image)
    _write_png(raw / "apo_masks_v1" / f"{leak}.png", np.maximum(mex[0].sup_mask, mex[0].deep_mask) * 255)
    _write_png(raw / "fasc_masks_v1" / f"{leak}.png", mex[0].fasc_mask * 255)
    _maps(root / "reports/phase3/chase018/cache/maps_train_overlap" / f"{leak}.npz", mex[0])

    ext = root / "data/external/umud_expert_arch_v0.1.0"
    rows = []
    for i, s in enumerate(mex):
        name = f"mex_{i:02d}.png"
        _write_png(ext / "benchmark_dataset_architecture_v0.1.0" / name, s.image)
        _maps(root / "reports/phase3/chase018/cache/maps_multexp" / f"{name}.npz", s)
        rows.append({"image_file": name, "Scale_pixel_per_cm": 10.0 / s.mm_per_px, "median_PA": s.labels["pa_deg"],
                     "median_FL": s.labels["fl_mm"], "median_MT": s.labels["mt_mm"]})
    pd.DataFrame(rows).to_csv(ext / "expert_consensus.csv", index=False)

    ids = [f"IMG_{i:05d}.png" for i in range(len(test))]
    for iid, s in zip(ids, test):
        _write_png(raw / "test_images_v2/test_set_v2" / iid, s.image)
        _maps(root / "reports/phase3/chase018/cache/maps_test" / f"{iid}.npz", s)
    (root / "reports/phase3").mkdir(parents=True, exist_ok=True)
    (root / "reports/phase3/official_test_file_names.txt").write_text("\n".join([*ids, "sample_submission.csv"]) + "\n")
    pd.DataFrame({"image_id": ids, "sx": [1 / s.mm_per_px for s in test], "sy": [1 / s.mm_per_px for s in test], "layout": "syn"}).to_csv(
        root / "reports/phase3/chase018/scale_aniso.csv", index=False
    )
    pd.DataFrame({"image_id": ids, "pa_deg": 15.0, "fl_mm": 60.0, "mt_mm": 15.0}).to_csv(
        root / "reports/phase3/chase018/submission_texture_rl.csv", index=False
    )


def test_all_stages_on_fake_layout(tmp_path: Path):
    _layout(tmp_path)
    mod = _load_script()
    common = ["--root", str(tmp_path), "--device", "cpu", "--grid", "48", "72"]
    assert mod.main(["prepare", *common]) == 0
    manifest = json.loads((tmp_path / "reports/phase3/chase018/attention_rl/lab/manifest.json").read_text())
    assert manifest["dropped_near_multexp"] == ["image_9999"]
    assert "image_9999" not in manifest["stems"]
    assert mod.main(["train", *common, "--epochs", "1", "--dqn-steps", "300", "--width", "8", "--depth", "2", "--batch", "8", "--min-val-scored", "0"]) == 0
    out = tmp_path / "reports/phase3/chase018/attention_rl/lab"
    report = json.loads((out / "val_report.json").read_text())
    assert set(report["gates"]) == {"val_attention_beats_maps", "val_rl_beats_init"}
    assert mod.main(["eval-multexp", *common, "--use-rl", "on"]) == 0
    mr = json.loads((out / "multexp_report.json").read_text())
    assert mr["n"] == 5 and mr["label"] == "S_expert_NOT_LB"
    assert mod.main(["infer-test", *common, "--use-rl", "on"]) == 0
    ids = [f"IMG_{i:05d}.png" for i in range(6)]
    for name in ("pa_only", "fl_only", "pa_fl"):
        df = pd.read_csv(out / f"submission_attn_rl_{name}.csv")
        assert df.image_id.tolist() == ids
        assert np.isfinite(df[["pa_deg", "fl_mm", "mt_mm"]].values).all()
    pa_only = pd.read_csv(out / "submission_attn_rl_pa_only.csv")
    assert (pa_only.fl_mm == 60.0).all() and (pa_only.mt_mm == 15.0).all()
    meta = json.loads((out / "test_meta.json").read_text())
    assert meta["submitted"] is False and meta["fl_affine"] is None
