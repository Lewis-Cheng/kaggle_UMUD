#!/usr/bin/env python3
"""Infer Track-A v5+Attention on official test → submission CSV (semicolon)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.ambrosm_scale_io import load_ambrosm_scale_rows, scale_map_by_stem  # noqa: E402
from umud.baselines.io import (  # noqa: E402
    load_test_ids,
    read_sample_ids,
    write_submission_csv,
)
from umud.baselines.mask_geometry import measure_masks  # noqa: E402
from umud.baselines.mask_proxy import FALLBACK_PRIORS, load_mask  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.models.anatomy_query_v5 import V5AttentionRegressor  # noqa: E402


def _mask_path(folder: Path | None, stem: str) -> Path | None:
    if folder is None or not folder.is_dir():
        return None
    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _load_gray(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("L")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _load_mask01(path: Path | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=np.float32)
    im = Image.open(path).convert("L")
    im = im.resize((size, size), Image.NEAREST)
    arr = np.asarray(im, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return (arr > 0.5).astype(np.float32)


def _measure(
    stem: str,
    *,
    fasc_dir: Path,
    apo_dir: Path,
    scale,
    priors: dict[str, float],
) -> dict[str, Any]:
    fasc = apo = None
    fp = _mask_path(fasc_dir, stem)
    ap = _mask_path(apo_dir, stem)
    if fp is not None:
        fasc = load_mask(fp)
    if ap is not None:
        apo = load_mask(ap)
    usable = scale is not None and getattr(scale, "usable", False)
    sx = sy = float(scale.mm_per_px) if usable else 1.0
    m = measure_masks(fasc_mask=fasc, apo_mask=apo, sx=sx, sy=sy, priors=priors)
    m["stem"] = stem
    m["usable"] = bool(usable and m.get("ok"))
    m["mm_per_px"] = float(scale.mm_per_px) if usable else 0.2
    return m


def resolve_test_ids(layout) -> list[str]:
    """Prefer full test image set; stub sample_submission (2 rows) is not enough."""
    test_stems = load_test_ids(layout.data_root)
    sample_ids: list[str] = []
    if layout.sample_submission is not None and layout.sample_submission.is_file():
        sample_ids = read_sample_ids(layout.sample_submission)
    if sample_ids and len(sample_ids) >= max(len(test_stems), 1):
        return sample_ids
    suffix = ".tif"
    if sample_ids:
        suffix = Path(sample_ids[0]).suffix or ".tif"
    # Preserve sample ordering when possible; else sorted stems.
    return [f"{s}{suffix}" for s in test_stems]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ambrosm-scale-csv", type=Path, required=True)
    ap.add_argument("--pred-mask-dir", type=Path, required=True, help="val+test v5 preds")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--delimiter", type=str, default=";")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    size = int(ck.get("size", 256))
    priors = dict(ck.get("priors") or FALLBACK_PRIORS)
    model = V5AttentionRegressor(size_hint=size).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    layout = discover_layout(args.data_root)
    scales = scale_map_by_stem(load_ambrosm_scale_rows(args.ambrosm_scale_csv))
    pred_apo = args.pred_mask_dir / "pred_apo_masks"
    pred_fasc = args.pred_mask_dir / "pred_fasc_masks"

    test_imgs = {p.stem: p for p in list_images(layout.test_imgs)}
    for p in list_images(layout.apo_imgs) + list_images(layout.fasc_imgs):
        test_imgs.setdefault(p.stem, p)

    test_ids = resolve_test_ids(layout)
    stems = [Path(i).stem for i in test_ids]
    id_for = {Path(i).stem: i for i in test_ids}
    print(f"n_test_ids={len(test_ids)} n_test_imgs={len(test_imgs)}", flush=True)

    rows_out = []
    n_attn = 0
    n_base = 0
    n_missing_mask = 0
    for i, stem in enumerate(stems):
        scale = scales.get(stem)
        base_m = _measure(stem, fasc_dir=pred_fasc, apo_dir=pred_apo, scale=scale, priors=priors)
        if _mask_path(pred_fasc, stem) is None and _mask_path(pred_apo, stem) is None:
            n_missing_mask += 1
        base = np.array(
            [
                float(base_m["pa_deg"]) if base_m.get("ok") else float(priors["pa_deg"]),
                float(base_m["fl_mm"]) if base_m.get("ok") else float(priors["fl_mm"]),
                float(base_m["mt_mm"]) if base_m.get("ok") else float(priors["mt_mm"]),
            ],
            dtype=np.float32,
        )
        img_path = test_imgs.get(stem)
        if img_path is None:
            pa, fl, mt = map(float, base)
            n_base += 1
        else:
            mm = float(base_m["mm_per_px"]) if base_m.get("mm_per_px") else 0.2
            if scale is not None and getattr(scale, "usable", False):
                mm = float(scale.mm_per_px)
            img = _load_gray(img_path, size)
            apo = _load_mask01(_mask_path(pred_apo, stem), size)
            fasc = _load_mask01(_mask_path(pred_fasc, stem), size)
            x = torch.from_numpy(np.stack([img, apo, fasc], 0))[None].to(device)
            base_t = torch.from_numpy(base)[None].to(device)
            log_mm = torch.tensor([[math.log(max(mm, 1e-6))]], device=device)
            spacing = torch.tensor([[mm, mm]], device=device, dtype=torch.float32)
            out = model(x, base_pa_fl_mt=base_t, log_mm_per_px=log_mm, spacing_mm=spacing)
            pred = out["pa_fl_mt"][0].cpu().numpy()
            pa, fl, mt = float(pred[0]), float(pred[1]), float(pred[2])
            pa = float(np.clip(pa, 0.0, 45.0))
            fl = float(np.clip(fl, 5.0, 200.0))
            mt = float(np.clip(mt, 1.0, 80.0))
            n_attn += 1
        rows_out.append(
            {
                "image_id": id_for[stem],
                "pa_deg": pa,
                "fl_mm": fl,
                "mt_mm": mt,
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(stems):
            print(f"inferred {i+1}/{len(stems)}", flush=True)

    write_submission_csv(args.out_csv, rows_out, delimiter=args.delimiter)
    meta = {
        "n": len(rows_out),
        "n_attn": n_attn,
        "n_base_only": n_base,
        "n_missing_mask": n_missing_mask,
        "ckpt": str(args.ckpt),
        "pred_mask_dir": str(args.pred_mask_dir),
        "out_csv": str(args.out_csv),
        "variant": "trackA_v5_attention",
        "val_metrics": ck.get("val"),
        "score": ck.get("score"),
    }
    args.out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
