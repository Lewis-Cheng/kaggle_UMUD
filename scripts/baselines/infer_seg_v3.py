#!/usr/bin/env python3
"""Infer apo (TinyUNet) + fasc (FascUNet/TinyUNet) with optional AmbrosM FOV crop for fasc."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet import TinyUNet  # noqa: E402
from umud.seg.unet_v2 import FascUNet  # noqa: E402


def load_fov_map(scale_csv: Path | None) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    if scale_csv is None or not scale_csv.is_file():
        return out
    with scale_csv.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                l, t, rr, b = int(r["fov_l"]), int(r["fov_t"]), int(r["fov_r"]), int(r["fov_b"])
            except (TypeError, ValueError, KeyError):
                continue
            if rr > l and b > t:
                out[str(r["stem"])] = (l, t, rr, b)
    return out


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    size = int(ckpt.get("size", 256))
    base = int(ckpt.get("base_ch", 32))
    arch = ckpt.get("arch") or ckpt.get("variant") or ""
    thr = float(ckpt.get("eval_thr_preferred", 0.5))
    if "FascUNet" in str(arch) or "fasc_v3" in str(ckpt.get("variant", "")):
        model = FascUNet(1, base, use_aspp=True).to(device)
    else:
        model = TinyUNet(1, base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, size, thr, ckpt


@torch.no_grad()
def predict_full(
    model,
    img_path: Path,
    *,
    size: int,
    device: torch.device,
    thr: float,
    fov: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    raw = Image.open(img_path).convert("L")
    w, h = raw.size
    full = np.zeros((h, w), dtype=np.uint8)
    arr = np.asarray(raw, dtype=np.uint8)
    if fov is not None:
        l0, t0, r0, b0 = fov
        # reject AmbrosM FOV templates that don't fit this image (e.g. 800x1200 box on 556x996)
        if r0 > w + 2 or b0 > h + 2 or (r0 - l0) < w * 0.5 or (b0 - t0) < h * 0.5:
            crop = arr
            l, t, r, b = 0, 0, w, h
        else:
            l, t = max(0, l0), max(0, t0)
            r, b = min(w, r0), min(h, b0)
            crop = arr[t:b, l:r]
            if crop.size == 0:
                crop = arr
                l, t, r, b = 0, 0, w, h
    else:
        crop = arr
        l, t, r, b = 0, 0, w, h
    resized = Image.fromarray(crop).resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)[None, None, ...].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    small = (prob >= thr).astype(np.uint8) * 255
    crop_pred = np.asarray(
        Image.fromarray(small).resize((r - l, b - t), Image.NEAREST), dtype=np.uint8
    )
    full[t:b, l:r] = crop_pred
    return full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--apo-ckpt", type=Path, required=True)
    ap.add_argument("--fasc-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--scale-csv", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--fasc-thr", type=float, default=0.0, help="0=use ckpt preferred")
    ap.add_argument("--apo-thr", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    apo_model, apo_size, _, _ = load_model(args.apo_ckpt, device)
    fasc_model, fasc_size, fasc_thr_ck, fasc_ckpt = load_model(args.fasc_ckpt, device)
    fasc_thr = args.fasc_thr if args.fasc_thr > 0 else fasc_thr_ck
    layout = discover_layout(args.data_root)
    fov_map = load_fov_map(args.scale_csv)
    # also index test/apo FOV by stem from any pool
    with args.scale_csv.open() as fh:
        for r in csv.DictReader(fh):
            try:
                l, t, rr, b = int(r["fov_l"]), int(r["fov_t"]), int(r["fov_r"]), int(r["fov_b"])
            except Exception:
                continue
            if rr > l and b > t:
                fov_map.setdefault(str(r["stem"]), (l, t, rr, b))

    out_apo = args.out_dir / "pred_apo_masks"
    out_fasc = args.out_dir / "pred_fasc_masks"
    out_apo.mkdir(parents=True, exist_ok=True)
    out_fasc.mkdir(parents=True, exist_ok=True)

    apo_imgs = {p.stem: p for p in list_images(layout.apo_imgs)}
    fasc_imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    test_imgs = {p.stem: p for p in list_images(layout.test_imgs)}

    work = []
    if "test" in args.splits:
        for stem, p in sorted(test_imgs.items()):
            work.append(("test", stem, p, p))
    if "val" in args.splits or "train" in args.splits:
        rows = load_split_assignments(args.split_assignments)
        stems = {
            str(r["image_id"])
            for r in rows
            if r.get("split") in args.splits and r.get("split") != "test"
        }
        for stem in sorted(stems):
            work.append(("val", stem, apo_imgs.get(stem), fasc_imgs.get(stem)))

    if args.max_images:
        work = work[: args.max_images]

    index = []
    for i, (split, stem, apo_src, fasc_src) in enumerate(work):
        fov = fov_map.get(stem)
        if apo_src is None and fasc_src is not None:
            apo_src = fasc_src
        if fasc_src is None and apo_src is not None:
            fasc_src = apo_src
        if apo_src is None or fasc_src is None:
            continue
        apo = predict_full(apo_model, apo_src, size=apo_size, device=device, thr=args.apo_thr, fov=None)
        fasc = predict_full(
            fasc_model, fasc_src, size=fasc_size, device=device, thr=fasc_thr, fov=fov
        )
        Image.fromarray(apo).save(out_apo / f"{stem}.png")
        Image.fromarray(fasc).save(out_fasc / f"{stem}.png")
        index.append(
            {
                "split": split,
                "stem": stem,
                "apo_nnz": int((apo > 0).sum()),
                "fasc_nnz": int((fasc > 0).sum()),
                "fov": list(fov) if fov else None,
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == len(work):
            print(f"inferred {i+1}/{len(work)}")

    meta = {
        "n": len(index),
        "apo_ckpt": str(args.apo_ckpt),
        "fasc_ckpt": str(args.fasc_ckpt),
        "fasc_thr": fasc_thr,
        "fasc_variant": fasc_ckpt.get("variant"),
        "device": str(device),
    }
    (args.out_dir / "pred_masks_index.json").write_text(
        json.dumps({"meta": meta, "items": index}, indent=2)
    )
    print(f"wrote {len(index)} pairs thr_fasc={fasc_thr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
