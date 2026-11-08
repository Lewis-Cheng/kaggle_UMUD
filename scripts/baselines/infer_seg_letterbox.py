#!/usr/bin/env python3
"""Infer apo (square) + fasc (letterbox if ckpt says so) masks at native resolution."""
from __future__ import annotations

import argparse
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
from umud.seg.ms_unet import MSTinyUNet  # noqa: E402
try:
    from umud.seg.timm_unet import TimmUNet  # noqa: E402
except Exception:  # pragma: no cover
    TimmUNet = None  # noqa: E402


def letterbox_img(im: Image.Image, size: int) -> tuple[Image.Image, dict]:
    w, h = im.size
    scale = min(size / h, size / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(resized, (left, top))
    meta = {"h": h, "w": w, "nh": nh, "nw": nw, "top": top, "left": left, "size": size}
    return canvas, meta


def unletterbox_mask(prob_sq: np.ndarray, meta: dict, thr: float) -> Image.Image:
    top, left, nh, nw = meta["top"], meta["left"], meta["nh"], meta["nw"]
    crop = prob_sq[top : top + nh, left : left + nw]
    up = np.asarray(
        Image.fromarray((crop * 255).astype(np.uint8)).resize((meta["w"], meta["h"]), Image.BILINEAR)
    )
    pred = (up > int(thr * 255)).astype(np.uint8) * 255
    return Image.fromarray(pred)


def load_model(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ck.get("arch", "TinyUNet")
    size = int(ck.get("size", 384))
    base_ch = int(ck.get("base_ch", 32))
    thr = float(ck.get("eval_thr_preferred", 0.5))
    letterbox = bool(ck.get("letterbox", False))
    if arch == "FascUNet":
        model = FascUNet(1, base_ch, True)
    elif arch == "MSTinyUNet":
        model = MSTinyUNet(1, base_ch)
    elif arch == "TimmUNet":
        if TimmUNet is None:
            raise RuntimeError("TimmUNet unavailable")
        model = TimmUNet(ck.get("encoder", "resnet18"), base=base_ch, pretrained=False)
    else:
        model = TinyUNet(1, base_ch)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, size, thr, arch, letterbox, ck


@torch.no_grad()
def predict(model, img_path: Path, size: int, thr: float, device, use_letterbox: bool) -> Image.Image:
    im = Image.open(img_path).convert("L")
    if use_letterbox:
        canvas, meta = letterbox_img(im, size)
        x = np.asarray(canvas, dtype=np.float32) / 255.0
        t = torch.from_numpy(x)[None, None].to(device)
        out = model(t)
        logits = out[0] if isinstance(out, tuple) else out
        if logits.shape[-2:] != (size, size):
            import torch.nn.functional as F
            logits = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return unletterbox_mask(prob, meta, thr)
    w, h = im.size
    x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(x)[None, None].to(device)
    out = model(t)
    logits = out[0] if isinstance(out, tuple) else out
    if logits.shape[-2:] != (size, size):
        import torch.nn.functional as F
        logits = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob > thr).astype(np.uint8) * 255
    return Image.fromarray(pred).resize((w, h), Image.NEAREST)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--apo-ckpt", type=Path, required=True)
    ap.add_argument("--fasc-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    apo, apo_size, apo_thr, apo_arch, apo_lb, apo_ck = load_model(args.apo_ckpt, device)
    fasc, fasc_size, fasc_thr, fasc_arch, fasc_lb, fasc_ck = load_model(args.fasc_ckpt, device)
    print(
        f"apo arch={apo_arch} size={apo_size} thr={apo_thr} dice={apo_ck.get('val_dice')} lb={apo_lb}",
        flush=True,
    )
    print(
        f"fasc arch={fasc_arch} size={fasc_size} thr={fasc_thr} dice={fasc_ck.get('val_dice')} "
        f"variant={fasc_ck.get('variant')} lb={fasc_lb}",
        flush=True,
    )

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    apo_imgs = {p.stem: p for p in list_images(layout.apo_imgs)}
    fasc_imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    test_imgs = {p.stem: p for p in list_images(layout.test_imgs)}
    out_apo = args.out_dir / "pred_apo_masks"
    out_fasc = args.out_dir / "pred_fasc_masks"
    out_apo.mkdir(parents=True, exist_ok=True)
    out_fasc.mkdir(parents=True, exist_ok=True)

    # split_assignments has train/val only; test stems live under layout.test_imgs
    stems = sorted({str(r["image_id"]) for r in rows if r.get("split") in args.splits})
    if "test" in args.splits:
        stems = sorted(set(stems) | set(test_imgs))
    index = []
    for i, s in enumerate(stems):
        rec = {"stem": s}
        # Test images are unlabeled: run BOTH heads (same as prior infer_seg path).
        src = apo_imgs.get(s) or fasc_imgs.get(s) or test_imgs.get(s)
        if s in apo_imgs or s in test_imgs:
            ip = apo_imgs.get(s, src)
            if ip is not None:
                predict(apo, ip, apo_size, apo_thr, device, apo_lb).save(out_apo / f"{s}.png")
                rec["apo"] = True
        if s in fasc_imgs or s in test_imgs:
            ip = fasc_imgs.get(s, src)
            if ip is not None:
                predict(fasc, ip, fasc_size, fasc_thr, device, fasc_lb).save(out_fasc / f"{s}.png")
                rec["fasc"] = True
        index.append(rec)
        if (i + 1) % 50 == 0 or (i + 1) == len(stems):
            print(f"inferred {i+1}/{len(stems)}", flush=True)

    (args.out_dir / "pred_masks_index.json").write_text(json.dumps({"n": len(index), "items": index}, indent=2))
    print(f"wrote {len(index)} under {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
