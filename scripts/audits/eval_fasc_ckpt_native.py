#!/usr/bin/env python3
"""Re-evaluate a fasc ckpt with native-res macro Dice (fair gate metric)."""
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
    from umud.seg.timm_unet import TimmUNet
except Exception:
    TimmUNet = None


def letterbox(arr: np.ndarray, size: int, *, nearest: bool):
    h, w = arr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resample = Image.NEAREST if nearest else Image.BILINEAR
    src = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    resized = Image.fromarray(src).resize((nw, nh), resample)
    canvas = Image.new("L", (size, size), 0)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas), {"h": h, "w": w, "nh": nh, "nw": nw, "top": top, "left": left, "size": size}


def unletterbox_prob(prob_sq, meta):
    top, left, nh, nw = meta["top"], meta["left"], meta["nh"], meta["nw"]
    crop = prob_sq[top : top + nh, left : left + nw]
    return (
        np.asarray(
            Image.fromarray((crop * 255).astype(np.uint8)).resize((meta["w"], meta["h"]), Image.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )



def _logits(model, t, size):
    import torch.nn.functional as F
    out = model(t)
    logits = out[0] if isinstance(out, tuple) else out
    if logits.shape[-2:] != (size, size):
        logits = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
    return logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force-letterbox", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = ck.get("arch", "TinyUNet")
    size = int(ck.get("size", 384))
    base = int(ck.get("base_ch", 32))
    use_lb = bool(ck.get("letterbox", False) or args.force_letterbox)
    if arch == "FascUNet":
        model = FascUNet(1, base, True)
    elif arch == "MSTinyUNet":
        model = MSTinyUNet(1, base)
    elif arch == "TimmUNet":
        model = TimmUNet(ck.get("encoder", "resnet18"), base=base, pretrained=False)
    else:
        model = TinyUNet(1, base)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}
    val = []
    for r in rows:
        if r.get("task") == "fascicle" and r.get("split") == "val":
            s = str(r["image_id"])
            if s in imgs and s in masks:
                val.append((s, imgs[s], masks[s]))

    def eval_thr(thr: float, letterbox_mode: bool):
        dices = []
        empty = 0
        with torch.no_grad():
            for stem, ip, mp in val:
                im = Image.open(ip).convert("L")
                hard = np.asarray(Image.open(mp).convert("L")) > 0
                w, h = im.size
                if hard.shape != (h, w):
                    hard = (
                        np.asarray(
                            Image.fromarray(hard.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
                        )
                        > 127
                    )
                if letterbox_mode:
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    sq, meta = letterbox(arr, size, nearest=False)
                    x = torch.from_numpy(sq.astype(np.float32) / 255.0)[None, None].to(device)
                    prob_sq = torch.sigmoid(_logits(model, x, size))[0, 0].cpu().numpy()
                    prob = unletterbox_prob(prob_sq, meta)
                else:
                    x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
                    t = torch.from_numpy(x)[None, None].to(device)
                    prob_sq = torch.sigmoid(_logits(model, t, size))[0, 0].cpu().numpy()
                    prob = (
                        np.asarray(
                            Image.fromarray((prob_sq * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
                pred = prob > thr
                if pred.sum() == 0:
                    empty += 1
                inter = float((pred & hard).sum())
                den = float(pred.sum() + hard.sum())
                dices.append((2 * inter + 1e-6) / (den + 1e-6))
        a = np.asarray(dices)
        return {
            "n": int(a.size),
            "macro_dice": float(a.mean()),
            "dice_p50": float(np.median(a)),
            "empty_pred_frac": float(empty / max(1, a.size)),
            "thr": thr,
            "letterbox": letterbox_mode,
        }

    report = {
        "ckpt": str(args.ckpt),
        "arch": arch,
        "size": size,
        "variant": ck.get("variant"),
        "stored_val_dice": ck.get("val_dice"),
        "square_resize_native": {str(t): eval_thr(t, False) for t in (0.3, 0.35, 0.5)},
        "letterbox_native": {str(t): eval_thr(t, True) for t in (0.3, 0.35, 0.5)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
