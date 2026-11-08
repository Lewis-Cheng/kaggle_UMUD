#!/usr/bin/env python3
"""Infer apo+fasc masks; supports TinyUNet and FascUNet ckpts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from umud.baselines.io import load_split_assignments
from umud.data.layout import discover_layout, list_images
from umud.seg.unet import TinyUNet
from umud.seg.unet_v2 import FascUNet


def load_model(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ck.get("arch", "TinyUNet")
    size = int(ck.get("size", 384))
    base_ch = int(ck.get("base_ch", 32))
    thr = float(ck.get("eval_thr_preferred", 0.5))
    if arch == "FascUNet":
        model = FascUNet(1, base_ch, True)
    else:
        model = TinyUNet(1, base_ch)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, size, thr, arch, ck


@torch.no_grad()
def predict_mask(model, img_path: Path, size: int, thr: float, device) -> Image.Image:
    im = Image.open(img_path).convert("L")
    w, h = im.size
    x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(x)[None, None].to(device)
    prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    pred = (prob > thr).astype(np.uint8) * 255
    out = Image.fromarray(pred).resize((w, h), Image.NEAREST)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--apo-ckpt", type=Path, required=True)
    ap.add_argument("--fasc-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    apo, apo_size, apo_thr, apo_arch, apo_ck = load_model(args.apo_ckpt, device)
    fasc, fasc_size, fasc_thr, fasc_arch, fasc_ck = load_model(args.fasc_ckpt, device)
    print(f"apo arch={apo_arch} size={apo_size} thr={apo_thr} dice={apo_ck.get('val_dice')}")
    print(f"fasc arch={fasc_arch} size={fasc_size} thr={fasc_thr} dice={fasc_ck.get('val_dice')} variant={fasc_ck.get('variant')}")

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    apo_imgs = {p.stem: p for p in list_images(layout.apo_imgs)}
    fasc_imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}

    out_apo = args.out_dir / "pred_apo_masks"
    out_fasc = args.out_dir / "pred_fasc_masks"
    out_apo.mkdir(parents=True, exist_ok=True)
    out_fasc.mkdir(parents=True, exist_ok=True)

    # gather stems for requested splits (union of apo/fasc tasks)
    stems = []
    for r in rows:
        if r.get("split") in args.splits:
            stems.append(str(r["image_id"]))
    stems = sorted(set(stems))
    index = []
    for i, s in enumerate(stems):
        rec = {"stem": s}
        if s in apo_imgs:
            outp = out_apo / f"{s}.png"
            predict_mask(apo, apo_imgs[s], apo_size, apo_thr, device).save(outp)
            rec["apo"] = str(outp)
        if s in fasc_imgs:
            outp = out_fasc / f"{s}.png"
            predict_mask(fasc, fasc_imgs[s], fasc_size, fasc_thr, device).save(outp)
            rec["fasc"] = str(outp)
        index.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(stems):
            print(f"inferred {i+1}/{len(stems)}")
    (args.out_dir / "pred_masks_index.json").write_text(json.dumps({"n": len(index), "fasc_ckpt": str(args.fasc_ckpt), "apo_ckpt": str(args.apo_ckpt), "fasc_variant": fasc_ck.get("variant"), "fasc_val_dice": fasc_ck.get("val_dice"), "items": index}, indent=2))
    print(f"wrote {len(index)} under {args.out_dir}")


if __name__ == "__main__":
    main()
