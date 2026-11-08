#!/usr/bin/env python3
"""Fasc seg v3e: NO FOV crop (FOV drops ~25% fasc GT), high pos_weight, dilate, FascUNet.

Writes seg_fasc_v3e_best.pt (does NOT overwrite v2/v3c).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet_v2 import FascUNet  # noqa: E402


def load_fov_meta(scale_csv: Path) -> dict[str, dict]:
    out = {}
    with scale_csv.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("pool") != "fasc":
                continue
            try:
                if not r.get("fov_l") or not r.get("fov_r"):
                    continue
                meta = {
                    "box": (int(float(r["fov_l"])), int(float(r["fov_t"])), int(float(r["fov_r"])), int(float(r["fov_b"]))),
                    "ch": int(float(r["height"])),
                    "cw": int(float(r["width"])),
                }
            except (TypeError, ValueError, KeyError):
                continue
            out[str(r["stem"])] = meta
    return out


def scaled_fov(meta: dict | None, h: int, w: int) -> tuple[int, int, int, int] | None:
    """DISABLED for v3e: AmbrosM FOV drops ~25% of fasc GT pixels."""
    return None
    if meta is None:
        return None
    l, t, r, b = meta["box"]
    ch, cw = max(1, meta["ch"]), max(1, meta["cw"])
    # If CSV size matches actual, use as-is (with clamp)
    if abs(ch - h) <= 2 and abs(cw - w) <= 2:
        ls, ts, rs, bs = max(0, l), max(0, t), min(w, r), min(h, b)
    else:
        sx, sy = w / cw, h / ch
        ls, ts = int(round(l * sx)), int(round(t * sy))
        rs, bs = int(round(r * sx)), int(round(b * sy))
        ls, ts = max(0, ls), max(0, ts)
        rs, bs = min(w, rs), min(h, bs)
    if rs - ls < max(32, int(0.4 * w)) or bs - ts < max(32, int(0.4 * h)):
        return None  # FOV too small → skip
    # Require FOV not tiny relative to image; allow moderate chrome strip
    return ls, ts, rs, bs


def crop(arr: np.ndarray, fov: tuple[int, int, int, int] | None) -> np.ndarray:
    if fov is None:
        return arr
    l, t, r, b = fov
    return arr[t:b, l:r]


def dilate_mask(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_u8
    return np.asarray(Image.fromarray(mask_u8).filter(ImageFilter.MaxFilter(size=2 * radius + 1)))


class FascDataset(Dataset):
    def __init__(self, pairs, fov_meta, *, size=512, dilate=3, augment=False):
        self.pairs = pairs
        self.fov_meta = fov_meta
        self.size = int(size)
        self.dilate = int(dilate)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        mask = (np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        h, w = img.shape[:2]
        fov = scaled_fov(self.fov_meta.get(stem), h, w)
        img = crop(img, fov)
        mask = crop(mask, fov)
        soft = dilate_mask(mask * 255, self.dilate).astype(np.float32) / 255.0
        hard = mask.astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                soft = np.ascontiguousarray(soft[:, ::-1])
                hard = np.ascontiguousarray(hard[:, ::-1])
            if random.random() < 0.5:
                # mild brightness / contrast
                gain = 0.85 + 0.3 * random.random()
                bias = (random.random() - 0.5) * 0.1
                img = np.clip(img * gain + bias, 0, 1)
            if random.random() < 0.3:
                # small rotation ±7 deg via PIL
                ang = (random.random() - 0.5) * 14
                img_p = Image.fromarray((img * 255).astype(np.uint8)).rotate(ang, resample=Image.BILINEAR, fillcolor=0)
                soft_p = Image.fromarray((soft * 255).astype(np.uint8)).rotate(ang, resample=Image.NEAREST, fillcolor=0)
                hard_p = Image.fromarray((hard * 255).astype(np.uint8)).rotate(ang, resample=Image.NEAREST, fillcolor=0)
                img = np.asarray(img_p, dtype=np.float32) / 255.0
                soft = np.asarray(soft_p, dtype=np.float32) / 255.0
                hard = (np.asarray(hard_p) > 127).astype(np.float32)

        img_p = Image.fromarray((img * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        soft_p = Image.fromarray((soft * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        hard_p = Image.fromarray((hard * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img_p, dtype=np.float32) / 255.0).unsqueeze(0)
        soft_t = torch.from_numpy(np.asarray(soft_p, dtype=np.float32) / 255.0).unsqueeze(0)
        hard_t = torch.from_numpy((np.asarray(hard_p) > 127).astype(np.float32)).unsqueeze(0)
        return x, soft_t, hard_t


def loss_fn(logits, soft, hard, pos_weight: float):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)
    probs = torch.sigmoid(logits)
    # soft Dice on dilated target + hard Dice on thin GT (equal)
    inter_s = (probs * soft).sum(dim=(2, 3))
    den_s = probs.sum(dim=(2, 3)) + soft.sum(dim=(2, 3))
    dice_s = 1.0 - (2 * inter_s + 1) / (den_s + 1)
    inter_h = (probs * hard).sum(dim=(2, 3))
    den_h = probs.sum(dim=(2, 3)) + hard.sum(dim=(2, 3))
    dice_h = 1.0 - (2 * inter_h + 1) / (den_h + 1)
    return bce + dice_s.mean() + 0.5 * dice_h.mean()


@torch.no_grad()
def eval_dice(model, loader, device, thr: float) -> float:
    model.eval()
    dices = []
    for x, soft, hard in loader:
        x = x.to(device)
        hard = hard.to(device)
        pred = (torch.sigmoid(model(x)) > thr).float()
        inter = (pred * hard).sum(dim=(1, 2, 3))
        den = pred.sum(dim=(1, 2, 3)) + hard.sum(dim=(1, 2, 3))
        d = (2 * inter + 1e-6) / (den + 1e-6)
        dices.extend(d.cpu().tolist())
    return float(np.mean(dices)) if dices else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--scale-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}
    fov_meta = load_fov_meta(args.scale_csv)

    def pairs(split):
        out = []
        for r in rows:
            if r.get("task") == "fascicle" and r.get("split") == split:
                s = str(r["image_id"])
                if s in imgs and s in masks:
                    out.append((s, imgs[s], masks[s]))
        return out

    train_pairs, val_pairs = pairs("train"), pairs("val")
    n_crop = 0
    drop_scaled = []
    for stem, ip, mp in train_pairs[:200]:
        a = np.asarray(Image.open(ip).convert("L"))
        m = (np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        fov = scaled_fov(fov_meta.get(stem), a.shape[0], a.shape[1])
        if fov is not None:
            n_crop += 1
            c = crop(m, fov)
            drop_scaled.append(1.0 - c.sum() / max(1, m.sum()))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    drop_m = float(np.mean(drop_scaled)) if drop_scaled else -1.0
    print(
        f"fasc_v3e train={len(train_pairs)} val={len(val_pairs)} fov_meta={len(fov_meta)} "
        f"crop_in_sample200={n_crop} mean_gt_drop={drop_m:.4f} "
        f"pos_w={args.pos_weight} dilate={args.dilate} size={args.size} device={device}"
    )

    train_loader = DataLoader(
        FascDataset(train_pairs, fov_meta, size=args.size, dilate=args.dilate, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        FascDataset(val_pairs, fov_meta, size=args.size, dilate=0, augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    model = FascUNet(1, args.base_ch, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v3e_best.pt"
    best = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, soft, hard in train_loader:
            x, soft, hard = x.to(device), soft.to(device), hard.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, soft, hard, args.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        d35 = eval_dice(model, val_loader, device, 0.35)
        d50 = eval_dice(model, val_loader, device, 0.5)
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "dice35": d35, "dice50": d50})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} dice@0.35={d35:.4f} dice@0.5={d50:.4f}", flush=True)
        score = max(d35, d50)
        if score > best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": "fasc",
                    "arch": "FascUNet",
                    "size": args.size,
                    "base_ch": args.base_ch,
                    "val_dice": best,
                    "val_dice_thr035": d35,
                    "val_dice_thr050": d50,
                    "epoch": epoch,
                    "variant": "fasc_v3e_nofov_hipw",
                    "dilate": args.dilate,
                    "pos_weight": args.pos_weight,
                    "eval_thr_preferred": 0.35 if d35 >= d50 else 0.5,
                },
                ckpt,
            )
    meta = {
        "variant": "fasc_v3e_nofov_hipw",
        "best_val_dice": best,
        "history": hist,
        "fov_meta": len(fov_meta),
        "n_train": len(train_pairs),
        "checkpoint": str(ckpt),
        "pos_weight": args.pos_weight,
        "dilate": args.dilate,
    }
    (args.out_dir / "seg_fasc_v3e_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_dice={best:.4f} ckpt={ckpt}")


if __name__ == "__main__":
    main()
