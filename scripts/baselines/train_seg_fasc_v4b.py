#!/usr/bin/env python3
"""Fasc seg v4b: align mask→img, then square TinyUNet like v2 (no dilate/letterbox).

Isolates the size-mismatch fix that v2 lacked. Writes seg_fasc_v4b_best.pt.
Gate metric: native-res per-image macro Dice after align.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet import TinyUNet  # noqa: E402


def align_mask_to_img(hard: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if hard.shape == (h, w):
        return hard.astype(np.uint8)
    return (
        np.asarray(Image.fromarray((hard.astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)) > 127
    ).astype(np.uint8)


class PairDataset(Dataset):
    def __init__(self, pairs, *, size=384, augment=False):
        self.pairs = pairs
        self.size = int(size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard = (np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        hard = align_mask_to_img(hard, img.shape[:2])
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                hard = np.ascontiguousarray(hard[:, ::-1])
            if random.random() < 0.35:
                img = np.clip(img * random.uniform(0.8, 1.25), 0, 1)
        img_p = Image.fromarray((img * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        hard_p = Image.fromarray(hard * 255).resize((self.size, self.size), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img_p, dtype=np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy((np.asarray(hard_p) > 127).astype(np.float32)).unsqueeze(0)
        return x, y


def loss_fn(logits, targets, pos_weight: float):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    den = probs.sum() + targets.sum()
    dice = 1.0 - (2.0 * inter + 1e-6) / (den + 1e-6)
    return bce + dice


@torch.no_grad()
def eval_macro_square(model, loader, device, thr=0.5):
    model.eval()
    dices = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x)) > thr).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        den = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dices.extend(((2 * inter + 1e-6) / (den + 1e-6)).cpu().tolist())
    return float(np.mean(dices)) if dices else 0.0


@torch.no_grad()
def eval_native_macro(model, pairs, *, size, device, thr=0.5, max_n=None):
    model.eval()
    dices = []
    empty = 0
    for i, (stem, ip, mp) in enumerate(pairs):
        if max_n is not None and i >= max_n:
            break
        im = Image.open(ip).convert("L")
        w, h = im.size
        hard = (np.asarray(Image.open(mp).convert("L")) > 0)
        hard = align_mask_to_img(hard.astype(np.uint8), (h, w)).astype(bool)
        x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        t = torch.from_numpy(x)[None, None].to(device)
        prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
        prob = (
            np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32)
            / 255.0
        )
        pred = prob > thr
        if pred.sum() == 0:
            empty += 1
        inter = float((pred & hard).sum())
        den = float(pred.sum() + hard.sum())
        dices.append((2 * inter + 1e-6) / (den + 1e-6))
    a = np.asarray(dices, dtype=np.float64)
    return {
        "n": int(a.size),
        "macro_dice": float(a.mean()) if a.size else 0.0,
        "dice_p50": float(np.median(a)) if a.size else 0.0,
        "empty_pred_frac": float(empty / max(1, a.size)),
        "thr": thr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--pos-weight", type=float, default=40.0)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--device", type=str, default="cuda:3")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}

    def pairs(split):
        out = []
        for r in rows:
            if r.get("task") == "fascicle" and r.get("split") == split:
                s = str(r["image_id"])
                if s in imgs and s in masks:
                    out.append((s, imgs[s], masks[s]))
        return out

    train_pairs, val_pairs = pairs("train"), pairs("val")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"fasc_v4b train={len(train_pairs)} val={len(val_pairs)} size={args.size} "
        f"pos_w={args.pos_weight} device={device} align=mask_to_img square=yes dilate=no",
        flush=True,
    )

    train_loader = DataLoader(
        PairDataset(train_pairs, size=args.size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs, size=args.size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v4b_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v4b_last.pt"
    best_native = -1.0
    hist = []
    start_epoch = 1
    resume_path = args.resume if args.resume is not None else (last_ckpt if last_ckpt.exists() else None)
    if resume_path is not None and Path(resume_path).exists():
        rd = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rd["model"])
        for key, obj in (("optimizer", opt), ("scheduler", sched)):
            if key in rd:
                try:
                    obj.load_state_dict(rd[key])
                except Exception as e:
                    print(f"warn: {key} resume failed: {e}", flush=True)
        start_epoch = int(rd.get("epoch", 0)) + 1
        best_native = float(rd.get("val_dice", -1.0) or -1.0)
        hist = list(rd.get("history", []))
        print(f"resumed from {resume_path} epoch={start_epoch-1} best_native={best_native:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, args.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        d35 = eval_macro_square(model, val_loader, device, 0.35)
        d50 = eval_macro_square(model, val_loader, device, 0.5)
        n35 = eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.35)
        n50 = eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.5)
        native = max(n35["macro_dice"], n50["macro_dice"])
        prefer = 0.35 if n35["macro_dice"] >= n50["macro_dice"] else 0.5
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "sq035": d35,
            "sq050": d50,
            "native035": n35,
            "native050": n50,
        }
        hist.append(row)
        print(
            f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
            f"sq@0.35={d35:.4f} sq@0.5={d50:.4f} "
            f"native@0.35={n35['macro_dice']:.4f} native@0.5={n50['macro_dice']:.4f}",
            flush=True,
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "task": "fasc",
            "arch": "TinyUNet",
            "size": args.size,
            "base_ch": args.base_ch,
            "val_dice": max(best_native, native),
            "val_native_035": n35,
            "val_native_050": n50,
            "epoch": epoch,
            "variant": "fasc_v4b_align_square",
            "pos_weight": args.pos_weight,
            "eval_thr_preferred": prefer,
            "letterbox": False,
            "align_mask_to_img": True,
            "history": hist,
        }
        torch.save(payload, last_ckpt)
        if native > best_native:
            best_native = native
            payload["val_dice"] = best_native
            torch.save(payload, ckpt)

    meta = {
        "variant": "fasc_v4b_align_square",
        "best_val_native_macro_dice": best_native,
        "history": hist,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "checkpoint": str(ckpt),
    }
    (args.out_dir / "seg_fasc_v4b_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_native_macro_dice={best_native:.4f} ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
