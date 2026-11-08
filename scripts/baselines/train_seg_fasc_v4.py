#!/usr/bin/env python3
"""Fasc seg v4: TinyUNet + letterbox (aspect) + soft dilate + native-res macro Dice.

Diagnosis (phase3/diag_fasc_seg.json):
  fg≈0.25%, thickness≈2px → square resize destroys diagonals; FascUNet/pos_w≥80
  collapsed. Keep TinyUNet + pos_w≈40–60; eval = per-image macro Dice at native HxW.
Does NOT overwrite seg_fasc_best.pt (writes seg_fasc_v4_best.pt).
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
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.unet import TinyUNet  # noqa: E402


def dilate_u8(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_u8
    return np.asarray(Image.fromarray(mask_u8).filter(ImageFilter.MaxFilter(size=2 * radius + 1)))


def letterbox(arr: np.ndarray, size: int, *, nearest: bool) -> tuple[np.ndarray, dict]:
    """Resize preserving aspect into size×size canvas; return array + meta for unletterbox."""
    h, w = arr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resample = Image.NEAREST if nearest else Image.BILINEAR
    if arr.dtype != np.uint8:
        # expect float 0..1 or 0..255
        if arr.max() <= 1.0:
            src = (arr * 255.0).astype(np.uint8)
        else:
            src = arr.astype(np.uint8)
    else:
        src = arr
    resized = Image.fromarray(src).resize((nw, nh), resample)
    canvas = Image.new("L", (size, size), 0)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas.paste(resized, (left, top))
    out = np.asarray(canvas)
    meta = {"h": h, "w": w, "nh": nh, "nw": nw, "top": top, "left": left, "size": size}
    return out, meta


def unletterbox_prob(prob_sq: np.ndarray, meta: dict) -> np.ndarray:
    """Map size×size probability back to native HxW."""
    size = meta["size"]
    top, left, nh, nw = meta["top"], meta["left"], meta["nh"], meta["nw"]
    crop = prob_sq[top : top + nh, left : left + nw]
    return np.asarray(
        Image.fromarray((crop * 255).astype(np.uint8)).resize((meta["w"], meta["h"]), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


class FascLetterboxDS(Dataset):
    def __init__(self, pairs, *, size=512, dilate=2, augment=False):
        self.pairs = pairs
        self.size = int(size)
        self.dilate = int(dilate)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard = (np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        # Critical: ~60% fasc pairs have different HxW (HUD screenshot vs annotation).
        # Alignment score audit: resize_to_img beats letterbox/center-crop (fg-bg +11.8).
        if hard.shape != img.shape:
            hard = (
                np.asarray(
                    Image.fromarray(hard * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)
                )
                > 127
            ).astype(np.uint8)
        soft = dilate_u8(hard * 255, self.dilate).astype(np.float32) / 255.0
        hard_f = hard.astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                soft = np.ascontiguousarray(soft[:, ::-1])
                hard_f = np.ascontiguousarray(hard_f[:, ::-1])
            if random.random() < 0.45:
                img = np.clip(img * random.uniform(0.8, 1.25) + random.uniform(-0.05, 0.05), 0, 1)
            if random.random() < 0.25:
                ang = random.uniform(-8, 8)
                img = np.asarray(
                    Image.fromarray((img * 255).astype(np.uint8)).rotate(ang, Image.BILINEAR, fillcolor=0),
                    dtype=np.float32,
                ) / 255.0
                soft = np.asarray(
                    Image.fromarray((soft * 255).astype(np.uint8)).rotate(ang, Image.NEAREST, fillcolor=0),
                    dtype=np.float32,
                ) / 255.0
                hard_f = (
                    np.asarray(
                        Image.fromarray((hard_f * 255).astype(np.uint8)).rotate(ang, Image.NEAREST, fillcolor=0)
                    )
                    > 127
                ).astype(np.float32)

        img_sq, meta = letterbox(img, self.size, nearest=False)
        soft_sq, _ = letterbox(soft, self.size, nearest=True)
        hard_sq, _ = letterbox(hard_f, self.size, nearest=True)
        x = torch.from_numpy(img_sq.astype(np.float32) / 255.0).unsqueeze(0)
        soft_t = torch.from_numpy(soft_sq.astype(np.float32) / 255.0).unsqueeze(0)
        hard_t = torch.from_numpy((hard_sq > 127).astype(np.float32)).unsqueeze(0)
        # stash native hard for native eval via paths — return stem index only in train
        return x, soft_t, hard_t, stem


def collate(batch):
    xs, softs, hards, stems = zip(*batch)
    return torch.stack(xs), torch.stack(softs), torch.stack(hards), list(stems)


def loss_fn(logits, soft, hard, pos_weight: float):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)
    probs = torch.sigmoid(logits)
    inter_s = (probs * soft).sum(dim=(2, 3))
    den_s = probs.sum(dim=(2, 3)) + soft.sum(dim=(2, 3))
    dice_s = 1.0 - (2 * inter_s + 1) / (den_s + 1)
    inter_h = (probs * hard).sum(dim=(2, 3))
    den_h = probs.sum(dim=(2, 3)) + hard.sum(dim=(2, 3))
    dice_h = 1.0 - (2 * inter_h + 1) / (den_h + 1)
    return bce + dice_s.mean() + 0.75 * dice_h.mean()


@torch.no_grad()
def eval_macro_dice_letterbox(model, loader, device, thr: float) -> float:
    """Per-image macro Dice on letterboxed hard masks (fast proxy)."""
    model.eval()
    dices = []
    for batch in loader:
        x, soft, hard = batch[0], batch[1], batch[2]
        x, hard = x.to(device), hard.to(device)
        pred = (torch.sigmoid(model(x)) > thr).float()
        inter = (pred * hard).sum(dim=(1, 2, 3))
        den = pred.sum(dim=(1, 2, 3)) + hard.sum(dim=(1, 2, 3))
        d = (2 * inter + 1e-6) / (den + 1e-6)
        dices.extend(d.cpu().tolist())
    return float(np.mean(dices)) if dices else 0.0


@torch.no_grad()
def eval_native_macro_dice(model, pairs, *, size: int, device, thr: float, max_n: int | None = None) -> dict:
    """Per-image macro Dice after unletterbox to native resolution (gate metric)."""
    model.eval()
    dices = []
    empty = 0
    for i, (stem, ip, mp) in enumerate(pairs):
        if max_n is not None and i >= max_n:
            break
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard = (np.asarray(Image.open(mp).convert("L")) > 0)
        if hard.shape != img.shape:
            hard = (
                np.asarray(
                    Image.fromarray(hard.astype(np.uint8) * 255).resize(
                        (img.shape[1], img.shape[0]), Image.NEAREST
                    )
                )
                > 127
            )
        img_sq, meta = letterbox(img, size, nearest=False)
        x = torch.from_numpy(img_sq.astype(np.float32) / 255.0)[None, None].to(device)
        prob_sq = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        prob = unletterbox_prob(prob_sq, meta)
        pred = prob > thr
        if pred.sum() == 0:
            empty += 1
        inter = float((pred & hard).sum())
        den = float(pred.sum() + hard.sum())
        dices.append((2 * inter + 1e-6) / (den + 1e-6))
    arr = np.asarray(dices, dtype=np.float64)
    return {
        "n": int(arr.size),
        "macro_dice": float(arr.mean()) if arr.size else 0.0,
        "dice_p50": float(np.median(arr)) if arr.size else 0.0,
        "empty_pred_frac": float(empty / max(1, arr.size)),
        "thr": thr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--pos-weight", type=float, default=45.0)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native-eval-every", type=int, default=5)
    ap.add_argument("--resume", type=Path, default=None, help="Resume from last/best ckpt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}

    def pairs(split: str):
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
        f"fasc_v4 train={len(train_pairs)} val={len(val_pairs)} size={args.size} "
        f"dilate={args.dilate} pos_w={args.pos_weight} device={device}",
        flush=True,
    )

    train_loader = DataLoader(
        FascLetterboxDS(train_pairs, size=args.size, dilate=args.dilate, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        FascLetterboxDS(val_pairs, size=args.size, dilate=0, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    model = TinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v4_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v4_last.pt"
    best = -1.0
    best_native = -1.0
    hist = []
    start_epoch = 1
    resume_path = args.resume
    if resume_path is None and last_ckpt.exists():
        resume_path = last_ckpt
    if resume_path is not None and Path(resume_path).exists():
        rd = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rd["model"])
        if "optimizer" in rd:
            try:
                opt.load_state_dict(rd["optimizer"])
            except Exception as e:
                print(f"warn: optimizer resume failed: {e}", flush=True)
        if "scheduler" in rd:
            try:
                sched.load_state_dict(rd["scheduler"])
            except Exception as e:
                print(f"warn: scheduler resume failed: {e}", flush=True)
        start_epoch = int(rd.get("epoch", 0)) + 1
        best_native = float(rd.get("val_dice", -1.0) or -1.0)
        best = float(rd.get("val_dice_letterbox_proxy", best_native) or best_native)
        hist = list(rd.get("history", []))
        print(f"resumed from {resume_path} epoch={start_epoch-1} best_native={best_native:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for x, soft, hard, _stems in train_loader:
            x, soft, hard = x.to(device), soft.to(device), hard.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, soft, hard, args.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        d35 = eval_macro_dice_letterbox(model, val_loader, device, 0.35)
        d50 = eval_macro_dice_letterbox(model, val_loader, device, 0.5)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "macro_dice_lb_035": d35,
            "macro_dice_lb_050": d50,
        }
        score = max(d35, d50)
        do_native = (epoch % args.native_eval_every == 0) or (epoch == args.epochs) or (score > best)
        if do_native:
            n35 = eval_native_macro_dice(model, val_pairs, size=args.size, device=device, thr=0.35)
            n50 = eval_native_macro_dice(model, val_pairs, size=args.size, device=device, thr=0.5)
            row["native_035"] = n35
            row["native_050"] = n50
            native_score = max(n35["macro_dice"], n50["macro_dice"])
            prefer_thr = 0.35 if n35["macro_dice"] >= n50["macro_dice"] else 0.5
            print(
                f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
                f"lb@0.35={d35:.4f} lb@0.5={d50:.4f} "
                f"native@0.35={n35['macro_dice']:.4f} native@0.5={n50['macro_dice']:.4f}",
                flush=True,
            )
            if native_score > best_native:
                best_native = native_score
                best = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "task": "fasc",
                        "arch": "TinyUNet",
                        "size": args.size,
                        "base_ch": args.base_ch,
                        "val_dice": best_native,
                        "val_dice_letterbox_proxy": score,
                        "val_native_035": n35,
                        "val_native_050": n50,
                        "epoch": epoch,
                        "variant": "fasc_v4_letterbox_dilate",
                        "dilate": args.dilate,
                        "pos_weight": args.pos_weight,
                        "eval_thr_preferred": prefer_thr,
                        "letterbox": True,
                    },
                    ckpt,
                )
        else:
            print(
                f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} lb@0.35={d35:.4f} lb@0.5={d50:.4f}",
                flush=True,
            )
        hist.append(row)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "task": "fasc",
                "arch": "TinyUNet",
                "size": args.size,
                "base_ch": args.base_ch,
                "val_dice": best_native,
                "val_dice_letterbox_proxy": best,
                "epoch": epoch,
                "variant": "fasc_v4_letterbox_dilate",
                "dilate": args.dilate,
                "pos_weight": args.pos_weight,
                "letterbox": True,
                "history": hist,
            },
            last_ckpt,
        )

    meta = {
        "variant": "fasc_v4_letterbox_dilate",
        "best_val_native_macro_dice": best_native,
        "best_val_letterbox_macro_dice": best,
        "history": hist,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "checkpoint": str(ckpt),
        "pos_weight": args.pos_weight,
        "dilate": args.dilate,
        "size": args.size,
        "note": "Gate metric = native-res per-image macro Dice; NOT LB",
    }
    (args.out_dir / "seg_fasc_v4_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_native_macro_dice={best_native:.4f} ckpt={ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
