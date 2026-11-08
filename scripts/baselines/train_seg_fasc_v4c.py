#!/usr/bin/env python3
"""Fasc v4c: align mask→img, square TinyUNet, strong soft dilate (r=5), lower pos_w.

Hypothesis: hard 2px labels are the ceiling; train on thick locus, eval on hard native.
Also reports soft-tolerance Dice (pred vs dilate-r GT) as locus quality.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from umud.baselines.io import load_split_assignments
from umud.data.layout import discover_layout, list_images
from umud.seg.unet import TinyUNet


def align_mask_to_img(hard: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if hard.shape == (h, w):
        return hard.astype(np.uint8)
    return (np.asarray(Image.fromarray(hard.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127).astype(np.uint8)


def dilate_u8(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_u8
    return np.asarray(Image.fromarray(mask_u8).filter(ImageFilter.MaxFilter(size=2 * radius + 1)))


class DS(Dataset):
    def __init__(self, pairs, *, size=512, dilate=5, augment=False):
        self.pairs = pairs
        self.size = int(size)
        self.dilate = int(dilate)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard = align_mask_to_img((np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8), img.shape[:2])
        soft = dilate_u8(hard * 255, self.dilate).astype(np.float32) / 255.0
        hard_f = hard.astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1]); soft = np.ascontiguousarray(soft[:, ::-1]); hard_f = np.ascontiguousarray(hard_f[:, ::-1])
            if random.random() < 0.4:
                img = np.clip(img * random.uniform(0.8, 1.25) + random.uniform(-0.05, 0.05), 0, 1)
        img_p = Image.fromarray((img * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        soft_p = Image.fromarray((soft * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        hard_p = Image.fromarray((hard_f * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        return (
            torch.from_numpy(np.asarray(img_p, dtype=np.float32) / 255.0).unsqueeze(0),
            torch.from_numpy(np.asarray(soft_p, dtype=np.float32) / 255.0).unsqueeze(0),
            torch.from_numpy((np.asarray(hard_p) > 127).astype(np.float32)).unsqueeze(0),
        )


def loss_fn(logits, soft, hard, pos_weight: float):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)
    probs = torch.sigmoid(logits)
    inter = (probs * soft).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + soft.sum(dim=(2, 3))
    dice_s = 1.0 - (2 * inter + 1) / (den + 1)
    # light hard term so center stays sharp
    inter_h = (probs * hard).sum(dim=(2, 3))
    den_h = probs.sum(dim=(2, 3)) + hard.sum(dim=(2, 3))
    dice_h = 1.0 - (2 * inter_h + 1) / (den_h + 1)
    return bce + dice_s.mean() + 0.25 * dice_h.mean()


@torch.no_grad()
def eval_native(model, pairs, *, size, device, thr, soft_r=0):
    model.eval()
    hard_d, soft_d = [], []
    empty = 0
    for stem, ip, mp in pairs:
        im = Image.open(ip).convert("L"); w, h = im.size
        hard = align_mask_to_img((np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8), (h, w)).astype(bool)
        soft = dilate_u8(hard.astype(np.uint8) * 255, soft_r) > 127 if soft_r else hard
        x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        prob = torch.sigmoid(model(torch.from_numpy(x)[None, None].to(device)))[0, 0].cpu().numpy()
        prob = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        pred = prob > thr
        if pred.sum() == 0:
            empty += 1
        inter = float((pred & hard).sum()); den = float(pred.sum() + hard.sum())
        hard_d.append((2 * inter + 1e-6) / (den + 1e-6))
        inter_s = float((pred & soft).sum()); den_s = float(pred.sum() + soft.sum())
        soft_d.append((2 * inter_s + 1e-6) / (den_s + 1e-6))
    return {
        "hard_macro": float(np.mean(hard_d)),
        "soft_tol_macro": float(np.mean(soft_d)),
        "empty_frac": float(empty / max(1, len(hard_d))),
        "thr": thr,
        "soft_r": soft_r,
        "n": len(hard_d),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--dilate", type=int, default=5)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

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
    print(f"fasc_v4c train={len(train_pairs)} val={len(val_pairs)} size={args.size} dilate={args.dilate} pos_w={args.pos_weight} device={device}", flush=True)

    train_loader = DataLoader(DS(train_pairs, size=args.size, dilate=args.dilate, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    model = TinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v4c_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v4c_last.pt"
    best = -1.0
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
        best = float(rd.get("val_dice", -1.0) or -1.0)
        hist = list(rd.get("history", []))
        print(f"resumed from {resume_path} epoch={start_epoch-1} best={best:.4f}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); losses = []
        for x, soft, hard in train_loader:
            x, soft, hard = x.to(device), soft.to(device), hard.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), soft, hard, args.pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.item()))
        sched.step()
        n35 = eval_native(model, val_pairs, size=args.size, device=device, thr=0.35, soft_r=args.dilate)
        n50 = eval_native(model, val_pairs, size=args.size, device=device, thr=0.5, soft_r=args.dilate)
        # gate on hard dice; also track soft tolerance
        score = max(n35["hard_macro"], n50["hard_macro"])
        prefer = 0.35 if n35["hard_macro"] >= n50["hard_macro"] else 0.5
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "n35": n35, "n50": n50}
        hist.append(row)
        print(f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} hard@0.35={n35['hard_macro']:.4f} hard@0.5={n50['hard_macro']:.4f} softTol@0.35={n35['soft_tol_macro']:.4f} softTol@0.5={n50['soft_tol_macro']:.4f}", flush=True)
        payload = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "task": "fasc", "arch": "TinyUNet", "size": args.size, "base_ch": args.base_ch, "val_dice": max(best, score), "val_native_035": n35, "val_native_050": n50, "epoch": epoch, "variant": "fasc_v4c_thick_soft", "dilate": args.dilate, "pos_weight": args.pos_weight, "eval_thr_preferred": prefer, "letterbox": False, "align_mask_to_img": True, "history": hist}
        torch.save(payload, last_ckpt)
        if score > best:
            best = score
            payload["val_dice"] = best
            torch.save(payload, ckpt)
    meta = {"variant": "fasc_v4c_thick_soft", "best_val_hard_macro_dice": best, "history": hist, "checkpoint": str(ckpt)}
    (args.out_dir / "seg_fasc_v4c_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_hard_macro={best:.4f} ckpt={ckpt}", flush=True)

if __name__ == "__main__":
    main()
