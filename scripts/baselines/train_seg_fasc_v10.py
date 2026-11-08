#!/usr/bin/env python3
"""Fasc v10: MSTinyUNet + Focal-Tversky + high clDice + fasc-centered multi-scale crops.

Distinct lever vs v5–v9 (loss + crop policy). Warm-start from v5_best when possible.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from umud.baselines.io import load_split_assignments
from umud.data.layout import discover_layout, list_images
from umud.seg.ms_unet import MSTinyUNet

import importlib.util
spec = importlib.util.spec_from_file_location("v7", ROOT / "scripts/baselines/train_seg_fasc_v7.py")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)


def focal_tversky(probs, hard, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-6):
    # probs/hard: B1HW
    p = probs.reshape(probs.shape[0], -1)
    t = hard.reshape(hard.shape[0], -1)
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1.0 - tv) ** gamma).mean()


def loss_fn(logits, soft, hard, skel, *, pos_weight=45.0, cldice_w=1.5, ft_w=1.0, bce_w=0.35):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits, soft, pos_weight=torch.tensor([pos_weight], device=logits.device)
    )
    ft = focal_tversky(probs, hard)
    cl = v7.soft_cldice(probs, soft, skel)
    return bce_w * bce + ft_w * ft + cldice_w * cl


class FascCropDS(Dataset):
    """Square resize + optional fasc-centered crop zoom for thin structure focus."""

    def __init__(self, pairs, *, size=576, edt_radius=3.5, augment=False, crop_zoom_p=0.55):
        self.pairs = pairs
        self.size = size
        self.edt_radius = edt_radius
        self.augment = augment
        self.crop_zoom_p = crop_zoom_p

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard0 = (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.uint8)
        hard0 = v7.align_mask_to_img(hard0, img.shape[:2])

        # fasc-centered zoom crop before final square resize
        if self.augment and hard0.max() > 0 and random.random() < self.crop_zoom_p:
            ys, xs = np.where(hard0 > 0)
            cy, cx = int(ys.mean()), int(xs.mean())
            H, W = img.shape[:2]
            # zoom factor 1.3–2.0 → crop smaller window then resize up
            z = random.uniform(1.3, 2.0)
            ch, cw = max(32, int(H / z)), max(32, int(W / z))
            y0 = max(0, min(H - ch, cy - ch // 2 + random.randint(-ch // 8, ch // 8)))
            x0 = max(0, min(W - cw, cx - cw // 2 + random.randint(-cw // 8, cw // 8)))
            img = img[y0 : y0 + ch, x0 : x0 + cw]
            hard0 = hard0[y0 : y0 + ch, x0 : x0 + cw]

        img_r = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255.0
        hard = np.asarray(Image.fromarray(hard0 * 255).resize((self.size, self.size), Image.NEAREST)) > 127
        hard = hard.astype(np.uint8)
        if self.augment:
            if random.random() < 0.5:
                img_r = np.fliplr(img_r).copy(); hard = np.fliplr(hard).copy()
            if random.random() < 0.3:
                img_r = np.flipud(img_r).copy(); hard = np.flipud(hard).copy()
            # mild intensity jitte
            if random.random() < 0.5:
                img_r = np.clip(img_r * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05), 0, 1)
        soft = v7.edt_soft(hard, self.edt_radius)
        skel = v7.gt_skeleton(hard)
        x = torch.from_numpy(img_r[None].astype(np.float32))
        return x, torch.from_numpy(soft[None]), torch.from_numpy(hard[None].astype(np.float32)), torch.from_numpy(skel[None])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--size", type=int, default=576)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--edt-radius", type=float, default=3.5)
    ap.add_argument("--pos-weight", type=float, default=45.0)
    ap.add_argument("--cldice-w", type=float, default=1.5)
    ap.add_argument("--ft-w", type=float, default=1.0)
    ap.add_argument("--bce-w", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    use_amp = bool(args.amp) and not bool(args.no_amp)

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
    print(
        f"fasc_v10 train={len(train_pairs)} val={len(val_pairs)} size={args.size} base={args.base_ch} "
        f"cldice_w={args.cldice_w} ft_w={args.ft_w} bce_w={args.bce_w} amp={use_amp} bs={args.batch_size} device={device}",
        flush=True,
    )

    loader = DataLoader(
        FascCropDS(train_pairs, size=args.size, edt_radius=args.edt_radius, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    model = MSTinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v10_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v10_last.pt"
    best_native = -1.0
    hist = []
    start_epoch = 1

    if args.init_from and Path(args.init_from).exists() and args.resume is None and not last_ckpt.exists():
        rd = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(rd["model"], strict=False)
        print(f"warm-start {args.init_from} miss={len(missing)} unexp={len(unexpected)} src={rd.get('val_dice')}", flush=True)

    resume_path = args.resume if args.resume is not None else (last_ckpt if last_ckpt.exists() else None)
    if resume_path is not None and Path(resume_path).exists():
        rd = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rd["model"])
        for key, obj in (("optimizer", opt), ("scheduler", sched)):
            if key in rd:
                try: obj.load_state_dict(rd[key])
                except Exception as e: print(f"warn {key}: {e}", flush=True)
        if "scaler" in rd and rd["scaler"] is not None and use_amp:
            try: scaler.load_state_dict(rd["scaler"])
            except Exception as e: print(f"warn scaler: {e}", flush=True)
        start_epoch = int(rd.get("epoch", 0)) + 1
        best_native = float(rd.get("val_dice", -1) or -1)
        hist = list(rd.get("history", []))
        print(f"resumed {resume_path} ep={start_epoch-1} best={best_native:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); losses = []
        for x, soft, hard, skel in loader:
            x = x.to(device, non_blocking=True); soft = soft.to(device, non_blocking=True)
            hard = hard.to(device, non_blocking=True); skel = skel.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                if logits.shape[-2:] != soft.shape[-2:]:
                    logits = F.interpolate(logits, size=soft.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(
                    logits, soft, hard, skel,
                    pos_weight=args.pos_weight, cldice_w=args.cldice_w, ft_w=args.ft_w, bce_w=args.bce_w,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            losses.append(float(loss.item()))
        sched.step()

        n35 = v7.eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.35)
        n50 = v7.eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.5)
        native = max(n35["macro_dice"], n50["macro_dice"])
        prefer = 0.35 if n35["macro_dice"] >= n50["macro_dice"] else 0.5
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "native035": n35, "native050": n50}
        hist.append(row)
        print(f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} native@0.35={n35['macro_dice']:.4f} native@0.5={n50['macro_dice']:.4f}", flush=True)

        payload = {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "task": "fasc", "arch": "MSTinyUNet", "size": args.size, "base_ch": args.base_ch,
            "val_dice": max(best_native, native),
            "val_native_035": n35, "val_native_050": n50, "epoch": epoch,
            "variant": "fasc_v10_focal_tversky_cropzoom",
            "pos_weight": args.pos_weight, "edt_radius": args.edt_radius,
            "cldice_w": args.cldice_w, "ft_w": args.ft_w, "bce_w": args.bce_w,
            "amp": use_amp, "batch_size": args.batch_size, "eval_thr_preferred": prefer,
            "letterbox": False, "align_mask_to_img": True, "crop_zoom": True, "history": hist,
        }
        torch.save(payload, last_ckpt)
        if native > best_native:
            best_native = native
            payload["val_dice"] = best_native
            torch.save(payload, ckpt)
            print(f"  saved best_native={best_native:.4f}", flush=True)

    meta = {
        "variant": "fasc_v10_focal_tversky_cropzoom",
        "best_val_native_macro_dice": best_native,
        "history": hist, "n_train": len(train_pairs), "n_val": len(val_pairs),
        "checkpoint": str(ckpt), "amp": use_amp, "batch_size": args.batch_size,
    }
    (args.out_dir / "seg_fasc_v10_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_native_macro_dice={best_native:.4f} ckpt={ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
