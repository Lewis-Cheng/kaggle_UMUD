#!/usr/bin/env python3
"""Fasc v7: AMP fp16 + stronger topology (clDice/EDT) + larger-batch MSTinyUNet on free GPU.

Nearest mask align (no letterbox/thick-soft). Gate: native-res macro Dice.
Warm-start from v5_best when present; auto-resume from seg_fasc_v7_last.pt.
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
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.seg.ms_unet import MSTinyUNet  # noqa: E402


def align_mask_to_img(hard: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if hard.shape == (h, w):
        return hard.astype(np.uint8)
    return (
        np.asarray(Image.fromarray(hard.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    ).astype(np.uint8)


def edt_soft(hard: np.ndarray, radius: float = 3.5) -> np.ndarray:
    if hard.max() == 0:
        return hard.astype(np.float32)
    dist = ndimage.distance_transform_edt(hard.astype(bool))
    return np.clip(dist / float(radius), 0.0, 1.0).astype(np.float32)


def gt_skeleton(hard: np.ndarray) -> np.ndarray:
    if hard.max() == 0:
        return hard.astype(np.float32)
    return skeletonize(hard.astype(bool)).astype(np.float32)


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_skel(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img = x
    skel = F.relu(img - soft_erode(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_erode(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(probs: torch.Tensor, soft_gt: torch.Tensor, skel_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    skel_p = soft_skel(probs)
    tprec = (skel_p * soft_gt).sum(dim=(2, 3)) / (skel_p.sum(dim=(2, 3)) + eps)
    tsens = (skel_gt * probs).sum(dim=(2, 3)) / (skel_gt.sum(dim=(2, 3)) + eps)
    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return 1.0 - cl.mean()


class FascEDTDS(Dataset):
    def __init__(self, pairs, *, size=576, edt_radius=3.5, augment=False):
        self.pairs = pairs
        self.size = size
        self.edt_radius = edt_radius
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard = align_mask_to_img((np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8), img.shape[:2])
        soft = edt_soft(hard, self.edt_radius)
        skel = gt_skeleton(hard)
        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy(); soft = np.fliplr(soft).copy(); hard = np.fliplr(hard).copy(); skel = np.fliplr(skel).copy()
            if random.random() < 0.3:
                ang = random.uniform(-12, 12)
                img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).rotate(ang, Image.BILINEAR, fillcolor=0), dtype=np.float32) / 255.0
                soft = np.asarray(Image.fromarray((soft * 255).astype(np.uint8)).rotate(ang, Image.BILINEAR, fillcolor=0), dtype=np.float32) / 255.0
                hard = (np.asarray(Image.fromarray(hard * 255).rotate(ang, Image.NEAREST, fillcolor=0)) > 127).astype(np.uint8)
                skel = (np.asarray(Image.fromarray((skel * 255).astype(np.uint8)).rotate(ang, Image.NEAREST, fillcolor=0)) > 127).astype(np.float32)
        img_p = Image.fromarray((img * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        soft_p = Image.fromarray((soft * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        hard_p = Image.fromarray(hard * 255).resize((self.size, self.size), Image.NEAREST)
        skel_p = Image.fromarray((skel * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img_p, dtype=np.float32) / 255.0)[None]
        soft_t = torch.from_numpy(np.asarray(soft_p, dtype=np.float32) / 255.0)[None]
        hard_t = torch.from_numpy((np.asarray(hard_p) > 127).astype(np.float32))[None]
        skel_t = torch.from_numpy((np.asarray(skel_p) > 127).astype(np.float32))[None]
        return x, soft_t, hard_t, skel_t


def loss_fn(logits, soft, hard, skel, *, pos_weight, cldice_w, aux=None):
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=torch.tensor([pos_weight], device=logits.device))
    probs = torch.sigmoid(logits)
    inter = (probs * hard).sum(dim=(2, 3))
    dice = 1.0 - (2 * inter + 1) / (probs.sum(dim=(2, 3)) + hard.sum(dim=(2, 3)) + 1)
    dice = dice.mean()
    cld = soft_cldice(probs, soft, skel)
    loss = bce + dice + cldice_w * cld
    if aux:
        for a_logits, w in aux:
            a_probs = torch.sigmoid(a_logits)
            if a_probs.shape[-2:] != soft.shape[-2:]:
                a_probs = F.interpolate(a_probs, size=soft.shape[-2:], mode="bilinear", align_corners=False)
                a_logits = F.interpolate(a_logits, size=soft.shape[-2:], mode="bilinear", align_corners=False)
            a_bce = F.binary_cross_entropy_with_logits(a_logits, soft, pos_weight=torch.tensor([pos_weight], device=logits.device))
            a_inter = (a_probs * hard).sum(dim=(2, 3))
            a_dice = 1.0 - (2 * a_inter + 1) / (a_probs.sum(dim=(2, 3)) + hard.sum(dim=(2, 3)) + 1)
            loss = loss + w * (a_bce + a_dice.mean())
    return loss


@torch.no_grad()
def eval_native_macro(model, pairs, *, size, device, thr=0.5):
    model.eval()
    dices = []
    empty = 0
    for _, ip, mp in pairs:
        im = Image.open(ip).convert("L")
        w, h = im.size
        hard = align_mask_to_img((np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8), (h, w)).astype(bool)
        x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        t = torch.from_numpy(x)[None, None].to(device)
        out = model(t)
        logits = out[0] if isinstance(out, tuple) else out
        if logits.shape[-2:] != (size, size):
            logits = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        prob = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        pred = prob >= thr
        if not pred.any():
            empty += 1
        inter = float((pred & hard).sum())
        denom = float(pred.sum() + hard.sum())
        dices.append((2 * inter / denom) if denom > 0 else (1.0 if not hard.any() else 0.0))
    arr = np.asarray(dices, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "macro_dice": float(arr.mean()) if len(arr) else 0.0,
        "dice_p50": float(np.median(arr)) if len(arr) else 0.0,
        "empty_pred_frac": float(empty / max(1, len(arr))),
        "thr": float(thr),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--size", type=int, default=576)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--edt-radius", type=float, default=3.5)
    ap.add_argument("--pos-weight", type=float, default=45.0)
    ap.add_argument("--cldice-w", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--native-every", type=int, default=1)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    use_amp = bool(args.amp) and not bool(args.no_amp)

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
        f"fasc_v7 train={len(train_pairs)} val={len(val_pairs)} size={args.size} "
        f"base={args.base_ch} edt_r={args.edt_radius} pos_w={args.pos_weight} "
        f"cldice_w={args.cldice_w} amp={use_amp} bs={args.batch_size} device={device}",
        flush=True,
    )

    train_loader = DataLoader(
        FascEDTDS(train_pairs, size=args.size, edt_radius=args.edt_radius, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    model = MSTinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v7_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v7_last.pt"
    best_native = -1.0
    hist = []
    start_epoch = 1

    if args.init_from and Path(args.init_from).exists() and args.resume is None and not last_ckpt.exists():
        rd = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(rd["model"], strict=False)
        print(
            f"warm-start {args.init_from} missing={len(missing)} unexpected={len(unexpected)} "
            f"src_dice={rd.get('val_dice')}",
            flush=True,
        )

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
        if "scaler" in rd and use_amp:
            try:
                scaler.load_state_dict(rd["scaler"])
            except Exception as e:
                print(f"warn: scaler resume failed: {e}", flush=True)
        start_epoch = int(rd.get("epoch", 0)) + 1
        best_native = float(rd.get("val_dice", -1.0) or -1.0)
        hist = list(rd.get("history", []))
        print(f"resumed {resume_path} ep={start_epoch-1} best={best_native:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for x, soft, hard, skel in train_loader:
            x, soft, hard, skel = x.to(device, non_blocking=True), soft.to(device, non_blocking=True), hard.to(device, non_blocking=True), skel.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits, a3, a2 = model(x, return_aux=True)
                loss = loss_fn(
                    logits, soft, hard, skel,
                    pos_weight=args.pos_weight,
                    cldice_w=args.cldice_w,
                    aux=[(a3, 0.3), (a2, 0.2)],
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        sched.step()

        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        if epoch % args.native_every == 0 or epoch == args.epochs:
            n35 = eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.35)
            n50 = eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.5)
            row["native035"] = n35
            row["native050"] = n50
            native = max(n35["macro_dice"], n50["macro_dice"])
            prefer = 0.35 if n35["macro_dice"] >= n50["macro_dice"] else 0.5
            print(
                f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} "
                f"native@0.35={n35['macro_dice']:.4f} native@0.5={n50['macro_dice']:.4f}",
                flush=True,
            )
        else:
            native = best_native
            prefer = 0.5
            n35 = n50 = None
            print(f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f}", flush=True)

        hist.append(row)
        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "task": "fasc",
            "arch": "MSTinyUNet",
            "size": args.size,
            "base_ch": args.base_ch,
            "val_dice": max(best_native, native),
            "val_native_035": n35,
            "val_native_050": n50,
            "epoch": epoch,
            "variant": "fasc_v7_amp_topo_ms",
            "pos_weight": args.pos_weight,
            "edt_radius": args.edt_radius,
            "cldice_w": args.cldice_w,
            "amp": use_amp,
            "batch_size": args.batch_size,
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
            print(f"  saved best_native={best_native:.4f}", flush=True)

    meta = {
        "variant": "fasc_v7_amp_topo_ms",
        "best_val_native_macro_dice": best_native,
        "history": hist,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "checkpoint": str(ckpt),
        "amp": use_amp,
        "batch_size": args.batch_size,
    }
    (args.out_dir / "seg_fasc_v7_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_native_macro_dice={best_native:.4f} ckpt={ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
