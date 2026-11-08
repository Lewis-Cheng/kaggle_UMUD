#!/usr/bin/env python3
"""Fasc v12: v5 structure + aux1@dec1 (w=0.1). Block fasc_thin_ceiling.

Frozen v5 knobs; only n_aux=3 + aux1 weight 0.1. Gate: beat v5 native 0.2846.
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


def edt_soft(hard: np.ndarray, radius: float = 2.5) -> np.ndarray:
    """Soft thickness target from distance transform (centerline≈1, fades to edge)."""
    if hard.max() == 0:
        return hard.astype(np.float32)
    dist = ndimage.distance_transform_edt(hard.astype(bool))
    return np.clip(dist / float(radius), 0.0, 1.0).astype(np.float32)


def gt_skeleton(hard: np.ndarray) -> np.ndarray:
    if hard.max() == 0:
        return hard.astype(np.float32)
    sk = skeletonize(hard.astype(bool))
    return sk.astype(np.float32)


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_skel(x: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Differentiable soft-skeleton (clDice-style); no forced connectivity."""
    img = x
    skel = F.relu(img - soft_erode(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_erode(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(probs: torch.Tensor, soft_gt: torch.Tensor, skel_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - soft-clDice; skel_gt is precomputed GT skeleton (may be disconnected)."""
    skel_p = soft_skel(probs)
    tprec = (skel_p * soft_gt).sum(dim=(2, 3)) / (skel_p.sum(dim=(2, 3)) + eps)
    tsens = (skel_gt * probs).sum(dim=(2, 3)) / (skel_gt.sum(dim=(2, 3)) + eps)
    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return 1.0 - cl.mean()


class FascEDTDS(Dataset):
    def __init__(self, pairs, *, size=512, edt_radius=2.5, augment=False):
        self.pairs = pairs
        self.size = int(size)
        self.edt_radius = float(edt_radius)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stem, ip, mp = self.pairs[idx]
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
        hard = align_mask_to_img((np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8), img.shape[:2])
        soft = edt_soft(hard, self.edt_radius)
        skel = gt_skeleton(hard)
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                soft = np.ascontiguousarray(soft[:, ::-1])
                hard = np.ascontiguousarray(hard[:, ::-1])
                skel = np.ascontiguousarray(skel[:, ::-1])
            if random.random() < 0.4:
                img = np.clip(img * random.uniform(0.8, 1.25) + random.uniform(-0.05, 0.05), 0, 1)
            if random.random() < 0.25:
                ang = random.uniform(-8, 8)
                img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).rotate(ang, Image.BILINEAR, fillcolor=0), dtype=np.float32) / 255.0
                soft = np.asarray(Image.fromarray((soft * 255).astype(np.uint8)).rotate(ang, Image.BILINEAR, fillcolor=0), dtype=np.float32) / 255.0
                hard = (np.asarray(Image.fromarray(hard * 255).rotate(ang, Image.NEAREST, fillcolor=0)) > 127).astype(np.uint8)
                skel = (np.asarray(Image.fromarray((skel * 255).astype(np.uint8)).rotate(ang, Image.NEAREST, fillcolor=0)) > 127).astype(np.float32)

        img_p = Image.fromarray((img * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        soft_p = Image.fromarray((soft * 255).astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        hard_p = Image.fromarray(hard * 255).resize((self.size, self.size), Image.NEAREST)
        skel_p = Image.fromarray((skel * 255).astype(np.uint8)).resize((self.size, self.size), Image.NEAREST)
        return (
            torch.from_numpy(np.asarray(img_p, dtype=np.float32) / 255.0).unsqueeze(0),
            torch.from_numpy(np.asarray(soft_p, dtype=np.float32) / 255.0).unsqueeze(0),
            torch.from_numpy((np.asarray(hard_p) > 127).astype(np.float32)).unsqueeze(0),
            torch.from_numpy((np.asarray(skel_p) > 127).astype(np.float32)).unsqueeze(0),
        )


def loss_fn(logits, soft, hard, skel, *, pos_weight: float, cldice_w: float, aux=None):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)
    inter_s = (probs * soft).sum(dim=(2, 3))
    den_s = probs.sum(dim=(2, 3)) + soft.sum(dim=(2, 3))
    dice_s = 1.0 - (2 * inter_s + 1) / (den_s + 1)
    inter_h = (probs * hard).sum(dim=(2, 3))
    den_h = probs.sum(dim=(2, 3)) + hard.sum(dim=(2, 3))
    dice_h = 1.0 - (2 * inter_h + 1) / (den_h + 1)
    cld = soft_cldice(probs, soft, skel)
    loss = bce + dice_s.mean() + 0.5 * dice_h.mean() + cldice_w * cld
    if aux:
        for a, w in aux:
            pa = torch.sigmoid(a)
            loss = loss + w * (
                F.binary_cross_entropy_with_logits(a, soft, pos_weight=pw)
                + (1.0 - (2 * (pa * soft).sum(dim=(2, 3)) + 1) / (pa.sum(dim=(2, 3)) + soft.sum(dim=(2, 3)) + 1)).mean()
            )
    return loss


@torch.no_grad()
def eval_native_macro(model, pairs, *, size, device, thr=0.5, max_n=None):
    model.eval()
    dices, empty = [], 0
    for i, (stem, ip, mp) in enumerate(pairs):
        if max_n is not None and i >= max_n:
            break
        im = Image.open(ip).convert("L")
        w, h = im.size
        hard = align_mask_to_img((np.asarray(Image.open(mp).convert("L")) > 0).astype(np.uint8), (h, w)).astype(bool)
        x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        t = torch.from_numpy(x)[None, None].to(device)
        out = model(t)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        if prob.shape != (size, size):
            # Timm may return different spatial size
            prob = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        prob = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--edt-radius", type=float, default=2.5)
    ap.add_argument("--pos-weight", type=float, default=35.0)
    ap.add_argument("--cldice-w", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--device", type=str, default="cuda:3")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--warm-start", type=Path, default=None,
                    help="v5 ckpt; load compatible weights (skip aux1)")
    ap.add_argument("--native-every", type=int, default=1)
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
        f"fasc_v12_aux1_ms train={len(train_pairs)} val={len(val_pairs)} size={args.size} "
        f"base={args.base_ch} edt_r={args.edt_radius} pos_w={args.pos_weight} "
        f"cldice_w={args.cldice_w} device={device}",
        flush=True,
    )

    train_loader = DataLoader(
        FascEDTDS(train_pairs, size=args.size, edt_radius=args.edt_radius, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    model = MSTinyUNet(1, args.base_ch, n_aux=3).to(device)
    if args.warm_start is not None and Path(args.warm_start).exists():
        wd = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd = wd["model"] if isinstance(wd, dict) and "model" in wd else wd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # aux1 is new — expected missing; anything else is suspicious
        bad = [k for k in missing if not k.startswith("aux1")]
        if bad:
            raise SystemExit(f"warm-start missing unexpected keys: {bad[:12]}")
        if unexpected:
            raise SystemExit(f"warm-start unexpected keys: {unexpected[:12]}")
        print(f"warm-started from {args.warm_start} (aux1 randomly init)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v12_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v12_last.pt"
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
        for x, soft, hard, skel in train_loader:
            x, soft, hard, skel = x.to(device), soft.to(device), hard.to(device), skel.to(device)
            opt.zero_grad(set_to_none=True)
            logits, a3, a2, a1 = model(x, return_aux=True)
            loss = loss_fn(
                logits, soft, hard, skel,
                pos_weight=args.pos_weight,
                cldice_w=args.cldice_w,
                aux=[(a3, 0.3), (a2, 0.2), (a1, 0.1)],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
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
            "task": "fasc",
            "arch": "MSTinyUNet",
            "n_aux": 3,
            "aux_weights": [0.3, 0.2, 0.1],
            "size": args.size,
            "base_ch": args.base_ch,
            "val_dice": max(best_native, native),
            "val_native_035": n35,
            "val_native_050": n50,
            "epoch": epoch,
            "variant": "fasc_v12_aux1_ms",
            "pos_weight": args.pos_weight,
            "edt_radius": args.edt_radius,
            "cldice_w": args.cldice_w,
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
        "variant": "fasc_v12_aux1_ms",
        "best_val_native_macro_dice": best_native,
        "history": hist,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "checkpoint": str(ckpt),
        "v5_floor": 0.2846309831975224,
        "block_id": "fasc_thin_ceiling",
        "beats_v5_floor": bool(best_native > 0.2846309831975224),
        "attention_blocked": True,
    }
    (args.out_dir / "seg_fasc_v12_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_native_macro_dice={best_native:.4f} ckpt={ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
