#!/usr/bin/env python3
"""Fasc v6: higher-res MSTinyUNet (640) + stronger soft-clDice/EDT; optional warm-start from v5."""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from umud.baselines.io import load_split_assignments
from umud.data.layout import discover_layout, list_images
from umud.seg.ms_unet import MSTinyUNet
import importlib.util
spec = importlib.util.spec_from_file_location("v5", ROOT / "scripts/baselines/train_seg_fasc_v5.py")
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--edt-radius", type=float, default=3.0)
    ap.add_argument("--pos-weight", type=float, default=40.0)
    ap.add_argument("--cldice-w", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--device", type=str, default="cuda:3")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--init-from", type=Path, default=None, help="warm-start weights (same arch)")
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
    print(
        f"fasc_v6 train={len(train_pairs)} val={len(val_pairs)} size={args.size} "
        f"base={args.base_ch} edt_r={args.edt_radius} pos_w={args.pos_weight} "
        f"cldice_w={args.cldice_w} device={device}",
        flush=True,
    )
    loader = DataLoader(
        v5.FascEDTDS(train_pairs, size=args.size, edt_radius=args.edt_radius, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    model = MSTinyUNet(1, args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v6_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v6_last.pt"
    best = -1.0
    hist = []
    start = 1

    if args.init_from and Path(args.init_from).exists() and args.resume is None and not last_ckpt.exists():
        rd = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(rd["model"], strict=False)
        print(f"warm-start {args.init_from} missing={len(missing)} unexpected={len(unexpected)} src_dice={rd.get('val_dice')}", flush=True)

    resume_path = args.resume if args.resume is not None else (last_ckpt if last_ckpt.exists() else None)
    if resume_path is not None and Path(resume_path).exists():
        rd = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rd["model"])
        for key, obj in (("optimizer", opt), ("scheduler", sched)):
            if key in rd:
                try: obj.load_state_dict(rd[key])
                except Exception as e: print("warn", key, e, flush=True)
        start = int(rd.get("epoch", 0)) + 1
        best = float(rd.get("val_dice", -1) or -1)
        hist = list(rd.get("history", []))
        print(f"resumed {resume_path} ep={start-1} best={best:.4f}", flush=True)

    for epoch in range(start, args.epochs + 1):
        model.train(); losses = []
        for x, soft, hard, skel in loader:
            x, soft, hard, skel = x.to(device), soft.to(device), hard.to(device), skel.to(device)
            opt.zero_grad(set_to_none=True)
            logits, a3, a2 = model(x, return_aux=True)
            loss = v5.loss_fn(logits, soft, hard, skel, pos_weight=args.pos_weight, cldice_w=args.cldice_w, aux=[(a3, 0.3), (a2, 0.2)])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.item()))
        sched.step()
        n35 = v5.eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.35)
        n50 = v5.eval_native_macro(model, val_pairs, size=args.size, device=device, thr=0.5)
        score = max(n35["macro_dice"], n50["macro_dice"])
        prefer = 0.35 if n35["macro_dice"] >= n50["macro_dice"] else 0.5
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "n35": n35, "n50": n50}
        hist.append(row)
        print(f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} native@0.35={n35['macro_dice']:.4f} native@0.5={n50['macro_dice']:.4f}", flush=True)
        payload = {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "task": "fasc", "arch": "MSTinyUNet", "size": args.size, "base_ch": args.base_ch,
            "val_dice": max(best, score), "val_native_035": n35, "val_native_050": n50,
            "epoch": epoch, "variant": "fasc_v6_hr_edt_cldice_ms", "pos_weight": args.pos_weight,
            "edt_radius": args.edt_radius, "cldice_w": args.cldice_w, "eval_thr_preferred": prefer,
            "letterbox": False, "align_mask_to_img": True, "history": hist,
        }
        torch.save(payload, last_ckpt)
        if score > best:
            best = score
            payload["val_dice"] = best
            torch.save(payload, ckpt)
            print(f"  saved best_native={best:.4f}", flush=True)

    meta = {"variant": "fasc_v6_hr_edt_cldice_ms", "best_val_native_macro_dice": best, "history": hist,
            "n_train": len(train_pairs), "n_val": len(val_pairs), "checkpoint": str(ckpt)}
    (args.out_dir / "seg_fasc_v6_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_native_macro_dice={best:.4f} ckpt={ckpt}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
