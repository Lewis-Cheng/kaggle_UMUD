#!/usr/bin/env python3
"""Fasc v8: AMP + stronger topology + timm resnet34 TimmUNet (NEAREST align)."""
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
from umud.seg.timm_unet import TimmUNet

import importlib.util
spec = importlib.util.spec_from_file_location("v7", ROOT / "scripts/baselines/train_seg_fasc_v7.py")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--edt-radius", type=float, default=3.5)
    ap.add_argument("--pos-weight", type=float, default=45.0)
    ap.add_argument("--cldice-w", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda:2")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
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
        f"fasc_v8 train={len(train_pairs)} val={len(val_pairs)} enc={args.encoder} "
        f"size={args.size} edt_r={args.edt_radius} cldice_w={args.cldice_w} "
        f"amp={use_amp} bs={args.batch_size} device={device}",
        flush=True,
    )

    loader = DataLoader(
        v7.FascEDTDS(train_pairs, size=args.size, edt_radius=args.edt_radius, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    model = TimmUNet(args.encoder, base=args.base_ch, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "seg_fasc_v8_best.pt"
    last_ckpt = args.out_dir / "seg_fasc_v8_last.pt"
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
                loss = v7.loss_fn(logits, soft, hard, skel, pos_weight=args.pos_weight, cldice_w=args.cldice_w, aux=None)
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
            "task": "fasc", "arch": "TimmUNet", "encoder": args.encoder,
            "size": args.size, "base_ch": args.base_ch,
            "val_dice": max(best_native, native),
            "val_native_035": n35, "val_native_050": n50, "epoch": epoch,
            "variant": "fasc_v8_amp_topo_resnet34",
            "pos_weight": args.pos_weight, "edt_radius": args.edt_radius, "cldice_w": args.cldice_w,
            "amp": use_amp, "batch_size": args.batch_size, "eval_thr_preferred": prefer,
            "letterbox": False, "align_mask_to_img": True, "history": hist,
        }
        torch.save(payload, last_ckpt)
        if native > best_native:
            best_native = native
            payload["val_dice"] = best_native
            torch.save(payload, ckpt)
            print(f"  saved best_native={best_native:.4f}", flush=True)

    meta = {
        "variant": "fasc_v8_amp_topo_resnet34",
        "best_val_native_macro_dice": best_native,
        "history": hist, "n_train": len(train_pairs), "n_val": len(val_pairs),
        "checkpoint": str(ckpt), "amp": use_amp, "batch_size": args.batch_size, "encoder": args.encoder,
    }
    (args.out_dir / "seg_fasc_v8_train.json").write_text(json.dumps(meta, indent=2))
    print(f"best_native_macro_dice={best_native:.4f} ckpt={ckpt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
