#!/usr/bin/env python3
"""Fine-tune LabTrain fasc ckpt on LabTrain + competition; MultExp never in train.

Loads ImageNet/LabTrain best, continues on union of LabTrain fasc + competition
fasc_imgs_v1 / fasc_masks_v1. Excludes any train image whose dHash is within
max_dist of a MultExp expert image. Then dump MultExp fasc + Frundles apo and
frozen LOOCV vs hold 0.13195 / gate 0.12.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.geometry.host_faithful import dhash  # noqa: E402
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402
from train_eval_labtrain_fasc import (  # noqa: E402
    FascDS,
    discover_pairs,
    dump_multexp,
    eval_frozen,
    groups_from_hashes,
    soft_dice,
)

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
H, W = 384, 576


def now():
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")


def multexp_hashes():
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    hs = []
    for _, er in expert.iterrows():
        g = cv2.imread(str(expert_dir / er["image_file"]), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            hs.append(dhash(g))
    return hs


def far_from_multexp(img_path: Path, mex_hs, max_dist=8) -> bool:
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return False
    h = dhash(g)
    for mh in mex_hs:
        if (np.asarray(h, dtype=np.uint8) != np.asarray(mh, dtype=np.uint8)).sum() <= max_dist:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-img", type=Path, default=ROOT / "data/external/osf_xbawc/labtrain_fasc_images")
    ap.add_argument("--lab-mask", type=Path, default=ROOT / "data/external/osf_xbawc/labtrain_fasc_masks")
    ap.add_argument("--comp-img", type=Path, default=ROOT / "data/raw/umud/fasc_imgs_v1")
    ap.add_argument("--comp-mask", type=Path, default=ROOT / "data/raw/umud/fasc_masks_v1")
    ap.add_argument(
        "--init-ckpt",
        type=Path,
        default=ROOT / "reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_labtrain_best.pt",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018/cache/maps_multexp_labtrain_comp_ft",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-dist", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[ft] init={args.init_ckpt}", flush=True)
    mex = multexp_hashes()
    print(f"[ft] MultExp hashes n={len(mex)}", flush=True)

    pairs = discover_pairs(args.lab_img, args.lab_mask) + discover_pairs(args.comp_img, args.comp_mask)
    # unique by stem
    seen, uniq = set(), []
    for s, ip, mp in pairs:
        if s in seen:
            continue
        seen.add(s)
        uniq.append((s, ip, mp))
    pairs = uniq
    print(f"[ft] discovered {len(pairs)} before MultExp filter", flush=True)

    kept = []
    dropped = 0
    for item in pairs:
        if far_from_multexp(item[1], mex, args.max_dist):
            kept.append(item)
        else:
            dropped += 1
    pairs = kept
    print(f"[ft] after MultExp dHash filter: keep={len(pairs)} drop={dropped}", flush=True)
    if len(pairs) < 50:
        raise SystemExit("too few pairs after filter")

    hs = []
    for _, ip, _ in pairs:
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        hs.append(dhash(g))
    grp = groups_from_hashes(hs)
    ug = np.unique(grp)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ug)
    n_val = max(1, int(round(len(ug) * args.val_frac)))
    val_g = set(ug[:n_val].tolist())
    train_pairs = [p for p, g in zip(pairs, grp) if g not in val_g]
    val_pairs = [p for p, g in zip(pairs, grp) if g in val_g]
    print(f"[ft] train={len(train_pairs)} val={len(val_pairs)} groups={len(ug)}", flush=True)

    device = args.device
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, (len(train_pairs) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    bce = torch.nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(FascDS(train_pairs, True, args.seed), batch_size=args.batch, shuffle=True, num_workers=2)
    va_loader = DataLoader(FascDS(val_pairs, False, args.seed + 1), batch_size=args.batch, shuffle=False, num_workers=2)

    best = -1.0
    hist = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "fasc_resnet34_384x576_labtrain_comp_ft_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            probs = torch.sigmoid(logits)
            loss = 0.5 * bce(logits, y) + 0.5 * soft_dice(probs, y)
            loss.backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            losses.append(float(loss.item()))
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                probs = torch.sigmoid(model(x))
                pred = (probs > 0.5).float()
                inter = (pred * y).sum((2, 3))
                den = pred.sum((2, 3)) + y.sum((2, 3)) + 1e-6
                dices.append((2 * inter / den).mean().item())
        vd = float(np.mean(dices)) if dices else 0.0
        hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_dice": vd})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} val_dice={vd:.4f}", flush=True)
        if vd > best:
            best = vd
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "smp_unet_resnet34",
                    "in_ch": 1,
                    "size": (H, W),
                    "best_val_dice": best,
                    "source": "labtrain_then_comp_ft",
                    "init": str(args.init_ckpt),
                    "encoder_weights": "imagenet->labtrain->comp",
                    "n_train": len(train_pairs),
                    "n_dropped_multexp": dropped,
                },
                ckpt_path,
            )
            print("saved", ckpt_path, "best", best, flush=True)

    (args.out_dir / "fasc_labtrain_comp_ft_train.json").write_text(
        json.dumps(
            {
                "best": best,
                "hist": hist,
                "n_train": len(train_pairs),
                "n_val": len(val_pairs),
                "dropped_multexp": dropped,
            },
            indent=2,
        )
    )

    dump_multexp(ckpt_path, args.cache_dir, args.device)
    results, best_ev, best_df = eval_frozen(args.cache_dir)
    out = ROOT / "reports/phase3/chase018"
    if best_df is not None:
        best_df.to_csv(out / "multexp_nonfrundles_fasc_labtrain_comp_ft_best_rows.csv", index=False)
    payload = {
        "ts": now(),
        "hold": HOLD,
        "gate": GATE,
        "ckpt": str(ckpt_path),
        "best_val_dice": best,
        "dropped_multexp_near": dropped,
        "cache": str(args.cache_dir),
        "results": results,
        "best": best_ev,
        "promoted": bool(best_ev and best_ev["S"] < HOLD),
        "gate_pass": bool(best_ev and best_ev["S"] < GATE),
        "elapsed_s": time.time() - t0,
        "prior_labtrain_S": 0.13817728585553185,
        "note": "LabTrain ImageNet fasc FT on LabTrain+competition; MultExp dHash-excluded; Frundles apo",
    }
    (out / "multexp_nonfrundles_fasc_labtrain_comp_ft.json").write_text(json.dumps(payload, indent=2))
    be = best_ev
    status = (
        f"# LabTrain+comp FT fasc MultExp\nUpdated: {now()}\n\n"
        f"- Protocol: ImageNet→LabTrain (1854)→FT LabTrain+comp, MultExp dHash-excluded (drop={dropped})\n"
        f"- Best MultExp S: **{be['S']:.5f}** thr={be['thr']} "
        f"PA={be['MAE']['pa_deg']:.3f} FL={be['MAE']['fl_mm']:.3f} MT={be['MAE']['mt_mm']:.3f}\n"
        f"- Prior LabTrain-only S=0.13818; Hold 0.13195: "
        f"{'BEAT' if payload['promoted'] else 'MISS'}; Gate 0.12: "
        f"{'PASS' if payload['gate_pass'] else 'FAIL'}; no Kaggle\n"
        f"- ckpt: `{ckpt_path}` val_dice={best}\n"
    )
    # also refresh primary status if better than prior
    prior = 0.13817728585553185
    if be["S"] < prior:
        (out / "multexp_nonfrundles_fasc_status.md").write_text(status)
    (out / "multexp_nonfrundles_fasc_labtrain_comp_ft_status.md").write_text(status)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
