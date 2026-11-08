#!/usr/bin/env python3
"""Short MultExp-domain fasc self-train (B).

No MultExp GT labels used (nested-safe for LOOCV geometry eval):
  teacher = Frundles fasc soft * belly + ridge seeds (blend_base_morph-like)
  student = Frundles fasc ResNet34 U-Net fine-tuned with Dice+BCE to teacher
  + mild consistency under flip/brightness.

Then dump MultExp maps and eval with frozen recipe (baseline + optional C later).
Timebox ~30-45 min on one GPU.
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
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.seg.frundles_smp import load_ckpt  # noqa: E402
from adapt_multexp_fasc import adapt_fasc, apo_belly_mask, structure_ridge  # noqa: E402
from adapt_multexp_fasc_v2 import adapt_fasc_v2  # noqa: E402
from adapt_multexp_fasc import FROZEN, PRIOR_BEST, predict_one, loocv_cal_pa_mt  # noqa: E402
import pandas as pd

HK = timezone(timedelta(hours=8))


def dice_loss(probs, gt, eps=1e-6):
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


def teacher_map(gray_full, apo, fasc):
    """Soft MultExp-domain teacher: blend baseline with morph_connect (best C partner)."""
    a = fasc.astype(np.float32)
    b = adapt_fasc(gray_full, apo, fasc, "morph_connect")
    # also mild ridge soft
    h, w = apo.shape
    g = cv2.resize(gray_full, (w, h), interpolation=cv2.INTER_AREA)
    belly = apo_belly_mask(apo, 0.60, 2).astype(np.float32)
    ridge, _, coher = structure_ridge(g, 1.5)
    soft = np.maximum(0.7 * a + 0.3 * b, 0.18 * ridge * coher * belly)
    return np.clip(soft, 0, 1).astype(np.float32)


class MultExpTeacherDS(Dataset):
    def __init__(self, expert_csv, expert_dir, cache_dir, size=(384, 576)):
        self.expert = pd.read_csv(expert_csv)
        self.expert_dir = Path(expert_dir)
        self.cache_dir = Path(cache_dir)
        self.H, self.W = size
        self.items = []
        for _, er in self.expert.iterrows():
            name = er["image_file"]
            gpath = self.expert_dir / name
            zpath = self.cache_dir / f"{name}.npz"
            if gpath.exists() and zpath.exists():
                self.items.append((name, gpath, zpath))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, gpath, zpath = self.items[idx]
        g = cv2.imread(str(gpath), cv2.IMREAD_GRAYSCALE)
        z = np.load(zpath)
        apo = z["apo"].astype(np.float32)
        fasc = z["fasc"].astype(np.float32)
        teach = teacher_map(g, apo, fasc)
        g_r = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        # apo/fasc already 384x576
        if teach.shape != (self.H, self.W):
            teach = cv2.resize(teach, (self.W, self.H), interpolation=cv2.INTER_AREA)
        # aug
        if np.random.rand() < 0.5:
            g_r = g_r[:, ::-1].copy()
            teach = teach[:, ::-1].copy()
        if np.random.rand() < 0.5:
            alpha = 0.85 + 0.3 * np.random.rand()
            beta = (np.random.rand() - 0.5) * 0.1
            g_r = np.clip(g_r * alpha + beta, 0, 1)
        x = torch.from_numpy(g_r[None]).float()
        y = torch.from_numpy(teach[None]).float()
        return x, y, name


@torch.no_grad()
def dump_maps(model, expert_csv, expert_dir, apo_cache, out_dir, device):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expert = pd.read_csv(expert_csv)
    expert_dir = Path(expert_dir)
    apo_cache = Path(apo_cache)
    model.eval()
    for _, er in expert.iterrows():
        name = er["image_file"]
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        z = np.load(apo_cache / f"{name}.npz")
        apo = z["apo"].astype(np.float32)
        H, W = apo.shape
        g_r = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = torch.from_numpy(g_r[None, None]).float().to(device)
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
        np.savez_compressed(out_dir / f"{name}.npz", apo=apo.astype(np.float16), fasc=prob.astype(np.float16))
    return out_dir


def eval_cache(cache_dir, expert_csv, expert_dir):
    expert = pd.read_csv(expert_csv)
    expert_dir = Path(expert_dir)
    cache_dir = Path(cache_dir)
    kw = dict(FROZEN)
    rows = []
    for _, er in expert.iterrows():
        name = er["image_file"]
        g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        z = np.load(cache_dir / f"{name}.npz")
        pred = predict_one(
            g,
            z["apo"].astype(np.float32),
            z["fasc"].astype(np.float32),
            float(er["Scale_pixel_per_cm"]) / 10.0,
            kw,
        )
        rows.append(
            dict(
                image_file=name,
                ok=pred["ok"],
                pred_pa=pred["pred_pa"],
                pred_fl=pred["pred_fl"],
                pred_mt=pred["pred_mt"],
                gt_pa=float(er["median_PA"]),
                gt_fl=float(er["median_FL"]),
                gt_mt=float(er["median_MT"]),
            )
        )
    df = pd.DataFrame(rows)
    comps, mae, _, _ = loocv_cal_pa_mt(df)
    return {
        "S_loocv_cal_pa_mt": float(comps["S"]),
        "MAE_loocv": mae,
        "gate_0.12": float(comps["S"]) < 0.12,
        "beats_0.13294": float(comps["S"]) < PRIOR_BEST,
        "beats_0.13195": float(comps["S"]) < 0.13195144580335014,
        "n": len(df),
    }, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--time-limit-min", type=float, default=40.0)
    args = ap.parse_args()

    expert_csv = ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    ckpt_in = ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt"
    out_dir = ROOT / "reports/phase3/chase018"
    seg_dir = out_dir / "seg_ckpts"
    seg_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} epochs={args.epochs}", flush=True)

    model, size = load_ckpt(ckpt_in, device)
    model.train()
    print(f"loaded frundles fasc size={size}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ds = MultExpTeacherDS(expert_csv, expert_dir, cache)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=False)
    print(f"n_images={len(ds)}", flush=True)

    t0 = time.time()
    best_loss = 9e9
    best_path = seg_dir / "fasc_multexp_selftrain_best.pt"
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            loss = 0.5 * dice_loss(probs, y) + 0.5 * F.binary_cross_entropy(probs, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        mean_l = float(np.mean(losses)) if losses else 9.0
        hist.append({"ep": ep, "loss": mean_l, "mins": (time.time() - t0) / 60.0})
        print(f"ep {ep}/{args.epochs} loss={mean_l:.4f} t={hist[-1]['mins']:.1f}m", flush=True)
        if mean_l < best_loss:
            best_loss = mean_l
            torch.save({"encoder": "resnet34", "size": list(size), "state": model.state_dict(), "ep": ep, "loss": mean_l}, best_path)
        if (time.time() - t0) / 60.0 >= args.time_limit_min:
            print("time limit hit", flush=True)
            break

    # load best and dump maps
    blob = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["state"])
    maps_out = out_dir / "cache" / "maps_multexp_selftrain_b"
    print("dumping maps...", flush=True)
    dump_maps(model, expert_csv, expert_dir, cache, maps_out, device)

    print("eval frozen recipe...", flush=True)
    summary, df = eval_cache(maps_out, expert_csv, expert_dir)
    print(summary, flush=True)
    df.to_csv(out_dir / "multexp_fasc_adapt_b_selftrain_rows.csv", index=False)

    # also eval Frundles baseline for sanity
    base_s, _ = eval_cache(cache, expert_csv, expert_dir)

    payload = {
        "asof": datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),
        "method": "B_multexp_selftrain_teacher_morph_ridge",
        "ckpt": str(best_path),
        "maps": str(maps_out),
        "train_hist": hist,
        "best_loss": best_loss,
        "S_selftrain": summary,
        "S_frundles_baseline_recheck": base_s,
        "prior_adapt_C": 0.13195144580335014,
        "prior_fixed": PRIOR_BEST,
        "gate_0.12": summary["gate_0.12"],
        "promote": bool(summary["S_loocv_cal_pa_mt"] < 0.13195144580335014),
        "kaggle_submit": False,
        "uses_multexp_gt": False,
        "note": "Self-train teacher from Frundles+morph/ridge on MultExp images only; no PA/FL/MT GT in loss",
    }
    (out_dir / "multexp_fasc_adapt_b_selftrain.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out_dir / "multexp_fasc_adapt_b_selftrain.json", flush=True)
    print(
        f"B_RESULT S={summary['S_loocv_cal_pa_mt']:.5f} "
        f"vs C={0.13195:.5f} vs fixed={PRIOR_BEST:.5f} gate={summary['gate_0.12']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
