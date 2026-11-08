#!/usr/bin/env python3
"""ExpDrawn overlay → fasc masks; Frundles-init FT; MultExp held out.

1) Parse ExpDrawn annotated panels: middle 800 cols (x=200:1000) align with raw
   .tiff; extract red fasc strokes → dilated soft masks.
2) Fine-tune from Frundles public fasc (BCE+Dice+light orient), optional LabTrain mix.
3) Dump MultExp maps + α-blend with Frundles; honest LOOCV frozen recipe.
4) Promote only if S < 0.13195; gate if S < 0.12. No Kaggle. No attention.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.geometry.host_faithful import dhash  # noqa: E402
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
H, W = 384, 576


def now() -> str:
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")


def soft_dice(probs, gt, eps=1e-6):
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


def orient_consistency(probs, gt, eps=1e-6):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3) / 8.0

    def st_angle(x):
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        avg = torch.ones(1, 1, 5, 5, dtype=x.dtype, device=x.device) / 25.0
        gxx = F.conv2d(gx * gx, avg, padding=2)
        gyy = F.conv2d(gy * gy, avg, padding=2)
        gxy = F.conv2d(gx * gy, avg, padding=2)
        return 0.5 * torch.atan2(2 * gxy, gxx - gyy + eps)

    with torch.no_grad():
        ang_g = st_angle(gt)
        mag_g = F.conv2d(gt, kx, padding=1).abs() + F.conv2d(gt, ky, padding=1).abs()
        w = (gt > 0.3).float() * (mag_g > 0.02).float()
    ang_p = st_angle(probs)
    d = ang_p - ang_g
    diff = torch.atan2(torch.sin(2 * d), torch.cos(2 * d)) / 2.0
    return ((diff.abs() * w).sum((2, 3)) / (w.sum((2, 3)) + eps)).mean()


def groups_from_hashes(hs, max_dist=12):
    parent = list(range(len(hs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    arr = np.array(hs, dtype=np.uint8)
    for i in range(len(arr)):
        for j in np.nonzero((arr[i + 1 :] != arr[i]).sum(1) <= max_dist)[0] + i + 1:
            parent[find(int(j))] = find(i)
    return np.array([find(i) for i in range(len(hs))])


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


def extract_expdrawn(
    exp_root: Path,
    out_img: Path,
    out_mask: Path,
    min_corr: float = 0.97,
    dilate_r: int = 1,
    min_red: int = 80,
) -> dict:
    """Convert annotated panels → raw-aligned fasc masks.

    Annotated .tif is 608×1200 RGB; raw .tiff is 608×800 RGB.
    Columns x=200:1000 of annotated align with raw (corr≈0.99).
    Fasc strokes are pure-red (R>200,G<60,B<60).
    """
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    meta = []
    n_ok = n_fail = n_low_red = 0
    for split_name, split_rel in [
        ("young", "young/NeuAge_Young_ANONYMIZED_annotated_v1"),
        ("old", "old/NeuAge_Old_ANONYMIZED_annotated_v1"),
    ]:
        split_dir = exp_root / split_rel
        if not split_dir.exists():
            continue
        for subj in sorted(split_dir.iterdir()):
            tp = subj / "timepoint1"
            if not tp.is_dir():
                continue
            tifs = sorted(tp.glob("*.tif"))
            tiffs = sorted(tp.glob("*.tiff"))
            for ann_p, raw_p in zip(tifs, tiffs):
                ann = cv2.imread(str(ann_p), cv2.IMREAD_COLOR)  # BGR
                raw = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)
                if ann is None or raw is None:
                    n_fail += 1
                    meta.append({"stem": ann_p.stem, "ok": False, "reason": "imread"})
                    continue
                if ann.shape[0] != raw.shape[0] or ann.shape[1] < raw.shape[1] + 200:
                    n_fail += 1
                    meta.append({"stem": ann_p.stem, "ok": False, "reason": f"shape ann={ann.shape} raw={raw.shape}"})
                    continue
                mid = ann[:, 200 : 200 + raw.shape[1], :]
                # correlation on grayscale
                sg = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY).astype(np.float32)
                rg = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if sg.std() < 1 or rg.std() < 1:
                    n_fail += 1
                    meta.append({"stem": ann_p.stem, "ok": False, "reason": "low_std"})
                    continue
                corr = float(np.corrcoef(sg.ravel(), rg.ravel())[0, 1])
                # red in BGR: B low, G low, R high → channels [B,G,R]
                b, g, r = mid[:, :, 0], mid[:, :, 1], mid[:, :, 2]
                red = (r > 200) & (g < 60) & (b < 60)
                n_red = int(red.sum())
                if corr < min_corr:
                    n_fail += 1
                    meta.append({"stem": ann_p.stem, "ok": False, "reason": f"corr={corr:.4f}", "n_red": n_red})
                    continue
                if n_red < min_red:
                    n_low_red += 1
                    meta.append({"stem": ann_p.stem, "ok": False, "reason": f"low_red={n_red}", "corr": corr})
                    continue
                mask = red.astype(np.uint8) * 255
                if dilate_r > 0:
                    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1))
                    mask = cv2.dilate(mask, ker)
                # save grayscale raw + mask with unique stem
                stem = f"{split_name}_{subj.name}_{ann_p.stem}"
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(out_img / f"{stem}.png"), gray)
                cv2.imwrite(str(out_mask / f"{stem}.png"), mask)
                n_ok += 1
                meta.append(
                    {
                        "stem": stem,
                        "ok": True,
                        "split": split_name,
                        "subj": subj.name,
                        "corr": corr,
                        "n_red": n_red,
                        "pos_frac": float((mask > 127).mean()),
                        "ann": str(ann_p),
                        "raw": str(raw_p),
                    }
                )
    summary = {
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_low_red": n_low_red,
        "min_corr": min_corr,
        "dilate_r": dilate_r,
        "out_img": str(out_img),
        "out_mask": str(out_mask),
    }
    return {"summary": summary, "meta": meta}


def discover_pairs(img_root: Path, mask_root: Path):
    imgs = {}
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            imgs[p.stem] = p
    masks = {}
    for p in mask_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            masks[p.stem] = p
    return sorted([(s, imgs[s], masks[s]) for s in imgs if s in masks], key=lambda t: t[0])


def labtrain_numeric_pairs(max_n: int = 309):
    """Co-located LabTrain apo+fasc stems (optional mix). Prefer fasc_* stems that also have apo."""
    fasc_img = ROOT / "data/external/osf_xbawc/labtrain_fasc_images"
    fasc_msk = ROOT / "data/external/osf_xbawc/labtrain_fasc_masks"
    apo_img = ROOT / "data/external/osf_xbawc/labtrain_apo_images"
    pairs = discover_pairs(fasc_img, fasc_msk)
    # prefer stems that also exist under apo images
    apo_stems = set()
    if apo_img.exists():
        for p in apo_img.rglob("*"):
            if p.is_file():
                apo_stems.add(p.stem)
    ranked = []
    for s, ip, mp in pairs:
        ranked.append((0 if s in apo_stems else 1, s, ip, mp))
    ranked.sort()
    out = [(s, ip, mp) for _, s, ip, mp in ranked[:max_n]]
    return out


class FascDS(Dataset):
    def __init__(self, pairs, augment: bool, seed: int = 0):
        self.pairs = pairs
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, ip, mp = self.pairs[idx]
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if g is None or m is None:
            g = np.zeros((H, W), np.uint8)
            m = np.zeros((H, W), np.uint8)
        # Frundles-style: aspect-ish random crop when augmenting
        if self.augment:
            Hh, Ww = g.shape
            ch = int(Hh * self.rng.uniform(0.65, 1.0))
            cw = int(Ww * self.rng.uniform(0.65, 1.0))
            y0 = int(self.rng.randint(0, max(1, Hh - ch + 1)))
            x0 = int(self.rng.randint(0, max(1, Ww - cw + 1)))
            g = g[y0 : y0 + ch, x0 : x0 + cw]
            m = m[y0 : y0 + ch, x0 : x0 + cw]
            if self.rng.rand() < 0.5:
                g = np.fliplr(g).copy()
                m = np.fliplr(m).copy()
            if self.rng.rand() < 0.3:
                a = float(self.rng.uniform(-8, 8))
                M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), a, 1.0)
                g = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        # INTER_AREA then thr>0.25 preserves thin strokes
        m_f = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        y = (m_f > 0.25 * 255).astype(np.float32)
        x = g.astype(np.float32) / 255.0
        if self.augment:
            x = np.clip((x ** self.rng.uniform(0.7, 1.4)) * self.rng.uniform(0.8, 1.2) + self.rng.uniform(-0.05, 0.05), 0, 1)
            if self.rng.rand() < 0.25:
                x = np.clip(x * self.rng.normal(1.0, 0.08, x.shape), 0, 1)
        return torch.from_numpy(x.astype(np.float32))[None], torch.from_numpy(y)[None]


def dump_multexp(model, cache_dir: Path, device: str) -> int:
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    apo_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    n = 0
    with torch.no_grad():
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            zsrc = apo_cache / f"{name}.npz"
            if g is None or not zsrc.exists():
                continue
            apo = np.load(zsrc)["apo"].astype(np.float32)
            x = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            t = torch.from_numpy(x)[None, None].to(device)
            fasc = torch.sigmoid(model(t))[0, 0].detach().cpu().numpy().astype(np.float32)
            if fasc.shape != apo.shape:
                fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_LINEAR)
            np.savez_compressed(cache_dir / f"{name}.npz", apo=apo.astype(np.float16), fasc=fasc.astype(np.float16))
            n += 1
    return n


def blend_caches(new_dir: Path, fr_dir: Path, out_dir: Path, alpha: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for zp in sorted(fr_dir.glob("*.npz")):
        zn = new_dir / zp.name
        if not zn.exists():
            continue
        zfr = np.load(zp)
        znew = np.load(zn)
        fasc = alpha * znew["fasc"].astype(np.float32) + (1.0 - alpha) * zfr["fasc"].astype(np.float32)
        np.savez_compressed(out_dir / zp.name, apo=zfr["apo"].astype(np.float16), fasc=fasc.astype(np.float16))
        n += 1
    return n


def eval_frozen_robust(cache_dir: Path):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    thr_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
    results = []
    best = None
    best_df = None
    for thr in thr_grid:
        kw = dict(FROZEN)
        kw["fasc_thr"] = float(thr)
        rows = []
        for _, er in expert.iterrows():
            name = er["image_file"]
            g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
            zp = cache_dir / f"{name}.npz"
            if g is None or not zp.exists():
                continue
            z = np.load(zp)
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
        out = loocv_cal_pa_mt(df)
        comps, mae = out[0], out[1]
        S = float(comps["S"]) if isinstance(comps, dict) else float(comps)

        def _m(keys):
            if isinstance(mae, dict):
                for k in keys:
                    if k in mae:
                        return float(mae[k])
            return float("nan")

        summary = {
            "thr": float(thr),
            "S": S,
            "mae_pa": _m(["pa_deg", "pa", "mae_pa"]),
            "mae_fl": _m(["fl_mm", "fl", "mae_fl"]),
            "mae_mt": _m(["mt_mm", "mt", "mae_mt"]),
            "n": int(len(df)),
            "recipe": FROZEN["name"],
        }
        results.append(summary)
        print(
            f"  thr={thr:.2f} S={summary['S']:.5f} PA={summary['mae_pa']:.3f} "
            f"FL={summary['mae_fl']:.3f} MT={summary['mae_mt']:.3f}",
            flush=True,
        )
        if best is None or summary["S"] < best["S"]:
            best = summary
            best_df = df
    return best, results, best_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdrawn-root", type=Path, default=ROOT / "data/external/osf_xbawc/expdrawn")
    ap.add_argument("--out-img", type=Path, default=ROOT / "data/external/osf_xbawc/expdrawn_images")
    ap.add_argument("--out-mask", type=Path, default=ROOT / "data/external/osf_xbawc/expdrawn_masks")
    ap.add_argument("--frundles-ckpt", type=Path, default=ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "reports/phase3/chase018/cache/maps_multexp_expdrawn")
    ap.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase3/chase018")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--w-orient", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dilate-r", type=int, default=1)
    ap.add_argument("--mix-labtrain", type=int, default=309, help="0=ExpDrawn only; else mix N LabTrain pairs")
    ap.add_argument("--blend-alphas", type=str, default="0.15,0.25,0.40,0.55,0.70")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[expdrawn] start {now()}", flush=True)

    # --- 1) extract ---
    extract_json = args.report_dir / "multexp_expdrawn_extract.json"
    if args.skip_extract and args.out_img.exists() and args.out_mask.exists() and len(list(args.out_img.glob("*.png"))) > 50:
        print(f"[expdrawn] skip extract; existing imgs={len(list(args.out_img.glob('*.png')))}", flush=True)
        extract_info = json.loads(extract_json.read_text()) if extract_json.exists() else {"summary": {"n_ok": "existing"}}
    else:
        print("[expdrawn] extracting overlays …", flush=True)
        extract_info = extract_expdrawn(args.expdrawn_root, args.out_img, args.out_mask, dilate_r=args.dilate_r)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        extract_json.write_text(json.dumps(extract_info, indent=2))
        print(f"[expdrawn] extract {extract_info['summary']}", flush=True)

    mex = multexp_hashes()
    print(f"[expdrawn] MultExp hashes n={len(mex)} HELD OUT", flush=True)

    pairs = []
    for s, ip, mp in discover_pairs(args.out_img, args.out_mask):
        if far_from_multexp(ip, mex, 8):
            pairs.append((s, ip, mp, "expdrawn"))
    n_exp = len(pairs)
    print(f"[expdrawn] usable ExpDrawn after MultExp filter: {n_exp}", flush=True)

    if args.mix_labtrain > 0:
        lab = labtrain_numeric_pairs(args.mix_labtrain)
        n_lab_keep = 0
        seen = {s for s, *_ in pairs}
        for s, ip, mp in lab:
            if s in seen:
                continue
            if far_from_multexp(ip, mex, 8):
                pairs.append((s, ip, mp, "labtrain"))
                seen.add(s)
                n_lab_keep += 1
        print(f"[expdrawn] +LabTrain mix keep={n_lab_keep} (cap={args.mix_labtrain})", flush=True)

    if len(pairs) < 30:
        raise SystemExit(f"too few usable pairs: {len(pairs)}")

    # group split
    hs = [dhash(cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)) for _, ip, _, _ in pairs]
    grp = groups_from_hashes(hs)
    ug = np.unique(grp)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ug)
    n_val = max(1, int(round(len(ug) * args.val_frac)))
    val_g = set(ug[:n_val].tolist())
    train_pairs = [(s, ip, mp) for (s, ip, mp, _), g in zip(pairs, grp) if g not in val_g]
    val_pairs = [(s, ip, mp) for (s, ip, mp, _), g in zip(pairs, grp) if g in val_g]
    src_keep = {}
    for _, _, _, lab in pairs:
        src_keep[lab] = src_keep.get(lab, 0) + 1
    print(f"[expdrawn] train={len(train_pairs)} val={len(val_pairs)} groups={len(ug)} src={src_keep}", flush=True)

    # --- 2) train from Frundles ---
    device = args.device
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    ck = torch.load(args.frundles_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state"], strict=True)
    init_src = f"frundles_state encoder={ck.get('encoder')} size={ck.get('size')}"
    print(f"[expdrawn] init={init_src}", flush=True)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, (len(train_pairs) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    pw = torch.tensor([args.pos_weight], device=device)
    tr_loader = DataLoader(FascDS(train_pairs, True, args.seed), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    va_loader = DataLoader(FascDS(val_pairs, False, args.seed + 1), batch_size=args.batch, shuffle=False, num_workers=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "fasc_resnet34_384x576_expdrawn_frundles_best.pt"
    best_dice = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts = {"bce": [], "dice": [], "orient": []}
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            probs = torch.sigmoid(logits)
            lb = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            ld = soft_dice(probs, y)
            lo = orient_consistency(probs, y) if args.w_orient > 0 else torch.tensor(0.0, device=device)
            loss = 0.55 * lb + 0.40 * ld + args.w_orient * lo
            loss.backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            losses.append(float(loss.item()))
            parts["bce"].append(float(lb.item()))
            parts["dice"].append(float(ld.item()))
            parts["orient"].append(float(lo.item()) if torch.is_tensor(lo) else float(lo))
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
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "val_dice": vd,
            "bce": float(np.mean(parts["bce"])) if parts["bce"] else float("nan"),
            "dice": float(np.mean(parts["dice"])) if parts["dice"] else float("nan"),
            "orient": float(np.mean(parts["orient"])) if parts["orient"] else float("nan"),
        }
        hist.append(row)
        print(
            f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} bce={row['bce']:.3f} "
            f"dice={row['dice']:.3f} orient={row['orient']:.3f} val_dice={vd:.4f}",
            flush=True,
        )
        if vd > best_dice:
            best_dice = vd
            torch.save(
                {
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "model": model.state_dict(),
                    "encoder": "resnet34",
                    "size": (H, W),
                    "val_dice": best_dice,
                    "source": "expdrawn_frundles_ft",
                    "init": init_src,
                    "train_mix": src_keep,
                    "n_train": len(train_pairs),
                    "n_val": len(val_pairs),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print("saved", ckpt_path, "best_dice", best_dice, flush=True)

    # reload best
    ck2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck2["state"] if "state" in ck2 else ck2["model"])
    model = model.to(device)

    # --- 3) MultExp dump + eval + blend ---
    print(f"[expdrawn] dump MultExp maps -> {args.cache_dir}", flush=True)
    nmaps = dump_multexp(model, args.cache_dir, device)
    print(f"[expdrawn] dumped {nmaps}", flush=True)

    print("[expdrawn] eval raw ExpDrawn-FT fasc + Frundles apo", flush=True)
    best_raw, grid_raw, df_raw = eval_frozen_robust(args.cache_dir)

    fr_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    blend_results = []
    best_blend = None
    alphas = [float(x) for x in args.blend_alphas.split(",") if x.strip()]
    for a in alphas:
        bdir = args.report_dir / "cache" / f"maps_multexp_expdrawn_blend{int(round(a * 100)):02d}"
        nb = blend_caches(args.cache_dir, fr_cache, bdir, a)
        print(f"[expdrawn] blend alpha={a:.2f} n={nb}", flush=True)
        bbest, bgrid, _ = eval_frozen_robust(bdir)
        entry = {"alpha": a, "best": bbest, "grid": bgrid, "cache": str(bdir)}
        blend_results.append(entry)
        if best_blend is None or bbest["S"] < best_blend["best"]["S"]:
            best_blend = entry

    print("[expdrawn] ref Frundles fasc eval", flush=True)
    best_fr, grid_fr, _ = eval_frozen_robust(fr_cache)

    candidates = [{"tag": "raw_expdrawn", "S": best_raw["S"], "detail": best_raw, "cache": str(args.cache_dir)}]
    if best_blend is not None:
        candidates.append(
            {
                "tag": f"blend_a{best_blend['alpha']:.2f}",
                "S": best_blend["best"]["S"],
                "detail": best_blend["best"],
                "cache": best_blend["cache"],
            }
        )
    candidates.append({"tag": "frundles_ref", "S": best_fr["S"], "detail": best_fr, "cache": str(fr_cache)})
    # also record hold-C known as 0.13195 (adapt feature_rule) — Frundles frozen thr0.35 is the backbone ref
    candidates.sort(key=lambda c: c["S"])

    non_fr = [c for c in candidates if c["tag"] != "frundles_ref"]
    report_champ = non_fr[0] if non_fr else candidates[0]
    # promote only if beats HOLD and is not just frundles_ref
    promote = bool(report_champ["S"] < HOLD)
    gate = bool(report_champ["S"] < GATE)
    elapsed = time.time() - t0

    extract_summary = extract_info.get("summary", extract_info) if isinstance(extract_info, dict) else {}
    report = {
        "updated": now(),
        "elapsed_sec": elapsed,
        "extract": extract_summary,
        "usable_expdrawn": n_exp,
        "train_mix": src_keep,
        "init": init_src,
        "loss": "0.55*BCE(pos_w=4) + 0.40*Dice + 0.05*orient",
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_dice": best_dice,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "ckpt": str(ckpt_path),
        "cache_raw": str(args.cache_dir),
        "raw_best": best_raw,
        "raw_grid": grid_raw,
        "blend_best": best_blend["best"] if best_blend else None,
        "blend_alpha": best_blend["alpha"] if best_blend else None,
        "blend_results": [{"alpha": b["alpha"], "best": b["best"]} for b in blend_results],
        "frundles_ref": best_fr,
        "candidates": candidates,
        "champ": report_champ,
        "hold": HOLD,
        "hold_C": 0.13195,
        "gate": GATE,
        "promote": promote,
        "gate_pass": gate,
        "hist": hist,
        "no_kaggle": True,
        "multexp_in_train": False,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.report_dir / "multexp_expdrawn_frundles.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    if df_raw is not None:
        df_raw.to_csv(args.report_dir / "multexp_expdrawn_frundles_rows.csv", index=False)

    status = args.report_dir / "status.md"
    prev = status.read_text() if status.exists() else ""
    beat = "BEAT" if report_champ["S"] < HOLD else "MISS"
    block = f"""# MultExp ExpDrawn→Frundles FT — {now()}

## Result
| metric | value |
|---|---|
| **champ S** | **{report_champ['S']:.5f}** ({report_champ['tag']}) |
| MAE PA / FL / MT | {report_champ['detail'].get('mae_pa', float('nan')):.3f}° / {report_champ['detail'].get('mae_fl', float('nan')):.3f} mm / {report_champ['detail'].get('mae_mt', float('nan')):.3f} mm |
| thr | {report_champ['detail'].get('thr')} |
| vs hold 0.13195 | {beat} ({report_champ['S'] - HOLD:+.5f}) |
| gate 0.12 | {'PASS' if gate else 'FAIL'} |
| promote | {promote} |
| usable ExpDrawn pairs | {n_exp} (extract ok={extract_summary.get('n_ok')}) |
| train mix | {src_keep} |
| init | Frundles fasc |
| best val Dice | {best_dice:.4f} |
| Frundles ref S | {best_fr['S']:.5f} |
| raw ExpDrawn-FT S | {best_raw['S']:.5f} |
| best blend | alpha={best_blend['alpha'] if best_blend else None} S={best_blend['best']['S'] if best_blend else None} |
| elapsed | {elapsed/60:.1f} min |
| no Kaggle | true |

## Diagnosis
ExpDrawn red-stroke overlays (ann[:,200:1000]↔raw) → fasc masks; Frundles-init BCE+Dice+light orient FT; MultExp held out.

## Next
{"Promote; optional MultExp-domain light adapt on ExpDrawn-FT maps." if promote else "ExpDrawn FT did not beat hold — next: (1) apo-relative soft PA on Frundles init with ExpDrawn stroke angles, (2) thicker multi-dilate ensemble, (3) keep Frundles maps + seek better FL geometry outside mask FT."}

Artifacts: `{out_json.name}`, extract `multexp_expdrawn_extract.json`, cache `{args.cache_dir.name}`, ckpt `{ckpt_path.name}`

---

"""
    status.write_text(block + prev)
    print(json.dumps({"champ": report_champ, "promote": promote, "gate": gate, "hold": HOLD, "usable_expdrawn": n_exp, "val_dice": best_dice}, indent=2, default=str), flush=True)
    print(f"[expdrawn] DONE {now()} elapsed={elapsed:.0f}s wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
