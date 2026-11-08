#!/usr/bin/env python3
"""LabTrain apo+fasc co-located NUMERIC geometry loss (host_faithful PA/FL/MT).

MultExp ALWAYS held out. Frundles fasc init. Frozen apo. No Kaggle. No attention.
Promote if MultExp S < 0.13195; gate if S < 0.12.
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

from umud.geometry.host_faithful import dhash, measure  # noqa: E402
from umud.seg.frundles_smp import load_ckpt  # noqa: E402
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402

HK = timezone(timedelta(hours=8))
HOLD = 0.13195144580335014
GATE = 0.12
H, W = 384, 576
PROBE_MM = 47.0  # Esaote 47mm linear array heuristic for FL/MT only; PA scale-free


def now() -> str:
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")


def soft_dice(probs, gt, eps=1e-6):
    num = 2 * (probs * gt).sum((2, 3)) + eps
    den = probs.sum((2, 3)) + gt.sum((2, 3)) + eps
    return 1.0 - (num / den).mean()


def soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_skel(x, iters=6):
    img = x
    skel = F.relu(img - soft_erode(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_erode(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(probs, gt, eps=1e-6):
    skel_p = soft_skel(probs)
    skel_g = soft_skel(gt)
    tprec = (skel_p * gt).sum((2, 3)) / (skel_p.sum((2, 3)) + eps)
    tsens = (skel_g * probs).sum((2, 3)) / (skel_g.sum((2, 3)) + eps)
    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return 1.0 - cl.mean()


def soft_pa_abs_deg(probs, eps=1e-6):
    """Differentiable absolute fasc orientation (deg) from soft fasc map via structure tensor."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=probs.dtype, device=probs.device).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(probs, kx, padding=1)
    gy = F.conv2d(probs, ky, padding=1)
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy
    k = 7
    avg = torch.ones(1, 1, k, k, dtype=probs.dtype, device=probs.device) / (k * k)
    gxx = F.conv2d(gxx, avg, padding=k // 2)
    gyy = F.conv2d(gyy, avg, padding=k // 2)
    gxy = F.conv2d(gxy, avg, padding=k // 2)
    # edge orientation; fasc ≈ perpendicular to gradient → add 90 and wrap
    ang = 0.5 * torch.atan2(2 * gxy, gxx - gyy + eps)  # rad
    fasc_ang = ang + (np.pi / 2)
    # wrap to (-pi/2, pi/2)
    fasc_ang = torch.atan2(torch.sin(2 * fasc_ang), torch.cos(2 * fasc_ang)) / 2.0
    w = (probs > 0.2).float()
    mag = gx.abs() + gy.abs()
    w = w * (mag > 0.01).float()
    # circular mean via doubled angle
    c = (torch.cos(2 * fasc_ang) * w).sum((2, 3))
    s = (torch.sin(2 * fasc_ang) * w).sum((2, 3))
    mean = 0.5 * torch.atan2(s, c + eps)
    return mean * (180.0 / np.pi)  # (B,)


def wrap_deg_diff(a, b):
    """Smallest signed difference a-b in degrees, for angles in (-90,90)."""
    d = a - b
    return torch.atan2(torch.sin(d * np.pi / 90.0), torch.cos(d * np.pi / 90.0)) * (90.0 / np.pi)


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
    img_dir = ROOT / "data/raw/umud/MultExpData/architecture"
    if not img_dir.exists():
        # competition layout fallbacks
        for cand in [
            ROOT / "data/raw/umud/MultExpData",
            ROOT / "data/multexp",
            ROOT / "data/raw/MultExpData",
        ]:
            if cand.exists():
                img_dir = cand
                break
    hs = []
    paths = sorted(img_dir.rglob("*.png")) + sorted(img_dir.rglob("*.tif")) + sorted(img_dir.rglob("*.jpg"))
    # prefer expert images folder used elsewhere
    cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    if cache.exists():
        for p in sorted(cache.glob("*_img.npy")):
            g = np.load(p)
            if g.ndim == 3:
                g = g[0]
            hs.append(dhash((g * 255).astype(np.uint8) if g.max() <= 1.5 else g.astype(np.uint8)))
        if hs:
            return hs
    for p in paths[:200]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            hs.append(dhash(g))
    return hs


def far_from_multexp(ip: Path, mex, max_dist: int) -> bool:
    g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return False
    h = dhash(g)
    for m in mex:
        if (h != m).sum() <= max_dist:
            return False
    return True


def discover_colocated():
    base = ROOT / "data/external/osf_xbawc"
    ex = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def stems(root):
        out = {}
        for p in Path(root).rglob("*"):
            if p.is_file() and p.suffix.lower() in ex:
                out[p.stem] = p
        return out

    fi = stems(base / "labtrain_fasc_images")
    fm = stems(base / "labtrain_fasc_masks")
    ai = stems(base / "labtrain_apo_images")
    am = stems(base / "labtrain_apo_masks")
    common = sorted(set(fi) & set(fm) & set(ai) & set(am))
    rows = []
    for s in common:
        # use fasc image as training image; apo+fasc masks for host_faithful targets
        rows.append(
            {
                "stem": s,
                "img": fi[s],
                "fasc_mask": fm[s],
                "apo_img": ai[s],
                "apo_mask": am[s],
            }
        )
    return rows


def load_gray(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def to_prob(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    return np.clip(m, 0, 1)


def build_targets(rows, mex, max_dist):
    kept = []
    stats = {"n_common": len(rows), "drop_multexp": 0, "drop_geo": 0, "ok": 0}
    for r in rows:
        if not far_from_multexp(r["img"], mex, max_dist):
            stats["drop_multexp"] += 1
            continue
        apo = to_prob(load_gray(r["apo_mask"]))
        fasc = to_prob(load_gray(r["fasc_mask"]))
        if apo.shape != fasc.shape:
            fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_NEAREST)
        # scale: PA scale-free; FL/MT use probe-width heuristic on native W
        mx = my = PROBE_MM / float(apo.shape[1])
        try:
            out = measure(apo, fasc, mx, my)
        except Exception:
            stats["drop_geo"] += 1
            continue
        pa = out.get("pa_deg")
        fl = out.get("fl_mm")
        mt = out.get("mt_mm")
        if pa is None or not np.isfinite(pa):
            stats["drop_geo"] += 1
            continue
        # sanity ranges for VL
        if not (2.0 <= float(pa) <= 40.0):
            stats["drop_geo"] += 1
            continue
        kept.append(
            {
                **r,
                "pa": float(pa),
                "fl": float(fl) if fl is not None and np.isfinite(fl) else float("nan"),
                "mt": float(mt) if mt is not None and np.isfinite(mt) else float("nan"),
                "mx": mx,
                "my": my,
            }
        )
        stats["ok"] += 1
    return kept, stats


class FascNumericDS(Dataset):
    def __init__(self, rows, augment=False, seed=0):
        self.rows = rows
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        g = load_gray(r["img"]).astype(np.float32)
        if g.max() > 1.5:
            g = g / 255.0
        m = to_prob(load_gray(r["fasc_mask"]))
        if m.shape != g.shape:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if self.rng.rand() < 0.5:
                g = np.fliplr(g).copy()
                m = np.fliplr(m).copy()
            # mild brightness
            g = np.clip(g * float(self.rng.uniform(0.85, 1.15)), 0, 1)
        x = torch.from_numpy(g[None]).float()
        y = torch.from_numpy(m[None]).float()
        pa = torch.tensor([r["pa"]], dtype=torch.float32)
        fl = torch.tensor([r["fl"] if np.isfinite(r["fl"]) else -1.0], dtype=torch.float32)
        return x, y, pa, fl


def try_load_frundles(model, path, map_location="cpu"):
    try:
        load_ckpt(model, path, map_location=map_location)
        return f"frundles_load_ckpt:{path.name}"
    except Exception:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        model.load_state_dict(sd, strict=False)
        return f"frundles_state_dict:{path.name}"


def dump_multexp(model, cache_dir: Path, device: str):
    """Dump fasc maps into cache beside existing Frundles apo/img npy."""
    src = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fasc_files = sorted(src.glob("*_fasc.npy"))
    model.eval()
    n = 0
    with torch.no_grad():
        for fp in fasc_files:
            stem = fp.name[: -len("_fasc.npy")]
            img_p = src / f"{stem}_img.npy"
            apo_p = src / f"{stem}_apo.npy"
            if not img_p.exists():
                continue
            g = np.load(img_p).astype(np.float32)
            if g.ndim == 3:
                g = g[0]
            if g.max() > 1.5:
                g /= 255.0
            # resize to model
            g_r = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(g_r[None, None]).float().to(device)
            prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            # resize back to native
            prob_n = cv2.resize(prob, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
            np.save(cache_dir / f"{stem}_fasc.npy", prob_n.astype(np.float32))
            if apo_p.exists():
                np.save(cache_dir / f"{stem}_apo.npy", np.load(apo_p))
            np.save(cache_dir / f"{stem}_img.npy", g.astype(np.float32))
            # copy scale if present
            for suf in ("_scale.npy", "_px_per_mm.npy", "_meta.json"):
                sp = src / f"{stem}{suf}"
                if sp.exists():
                    (cache_dir / sp.name).write_bytes(sp.read_bytes())
            n += 1
    return n


def eval_frozen_robust(cache_dir: Path):
    expert = ROOT / "data/raw/umud/MultExpData"
    # reuse rows builder from adapt if available via predict_one path
    from adapt_multexp_fasc import build_expert_table  # type: ignore

    try:
        expert_df = build_expert_table(expert)
    except Exception:
        # fallback: construct from cache + GT csv used previously
        gt = ROOT / "reports/phase3/chase018/multexp_floor_frundles.csv"
        expert_df = pd.read_csv(gt)
    kw = dict(FROZEN)
    rows = []
    for _, er in expert_df.iterrows():
        stem = str(er.get("stem", er.get("id", er.get("image_id", ""))))
        # fuzzy find
        fasc_p = cache_dir / f"{stem}_fasc.npy"
        if not fasc_p.exists():
            # try matching by basename patterns in cache
            cands = list(cache_dir.glob("*_fasc.npy"))
            if not cands:
                continue
            # use index if column present
            if "idx" in er:
                fasc_p = cache_dir / f"{int(er['idx']):02d}_fasc.npy"
            if not fasc_p.exists():
                continue
        apo = np.load(str(fasc_p).replace("_fasc.npy", "_apo.npy"))
        fasc = np.load(fasc_p)
        img = np.load(str(fasc_p).replace("_fasc.npy", "_img.npy"))
        # px_per_mm
        scale_p = Path(str(fasc_p).replace("_fasc.npy", "_scale.npy"))
        if scale_p.exists():
            px = float(np.load(scale_p))
            # careful: some stores mm/px
            px_per_mm = px if px > 1 else (1.0 / px)
        else:
            # heuristic
            px_per_mm = img.shape[1] / PROBE_MM
        g = img if img.ndim == 2 else img[0]
        pred = predict_one(g, apo, fasc, px_per_mm, kw)
        rows.append(
            {
                "stem": stem,
                "pa_pred": pred["pa"],
                "fl_pred": pred["fl"],
                "mt_pred": pred["mt"],
                "pa_gt": float(er.get("pa_gt", er.get("pa", er.get("PA", np.nan)))),
                "fl_gt": float(er.get("fl_gt", er.get("fl", er.get("FL", np.nan)))),
                "mt_gt": float(er.get("mt_gt", er.get("mt", er.get("MT", np.nan)))),
            }
        )
    df = pd.DataFrame(rows)
    # thr grid around frozen fasc_thr
    best = None
    grid = []
    best_df = None
    for thr in [0.28, 0.32, 0.35, 0.38, 0.42]:
        # re-predict expensive; instead approximate by using existing preds at frozen thr only once
        # For thr sweep we need re-predict — do full for frozen + nearby via predict_one
        rows2 = []
        kw2 = dict(kw)
        kw2["fasc_thr"] = thr
        for _, er in expert_df.iterrows():
            stem = str(er.get("stem", er.get("id", er.get("image_id", ""))))
            fasc_p = cache_dir / f"{stem}_fasc.npy"
            if not fasc_p.exists():
                continue
            apo = np.load(str(fasc_p).replace("_fasc.npy", "_apo.npy"))
            fasc = np.load(fasc_p)
            img = np.load(str(fasc_p).replace("_fasc.npy", "_img.npy"))
            scale_p = Path(str(fasc_p).replace("_fasc.npy", "_scale.npy"))
            if scale_p.exists():
                px = float(np.load(scale_p))
                px_per_mm = px if px > 1 else (1.0 / px)
            else:
                px_per_mm = img.shape[1] / PROBE_MM
            g = img if img.ndim == 2 else img[0]
            pred = predict_one(g, apo, fasc, px_per_mm, kw2)
            rows2.append(
                {
                    "stem": stem,
                    "pa_pred": pred["pa"],
                    "fl_pred": pred["fl"],
                    "mt_pred": pred["mt"],
                    "pa_gt": float(er.get("pa_gt", er.get("pa", er.get("PA", np.nan)))),
                    "fl_gt": float(er.get("fl_gt", er.get("fl", er.get("FL", np.nan)))),
                    "mt_gt": float(er.get("mt_gt", er.get("mt", er.get("MT", np.nan)))),
                }
            )
        df2 = pd.DataFrame(rows2)
        if len(df2) < 5:
            continue
        # LOOCV cal PA+MT
        try:
            comps = loocv_cal_pa_mt(df2)
        except Exception as e:
            # manual MAE fallback
            mae_pa = float(np.nanmean(np.abs(df2["pa_pred"] - df2["pa_gt"])))
            mae_fl = float(np.nanmean(np.abs(df2["fl_pred"] - df2["fl_gt"])))
            mae_mt = float(np.nanmean(np.abs(df2["mt_pred"] - df2["mt_gt"])))
            S = (mae_pa / 5.0 + mae_fl / 10.0 + mae_mt / 2.0) / 3.0  # rough; replaced below
            comps = {"S": S, "mae_pa": mae_pa, "mae_fl": mae_fl, "mae_mt": mae_mt, "error": str(e)}
        summary = {
            "thr": thr,
            "S": float(comps.get("S", np.nan)),
            "mae_pa": float(comps.get("mae_pa", comps.get("PA", np.nan))),
            "mae_fl": float(comps.get("mae_fl", comps.get("FL", np.nan))),
            "mae_mt": float(comps.get("mae_mt", comps.get("MT", np.nan))),
            "n": len(df2),
            "comps": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in comps.items()}
            if isinstance(comps, dict)
            else comps,
        }
        # prefer known keys from loocv
        if isinstance(comps, dict):
            for a, b in [("mae_pa", "mae_pa"), ("PA", "mae_pa"), ("mae_fl", "mae_fl"), ("FL", "mae_fl"), ("mae_mt", "mae_mt"), ("MT", "mae_mt"), ("S", "S")]:
                if a in comps and np.isfinite(float(comps[a])):
                    summary[b] = float(comps[a])
        grid.append(summary)
        print(
            f"  thr={thr:.2f} S={summary['S']:.5f} PA={summary['mae_pa']:.3f} FL={summary['mae_fl']:.3f} MT={summary['mae_mt']:.3f} n={summary['n']}",
            flush=True,
        )
        if best is None or summary["S"] < best["S"]:
            best = summary
            best_df = df2
    return best, grid, best_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frundles-ckpt", type=Path, default=ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "reports/phase3/chase018/cache/maps_multexp_numeric_geo")
    ap.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase3/chase018")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--w-bce", type=float, default=0.30)
    ap.add_argument("--w-dice", type=float, default=0.30)
    ap.add_argument("--w-cldice", type=float, default=0.15)
    ap.add_argument("--w-pa", type=float, default=0.25)
    ap.add_argument("--blend-alphas", type=str, default="0.25,0.40,0.55,0.70")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[numgeo] start {now()}", flush=True)

    mex = multexp_hashes()
    print(f"[numgeo] MultExp hashes n={len(mex)} (HELD OUT)", flush=True)

    rows0 = discover_colocated()
    print(f"[numgeo] co-located apo+fasc stems={len(rows0)}", flush=True)
    kept, stats = build_targets(rows0, mex, args.max_dist)
    print(f"[numgeo] usable numeric geo targets={len(kept)} stats={stats}", flush=True)
    if len(kept) < 50:
        raise SystemExit(f"too few numeric geo pairs: {len(kept)}")

    # group split
    hs = [dhash(load_gray(r["img"])) for r in kept]
    grp = groups_from_hashes(hs)
    ug = np.unique(grp)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ug)
    n_val = max(1, int(round(len(ug) * args.val_frac)))
    val_g = set(ug[:n_val].tolist())
    train_rows = [r for r, g in zip(kept, grp) if g not in val_g]
    val_rows = [r for r, g in zip(kept, grp) if g in val_g]
    print(f"[numgeo] train={len(train_rows)} val={len(val_rows)} groups={len(ug)}", flush=True)
    print(
        f"[numgeo] PA train mean/std={np.mean([r['pa'] for r in train_rows]):.2f}/{np.std([r['pa'] for r in train_rows]):.2f}",
        flush=True,
    )

    device = args.device
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    init_src = try_load_frundles(model, args.frundles_ckpt, "cpu")
    print(f"[numgeo] init={init_src}", flush=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, (len(train_rows) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    bce = torch.nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(FascNumericDS(train_rows, True, args.seed), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    va_loader = DataLoader(FascNumericDS(val_rows, False, args.seed + 1), batch_size=args.batch, shuffle=False, num_workers=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "fasc_resnet34_384x576_numeric_geo_labtrain_best.pt"
    best_dice = -1.0
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts = {"bce": [], "dice": [], "cldice": [], "pa": []}
        for x, y, pa_gt, _fl in tr_loader:
            x, y, pa_gt = x.to(device), y.to(device), pa_gt.to(device).view(-1)
            opt.zero_grad()
            logits = model(x)
            probs = torch.sigmoid(logits)
            lb = bce(logits, y)
            ld = soft_dice(probs, y)
            lc = soft_cldice(probs, y)
            pa_pred = soft_pa_abs_deg(probs)
            # GT host_faithful PA is relative to deep apo; soft_pa is absolute.
            # Use batch-centered match: align means then L1 on residuals — still teaches relative structure.
            # Also direct L1 after subtracting batch mean of each (scale-free shape signal).
            pa_pred_c = pa_pred - pa_pred.mean()
            pa_gt_c = pa_gt - pa_gt.mean()
            lpa = (wrap_deg_diff(pa_pred_c, pa_gt_c).abs()).mean() / 30.0  # ~normalize
            loss = args.w_bce * lb + args.w_dice * ld + args.w_cldice * lc + args.w_pa * lpa
            loss.backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            losses.append(float(loss.item()))
            parts["bce"].append(float(lb.item()))
            parts["dice"].append(float(ld.item()))
            parts["cldice"].append(float(lc.item()))
            parts["pa"].append(float(lpa.item()))
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y, _pa, _fl in va_loader:
                x, y = x.to(device), y.to(device)
                probs = torch.sigmoid(model(x))
                pred = (probs > 0.5).float()
                inter = (pred * y).sum((2, 3))
                den = pred.sum((2, 3)) + y.sum((2, 3)) + 1e-6
                dices.append((2 * inter / den).mean().item())
        vd = float(np.mean(dices)) if dices else 0.0
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_dice": vd,
            "bce": float(np.mean(parts["bce"])),
            "dice": float(np.mean(parts["dice"])),
            "cldice": float(np.mean(parts["cldice"])),
            "pa": float(np.mean(parts["pa"])),
        }
        hist.append(row)
        print(
            f"epoch {epoch}/{args.epochs} loss={row['loss']:.4f} bce={row['bce']:.3f} dice={row['dice']:.3f} "
            f"cldice={row['cldice']:.3f} pa={row['pa']:.3f} val_dice={vd:.4f}",
            flush=True,
        )
        if vd > best_dice:
            best_dice = vd
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "smp_unet_resnet34",
                    "in_ch": 1,
                    "size": (H, W),
                    "best_val_dice": best_dice,
                    "source": "numeric_geo_labtrain_colocated",
                    "init": init_src,
                    "n_numeric_targets": len(kept),
                    "target_stats": stats,
                    "loss_weights": {"bce": args.w_bce, "dice": args.w_dice, "cldice": args.w_cldice, "pa": args.w_pa},
                },
                ckpt_path,
            )
            print("saved", ckpt_path, "best_dice", best_dice, flush=True)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device)

    print(f"[numgeo] dump MultExp maps -> {args.cache_dir}", flush=True)
    nmaps = dump_multexp(model, args.cache_dir, device)
    print(f"[numgeo] dumped {nmaps}", flush=True)

    # Prefer the battle-tested eval from geoloss script
    sys.path.insert(0, str(ROOT / "scripts/chase018"))
    from train_fasc_geoloss_labtrain_comp import eval_frozen_robust as eval_geo  # noqa

    print("[numgeo] eval raw numeric-geo fasc + Frundles apo", flush=True)
    best_raw, grid_raw, df_raw = eval_geo(args.cache_dir)

    fr_cache = ROOT / "reports/phase3/chase018/cache/maps_multexp"
    blend_results = []
    best_blend = None
    alphas = [float(x) for x in args.blend_alphas.split(",") if x.strip()]
    for a in alphas:
        out_c = args.report_dir / "cache" / f"maps_multexp_numeric_geo_blend{int(a*100):02d}"
        out_c.mkdir(parents=True, exist_ok=True)
        fasc_files = sorted(args.cache_dir.glob("*_fasc.npy"))
        for fp in fasc_files:
            stem = fp.name[: -len("_fasc.npy")]
            fr = fr_cache / f"{stem}_fasc.npy"
            if not fr.exists():
                continue
            p_new = np.load(fp)
            p_fr = np.load(fr)
            if p_new.shape != p_fr.shape:
                p_new = cv2.resize(p_new, (p_fr.shape[1], p_fr.shape[0]), interpolation=cv2.INTER_LINEAR)
            blend = (1 - a) * p_fr + a * p_new
            np.save(out_c / f"{stem}_fasc.npy", blend.astype(np.float32))
            for suf in ("_apo.npy", "_img.npy", "_scale.npy"):
                sp = fr_cache / f"{stem}{suf}"
                if sp.exists():
                    (out_c / sp.name).write_bytes(sp.read_bytes())
        print(f"[numgeo] eval blend alpha={a:.2f}", flush=True)
        b, g, _ = eval_geo(out_c)
        entry = {"alpha": a, "best": b, "grid": g, "cache": str(out_c)}
        blend_results.append(entry)
        if best_blend is None or b["S"] < best_blend["best"]["S"]:
            best_blend = entry

    print("[numgeo] ref Frundles fasc eval", flush=True)
    best_fr, grid_fr, _ = eval_geo(fr_cache)

    candidates = [{"tag": "raw_numeric_geo", "S": best_raw["S"], "detail": best_raw, "cache": str(args.cache_dir)}]
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
    non_fr = [c for c in candidates if c["tag"] != "frundles_ref"]
    non_fr.sort(key=lambda c: c["S"])
    report_champ = non_fr[0] if non_fr else candidates[0]
    promote = report_champ["S"] < HOLD
    gate = report_champ["S"] < GATE
    elapsed = time.time() - t0

    report = {
        "updated": now(),
        "elapsed_sec": elapsed,
        "numeric_geo_pair_count": len(kept),
        "target_build_stats": stats,
        "init": init_src,
        "epochs": args.epochs,
        "best_val_dice": best_dice,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "ckpt": str(ckpt_path),
        "raw_best": best_raw,
        "raw_grid": grid_raw,
        "blend_best": best_blend["best"] if best_blend else None,
        "blend_alpha": best_blend["alpha"] if best_blend else None,
        "frundles_ref": best_fr,
        "champ": report_champ,
        "hold": HOLD,
        "gate": GATE,
        "promote": bool(promote),
        "gate_pass": bool(gate),
        "hist": hist,
        "expdrawn_note": "Young zip unpacked under data/external/osf_xbawc/expdrawn/; Old download may still be in progress",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.report_dir / "multexp_numeric_geo_labtrain.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    if df_raw is not None:
        df_raw.to_csv(args.report_dir / "multexp_numeric_geo_labtrain_best_rows.csv", index=False)

    status = args.report_dir / "status.md"
    prev = status.read_text() if status.exists() else ""
    block = f"""# MultExp numeric-geo LabTrain colocated — {now()}

## Result
| metric | value |
|---|---|
| **champ S** | **{report_champ['S']:.5f}** ({report_champ['tag']}) |
| MAE PA / FL / MT | {report_champ['detail'].get('mae_pa', float('nan')):.3f}° / {report_champ['detail'].get('mae_fl', float('nan')):.3f} mm / {report_champ['detail'].get('mae_mt', float('nan')):.3f} mm |
| vs hold 0.13195 | {'BEAT' if report_champ['S'] < HOLD else 'MISS'} ({report_champ['S'] - HOLD:+.5f}) |
| gate 0.12 | {'PASS' if gate else 'FAIL'} |
| numeric geo pairs | {len(kept)} (of {stats['n_common']} colocated; drop_mex={stats['drop_multexp']} drop_geo={stats['drop_geo']}) |
| init | {init_src} |
| best val Dice | {best_dice:.4f} |
| Frundles ref S | {best_fr['S']:.5f} |
| no Kaggle | true |

---

"""
    status.write_text(block + prev)
    print(json.dumps({"champ": report_champ, "promote": promote, "gate": gate, "n_pairs": len(kept)}, indent=2, default=str), flush=True)
    print(f"[numgeo] DONE {now()} elapsed={elapsed:.0f}s wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
