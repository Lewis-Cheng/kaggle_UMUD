#!/usr/bin/env python3
"""MultExp-held-out Frundles fasc FT with crop/scale + tiny geometry-consistent rot.

Reuses train_fasc_frundles_recipe.py (crop/scale + ±8 deg) and infer_multexp_maps.py.
Train data: LabTrain fasc + ExpDrawn overlays, dHash-filtered away from MultExp.
Eval: frozen hough_tex0.4_thr035_apo60 / fl_factor=0.9 + cheap nested sin_w.
Promote only if honest S < 0.13449. Gate S<0.12. No Kaggle.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from umud.geometry.host_faithful import dhash  # noqa: E402
from adapt_multexp_fasc import FROZEN, loocv_cal_pa_mt, predict_one  # noqa: E402

HK = timezone(timedelta(hours=8))
HOLD = 0.13449
GATE = 0.12
SIN_WS = [0.25, 0.28, 0.30, 0.35, 0.40]


def now() -> str:
    return datetime.now(HK).strftime("%Y-%m-%d %H:%M:%S %Z")


def pick_gpu() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as e:
        print(f"[crop_aug] nvidia-smi failed ({e}); default cuda:0", flush=True)
        return "cuda:0"
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, used, total, util = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        free = total - used
        rows.append((free, -util, idx, used, total, util))
        print(
            f"[crop_aug] gpu{idx} used={used:.0f}/{total:.0f} MiB free={free:.0f} util={util:.0f}%",
            flush=True,
        )
    if not rows:
        return "cuda:0"
    rows.sort(reverse=True)
    best = rows[0]
    if best[0] < 6000:
        print("[crop_aug] WARNING: less than 6 GiB free on best GPU", flush=True)
    return f"cuda:{best[2]}"


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


def stage_pairs(pairs, img_dir: Path, mask_dir: Path) -> int:
    if img_dir.exists():
        shutil.rmtree(img_dir)
    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem, ip, mp, _src in pairs:
        ext_i = ip.suffix.lower() if ip.suffix else ".png"
        ext_m = mp.suffix.lower() if mp.suffix else ".png"
        dst_i = img_dir / f"{stem}{ext_i}"
        dst_m = mask_dir / f"{stem}{ext_m}"
        if dst_i.exists() or dst_m.exists():
            stem = f"{_src}_{stem}"
            dst_i = img_dir / f"{stem}{ext_i}"
            dst_m = mask_dir / f"{stem}{ext_m}"
        os.symlink(ip.resolve(), dst_i)
        os.symlink(mp.resolve(), dst_m)
        n += 1
    return n


def collect_pairs(mix_labtrain: bool, mix_expdrawn: bool):
    mex = multexp_hashes()
    print(f"[crop_aug] MultExp hashes n={len(mex)} HELD OUT", flush=True)
    pairs = []
    dropped = {"labtrain": 0, "expdrawn": 0}
    if mix_labtrain:
        lab = discover_pairs(
            ROOT / "data/external/osf_xbawc/labtrain_fasc_images",
            ROOT / "data/external/osf_xbawc/labtrain_fasc_masks",
        )
        for s, ip, mp in lab:
            if far_from_multexp(ip, mex, 8):
                pairs.append((s, ip, mp, "labtrain"))
            else:
                dropped["labtrain"] += 1
    if mix_expdrawn:
        exp = discover_pairs(
            ROOT / "data/external/osf_xbawc/expdrawn_images",
            ROOT / "data/external/osf_xbawc/expdrawn_masks",
        )
        seen = {s for s, *_ in pairs}
        for s, ip, mp in exp:
            if s in seen:
                s2 = f"exp_{s}"
            else:
                s2 = s
            if far_from_multexp(ip, mex, 8):
                pairs.append((s2, ip, mp, "expdrawn"))
            else:
                dropped["expdrawn"] += 1
    src_keep = {}
    for _, _, _, src in pairs:
        src_keep[src] = src_keep.get(src, 0) + 1
    print(f"[crop_aug] kept={src_keep} dropped_near_multexp={dropped}", flush=True)
    return pairs, src_keep, dropped


def mae_of(mae, keys):
    if isinstance(mae, dict):
        for k in keys:
            if k in mae:
                return float(mae[k])
    return float("nan")


def score_df(df):
    out = loocv_cal_pa_mt(df)
    comps, mae = out[0], out[1]
    S = float(comps["S"]) if isinstance(comps, dict) else float(comps)
    return {
        "S": S,
        "mae_pa": mae_of(mae, ["pa_deg", "pa", "mae_pa"]),
        "mae_fl": mae_of(mae, ["fl_mm", "fl", "mae_fl"]),
        "mae_mt": mae_of(mae, ["mt_mm", "mt", "mae_mt"]),
        "n": int(len(df)),
    }


def predict_rows(cache_dir: Path, kw: dict):
    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
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
    return pd.DataFrame(rows)


def eval_frozen(cache_dir: Path, tag: str):
    kw = dict(FROZEN)
    df = predict_rows(cache_dir, kw)
    summary = score_df(df)
    summary.update({"tag": tag, "recipe": FROZEN["name"], "fl_sin_w": float(kw["fl_sin_w"]), "cache": str(cache_dir)})
    print(
        f"  [{tag}] frozen S={summary['S']:.5f} PA={summary['mae_pa']:.3f} "
        f"FL={summary['mae_fl']:.3f} MT={summary['mae_mt']:.3f} n={summary['n']}",
        flush=True,
    )
    return summary, df


def nested_sin_w(cache_dir: Path, tag: str):
    by_w = {}
    for w in SIN_WS:
        kw = dict(FROZEN)
        kw["fl_sin_w"] = float(w)
        kw["name"] = FROZEN["name"] + f"_sin{w:.2f}"
        df = predict_rows(cache_dir, kw)
        by_w[w] = df.reset_index(drop=True)
        s = score_df(df)
        print(f"  [{tag}] sin_w={w:.2f} (full, diagnostic) S={s['S']:.5f}", flush=True)
    names = list(by_w[SIN_WS[0]]["image_file"])
    n = len(names)
    picked_rows = []
    picked_w = []
    for i in range(n):
        best_w, best_s = None, 1e9
        for w in SIN_WS:
            df_tr = by_w[w].drop(index=i)
            s = score_df(df_tr)["S"]
            if s < best_s:
                best_s, best_w = s, w
        row = by_w[best_w].iloc[i].to_dict()
        picked_rows.append(row)
        picked_w.append(float(best_w))
    df = pd.DataFrame(picked_rows)
    summary = score_df(df)
    from collections import Counter

    summary.update(
        {
            "tag": f"{tag}_nested_sinw",
            "recipe": FROZEN["name"] + "_nested_sinw",
            "sin_w_grid": SIN_WS,
            "picked_counts": dict(Counter(picked_w)),
            "cache": str(cache_dir),
        }
    )
    print(
        f"  [{tag}] NESTED sin_w S={summary['S']:.5f} PA={summary['mae_pa']:.3f} "
        f"FL={summary['mae_fl']:.3f} MT={summary['mae_mt']:.3f} picks={summary['picked_counts']}",
        flush=True,
    )
    return summary, df


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


def write_status(report_dir: Path, body: str, prepend_main: bool = False):
    fp = report_dir / "multexp_crop_scale_aug_status.md"
    fp.write_text(body)
    if prepend_main:
        main = report_dir / "status.md"
        prev = main.read_text() if main.exists() else ""
        main.write_text(body + "\n---\n\n" + prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="")
    ap.add_argument("--crop-min", type=float, default=0.6)
    ap.add_argument("--rot-deg", type=float, default=8.0)
    ap.add_argument("--rot-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-labtrain", action="store_true")
    ap.add_argument("--no-expdrawn", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    report_dir = ROOT / "reports/phase3/chase018"
    report_dir.mkdir(parents=True, exist_ok=True)
    stage_img = ROOT / "reports/phase3/chase018/cache/crop_aug_stage/images"
    stage_msk = ROOT / "reports/phase3/chase018/cache/crop_aug_stage/masks"
    ckpt_dir = report_dir / "seg_ckpts"
    tag = "fasc_resnet34_384x576_crop_scale_aug"
    ckpt_path = ckpt_dir / f"{tag}_best.pt"
    cache_dir = report_dir / "cache" / "maps_multexp_crop_scale_aug"
    fr_cache = report_dir / "cache" / "maps_multexp"
    init_ckpt = ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt"
    py = str(ROOT / ".venv/bin/python")

    device = args.device or pick_gpu()
    print(f"[crop_aug] start {now()} device={device}", flush=True)
    write_status(
        report_dir,
        f"# MultExp crop/scale + tiny-rot Frundles FT — running\n\n"
        f"Started {now()}. Device {device}. MultExp held out. No Kaggle.\n",
    )

    pairs, src_keep, dropped = collect_pairs(not args.no_labtrain, not args.no_expdrawn)
    if len(pairs) < 30:
        raise SystemExit(f"too few pairs: {len(pairs)}")
    n_stage = stage_pairs(pairs, stage_img, stage_msk)
    print(f"[crop_aug] staged {n_stage} pairs under {stage_img}", flush=True)

    train_hist = None
    if not args.skip_train:
        cmd = [
            py,
            str(ROOT / "scripts/chase018/train_fasc_frundles_recipe.py"),
            "--img-dir",
            str(stage_img),
            "--mask-dir",
            str(stage_msk),
            "--out-dir",
            str(ckpt_dir),
            "--init",
            str(init_ckpt),
            "--tag",
            tag,
            "--epochs",
            str(args.epochs),
            "--bs",
            str(args.bs),
            "--lr",
            str(args.lr),
            "--device",
            device,
            "--seed",
            str(args.seed),
            "--crop-min",
            str(args.crop_min),
            "--rot-deg",
            str(args.rot_deg),
            "--rot-prob",
            str(args.rot_prob),
        ]
        print("[crop_aug] TRAIN", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            partial = {}
            tj = ckpt_dir / f"{tag}_train.json"
            if tj.exists():
                partial = json.loads(tj.read_text())
            report = {
                "updated": now(),
                "status": "train_failed",
                "returncode": r.returncode,
                "device": device,
                "partial": partial,
                "promote": False,
                "gate_pass": False,
                "no_kaggle": True,
            }
            (report_dir / "multexp_crop_scale_aug.json").write_text(json.dumps(report, indent=2))
            write_status(
                report_dir,
                f"# MultExp crop/scale + tiny-rot Frundles FT — TRAIN FAILED\n\n"
                f"Updated {now()}. returncode={r.returncode}. device={device}.\n"
                f"Partial train json: `{tj}` best={partial.get('best')}.\n",
                prepend_main=True,
            )
            raise SystemExit(r.returncode)
        tj = ckpt_dir / f"{tag}_train.json"
        train_hist = json.loads(tj.read_text()) if tj.exists() else None
    else:
        print("[crop_aug] skip train; using existing ckpt", flush=True)
        if not ckpt_path.exists():
            raise SystemExit(f"missing ckpt {ckpt_path}")

    if not ckpt_path.exists():
        raise SystemExit(f"missing ckpt after train: {ckpt_path}")

    infer_cmd = [
        py,
        str(ROOT / "scripts/chase018/infer_multexp_maps.py"),
        "--fasc-ckpt",
        str(ckpt_path),
        "--cache-dir",
        str(cache_dir),
        "--device",
        device,
        "--force",
    ]
    print("[crop_aug] INFER", " ".join(infer_cmd), flush=True)
    r = subprocess.run(infer_cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"infer failed rc={r.returncode}")

    print("[crop_aug] eval raw FT maps with frozen recipe", flush=True)
    raw_frozen, df_raw = eval_frozen(cache_dir, "raw_crop_aug")
    raw_nested, _ = nested_sin_w(cache_dir, "raw_crop_aug")

    print("[crop_aug] eval Frundles baseline maps", flush=True)
    fr_frozen, _ = eval_frozen(fr_cache, "frundles_ref")

    blend_diag = []
    for a in (0.15, 0.25, 0.40):
        bdir = report_dir / "cache" / f"maps_multexp_crop_scale_aug_blend{int(round(a * 100)):02d}"
        nb = blend_caches(cache_dir, fr_cache, bdir, a)
        print(f"[crop_aug] diagnostic blend alpha={a:.2f} n={nb}", flush=True)
        bs, _ = eval_frozen(bdir, f"blend_a{a:.2f}")
        blend_diag.append(bs)

    # Honest candidates: predeclared frozen raw + nested sin_w only (no blend oracle)
    honest = [raw_frozen, raw_nested]
    champ = min(honest, key=lambda s: s["S"])
    promote = bool(champ["S"] < HOLD)
    gate = bool(champ["S"] < GATE)
    elapsed = time.time() - t0
    best_dice = None
    if train_hist and isinstance(train_hist, dict):
        best_dice = train_hist.get("best")

    report = {
        "updated": now(),
        "elapsed_sec": elapsed,
        "hold": HOLD,
        "hold_note": "honest MultExp nested joint-search S=0.13449 (leaky 0.13195 is not the comparator)",
        "gate": GATE,
        "promote": promote,
        "gate_pass": gate,
        "no_kaggle": True,
        "multexp_in_train": False,
        "init": str(init_ckpt),
        "ckpt": str(ckpt_path),
        "cache_raw": str(cache_dir),
        "device": device,
        "epochs": args.epochs,
        "bs": args.bs,
        "lr": args.lr,
        "crop_min": args.crop_min,
        "rot_deg": min(8.0, args.rot_deg),
        "rot_prob": args.rot_prob,
        "train_mix": src_keep,
        "dropped_near_multexp": dropped,
        "n_pairs": len(pairs),
        "best_val_dice": best_dice,
        "raw_frozen": raw_frozen,
        "raw_nested_sinw": raw_nested,
        "frundles_ref": fr_frozen,
        "blend_diag_not_for_promote": blend_diag,
        "champ": champ,
        "beats_frundles_maps": bool(champ["S"] < fr_frozen["S"]),
        "domain_shift_loses_to_frundles": bool(champ["S"] > fr_frozen["S"]),
        "train_hist_tail": (train_hist.get("hist", [])[-5:] if train_hist else None),
    }
    out_json = report_dir / "multexp_crop_scale_aug.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    if df_raw is not None:
        df_raw.to_csv(report_dir / "multexp_crop_scale_aug_rows.csv", index=False)

    beat = "BEAT" if champ["S"] < HOLD else "MISS"
    body = f"""# MultExp crop/scale + tiny-rot Frundles FT — {now()}

## Result
| metric | value |
|---|---|
| **honest champ S** | **{champ['S']:.5f}** ({champ['tag']}) |
| MAE PA / FL / MT | {champ.get('mae_pa', float('nan')):.3f}° / {champ.get('mae_fl', float('nan')):.3f} mm / {champ.get('mae_mt', float('nan')):.3f} mm |
| raw frozen S | {raw_frozen['S']:.5f} (PA {raw_frozen['mae_pa']:.3f} / FL {raw_frozen['mae_fl']:.3f} / MT {raw_frozen['mae_mt']:.3f}) |
| nested sin_w S | {raw_nested['S']:.5f} |
| Frundles map ref S | {fr_frozen['S']:.5f} |
| vs honest hold 0.13449 | {beat} ({champ['S'] - HOLD:+.5f}) |
| gate 0.12 | {'PASS' if gate else 'FAIL'} |
| promote | {promote} |
| domain-shift vs Frundles maps | {'WINS' if champ['S'] < fr_frozen['S'] else 'LOSES'} ({champ['S'] - fr_frozen['S']:+.5f}) |
| train mix | {src_keep} |
| dropped near MultExp | {dropped} |
| best val Dice | {best_dice} |
| init | Frundles fasc_resnet34_384x576_f0.pt |
| aug | crop/scale [{args.crop_min}, 1.0] + rot ±{min(8.0, args.rot_deg):.0f}° p={args.rot_prob} (same affine on image+mask) |
| epochs / bs / lr / device | {args.epochs} / {args.bs} / {args.lr} / {device} |
| elapsed | {elapsed/60:.1f} min |
| no Kaggle | true |

## Protocol
MultExp dHash-held-out. Frozen recipe `{FROZEN['name']}`. Nested sin_w over {SIN_WS} is honest LOOCV (pick on 34, score 1). Blend alphas are diagnostic only and not used for promotion. Comparator is honest joint S=0.13449, not the leaky 0.13195.

## Artifacts
- ckpt: `{ckpt_path}`
- maps: `{cache_dir}`
- json: `{out_json}`
"""
    write_status(report_dir, body, prepend_main=True)
    print(json.dumps({"champ": champ, "promote": promote, "gate": gate, "hold": HOLD, "frundles_ref": fr_frozen}, indent=2, default=str), flush=True)
    print(f"[crop_aug] DONE {now()} elapsed={elapsed:.0f}s wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()