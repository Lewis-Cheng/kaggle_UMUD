#!/usr/bin/env python3
"""Diagnose fasc segmentation failure modes (imbalance, thinness, FOV)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from umud.baselines.io import load_split_assignments
from umud.data.layout import discover_layout, list_images

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--scale-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    layout = discover_layout(args.data_root)
    imgs = {p.stem: p for p in list_images(layout.fasc_imgs)}
    masks = {p.stem: p for p in list_images(layout.fasc_masks)}
    rows = load_split_assignments(args.split)
    val = [str(r["image_id"]) for r in rows if r.get("task")=="fascicle" and r.get("split")=="val"]
    train = [str(r["image_id"]) for r in rows if r.get("task")=="fascicle" and r.get("split")=="train"]

    scale = {}
    with args.scale_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("pool") == "fasc":
                scale[r["stem"]] = r

    def stats(stems, tag):
        fg_fracs, widths, heights, thicknesses, fov_cover = [], [], [], [], 0
        for s in stems[: args.n]:
            if s not in masks: continue
            m = np.asarray(Image.open(masks[s]))
            if m.ndim == 3: m = m[...,0]
            binm = m > 0
            h, w = binm.shape
            heights.append(h); widths.append(w)
            fg_fracs.append(float(binm.mean()))
            # thickness: mean run length along columns where fg present
            runs = []
            ys, xs = np.where(binm)
            if len(ys):
                # approximate thickness via distance to background
                from scipy import ndimage
                try:
                    dist = ndimage.distance_transform_edt(binm)
                    thicknesses.append(float(2*dist[binm].mean()))
                except Exception:
                    thicknesses.append(float("nan"))
            sc = scale.get(s)
            if sc and sc.get("fov_l") not in (None, "", "None"):
                fov_cover += 1
                # fg outside FOV?
        return {
            "tag": tag,
            "n_sampled": len(fg_fracs),
            "fg_frac_mean": float(np.mean(fg_fracs)) if fg_fracs else None,
            "fg_frac_p50": float(np.median(fg_fracs)) if fg_fracs else None,
            "fg_frac_p10": float(np.percentile(fg_fracs, 10)) if fg_fracs else None,
            "size_h_p50": float(np.median(heights)) if heights else None,
            "size_w_p50": float(np.median(widths)) if widths else None,
            "thickness_px_p50": float(np.nanmedian(thicknesses)) if thicknesses else None,
            "fov_rows_in_scale_csv": fov_cover,
            "implied_pos_weight": float((1-np.mean(fg_fracs))/max(np.mean(fg_fracs),1e-8)) if fg_fracs else None,
        }

    # also check current preds if exist
    pred_dir = ROOT / "reports/phase3/seg_preds/pred_fasc_masks"
    pred_stats = None
    if pred_dir.is_dir():
        dices = []
        for s in val[: args.n]:
            gp = masks.get(s); pp = pred_dir / f"{s}.png"
            if gp is None or not pp.is_file(): continue
            g = (np.asarray(Image.open(gp))>0)
            if g.ndim==3: g=g[...,0]
            p = (np.asarray(Image.open(pp))>0)
            # resize pred to gt
            if p.shape != g.shape:
                p = np.asarray(Image.fromarray(p.astype(np.uint8)*255).resize((g.shape[1], g.shape[0]), Image.NEAREST))>0
            inter = float((p&g).sum()); den=float(p.sum()+g.sum())
            dices.append((2*inter+1e-6)/(den+1e-6))
        pred_stats = {
            "n": len(dices),
            "dice_mean": float(np.mean(dices)) if dices else None,
            "dice_p50": float(np.median(dices)) if dices else None,
            "empty_pred_frac": float(np.mean([d<1e-4 for d in dices])) if dices else None,
        }

    report = {
        "train": stats(train, "train"),
        "val": stats(val, "val"),
        "current_pred_vs_gt": pred_stats,
        "hypotheses": [
            "fg_frac << 1% → extreme imbalance; need pos_weight ~1/fg or dilated soft targets",
            "thickness few px → bilinear resize to 256/384 destroys lines; prefer FOV crop + higher res or nearest+dilate",
            "HUD chrome outside FOV confuses model; AmbrosM fov_* crop should help",
            "Dice-only on thin lines under-penalizes misses; use weighted BCE + soft-clDice or dilate-target Dice",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
