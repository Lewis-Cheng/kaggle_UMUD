#!/usr/bin/env python3
"""Offline MultExpData S_expert_NOT_LB for floor vs texture / fasc-ft variants.

Does NOT touch the public LB. Uses official score.py tolerances.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.calibration.anisotropic_scale import mm_per_prob_pixel  # noqa: E402
from umud.geometry.host_faithful import measure, pa_depth_fit  # noqa: E402
from umud.geometry.texture_angle import texture_angle  # noqa: E402
from umud.metric.score import score_components, TOLERANCES  # noqa: E402
from umud.seg.frundles_smp import load_ckpt, load_pair, predict_prob  # noqa: E402


def run_variant(name, expert, expert_dir, m_apo, s_apo, m_fasc, s_fasc, device, *, tex_w, fl_sin_w, fl_factor):
    rows = []
    for _, er in expert.iterrows():
        path = expert_dir / er["image_file"]
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H0, W0 = g.shape
        px_per_mm = float(er["Scale_pixel_per_cm"]) / 10.0
        pa = predict_prob(m_apo, s_apo, g, device)
        pf = predict_prob(m_fasc, s_fasc, g, device)
        h, w = pa.shape
        mx, my = mm_per_prob_pixel(px_per_mm, px_per_mm, W0, H0, w, h)
        res = measure(pa, pf, mx, my)
        apo = res.get("apo")
        pa_deg = float(res["pa_deg"])
        # single-image depth fit when enough segs
        segs = res.get("segs")
        if segs is not None and len(segs) >= 5:
            v = pa_depth_fit(segs)
            if v is not None:
                pa_deg = v
        if tex_w > 0 and apo is not None:
            pa_tex = texture_angle(g, px_per_mm, px_per_mm, apo)[1]
            if np.isfinite(pa_tex):
                pa_deg = (1.0 - tex_w) * pa_deg + tex_w * float(pa_tex)
        fl = float(res["fl_mm"]) * fl_factor
        mt = float(res["mt_mm"])
        if fl_sin_w > 0:
            fl_sin = mt / np.sin(np.radians(np.clip(pa_deg, 5, 45)))
            fl = (1.0 - fl_sin_w) * fl + fl_sin_w * fl_sin
        rows.append(
            dict(
                image_file=er["image_file"],
                ok=bool(res["ok"]),
                pred_pa=pa_deg,
                pred_fl=fl,
                pred_mt=mt,
                gt_pa=float(er["median_PA"]),
                gt_fl=float(er["median_FL"]),
                gt_mt=float(er["median_MT"]),
            )
        )
    cal = pd.DataFrame(rows)
    err = {
        "pa_deg": np.abs(cal.pred_pa - cal.gt_pa).to_numpy(),
        "fl_mm": np.abs(cal.pred_fl - cal.gt_fl).to_numpy(),
        "mt_mm": np.abs(cal.pred_mt - cal.gt_mt).to_numpy(),
    }
    comps = score_components(err)
    mae = {k: float(np.mean(err[k])) for k in err}
    return {
        "name": name,
        "n": len(cal),
        "ok": int(cal.ok.sum()),
        "S_expert_NOT_LB": comps["S"],
        "S1": comps["S1"],
        "MAE": mae,
        "tex_w": tex_w,
        "fl_sin_w": fl_sin_w,
        "fl_factor": fl_factor,
    }, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-csv", type=Path, default=ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    ap.add_argument("--expert-dir", type=Path, default=ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0")
    ap.add_argument("--weights", type=Path, default=ROOT / "data/external/frundles_weights")
    ap.add_argument("--fasc-ckpt", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=ROOT / "reports/phase3/chase018/multexp_variant_scores.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    expert = pd.read_csv(args.expert_csv)
    pair = load_pair(args.weights, device=args.device)
    m_apo, s_apo = pair["apo"]
    m_fasc, s_fasc = pair["fasc"]
    if args.fasc_ckpt:
        m_fasc, s_fasc = load_ckpt(args.fasc_ckpt, args.device)

    variants = [
        ("floor_frundles", 0.0, 0.0, 0.9),
        ("texture_pa_tex0.5_flsin0.25", 0.5, 0.25, 0.9),
        ("texture_pa_tex0.5_only", 0.5, 0.0, 0.9),
        ("texture_pa_tex0.7_flsin0.25", 0.7, 0.25, 0.9),
    ]
    results = []
    for name, tw, fw, flf in variants:
        print("running", name, flush=True)
        summary, cal = run_variant(name, expert, args.expert_dir, m_apo, s_apo, m_fasc, s_fasc, args.device, tex_w=tw, fl_sin_w=fw, fl_factor=flf)
        print(json.dumps(summary, indent=2), flush=True)
        results.append(summary)
        cal.to_csv(args.out_json.with_name(f"multexp_{name}.csv"), index=False)

    args.out_json.write_text(json.dumps(results, indent=2))
    print("wrote", args.out_json)
    # ranking
    results_sorted = sorted(results, key=lambda r: r["S_expert_NOT_LB"])
    print("RANK (lower S better):")
    for r in results_sorted:
        print(f"  {r['S_expert_NOT_LB']:.5f}  {r['name']}  MAE_pa={r['MAE']['pa_deg']:.3f} fl={r['MAE']['fl_mm']:.3f} mt={r['MAE']['mt_mm']:.3f}")


if __name__ == "__main__":
    main()
