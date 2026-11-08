#!/usr/bin/env python3
"""Chase-0.18 geometry floor: Frundles aniso scale + dual U-Net + host-faithful geometry."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.calibration.anisotropic_scale import (  # noqa: E402
    estimate_anisotropic_scales,
    mm_per_prob_pixel,
)
from umud.geometry.host_faithful import (  # noqa: E402
    dhash,
    groups_from_hashes,
    measure,
    pa_depth_fit,
)
from umud.seg.frundles_smp import load_pair, predict_prob  # noqa: E402

RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
FL_FACTOR_DEFAULT = 0.9


def clip_row(r: dict) -> dict:
    for k, (lo, hi) in RANGES.items():
        r[k] = float(np.clip(r[k], lo, hi))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/umud/test_images_v2/test_set_v2")
    ap.add_argument("--ids", type=Path, default=ROOT / "reports/phase3/official_test_file_names.txt")
    ap.add_argument("--scale-csv", type=Path, default=ROOT / "reports/phase3/chase018/scale_aniso.csv")
    ap.add_argument("--weights", type=Path, default=ROOT / "data/external/frundles_weights")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/phase3/chase018/submission_geo_floor.csv")
    ap.add_argument("--fl-factor", type=float, default=FL_FACTOR_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-pa-depth-fit", action="store_true")
    args = ap.parse_args()

    # Resolve official basenames
    raw = [ln.strip() for ln in args.ids.read_text().splitlines() if ln.strip() and not ln.endswith(".csv")]
    basenames = [Path(ln).name for ln in raw]
    files = [args.test_dir / b for b in basenames]
    missing = [p for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} images e.g. {missing[0]}")
    if args.limit:
        files = files[: args.limit]
        basenames = basenames[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.scale_csv.exists():
        scale = pd.read_csv(args.scale_csv).set_index("image_id")
        print("loaded scale", args.scale_csv, len(scale))
    else:
        print("computing anisotropic scale …", flush=True)
        scale_df = estimate_anisotropic_scales(files, show_progress=True)
        scale_df.to_csv(args.scale_csv, index=False)
        print("wrote", args.scale_csv)
        scale = scale_df.set_index("image_id")

    device = args.device
    print("loading weights on", device, flush=True)
    pair = load_pair(args.weights, device=device)
    (m_apo, s_apo) = pair["apo"]
    (m_fasc, s_fasc) = pair["fasc"]

    rows, segs, hashes = [], [], []
    for i, p in enumerate(files):
        if i % 25 == 0 or i + 1 == len(files):
            print(f"infer {i+1}/{len(files)} {p.name}", flush=True)
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        H0, W0 = g.shape
        pa = predict_prob(m_apo, s_apo, g, device)
        pf = predict_prob(m_fasc, s_fasc, g, device)
        h, w = pa.shape
        sx = float(scale.loc[p.name, "sx"])
        sy = float(scale.loc[p.name, "sy"])
        mx, my = mm_per_prob_pixel(sx, sy, W0, H0, w, h)
        res = measure(pa, pf, mx, my)
        segs.append(res.pop("segs"))
        hashes.append(dhash(g))
        rows.append(
            dict(
                image_id=p.name,
                layout=scale.loc[p.name, "layout"],
                pa_deg=res["pa_deg"],
                fl_mm=res["fl_mm"],
                mt_mm=res["mt_mm"],
                n_fasc=res["n_fasc"],
                ok=res["ok"],
            )
        )

    df = pd.DataFrame(rows)
    print("fallback (not ok):", int((~df.ok).sum()))

    if not args.skip_pa_depth_fit:
        df["run"] = groups_from_hashes(hashes)
        for _, idx in df.groupby("run").groups.items():
            v = pa_depth_fit(np.concatenate([segs[i] for i in idx]))
            if v is not None:
                df.loc[list(idx), "pa_deg"] = v

    df["fl_mm"] = df["fl_mm"] * args.fl_factor
    sub_rows = []
    for _, r in df.iterrows():
        rr = clip_row({"pa_deg": r.pa_deg, "fl_mm": r.fl_mm, "mt_mm": r.mt_mm})
        sub_rows.append({"image_id": r.image_id, **rr})
    sub = pd.DataFrame(sub_rows)[["image_id", "pa_deg", "fl_mm", "mt_mm"]]
    # official order
    order = {b: i for i, b in enumerate(basenames)}
    sub["_o"] = sub.image_id.map(order)
    sub = sub.sort_values("_o").drop(columns="_o")
    # comma, no BOM
    sub.to_csv(args.out, index=False, lineterminator="\n")
    diag = args.out.with_name(args.out.stem + "_diag.csv")
    df.to_csv(diag, index=False)
    print("wrote", args.out, "rows", len(sub))
    print(sub.describe().round(2))
    print("layouts ok rate:")
    print(df.groupby("layout").ok.mean().round(3))


if __name__ == "__main__":
    main()
