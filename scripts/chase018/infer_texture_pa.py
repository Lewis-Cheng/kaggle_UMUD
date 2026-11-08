#!/usr/bin/env python3
"""Chase-0.18: floor geometry + Mawra texture-angle PA blend (+ optional FL sin).

Writes reports/phase3/chase018/submission_texture_pa.csv (comma, no BOM).
Also caches float16 prob maps under reports/phase3/chase018/cache/maps_test/
so fasc-ft re-runs skip apo/fasc forward if maps exist.
"""
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

from umud.calibration.anisotropic_scale import mm_per_prob_pixel  # noqa: E402
from umud.geometry.host_faithful import (  # noqa: E402
    dhash,
    groups_from_hashes,
    measure,
    pa_depth_fit,
)
from umud.geometry.texture_angle import texture_angle  # noqa: E402
from umud.seg.frundles_smp import load_ckpt, load_pair, predict_prob  # noqa: E402

RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def clip_row(r: dict) -> dict:
    for k, (lo, hi) in RANGES.items():
        r[k] = float(np.clip(r[k], lo, hi))
    return r


def load_or_predict(cache_dir: Path, name: str, m_apo, s_apo, m_fasc, s_fasc, g, device):
    npz = cache_dir / f"{name}.npz"
    if npz.exists():
        z = np.load(npz)
        return z["apo"].astype(np.float32), z["fasc"].astype(np.float32)
    pa = predict_prob(m_apo, s_apo, g, device)
    pf = predict_prob(m_fasc, s_fasc, g, device)
    np.savez_compressed(npz, apo=pa.astype(np.float16), fasc=pf.astype(np.float16))
    return pa, pf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/umud/test_images_v2/test_set_v2")
    ap.add_argument("--ids", type=Path, default=ROOT / "reports/phase3/official_test_file_names.txt")
    ap.add_argument("--scale-csv", type=Path, default=ROOT / "reports/phase3/chase018/scale_aniso.csv")
    ap.add_argument("--weights", type=Path, default=ROOT / "data/external/frundles_weights")
    ap.add_argument("--fasc-ckpt", type=Path, default=None, help="override fasc weights (fine-tuned)")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "reports/phase3/chase018/cache/maps_test")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/phase3/chase018/submission_texture_pa.csv")
    ap.add_argument("--fl-factor", type=float, default=0.9)
    ap.add_argument("--tex-w", type=float, default=0.5)
    ap.add_argument("--fl-sin-w", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache-write", action="store_true")
    args = ap.parse_args()

    raw = [ln.strip() for ln in args.ids.read_text().splitlines() if ln.strip() and not ln.endswith(".csv")]
    basenames = [Path(ln).name for ln in raw]
    files = [args.test_dir / b for b in basenames]
    missing = [p for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} images e.g. {missing[0]}")
    if args.limit:
        files = files[: args.limit]
        basenames = basenames[: args.limit]

    scale = pd.read_csv(args.scale_csv).set_index("image_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print("loading weights on", device, flush=True)
    pair = load_pair(args.weights, device=device)
    m_apo, s_apo = pair["apo"]
    m_fasc, s_fasc = pair["fasc"]
    if args.fasc_ckpt:
        print("override fasc", args.fasc_ckpt, flush=True)
        m_fasc, s_fasc = load_ckpt(args.fasc_ckpt, device)
        # invalidate fasc half of cache by using a different cache dir name tag
        tag = args.fasc_ckpt.stem
        args.cache_dir = args.cache_dir.parent / f"maps_test_{tag}"
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows, segs, hashes = [], [], []
    for i, p in enumerate(files):
        if i % 25 == 0 or i + 1 == len(files):
            print(f"infer {i+1}/{len(files)} {p.name}", flush=True)
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        H0, W0 = g.shape
        if args.no_cache_write:
            pa = predict_prob(m_apo, s_apo, g, device)
            pf = predict_prob(m_fasc, s_fasc, g, device)
        else:
            pa, pf = load_or_predict(args.cache_dir, p.name, m_apo, s_apo, m_fasc, s_fasc, g, device)
        h, w = pa.shape
        sx = float(scale.loc[p.name, "sx"])
        sy = float(scale.loc[p.name, "sy"])
        mx, my = mm_per_prob_pixel(sx, sy, W0, H0, w, h)
        res = measure(pa, pf, mx, my)
        apo = res.pop("apo", None)
        segs.append(res.pop("segs"))
        hashes.append(dhash(g))
        pa_tex = float("nan")
        if apo is not None and args.tex_w > 0:
            pa_tex = texture_angle(g, sx, sy, apo)[1]
        rows.append(
            dict(
                image_id=p.name,
                layout=scale.loc[p.name, "layout"],
                pa_deg=res["pa_deg"],
                fl_mm=res["fl_mm"],
                mt_mm=res["mt_mm"],
                n_fasc=res["n_fasc"],
                ok=res["ok"],
                pa_tex=pa_tex,
            )
        )

    df = pd.DataFrame(rows)
    print("fallback (not ok):", int((~df.ok).sum()), "tex finite:", int(np.isfinite(df.pa_tex).sum()), flush=True)

    df["run"] = groups_from_hashes(hashes)
    for _, idx in df.groupby("run").groups.items():
        v = pa_depth_fit(np.concatenate([segs[i] for i in idx]))
        if v is not None:
            df.loc[list(idx), "pa_deg"] = v

    if args.tex_w > 0:
        tex = df.groupby("run").pa_tex.transform("median")
        m = tex.notna() & np.isfinite(tex)
        df.loc[m, "pa_deg"] = (1.0 - args.tex_w) * df.loc[m, "pa_deg"] + args.tex_w * tex[m]

    df["fl_mm"] = df["fl_mm"] * args.fl_factor
    if args.fl_sin_w > 0:
        fl_sin = df.mt_mm / np.sin(np.radians(df.pa_deg.clip(5, 45)))
        df["fl_mm"] = (1.0 - args.fl_sin_w) * df.fl_mm + args.fl_sin_w * fl_sin

    sub_rows = []
    for _, r in df.iterrows():
        rr = clip_row({"pa_deg": float(r.pa_deg), "fl_mm": float(r.fl_mm), "mt_mm": float(r.mt_mm)})
        sub_rows.append({"image_id": r.image_id, **rr})
    sub = pd.DataFrame(sub_rows)[["image_id", "pa_deg", "fl_mm", "mt_mm"]]
    order = {b: i for i, b in enumerate(basenames)}
    sub["_o"] = sub.image_id.map(order)
    sub = sub.sort_values("_o").drop(columns="_o")
    sub.to_csv(args.out, index=False, lineterminator="\n")
    diag = args.out.with_name(args.out.stem + "_diag.csv")
    df.to_csv(diag, index=False)
    print("wrote", args.out, "rows", len(sub), flush=True)
    print(sub.describe().round(2), flush=True)
    print("layouts ok rate:", flush=True)
    print(df.groupby("layout").ok.mean().round(3), flush=True)


if __name__ == "__main__":
    main()
