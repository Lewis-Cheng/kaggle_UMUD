#!/usr/bin/env python3
"""Fit ExtraTrees on UMUD train (same recipe) and predict MultExpData images.

Produces multexp_et_preds.csv for RL state. No competition test labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from extratrees_mask_proxy import (  # noqa: E402
    _build_apo_examples,
    _build_fasc_examples,
    _clip_to_quantiles,
)
from umud.baselines.features import image_feature_vector  # noqa: E402
from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.baselines.mask_proxy import load_gray  # noqa: E402
from umud.data.layout import discover_layout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--multexp-images", type=Path, required=True)
    ap.add_argument("--consensus-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--mm-per-px", type=float, default=0.07225)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.ensemble import ExtraTreesRegressor

    layout = discover_layout(args.data_root)
    rows = load_split_assignments(args.split_assignments)
    fasc_train = [r for r in rows if r.get("task") == "fascicle" and r.get("split") == "train"]
    apo_train = [r for r in rows if r.get("task") == "aponeurosis" and r.get("split") == "train"]

    Xf, Yf, _ = _build_fasc_examples(
        fasc_train, img_dir=layout.fasc_imgs, mask_dir=layout.fasc_masks, mm_per_px=args.mm_per_px
    )
    Xa, Ya, _ = _build_apo_examples(
        apo_train, img_dir=layout.apo_imgs, mask_dir=layout.apo_masks, mm_per_px=args.mm_per_px
    )
    print(f"ET train fasc={Xf.shape[0]} apo={Xa.shape[0]}")
    fasc_model = ExtraTreesRegressor(
        n_estimators=160, min_samples_leaf=3, max_features=0.72, random_state=args.seed, n_jobs=-1
    )
    apo_model = ExtraTreesRegressor(
        n_estimators=160, min_samples_leaf=3, max_features=0.72, random_state=args.seed, n_jobs=-1
    )
    fasc_model.fit(Xf, Yf)
    apo_model.fit(Xa, Ya)
    pa_med, fl_med, mt_med = float(np.median(Yf[:, 0])), float(np.median(Yf[:, 1])), float(np.median(Ya))

    cons = pd.read_csv(args.consensus_csv)
    out_rows = []
    for _, r in cons.iterrows():
        stem = str(r["ImageID"])
        ip = args.multexp_images / f"{stem}.tif"
        if not ip.is_file():
            out_rows.append(
                {"ImageID": stem, "et_pa": pa_med, "et_fl": fl_med, "et_mt": mt_med, "ok": False}
            )
            continue
        feat = image_feature_vector(load_gray(ip)).reshape(1, -1)
        pf = _clip_to_quantiles(fasc_model.predict(feat), Yf)[0]
        pm = float(_clip_to_quantiles(apo_model.predict(feat).reshape(-1), Ya)[0])
        out_rows.append(
            {
                "ImageID": stem,
                "et_pa": float(pf[0]),
                "et_fl": float(pf[1]),
                "et_mt": pm,
                "ok": True,
            }
        )
        print(f"{stem}: ET PA={pf[0]:.2f} FL={pf[1]:.2f} MT={pm:.2f}")

    df = pd.DataFrame(out_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    meta = {
        "train_medians": {"pa_deg": pa_med, "fl_mm": fl_med, "mt_mm": mt_med},
        "n": int(len(df)),
        "n_ok": int(df["ok"].sum()),
        "out_csv": str(args.out_csv),
    }
    print(meta)
    (args.out_csv.with_suffix(".json")).write_text(__import__("json").dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
