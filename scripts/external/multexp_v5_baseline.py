#!/usr/bin/env python3
"""Frozen apo + fasc_v5 (MSTinyUNet) → GeometryHead on MultExpData; S_expert_NOT_LB.

Uses Scale_pixel_per_cm from expert_consensus.csv (not AmbrosM).
Does NOT train, does NOT touch competition test labels, does NOT claim LB.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.mask_geometry import measure_masks  # noqa: E402
from umud.baselines.mask_proxy import FALLBACK_PRIORS  # noqa: E402
from umud.metric.score import (  # noqa: E402
    TOLERANCES,
    per_target_error_stats,
    score_components,
)
from umud.seg.ms_unet import MSTinyUNet  # noqa: E402
from umud.seg.unet import TinyUNet  # noqa: E402


def load_seg(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = str(ck.get("arch", "TinyUNet"))
    size = int(ck.get("size", 512))
    base_ch = int(ck.get("base_ch", 32))
    thr = float(ck.get("eval_thr_preferred", ck.get("best_thr", 0.5)))
    if arch in ("MSTinyUNet", "ms_tiny_unet"):
        model = MSTinyUNet(1, base_ch)
    else:
        model = TinyUNet(1, base_ch)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, size, thr, arch, ck


@torch.no_grad()
def predict_mask(model, img_path: Path, size: int, thr: float, device: torch.device) -> np.ndarray:
    im = Image.open(img_path).convert("L")
    w, h = im.size
    x = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(x)[None, None].to(device)
    out = model(t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    prob = torch.sigmoid(out)[0, 0].cpu().numpy()
    pred = (prob > thr).astype(np.uint8)
    return (
        np.asarray(Image.fromarray(pred * 255).resize((w, h), Image.NEAREST)) > 127
    ).astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--consensus-csv", type=Path, required=True)
    ap.add_argument("--apo-ckpt", type=Path, required=True)
    ap.add_argument("--fasc-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    apo, apo_size, apo_thr, apo_arch, apo_ck = load_seg(args.apo_ckpt, device)
    fasc, fasc_size, fasc_thr, fasc_arch, fasc_ck = load_seg(args.fasc_ckpt, device)
    print(
        f"apo arch={apo_arch} size={apo_size} thr={apo_thr} "
        f"fasc arch={fasc_arch} size={fasc_size} thr={fasc_thr} "
        f"variant={fasc_ck.get('variant')} native={fasc_ck.get('best_native_dice')}"
    )

    cons = pd.read_csv(args.consensus_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_apo = args.out_dir / "pred_apo_masks"
    mask_fasc = args.out_dir / "pred_fasc_masks"
    mask_apo.mkdir(exist_ok=True)
    mask_fasc.mkdir(exist_ok=True)

    rows = []
    for _, r in cons.iterrows():
        stem = str(r["ImageID"])
        img = args.images_dir / f"{stem}.tif"
        if not img.is_file():
            rows.append({"ImageID": stem, "ok": False, "failure_reason": "missing_image"})
            continue
        scale = float(r["Scale_pixel_per_cm"])
        # px per cm → mm per px
        mm_per_px = 10.0 / scale
        sx = sy = mm_per_px

        apo_m = predict_mask(apo, img, apo_size, apo_thr, device)
        fasc_m = predict_mask(fasc, img, fasc_size, fasc_thr, device)
        Image.fromarray(apo_m.astype(np.uint8) * 255).save(mask_apo / f"{stem}.png")
        Image.fromarray(fasc_m.astype(np.uint8) * 255).save(mask_fasc / f"{stem}.png")

        meas = measure_masks(
            fasc_mask=fasc_m,
            apo_mask=apo_m,
            sx=sx,
            sy=sy,
            priors=FALLBACK_PRIORS,
        )
        rows.append(
            {
                "ImageID": stem,
                "Scale_pixel_per_cm": scale,
                "mm_per_px": mm_per_px,
                "pred_PA": float(meas["pa_deg"]),
                "pred_FL": float(meas["fl_mm"]),
                "pred_MT": float(meas["mt_mm"]),
                "gt_PA": float(r["median_PA"]),
                "gt_FL": float(r["median_FL"]),
                "gt_MT": float(r["median_MT"]),
                "ok": bool(meas.get("ok")),
                "failure_reason": meas.get("failure_reason"),
                "used_prior_targets": meas.get("used_prior_targets"),
                "apo_fg_frac": float(apo_m.mean()),
                "fasc_fg_frac": float(fasc_m.mean()),
            }
        )
        print(
            f"{stem}: ok={meas.get('ok')} PA={meas['pa_deg']:.2f}/{r['median_PA']:.2f} "
            f"FL={meas['fl_mm']:.2f}/{r['median_FL']:.2f} MT={meas['mt_mm']:.2f}/{r['median_MT']:.2f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "multexp_v5_rows.csv", index=False)

    scored = df.dropna(subset=["pred_PA", "pred_FL", "pred_MT", "gt_PA", "gt_FL", "gt_MT"])
    errors = {
        "pa_deg": (scored["pred_PA"] - scored["gt_PA"]).abs().to_numpy(),
        "fl_mm": (scored["pred_FL"] - scored["gt_FL"]).abs().to_numpy(),
        "mt_mm": (scored["pred_MT"] - scored["gt_MT"]).abs().to_numpy(),
    }
    comps = score_components(errors)
    stats = per_target_error_stats(errors)
    report = {
        "label": "S_expert_NOT_LB",
        "pipeline": "frozen apo + fasc_v5 MSTinyUNet → GeometryHead + MultExpData Scale_pixel_per_cm",
        "gt": "median R1-R7 MultExpData architecture v0.1.0",
        "n_images": int(len(df)),
        "n_scored": int(len(scored)),
        "n_ok_geometry": int(df["ok"].fillna(False).sum()) if "ok" in df else None,
        "MAE_PA": stats["pa_deg"]["MAE"],
        "MAE_FL": stats["fl_mm"]["MAE"],
        "MAE_MT": stats["mt_mm"]["MAE"],
        "S_expert_NOT_LB": comps["S"],
        "S1": comps["S1"],
        "S2": comps["S2"],
        "S3": comps["S3"],
        "tolerances": dict(TOLERANCES),
        "apo_ckpt": str(args.apo_ckpt),
        "fasc_ckpt": str(args.fasc_ckpt),
        "fasc_arch": fasc_arch,
        "fasc_variant": fasc_ck.get("variant"),
        "fasc_best_native_dice": fasc_ck.get("best_native_dice"),
        "claims_kaggle_lb": False,
        "attention_still_blocked": True,
    }
    (args.out_dir / "multexp_v5_baseline.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
