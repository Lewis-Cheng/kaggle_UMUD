#!/usr/bin/env python3
"""New reference maps, then the cal_pa_mt operator that beat texture_pa on the public board.

texture_rl is exactly cal_pa_mt. Against texture_pa its fascicle length is unchanged,
pennation is the affine, and thickness is shifted by about +0.18 mm. That file scored
0.29895. Scaling both lengths by 0.97 scored 0.35218. This script does not scale length.
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

from umud.baselines.io import load_official_submission_ids, write_submission_csv  # noqa: E402
from umud.calibration.anisotropic_scale import mm_per_prob_pixel  # noqa: E402
from umud.calibration.executable import require_executable_calibration  # noqa: E402
from umud.geometry.host_faithful import dhash, groups_from_hashes, measure, pa_depth_fit  # noqa: E402
from umud.geometry.texture_angle import texture_angle  # noqa: E402
from umud.seg.frundles_smp import load_ckpt  # noqa: E402

RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
INPUT_MEAN = 0.485
INPUT_STD = 0.229


def clip_values(pa: float, fl: float, mt: float) -> tuple[float, float, float]:
    return (
        float(np.clip(pa, *RANGES["pa_deg"])),
        float(np.clip(fl, *RANGES["fl_mm"])),
        float(np.clip(mt, *RANGES["mt_mm"])),
    )


@torch.no_grad()
def predict_prob(model, size: tuple[int, int], gray: np.ndarray, device: torch.device, mean: float, std: float) -> np.ndarray:
    oh, ow = size
    image = cv2.resize(gray, (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    image = (image - mean) / std
    batch = torch.from_numpy(image)[None, None].to(device)
    prob = (torch.sigmoid(model(batch)) + torch.sigmoid(model(batch.flip(-1))).flip(-1)) / 2
    return prob[0, 0].detach().cpu().numpy()


def main() -> int:
    parser = argparse_parser()
    args = parser.parse_args()
    device = torch.device(args.device)
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    ids = load_official_submission_ids(args.ids)
    files = [args.test_dir / image_id for image_id in ids]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} test images, e.g. {missing[0]}")
    scale = pd.read_csv(args.scale_csv).set_index("image_id")
    coeff = require_executable_calibration(json.loads(args.affine_json.read_text()))
    pa_a, pa_b = coeff["pa_deg"]
    mt_a, mt_b = coeff["mt_mm"]
    if coeff["fl_mm"] is not None or pa_a is None or mt_a is None:
        raise SystemExit("calibration must be PA/MT affine and FL identity")

    apo_model, apo_size = load_ckpt(args.apo_ckpt, device)
    fasc_model, fasc_size = load_ckpt(args.fasc_ckpt, device)
    apo_model.eval()
    fasc_model.eval()
    apo_norm = _norm_from_ckpt(args.apo_ckpt)
    fasc_norm = _norm_from_ckpt(args.fasc_ckpt)

    rows = []
    segs = []
    hashes = []
    for index, path in enumerate(files):
        if index % 25 == 0 or index + 1 == len(files):
            print(f"infer {index + 1}/{len(files)} {path.name}", flush=True)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"unreadable {path}")
        height, width = gray.shape
        apo = predict_prob(apo_model, apo_size, gray, device, *apo_norm)
        fasc = predict_prob(fasc_model, fasc_size, gray, device, *fasc_norm)
        map_h, map_w = apo.shape
        sx = float(scale.loc[path.name, "sx"])
        sy = float(scale.loc[path.name, "sy"])
        mx, my = mm_per_prob_pixel(sx, sy, width, height, map_w, map_h)
        measured = measure(apo, fasc, mx, my)
        apo_bands = measured.pop("apo", None)
        segs.append(measured.pop("segs"))
        hashes.append(dhash(gray))
        pa_tex = float("nan")
        if apo_bands is not None:
            pa_tex = texture_angle(gray, sx, sy, apo_bands)[1]
        rows.append(
            {
                "image_id": path.name,
                "pa_deg": float(measured["pa_deg"]),
                "fl_mm": float(measured["fl_mm"]),
                "mt_mm": float(measured["mt_mm"]),
                "ok": bool(measured["ok"]),
                "pa_tex": pa_tex,
            }
        )

    frame = pd.DataFrame(rows)
    frame["run"] = groups_from_hashes(hashes)
    for _, group_index in frame.groupby("run").groups.items():
        fitted = pa_depth_fit(np.concatenate([segs[i] for i in group_index]))
        if fitted is not None:
            frame.loc[list(group_index), "pa_deg"] = fitted
    texture = frame.groupby("run").pa_tex.transform("median")
    use_texture = texture.notna() & np.isfinite(texture)
    frame.loc[use_texture, "pa_deg"] = (1.0 - args.tex_w) * frame.loc[use_texture, "pa_deg"] + args.tex_w * texture[use_texture]
    frame["fl_mm"] = frame["fl_mm"] * args.fl_factor
    if args.fl_sin_w > 0:
        fl_sin = frame.mt_mm / np.sin(np.radians(frame.pa_deg.clip(5, 45)))
        frame["fl_mm"] = (1.0 - args.fl_sin_w) * frame.fl_mm + args.fl_sin_w * fl_sin

    out_rows = []
    for _, row in frame.iterrows():
        pa, fl, mt = clip_values(
            pa_a * float(row.pa_deg) + pa_b,
            float(row.fl_mm),
            mt_a * float(row.mt_mm) + mt_b,
        )
        out_rows.append({"image_id": row.image_id, "pa_deg": pa, "fl_mm": fl, "mt_mm": mt})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(args.out, out_rows, validate_against=ids)
    text = args.out.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        raise SystemExit("submission acquired a BOM")
    meta = {
        "out": str(args.out),
        "n": len(out_rows),
        "ok_rate": float(frame.ok.mean()),
        "apo_ckpt": str(args.apo_ckpt),
        "fasc_ckpt": str(args.fasc_ckpt),
        "calibration": "pa_mt_affine_fl_identity",
        "fl_factor": args.fl_factor,
        "fl_sin_w": args.fl_sin_w,
        "tex_w": args.tex_w,
        "length_scale": 1.0,
        "why": "Public 0.35218 showed a 0.97 length scale hurts. texture_rl gained by PA affine and +0.18 mm MT, with FL unchanged.",
        "claims_kaggle_lb": False,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


def _norm_from_ckpt(path: Path) -> tuple[float, float]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return float(blob.get("input_mean", INPUT_MEAN)), float(blob.get("input_std", INPUT_STD))


def argparse_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/raw/umud/test_images_v2/test_set_v2")
    parser.add_argument("--ids", type=Path, default=ROOT / "reports/phase3/official_test_file_names.txt")
    parser.add_argument("--scale-csv", type=Path, default=ROOT / "reports/phase3/chase018/scale_aniso.csv")
    parser.add_argument("--affine-json", type=Path, default=ROOT / "reports/phase3/chase018/multexp_chase_affine.json")
    parser.add_argument("--apo-ckpt", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts/ref_apo_optuna_best.pt")
    parser.add_argument("--fasc-ckpt", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts/ref_fasc_optuna_best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/phase3/chase018/submission_ref_cal_pa_mt.csv")
    parser.add_argument("--fl-factor", type=float, default=0.9)
    parser.add_argument("--fl-sin-w", type=float, default=0.25)
    parser.add_argument("--tex-w", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
