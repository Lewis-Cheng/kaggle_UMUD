#!/usr/bin/env python3
"""Dump MultExp apo+fasc probability maps (float16 npz) for offline geometry eval.

Writes one <image_file>.npz per expert image with keys apo, fasc (H=384,W=576).
Frundles apo always; fasc from public Frundles or --fasc-ckpt override.
No Kaggle submit.
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

from umud.seg.frundles_smp import load_ckpt, load_pair, predict_prob  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--expert-csv",
        type=Path,
        default=ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv",
    )
    ap.add_argument(
        "--expert-dir",
        type=Path,
        default=ROOT
        / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0",
    )
    ap.add_argument("--weights", type=Path, default=ROOT / "data/external/frundles_weights")
    ap.add_argument("--fasc-ckpt", type=Path, default=None)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "reports/phase3/chase018/cache/maps_multexp",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="re-infer even if npz exists")
    args = ap.parse_args()

    expert = pd.read_csv(args.expert_csv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"[infer_multexp_maps] device={device} n={len(expert)} cache={args.cache_dir}", flush=True)

    pair = load_pair(args.weights, device=device)
    m_apo, s_apo = pair["apo"]
    m_fasc, s_fasc = pair["fasc"]
    if args.fasc_ckpt is not None:
        ck = Path(args.fasc_ckpt)
        if not ck.exists():
            raise SystemExit(f"fasc ckpt missing: {ck}")
        print(f"[infer_multexp_maps] fasc override {ck}", flush=True)
        m_fasc, s_fasc = load_ckpt(ck, device)

    n_ok = n_skip = 0
    for i, er in expert.iterrows():
        name = er["image_file"]
        out = args.cache_dir / f"{name}.npz"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        path = args.expert_dir / name
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"WARN missing image {path}", flush=True)
            continue
        pa = predict_prob(m_apo, s_apo, g, device)
        pf = predict_prob(m_fasc, s_fasc, g, device)
        np.savez_compressed(out, apo=pa.astype(np.float16), fasc=pf.astype(np.float16))
        n_ok += 1
        if n_ok % 5 == 0 or n_ok == 1:
            print(f"  wrote {n_ok} {name} apo={pa.shape} fasc={pf.shape}", flush=True)

    print(
        f"[infer_multexp_maps] DONE wrote={n_ok} skipped_existing={n_skip} "
        f"cache_count={len(list(args.cache_dir.glob('*.npz')))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
