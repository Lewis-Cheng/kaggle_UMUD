#!/usr/bin/env python3
"""Spatial-attention + RL pipeline end to end on synthetic images (CPU).

Checks that the pipeline learns what it is meant to learn when the truth is
known: which fascicles are annotated, which band is deep, and how to refine a
line. Synthetic numbers are not evidence about the public leaderboard.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.data.synthetic_us import make_dataset  # noqa: E402
from umud.models.attention_rl_train import (  # noqa: E402
    TrainConfig,
    agent_channels,
    band_poly_px,
    ff_outputs,
    from_synthetic,
    geometry_rows,
    make_episode_source,
    rl_by_init_error,
    score_rows,
    train_ff,
)
from umud.models.spatial_attention_geo import AttnGeoConfig, SpatialAttentionGeoNet  # noqa: E402
from umud.rl.fascicle_agent import DQNConfig, QNet, refine_lines, train_dqn  # noqa: E402

OUT_DIR = ROOT / "reports/phase3/chase018/attention_rl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dqn-steps", type=int, default=40000)
    ap.add_argument("--ff-ckpt", type=Path, default=None, help="reuse a trained feed-forward checkpoint")
    ap.add_argument("--qnet-ckpt", type=Path, default=None, help="reuse a trained Q-net checkpoint")
    ap.add_argument("--gamma", type=float, default=DQNConfig.gamma)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "synthetic_run.json")
    ap.add_argument("--ckpt-dir", type=Path, default=OUT_DIR / "synthetic_ckpts")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_all = time.time()

    def wrap(ds, tag, offset):
        return [from_synthetic(s, f"{tag}{i:04d}", offset + i) for i, s in enumerate(ds)]

    train = wrap(make_dataset(args.n_train, seed=args.seed + 1), "tr", 0)
    val = wrap(make_dataset(args.n_val, seed=args.seed + 2), "va", 10**6)
    test = wrap(make_dataset(args.n_test, seed=args.seed + 3), "te", 2 * 10**6)

    cfg = TrainConfig(epochs=args.epochs, batch=args.batch, seed=args.seed)
    t0 = time.time()
    if args.ff_ckpt is not None:
        ck = torch.load(args.ff_ckpt, map_location="cpu", weights_only=False)
        cfg = TrainConfig(**ck["cfg"])
        model = SpatialAttentionGeoNet(AttnGeoConfig(width=cfg.width, depth=cfg.depth))
        model.load_state_dict(ck["state"])
        model.eval()
        hist = ck.get("hist", [])
    else:
        model, hist = train_ff(train, val, cfg, log=lambda r: print(json.dumps(r), flush=True))
    ff_time = time.time() - t0

    ff = {s.key: ff_outputs(model, s) for s in [*train, *val, *test]}
    source = make_episode_source(train, ff)
    t0 = time.time()
    dqn_cfg = DQNConfig(steps=args.dqn_steps, eps_steps=args.dqn_steps // 2, gamma=args.gamma, seed=args.seed)
    dqn_hist: list[dict] = []

    def log_rl(r: dict) -> None:
        dqn_hist.append(r)
        print(json.dumps(r), flush=True)

    if args.qnet_ckpt is not None:
        qk = torch.load(args.qnet_ckpt, map_location="cpu", weights_only=False)
        dqn_cfg = DQNConfig(**qk["cfg"])
        dqn_hist = qk.get("hist", [])
        qnet = QNet(3)
        qnet.load_state_dict(qk["state"])
        qnet.eval()
    else:
        qnet = train_dqn(source, dqn_cfg, log=log_rl)
    rl_time = time.time() - t0

    def refine(sample, lines):
        o = ff[sample.key]
        return refine_lines(qnet, agent_channels(sample, o), lines, band_poly_px(o["sup"]), band_poly_px(o["deep"]))

    result = {"label": "SYNTHETIC_NOT_LB", "claims_kaggle_lb": False}
    for name, split in (("val", val), ("test", test)):
        rows = geometry_rows(split, ff, refine=refine, fl_factor_maps=1.0)

        def mean_finite(key: str) -> float:
            vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
            return float(np.mean(vals)) if vals else float("nan")

        result[name] = {
            "maps_host_measure": score_rows(rows, "maps"),
            "attention_readout": score_rows(rows, "attn"),
            "attention_rl_readout": score_rows(rows, "rl"),
            "line_angle_err_deg": {"init": mean_finite("init_ang_err"), "rl": mean_finite("rl_ang_err")},
            "line_offset_err_px": {"init": mean_finite("init_off_err"), "rl": mean_finite("rl_off_err")},
            "rl_by_init_error": rl_by_init_error(rows),
            "attn_ok_rate": float(np.mean([r["attn_ok"] for r in rows])),
        }
    v = result["val"]
    result["gates"] = {
        "val_attention_beats_maps": bool(
            v["attention_readout"]["MAE_PA"] < v["maps_host_measure"]["MAE_PA"]
            and v["attention_readout"]["MAE_FL"] < v["maps_host_measure"]["MAE_FL"]
        ),
        "val_rl_beats_init": bool(
            v["line_angle_err_deg"]["rl"] < v["line_angle_err_deg"]["init"]
            and v["line_offset_err_px"]["rl"] <= v["line_offset_err_px"]["init"]
        ),
    }
    result["ff_history"] = hist
    result["dqn_history"] = dqn_hist
    result["config"] = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_cfg": cfg.__dict__,
        "dqn_cfg": dqn_cfg.__dict__,
    }
    result["seconds"] = {"ff": ff_time, "rl": rl_time, "total": time.time() - t_all}
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": cfg.__dict__, "hist": hist}, args.ckpt_dir / "ff.pt")
    torch.save({"state": qnet.state_dict(), "cfg": dqn_cfg.__dict__, "hist": dqn_hist}, args.ckpt_dir / "qnet.pt")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("val", "test", "gates", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
