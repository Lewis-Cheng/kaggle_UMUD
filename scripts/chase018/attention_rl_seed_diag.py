#!/usr/bin/env python3
"""Where does the RL line agent help? Error before and after RL, by seed type.

Runs a trained Q-net greedily on held-out synthetic test images from three
kinds of seed: feed-forward lines, small perturbations of an annotated line
(about the size of synthetic feed-forward errors), and large ones (closer to
what real feed-forward lines may look like). The target line is used only to
measure error; the agent never sees it. Synthetic only, not leaderboard
evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.data.synthetic_us import make_dataset  # noqa: E402
from umud.geometry.attended_readout import fold, lines_from_attention  # noqa: E402
from umud.models.attention_rl_train import (  # noqa: E402
    TrainConfig,
    agent_channels,
    band_poly_px,
    ff_outputs,
    from_synthetic,
    lines_from_mask,
)
from umud.models.spatial_attention_geo import AttnGeoConfig, SpatialAttentionGeoNet  # noqa: E402
from umud.rl.fascicle_agent import FascicleEnv, QNet, greedy_policy, nearest_target, perp_distance, rollout  # noqa: E402

OUT_DIR = ROOT / "reports/phase3/chase018/attention_rl"


def errors(env: FascicleEnv) -> tuple[float, float]:
    tg = env.target
    return abs(float(fold(env.t - float(tg[4])))), perp_distance(env.x0, env.y, tg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ff-ckpt", type=Path, default=OUT_DIR / "synthetic_ckpts/ff_v1.pt")
    ap.add_argument("--qnet", action="append", required=True, help="name=path; repeatable")
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "rl_seed_diag.json")
    args = ap.parse_args()
    torch.set_num_threads(4)

    test = [from_synthetic(s, f"te{i:04d}", 2 * 10**6 + i) for i, s in enumerate(make_dataset(args.n_test, seed=args.seed + 3))]
    ck = torch.load(args.ff_ckpt, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["cfg"])
    model = SpatialAttentionGeoNet(AttnGeoConfig(width=cfg.width, depth=cfg.depth))
    model.load_state_dict(ck["state"])
    model.eval()

    rng = np.random.default_rng(args.seed + 99)
    seeds: dict[str, list] = {"ff": [], "small": [], "large": []}
    for s in test:
        targets = lines_from_mask(s.fasc_mask)
        if len(targets) == 0:
            continue
        o = ff_outputs(model, s)
        ch, sp, dp = agent_channels(s, o), band_poly_px(o["sup"]), band_poly_px(o["deep"])
        for ln in lines_from_attention(o["attn"], o["orient"]):
            tg = targets[nearest_target(ln.x, ln.y, ln.t_deg, targets)]
            seeds["ff"].append((ch, ln.x, ln.y, ln.t_deg, tg, sp, dp))
        for _ in range(args.per_image):
            for kind, (sd_y, sd_t) in (("small", (2.0, 2.0)), ("large", (4.0, None))):
                tg = targets[int(rng.integers(len(targets)))]
                u = rng.uniform(0.2, 0.8)
                x0 = tg[0] + u * (tg[2] - tg[0])
                y0 = tg[1] + u * (tg[3] - tg[1]) + float(np.clip(rng.normal(0, sd_y), -8, 8))
                dt = rng.uniform(-12, 12) if sd_t is None else float(np.clip(rng.normal(0, sd_t), -12, 12))
                seeds[kind].append((ch, float(x0), float(y0), float(tg[4]) + dt, tg, sp, dp))

    result: dict = {"label": "SYNTHETIC_NOT_LB", "n_test_images": len(test), "agents": {}}
    for spec in args.qnet:
        name, path = spec.split("=", 1)
        q = QNet(3)
        q.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state"])
        q.eval()
        policy = greedy_policy(q)
        per_kind = {}
        for kind, rows in seeds.items():
            a0, o0, a1, o1, moves = [], [], [], [], []
            for ch, x, y, t, tg, sp, dp in rows:
                env = FascicleEnv(ch, x, y, t, tg, sp, dp)
                e0 = errors(env)
                rollout(env, policy)
                e1 = errors(env)
                a0.append(e0[0]), o0.append(e0[1]), a1.append(e1[0]), o1.append(e1[1]), moves.append(env.moves)
            a0, o0, a1, o1 = map(np.asarray, (a0, o0, a1, o1))
            per_kind[kind] = {
                "n": len(rows),
                "angle_err_deg": {"init": float(a0.mean()), "rl": float(a1.mean())},
                "offset_err_px": {"init": float(o0.mean()), "rl": float(o1.mean())},
                "frac_angle_improved": float(np.mean(a1 < a0 - 1e-9)),
                "frac_angle_worse": float(np.mean(a1 > a0 + 1e-9)),
                "mean_moves": float(np.mean(moves)),
                "frac_stop_at_once": float(np.mean(np.asarray(moves) == 0)),
            }
        result["agents"][name] = per_kind
        print(name, json.dumps(per_kind, indent=1), flush=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
