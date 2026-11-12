"""Training and evaluation shared by the synthetic run and the lab run.

Supervision comes from masks only: annotated fascicle mask, superficial and
deep band masks. Orientation targets are recomputed from the augmented mask
so anisotropic crops cannot desynchronise them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from umud.geometry.attended_readout import PixelLine, fold, lines_from_attention, readout
from umud.geometry.host_faithful import measure
from umud.models.spatial_attention_geo import AttnGeoConfig, SpatialAttentionGeoNet, predict_symmetric
from umud.rl.fascicle_agent import (
    FascicleEnv,
    line_errors,
    nearest_target,
    perp_distance,
)


@dataclass
class GeoSample:
    key: str
    image: np.ndarray  # float32 HxW in [0, 1]
    p_apo: np.ndarray
    p_fasc: np.ndarray
    mx: float
    my: float
    group: int = 0
    sup_mask: np.ndarray | None = None
    deep_mask: np.ndarray | None = None
    fasc_mask: np.ndarray | None = None
    labels: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def x(self) -> np.ndarray:
        return np.stack([self.image, self.p_apo, self.p_fasc]).astype(np.float32)


def from_synthetic(s, key: str, group: int) -> GeoSample:
    return GeoSample(
        key=key,
        image=s.image.astype(np.float32) / 255.0,
        p_apo=s.p_apo,
        p_fasc=s.p_fasc,
        mx=s.mm_per_px,
        my=s.mm_per_px,
        group=group,
        sup_mask=s.sup_mask,
        deep_mask=s.deep_mask,
        fasc_mask=s.fasc_mask,
        labels=dict(s.labels),
    )


def lines_from_mask(mask: np.ndarray, min_px: int = 6) -> np.ndarray:
    """Annotated segments as (x1, y1, x2, y2, angle_deg) from mask components."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    rows = []
    for i in range(1, n):
        if st[i, 4] < min_px:
            continue
        ys, xs = np.nonzero(lab == i)
        P = np.stack([xs, ys], 1).astype(np.float64)
        c = P.mean(0)
        _, s, vt = np.linalg.svd(P - c, full_matrices=False)
        if s[0] < 2 * (s[1] + 1e-6):
            continue
        proj = (P - c) @ vt[0]
        p1, p2 = c + proj.min() * vt[0], c + proj.max() * vt[0]
        ang = float(fold(np.degrees(np.arctan2(vt[0][1], vt[0][0]))))
        rows.append((p1[0], p1[1], p2[0], p2[1], ang))
    return np.asarray(rows, np.float64).reshape(-1, 5)


def orientation_target(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    o = np.zeros((2, h, w), np.float32)
    valid = np.zeros((h, w), bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if st[i, 4] < 4:
            continue
        m = lab == i
        ys, xs = np.nonzero(m)
        P = np.stack([xs, ys], 1).astype(np.float64)
        _, s, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
        if s[0] < 2 * (s[1] + 1e-6):
            continue
        t = np.arctan2(vt[0][1], vt[0][0])
        o[0][m], o[1][m] = np.cos(2 * t), np.sin(2 * t)
        valid |= m
    return o, valid


def augment(sample: GeoSample, rng: np.random.Generator, crop_min: float = 0.75) -> dict[str, np.ndarray]:
    h, w = sample.image.shape
    stack = np.stack(
        [sample.image, sample.p_apo, sample.p_fasc, sample.fasc_mask, sample.sup_mask, sample.deep_mask]
    ).astype(np.float32)
    ch, cw = int(h * rng.uniform(crop_min, 1.0)), int(w * rng.uniform(crop_min, 1.0))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    stack = stack[:, y0 : y0 + ch, x0 : x0 + cw]
    stack = np.stack(
        [cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR if i < 3 else cv2.INTER_NEAREST) for i, c in enumerate(stack)]
    )
    if rng.random() < 0.5:
        stack = stack[:, :, ::-1].copy()
    img = stack[0]
    img = np.clip((img ** rng.uniform(0.7, 1.4)) * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1)
    if rng.random() < 0.3:
        img = np.clip(img * rng.normal(1, 0.08, img.shape), 0, 1)
    stack[0] = img
    o, valid = orientation_target(stack[3])
    return {"x": stack[:3], "attn": stack[3], "sup": stack[4], "deep": stack[5], "orient": o, "valid": valid}


def dice_loss(logit: torch.Tensor, y: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2)) + y.sum((1, 2)) + eps)).mean()


@dataclass
class TrainConfig:
    epochs: int = 20
    batch: int = 16
    lr: float = 2e-3
    l2: float = 1e-4
    pos_weight: float = 4.0
    crop_min: float = 0.75
    w_orient: float = 1.0
    seed: int = 0
    width: int = 16
    depth: int = 3


def ff_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: TrainConfig) -> tuple[torch.Tensor, dict[str, float]]:
    pw = torch.tensor(cfg.pos_weight, device=out["attn"].device)
    l_attn = F.binary_cross_entropy_with_logits(out["attn"], batch["attn"], pos_weight=pw) + dice_loss(out["attn"], batch["attn"])
    l_sup = F.binary_cross_entropy_with_logits(out["sup"], batch["sup"]) + dice_loss(out["sup"], batch["sup"])
    l_deep = F.binary_cross_entropy_with_logits(out["deep"], batch["deep"]) + dice_loss(out["deep"], batch["deep"])
    valid = batch["valid"]
    if valid.any():
        diff = ((out["orient"] - batch["orient"]) ** 2).sum(1)
        l_or = diff[valid].mean()
    else:
        l_or = out["orient"].sum() * 0.0
    loss = l_attn + l_sup + l_deep + cfg.w_orient * l_or
    return loss, {"attn": l_attn.item(), "sup": l_sup.item(), "deep": l_deep.item(), "orient": l_or.item()}


def _batch(items: list[dict[str, np.ndarray]], device) -> dict[str, torch.Tensor]:
    return {
        "x": torch.from_numpy(np.stack([i["x"] for i in items])).to(device),
        "attn": torch.from_numpy(np.stack([i["attn"] for i in items])).float().to(device),
        "sup": torch.from_numpy(np.stack([i["sup"] for i in items])).float().to(device),
        "deep": torch.from_numpy(np.stack([i["deep"] for i in items])).float().to(device),
        "orient": torch.from_numpy(np.stack([i["orient"] for i in items])).to(device),
        "valid": torch.from_numpy(np.stack([i["valid"] for i in items])).to(device),
    }


def _dice(p: np.ndarray, y: np.ndarray) -> float:
    p, y = p > 0.5, y > 0.5
    return float((2 * (p & y).sum() + 1) / (p.sum() + y.sum() + 1))


def train_ff(
    train: Sequence[GeoSample],
    val: Sequence[GeoSample],
    cfg: TrainConfig,
    device: str | torch.device = "cpu",
    log: Callable[[dict], None] | None = None,
) -> tuple[SpatialAttentionGeoNet, list[dict]]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = SpatialAttentionGeoNet(AttnGeoConfig(width=cfg.width, depth=cfg.depth)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    steps = max(1, len(train) // cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.epochs * steps, pct_start=0.15)
    best, best_state, hist = -1.0, None, []
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(train))
        tot = 0.0
        for b in range(steps):
            items = [augment(train[i], rng, cfg.crop_min) for i in perm[b * cfg.batch : (b + 1) * cfg.batch]]
            batch = _batch(items, device)
            out = model(batch["x"])
            loss, _ = ff_loss(out, batch, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.item())
        metrics = validate_ff(model, val, device)
        row = {"epoch": ep, "train_loss": tot / steps, **metrics}
        hist.append(row)
        if log is not None:
            log(row)
        score = metrics["val_attn_dice"] + 0.5 * (metrics["val_sup_dice"] + metrics["val_deep_dice"])
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def ff_outputs(model: SpatialAttentionGeoNet, sample: GeoSample, device: str | torch.device = "cpu") -> dict[str, np.ndarray]:
    out = predict_symmetric(model, torch.from_numpy(sample.x())[None].to(device))
    return {k: v[0].cpu().numpy() for k, v in out.items()}


def validate_ff(model: SpatialAttentionGeoNet, val: Sequence[GeoSample], device) -> dict[str, float]:
    d_attn, d_sup, d_deep, ang = [], [], [], []
    for s in val:
        o = ff_outputs(model, s, device)
        d_attn.append(_dice(o["attn"], s.fasc_mask))
        d_sup.append(_dice(o["sup"], s.sup_mask))
        d_deep.append(_dice(o["deep"], s.deep_mask))
        tgt, valid = orientation_target(s.fasc_mask)
        if valid.any():
            tp = 0.5 * np.degrees(np.arctan2(o["orient"][1][valid], o["orient"][0][valid]))
            tt = 0.5 * np.degrees(np.arctan2(tgt[1][valid], tgt[0][valid]))
            ang.append(float(np.mean(np.abs(fold(tp - tt)))))
    return {
        "val_attn_dice": float(np.mean(d_attn)),
        "val_sup_dice": float(np.mean(d_sup)),
        "val_deep_dice": float(np.mean(d_deep)),
        "val_orient_err_deg": float(np.mean(ang)) if ang else float("nan"),
    }


def band_poly_px(p: np.ndarray, thr: float = 0.5) -> np.ndarray | None:
    ys, xs = np.nonzero(p > thr)
    if len(xs) < 10 or np.ptp(xs) < 5:
        return None
    return np.polyfit(xs.astype(float), ys.astype(float), 2 if np.ptp(xs) > 30 else 1)


def agent_channels(sample: GeoSample, ff: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([sample.image, sample.p_fasc, ff["attn"]]).astype(np.float32)


def make_episode_source(
    samples: Sequence[GeoSample],
    ff_by_key: dict[str, dict[str, np.ndarray]],
    p_ff_seed: float = 0.3,
    p_small: float = 0.4,
    small_angle_sd: float = 2.0,
    small_offset_sd: float = 2.0,
    max_seed_angle: float = 12.0,
    max_seed_offset: float = 8.0,
) -> Callable[[np.random.Generator], FascicleEnv]:
    """Seeds: feed-forward lines matched to a target, small target perturbations
    (the size of feed-forward errors), and large ones."""
    prepared = []
    for s in samples:
        targets = lines_from_mask(s.fasc_mask)
        if len(targets) == 0:
            continue
        ff = ff_by_key[s.key]
        ff_lines = lines_from_attention(ff["attn"], ff["orient"])
        prepared.append((s, targets, ff_lines, agent_channels(s, ff), band_poly_px(ff["sup"]), band_poly_px(ff["deep"])))
    if not prepared:
        raise ValueError("no sample has annotated lines")

    def source(rng: np.random.Generator) -> FascicleEnv:
        s, targets, ff_lines, ch, sp, dp = prepared[int(rng.integers(len(prepared)))]
        if ff_lines and rng.random() < p_ff_seed:
            ln = ff_lines[int(rng.integers(len(ff_lines)))]
            k = nearest_target(ln.x, ln.y, ln.t_deg, targets)
            tg = targets[k]
            if abs(float(fold(ln.t_deg - tg[4]))) <= max_seed_angle and perp_distance(ln.x, ln.y, tg) <= 2 * max_seed_offset:
                return FascicleEnv(ch, ln.x, ln.y, ln.t_deg, tg, sp, dp)
        tg = targets[int(rng.integers(len(targets)))]
        u = rng.uniform(0.2, 0.8)
        x0 = tg[0] + u * (tg[2] - tg[0])
        y_on = tg[1] + u * (tg[3] - tg[1])
        if rng.random() < p_small / max(1e-9, 1.0 - p_ff_seed):
            dy = rng.normal(0, small_offset_sd)
            dt = rng.normal(0, small_angle_sd)
        else:
            dy = rng.normal(0, max_seed_offset / 2)
            dt = rng.uniform(-max_seed_angle, max_seed_angle)
        y0 = y_on + float(np.clip(dy, -max_seed_offset, max_seed_offset))
        t0 = float(tg[4]) + float(np.clip(dt, -max_seed_angle, max_seed_angle))
        return FascicleEnv(ch, float(x0), float(y0), t0, tg, sp, dp)

    return source


def geometry_rows(
    samples: Sequence[GeoSample],
    ff_by_key: dict[str, dict[str, np.ndarray]],
    refine: Callable[[GeoSample, list[PixelLine]], list[PixelLine]] | None = None,
    fl_factor_maps: float = 0.9,
) -> list[dict[str, Any]]:
    """Per sample: host measure on maps, attended readout, and attended+RL readout."""
    rows = []
    for s in samples:
        ff = ff_by_key[s.key]
        base = measure(s.p_apo, s.p_fasc, s.mx, s.my)
        lines = lines_from_attention(ff["attn"], ff["orient"])
        att = readout(ff["sup"], ff["deep"], lines, s.mx, s.my)
        row = {
            "key": s.key,
            "maps_pa": float(base["pa_deg"]),
            "maps_fl": float(base["fl_mm"]) * fl_factor_maps,
            "maps_mt": float(base["mt_mm"]),
            "maps_ok": bool(base["ok"]),
            "attn_pa": att["pa_deg"],
            "attn_fl": att["fl_mm"],
            "attn_mt": att["mt_mm"],
            "attn_ok": att["ok"],
            "n_lines": att["n_lines"],
        }
        if refine is not None:
            rl_lines = refine(s, lines)
            rl = readout(ff["sup"], ff["deep"], rl_lines, s.mx, s.my)
            row.update(rl_pa=rl["pa_deg"], rl_fl=rl["fl_mm"], rl_mt=rl["mt_mm"], rl_ok=rl["ok"])
            if s.fasc_mask is not None:
                targets = lines_from_mask(s.fasc_mask)
                row["init_ang_err"], row["init_off_err"] = line_errors(lines, targets)
                row["rl_ang_err"], row["rl_off_err"] = line_errors(rl_lines, targets)
        if s.labels is not None:
            row.update({f"gt_{k.split('_')[0]}": v for k, v in s.labels.items()})
        rows.append(row)
    return rows


def rl_by_init_error(rows: list[dict[str, Any]], edges: Sequence[float] = (0.0, 1.0, 2.0, 4.0, float("inf"))) -> list[dict[str, float]]:
    """Mean line angle error before and after RL, binned by the image's feed-forward error.

    On synthetic data the agent hurt seeds under about 1 deg and helped seeds
    of several degrees, so the useful question on real data is which bin the
    feed-forward lines fall into.
    """
    out = []
    pairs = [(r["init_ang_err"], r["rl_ang_err"]) for r in rows if np.isfinite(r.get("init_ang_err", np.nan)) and np.isfinite(r.get("rl_ang_err", np.nan))]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(a, b) for a, b in pairs if lo <= a < hi]
        out.append({
            "init_deg_from": lo,
            "init_deg_to": hi,
            "n": len(sel),
            "init": float(np.mean([a for a, _ in sel])) if sel else float("nan"),
            "rl": float(np.mean([b for _, b in sel])) if sel else float("nan"),
        })
    return out


def score_rows(rows: list[dict[str, Any]], prefix: str, fallback_prefix: str = "maps") -> dict[str, float]:
    """MAE and S against gt_*; missing values fall back to the maps prediction (abstain)."""
    if not rows:
        nan = float("nan")
        return {"MAE_PA": nan, "MAE_FL": nan, "MAE_MT": nan, "S1": nan, "n": 0, "n_fallback_values": 0}
    err = {"pa": [], "fl": [], "mt": []}
    n_fallback = 0
    for r in rows:
        for k in err:
            v = r.get(f"{prefix}_{k}", float("nan"))
            if not np.isfinite(v):
                v = r[f"{fallback_prefix}_{k}"]
                n_fallback += 1
            err[k].append(abs(v - r[f"gt_{k}"]))
    mae = {k: float(np.mean(v)) for k, v in err.items()}
    return {
        "MAE_PA": mae["pa"],
        "MAE_FL": mae["fl"],
        "MAE_MT": mae["mt"],
        "S1": float(np.mean([mae["pa"] / 6.0, mae["fl"] / 12.0, mae["mt"] / 3.0])),
        "n": len(rows),
        "n_fallback_values": n_fallback,
    }
