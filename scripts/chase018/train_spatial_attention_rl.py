#!/usr/bin/env python3
"""Spatial-attention + medical-RL geometry on real data (lab GPU).

Stages, run in order:

  prepare       apo∩fasc competition images -> grid cache with Frundles maps,
                annotated fascicle mask, and superficial/deep band masks.
                MultExp near-duplicates (dHash <= 8) are dropped and re-checked.
  train         feed-forward attention net, then the DQN line agent, then the
                held-out (dHash group) validation against mask geometry.
  eval-multexp  stress test on the 35 MultExp images no model saw.
  infer-test    309 official ids -> PA-only, FL-only, and PA+FL CSVs. Targets
                that are not replaced, and images where the readout abstains,
                are copied from texture_rl.

Spec: reports/phase3/chase018/attention_rl/SPEC.md. Gates are enforced by
scripts/chase018/fv_attention_rl_stages.py. No Kaggle submission happens here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_official_submission_ids, write_submission_csv  # noqa: E402
from umud.calibration.anisotropic_scale import mm_per_prob_pixel  # noqa: E402
from umud.data.holdout import MULTEXP_MAX_DIST, assert_multexp_excluded, near_any  # noqa: E402
from umud.geometry.attended_readout import lines_from_attention, readout  # noqa: E402
from umud.geometry.host_faithful import dhash, groups_from_hashes, measure  # noqa: E402
from umud.metric.score import score_components  # noqa: E402
from umud.models.attention_rl_train import (  # noqa: E402
    GeoSample,
    TrainConfig,
    agent_channels,
    band_poly_px,
    ff_outputs,
    geometry_rows,
    make_episode_source,
    rl_by_init_error,
    score_rows,
    train_ff,
)
from umud.models.spatial_attention_geo import AttnGeoConfig, SpatialAttentionGeoNet  # noqa: E402
from umud.rl.fascicle_agent import DQNConfig, QNet, refine_lines, train_dqn  # noqa: E402

MAP_SIZE = (384, 576)
MASK_PROXY_MM_PER_PX = 1.0 / 14.0
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class Layout:
    root: Path
    out: Path

    def p(self, rel: str) -> Path:
        return self.root / rel

    @property
    def apo_img(self) -> Path:
        return self.p("data/raw/umud/apo_imgs_v1")

    @property
    def apo_mask(self) -> Path:
        return self.p("data/raw/umud/apo_masks_v1")

    @property
    def fasc_img(self) -> Path:
        return self.p("data/raw/umud/fasc_imgs_v1")

    @property
    def fasc_mask(self) -> Path:
        return self.p("data/raw/umud/fasc_masks_v1")

    @property
    def weights(self) -> Path:
        return self.p("data/external/frundles_weights")

    @property
    def expert_csv(self) -> Path:
        return self.p("data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")

    @property
    def expert_dir(self) -> Path:
        return self.p("data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0")

    @property
    def multexp_maps(self) -> Path:
        return self.p("reports/phase3/chase018/cache/maps_multexp")

    @property
    def train_maps(self) -> Path:
        return self.p("reports/phase3/chase018/cache/maps_train_overlap")

    @property
    def test_dir(self) -> Path:
        return self.p("data/raw/umud/test_images_v2/test_set_v2")

    @property
    def test_ids(self) -> Path:
        return self.p("reports/phase3/official_test_file_names.txt")

    @property
    def scale_csv(self) -> Path:
        return self.p("reports/phase3/chase018/scale_aniso.csv")

    @property
    def test_maps(self) -> Path:
        return self.p("reports/phase3/chase018/cache/maps_test")

    @property
    def texture_rl(self) -> Path:
        return self.p("reports/phase3/chase018/submission_texture_rl.csv")

    @property
    def grid_cache(self) -> Path:
        return self.out / "grid_cache"


def _index(folder: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if folder.exists():
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMG_EXT:
                found.setdefault(path.stem, path)
    return found


class MapSource:
    """Frundles apo+fasc maps at 384x576, read from an npz cache when present."""

    def __init__(self, weights: Path, cache: Path, device: str):
        self.weights, self.cache, self.device = weights, cache, device
        self.models = None
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        path = self.cache / f"{name}.npz"
        if path.exists():
            z = np.load(path)
            return z["apo"].astype(np.float32), z["fasc"].astype(np.float32)
        if self.models is None:
            from umud.seg.frundles_smp import load_pair

            self.models = load_pair(self.weights, device=self.device)
        from umud.seg.frundles_smp import predict_prob

        m_apo, s_apo = self.models["apo"]
        m_fasc, s_fasc = self.models["fasc"]
        pa = predict_prob(m_apo, s_apo, gray, self.device)
        pf = predict_prob(m_fasc, s_fasc, gray, self.device)
        np.savez_compressed(path, apo=pa.astype(np.float16), fasc=pf.astype(np.float16))
        return pa, pf


def split_bands(apo: np.ndarray, merge_frac: float = 0.04) -> tuple[np.ndarray, np.ndarray] | None:
    """Shallowest and deepest band groups of a GT apo mask."""
    h, w = apo.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(apo.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        if st[i, 2] < 0.03 * w:
            continue
        ys, xs = np.nonzero(lab == i)
        depth = float(np.polyval(np.polyfit(xs, ys, 1), w / 2.0)) if np.ptp(xs) > 0 else float(ys.mean())
        comps.append((depth, i))
    if len(comps) < 2:
        return None
    comps.sort()
    groups: list[list[tuple[float, int]]] = [[comps[0]]]
    for d, i in comps[1:]:
        if d - groups[-1][-1][0] < merge_frac * h:
            groups[-1].append((d, i))
        else:
            groups.append([(d, i)])
    if len(groups) < 2:
        return None
    sup = np.isin(lab, [i for _, i in groups[0]]).astype(np.uint8)
    deep = np.isin(lab, [i for _, i in groups[-1]]).astype(np.uint8)
    return sup, deep


def to_grid(arr: np.ndarray, grid: tuple[int, int], kind: str) -> np.ndarray:
    gh, gw = grid
    if kind == "image":
        return cv2.resize(arr, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if kind == "map":
        return cv2.resize(arr.astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
    if kind == "thin":
        return (cv2.resize(arr.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.25).astype(np.uint8)
    return (cv2.resize(arr.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"unreadable mask {path}")
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def multexp_hashes(lay: Layout) -> tuple[pd.DataFrame, list[np.ndarray]]:
    expert = pd.read_csv(lay.expert_csv)
    hashes = []
    for name in expert["image_file"].astype(str):
        g = cv2.imread(str(lay.expert_dir / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise SystemExit(f"MultExp image missing: {name}; the holdout cannot be checked")
        hashes.append(dhash(g))
    return expert, hashes


def cmd_prepare(args, lay: Layout) -> int:
    grid = tuple(args.grid)
    apo_i, apo_m, fasc_i, fasc_m = map(_index, (lay.apo_img, lay.apo_mask, lay.fasc_img, lay.fasc_mask))
    stems = sorted(set(apo_i) & set(apo_m) & set(fasc_i) & set(fasc_m))
    if args.limit:
        stems = stems[: args.limit]
    _, mex = multexp_hashes(lay)
    maps = MapSource(lay.weights, lay.train_maps, args.device)
    lay.grid_cache.mkdir(parents=True, exist_ok=True)
    kept, hashes, shapes, dropped, no_bands = [], [], {}, [], []
    for k, stem in enumerate(stems):
        gray = cv2.imread(str(apo_i[stem]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h = dhash(gray)
        if near_any(h, mex, MULTEXP_MAX_DIST):
            dropped.append(stem)
            continue
        bands = split_bands(read_mask(apo_m[stem], gray.shape))
        if bands is None:
            no_bands.append(stem)
            continue
        fasc = read_mask(fasc_m[stem], gray.shape)
        p_apo, p_fasc = maps.get(stem, gray)
        np.savez_compressed(
            lay.grid_cache / f"{stem}.npz",
            image=(to_grid(gray, grid, "image") * 255).astype(np.uint8),
            p_apo=to_grid(p_apo, grid, "map").astype(np.float16),
            p_fasc=to_grid(p_fasc, grid, "map").astype(np.float16),
            fasc=to_grid(fasc, grid, "thin"),
            sup=to_grid(bands[0], grid, "band"),
            deep=to_grid(bands[1], grid, "band"),
        )
        kept.append(stem)
        hashes.append(h)
        shapes[stem] = list(gray.shape)
        if k % 100 == 0:
            print(json.dumps({"prepared": k + 1, "of": len(stems)}), flush=True)
    assert_multexp_excluded(hashes, mex, kept)
    groups = groups_from_hashes(hashes) if hashes else np.zeros(0, int)
    manifest = {
        "grid": list(grid),
        "stems": kept,
        "orig_shape": shapes,
        "group": {s: int(g) for s, g in zip(kept, groups)},
        "dropped_near_multexp": dropped,
        "no_two_bands": no_bands,
        "multexp_max_dist": MULTEXP_MAX_DIST,
        "multexp_in_train": False,
    }
    (lay.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"kept": len(kept), "dropped_near_multexp": len(dropped), "no_two_bands": len(no_bands)}), flush=True)
    return 0


def load_samples(lay: Layout) -> tuple[list[GeoSample], dict]:
    manifest = json.loads((lay.out / "manifest.json").read_text())
    gh, gw = manifest["grid"]
    samples = []
    for stem in manifest["stems"]:
        z = np.load(lay.grid_cache / f"{stem}.npz")
        h0, w0 = manifest["orig_shape"][stem]
        mx = (w0 / gw) * MASK_PROXY_MM_PER_PX
        my = (h0 / gh) * MASK_PROXY_MM_PER_PX
        s = GeoSample(
            key=stem,
            image=z["image"].astype(np.float32) / 255.0,
            p_apo=z["p_apo"].astype(np.float32),
            p_fasc=z["p_fasc"].astype(np.float32),
            mx=mx,
            my=my,
            group=int(manifest["group"][stem]),
            sup_mask=z["sup"],
            deep_mask=z["deep"],
            fasc_mask=z["fasc"],
        )
        ref = measure(np.maximum(s.sup_mask, s.deep_mask).astype(np.float32), s.fasc_mask.astype(np.float32), mx, my)
        if ref["ok"]:
            s.labels = {"pa_deg": float(ref["pa_deg"]), "fl_mm": float(ref["fl_mm"]), "mt_mm": float(ref["mt_mm"])}
        samples.append(s)
    return samples, manifest


def group_split(samples: list[GeoSample], val_frac: float, seed: int) -> tuple[list[GeoSample], list[GeoSample]]:
    groups = sorted({s.group for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_g = set(groups[: max(1, int(round(len(groups) * val_frac)))])
    train = [s for s in samples if s.group not in val_g]
    val = [s for s in samples if s.group in val_g]
    if {s.group for s in train} & {s.group for s in val}:
        raise SystemExit("dHash group in both train and val")
    return train, val


def _writer(lay: Layout):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(lay.out / "tb"))
    except Exception:
        return None


def load_models(lay: Layout, device: str) -> tuple[SpatialAttentionGeoNet, QNet | None, dict, tuple[int, int]]:
    ck = torch.load(lay.out / "ff.pt", map_location="cpu", weights_only=False)
    model = SpatialAttentionGeoNet(AttnGeoConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    qnet = None
    qpath = lay.out / "qnet.pt"
    if qpath.exists():
        qnet = QNet(3)
        qnet.load_state_dict(torch.load(qpath, map_location="cpu", weights_only=False)["state"])
        qnet.to(device).eval()
    report_path = lay.out / "val_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return model, qnet, report, (int(ck["grid"][0]), int(ck["grid"][1]))


def cmd_train(args, lay: Layout) -> int:
    samples, manifest = load_samples(lay)
    train, val = group_split(samples, args.val_frac, args.seed)
    print(json.dumps({"train": len(train), "val": len(val), "val_with_mask_geometry": sum(s.labels is not None for s in val)}), flush=True)
    writer = _writer(lay)

    def log_ff(row: dict) -> None:
        print(json.dumps(row), flush=True)
        if writer is not None:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(f"ff/{k}", v, row["epoch"])
            writer.flush()

    cfg = TrainConfig(
        epochs=args.epochs, batch=args.batch, lr=args.lr, l2=args.l2, pos_weight=args.pos_weight,
        crop_min=args.crop_min, seed=args.seed, width=args.width, depth=args.depth,
    )
    t0 = time.time()
    model, hist = train_ff(train, val, cfg, device=args.device, log=log_ff)
    torch.save(
        {"state": {k: v.cpu() for k, v in model.state_dict().items()}, "model_cfg": asdict(AttnGeoConfig(width=cfg.width, depth=cfg.depth)),
         "train_cfg": asdict(cfg), "grid": manifest["grid"], "multexp_in_train": False},
        lay.out / "ff.pt",
    )
    ff = {s.key: ff_outputs(model, s, args.device) for s in [*train, *val]}

    def log_rl(row: dict) -> None:
        print(json.dumps(row), flush=True)
        if writer is not None:
            for k, v in row.items():
                if k != "step":
                    writer.add_scalar(f"rl/{k}", v, row["step"])
            writer.flush()

    qnet = train_dqn(
        make_episode_source(train, ff),
        DQNConfig(steps=args.dqn_steps, eps_steps=args.dqn_steps // 2, gamma=args.gamma, seed=args.seed),
        device=args.device,
        log=log_rl,
    )
    torch.save({"state": {k: v.cpu() for k, v in qnet.state_dict().items()}}, lay.out / "qnet.pt")

    def refine(sample, lines):
        o = ff[sample.key]
        return refine_lines(qnet, agent_channels(sample, o), lines, band_poly_px(o["sup"]), band_poly_px(o["deep"]), args.device)

    scored = [s for s in val if s.labels is not None]
    if len(scored) < args.min_val_scored:
        raise SystemExit(f"only {len(scored)} validation images have mask geometry; gates need {args.min_val_scored}")
    rows = geometry_rows(scored, ff, refine=refine, fl_factor_maps=1.0)

    def mean_finite(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    maps, attn, rl = score_rows(rows, "maps"), score_rows(rows, "attn"), score_rows(rows, "rl")
    ang = {"init": mean_finite("init_ang_err"), "rl": mean_finite("rl_ang_err")}
    off = {"init": mean_finite("init_off_err"), "rl": mean_finite("rl_off_err")}
    gates = {
        "val_attention_beats_maps": bool(attn["MAE_PA"] < maps["MAE_PA"] and attn["MAE_FL"] < maps["MAE_FL"]),
        "val_rl_beats_init": bool(ang["rl"] < ang["init"] and off["rl"] <= off["init"]),
    }
    report = {
        "label": "MASK_GEOMETRY_PROXY_NOT_LB",
        "claims_kaggle_lb": False,
        "mm_per_px_proxy": MASK_PROXY_MM_PER_PX,
        "n_train": len(train),
        "n_val_scored": len(rows),
        "vs_mask_geometry": {"maps_host_measure": maps, "attention_readout": attn, "attention_rl_readout": rl},
        "line_angle_err_deg": ang,
        "line_offset_err_px": off,
        "rl_by_init_error": rl_by_init_error(rows),
        "gates": gates,
        "rl_enabled": gates["val_rl_beats_init"],
        "ff_history": hist,
        "seconds": time.time() - t0,
    }
    (lay.out / "val_report.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(lay.out / "val_rows.csv", index=False)
    print(json.dumps({k: report[k] for k in ("vs_mask_geometry", "line_angle_err_deg", "line_offset_err_px", "gates")}, indent=2), flush=True)
    return 0


def predict_image(
    gray: np.ndarray,
    p_apo: np.ndarray,
    p_fasc: np.ndarray,
    sx: float,
    sy: float,
    model: SpatialAttentionGeoNet,
    qnet: QNet | None,
    grid: tuple[int, int],
    device: str,
) -> dict:
    h0, w0 = gray.shape
    gh, gw = grid
    mx, my = mm_per_prob_pixel(sx, sy, w0, h0, gw, gh)
    s = GeoSample(key="x", image=to_grid(gray, grid, "image"), p_apo=to_grid(p_apo, grid, "map"), p_fasc=to_grid(p_fasc, grid, "map"), mx=mx, my=my)
    o = ff_outputs(model, s, device)
    lines = lines_from_attention(o["attn"], o["orient"])
    if qnet is not None and lines:
        lines = refine_lines(qnet, agent_channels(s, o), lines, band_poly_px(o["sup"]), band_poly_px(o["deep"]), device)
    out = readout(o["sup"], o["deep"], lines, mx, my)
    bh, bw = p_apo.shape
    bmx, bmy = mm_per_prob_pixel(sx, sy, w0, h0, bw, bh)
    base = measure(p_apo, p_fasc, bmx, bmy)
    return {
        "pa_deg": out["pa_deg"], "fl_mm": out["fl_mm"], "mt_mm": out["mt_mm"], "ok": out["ok"], "n_lines": out["n_lines"],
        "base_pa": float(base["pa_deg"]), "base_fl": float(base["fl_mm"]) * 0.9, "base_mt": float(base["mt_mm"]), "base_ok": bool(base["ok"]),
    }


def _use_rl(args, qnet, report) -> QNet | None:
    if args.use_rl == "off" or qnet is None:
        return None
    if args.use_rl == "auto" and not report.get("rl_enabled", False):
        return None
    return qnet


def _loocv_affine(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred)
    for i in range(len(pred)):
        m = np.ones(len(pred), bool)
        m[i] = False
        a, b = np.polyfit(pred[m], gt[m], 1)
        out[i] = a * pred[i] + b
    return out


def _s(pa, fl, mt, gpa, gfl, gmt) -> dict:
    err = {"pa_deg": np.abs(pa - gpa), "fl_mm": np.abs(fl - gfl), "mt_mm": np.abs(mt - gmt)}
    comps = score_components(err)
    return {"S_expert_NOT_LB": float(comps["S"]), **{f"MAE_{k}": float(np.mean(v)) for k, v in err.items()}}


def cmd_eval_multexp(args, lay: Layout) -> int:
    model, qnet, report, grid = load_models(lay, args.device)
    qnet = _use_rl(args, qnet, report)
    expert = pd.read_csv(lay.expert_csv)
    rows = []
    for _, er in expert.iterrows():
        name = str(er["image_file"])
        gray = cv2.imread(str(lay.expert_dir / name), cv2.IMREAD_GRAYSCALE)
        zpath = lay.multexp_maps / f"{name}.npz"
        if gray is None or not zpath.exists():
            continue
        z = np.load(zpath)
        px_per_mm = float(er["Scale_pixel_per_cm"]) / 10.0
        r = predict_image(gray, z["apo"].astype(np.float32), z["fasc"].astype(np.float32), px_per_mm, px_per_mm, model, qnet, grid, args.device)
        r.update(image_file=name, gt_pa=float(er["median_PA"]), gt_fl=float(er["median_FL"]), gt_mt=float(er["median_MT"]))
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no MultExp image with cached maps")
    cand = {k: np.where(np.isfinite(df[k]), df[k], df[f"base_{k.split('_')[0]}"]) for k in ("pa_deg", "fl_mm", "mt_mm")}
    g = (df.gt_pa.values, df.gt_fl.values, df.gt_mt.values)
    base_raw = _s(df.base_pa.values, df.base_fl.values, df.base_mt.values, *g)
    cand_raw = _s(cand["pa_deg"], cand["fl_mm"], cand["mt_mm"], *g)
    base_cv = _s(_loocv_affine(df.base_pa.values, g[0]), df.base_fl.values, _loocv_affine(df.base_mt.values, g[2]), *g)
    cand_cv = _s(_loocv_affine(cand["pa_deg"], g[0]), cand["fl_mm"], _loocv_affine(cand["mt_mm"], g[2]), *g)
    pa_a, pa_b = np.polyfit(cand["pa_deg"], g[0], 1)
    out = {
        "label": "S_expert_NOT_LB",
        "claims_kaggle_lb": False,
        "n": int(len(df)),
        "rl_used": qnet is not None,
        "readout_ok_rate": float(df.ok.mean()),
        "baseline_host_measure": {"raw": base_raw, "loocv_pa_mt_affine": base_cv},
        "candidate": {"raw": cand_raw, "loocv_pa_mt_affine": cand_cv},
        "candidate_pa_affine_full_fit": {"a": float(pa_a), "b": float(pa_b), "note": "fit on all MultExp; only for --pa-cal multexp"},
        "gate_multexp_not_worse": bool(cand_cv["S_expert_NOT_LB"] <= base_cv["S_expert_NOT_LB"]),
        "note": "MultExp is a stress test. Honest MultExp 0.134 coexisted with public ~0.44.",
    }
    (lay.out / "multexp_report.json").write_text(json.dumps(out, indent=2) + "\n")
    df.to_csv(lay.out / "multexp_rows.csv", index=False)
    print(json.dumps(out, indent=2), flush=True)
    return 0


def _read_incumbent(path: Path, ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str).str.strip()
    if not df.image_id.is_unique:
        raise SystemExit("incumbent image_id collision")
    missing = sorted(set(ids) - set(df.image_id))
    if missing:
        raise SystemExit(f"incumbent misses {len(missing)} official ids, e.g. {missing[0]}")
    return df.set_index("image_id").loc[ids]


def cmd_infer_test(args, lay: Layout) -> int:
    ids = load_official_submission_ids(lay.test_ids)
    rl_df = _read_incumbent(lay.texture_rl, ids)
    model, qnet, report, grid = load_models(lay, args.device)
    qnet = _use_rl(args, qnet, report)
    scale = pd.read_csv(lay.scale_csv).set_index("image_id")
    maps = MapSource(lay.weights, lay.test_maps, args.device)
    pa_cal = None
    if args.pa_cal == "multexp":
        mr = json.loads((lay.out / "multexp_report.json").read_text())
        pa_cal = (mr["candidate_pa_affine_full_fit"]["a"], mr["candidate_pa_affine_full_fit"]["b"])
    diag = []
    for k, iid in enumerate(ids):
        gray = cv2.imread(str(lay.test_dir / iid), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"missing test image {iid}")
        p_apo, p_fasc = maps.get(iid, gray)
        r = predict_image(gray, p_apo, p_fasc, float(scale.loc[iid, "sx"]), float(scale.loc[iid, "sy"]), model, qnet, grid, args.device)
        r["image_id"] = iid
        diag.append(r)
        if k % 50 == 0:
            print(json.dumps({"infer": k + 1, "of": len(ids)}), flush=True)
    d = pd.DataFrame(diag).set_index("image_id").loc[ids]
    pa_new = d.pa_deg.values.copy()
    if pa_cal is not None:
        pa_new = pa_cal[0] * pa_new + pa_cal[1]
    fl_new = d.fl_mm.values * args.fl_factor
    use_pa = np.isfinite(pa_new)
    use_fl = np.isfinite(fl_new) & d.ok.values.astype(bool)
    pa_out = np.clip(np.where(use_pa, pa_new, rl_df.pa_deg.values), *RANGES["pa_deg"])
    fl_out = np.clip(np.where(use_fl, fl_new, rl_df.fl_mm.values), *RANGES["fl_mm"])
    variants = {
        "pa_only": (pa_out, rl_df.fl_mm.values),
        "fl_only": (rl_df.pa_deg.values, fl_out),
        "pa_fl": (pa_out, fl_out),
    }
    written = {}
    for name, (pa, fl) in variants.items():
        rows = [{"image_id": i, "pa_deg": float(a), "fl_mm": float(b), "mt_mm": float(c)} for i, a, b, c in zip(ids, pa, fl, rl_df.mt_mm.values)]
        path = lay.out / f"submission_attn_rl_{name}.csv"
        write_submission_csv(path, rows, validate_against=ids)
        written[name] = str(path)
    d.assign(pa_used=use_pa, fl_used=use_fl).to_csv(lay.out / "test_diag.csv")
    meta = {
        "rl_used": qnet is not None,
        "pa_cal": args.pa_cal,
        "pa_affine": pa_cal,
        "fl_factor": args.fl_factor,
        "fl_affine": None,
        "mt_source": "texture_rl",
        "abstain_source": "texture_rl",
        "pa_replaced": int(use_pa.sum()),
        "fl_replaced": int(use_fl.sum()),
        "written": written,
        "submitted": False,
    }
    (lay.out / "test_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("prepare", "train", "eval-multexp", "infer-test"))
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid", type=int, nargs=2, default=[192, 288])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--crop-min", type=float, default=0.75)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dqn-steps", type=int, default=60000)
    ap.add_argument("--gamma", type=float, default=DQNConfig.gamma, help="discount; the reward shaping uses the same value")
    ap.add_argument("--min-val-scored", type=int, default=30)
    ap.add_argument("--use-rl", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--pa-cal", choices=("none", "multexp"), default="none")
    ap.add_argument("--fl-factor", type=float, default=1.0)
    args = ap.parse_args(argv)
    lay = Layout(args.root, args.out or args.root / "reports/phase3/chase018/attention_rl/lab")
    lay.out.mkdir(parents=True, exist_ok=True)
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    stage = {"prepare": cmd_prepare, "train": cmd_train, "eval-multexp": cmd_eval_multexp, "infer-test": cmd_infer_test}[args.stage]
    return stage(args, lay)


if __name__ == "__main__":
    raise SystemExit(main())
