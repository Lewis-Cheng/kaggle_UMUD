#!/usr/bin/env python3
"""Train Track-A v5+Attention residual regressor (proxy GH AmbrosM targets).

Priors: predicted apo + fasc v5 masks (train/val dirs). Targets: GeometryHead on
GT masks + AmbrosM when usable. Loss = SmoothL1 on tolerance-normalized errors.
NOT expert GT / NOT LB — used only to produce a real Kaggle public submission.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.ambrosm_scale_io import load_ambrosm_scale_rows, scale_map_by_stem  # noqa: E402
from umud.baselines.io import load_split_assignments  # noqa: E402
from umud.baselines.mask_geometry import measure_masks  # noqa: E402
from umud.baselines.mask_proxy import FALLBACK_PRIORS, load_mask  # noqa: E402
from umud.data.layout import discover_layout, list_images  # noqa: E402
from umud.models.anatomy_query_v5 import V5AttentionRegressor  # noqa: E402

TAU = np.array([6.0, 12.0, 3.0], dtype=np.float32)  # official S tolerances


def _mask_path(folder: Path | None, stem: str) -> Path | None:
    if folder is None or not folder.is_dir():
        return None
    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _load_gray(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("L")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _load_mask01(path: Path | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=np.float32)
    im = Image.open(path).convert("L")
    im = im.resize((size, size), Image.NEAREST)
    arr = np.asarray(im, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return (arr > 0.5).astype(np.float32)


def _measure_stem(
    stem: str,
    *,
    fasc_dir: Path | None,
    apo_dir: Path | None,
    scale,
    priors: dict[str, float],
) -> dict[str, Any]:
    fasc = apo = None
    fp = _mask_path(fasc_dir, stem)
    ap = _mask_path(apo_dir, stem)
    if fp is not None:
        fasc = load_mask(fp)
    if ap is not None:
        apo = load_mask(ap)
    usable = scale is not None and getattr(scale, "usable", False)
    sx = sy = float(scale.mm_per_px) if usable else 1.0
    m = measure_masks(fasc_mask=fasc, apo_mask=apo, sx=sx, sy=sy, priors=priors)
    m["stem"] = stem
    m["usable_for_mm_targets"] = bool(usable and m.get("ok"))
    m["mm_per_px"] = float(scale.mm_per_px) if usable else None
    return m


class AttnDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        img_index: dict[str, Path],
        pred_apo: Path,
        pred_fasc: Path,
        size: int,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.img_index = img_index
        self.pred_apo = pred_apo
        self.pred_fasc = pred_fasc
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        stem = r["stem"]
        img = _load_gray(self.img_index[stem], self.size)
        apo = _load_mask01(_mask_path(self.pred_apo, stem), self.size)
        fasc = _load_mask01(_mask_path(self.pred_fasc, stem), self.size)
        if self.augment and random.random() < 0.5:
            img = np.fliplr(img).copy()
            apo = np.fliplr(apo).copy()
            fasc = np.fliplr(fasc).copy()
        x = np.stack([img, apo, fasc], axis=0)
        base = np.array([r["base_pa"], r["base_fl"], r["base_mt"]], dtype=np.float32)
        tgt = np.array([r["tgt_pa"], r["tgt_fl"], r["tgt_mt"]], dtype=np.float32)
        mm = float(r["mm_per_px"])
        log_mm = math.log(max(mm, 1e-6))
        return {
            "x": torch.from_numpy(x),
            "base": torch.from_numpy(base),
            "tgt": torch.from_numpy(tgt),
            "log_mm": torch.tensor([log_mm], dtype=torch.float32),
            "spacing": torch.tensor([mm, mm], dtype=torch.float32),
        }


def build_rows(
    stems: list[str],
    *,
    layout,
    scales,
    pred_apo: Path,
    pred_fasc: Path,
    priors: dict[str, float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stem in stems:
        tgt = _measure_stem(
            stem,
            fasc_dir=layout.fasc_masks,
            apo_dir=layout.apo_masks,
            scale=scales.get(stem),
            priors=priors,
        )
        if not tgt.get("usable_for_mm_targets"):
            continue
        base = _measure_stem(
            stem,
            fasc_dir=pred_fasc,
            apo_dir=pred_apo,
            scale=scales.get(stem),
            priors=priors,
        )
        # Need a base prediction even if GH failed — fall back to priors.
        base_pa = float(base["pa_deg"]) if base.get("ok") else float(priors["pa_deg"])
        base_fl = float(base["fl_mm"]) if base.get("ok") else float(priors["fl_mm"])
        base_mt = float(base["mt_mm"]) if base.get("ok") else float(priors["mt_mm"])
        out.append(
            {
                "stem": stem,
                "tgt_pa": float(tgt["pa_deg"]),
                "tgt_fl": float(tgt["fl_mm"]),
                "tgt_mt": float(tgt["mt_mm"]),
                "base_pa": base_pa,
                "base_fl": base_fl,
                "base_mt": base_mt,
                "mm_per_px": float(tgt["mm_per_px"]),
            }
        )
    return out


def mae_components(pred: np.ndarray, tgt: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - tgt)
    return {
        "pa_mae": float(err[:, 0].mean()) if len(err) else float("nan"),
        "fl_mae": float(err[:, 1].mean()) if len(err) else float("nan"),
        "mt_mae": float(err[:, 2].mean()) if len(err) else float("nan"),
        "n": int(len(err)),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    preds, tgts, bases = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        base = batch["base"].to(device)
        log_mm = batch["log_mm"].to(device)
        spacing = batch["spacing"].to(device)
        out = model(x, base_pa_fl_mt=base, log_mm_per_px=log_mm, spacing_mm=spacing)
        preds.append(out["pa_fl_mt"].cpu().numpy())
        tgts.append(batch["tgt"].numpy())
        bases.append(batch["base"].numpy())
    if not preds:
        return {"pa_mae": float("nan"), "fl_mae": float("nan"), "mt_mae": float("nan"), "n": 0}
    p = np.concatenate(preds)
    t = np.concatenate(tgts)
    b = np.concatenate(bases)
    m = mae_components(p, t)
    m_base = mae_components(b, t)
    m["base_pa_mae"] = m_base["pa_mae"]
    m["base_fl_mae"] = m_base["fl_mae"]
    m["base_mt_mae"] = m_base["mt_mae"]
    # diagnostic S1 vs GT-mask GH
    s1 = (
        np.maximum(np.abs(p - t) / TAU, 0.0).sum(axis=1).mean() / 3.0
        if len(p)
        else float("nan")
    )
    m["diagnostic_S1"] = float(s1)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-assignments", type=Path, required=True)
    ap.add_argument("--ambrosm-scale-csv", type=Path, required=True)
    ap.add_argument("--pred-train-dir", type=Path, required=True)
    ap.add_argument("--pred-val-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layout = discover_layout(args.data_root)
    scales = scale_map_by_stem(load_ambrosm_scale_rows(args.ambrosm_scale_csv))
    split_rows = load_split_assignments(args.split_assignments)

    train_stems = sorted({str(r["image_id"]) for r in split_rows if r.get("split") == "train"})
    val_stems = sorted({str(r["image_id"]) for r in split_rows if r.get("split") == "val"})

    # Image index: apo ∪ fasc
    img_index: dict[str, Path] = {}
    for p in list_images(layout.apo_imgs) + list_images(layout.fasc_imgs):
        img_index[p.stem] = p

    # Priors from GT mm medians (same as tier_b)
    probe = []
    for s in val_stems[: min(400, len(val_stems))]:
        m = _measure_stem(
            s,
            fasc_dir=layout.fasc_masks,
            apo_dir=layout.apo_masks,
            scale=scales.get(s),
            priors=FALLBACK_PRIORS,
        )
        if m.get("usable_for_mm_targets"):
            probe.append(m)
    priors = {
        "pa_deg": float(np.median([m["pa_deg"] for m in probe])) if probe else float(FALLBACK_PRIORS["pa_deg"]),
        "fl_mm": float(np.median([m["fl_mm"] for m in probe])) if probe else float(FALLBACK_PRIORS["fl_mm"]),
        "mt_mm": float(np.median([m["mt_mm"] for m in probe])) if probe else float(FALLBACK_PRIORS["mt_mm"]),
    }

    pred_train_apo = args.pred_train_dir / "pred_apo_masks"
    pred_train_fasc = args.pred_train_dir / "pred_fasc_masks"
    pred_val_apo = args.pred_val_dir / "pred_apo_masks"
    pred_val_fasc = args.pred_val_dir / "pred_fasc_masks"

    train_rows = build_rows(
        train_stems,
        layout=layout,
        scales=scales,
        pred_apo=pred_train_apo,
        pred_fasc=pred_train_fasc,
        priors=priors,
    )
    val_rows = build_rows(
        val_stems,
        layout=layout,
        scales=scales,
        pred_apo=pred_val_apo,
        pred_fasc=pred_val_fasc,
        priors=priors,
    )
    # Keep only stems with an image
    train_rows = [r for r in train_rows if r["stem"] in img_index]
    val_rows = [r for r in val_rows if r["stem"] in img_index]
    print(f"train_rows={len(train_rows)} val_rows={len(val_rows)} priors={priors}", flush=True)
    if len(train_rows) < 50:
        print("ERROR: too few train rows — is pred-train-dir ready?", file=sys.stderr)
        return 1

    train_ds = AttnDataset(
        train_rows,
        img_index=img_index,
        pred_apo=pred_train_apo,
        pred_fasc=pred_train_fasc,
        size=args.size,
        augment=True,
    )
    val_ds = AttnDataset(
        val_rows,
        img_index=img_index,
        pred_apo=pred_val_apo,
        pred_fasc=pred_val_fasc,
        size=args.size,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = V5AttentionRegressor(size_hint=args.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    tau_t = torch.tensor(TAU, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            base = batch["base"].to(device)
            tgt = batch["tgt"].to(device)
            log_mm = batch["log_mm"].to(device)
            spacing = batch["spacing"].to(device)
            out = model(x, base_pa_fl_mt=base, log_mm_per_px=log_mm, spacing_mm=spacing)
            pred = out["pa_fl_mt"]
            # tolerance-normalized SmoothL1 + small residual L2
            err = (pred - tgt) / tau_t
            loss = F.smooth_l1_loss(err, torch.zeros_like(err))
            loss = loss + 0.05 * F.mse_loss(out["residual"] / tau_t, torch.zeros_like(err))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        metrics = evaluate(model, val_loader, device)
        # Selection: mean normalized MAE (proxy S1)
        score = (
            metrics["pa_mae"] / 6.0 + metrics["fl_mae"] / 12.0 + metrics["mt_mae"] / 3.0
        ) / 3.0
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val": metrics,
            "score": float(score),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val": metrics,
            "score": float(score),
            "priors": priors,
            "size": args.size,
            "variant": "trackA_v5_attention",
        }
        torch.save(ckpt, args.out_dir / "attn_v5_last.pt")
        if score < best:
            best = score
            torch.save(ckpt, args.out_dir / "attn_v5_best.pt")
            print(f"NEW_BEST score={score:.4f}", flush=True)

    summary = {
        "best_score": best,
        "history": history,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "priors": priors,
        "pred_train_dir": str(args.pred_train_dir),
        "pred_val_dir": str(args.pred_val_dir),
        "device": str(device),
    }
    (args.out_dir / "attn_v5_train.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps({"best_score": best, "n_train": len(train_rows), "n_val": len(val_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
