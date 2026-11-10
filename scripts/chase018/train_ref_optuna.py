#!/usr/bin/env python3
"""Train a new apo or fasc U-Net. MultExp is held out and is not the objective.

ImageNet initialization, not Frundles weights. Every trial uses crop/flip/gamma
augmentation, AdamW L2 (weight decay), ImageNet input normalization, and the
ResNet BatchNorm. Optuna picks the learning rate, L2, crop scale, rotation
probability, and positive weight by validation Dice on competition images.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# This Python build has no stdlib sqlite3. Optuna's study file needs it.
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import cv2
import numpy as np
import optuna
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/chase018"))

from train_fasc_frundles_recipe import (  # noqa: E402
    augment,
    dice_loss,
    groups_from_hashes,
    load_xy,
    prep_eval,
)
from umud.geometry.host_faithful import dhash  # noqa: E402

# Matches the ImageNet ResNet encoder. BatchNorm inside the encoder stays on.
INPUT_MEAN = 0.485
INPUT_STD = 0.229
TB_DIR = ROOT / "reports/phase3/chase018/ref_optuna/tb"
_writers: dict[str, SummaryWriter] = {}
TASK_DIRS = {
    "fasc": (
        ROOT / "data/raw/umud/fasc_imgs_v1",
        ROOT / "data/raw/umud/fasc_masks_v1",
    ),
    "apo": (
        ROOT / "data/raw/umud/apo_imgs_v1",
        ROOT / "data/raw/umud/apo_masks_v1",
    ),
}


def discover_pairs(img_root: Path, mask_root: Path) -> list[tuple[str, Path, Path]]:
    imgs: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    for folder, bucket in ((img_root, imgs), (mask_root, masks)):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                bucket.setdefault(path.stem, path)
    return [(stem, imgs[stem], masks[stem]) for stem in sorted(imgs) if stem in masks]


def multiexp_hashes() -> list[np.ndarray]:
    import pandas as pd

    expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
    expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
    hashes = []
    for name in expert["image_file"].astype(str):
        gray = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            hashes.append(dhash(gray))
    return hashes


def near_multiexp(image_hash: np.ndarray, mex: list[np.ndarray], max_dist: int = 8) -> bool:
    image_hash = np.asarray(image_hash, dtype=np.uint8)
    for other in mex:
        if int((image_hash != np.asarray(other, dtype=np.uint8)).sum()) <= max_dist:
            return True
    return False


def normalize(image: np.ndarray) -> np.ndarray:
    return ((image - INPUT_MEAN) / INPUT_STD).astype(np.float32)


def tb_writer(tag: str) -> SummaryWriter:
    writer = _writers.get(tag)
    if writer is None:
        path = TB_DIR / "train" / tag
        path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(path))
        _writers[tag] = writer
    return writer


def suggest_params(trial: optuna.Trial) -> dict[str, float]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
        "crop_min": trial.suggest_float("crop_min", 0.55, 0.9),
        "rot_prob": trial.suggest_float("rot_prob", 0.3, 0.8),
        "pos_weight": trial.suggest_float("pos_weight", 2.0, 8.0),
    }


def make_model(device: torch.device) -> torch.nn.Module:
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1).to(device)
    if not any("bn" in name for name in model.state_dict()):
        raise RuntimeError("ResNet encoder has no BatchNorm")
    return model


def evaluate(model: torch.nn.Module, images: torch.Tensor, masks: torch.Tensor, device: torch.device) -> float:
    dices: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), 16):
            pred = torch.sigmoid(model(images[start : start + 16].to(device))).cpu() > 0.5
            truth = masks[start : start + 16] > 0.5
            inter = (pred & truth).sum((1, 2, 3)).float()
            dices += ((2 * inter + 1) / (pred.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)).tolist()
    return float(np.mean(dices))


def train_epochs(
    *,
    params: dict[str, float],
    epochs: int,
    data: dict,
    device: torch.device,
    seed: int,
    trial: optuna.Trial | None = None,
    save_path: Path | None = None,
    tag: str = "",
) -> float:
    model = make_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["l2"])
    steps = max(1, len(data["train_index"]) // data["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, params["lr"], total_steps=epochs * steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    pos_weight = torch.tensor(params["pos_weight"], device=device)
    best = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        started = time.time()
        total = 0.0
        perm = rng.permutation(data["train_index"])
        for step in range(steps):
            chosen = perm[step * data["batch"] : (step + 1) * data["batch"]]
            batch_x, batch_y = [], []
            for index in chosen:
                image, mask = augment(
                    data["images"][index],
                    data["masks"][index],
                    rng,
                    data["size"],
                    params["crop_min"],
                    8.0,
                    params["rot_prob"],
                )
                batch_x.append(normalize(image))
                batch_y.append(mask)
            xb = torch.from_numpy(np.stack(batch_x))[:, None].to(device)
            yb = torch.from_numpy(np.stack(batch_y))[:, None].to(device)
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight) + dice_loss(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.item())
        score = evaluate(model, data["val_x"], data["val_y"], device)
        train_loss = total / steps
        history.append({"epoch": epoch, "train_loss": train_loss, "val_hard_dice": score})
        print(
            json.dumps(
                {
                    "tag": tag,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_hard_dice": score,
                    "l2": params["l2"],
                }
            )
            + f" ({time.time() - started:.0f}s)",
            flush=True,
        )
        if tag:
            writer = tb_writer(tag)
            writer.add_scalar("val_hard_dice", score, epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("l2", params["l2"], epoch)
            writer.flush()
        if score > best:
            best = score
            if save_path is not None:
                torch.save(
                    {
                        "encoder": "resnet34",
                        "encoder_weights": "imagenet",
                        "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "size": list(data["size"]),
                        "input_mean": INPUT_MEAN,
                        "input_std": INPUT_STD,
                        "batch_norm": True,
                        "l2_weight_decay": params["l2"],
                        "params": params,
                        "val_dice": best,
                        "epoch": epoch,
                        "task": data["task"],
                        "multiexp_in_train": False,
                        "objective": "competition_val_hard_dice",
                        "note": "New reference. Augmentation, AdamW L2, ImageNet input norm, BatchNorm, Optuna. MultExp held out.",
                    },
                    save_path,
                )
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if save_path is not None:
        sidecar = save_path.with_suffix(".json")
        sidecar.write_text(json.dumps({"best": best, "params": params, "hist": history}, indent=2) + "\n")
    return best


def prepare(args: argparse.Namespace) -> dict:
    img_root, mask_root = TASK_DIRS[args.task]
    pairs = discover_pairs(img_root, mask_root)
    if args.limit:
        pairs = pairs[: args.limit]
    mex = multiexp_hashes()
    images, masks, hashes = [], [], []
    dropped = 0
    for _stem, image_path, mask_path in pairs:
        image, mask = load_xy(image_path, mask_path)
        image_hash = dhash(image)
        if near_multiexp(image_hash, mex):
            dropped += 1
            continue
        images.append(image)
        masks.append(mask)
        hashes.append(image_hash)
    if len(images) < 8:
        raise SystemExit(f"too few images after MultExp holdout: {len(images)}")
    groups = groups_from_hashes(hashes)
    unique = sorted(set(groups.tolist()))
    splitter = np.random.default_rng(args.seed)
    splitter.shuffle(unique)
    n_val = max(1, int(round(len(unique) * args.val_frac)))
    val_groups = set(unique[:n_val])
    train_index = [i for i, group in enumerate(groups) if group not in val_groups]
    val_index = [i for i, group in enumerate(groups) if group in val_groups]
    size = (args.size[0], args.size[1])
    val_x, val_y = [], []
    for index in val_index:
        image, mask = prep_eval(images[index], masks[index], size)
        val_x.append(normalize(image))
        val_y.append(mask)
    print(
        json.dumps(
            {
                "task": args.task,
                "kept": len(images),
                "dropped_near_multexp": dropped,
                "train": len(train_index),
                "val": len(val_index),
                "multiexp_hashes": len(mex),
            }
        ),
        flush=True,
    )
    return {
        "task": args.task,
        "images": images,
        "masks": masks,
        "train_index": train_index,
        "val_x": torch.from_numpy(np.stack(val_x))[:, None],
        "val_y": torch.from_numpy(np.stack(val_y))[:, None],
        "size": size,
        "batch": args.batch,
    }


def open_study(args: argparse.Namespace) -> optuna.Study:
    args.storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{args.storage_path}"
    last: Exception | None = None
    for _ in range(30):
        try:
            return optuna.create_study(
                study_name=args.study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
            )
        except Exception as exc:
            last = exc
            time.sleep(0.5)
    raise SystemExit(f"could not open Optuna study: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_DIRS), required=True)
    parser.add_argument("--mode", choices=("study", "final"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--trial-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, nargs=2, default=[384, 576])
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--study-name", default="")
    parser.add_argument("--storage", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/phase3/chase018/seg_ckpts")
    args = parser.parse_args()
    args.study_name = args.study_name or f"ref_{args.task}"
    args.storage_path = (args.storage or (ROOT / "reports/phase3/chase018/ref_optuna" / f"{args.study_name}.db")).resolve()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    device = torch.device(args.device)
    data = prepare(args)
    study = open_study(args)
    if args.mode == "study":

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            return train_epochs(
                params=params,
                epochs=args.trial_epochs,
                data=data,
                device=device,
                seed=args.seed + trial.number,
                trial=trial,
                tag=f"{args.task}-t{trial.number}",
            )

        study.optimize(objective, n_trials=args.trials)
        print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}), flush=True)
        return 0
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise SystemExit("no complete Optuna trial; refusing a final train")
    params = dict(study.best_params)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.out_dir / f"ref_{args.task}_optuna_best.pt"
    best = train_epochs(
        params=params,
        epochs=args.final_epochs,
        data=data,
        device=device,
        seed=args.seed,
        save_path=save_path,
        tag=f"{args.task}-final",
    )
    print(json.dumps({"saved": str(save_path), "val_hard_dice": best, "params": params}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
