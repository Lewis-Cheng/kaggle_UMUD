"""Load Frundles public ResNet34 U-Nets (apo / fasc) for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


def load_ckpt(path: Path | str, device: str | torch.device = "cpu"):
    c = torch.load(path, map_location="cpu", weights_only=False)
    m = smp.Unet(c["encoder"], encoder_weights=None, in_channels=1, classes=1)
    m.load_state_dict(c["state"])
    return m.to(device).eval(), tuple(c["size"])


@torch.no_grad()
def predict_prob(model, size: tuple[int, int], gray: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
    """TTA: average forward + horizontal flip. gray is HxW uint8."""
    oh, ow = size
    x = torch.from_numpy(
        cv2.resize(gray, (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    )[None, None].to(device)
    p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
    return p[0, 0].detach().cpu().numpy()


DEFAULT_WEIGHTS_DIR = Path("/workspace/kaggle_UMUD/data/external/frundles_weights")


def load_pair(
    weights_dir: Path | str = DEFAULT_WEIGHTS_DIR,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    weights_dir = Path(weights_dir)
    m_apo, s_apo = load_ckpt(weights_dir / "apo_resnet34_384x576_f0.pt", device)
    m_fasc, s_fasc = load_ckpt(weights_dir / "fasc_resnet34_384x576_f0.pt", device)
    return {
        "apo": (m_apo, s_apo),
        "fasc": (m_fasc, s_fasc),
        "device": device,
        "weights_dir": str(weights_dir),
        "note": "Frundles public umud-r34-unet-weights (CC BY-NC-SA-4.0)",
    }
