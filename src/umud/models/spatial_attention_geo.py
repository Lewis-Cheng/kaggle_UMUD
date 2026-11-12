"""Feed-forward spatial-attention geometry network.

Input channels: image, frozen ``p_apo``, frozen ``p_fasc`` (all in [0, 1]).
Outputs on the input grid:

* ``attn``   logit that a pixel lies on an annotated fascicle,
* ``orient`` unit ``(cos 2t, sin 2t)`` of the fascicle direction in pixel axes,
* ``sup`` / ``deep`` logits of the two aponeurosis bands.

Every encoder/decoder block ends with CBAM-style channel and spatial
attention. The measurement itself stays in ``umud.geometry.attended_readout``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_MEAN = 0.3
INPUT_STD = 0.25


@dataclass
class AttnGeoConfig:
    in_ch: int = 3
    width: int = 16
    depth: int = 3
    dropout: float = 0.1


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        hidden = max(4, ch // r)
        self.mlp = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean((2, 3)))
        mx = self.mlp(x.amax((2, 3)))
        return x * torch.sigmoid(avg + mx)[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * a, a


class AttnBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )
        self.ca = ChannelAttention(cout)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.ca(self.body(x))
        y, a = self.sa(y)
        return F.relu(y + self.skip(x)), a


class SpatialAttentionGeoNet(nn.Module):
    def __init__(self, cfg: AttnGeoConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or AttnGeoConfig()
        w = cfg.width
        chans = [w * (2**i) for i in range(cfg.depth + 1)]
        self.stem = AttnBlock(cfg.in_ch, chans[0])
        self.down = nn.ModuleList(
            [AttnBlock(chans[i], chans[i + 1], stride=2, dropout=cfg.dropout) for i in range(cfg.depth)]
        )
        self.up = nn.ModuleList(
            [AttnBlock(chans[i + 1] + chans[i], chans[i]) for i in reversed(range(cfg.depth))]
        )
        self.head_attn = nn.Conv2d(chans[0], 1, 1)
        self.head_orient = nn.Conv2d(chans[0], 2, 1)
        self.head_apo = nn.Conv2d(chans[0], 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - INPUT_MEAN) / INPUT_STD
        feats = []
        h, a = self.stem(x)
        feats.append(h)
        for blk in self.down:
            h, a = blk(h)
            feats.append(h)
        h = feats.pop()
        for blk in self.up:
            skip = feats.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h, a = blk(torch.cat([h, skip], 1))
        o = self.head_orient(h)
        o = o / (o.norm(dim=1, keepdim=True) + 1e-6)
        apo = self.head_apo(h)
        return {
            "attn": self.head_attn(h)[:, 0],
            "orient": o,
            "sup": apo[:, 0],
            "deep": apo[:, 1],
            "spatial_attention": a[:, 0],
        }


def flip_output(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map an output computed on a flipped image back to the original frame."""
    o = out["orient"].flip(-1)
    o = torch.stack([o[:, 0], -o[:, 1]], 1)
    return {
        "attn": out["attn"].flip(-1),
        "orient": o,
        "sup": out["sup"].flip(-1),
        "deep": out["deep"].flip(-1),
        "spatial_attention": out["spatial_attention"].flip(-1),
    }


@torch.no_grad()
def predict_symmetric(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Flip-TTA in probability space; equivariant to a horizontal flip of ``x``."""
    model.eval()
    a = model(x)
    b = flip_output(model(x.flip(-1)))
    o = a["orient"] + b["orient"]
    o = o / (o.norm(dim=1, keepdim=True) + 1e-6)
    return {
        "attn": 0.5 * (torch.sigmoid(a["attn"]) + torch.sigmoid(b["attn"])),
        "orient": o,
        "sup": 0.5 * (torch.sigmoid(a["sup"]) + torch.sigmoid(b["sup"])),
        "deep": 0.5 * (torch.sigmoid(a["deep"]) + torch.sigmoid(b["deep"])),
        "spatial_attention": 0.5 * (a["spatial_attention"] + b["spatial_attention"]),
    }
