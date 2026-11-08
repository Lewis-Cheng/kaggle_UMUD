"""Tiny U-Net for binary ultrasound mask prediction (pure torch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """4-level U-Net, ~2–4M params depending on base channels."""

    def __init__(self, in_ch: int = 1, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBNReLU(in_ch, base)
        self.enc2 = ConvBNReLU(base, base * 2)
        self.enc3 = ConvBNReLU(base * 2, base * 4)
        self.enc4 = ConvBNReLU(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBNReLU(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBNReLU(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBNReLU(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBNReLU(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBNReLU(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def dice_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """BCE-with-logits + soft Dice on probabilities."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3) if probs.ndim == 4 else (0, 1)
    inter = (probs * targets).sum(dim=dims)
    den = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = 1.0 - (2.0 * inter + eps) / (den + eps)
    return bce + dice.mean()
