"""Multi-scale TinyUNet with deep-supervision heads (pure torch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from umud.seg.unet import ConvBNReLU


class MSTinyUNet(nn.Module):
    """4-level U-Net with aux heads at dec3/dec2/(optional dec1) for thin structure.

    Deep-sup docstring completeness:
    - n_aux=2 (default): aux@dec3 + aux@dec2 — v5-compatible state_dict.
    - n_aux=3: also aux@dec1 (v12 mutant); new ``aux1`` weights only.
    """

    def __init__(self, in_ch: int = 1, base: int = 48, *, n_aux: int = 2) -> None:
        super().__init__()
        if n_aux not in (2, 3):
            raise ValueError(f"n_aux must be 2 or 3, got {n_aux}")
        self.base = int(base)
        self.n_aux = int(n_aux)
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
        self.aux3 = nn.Conv2d(base * 4, 1, 1)
        self.aux2 = nn.Conv2d(base * 2, 1, 1)
        if self.n_aux >= 3:
            self.aux1 = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        if not return_aux:
            return logits
        # upsample aux to full res
        a3 = F.interpolate(self.aux3(d3), size=logits.shape[-2:], mode="bilinear", align_corners=False)
        a2 = F.interpolate(self.aux2(d2), size=logits.shape[-2:], mode="bilinear", align_corners=False)
        if self.n_aux >= 3:
            a1 = self.aux1(d1)  # already full-res
            return logits, a3, a2, a1
        return logits, a3, a2
