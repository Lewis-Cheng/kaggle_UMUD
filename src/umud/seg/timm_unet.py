"""Lightweight U-Net decoder on a timm encoder (ImageNet pretrained)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None

from umud.seg.unet import ConvBNReLU


class TimmUNet(nn.Module):
    """1-channel ultrasound → repeat to 3ch → timm features → U-Net decoder."""

    def __init__(self, encoder: str = "resnet18", base: int = 32, pretrained: bool = True) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm required for TimmUNet")
        self.encoder_name = encoder
        self.enc = timm.create_model(
            encoder,
            pretrained=pretrained,
            features_only=True,
            in_chans=3,
            out_indices=(0, 1, 2, 3, 4),
        )
        chs = self.enc.feature_info.channels()  # 5 scales
        # project encoder feats to decoder widths
        self.p0 = ConvBNReLU(chs[0], base)
        self.p1 = ConvBNReLU(chs[1], base * 2)
        self.p2 = ConvBNReLU(chs[2], base * 4)
        self.p3 = ConvBNReLU(chs[3], base * 8)
        self.p4 = ConvBNReLU(chs[4], base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = ConvBNReLU(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBNReLU(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBNReLU(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBNReLU(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        f0, f1, f2, f3, f4 = self.enc(x)
        e0, e1, e2, e3, e4 = self.p0(f0), self.p1(f1), self.p2(f2), self.p3(f3), self.p4(f4)
        # align spatial if encoder strides differ slightly
        def cat_up(up, skip):
            u = up
            if u.shape[-2:] != skip.shape[-2:]:
                u = F.interpolate(u, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([u, skip], dim=1)

        x = self.d4(cat_up(self.up4(e4), e3))
        x = self.d3(cat_up(self.up3(x), e2))
        x = self.d2(cat_up(self.up2(x), e1))
        x = self.d1(cat_up(self.up1(x), e0))
        logits = self.head(x)
        # final size may be H/2 for some encoders — upsample to input
        return logits
