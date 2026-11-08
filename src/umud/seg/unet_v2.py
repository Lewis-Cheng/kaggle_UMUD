"""Deeper U-Net with ASPP bottleneck for thin fascicle lines."""

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


class ASPP(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(cout * 3, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1))


class FascUNet(nn.Module):
    """4-level U-Net + ASPP (channel-consistent skips)."""

    def __init__(self, in_ch: int = 1, base: int = 48, use_aspp: bool = True) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ConvBNReLU(in_ch, c1)
        self.enc2 = ConvBNReLU(c1, c2)
        self.enc3 = ConvBNReLU(c2, c3)
        self.enc4 = ConvBNReLU(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ASPP(c4, c5) if use_aspp else ConvBNReLU(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = ConvBNReLU(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBNReLU(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBNReLU(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBNReLU(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self._match(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._match(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        return self.head(d1)

    @staticmethod
    def _match(up: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != ref.shape[-2:]:
            up = F.interpolate(up, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return up
