"""v5-prior Anatomy-Query regressor (Track A public submit).

Frozen MSTinyUNet v5 fasc (+ apo) masks are input priors. Attention is enabled
(image SA → anatomy-query CA → query SA). The metric head predicts a residual
on top of GeometryHead(v5 priors)+AmbrosM so we keep the v5 floor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from umud.models.anatomy_query import AnatomyQueryConfig, AnatomyQueryModel


class V5AttentionRegressor(nn.Module):
    """Image + apo/fasc prior → attention queries → residual PA/FL/MT."""

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        num_heads: int = 4,
        size_hint: int = 256,
    ) -> None:
        super().__init__()
        self.size_hint = int(size_hint)
        cfg = AnatomyQueryConfig(
            in_channels=3,
            backbone_name="v5_prior_cnn",
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_image_sa_blocks=2,
            num_cross_blocks=2,
            num_query_sa_blocks=1,
            num_queries=3,
            curve_points=32,
            mlp_ratio=2.0,
            dropout=0.05,
            enable_image_self_attn=True,
            enable_cross_attn=True,
            enable_query_self_attn=True,
        )
        self.backbone = AnatomyQueryModel(cfg)
        # query tokens (3*D) + base(pa,fl,mt) + log_mm_per_px + confidence mean
        in_dim = embed_dim * 3 + 3 + 1 + 1
        self.residual_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
        )
        # Init residual near zero so early predictions ≈ v5 base.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        image_priors: torch.Tensor,
        *,
        base_pa_fl_mt: torch.Tensor,
        log_mm_per_px: torch.Tensor,
        spacing_mm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """image_priors: [B,3,H,W] = (img, apo_prior, fasc_prior) in [0,1]."""
        out = self.backbone(image_priors, spacing_mm=spacing_mm)
        q = out["query_tokens"].flatten(1)  # [B, 3D]
        conf = out["confidence"].mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        feat = torch.cat(
            [q, base_pa_fl_mt, log_mm_per_px.view(-1, 1), conf], dim=1
        )
        residual = self.residual_head(feat)
        pred = base_pa_fl_mt + residual
        out["residual"] = residual
        out["pa_fl_mt"] = pred
        out["base_pa_fl_mt"] = base_pa_fl_mt
        return out
