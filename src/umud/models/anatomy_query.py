"""Anatomy-query attention stack (PyTorch skeleton; no trained weights).

Proposed (not validated) architecture — do NOT treat as a proven win:

    Ultrasound image + calibration
                │
      High-resolution feature encoder
                │
      Multi-scale image self-attention
                │
      Anatomy queries cross-attend to features
                │
      Self-attention between anatomy queries
                │
     ┌──────────┼──────────────────┐
     │          │                  │
  Superficial/ Fascicle lines /  Confidence /
  deep curves  orientation field uncertainty
     └──────────┼──────────────────┘
                │
      Physical-coordinate geometry layer  (OUTSIDE this module)
                │
             PA, FL, MT

Ablate in order under a frozen protocol (see configs/model_anatomy_query.yaml):
  1) image self-attention
  2) anatomy-query cross-attention (vs dense decoder / param-budget control)
  3) query self-attention

Verification-first: attention is a hypothesis. Pass metric / data / baseline
gates before enabling expensive training. Not a Fast3R / VGGT / DUSt3R transplant.

Forward I/O contract
--------------------
Input:
  image:      FloatTensor [B, C, H, W]  (C usually 1 for ultrasound)
  spacing_mm: optional FloatTensor [B, 2] of (sx, sy) mm/px — stored in the
              output dict for downstream geometry; NOT applied inside forward.

Output (dict of tensors; physical PA/FL/MT are applied outside via geometry_head):
  features_ms:      list of [B, D, H_i, W_i] multi-scale maps (after image SA)
  query_tokens:     [B, Q, D] refined anatomy queries
  superficial_curve:[B, N_pts, 2] placeholder curve samples (pixel coords)
  deep_curve:       [B, N_pts, 2]
  fascicle_curve:   [B, N_pts, 2]
  apo_mask_logits:  [B, 2, H, W]  superficial/deep apo logits (placeholder)
  fascicle_mask_logits: [B, 1, H, W]
  orientation_director: [B, 2, H, W]  (cos 2θ, sin 2θ) field placeholder
  confidence:       [B, 1, H, W]
  spacing_mm:       echoed input or None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_QUERY_NAMES = ("superficial_apo", "deep_apo", "fascicle")


@dataclass
class AnatomyQueryConfig:
    """Layer sizes and staged ablation flags.

    Ablation defaults follow verification order: start with all attention
    blocks off (dense encoder + linear query projection only), then enable
    image_self_attn → cross_attn → query_self_attn under a frozen protocol.
    """

    in_channels: int = 1
    backbone_name: str = "placeholder_cnn"
    embed_dim: int = 128
    num_heads: int = 4
    num_image_sa_blocks: int = 1
    num_cross_blocks: int = 1
    num_query_sa_blocks: int = 1
    num_queries: int = 3  # superficial, deep, fascicle
    curve_points: int = 32
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    # Staged enablement (verification order)
    enable_image_self_attn: bool = False
    enable_cross_attn: bool = False
    enable_query_self_attn: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AnatomyQueryConfig":
        if not raw:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageSelfAttentionBlock(nn.Module):
    """Token-wise MHSA over a flattened spatial feature map."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm1(tokens)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class CrossAttentionBlock(nn.Module):
    """Anatomy queries attend to image tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(queries)
        kv = self.norm_kv(image_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))
        return queries


class QuerySelfAttentionBlock(nn.Module):
    """Self-attention among anatomy queries only."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        h = self.norm1(queries)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))
        return queries


class PlaceholderEncoder(nn.Module):
    """Configurable high-res encoder placeholder (not a pretrained backbone).

    ``backbone_name`` is recorded for future swaps (e.g. convnext / hrnet);
    this skeleton always uses a small stride-aware CNN so CPU smoke tests run
    without downloading weights.
    """

    def __init__(self, in_channels: int, embed_dim: int, backbone_name: str) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.lateral = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feat = self.stem(x)
        # Two-scale placeholder: full stem res + 2× pooled.
        fine = self.lateral(feat)
        coarse = F.avg_pool2d(fine, kernel_size=2, stride=2)
        return [fine, coarse]


class AnatomyQueryModel(nn.Module):
    """Image SA → query–image CA → query SA skeleton (hypothesis only)."""

    def __init__(self, config: AnatomyQueryConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if isinstance(config, AnatomyQueryConfig):
            cfg = config
        else:
            cfg = AnatomyQueryConfig.from_dict(config)
        self.cfg = cfg

        self.encoder = PlaceholderEncoder(
            cfg.in_channels, cfg.embed_dim, cfg.backbone_name
        )
        self.image_sa = nn.ModuleList(
            [
                ImageSelfAttentionBlock(
                    cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout
                )
                for _ in range(cfg.num_image_sa_blocks)
            ]
        )
        self.query_embed = nn.Parameter(
            torch.randn(1, cfg.num_queries, cfg.embed_dim) * 0.02
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout
                )
                for _ in range(cfg.num_cross_blocks)
            ]
        )
        self.query_sa = nn.ModuleList(
            [
                QuerySelfAttentionBlock(
                    cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout
                )
                for _ in range(cfg.num_query_sa_blocks)
            ]
        )
        # Dense decode head placeholders (pixel space, same HxW as input after upsample)
        self.to_apo_logits = nn.Conv2d(cfg.embed_dim, 2, kernel_size=1)
        self.to_fascicle_logits = nn.Conv2d(cfg.embed_dim, 1, kernel_size=1)
        self.to_orientation = nn.Conv2d(cfg.embed_dim, 2, kernel_size=1)
        self.to_confidence = nn.Conv2d(cfg.embed_dim, 1, kernel_size=1)
        # Per-query curve heads: map query token → N_pts x 2 in normalized [0,1]
        self.curve_head = nn.Linear(cfg.embed_dim, cfg.curve_points * 2)

    def _tokens_from_feature(self, feat: torch.Tensor) -> torch.Tensor:
        b, d, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2)  # [B, H*W, D]

    def _feature_from_tokens(
        self, tokens: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        b, _, d = tokens.shape
        return tokens.transpose(1, 2).reshape(b, d, h, w)

    def forward(
        self,
        image: torch.Tensor,
        spacing_mm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run the structure stack. Physical geometry is applied OUTSIDE.

        Parameters
        ----------
        image:
            [B, C, H, W] float tensor.
        spacing_mm:
            Optional [B, 2] (sx, sy). Echoed into the output for geometry_head;
            this module does not convert pixels to millimetres.
        """
        if image.ndim != 4:
            raise ValueError(f"image must be [B,C,H,W], got shape {tuple(image.shape)}")
        b, _, h_in, w_in = image.shape
        features_ms = self.encoder(image)

        # Image self-attention on the finest scale (ablate via flag).
        fine = features_ms[0]
        _, _, fh, fw = fine.shape
        tokens = self._tokens_from_feature(fine)
        if self.cfg.enable_image_self_attn:
            for block in self.image_sa:
                tokens = block(tokens)
            fine = self._feature_from_tokens(tokens, fh, fw)
            features_ms = [fine, *features_ms[1:]]

        queries = self.query_embed.expand(b, -1, -1)
        if self.cfg.enable_cross_attn:
            for block in self.cross_blocks:
                queries = block(queries, tokens)
        if self.cfg.enable_query_self_attn:
            for block in self.query_sa:
                queries = block(queries)

        # Upsample finest features to input resolution for dense heads.
        dense = F.interpolate(fine, size=(h_in, w_in), mode="bilinear", align_corners=False)
        apo_logits = self.to_apo_logits(dense)
        fasc_logits = self.to_fascicle_logits(dense)
        orientation = self.to_orientation(dense)
        # L2-normalize director channels for numerical stability (still a placeholder).
        orientation = F.normalize(orientation, dim=1, eps=1e-6)
        confidence = torch.sigmoid(self.to_confidence(dense))

        # Curve samples from each query token → pixel coordinates in [0, W]/ [0, H]
        curves = torch.sigmoid(self.curve_head(queries))  # [B, Q, N*2] in (0,1)
        curves = curves.view(b, self.cfg.num_queries, self.cfg.curve_points, 2)
        scale = torch.tensor(
            [w_in - 1, h_in - 1], device=image.device, dtype=image.dtype
        ).view(1, 1, 1, 2)
        curves_px = curves * scale

        # Query order: 0 superficial, 1 deep, 2 fascicle (see DEFAULT_QUERY_NAMES)
        superficial = curves_px[:, 0]
        deep = curves_px[:, 1]
        fascicle = curves_px[:, 2] if self.cfg.num_queries > 2 else curves_px[:, -1]

        return {
            "features_ms": features_ms,
            "query_tokens": queries,
            "superficial_curve": superficial,
            "deep_curve": deep,
            "fascicle_curve": fascicle,
            "apo_mask_logits": apo_logits,
            "fascicle_mask_logits": fasc_logits,
            "orientation_director": orientation,
            "confidence": confidence,
            "spacing_mm": spacing_mm,
            "query_names": DEFAULT_QUERY_NAMES[: self.cfg.num_queries],
        }
