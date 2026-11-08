"""CPU smoke test for AnatomyQueryModel forward I/O contract.

Requires torch (CPU wheel is fine). Skips cleanly only if torch is absent;
the project venv is expected to install torch for this gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from umud.models.anatomy_query import AnatomyQueryConfig, AnatomyQueryModel


REQUIRED_KEYS = {
    "features_ms",
    "query_tokens",
    "superficial_curve",
    "deep_curve",
    "fascicle_curve",
    "apo_mask_logits",
    "fascicle_mask_logits",
    "orientation_director",
    "confidence",
    "spacing_mm",
    "query_names",
}


def test_anatomy_query_forward_tiny_cpu():
    cfg = AnatomyQueryConfig(
        in_channels=1,
        embed_dim=32,
        num_heads=4,
        num_image_sa_blocks=1,
        num_cross_blocks=1,
        num_query_sa_blocks=1,
        num_queries=3,
        curve_points=8,
        # Enable all blocks once for a smoke path; production defaults stay off.
        enable_image_self_attn=True,
        enable_cross_attn=True,
        enable_query_self_attn=True,
    )
    model = AnatomyQueryModel(cfg)
    model.eval()
    image = torch.randn(2, 1, 64, 64)
    spacing = torch.tensor([[0.1, 0.15], [0.1, 0.15]], dtype=torch.float32)
    with torch.no_grad():
        out = model(image, spacing_mm=spacing)

    assert REQUIRED_KEYS <= set(out)
    assert out["query_tokens"].shape == (2, 3, 32)
    assert out["superficial_curve"].shape == (2, 8, 2)
    assert out["deep_curve"].shape == (2, 8, 2)
    assert out["fascicle_curve"].shape == (2, 8, 2)
    assert out["apo_mask_logits"].shape == (2, 2, 64, 64)
    assert out["fascicle_mask_logits"].shape == (2, 1, 64, 64)
    assert out["orientation_director"].shape == (2, 2, 64, 64)
    assert out["confidence"].shape == (2, 1, 64, 64)
    assert out["spacing_mm"] is spacing
    # Physical geometry is outside: model must not emit pa_deg/fl_mm/mt_mm.
    assert "pa_deg" not in out
    assert "fl_mm" not in out
    assert "mt_mm" not in out


def test_anatomy_query_ablation_defaults_all_off():
    cfg = AnatomyQueryConfig()
    assert cfg.enable_image_self_attn is False
    assert cfg.enable_cross_attn is False
    assert cfg.enable_query_self_attn is False
    model = AnatomyQueryModel(cfg)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 32, 32))
    assert out["query_tokens"].shape[1] == 3
