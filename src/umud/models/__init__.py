"""Model package.

Heavy modules (torch AnatomyQuery) are imported lazily so CPU baselines that
only need GeometryHead do not require a GPU stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AnatomyQueryConfig",
    "AnatomyQueryModel",
    "SegGeometryBaseline",
    "SegGeometryBaselineConfig",
    "GeometryHead",
    "GeometryHeadResult",
]


def __getattr__(name: str) -> Any:
    if name in ("AnatomyQueryConfig", "AnatomyQueryModel"):
        from .anatomy_query import AnatomyQueryConfig, AnatomyQueryModel

        return {"AnatomyQueryConfig": AnatomyQueryConfig, "AnatomyQueryModel": AnatomyQueryModel}[name]
    if name in ("SegGeometryBaseline", "SegGeometryBaselineConfig"):
        from .baseline import SegGeometryBaseline, SegGeometryBaselineConfig

        return {
            "SegGeometryBaseline": SegGeometryBaseline,
            "SegGeometryBaselineConfig": SegGeometryBaselineConfig,
        }[name]
    if name in ("GeometryHead", "GeometryHeadResult"):
        from .geometry_head import GeometryHead, GeometryHeadResult

        return {"GeometryHead": GeometryHead, "GeometryHeadResult": GeometryHeadResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
