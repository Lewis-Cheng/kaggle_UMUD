"""Shared CLI helpers for Phase 2 audit scripts."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def update_protocol_phase2(
    protocol_path: Path,
    *,
    phase2_block: dict[str, Any],
) -> None:
    """Merge/replace the top-level ``phase2_audit:`` mapping in protocol.yaml."""
    import yaml

    text = protocol_path.read_text() if protocol_path.is_file() else ""
    data = yaml.safe_load(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise ValueError(f"protocol.yaml root must be a mapping: {protocol_path}")
    data["phase2_audit"] = phase2_block
    protocol_path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    )
