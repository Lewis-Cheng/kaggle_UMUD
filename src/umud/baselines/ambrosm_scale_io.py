"""Load AmbrosM per-image mm/px rows for GeometryHead Tier B."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

USABLE = frozenset({"measured", "family_prior"})
_PROV_RANK = {"measured": 0, "family_prior": 1, "unknown": 2}


@dataclass(frozen=True)
class AmbrosmScaleRow:
    stem: str
    pool: str
    family: str
    provenance: str
    mm_per_px: float | None
    px_per_mm: float | None
    assumed_mm_per_tick: float | None
    q_provenance: str
    scale_note: str

    @property
    def usable(self) -> bool:
        return (
            self.provenance in USABLE
            and self.mm_per_px is not None
            and float(self.mm_per_px) > 0
        )


def _f(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_ambrosm_scale_rows(path: Path | str) -> list[AmbrosmScaleRow]:
    """Parse ``ambrosm_scale_rows.csv`` (or equivalent)."""
    path = Path(path)
    out: list[AmbrosmScaleRow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            out.append(
                AmbrosmScaleRow(
                    stem=str(raw.get("stem") or Path(str(raw.get("name", ""))).stem).strip(),
                    pool=str(raw.get("pool", "")).strip(),
                    family=str(raw.get("family", "")).strip(),
                    provenance=str(raw.get("provenance", "unknown")).strip(),
                    mm_per_px=_f(raw.get("mm_per_px")),
                    px_per_mm=_f(raw.get("px_per_mm")),
                    assumed_mm_per_tick=_f(raw.get("assumed_mm_per_tick")),
                    q_provenance=str(raw.get("q_provenance", "unknown")).strip(),
                    scale_note=str(raw.get("scale_note", "")).strip(),
                )
            )
    return out


def scale_map_by_stem(
    rows: Iterable[AmbrosmScaleRow],
    *,
    prefer_pools: tuple[str, ...] = ("test", "apo", "fasc"),
) -> dict[str, AmbrosmScaleRow]:
    """One row per stem: best provenance, then preferred pool order."""
    pool_rank = {p: i for i, p in enumerate(prefer_pools)}
    best: dict[str, AmbrosmScaleRow] = {}
    for row in rows:
        if not row.stem:
            continue
        cur = best.get(row.stem)
        if cur is None:
            best[row.stem] = row
            continue
        pr = _PROV_RANK.get(row.provenance, 9)
        pc = _PROV_RANK.get(cur.provenance, 9)
        if pr < pc:
            best[row.stem] = row
            continue
        if pr > pc:
            continue
        # same provenance: prefer listed pools, else keep existing
        if pool_rank.get(row.pool, 99) < pool_rank.get(cur.pool, 99):
            best[row.stem] = row
    return best
