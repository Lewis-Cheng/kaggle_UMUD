#!/usr/bin/env python3
"""Mirror ref-optuna log lines into TensorBoard while a run is already in progress."""
from __future__ import annotations

import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "reports/phase3/chase018/ref_optuna"
TB_DIR = LOG_DIR / "tb" / "live"


def parse_row(line: str) -> dict | None:
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        row = json.loads(line[start : end + 1])
    except json.JSONDecodeError:
        return None
    if "tag" not in row or "epoch" not in row:
        return None
    return row


def main() -> int:
    writers: dict[str, SummaryWriter] = {}
    offsets: dict[Path, int] = {}
    seen: set[tuple[str, int]] = set()
    TB_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for path in sorted(LOG_DIR.glob("*.log")):
            size = path.stat().st_size
            offset = offsets.get(path, 0)
            if size < offset:
                offset = 0
            if size == offset:
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                text = handle.read()
                offsets[path] = handle.tell()
            for line in text.splitlines():
                row = parse_row(line)
                if row is None:
                    continue
                tag = str(row["tag"])
                epoch = int(row["epoch"])
                key = (tag, epoch)
                if key in seen:
                    continue
                seen.add(key)
                writer = writers.get(tag)
                if writer is None:
                    writer = SummaryWriter(log_dir=str(TB_DIR / tag))
                    writers[tag] = writer
                if "val_hard_dice" in row:
                    writer.add_scalar("val_hard_dice", float(row["val_hard_dice"]), epoch)
                if "train_loss" in row:
                    writer.add_scalar("train_loss", float(row["train_loss"]), epoch)
                if "l2" in row:
                    writer.add_scalar("l2", float(row["l2"]), epoch)
                writer.flush()
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
