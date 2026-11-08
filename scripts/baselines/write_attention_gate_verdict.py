#!/usr/bin/env python3
"""Write honest attention_gate_verdict.md from fasc rank + seg→GH metrics."""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HK = ZoneInfo("Asia/Hong_Kong")

def dig(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-json", type=Path, required=True)
    ap.add_argument("--metrics-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fasc-winner-ckpt", type=str, default="")
    args = ap.parse_args()

    rank = json.loads(args.rank_json.read_text())
    winner = rank.get("winner") or {}
    dice = float(winner.get("best_native_macro_dice") or 0)
    variant = winner.get("variant") or "?"
    ckpt = args.fasc_winner_ckpt or winner.get("ckpt") or "?"
    rows = rank.get("ranked") or []

    m = json.loads(args.metrics_json.read_text()) if args.metrics_json.exists() else {}
    # try common shapes
    vm = m.get("val_metrics") or m.get("val_mae") or m
    pa = dig(vm, "pa_mae", "pa_deg", "PA", "pa")
    fl = dig(vm, "fl_mae", "fl_mm", "FL", "fl")
    mt = dig(vm, "mt_mae", "mt_mm", "MT", "mt")
    s1 = dig(vm, "diagnostic_S1_vs_gt_mask_gh", "diagnostic_s1", "s1", "S1")
    if pa is None and isinstance(m.get("metrics"), dict):
        mm = m["metrics"]
        pa = dig(mm, "pa_mae", "pa_deg", "PA"); fl = dig(mm, "fl_mae", "fl_mm", "FL"); mt = dig(mm, "mt_mae", "mt_mm", "MT")
        s1 = dig(mm, "diagnostic_S1_vs_gt_mask_gh", "diagnostic_s1", "s1")

    def fnum(x, nd=2):
        try: return f"{float(x):.{nd}f}"
        except Exception: return "n/a"

    et_pa, et_fl, et_mt = 2.23, 9.86, 1.00
    v4b_pa, v4b_fl, v4b_mt = 3.79, 8.48, 2.96

    fasc_usable = dice >= 0.40  # honest bar for thin fasc geometry path
    et_still_better = True
    try:
        et_still_better = (float(pa) > et_pa + 0.2) or (float(mt) > et_mt + 0.3)
    except Exception:
        et_still_better = True
    expert_gt = False  # known null
    attention_blocked = (not fasc_usable) or et_still_better or (not expert_gt)
    # even if fasc_usable and beats ET on proxies, without expert GT keep blocked unless waiver — user said no fake waiver
    # Only NO if fasc clearly usable AND gates otherwise allow. Expert GT missing → still YES.
    attention_blocked = True  # hard: no expert GT → S=null; also require fasc usable for geometry
    reason_unblock_parts = []
    if fasc_usable and (not et_still_better):
        # still YES because expert GT null — document near-miss
        attention_blocked = True
        hard_stop_note = "Near-miss: fasc Dice≥0.40 and seg→GH PA/MT competitive with ET, but official S still null (no expert GT) → attention_blocked stays YES (no fake waiver)."
    else:
        hard_stop_note = ""

    # Actually re-read user criteria: "Only NO if fasc clearly usable for MT/geometry path AND gates otherwise allow."
    # Gates include expert GT / S pin. So without expert GT, always YES.
    attention_blocked = True
    if fasc_usable and not et_still_better:
        status_extra = "Fasc/geometry proxy improved enough that *if* expert GT appeared, unblock could be reconsidered; until then YES."
    else:
        status_extra = "Fasc still too weak and/or ExtraTrees still dominates PA/MT proxy; keep YES."

    now = datetime.now(HK).strftime("%Y-%m-%d %H:%M HKT")
    table = "\n".join(
        f"| {r.get('variant')} | {float(r.get('best_native_macro_dice') or 0):.4f} | {r.get('arch')} | {r.get('size')} |"
        for r in rows[:10]
    )
    md = f"""# Attention go / no-go verdict

**Date (Asia/Hong_Kong):** {now}  
**Host:** SSH `chocheng9@144.214.209.211:~/kaggle_UMUD`  
**Decision: `attention_blocked = YES`** — do **not** start expensive attention training.

Win-plan gates before image SA + anatomy-query CA + query SA:

| Gate | Status | Evidence |
|---|---|---|
| Official scorer pinned (τ 6°/12mm/3mm, notebook SHA) | **PASS** | `configs/protocol.yaml` |
| Duplicate-aware splits | **PASS** | `reports/phase2/split_assignments.jsonl` |
| Calibration / scale honest | **PARTIAL** | AmbrosM v2: test 309/309; residual unknown — no invented parsers. |
| Mask alignment / seg → GeometryHead | **PARTIAL** | Apo Dice ~0.969. Fasc **native macro Dice {dice:.4f}** (`{variant}`). |
| Tier B meaningful vs independent labels | **FAIL** | **No expert PA/FL/MT GT** → official **S = null**. |
| Held-out proxy quality (pred GH vs GT-mask GH) | **proxy only** | Val MAE PA **{fnum(pa)}°** · FL **{fnum(fl)} mm** · MT **{fnum(mt)} mm** · diagnostic S1 **{fnum(s1,3)}** (NOT expert / NOT LB). |

## Fasc native macro Dice ranking (gate metric)

| Variant | Native macro Dice | Arch | Size |
|---|---:|---|---:|
{table}

Winner ckpt: `{ckpt}`

## Seg → GeometryHead + AmbrosM

| | ExtraTrees Tier B v2 | Seg→GH v4b | **Seg→GH current** |
|---|---:|---:|---:|
| PA ° | **{et_pa}** | {v4b_pa} | **{fnum(pa)}** |
| FL mm | {et_fl} | {v4b_fl} | **{fnum(fl)}** |
| MT mm | **{et_mt}** | {v4b_mt} | **{fnum(mt)}** |
| Official S | null | null | null |

## Why attention stays blocked

1. **No independent expert GT** → cannot pin official S / true LB (no fake waiver).
2. **Fasc segmentation** native macro Dice **{dice:.4f}** (usable bar ~0.40 for MT/geometry path): {"PASS" if fasc_usable else "FAIL"}.
3. **ExtraTrees** still {"dominates" if et_still_better else "no longer clearly dominates"} PA/MT proxy vs seg→GH.
4. **`q` remains AmbrosM UI prior**; absolute scale not independently verified.
5. {status_extra}

## Attention training

- **Blocked = YES**
- Do **not** scaffold full attention train.
- Next levers: stronger fasc (HR multi-scale, HRNet/timm thin-structure, topology loss tune), apo boundary refine for MT, optional Kaggle tiny held-out for S pin.

## Protocol stamp

```yaml
attention_gate_verdict:
  date: {datetime.now(HK).strftime("%Y-%m-%d")}
  attention_blocked: true
  tier_b_test_path: direct_seg_geometry_head_ambrosm_v2
  fasc_winner: {Path(ckpt).name}
  fasc_variant: {variant}
  fasc_val_native_macro_dice: {dice:.4f}
  apo_val_dice: 0.9689
  interim_proxy_floor: extratrees_tier_b_v2
  official_S: null
  seg_gh_val_mae: {{pa_deg: {fnum(pa)}, fl_mm: {fnum(fl)}, mt_mm: {fnum(mt)}}}
  test_ok_mm: 309/309
  baseline_fasc_dice_compare: 0.2676
```
"""
    args.out.write_text(md)
    # also dump machine-readable
    stamp = {
        "attention_blocked": True,
        "fasc_val_native_macro_dice": dice,
        "fasc_variant": variant,
        "fasc_ckpt": ckpt,
        "seg_gh_val_mae": {"pa_deg": pa, "fl_mm": fl, "mt_mm": mt, "diagnostic_s1": s1},
        "official_S": None,
        "fasc_usable_bar_0_40": fasc_usable,
        "et_still_better_pa_mt": et_still_better,
        "reason": status_extra,
    }
    args.out.with_suffix(".json").write_text(json.dumps(stamp, indent=2))
    print(json.dumps(stamp, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
