# Attention gate verdict — HARD STOP

**Updated:** 2026-11-04 20:12 HKT
**Decision: `attention_blocked = YES`** — do **not** start expensive attention training.

## Gates
| Gate | Status | Notes |
|---|---|---|
| Official metric / scorer pinned | PASS | PA/6° FL/12mm MT/3mm |
| Split / leak audit | PASS | duplicate-aware SHA split |
| Physical scale (AmbrosM) | PASS enough | test 309/309 |
| Fasc segmentation usable (~0.40 Dice) | **FAIL** | winner v5 **0.2846**; v6–v11 no unlock; v11 **0.2244** |
| Expert GT / official S on val | **FAIL** | S=null |
| Honest waiver | **NO** | would be fake |

## Floors (NOT LB / NOT expert)
- ExtraTrees Tier B v2: PA **2.23°** FL 9.86mm MT **1.00mm**
- Seg→GH v5 vs GT-mask GH: PA 3.93° FL **7.69mm** MT 2.96mm

## Hard stop
Fasc stronger campaign closed after v11. Thin-label ceiling.

```yaml
attention_blocked: true
hard_stop: true
fasc_winner: seg_fasc_v5_best.pt
fasc_val_native_macro_dice: 0.2846
official_S: null
interim_floor: extratrees_tier_b_v2
test_policy: direct_seg_gh_v5_ambrosm
```
