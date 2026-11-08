# Fasc stronger status — HARD STOP

**Date:** 2026-11-04 20:12 HKT
**attention_blocked:** YES
**Official S:** null
**hard_stop:** YES

## Decision
Stop inventing fasc architecture variants. Plateau ~0.27-0.29 (winner **v5 = 0.2846**); usable bar ~0.40 unmet. v11 skeleton lever **failed** (native Dice **0.2244**).

## Native Dice
| Variant | Dice |
|---|---:|
| **v5 MSTinyUNet (WINNER)** | **0.2846** |
| v10 Focal-Tversky+crop | 0.2837 |
| v6b HRNet | 0.2835 |
| v7 / v6 | ~0.283 |
| v9 EffNet-B3 | 0.2771 |
| v4b | 0.2676 |
| **v11 skeleton/centerline** | **0.2244** |

## Why still blocked
1. Fasc Dice << 0.40 for MT/geometry attention path.
2. No independent expert GT → official S=null (no honest waiver).
3. ExtraTrees Tier B v2 still better on PA/MT vs seg→GH v5.

## Frozen policy
- attention_blocked = YES — do not start expensive attention training
- Interim floor: ExtraTrees Tier B v2
- Test policy: direct seg→GH with v5 + AmbrosM
- Winner ckpt: reports/phase3/seg_ckpts/seg_fasc_v5_best.pt

## Reopen only if
- Better fasc labels / SAM install with real Dice past ~0.40
- Expert GT CSV enabling official S
- Kaggle tiny held-out that honestly pins S
- Not another random UNet on the same thin labels
