# MultExp nested fasc FT — 2026-11-05 14:02 Asia/Hong_Kong

## Result (honest nested-safe)
- **Best still prior C S = 0.13195** (
ested_feature_rule_fasc_adapt)
  PA 0.912° / FL 2.292 mm / MT 0.158 mm
- This job **did not promote** (no nested-safe S < 0.13195)
- **Gate 0.12: FAIL**
- **No Kaggle** | apo frozen | recipe unchanged

## Protocol tried
1. Group-safe 5-fold / 35-fold LOOCV fasc FT from Frundles weights (not lab scratch_best)
2. Geo proxies: stripe PA/FL paint + orient-reweight (abandoned — targets worse than Frundles)
3. GT-selected A/blend teachers on train folds only + U-Net distill + nested Frundles blend
4. Expanded 25-variant nested trainS / knn / feature_rule (no FT)

## Key numbers
| Track | S | Notes |
|-------|---|-------|
| Prior C (hold) | **0.13195** | still best honest |
| Stripe smoke FT | 0.579 | abandoned |
| 5-fold GT-teacher FT | 0.14469 | worse |
| LOOCV FT pure | 0.13566 | worse |
| LOOCV nested blend | 0.13504 | mostly α=0 |
| Expanded nested trainS | 0.13506 | worse than C |
| Expanded feature_rule | 0.14059 | worse |
| Leaky expanded oracle | **0.10485** | gate-pass upper bound (not nested) |

## Diagnosis
Differentiable/stripe PA/FL geo-paint proxy is a bad MultExp teacher (target S 0.14-0.29). GT-selected A/blend teachers have leaky oracle S=0.1049 (clears 0.12) but U-Net distill (5-fold and 35-fold LOOCV) and nested selectors on the expanded pool all land 0.135-0.145, worse than prior C 0.13195. Transferable per-image variant choice is the bottleneck, not more Frundles FT.

## Next lever
Build a stronger fold-pure variant predictor (richer features / small calibrated classifier) aimed at the expanded 25-variant GT labels (leaky ceiling 0.105), or generate fasc maps outside the Frundles+A family (e.g. MultExp-paired public labels / geometry-aware inference that does not rely on U-Net FT). Do not repeat stripe paint or morph-teacher self-train.
