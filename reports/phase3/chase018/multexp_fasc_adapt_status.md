# MultExp fasc adaptation — 2026-11-05 13:23 Asia/Hong_Kong

## Result
- **Best honest S = 0.13195** (`nested_feature_rule_fasc_adapt`)  
  PA 0.912° / FL 2.292 mm / MT 0.158 mm  
  vs prior fixed Frundles **0.13294** (Δ -0.00098)
- **Gate 0.12: FAIL** (gap 0.01195)
- Recipe JSON updated; **no Kaggle**

## Tried
| Track | Best S | Notes |
|-------|--------|-------|
| A single post-filters | 0.13294 | none beat baseline |
| C nested feature_rule | **0.13195** | promoted; mostly baseline↔blend_base_morph |
| C knn k7 | 0.13230 | slight honest gain |
| B self-train (teacher morph+ridge) | 0.15213 | worse; discarded |
| v1 leaky rule | 0.12963 | discarded (test leak) |
| Oracle A pool / full | 0.122 / 0.107 | not transferable |

## Diagnosis
Frundles MultExp fasc + morph/ridge/corridor post-filters are near ceiling for transferable S (~0.132). Nested C buys ~0.001. B distilling those teachers into a U-Net **hurt** FL (2.80 mm). Remaining gap to 0.12 needs fold-safe geometry-supervised fasc adaptation, not more unsupervised teacher morph.

## Next lever (≤3 sentences)
Run nested/group-safe MultExp fasc fine-tune that optimizes a PA/FL geometry proxy on train folds only (freeze apo; keep frozen hough_tex0.4/thr035/sin0.35 recipe). Morph/self-train teachers are exhausted. Still no Kaggle until S&lt;0.12.
