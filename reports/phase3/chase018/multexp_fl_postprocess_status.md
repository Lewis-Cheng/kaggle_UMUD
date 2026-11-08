# MultExp FL post-process status — 2026-11-05 13:02 Asia/Hong_Kong

## Result (Frundles maps only; no lab fasc ckpt; no Kaggle)
| metric | value |
|---|---|
| **best S_loocv** | **0.13294** |
| recipe | hough_tex0.4_thr035_apo60_ff0.9_sin0.35_agNone |
| MAE PA / FL / MT | 0.928° / 2.297 mm / 0.158 mm |
| vs prior floor 0.13761 | **BEAT** (-0.00467) |
| vs prior selector 0.13550 | **BEAT** (-0.00256) |
| gate S&lt;0.12 | **FAIL** (gap +0.01294) |

### Kw
{"fasc_thr": 0.35, "apo_thr": 0.6, "tex_w": 0.4, "fl_factor": 0.9, "fl_sin_w": 0.35, "agree_thr": null, "pa_mode": "hough"}

## Ladder
| step | S |
|---|---|
| Re-inferred promoted hough_tex03_thr032_apo60 | 0.13851 |
| thr/apo/tex grid best (thr035_apo60 tex0.3) | 0.13422 |
| **FL sin_w/agree post on tex0.4 thr035** | **0.13294** |
| FL-source oracle (ffsweep cands) | 0.08612 |
| Transferable FL selector (train-best / ols5) | 0.13744 / 0.14100 |

## Tried (FL post-process)
- Endpoint protocols (centre/deep_insert/shallow/both_inner), fl_factor grid, sin blends, ill-conditioned reject→fallback
- Fold-pure Softmax/L1 residual / kNN / simplex / nested cal-PA blends — none transferable below ~0.137
- Oracle proves headroom (~0.08–0.09) but sources are image-idiosyncratic (no stable selector)

## Ceiling
Frundles-map FL post-process + thr/tex micro-search ceiling **~0.133**. Gate 0.12 not reachable without better MultExp-domain fasc geometry.

## Next lever (non-FL-post)
**MultExp-domain fasc adaptation** (do not swap lab scratch/FT MultExp maps — already failed 0.14–0.15). Adapt fasc to MultExp appearance while keeping Frundles apo + current thr035/tex0.4/sin0.35 recipe.

## Artifacts
- cache/maps_multexp (Frundles re-infer)
- multexp_fl_postprocess_ffsweep.json / _scores.json / _grid.json / _summary.json
- math_map_promoted_recipe.json (new best)
- multexp_fl_postprocess_best_rows.csv
- scripts: eval_multexp_fl_postprocess.py, eval_multexp_fl_ffsweep.py
