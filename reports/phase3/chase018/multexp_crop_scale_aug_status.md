# MultExp crop/scale + tiny-rot Frundles FT — 2026-11-06 22:27:25 UTC+08:00

## Result
| metric | value |
|---|---|
| **honest champ S** | **0.15111** (raw_crop_aug) |
| MAE PA / FL / MT | 1.027° / 2.754 mm / 0.158 mm |
| raw frozen S | 0.15111 (PA 1.027 / FL 2.754 / MT 0.158) |
| nested sin_w S | 0.15196 |
| Frundles map ref S | 0.13294 |
| vs honest hold 0.13449 | MISS (+0.01662) |
| gate 0.12 | FAIL |
| promote | False |
| domain-shift vs Frundles maps | LOSES (+0.01817) |
| train mix | {'labtrain': 1841, 'expdrawn': 199} |
| dropped near MultExp | {'labtrain': 13, 'expdrawn': 0} |
| best val Dice | 0.48484038871570223 |
| init | Frundles fasc_resnet34_384x576_f0.pt |
| aug | crop/scale [0.6, 1.0] + rot ±8° p=0.5 (same affine on image+mask) |
| epochs / bs / lr / device | 25 / 6 / 0.0001 / cuda:2 |
| elapsed | 16.0 min |
| no Kaggle | true |

## Protocol
MultExp dHash-held-out. Frozen recipe `hough_tex0.4_thr035_apo60_ff0.9_sin0.35_agNone`. Nested sin_w over [0.25, 0.28, 0.3, 0.35, 0.4] is honest LOOCV (pick on 34, score 1). Blend alphas are diagnostic only and not used for promotion. Comparator is honest joint S=0.13449, not the leaky 0.13195.

## Artifacts
- ckpt: `/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/seg_ckpts/fasc_resnet34_384x576_crop_scale_aug_best.pt`
- maps: `/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/cache/maps_multexp_crop_scale_aug`
- json: `/home/chocheng9/kaggle_UMUD/reports/phase3/chase018/multexp_crop_scale_aug.json`
