# MultExp scratch_best status

- asof: 2026-11-05 12:23 Asia/Hong_Kong
- cache: reports/phase3/chase018/cache/maps_multexp_scratch_best
- ckpt: fasc_resnet34_384x576_scratch_best.pt (Dice ~0.4136)
- best variant: hough_tex0.4_thr028_apo60 S_loocv=0.14444 S_raw=0.15512 MAE PA=0.988 FL=2.590 MT=0.158
- promoted hough_tex03_thr032_apo60: S_loocv=0.14956 S_raw=0.15604 MAE PA=1.007 FL=2.737 MT=0.158
- band_agree_v1: S_loocv=0.15754 S_raw=0.16662 MAE PA=1.167 FL=2.691 MT=0.162
- gate_0.12: False
- vs Frundles floor 0.13761: FAIL (best gap +0.00683)
- vs selector 0.13550: FAIL
- vs overnight FT 0.15558 / scratch60_s7 0.13993: best 0.14444 beats FT, loses to scratch60_s7 and Frundles
- kaggle_submit: false

