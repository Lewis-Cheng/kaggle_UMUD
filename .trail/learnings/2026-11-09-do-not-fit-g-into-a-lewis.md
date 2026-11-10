---
title: Do not fit G into a submission
author: Lewis-Cheng
date: 2026-11-09
branch: main
id: l-a076f673
scope: repo
status: active
src: agent
touches:
  - src/umud/geometry/physical.py
  - scripts/chase018/infer_rl_refiner.py
  - src/umud/geometry/host_faithful.py
---

Recovering a physical metric G does not authorize a new Kaggle CSV. src/umud/geometry/physical.py already implements the diagonal spacing metric and the inverse-affine pullback; leave that algebra frozen, because the missing piece is an identified anchor, not a different formula.

The public base is reports/phase3/chase018/submission_texture_rl.csv (submission 58005465, public 0.29895). The next file allowed to be submitted is one single-factor probe that multiplies fl_mm and mt_mm by one scale and copies pa_deg unchanged. Do not submit a G-fit, a T/sin(phi) rewrite, another fascicle network, or the fold-pure joint stack.

MultExp S is not the public-score gate. Honest MultExp S 0.13449 scored about 0.43–0.44 public, while texture_rl at offline S about 0.157 scored 0.29895. Promote a probe only when its public score beats 0.29895.

scripts/chase018/infer_rl_refiner.py must copy official image ids literally and keep FL as the identity (cal_fl = raw FL). PA and MT use the declared affine. If the manifest says FL is applied, abort instead of enabling FL affine to paper over the mismatch.
