---
title: MultExp is not the only reference
author: Lewis-Cheng
date: 2026-11-09
branch: main
id: l-b2718548
scope: repo
status: active
src: agent
touches:
  - src/umud/calibration/scale_probe.py
  - scripts/chase018/build_scale_probe.py
---

MultExpData under data/external/umud_expert_arch_v0.1.0 is one external expert set of 35 images. It is not the reference model to train or to select against. scripts/chase018/build_scale_probe.py uses its median gt/pred only to pick the sign of the single 0.97 FL/MT probe in src/umud/calibration/scale_probe.py. That sign does not mean MultExp anatomy is the one to match.

A replacement reference should be trained from scratch on the competition apo and fasc masks, with MultExp held out of the fit. Do not drop that model because its MultExp S is worse than Frundles or texture_rl. The promotion check stays the public score against reports/phase3/chase018/submission_texture_rl.csv (0.29895). Do not fine-tune the new net on MultExp labels to chase S_expert_NOT_LB.
