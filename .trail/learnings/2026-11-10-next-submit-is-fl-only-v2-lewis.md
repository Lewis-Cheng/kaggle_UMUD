---
title: Next submit is FL-only v2
author: Lewis-Cheng
date: 2026-11-10
branch: main
id: l-2e9a7470
scope: repo
status: active
src: agent
---

Public target is 0.16 or lower. Best public file is still texture_rl at 0.29895. The 0.97 FL+MT probe scored 0.35218, so do not shrink both lengths.

build_single_target_and_level_probes.py in scripts/chase018 writes submission_v2_fl_mm_only.csv: FL ordering from the v2 geometry file, PA and MT copied from texture_rl. Submit that file alone. S is additive, so one column is the only score you can attribute. The other single-target splits and the FL/MT level probes wait until that public score is back.
