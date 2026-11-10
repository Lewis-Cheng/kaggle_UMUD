---
title: Retrain with aug L2 norm and Optuna
author: Lewis-Cheng
date: 2026-11-09
branch: main
id: l-c5f89dc9
scope: repo
status: active
src: agent
touches:
  - scripts/chase018/train_ref_optuna.py
  - scripts/chase018/run_ref_optuna.sh
---

The new reference in scripts/chase018/train_ref_optuna.py is trained from ImageNet on the competition apo and fasc masks. MultExp stays out of the fit and out of the objective.

Keep all four of these on every trial: data augmentation (crop, flip, gamma, brightness, speckle, small rotation), L2 through AdamW weight decay, normalization (ImageNet input mean/std plus ResNet BatchNorm), and Optuna for lr, L2, crop scale, rotation probability, and positive weight. Do not drop one of them to chase a MultExp score. Promotion remains the public score against submission_texture_rl.csv.
