"""Discrete-action geometry refiner env after frozen apo + fasc_v5.

Actions blend GeometryHead (GH) with an ExtraTrees/median prior, or abort
to the prior. Reward uses official UMUD S formulas vs MultExpData expert
consensus (label S_expert_NOT_LB). Does not update seg weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from umud.metric.score import TOLERANCES, score_components

# Discrete actions (interpretable post-v5 knobs).
ACTION_NAMES = (
    "gh_only",           # 0
    "blend_gh75_et25",   # 1
    "blend_gh50_et50",   # 2
    "blend_gh25_et75",   # 3
    "et_only",           # 4
    "median_prior",      # 5
    "unable_use_prior",  # 6  (same as et_only but tagged; shaped separately)
    "target_mix_et_pa_mt_gh_fl",  # 7 PA/MT from ET, FL from GH
)

ALPHA_BY_ACTION = {
    0: 1.0,
    1: 0.75,
    2: 0.50,
    3: 0.25,
    4: 0.0,
    5: None,  # median
    6: 0.0,   # unable → prior
    7: None,  # custom mix
}

N_ACTIONS = len(ACTION_NAMES)


@dataclass
class GeometrySample:
    image_id: str
    gh_pa: float
    gh_fl: float
    gh_mt: float
    et_pa: float
    et_fl: float
    et_mt: float
    med_pa: float
    med_fl: float
    med_mt: float
    gt_pa: float | None = None
    gt_fl: float | None = None
    gt_mt: float | None = None
    apo_fg_frac: float = 0.0
    fasc_fg_frac: float = 0.0
    ok_geometry: float = 1.0
    unknown_scale: float = 0.0
    # optional extras
    iqr_pa: float = 0.0
    iqr_fl: float = 0.0
    iqr_mt: float = 0.0


def build_state(sample: GeometrySample) -> np.ndarray:
    """Compact state vector for contextual bandit / REINFORCE."""
    d_pa = sample.gh_pa - sample.et_pa
    d_fl = sample.gh_fl - sample.et_fl
    d_mt = sample.gh_mt - sample.et_mt
    return np.asarray(
        [
            sample.gh_pa / 30.0,
            sample.gh_fl / 80.0,
            sample.gh_mt / 30.0,
            sample.et_pa / 30.0,
            sample.et_fl / 80.0,
            sample.et_mt / 30.0,
            d_pa / 20.0,
            d_fl / 30.0,
            d_mt / 15.0,
            sample.apo_fg_frac,
            sample.fasc_fg_frac,
            sample.ok_geometry,
            sample.unknown_scale,
            abs(d_pa) / 20.0,
            abs(d_fl) / 30.0,
            abs(d_mt) / 15.0,
            1.0 if sample.fasc_fg_frac < 0.01 else 0.0,
            1.0 if sample.fasc_fg_frac < 0.005 else 0.0,
        ],
        dtype=np.float64,
    )


STATE_DIM = 18


def apply_action(sample: GeometrySample, action: int) -> dict[str, float | int | str]:
    """Return refined PA/FL/MT for a discrete action."""
    if action < 0 or action >= N_ACTIONS:
        raise ValueError(f"bad action {action}")
    name = ACTION_NAMES[action]
    unable = action == 6
    if action == 5 or action == 6:
        pa, fl, mt = sample.med_pa, sample.med_fl, sample.med_mt
        if action == 6:
            # unable prefers ET if available else median
            pa, fl, mt = sample.et_pa, sample.et_fl, sample.et_mt
    elif action == 7:
        pa, fl, mt = sample.et_pa, sample.gh_fl, sample.et_mt
    else:
        a = float(ALPHA_BY_ACTION[action])  # type: ignore[arg-type]
        pa = a * sample.gh_pa + (1.0 - a) * sample.et_pa
        fl = a * sample.gh_fl + (1.0 - a) * sample.et_fl
        mt = a * sample.gh_mt + (1.0 - a) * sample.et_mt
    return {
        "pa_deg": float(pa),
        "fl_mm": float(fl),
        "mt_mm": float(mt),
        "action": int(action),
        "action_name": name,
        "unable": bool(unable),
    }


def reward_expert(
    pred: Mapping[str, float],
    sample: GeometrySample,
    *,
    lambda_mae: float = 0.15,
    lambda_unable: float = 0.25,
) -> dict[str, Any]:
    """Reward from MultExpData expert consensus. Higher is better for RL.

    Base: -S_expert (official S is lower-better). MAE penalty + unable penalty
    when experts agree the image is measurable (IQR not huge).
    """
    if sample.gt_pa is None or sample.gt_fl is None or sample.gt_mt is None:
        raise ValueError("reward_expert requires GT on sample")
    err = {
        "pa_deg": np.asarray([abs(float(pred["pa_deg"]) - float(sample.gt_pa))]),
        "fl_mm": np.asarray([abs(float(pred["fl_mm"]) - float(sample.gt_fl))]),
        "mt_mm": np.asarray([abs(float(pred["mt_mm"]) - float(sample.gt_mt))]),
    }
    comps = score_components(err)
    s = float(comps["S"])
    mae_term = (
        err["pa_deg"][0] / TOLERANCES["pa_deg"]
        + err["fl_mm"][0] / TOLERANCES["fl_mm"]
        + err["mt_mm"][0] / TOLERANCES["mt_mm"]
    ) / 3.0
    experts_agree = (
        sample.iqr_pa < 4.0 and sample.iqr_fl < 15.0 and sample.iqr_mt < 2.0
    )
    unable = bool(pred.get("unable", False))
    unable_pen = lambda_unable if (unable and experts_agree) else 0.0
    r = -s - lambda_mae * mae_term - unable_pen
    return {
        "reward": float(r),
        "S_expert_NOT_LB": s,
        "S1": float(comps["S1"]),
        "mae_term": float(mae_term),
        "unable_pen": float(unable_pen),
    }


def batch_score(
    preds: Sequence[Mapping[str, float]],
    samples: Sequence[GeometrySample],
) -> dict[str, float]:
    """Official S over a set of predictions (lower better)."""
    pa = np.asarray([abs(p["pa_deg"] - s.gt_pa) for p, s in zip(preds, samples)], dtype=np.float64)
    fl = np.asarray([abs(p["fl_mm"] - s.gt_fl) for p, s in zip(preds, samples)], dtype=np.float64)
    mt = np.asarray([abs(p["mt_mm"] - s.gt_mt) for p, s in zip(preds, samples)], dtype=np.float64)
    comps = score_components({"pa_deg": pa, "fl_mm": fl, "mt_mm": mt})
    return {
        "S_expert_NOT_LB": float(comps["S"]),
        "S1": float(comps["S1"]),
        "S2": float(comps["S2"]),
        "S3": float(comps["S3"]),
        "MAE_PA": float(pa.mean()),
        "MAE_FL": float(fl.mean()),
        "MAE_MT": float(mt.mean()),
        "n": float(len(samples)),
    }
