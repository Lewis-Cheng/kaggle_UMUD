"""Chase018 SoftmaxBandit: refine texture_pa AFTER host-faithful+texture (not encoder replace).

Does not mutate Track B geometry_env / bandit_policy. Action space is conservative
and centered on chase018 ch_* as primary; v5/ET are weak priors only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from umud.metric.score import TOLERANCES, score_components

ACTION_NAMES = (
    "ch_only",                 # 0
    "cal_pa_mt",               # 1 PA/MT affine; keep FL from ch
    "blend_ch75_v5_25",        # 2
    "blend_ch50_v5_50",        # 3
    "blend_ch75_et_25",        # 4
    "ch_fl_factor_0.85",       # 5
    "ch_fl_factor_0.95",       # 6
    "residual_or_scratch",    # 7 residual_small OR blend_ch_scratch50
)

N_ACTIONS = len(ACTION_NAMES)
STATE_DIM = 20

# Affine fits (texture_pa MultExp); FL intentionally unused (mae_after worse on floor fits).
# Overridden at train/infer time when rows carry cal_* already applied.
DEFAULT_PA_A, DEFAULT_PA_B = 0.8600171482013663, 2.8709384995145024
DEFAULT_MT_A, DEFAULT_MT_B = 1.005524099515498, 0.11485469843907695

RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


@dataclass
class GeometrySample:
    image_id: str
    ch_pa: float
    ch_fl: float
    ch_mt: float
    v5_pa: float
    v5_fl: float
    v5_mt: float
    et_pa: float
    et_fl: float
    et_mt: float
    med_pa: float
    med_fl: float
    med_mt: float
    cal_pa: float
    cal_fl: float
    cal_mt: float
    scratch_pa: float | None = None
    scratch_fl: float | None = None
    scratch_mt: float | None = None
    gt_pa: float | None = None
    gt_fl: float | None = None
    gt_mt: float | None = None
    apo_fg_frac: float = 0.05
    fasc_fg_frac: float = 0.02
    ok_geometry: float = 1.0
    iqr_pa: float = 0.0
    iqr_fl: float = 0.0
    iqr_mt: float = 0.0
    has_scratch: float = 0.0


def _clip(pa: float, fl: float, mt: float) -> tuple[float, float, float]:
    return (
        float(np.clip(pa, *RANGES["pa_deg"])),
        float(np.clip(fl, *RANGES["fl_mm"])),
        float(np.clip(mt, *RANGES["mt_mm"])),
    )


def build_state(sample: GeometrySample) -> np.ndarray:
    d_v5_pa = sample.ch_pa - sample.v5_pa
    d_v5_fl = sample.ch_fl - sample.v5_fl
    d_v5_mt = sample.ch_mt - sample.v5_mt
    d_et_pa = sample.ch_pa - sample.et_pa
    d_et_fl = sample.ch_fl - sample.et_fl
    d_et_mt = sample.ch_mt - sample.et_mt
    return np.asarray(
        [
            sample.ch_pa / 30.0,
            sample.ch_fl / 80.0,
            sample.ch_mt / 30.0,
            sample.v5_pa / 30.0,
            sample.v5_fl / 80.0,
            sample.v5_mt / 30.0,
            sample.et_pa / 30.0,
            sample.et_fl / 80.0,
            sample.et_mt / 30.0,
            d_v5_pa / 20.0,
            d_v5_fl / 30.0,
            d_v5_mt / 15.0,
            abs(d_et_pa) / 20.0,
            abs(d_et_fl) / 30.0,
            abs(d_et_mt) / 15.0,
            sample.apo_fg_frac,
            sample.fasc_fg_frac,
            sample.ok_geometry,
            sample.has_scratch,
            (sample.cal_pa - sample.ch_pa) / 10.0,
        ],
        dtype=np.float64,
    )


def apply_action(sample: GeometrySample, action: int) -> dict[str, float | int | str]:
    if action < 0 or action >= N_ACTIONS:
        raise ValueError(f"bad action {action}")
    name = ACTION_NAMES[action]
    if action == 0:
        pa, fl, mt = sample.ch_pa, sample.ch_fl, sample.ch_mt
    elif action == 1:
        pa, fl, mt = sample.cal_pa, sample.ch_fl, sample.cal_mt
    elif action == 2:
        a = 0.75
        pa = a * sample.ch_pa + (1 - a) * sample.v5_pa
        fl = a * sample.ch_fl + (1 - a) * sample.v5_fl
        mt = a * sample.ch_mt + (1 - a) * sample.v5_mt
    elif action == 3:
        a = 0.50
        pa = a * sample.ch_pa + (1 - a) * sample.v5_pa
        fl = a * sample.ch_fl + (1 - a) * sample.v5_fl
        mt = a * sample.ch_mt + (1 - a) * sample.v5_mt
    elif action == 4:
        a = 0.75
        pa = a * sample.ch_pa + (1 - a) * sample.et_pa
        fl = a * sample.ch_fl + (1 - a) * sample.et_fl
        mt = a * sample.ch_mt + (1 - a) * sample.et_mt
    elif action == 5:
        pa, fl, mt = sample.ch_pa, sample.ch_fl * 0.85 / 0.9, sample.ch_mt
        # ch already has fl_factor 0.9 baked in; 0.85 absolute ⇒ scale by 0.85/0.9
    elif action == 6:
        pa, fl, mt = sample.ch_pa, sample.ch_fl * 0.95 / 0.9, sample.ch_mt
    else:  # 7
        if (
            sample.has_scratch > 0.5
            and sample.scratch_pa is not None
            and sample.scratch_fl is not None
            and sample.scratch_mt is not None
            and np.isfinite(sample.scratch_pa)
        ):
            pa = 0.5 * sample.ch_pa + 0.5 * float(sample.scratch_pa)
            fl = 0.5 * sample.ch_fl + 0.5 * float(sample.scratch_fl)
            mt = 0.5 * sample.ch_mt + 0.5 * float(sample.scratch_mt)
            name = "blend_ch_scratch50"
        else:
            # residual_small: ch + clip(med(ch-v5)*0.25)
            d_pa = np.clip((sample.ch_pa - sample.v5_pa) * 0.25, -3.0, 3.0)
            d_fl = np.clip((sample.ch_fl - sample.v5_fl) * 0.25, -6.0, 6.0)
            d_mt = np.clip((sample.ch_mt - sample.v5_mt) * 0.25, -1.5, 1.5)
            # "med(ch-v5)" interpreted as the per-sample delta (single value); keep
            # conservative residual nudge toward disagreement direction.
            pa = sample.ch_pa + float(d_pa) * 0.0 + float(
                np.clip(np.median([sample.ch_pa - sample.v5_pa]) * 0.25, -3.0, 3.0)
            )
            fl = sample.ch_fl + float(
                np.clip(np.median([sample.ch_fl - sample.v5_fl]) * 0.25, -6.0, 6.0)
            )
            mt = sample.ch_mt + float(
                np.clip(np.median([sample.ch_mt - sample.v5_mt]) * 0.25, -1.5, 1.5)
            )
            name = "residual_small"
    pa, fl, mt = _clip(pa, fl, mt)
    return {
        "pa_deg": pa,
        "fl_mm": fl,
        "mt_mm": mt,
        "action": int(action),
        "action_name": name,
        "unable": False,
    }


def reward_expert(
    pred: Mapping[str, float],
    sample: GeometrySample,
    *,
    lambda_mae: float = 0.15,
) -> dict[str, Any]:
    if sample.gt_pa is None or sample.gt_fl is None or sample.gt_mt is None:
        raise ValueError("reward_expert requires GT")
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
    r = -s - lambda_mae * mae_term
    return {
        "reward": float(r),
        "S_expert_NOT_LB": s,
        "S1": float(comps["S1"]),
        "mae_term": float(mae_term),
    }


def batch_score(
    preds: Sequence[Mapping[str, float]],
    samples: Sequence[GeometrySample],
) -> dict[str, float]:
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


@dataclass
class BanditConfig:
    n_actions: int = N_ACTIONS
    state_dim: int = STATE_DIM
    lr: float = 0.05
    l2: float = 0.08
    epochs: int = 80
    seed: int = 0
    temperature: float = 1.0


class SoftmaxBandit:
    def __init__(self, cfg: BanditConfig | None = None):
        self.cfg = cfg or BanditConfig()
        rng = np.random.default_rng(self.cfg.seed)
        self.W = 0.01 * rng.normal(size=(self.cfg.n_actions, self.cfg.state_dim))
        self.b = np.zeros(self.cfg.n_actions, dtype=np.float64)

    def logits(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def probs(self, state: np.ndarray) -> np.ndarray:
        z = self.logits(state) / max(self.cfg.temperature, 1e-6)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, state: np.ndarray, *, greedy: bool = False, rng: np.random.Generator | None = None) -> int:
        p = self.probs(state)
        if greedy:
            return int(np.argmax(p))
        rng = rng or np.random.default_rng()
        return int(rng.choice(self.cfg.n_actions, p=p))

    def update(self, state: np.ndarray, action: int, advantage: float) -> None:
        p = self.probs(state)
        g = -p
        g[action] += 1.0
        g *= advantage
        self.W += self.cfg.lr * (np.outer(g, state) - self.cfg.l2 * self.W)
        self.b += self.cfg.lr * g

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b)
        path.with_suffix(path.suffix + ".cfg.json").write_text(
            json.dumps({"cfg": asdict(self.cfg), "action_names": list(ACTION_NAMES)}, indent=2)
        )

    @classmethod
    def load(cls, path: Path | str) -> "SoftmaxBandit":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        cfg_path = path.with_suffix(path.suffix + ".cfg.json")
        if cfg_path.is_file():
            meta = json.loads(cfg_path.read_text())
            cfg = BanditConfig(**meta["cfg"])
        else:
            cfg = BanditConfig()
        pol = cls(cfg)
        pol.W = np.asarray(data["W"], dtype=np.float64)
        pol.b = np.asarray(data["b"], dtype=np.float64)
        return pol


def oracle_best_action(sample: GeometrySample) -> tuple[int, dict[str, Any]]:
    best_a, best_r, best_info = 0, -1e18, {}
    for a in range(N_ACTIONS):
        pred = apply_action(sample, a)
        info = reward_expert(pred, sample)
        if info["reward"] > best_r:
            best_r = info["reward"]
            best_a = a
            best_info = {**info, **pred}
    return best_a, best_info


def evaluate_policy(
    policy: SoftmaxBandit,
    samples: list[GeometrySample],
    *,
    greedy: bool = True,
) -> dict[str, Any]:
    preds, actions = [], []
    for s in samples:
        a = policy.act(build_state(s), greedy=greedy)
        preds.append(apply_action(s, a))
        actions.append(a)
    score = batch_score(preds, samples)
    score["action_hist"] = {
        ACTION_NAMES[i]: int((np.asarray(actions) == i).sum()) for i in range(N_ACTIONS)
    }
    return score


def train_reinforce_loocv(
    samples: list[GeometrySample],
    *,
    cfg: BanditConfig | None = None,
    gate_baseline_s: float | None = None,
) -> dict[str, Any]:
    """LOOCV SoftmaxBandit. Gate: loocv S < ch_only S (or gate_baseline_s)."""
    cfg = cfg or BanditConfig()
    n = len(samples)
    holdout_preds: list[dict[str, float]] = []
    holdout_samples: list[GeometrySample] = []
    holdout_actions: list[int] = []
    fold_rows: list[dict[str, Any]] = []

    ch_preds = [apply_action(s, 0) for s in samples]
    cal_preds = [apply_action(s, 1) for s in samples]
    ch_score = batch_score(ch_preds, samples)
    cal_score = batch_score(cal_preds, samples)
    # also score each fixed action as reference
    action_baselines = {}
    for a in range(N_ACTIONS):
        action_baselines[ACTION_NAMES[a]] = batch_score(
            [apply_action(s, a) for s in samples], samples
        )

    rng = np.random.default_rng(cfg.seed)

    for i in range(n):
        train = [samples[j] for j in range(n) if j != i]
        test = samples[i]
        pol = SoftmaxBandit(BanditConfig(**{**asdict(cfg), "seed": cfg.seed + i}))

        for _ in range(max(10, cfg.epochs // 4)):
            s = train[int(rng.integers(0, len(train)))]
            st = build_state(s)
            a_star, _ = oracle_best_action(s)
            p = pol.probs(st)
            g = -p
            g[a_star] += 1.0
            pol.W += cfg.lr * (np.outer(g, st) - cfg.l2 * pol.W)
            pol.b += cfg.lr * g

        for _ep in range(cfg.epochs):
            order = rng.permutation(len(train))
            rewards = []
            batch: list[tuple[np.ndarray, int, float]] = []
            for idx in order:
                s = train[int(idx)]
                st = build_state(s)
                a = pol.act(st, greedy=False, rng=rng)
                pred = apply_action(s, a)
                info = reward_expert(pred, s)
                rewards.append(info["reward"])
                batch.append((st, a, info["reward"]))
            baseline = float(np.mean(rewards)) if rewards else 0.0
            for st, a, r in batch:
                pol.update(st, a, r - baseline)

        st = build_state(test)
        a = pol.act(st, greedy=True)
        pred = apply_action(test, a)
        holdout_preds.append(pred)
        holdout_samples.append(test)
        holdout_actions.append(a)
        info = reward_expert(pred, test)
        fold_rows.append(
            {
                "image_id": test.image_id,
                "action": a,
                "action_name": ACTION_NAMES[a],
                "pa_deg": pred["pa_deg"],
                "fl_mm": pred["fl_mm"],
                "mt_mm": pred["mt_mm"],
                "S_i": info["S_expert_NOT_LB"],
                "reward": info["reward"],
            }
        )

    loocv_score = batch_score(holdout_preds, holdout_samples)
    hist = {ACTION_NAMES[i]: int((np.asarray(holdout_actions) == i).sum()) for i in range(N_ACTIONS)}

    final = SoftmaxBandit(cfg)
    for _ in range(max(10, cfg.epochs // 4)):
        s = samples[int(rng.integers(0, n))]
        st = build_state(s)
        a_star, _ = oracle_best_action(s)
        p = final.probs(st)
        g = -p
        g[a_star] += 1.0
        final.W += cfg.lr * (np.outer(g, st) - cfg.l2 * final.W)
        final.b += cfg.lr * g
    for _ep in range(cfg.epochs):
        order = rng.permutation(n)
        rewards = []
        batch = []
        for idx in order:
            s = samples[int(idx)]
            st = build_state(s)
            a = final.act(st, greedy=False, rng=rng)
            pred = apply_action(s, a)
            info = reward_expert(pred, s)
            rewards.append(info["reward"])
            batch.append((st, a, info["reward"]))
        baseline = float(np.mean(rewards)) if rewards else 0.0
        for st, a, r in batch:
            final.update(st, a, r - baseline)

    full_eval = evaluate_policy(final, samples, greedy=True)
    baseline_s = float(gate_baseline_s) if gate_baseline_s is not None else ch_score["S_expert_NOT_LB"]
    gate_pass = loocv_score["S_expert_NOT_LB"] < baseline_s - 1e-6

    return {
        "label": "S_expert_NOT_LB",
        "claims_kaggle_lb": False,
        "n": n,
        "loocv": loocv_score,
        "loocv_action_hist": hist,
        "baseline_ch_only": ch_score,
        "baseline_cal_pa_mt": cal_score,
        "action_baselines": action_baselines,
        "final_policy_full_greedy": full_eval,
        "gate_beat_ch": gate_pass,
        "gate_rule": "loocv_S < ch_only_S",
        "gate_baseline_s": baseline_s,
        "fold_rows": fold_rows,
        "policy": final,
        "action_names": list(ACTION_NAMES),
    }
