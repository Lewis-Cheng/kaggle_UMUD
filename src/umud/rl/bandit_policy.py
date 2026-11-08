"""Softmax linear contextual bandit / REINFORCE for discrete geometry actions.

n=35 → heavy L2 reg, few actions, LOOCV. Pure numpy (no torch required).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from umud.rl.geometry_env import (
    ACTION_NAMES,
    N_ACTIONS,
    STATE_DIM,
    GeometrySample,
    apply_action,
    batch_score,
    build_state,
    reward_expert,
)


@dataclass
class BanditConfig:
    n_actions: int = N_ACTIONS
    state_dim: int = STATE_DIM
    lr: float = 0.05
    l2: float = 0.05
    epochs: int = 80
    seed: int = 0
    temperature: float = 1.0


class SoftmaxBandit:
    """Linear scores → softmax policy; REINFORCE with baseline."""

    def __init__(self, cfg: BanditConfig | None = None):
        self.cfg = cfg or BanditConfig()
        rng = np.random.default_rng(self.cfg.seed)
        # W: [n_actions, state_dim]
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
        """REINFORCE: grad log π(a|s) * advantage, with L2 on W."""
        p = self.probs(state)
        # d logits: one-hot - probs
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
    """Enumerate actions; pick lowest S (then highest reward)."""
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
    preds = []
    actions = []
    for s in samples:
        st = build_state(s)
        a = policy.act(st, greedy=greedy)
        pred = apply_action(s, a)
        preds.append(pred)
        actions.append(a)
    score = batch_score(preds, samples)
    # action histogram
    hist = {ACTION_NAMES[i]: int((np.asarray(actions) == i).sum()) for i in range(N_ACTIONS)}
    score["action_hist"] = hist
    return score


def train_reinforce_loocv(
    samples: list[GeometrySample],
    *,
    cfg: BanditConfig | None = None,
) -> dict[str, Any]:
    """Leave-one-out: train on 34, greedy-eval on held-out; aggregate S."""
    cfg = cfg or BanditConfig()
    n = len(samples)
    holdout_preds: list[dict[str, float]] = []
    holdout_samples: list[GeometrySample] = []
    holdout_actions: list[int] = []
    fold_rows: list[dict[str, Any]] = []

    # Baselines on full set (for gate)
    gh_preds = [apply_action(s, 0) for s in samples]
    et_preds = [apply_action(s, 4) for s in samples]
    med_preds = [apply_action(s, 5) for s in samples]
    gh_score = batch_score(gh_preds, samples)
    et_score = batch_score(et_preds, samples)
    med_score = batch_score(med_preds, samples)

    rng = np.random.default_rng(cfg.seed)

    for i in range(n):
        train = [samples[j] for j in range(n) if j != i]
        test = samples[i]
        pol = SoftmaxBandit(BanditConfig(**{**asdict(cfg), "seed": cfg.seed + i}))

        # behavior clone toward oracle for a few epochs, then REINFORCE
        for _ in range(max(10, cfg.epochs // 4)):
            s = train[int(rng.integers(0, len(train)))]
            st = build_state(s)
            a_star, _ = oracle_best_action(s)
            # supervised nudge
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

    # Final policy trained on all 35 (for test inference)
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

    # Gate: LOOCV must beat frozen GH-only on same MultExpData folds
    gate_pass = loocv_score["S_expert_NOT_LB"] < gh_score["S_expert_NOT_LB"] - 1e-6

    return {
        "label": "S_expert_NOT_LB",
        "claims_kaggle_lb": False,
        "n": n,
        "loocv": loocv_score,
        "loocv_action_hist": hist,
        "baseline_gh_only": gh_score,
        "baseline_et_only": et_score,
        "baseline_median": med_score,
        "final_policy_full_greedy": full_eval,
        "gate_beat_gh": gate_pass,
        "gate_rule": "loocv_S < gh_only_S",
        "fold_rows": fold_rows,
        "policy": final,
        "action_names": list(ACTION_NAMES),
    }
