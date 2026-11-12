"""Medical-RL fascicle-line agent (landmark-agent style DQN).

A line hypothesis is ``(y0 + ky, t0 + kt * DELTA_DEG)`` through a fixed anchor
column ``x0``. Each move changes ``ky`` or ``kt`` by one unit, matching
``UmudRl.Act`` in formal/umud_rl. The reward is potential-based and matched
to the learner's discount: ``phi(s) - gamma * phi(s') - step_cost``. The
discounted return then equals ``phi(first) - gamma**T * phi(last) -
step_cost * (1 + ... + gamma**(T-1))`` (``disc_return_telescopes``).
Only ``gamma = 1`` ranks episodes of different lengths by the deployment
objective ``phi(last) + step_cost * T`` (``shaped_return_objective``); with
``gamma < 1`` the ``gamma**T`` weight pays the agent for running out the clock,
which a trained agent exploited on synthetic data. DQN therefore trains with
``gamma = 1``, stabilised by reward scaling, a lower learning rate, and slower
target updates (unscaled ``gamma = 1`` diverged:
reports/phase3/chase018/attention_rl/dqn_gamma1_diverged.log).
A move into an already visited state ends the episode as ``stop``, in training
and at inference alike; in the Lean model that is a history-dependent policy.

Targets are annotated lines from competition masks, so the agent trains on
thousands of lines, not on the 35 MultExp labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from umud.geometry.attended_readout import PixelLine, fold

DELTA_DEG = 0.5
T_MAX = 40
ACTIONS = ("up", "down", "rot+", "rot-", "stop")
STOP = 4
STEP_COST = 0.05
W_ANGLE = 1.0  # per DELTA_DEG
W_OFFSET = 1.0  # per grid pixel
PATCH_ALONG = 32
PATCH_PERP = 17
ALONG_STEP = 1.5
N_SCALARS = 5

MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (0, 0)}


def perp_distance(x: float, y: float, target: np.ndarray) -> float:
    x1, y1, x2, y2 = (float(v) for v in target[:4])
    dx, dy = x2 - x1, y2 - y1
    n = np.hypot(dx, dy)
    if n < 1e-9:
        return float(np.hypot(x - x1, y - y1))
    return float(abs(dy * (x - x1) - dx * (y - y1)) / n)


def potential(x: float, y: float, t_deg: float, target: np.ndarray) -> float:
    return W_ANGLE * abs(float(fold(t_deg - float(target[4])))) / DELTA_DEG + W_OFFSET * perp_distance(x, y, target)


def nearest_target(x: float, y: float, t_deg: float, targets: np.ndarray) -> int:
    return int(np.argmin([potential(x, y, t_deg, tg) for tg in targets]))


def sample_patch(channels: np.ndarray, x: float, y: float, t_deg: float) -> np.ndarray:
    t = np.radians(t_deg)
    u = (np.arange(PATCH_ALONG) - (PATCH_ALONG - 1) / 2.0) * ALONG_STEP
    v = np.arange(PATCH_PERP) - (PATCH_PERP - 1) / 2.0
    vv, uu = np.meshgrid(v, u, indexing="ij")
    map_x = (x + uu * np.cos(t) - vv * np.sin(t)).astype(np.float32)
    map_y = (y + uu * np.sin(t) + vv * np.cos(t)).astype(np.float32)
    return np.stack(
        [cv2.remap(c, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) for c in channels]
    ).astype(np.float32)


@dataclass
class FascicleEnv:
    channels: np.ndarray  # (C, H, W) float32
    x0: float
    y0: float
    t0: float
    target: np.ndarray | None = None
    sup_poly: np.ndarray | None = None  # centre curves in grid pixels
    deep_poly: np.ndarray | None = None
    t_max: int = T_MAX
    step_cost: float = STEP_COST
    gamma: float = 1.0
    ky: int = 0
    kt: int = 0
    moves: int = 0
    done: bool = False
    history: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.t_max < 1:
            raise ValueError("t_max must be at least 1")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.history = [(0, 0)]

    @property
    def y(self) -> float:
        return self.y0 + self.ky

    @property
    def t(self) -> float:
        return self.t0 + self.kt * DELTA_DEG

    def line(self, weight: float = 1.0) -> PixelLine:
        return PixelLine(self.x0, self.y, float(fold(self.t)), weight)

    def phi(self) -> float:
        if self.target is None:
            raise ValueError("potential needs a target line")
        return potential(self.x0, self.y, self.t, self.target)

    def scalars(self) -> np.ndarray:
        rel_ang, rel_depth = 0.0, 0.5
        if self.deep_poly is not None:
            beta = float(np.degrees(np.arctan(np.polyval(np.polyder(self.deep_poly), self.x0))))
            rel_ang = float(fold(self.t - beta)) / 45.0
        if self.sup_poly is not None and self.deep_poly is not None:
            ys, yd = float(np.polyval(self.sup_poly, self.x0)), float(np.polyval(self.deep_poly, self.x0))
            if yd - ys > 1e-6:
                rel_depth = (self.y - ys) / (yd - ys)
        return np.array([rel_ang, rel_depth, self.ky / self.t_max, self.kt / self.t_max, self.moves / self.t_max], np.float32)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        return sample_patch(self.channels, self.x0, self.y, self.t), self.scalars()

    def next_state(self, action: int) -> tuple[int, int]:
        dy, dt = MOVES[action]
        return self.ky + dy, self.kt + dt

    def step(self, action: int) -> tuple[float, bool]:
        if self.done:
            raise RuntimeError("episode finished")
        if action == STOP or self.next_state(action) in self.history:
            self.done = True
            return 0.0, True
        before = self.phi() if self.target is not None else 0.0
        self.ky, self.kt = self.next_state(action)
        self.moves += 1
        self.history.append((self.ky, self.kt))
        after = self.phi() if self.target is not None else 0.0
        self.done = self.moves >= self.t_max
        return before - self.gamma * after - self.step_cost, self.done


class QNet(nn.Module):
    def __init__(self, in_ch: int = 3, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_ch, PATCH_PERP, PATCH_ALONG)).numel()
        self.fc = nn.Sequential(nn.Linear(n_flat + N_SCALARS, 128), nn.ReLU(inplace=True), nn.Linear(128, n_actions))

    def forward(self, patch: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.conv(patch).flatten(1), scalars], 1))


Policy = Callable[[FascicleEnv], int]


def greedy_policy(qnet: QNet, device: str | torch.device = "cpu") -> Policy:
    def act(env: FascicleEnv) -> int:
        patch, sc = env.observe()
        with torch.no_grad():
            q = qnet(torch.from_numpy(patch)[None].to(device), torch.from_numpy(sc)[None].to(device))[0]
        return int(q.argmax())

    return act


def rollout(env: FascicleEnv, policy: Policy) -> dict[str, list]:
    states = [(env.ky, env.kt)]
    rewards: list[float] = []
    phis = [env.phi()] if env.target is not None else []
    while not env.done:
        moves = env.moves
        r, _ = env.step(policy(env))
        rewards.append(r)
        if env.moves == moves:
            break
        states.append((env.ky, env.kt))
        if env.target is not None:
            phis.append(env.phi())
    return {"states": states, "rewards": rewards, "phis": phis}


@dataclass
class DQNConfig:
    steps: int = 20000
    batch: int = 64
    gamma: float = 1.0
    reward_scale: float = 0.1  # positive scaling leaves the greedy policy unchanged
    lr: float = 5e-4
    buffer: int = 30000
    warmup: int = 500
    target_every: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_steps: int = 10000
    seed: int = 0


EpisodeSource = Callable[[np.random.Generator], FascicleEnv]


def train_dqn(source: EpisodeSource, cfg: DQNConfig, device: str | torch.device = "cpu", log: Callable[[dict], None] | None = None) -> QNet:
    if not 0.0 < cfg.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if cfg.reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    def fresh() -> FascicleEnv:
        e = source(rng)
        e.gamma = cfg.gamma
        return e

    env = fresh()
    in_ch = env.channels.shape[0]
    q, q_tgt = QNet(in_ch).to(device), QNet(in_ch).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    P = np.zeros((cfg.buffer, in_ch, PATCH_PERP, PATCH_ALONG), np.float32)
    S = np.zeros((cfg.buffer, N_SCALARS), np.float32)
    P2, S2 = np.zeros_like(P), np.zeros_like(S)
    A = np.zeros(cfg.buffer, np.int64)
    R = np.zeros(cfg.buffer, np.float32)
    D = np.zeros(cfg.buffer, np.float32)
    n, ptr = 0, 0
    ep_returns: list[float] = []
    ep_ret = 0.0
    obs = env.observe()
    for step in range(cfg.steps):
        eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * max(0.0, 1.0 - step / cfg.eps_steps)
        if rng.random() < eps:
            a = int(rng.integers(len(ACTIONS)))
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(obs[0])[None].to(device), torch.from_numpy(obs[1])[None].to(device))[0].argmax())
        r, done = env.step(a)
        nxt = env.observe()
        P[ptr], S[ptr], A[ptr], R[ptr], D[ptr] = obs[0], obs[1], a, r * cfg.reward_scale, float(done)
        P2[ptr], S2[ptr] = nxt
        ptr = (ptr + 1) % cfg.buffer
        n = min(n + 1, cfg.buffer)
        ep_ret += r
        obs = nxt
        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            env = fresh()
            obs = env.observe()
        if n >= cfg.warmup:
            idx = rng.integers(0, n, cfg.batch)
            p, s = torch.from_numpy(P[idx]).to(device), torch.from_numpy(S[idx]).to(device)
            p2, s2 = torch.from_numpy(P2[idx]).to(device), torch.from_numpy(S2[idx]).to(device)
            a_t = torch.from_numpy(A[idx]).to(device)
            r_t = torch.from_numpy(R[idx]).to(device)
            d_t = torch.from_numpy(D[idx]).to(device)
            with torch.no_grad():
                a2 = q(p2, s2).argmax(1, keepdim=True)
                y = r_t + cfg.gamma * (1 - d_t) * q_tgt(p2, s2).gather(1, a2)[:, 0]
            pred = q(p, s).gather(1, a_t[:, None])[:, 0]
            loss = F.smooth_l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            if step % cfg.target_every == 0:
                q_tgt.load_state_dict(q.state_dict())
            if log is not None and step % 1000 == 0:
                recent = ep_returns[-200:]
                log({"step": step, "loss": float(loss.item()), "eps": eps, "mean_return_200": float(np.mean(recent)) if recent else 0.0})
    q.eval()
    return q


def refine_lines(
    qnet: QNet,
    channels: np.ndarray,
    lines: Sequence[PixelLine],
    sup_poly: np.ndarray | None = None,
    deep_poly: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> list[PixelLine]:
    policy = greedy_policy(qnet, device)
    out = []
    for ln in lines:
        env = FascicleEnv(channels, ln.x, ln.y, ln.t_deg, None, sup_poly, deep_poly)
        rollout(env, policy)
        out.append(env.line(ln.weight))
    return out


def line_errors(lines: Sequence[PixelLine], targets: np.ndarray) -> tuple[float, float]:
    """Mean absolute angle error (deg) and perpendicular offset (px) to the nearest target."""
    if len(lines) == 0 or len(targets) == 0:
        return float("nan"), float("nan")
    angs, offs = [], []
    for ln in lines:
        k = nearest_target(ln.x, ln.y, ln.t_deg, targets)
        angs.append(abs(float(fold(ln.t_deg - float(targets[k][4])))))
        offs.append(perp_distance(ln.x, ln.y, targets[k]))
    return float(np.mean(angs)), float(np.mean(offs))
