"""Executable mirrors of formal/umud_rl statements, each with a mutant that must fail.

The Python code is not extracted to Lean. These tests run the real Python
functions against the same statements; mutants are injected into the code
path (environment subclasses, a discounted return, a flip that forgets a
sign), not into the test's own arithmetic. Fixtures only, no real images.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from umud.data.holdout import HoldoutLeak, assert_multexp_excluded  # noqa: E402
from umud.data.synthetic_us import make_sample  # noqa: E402
from umud.geometry.attended_readout import (  # noqa: E402
    PixelLine,
    fold,
    lines_from_attention,
    pa_of,
    readout,
)
from umud.geometry.host_faithful import line_curve_intersection, measure  # noqa: E402
from umud.models.spatial_attention_geo import (  # noqa: E402
    AttnGeoConfig,
    SpatialAttentionGeoNet,
    flip_output,
    predict_symmetric,
)
from umud.rl.fascicle_agent import (  # noqa: E402
    ACTIONS,
    MOVES,
    STEP_COST,
    STOP,
    T_MAX,
    FascicleEnv,
    QNet,
    greedy_policy,
    rollout,
)

ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = ROOT / "formal/umud_rl"

# Pinned Lean statements (whitespace-normalised). A silent change of what is
# proven fails here even when the proof still builds.
PINNED = {
    "rollout_length_le": "(pi : List Line → Act) : ∀ (n : Nat) (h : List Line) (l : Line), (rollout pi n h l).length ≤ n + 1",
    "rollout_drift": "(pi : List Line → Act) : ∀ (n : Nat) (h : List Line) (l m : Line), m ∈ rollout pi n h l → l.y - (n : Int) ≤ m.y ∧ m.y ≤ l.y + (n : Int) ∧ l.t - (n : Int) ≤ m.t ∧ m.t ≤ l.t + (n : Int)",
    "shaped_return_telescopes": "(phi : Line → Int) (c : Int) : ∀ (s : Line) (rest : List Line), shapedReturn phi c (s :: rest) = phi s - phi (final s rest) - c * (rest.length : Int)",
    "shaped_return_order": "(phi : Line → Int) (c : Int) (s : Line) (r1 r2 : List Line) (hlen : r1.length = r2.length) : shapedReturn phi c (s :: r1) ≤ shapedReturn phi c (s :: r2) ↔ phi (final s r2) ≤ phi (final s r1)",
    "shaped_return_order_free": "(phi : Line → Int) (s : Line) (r1 r2 : List Line) : shapedReturn phi 0 (s :: r1) ≤ shapedReturn phi 0 (s :: r2) ↔ phi (final s r2) ≤ phi (final s r1)",
    "shaped_return_objective": "(phi : Line → Int) (c : Int) (s : Line) (r1 r2 : List Line) : shapedReturn phi c (s :: r1) ≤ shapedReturn phi c (s :: r2) ↔ phi (final s r2) + c * (r2.length : Int) ≤ phi (final s r1) + c * (r1.length : Int)",
    "disc_return_telescopes": "(phi : Line → Int) (g c : Int) : ∀ (s : Line) (rest : List Line), discReturn phi g c (s :: rest) = phi s - g ^ rest.length * phi (final s rest) - c * geom g rest.length",
    "disc_return_order": "(phi : Line → Int) (g c : Int) (hg : 0 < g) (s : Line) (r1 r2 : List Line) (hlen : r1.length = r2.length) : discReturn phi g c (s :: r1) ≤ discReturn phi g c (s :: r2) ↔ phi (final s r2) ≤ phi (final s r1)",
    "fold_range": "(a : Int) : -90 ≤ fold a ∧ fold a < 90",
    "pa_range": "(a : Int) : 0 ≤ pa a ∧ pa a ≤ 90",
    "pa_flip": "(a : Int) : pa (-a) = pa a",
}


def _env(rng: np.random.Generator, with_target: bool = True, cls=FascicleEnv) -> FascicleEnv:
    ch = rng.random((3, 48, 72)).astype(np.float32)
    target = np.array([10.0, 30.0, 60.0, 15.0, float(fold(np.degrees(np.arctan2(-15.0, 50.0))))])
    return cls(ch, rng.uniform(15, 55), rng.uniform(10, 40), rng.uniform(-40, 40), target if with_target else None)


def _random_policy(rng: np.random.Generator, p_stop: float = 0.05):
    def act(env: FascicleEnv) -> int:
        if rng.random() < p_stop:
            return STOP
        return int(rng.integers(0, STOP))

    return act


class _TerminalBonusEnv(FascicleEnv):
    def step(self, action):
        r, done = super().step(action)
        return (r + 1.0 if done and self.moves else r), done


class _DoubleStepEnv(FascicleEnv):
    def next_state(self, action):
        dy, dt = MOVES[action]
        return self.ky + 2 * dy, self.kt + 2 * dt


def _telescopes(tr) -> bool:
    moves = len(tr["states"]) - 1
    rhs = tr["phis"][0] - tr["phis"][-1] - STEP_COST * moves
    return len(tr["rewards"]) in (moves, moves + 1) and abs(float(np.sum(tr["rewards"])) - rhs) < 1e-9


def test_shaped_return_telescopes():
    rng = np.random.default_rng(0)
    for _ in range(300):
        assert _telescopes(rollout(_env(rng), _random_policy(rng)))


def test_shaped_return_telescopes_mutant_terminal_bonus():
    rng = np.random.default_rng(10)
    broken = 0
    for _ in range(100):
        tr = rollout(_env(rng, cls=_TerminalBonusEnv), _random_policy(rng, p_stop=0.0))
        broken += not _telescopes(tr)
    assert broken > 50


def _play(env: FascicleEnv, acts, discount: float = 1.0) -> float:
    total, g = 0.0, 1.0
    for a in acts:
        if env.done:
            break
        total += g * env.step(int(a))[0]
        g *= discount
    return total


def _twin(env: FascicleEnv, gamma: float) -> FascicleEnv:
    return FascicleEnv(env.channels, env.x0, env.y0, env.t0, env.target, gamma=gamma)


def test_discounted_return_telescopes_with_matched_shaping():
    rng = np.random.default_rng(11)
    for gamma in (0.9, 0.95, 0.5):
        for _ in range(100):
            env = _twin(_env(rng), gamma)
            phi0 = env.phi()
            ret = _play(env, rng.integers(0, STOP, size=30), discount=gamma)
            n = env.moves
            geom = (1 - gamma**n) / (1 - gamma)
            assert abs(ret - (phi0 - gamma**n * env.phi() - STEP_COST * geom)) < 1e-9
    with pytest.raises(ValueError):
        _twin(_env(rng), 0.0)


def _monotone_path(rng: np.random.Generator, n: int) -> np.ndarray:
    """One direction per axis, so the path never revisits a state."""
    y_act, t_act = int(rng.choice([0, 1])), int(rng.choice([2, 3]))
    return np.where(rng.random(n) < 0.5, y_act, t_act)


def test_return_order_same_moves_and_mismatched_discount_mutant():
    rng = np.random.default_rng(1)
    mismatched_violations = 0
    checked = 0
    for _ in range(400):
        base = _env(rng)
        acts_a, acts_b = _monotone_path(rng, 10), _monotone_path(rng, 10)
        a, b = _twin(base, 1.0), _twin(base, 1.0)
        ra, rb = _play(a, acts_a), _play(b, acts_b)
        if a.moves != b.moves:
            continue
        checked += 1
        closer_b = b.phi() <= a.phi() + 1e-12
        assert (ra <= rb + 1e-12) == closer_b
        ga, gb = _twin(base, 0.9), _twin(base, 0.9)
        assert (_play(ga, acts_a, 0.9) <= _play(gb, acts_b, 0.9) + 1e-12) == closer_b
        # Mutant: undiscounted shaping fed to a discounted learner.
        ma, mb = _twin(base, 1.0), _twin(base, 1.0)
        if (_play(ma, acts_a, 0.9) <= _play(mb, acts_b, 0.9) + 1e-12) != closer_b:
            mismatched_violations += 1
    assert checked >= 20
    assert mismatched_violations > 0


def test_undiscounted_return_ranks_any_lengths_by_objective_and_discount_witness():
    rng = np.random.default_rng(13)
    witness = False
    for _ in range(400):
        base = _env(rng)
        acts_a = _monotone_path(rng, int(rng.integers(1, 6)))
        acts_b = _monotone_path(rng, int(rng.integers(20, 40)))
        a, b = _twin(base, 1.0), _twin(base, 1.0)
        ra, rb = _play(a, acts_a), _play(b, acts_b)
        obj_a, obj_b = a.phi() + STEP_COST * a.moves, b.phi() + STEP_COST * b.moves
        assert (ra <= rb + 1e-9) == (obj_b <= obj_a + 1e-9)
        # gamma < 1: the longer episode can end farther away and still earn more.
        ga, gb = _twin(base, 0.95), _twin(base, 0.95)
        da, db = _play(ga, acts_a, 0.95), _play(gb, acts_b, 0.95)
        if gb.moves > ga.moves and gb.phi() > ga.phi() + 1.0 and db > da:
            witness = True
    assert witness


def test_rollout_bounds_random_and_greedy():
    rng = np.random.default_rng(2)
    for _ in range(200):
        tr = rollout(_env(rng), _random_policy(rng, p_stop=0.0))
        assert len(tr["states"]) <= T_MAX + 1
        assert all(abs(ky) <= T_MAX and abs(kt) <= T_MAX for ky, kt in tr["states"])
        assert len(set(tr["states"])) == len(tr["states"])
    torch.manual_seed(0)
    policy = greedy_policy(QNet(3).eval())
    for _ in range(20):
        tr = rollout(_env(rng, with_target=False), policy)
        assert len(tr["states"]) <= T_MAX + 1
        assert len(set(tr["states"])) == len(tr["states"])


def test_rollout_mutants_are_caught():
    rng = np.random.default_rng(3)
    env = _env(rng)
    env.t_max = T_MAX + 5
    tr = rollout(env, lambda e: 2)
    assert len(tr["states"]) > T_MAX + 1
    drift = 0
    for _ in range(50):
        e = _env(rng, cls=_DoubleStepEnv)
        tr = rollout(e, lambda e: 2)
        drift = max(drift, max(abs(kt) for _, kt in tr["states"]))
    assert drift > T_MAX
    with pytest.raises(ValueError):
        FascicleEnv(np.zeros((3, 8, 8), np.float32), 1.0, 1.0, 0.0, t_max=0)


def _lean_fold(a: int) -> int:
    return (a + 90) % 180 - 90


def _lean_pa(a: int) -> int:
    f = _lean_fold(a)
    return -f if f < 0 else f


def test_fold_and_pa_mirror_lean():
    for a in range(-1000, 1001):
        assert float(fold(a)) == _lean_fold(a)
        assert float(pa_of(a)) == _lean_pa(a)
        assert -90 <= _lean_fold(a) < 90
        assert 0 <= _lean_pa(a) <= 90
        assert _lean_pa(-a) == _lean_pa(a)
    rng = np.random.default_rng(4)
    a = np.concatenate([rng.uniform(-1e4, 1e4, 100000), 90.0 + 180.0 * rng.integers(-50, 50, 1000) + rng.normal(0, 1e-12, 1000)])
    f = fold(a)
    assert np.all((f >= -90) & (f < 90))
    assert np.max(np.abs(pa_of(-a) - pa_of(a))) < 1e-9
    wrong = lambda x: (np.asarray(x) + 90.0) % 180.0  # noqa: E731
    assert np.max(np.abs(np.abs(wrong(-a)) - np.abs(wrong(a)))) > 1.0


def test_fold_float_boundary():
    for x in (-90.0, 90.0, 270.0, -270.0, -90.0 - 1e-14, 90.0 - 1e-14, -1e-14, 1e-300):
        assert -90.0 <= float(fold(x)) < 90.0, x
    assert float(fold(90.0)) == -90.0
    naive = (-90.0 - 1e-14 + 90.0) % 180.0 - 90.0
    assert naive == 90.0


def test_flip_equivariance_and_mutant():
    torch.manual_seed(0)
    model = SpatialAttentionGeoNet(AttnGeoConfig(width=8, depth=2)).eval()
    x = torch.rand(2, 3, 32, 48)
    a = predict_symmetric(model, x)
    b = predict_symmetric(model, x.flip(-1))
    mapped = flip_output(b)
    for k in ("attn", "sup", "deep"):
        assert torch.max(torch.abs(mapped[k] - a[k])) < 1e-5
    assert torch.max(torch.abs(mapped["orient"] - a["orient"])) < 1e-5
    o = b["orient"].flip(-1)
    no_negate = torch.stack([o[:, 0], o[:, 1]], 1)
    assert torch.max(torch.abs(no_negate - a["orient"])) > 1e-3


def test_readout_flip_invariance_end_to_end():
    rng = np.random.default_rng(12)
    checked = 0
    for _ in range(20):
        s = make_sample(rng, p_central=0.0)
        mm, w = s.mm_per_px, s.image.shape[1]
        lines = lines_from_attention(s.fasc_mask.astype(np.float32), None, min_len_px=4.0 / mm)
        out = readout(s.sup_mask.astype(np.float32), s.deep_mask.astype(np.float32), lines, mm, mm)
        flipped_lines = [PixelLine(w - 1 - ln.x, ln.y, float(fold(-ln.t_deg)), ln.weight) for ln in lines]
        out_f = readout(s.sup_mask[:, ::-1].astype(np.float32), s.deep_mask[:, ::-1].astype(np.float32), flipped_lines, mm, mm)
        if not (out["ok"] and out_f["ok"]):
            continue
        checked += 1
        for k in ("pa_deg", "fl_mm", "mt_mm"):
            assert abs(out[k] - out_f[k]) < 1e-6, k
        unflipped = [PixelLine(w - 1 - ln.x, ln.y, ln.t_deg, ln.weight) for ln in lines]
        out_bad = readout(s.sup_mask[:, ::-1].astype(np.float32), s.deep_mask[:, ::-1].astype(np.float32), unflipped, mm, mm)
        if out_bad["ok"]:
            assert abs(out_bad["pa_deg"] - out["pa_deg"]) > 1.0 or abs(out_bad["fl_mm"] - out["fl_mm"]) > 1.0
    assert checked >= 15


def _two_band_masks(h=96, w=144, sup=(0.0, 12.0), deep=(0.18, 58.0), th=3.0):
    xs = np.arange(w, dtype=float)
    yy = np.arange(h, dtype=float)[:, None]
    sup_poly, deep_poly = np.array(sup), np.array(deep)
    sup_m = (np.abs(yy - np.polyval(sup_poly, xs)[None]) <= th / 2).astype(np.uint8)
    deep_m = (np.abs(yy - np.polyval(deep_poly, xs)[None]) <= th / 2).astype(np.uint8)
    return sup_m, deep_m, sup_poly, deep_poly


def test_readout_fl_is_extrapolated_line_not_sine():
    mm = 0.25
    sup_m, deep_m, sup_poly, deep_poly = _two_band_masks()
    cx, cy, ang = 70.0, 40.0, -30.0
    out = readout(sup_m.astype(np.float32), deep_m.astype(np.float32), [PixelLine(cx, cy, ang, 1.0)], mm, mm)
    p1 = line_curve_intersection(cx * mm, cy * mm, ang, sup_poly * np.array([1.0, mm]))
    p2 = line_curve_intersection(cx * mm, cy * mm, ang, deep_poly * np.array([1.0, mm]))
    fl_line = float(np.linalg.norm(p1 - p2))
    fl_sine = out["mt_mm"] / np.sin(np.radians(out["pa_deg"]))
    assert out["ok"]
    assert abs(out["fl_mm"] - fl_line) < 0.6
    assert abs(fl_sine - fl_line) > 3.0 * abs(out["fl_mm"] - fl_line) + 1.0


def test_readout_equals_host_measure_on_shared_inputs():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(30):
        s = make_sample(rng, p_central=0.0)
        mm = s.mm_per_px
        apo = np.maximum(s.sup_mask, s.deep_mask).astype(np.float32)
        fasc = s.fasc_mask.astype(np.float32)
        ref = measure(apo, fasc, mm, mm)
        lines = lines_from_attention(fasc, None, thr=0.5, min_len_px=4.0 / mm)
        out = readout(s.sup_mask.astype(np.float32), s.deep_mask.astype(np.float32), lines, mm, mm)
        if not (ref["ok"] and out["ok"]):
            continue
        checked += 1
        assert abs(out["pa_deg"] - ref["pa_deg"]) < 1e-6
        assert abs(out["fl_mm"] - ref["fl_mm"]) < 1e-6
        assert abs(out["mt_mm"] - ref["mt_mm"]) < 1e-6
    assert checked >= 20


def test_multexp_holdout_and_mutant():
    rng = np.random.default_rng(8)
    mex = [rng.integers(0, 2, 256).astype(bool) for _ in range(5)]
    train = [rng.integers(0, 2, 256).astype(bool) for _ in range(50)]
    assert_multexp_excluded(train, mex)
    leak = mex[2].copy()
    leak[:5] = ~leak[:5]
    with pytest.raises(HoldoutLeak):
        assert_multexp_excluded([*train, leak], mex)


def test_action_space_matches_lean():
    src = (LEAN_DIR / "UmudRl/Basic.lean").read_text()
    lean_actions = re.search(r"inductive Act where\s*\n\s*\|(.+)\n", src).group(1)
    assert [a.strip() for a in lean_actions.split("|")] == ["up", "down", "rotP", "rotN", "stop"]
    assert ACTIONS == ("up", "down", "rot+", "rot-", "stop")
    assert [MOVES[i] for i in range(5)] == [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]


def _statement(src: str, name: str) -> str:
    m = re.search(rf"theorem {name}\b(.*?)(?::= by|\n  \|)", src, re.S)
    assert m, name
    return " ".join(m.group(1).split())


def test_lean_statements_pinned_and_no_escape_hatch():
    src = (LEAN_DIR / "UmudRl/Basic.lean").read_text()
    for name, text in PINNED.items():
        assert _statement(src, name) == text, name
    assert not re.search(r"\b(sorry|admit|native_decide)\b", src)
    assert "axiom " not in src


def test_fv_csv_gates_fail_on_mutants():
    import pandas as pd

    spec = importlib.util.spec_from_file_location("fv_attention_rl_stages", ROOT / "scripts/chase018/fv_attention_rl_stages.py")
    fv = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fv
    spec.loader.exec_module(fv)
    verifier = fv._verifier()
    rng = np.random.default_rng(9)
    ids = [f"IMG_{i:05d}.tif" for i in range(309)]
    rl = pd.DataFrame({"pa_deg": rng.uniform(10, 25, 309), "fl_mm": rng.uniform(40, 90, 309), "mt_mm": rng.uniform(12, 25, 309)}, index=ids)
    moved = rl.copy()
    moved["pa_deg"] += rng.choice([-1, 1], 309) * 3.0
    moved["fl_mm"] += rng.choice([-1, 1], 309) * 6.0
    good = fv.csv_gates(moved, rl, ids, 0.3, verifier)
    assert {k: v[0] for k, v in good.items()} == {"official_ids": "PASS", "movement_budget": "PASS", "no_joint_length_shrink": "PASS"}
    assert fv.csv_gates(rl.copy(), rl, ids, 0.3, verifier)["movement_budget"][0] == "FAIL"
    shrunk = moved.copy()
    shrunk["fl_mm"] = rl.fl_mm * 0.97
    shrunk["mt_mm"] = rl.mt_mm * 0.97
    assert fv.csv_gates(shrunk, rl, ids, 0.3, verifier)["no_joint_length_shrink"][0] == "FAIL"
    renamed = moved.rename(index={ids[0]: "IMG_00000.png"})
    assert fv.csv_gates(renamed, rl, ids, 0.3, verifier)["official_ids"][0] == "FAIL"


def _lake() -> str | None:
    found = shutil.which("lake")
    if found:
        return found
    fallback = Path.home() / ".elan/bin/lake"
    return str(fallback) if fallback.exists() else None


@pytest.mark.skipif(_lake() is None, reason="Lean toolchain not installed")
def test_lean_build_and_axioms():
    lake = _lake()
    subprocess.run([lake, "build"], cwd=LEAN_DIR, check=True, capture_output=True)
    out = subprocess.run([lake, "env", "lean", "Audit.lean"], cwd=LEAN_DIR, check=True, capture_output=True, text=True)
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    lines = [ln for ln in out.stdout.splitlines() if "depends on axioms" in ln]
    assert len(lines) == len(PINNED)
    for ln in lines:
        used = set(re.search(r"\[(.*)\]", ln).group(1).replace(" ", "").split(","))
        assert used <= allowed, ln
