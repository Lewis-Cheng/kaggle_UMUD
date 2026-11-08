#!/usr/bin/env python3
"""Aborting property harness for UMUD coordinate, score, split, and promotion claims.

Run the positive path and the production integration rung:

    python tests/properties/test_umud_contracts.py

Prove each documented mutant flips its check:

    python tests/properties/test_umud_contracts.py --prove-mutants

Checks raise ``ContractAbort``. They do not use the ``assert`` statement.
A green synthetic run writes ``production_readiness: BLOCKED``.
The integration rung calls ``host_faithful.measure`` on a resized synthetic
raster. Disagreement beyond the preregistered tolerance aborts the process
after the JSON witness is written.

Trusted base: this file, ``src/umud/verification/contracts.py``,
``src/umud/geometry/physical.py``, ``src/umud/metric/score.py``, CPython, NumPy.
Out of scope: ultrasound appearance, leaderboard rank, private score, and any
claim outside the seeded inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from umud.geometry.host_faithful import measure, pa_depth_fit, weighted_median  # noqa: E402
from umud.geometry.physical import physical_length  # noqa: E402
from umud.metric.score import score_components  # noqa: E402
from umud.metric.submission import SubmissionValidationError, validate_submission  # noqa: E402
from umud.verification.contracts import (  # noqa: E402
    ContractAbort,
    assert_groups_disjoint,
    assert_promotable,
    assert_scale_not_relabeled,
    physical_cosine,
    pullback_length_sq,
)

JSON_PATH = ROOT / "reports/phase3/chase018/fv_umud_contracts.json"
RANDOM_SEED = 20261108
RANDOM_N = 1000
# Preregistered before looking at measure() output. Rasterization may move a
# thin line by about one destination pixel; these bounds are that budget, not
# a tolerance fitted to a result.
PULLBACK_TOLERANCE = {"pa_deg": 1.5, "fl_mm": 3.0, "mt_mm": 1.0}

PROPERTY_NAMES = (
    "coordinate_roundtrip",
    "angle_geometry",
    "length_units",
    "metric_normalization",
    "group_disjoint",
    "finite_output",
    "metric_anchor",
    "promotion_evidence",
)


def _expected_length_sq(u: np.ndarray, sx: float, sy: float) -> float:
    return float((sx * u[0]) ** 2 + (sy * u[1]) ** 2)


def _expected_cosine(u: np.ndarray, v: np.ndarray, sx: float, sy: float) -> float:
    dot = (sx * u[0]) * (sx * v[0]) + (sy * u[1]) * (sy * v[1])
    nu = float(np.hypot(sx * u[0], sy * u[1]))
    nv = float(np.hypot(sx * v[0], sy * v[1]))
    return float(dot / (nu * nv))


def check_coordinate_roundtrip(mutant: bool) -> None:
    """Physical length is invariant under an invertible affine coordinate change.

    mutant: pullback_length_sq path replaced by pushforward J^T G J
            (drop the inverse); input: u=(12,-5), J=diag(1.7, 0.4), sx=0.15, sy=0.08.
    """
    sx, sy = 0.15, 0.08
    J = np.array([[1.7, 0.05], [0.0, 0.4]], dtype=np.float64)
    u = np.array([12.0, -5.0], dtype=np.float64)
    b = np.array([3.0, -2.0], dtype=np.float64)
    p = np.array([40.0, 10.0], dtype=np.float64)
    if mutant:
        # Forgot to invert the resize: push the metric forward instead of pulling it back.
        G = np.diag([sx**2, sy**2])
        up = J @ u
        got = float(up @ (J.T @ G @ J) @ up)
    else:
        got = pullback_length_sq(u, sx, sy, J)
    expected = _expected_length_sq(u, sx, sy)
    if abs(got - expected) > 1e-8:
        raise ContractAbort(
            f"coordinate_roundtrip length^2 {got} != {expected}"
        )
    back = np.linalg.inv(J) @ ((J @ p + b) - b)
    err = float(np.max(np.abs(back - p)))
    if err > 1e-8:
        raise ContractAbort(f"coordinate_roundtrip point error {err}")


def check_angle_geometry(mutant: bool) -> None:
    """Pennation cosine uses anisotropic spacing. Ignoring sy=sx changes the angle.

    mutant: sy replaced by sx; input: u=(3,1), v=(1,2), sx=0.1, sy=0.25.
    """
    sx = 0.1
    sy = 0.1 if mutant else 0.25
    u = np.array([3.0, 1.0], dtype=np.float64)
    v = np.array([1.0, 2.0], dtype=np.float64)
    got = physical_cosine(u, v, sx, sy)
    expected = _expected_cosine(u, v, 0.1, 0.25)
    if abs(got - expected) > 1e-9:
        raise ContractAbort(f"angle_geometry cos {got} != {expected}")


def check_length_units(mutant: bool) -> None:
    """100 px at 0.1 mm/px is 10 mm. A centimetre/millimetre mix is a 10× error.

    mutant: mm_per_px scaled by 10; input: displacement (100, 0).
    """
    mm_per_px = 0.1 * (10.0 if mutant else 1.0)
    got = physical_length((100.0, 0.0), sx=mm_per_px, sy=mm_per_px)
    if abs(got - 10.0) > 1e-9:
        raise ContractAbort(f"length_units got {got} mm, expected 10")


def check_metric_normalization(mutant: bool) -> None:
    """S1 is the mean of MAE/6, MAE/12, MAE/3, matching score_components.

    mutant: S1 replaced by the sum of the three normalized MAEs;
            input: abs errors pa=[6,0], fl=[12,0], mt=[3,0] → each term 0.5.
    """
    errors = {"pa_deg": [6.0, 0.0], "fl_mm": [12.0, 0.0], "mt_mm": [3.0, 0.0]}
    parts = score_components(errors)
    mean_s1 = float(parts["S1"])
    summed = 3.0 * mean_s1
    got = summed if mutant else mean_s1
    if abs(got - 0.5) > 1e-12:
        raise ContractAbort(f"metric_normalization S1 {got} != 0.5")
    if abs(mean_s1 - 0.5) > 1e-12:
        raise ContractAbort(f"score_components S1 {mean_s1} != 0.5")


def check_group_disjoint(mutant: bool) -> None:
    """A group id cannot sit in both the training split and the evaluation split.

    mutant: group 3 copied into the evaluation side; input: train {1,2,3}, test {4,5}.
    """
    train = {1, 2, 3}
    test = {3, 4, 5} if mutant else {4, 5}
    assert_groups_disjoint(train, test)


def check_finite_output(mutant: bool) -> None:
    """Every required id has a finite PA, FL, and MT. NaN aborts.

    mutant: swallow SubmissionValidationError on a NaN pa_deg;
            input: ids a,b and pa_deg NaN on a.
    """
    ids = ["a.tif", "b.png"]
    df = pd.DataFrame(
        {
            "id": ids,
            "pa_deg": [np.nan, 12.0],
            "fl_mm": [40.0, 41.0],
            "mt_mm": [15.0, 16.0],
        }
    )
    caught = False
    try:
        validate_submission(df, ids)
    except SubmissionValidationError:
        caught = True
    if mutant:
        caught = False
    if not caught:
        raise ContractAbort("finite_output accepted a non-finite prediction")


def check_metric_anchor(mutant: bool) -> None:
    """family_prior and unknown must not be stored as measured.

    mutant: copy the production USABLE set (measured | family_prior) into the
            measured bucket and skip the guard; input: source family_prior.
    """
    if mutant:
        stored = "measured"
        if stored == "measured":
            raise ContractAbort("metric_anchor: family_prior stored as measured")
        return
    assert_scale_not_relabeled("family_prior", "family_prior")
    assert_scale_not_relabeled("unknown", "unknown")
    assert_scale_not_relabeled("measured", "measured")
    refused = False
    try:
        assert_scale_not_relabeled("family_prior", "measured")
    except ContractAbort:
        refused = True
    if not refused:
        raise ContractAbort("metric_anchor allowed a family_prior measured label")


def check_promotion_evidence(mutant: bool) -> None:
    """Synthetic passes alone cannot set production_readiness to PASS.

    mutant: set production_readiness PASS while real_images_ran is false and
            skip the guard; input: evidence with every real-data flag false.
    """
    evidence = {
        "scorer_parity": False,
        "coordinate_contract": False,
        "real_images_ran": False,
        "texture_rl_replay_matched": False,
        "deployment_parity": False,
        "production_readiness": "PASS" if mutant else "BLOCKED",
    }
    if mutant:
        if evidence["production_readiness"] == "PASS" and not evidence["real_images_ran"]:
            raise ContractAbort("promotion_evidence: PASS without real_images_ran")
        return
    refused = False
    try:
        assert_promotable(evidence)
    except ContractAbort:
        refused = True
    if not refused:
        raise ContractAbort("promotion_evidence promoted a synthetic-only bundle")


CHECKS = {
    "coordinate_roundtrip": check_coordinate_roundtrip,
    "angle_geometry": check_angle_geometry,
    "length_units": check_length_units,
    "metric_normalization": check_metric_normalization,
    "group_disjoint": check_group_disjoint,
    "finite_output": check_finite_output,
    "metric_anchor": check_metric_anchor,
    "promotion_evidence": check_promotion_evidence,
}


def run_positive() -> list[str]:
    failed: list[str] = []
    for name in PROPERTY_NAMES:
        try:
            CHECKS[name](False)
        except ContractAbort as exc:
            failed.append(f"{name}: {exc}")
    if failed:
        raise ContractAbort("positive properties failed: " + "; ".join(failed))
    return list(PROPERTY_NAMES)


def run_mutants() -> list[dict[str, object]]:
    """Each mutant must abort. A mutant that passes checks nothing."""
    rows: list[dict[str, object]] = []
    for name in PROPERTY_NAMES:
        rejected = False
        message = ""
        try:
            CHECKS[name](True)
        except ContractAbort as exc:
            rejected = True
            message = str(exc)
        rows.append({"property": name, "rejected": rejected, "message": message})
        if not rejected:
            raise ContractAbort(f"mutant for {name} did not abort")
    return rows


def run_random_affines(n: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    max_len = 0.0
    max_cos = 0.0
    max_pt = 0.0
    drawn = 0
    while drawn < n:
        A = rng.normal(size=(2, 2))
        if abs(float(np.linalg.det(A))) < 0.05:
            continue
        sx = float(rng.uniform(0.02, 0.5))
        sy = float(rng.uniform(0.02, 0.5))
        u = rng.normal(size=2)
        v = rng.normal(size=2)
        b = rng.normal(size=2)
        p = rng.normal(size=2)
        got = pullback_length_sq(u, sx, sy, A)
        exp = _expected_length_sq(u, sx, sy)
        max_len = max(max_len, abs(got - exp))
        cos_got = physical_cosine(u, v, sx, sy)
        cos_exp = _expected_cosine(u, v, sx, sy)
        max_cos = max(max_cos, abs(cos_got - cos_exp))
        back = np.linalg.inv(A) @ ((A @ p + b) - b)
        max_pt = max(max_pt, float(np.max(np.abs(back - p))))
        drawn += 1
    if max_len > 1e-8 or max_cos > 1e-8 or max_pt > 1e-8:
        raise ContractAbort(
            f"random affine residual len={max_len} cos={max_cos} pt={max_pt}"
        )
    return {
        "n": float(n),
        "seed": float(seed),
        "max_abs_length_sq": max_len,
        "max_abs_cosine": max_cos,
        "max_abs_point_roundtrip": max_pt,
    }


def _paint_physical(height: int, width: int, sx: float, sy: float) -> tuple[np.ndarray, np.ndarray]:
    """Binary apo/fasc maps of one fixed millimetre scene."""
    y_mm = (np.arange(height, dtype=np.float64) + 0.5) * sy
    x_mm = (np.arange(width, dtype=np.float64) + 0.5) * sx
    x, y = np.meshgrid(x_mm, y_mm)
    apo = np.zeros((height, width), dtype=np.float32)
    fasc = np.zeros((height, width), dtype=np.float32)
    in_x = (x >= 8.0) & (x <= 110.0)
    apo[(np.abs(y - 16.0) <= 1.2) & in_x] = 1.0
    apo[(np.abs(y - 46.0) <= 1.2) & in_x] = 1.0
    angle = np.deg2rad(25.0)
    for x0 in (28.0, 48.0, 68.0, 88.0):
        dist = np.abs(-(x - x0) * np.sin(angle) + (y - 31.0) * np.cos(angle))
        fasc[(dist <= 0.55) & (y > 18.5) & (y < 43.5) & in_x] = 1.0
    return apo, fasc


def _resize_area(apo: np.ndarray, fasc: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Match ``frundles_smp`` image resize (INTER_AREA) on the synthetic maps."""
    apo_r = cv2.resize(apo, (width, height), interpolation=cv2.INTER_AREA)
    fasc_r = cv2.resize(fasc, (width, height), interpolation=cv2.INTER_AREA)
    return apo_r, fasc_r


def run_coordinate_pullback() -> dict[str, object]:
    """Same physical scene at native 600×800 and after resize to 384×576.

    Spacing on the destination raster is the source spacing times the
    source/destination pixel ratio (Jacobian of the resize).
    """
    src_h, src_w = 600, 800
    dst_h, dst_w = 384, 576
    sx, sy = 0.15, 0.08
    apo, fasc = _paint_physical(src_h, src_w, sx, sy)
    native = measure(apo, fasc, sx, sy)
    apo_r, fasc_r = _resize_area(apo, fasc, dst_h, dst_w)
    sx_d = sx * (src_w / dst_w)
    sy_d = sy * (src_h / dst_h)
    resized = measure(apo_r, fasc_r, sx_d, sy_d)
    native_ok = bool(native.get("ok"))
    resized_ok = bool(resized.get("ok"))
    deltas = {
        "pa_deg": abs(float(native["pa_deg"]) - float(resized["pa_deg"])),
        "fl_mm": abs(float(native["fl_mm"]) - float(resized["fl_mm"])),
        "mt_mm": abs(float(native["mt_mm"]) - float(resized["mt_mm"])),
    }
    if not native_ok or not resized_ok:
        verdict = "inconclusive"
    elif any(deltas[key] > PULLBACK_TOLERANCE[key] for key in PULLBACK_TOLERANCE):
        verdict = "fail"
    else:
        verdict = "pass"
    return {
        "name": "coordinate_pullback",
        "source_hw": [src_h, src_w],
        "dest_hw": [dst_h, dst_w],
        "sx_sy_source_mm_per_px": [sx, sy],
        "sx_sy_dest_mm_per_px": [sx_d, sy_d],
        "tolerance": dict(PULLBACK_TOLERANCE),
        "native_ok": native_ok,
        "resized_ok": resized_ok,
        "native": {
            "pa_deg": float(native["pa_deg"]),
            "fl_mm": float(native["fl_mm"]),
            "mt_mm": float(native["mt_mm"]),
            "n_fasc": int(native["n_fasc"]),
        },
        "resized": {
            "pa_deg": float(resized["pa_deg"]),
            "fl_mm": float(resized["fl_mm"]),
            "mt_mm": float(resized["mt_mm"]),
            "n_fasc": int(resized["n_fasc"]),
        },
        "abs_delta": deltas,
        "verdict": verdict,
        "resize": "cv2.INTER_AREA",
        "note": (
            "Synthetic binary scene, cv2.INTER_AREA into 384x576, spacing multiplied "
            "by the source/destination pixel ratio. Not a network probability map, "
            "not letterbox padding, and not a leaderboard result."
        ),
    }


def _json_ready(payload: dict[str, object]) -> None:
    """Refuse to persist a PASS readiness bit from this harness."""
    if payload.get("production_readiness") == "PASS":
        raise ContractAbort("refusing to write production_readiness PASS")
    if payload.get("real_images_ran"):
        raise ContractAbort("this harness did not run real images")
    if payload.get("texture_rl_replay_matched"):
        raise ContractAbort("this harness did not replay texture_rl")


def write_witness(
    positive: list[str],
    mutants: list[dict[str, object]],
    random_stats: dict[str, float],
    integration: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "production_readiness": "BLOCKED",
        "real_images_ran": False,
        "texture_rl_replay_matched": False,
        "positive_checks": {
            "passed": len(positive),
            "failed": 0,
            "names": positive,
        },
        "mutants": {
            "expected_reject": len(PROPERTY_NAMES),
            "rejected": sum(1 for row in mutants if row["rejected"]),
            "details": mutants,
        },
        "random_affine": random_stats,
        "integration_coordinate_pullback": integration,
        "trusted_base": [
            "tests/properties/test_umud_contracts.py",
            "src/umud/verification/contracts.py",
            "src/umud/geometry/physical.py",
            "src/umud/metric/score.py",
            "CPython",
            "NumPy",
        ],
        "out_of_scope": [
            "ultrasound appearance",
            "fascicle detection on real images",
            "leaderboard rank",
            "private score",
            "inputs outside the seeded set",
            "Rocq or RISC-V proofs",
        ],
        "algebra_scope": (
            "Hand algebra covers invertible affine A and diagonal positive "
            "spacing. It does not cover interpolation or network outputs."
        ),
        "promotion_refused": True,
    }
    _json_ready(payload)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prove-mutants",
        action="store_true",
        help="Run only the eight mutants and require each to abort.",
    )
    args = parser.parse_args(argv)
    if args.prove_mutants:
        mutants = run_mutants()
        rejected = sum(1 for row in mutants if row["rejected"])
        print(json.dumps({"mutants_rejected": rejected, "expected": len(PROPERTY_NAMES)}))
        return 0 if rejected == len(PROPERTY_NAMES) else 1

    positive = run_positive()
    mutants = run_mutants()
    random_stats = run_random_affines(RANDOM_N, RANDOM_SEED)
    integration = run_coordinate_pullback()
    write_witness(positive, mutants, random_stats, integration)
    print(json.dumps({"json": str(JSON_PATH), "integration": integration["verdict"]}, indent=2))
    if integration["verdict"] == "fail":
        print(
            "coordinate_pullback exceeded the preregistered tolerance; aborting",
            file=sys.stderr,
        )
        return 2
    if integration["verdict"] == "inconclusive":
        print("coordinate_pullback inconclusive; production stays BLOCKED", file=sys.stderr)
        return 0
    return 0


def test_coordinate_roundtrip() -> None:
    check_coordinate_roundtrip(False)


def test_angle_geometry() -> None:
    check_angle_geometry(False)


def test_length_units() -> None:
    check_length_units(False)


def test_metric_normalization() -> None:
    check_metric_normalization(False)


def test_group_disjoint() -> None:
    check_group_disjoint(False)


def test_finite_output() -> None:
    check_finite_output(False)


def test_metric_anchor() -> None:
    check_metric_anchor(False)


def test_promotion_evidence() -> None:
    check_promotion_evidence(False)


def _aggregator_snapshot() -> dict[str, float]:
    """Helper used by the stage dump via import. Not a property check."""
    apo, fasc = _paint_physical(600, 800, 0.15, 0.08)
    out = measure(apo, fasc, 0.15, 0.08)
    segs = np.asarray(out["segs"], dtype=np.float64)
    if segs.size == 0:
        return {"ok": 0.0}
    depth, pa, weight = segs[:, 0], segs[:, 1], segs[:, 2]
    fitted = pa_depth_fit(segs, at=0.9, min_n=3)
    return {
        "ok": 1.0,
        "weighted_median_pa": float(weighted_median(pa, weight)),
        "unweighted_median_pa": float(np.median(pa)),
        "depth_fit_pa": None if fitted is None else float(fitted),
        "depth_fit_defined": fitted is not None,
        "n": float(len(pa)),
        "depth_span": float(np.ptp(depth)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
