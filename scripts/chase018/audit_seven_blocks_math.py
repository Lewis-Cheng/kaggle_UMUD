#!/usr/bin/env python3
"""Check the seven fault blocks against the coupled PA/FL diagnosis.

Does not fit scales, does not train, and does not read private images.
A block is solved only when this tree both implements the check and has
run it on real held-out images. None of the seven meet that bar.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.geometry.physical import physical_angle_to_horizontal_deg  # noqa: E402
from umud.metric.score import score_components  # noqa: E402
from umud.verification.coupling import (  # noqa: E402
    explained_length_residual_mm,
    first_order_length_change_mm,
    normalized_mean_score,
)

OUT = ROOT / "reports/phase3/chase018/seven_blocks_math_check.json"
HOST = ROOT / "src/umud/geometry/host_faithful.py"
AFFINE = ROOT / "scripts/chase018/build_multexp_chase_rows.py"

# Supplied aggregate MAEs from the diagnosis note. Not remeasured here.
SUPPLIED_MAE = {"pa_deg": 1.127, "fl_mm": 2.489, "mt_mm": 0.162}


def _score_check() -> dict[str, float | bool]:
    hand = normalized_mean_score(**{
        "mae_pa_deg": SUPPLIED_MAE["pa_deg"],
        "mae_fl_mm": SUPPLIED_MAE["fl_mm"],
        "mae_mt_mm": SUPPLIED_MAE["mt_mm"],
    })
    official = score_components(
        {key: [value] for key, value in SUPPLIED_MAE.items()}
    )["S1"]
    parts = {
        "pa": SUPPLIED_MAE["pa_deg"] / 6.0,
        "fl": SUPPLIED_MAE["fl_mm"] / 12.0,
        "mt": SUPPLIED_MAE["mt_mm"] / 3.0,
    }
    total = parts["pa"] + parts["fl"] + parts["mt"]
    return {
        "hand_mean": hand,
        "score_components_S1": official,
        "matches_official_mean": abs(hand - official) < 1e-12,
        "sum_not_the_score": total,
        "share_pa": parts["pa"] / total,
        "share_fl": parts["fl"] / total,
        "share_mt": parts["mt"] / total,
    }


def _aspect_fixture() -> dict[str, float]:
    """Pixel step (1, 1). sx, sy here are mm/px. Forcing them equal changes PA."""
    anisotropic = physical_angle_to_horizontal_deg((1.0, 1.0), sx=0.1, sy=0.2)
    forced = physical_angle_to_horizontal_deg((1.0, 1.0), sx=0.1, sy=0.1)
    return {
        "angle_deg_sx_0.1_sy_0.2": anisotropic,
        "angle_deg_forced_sy_eq_sx": forced,
        "abs_delta_deg": abs(anisotropic - forced),
    }


def _isotropic_calls() -> list[str]:
    hits: list[str] = []
    for path in (ROOT / "scripts").rglob("*.py"):
        if path.name == "audit_seven_blocks_math.py":
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            compact = line.replace(" ", "")
            if "mm_per_prob_pixel(px_per_mm,px_per_mm" in compact:
                hits.append(f"{path.relative_to(ROOT)}:{lineno}")
    return hits


def _separate_scale_calls() -> list[str]:
    hits: list[str] = []
    for path in (ROOT / "scripts").rglob("*.py"):
        if path.name == "audit_seven_blocks_math.py":
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "mm_per_prob_pixel(sx,sy" in line.replace(" ", ""):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}")
    return hits


def _measure_uses() -> dict[str, bool]:
    src = HOST.read_text(encoding="utf-8")
    tree = ast.parse(src)
    measure_src = ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "measure":
            measure_src = ast.get_source_segment(src, node) or ""
    return {
        "pa_uses_centreline_slope_deg": "slope_deg(deep[0]" in measure_src,
        "mt_uses_centre_minus_half_thickness": "centre_minus_half" in measure_src or "(th(sup_r) + th(deep_r)) / 2" in measure_src,
        "pa_uses_rich_inner_edge_tangent": 'fit"]["top"]' in measure_src or 'fit"]["bot"]' in measure_src,
        "rejects_near_parallel_intersection": "IntersectionRejected" in measure_src,
    }


def _affine_is_least_squares() -> bool:
    text = AFFINE.read_text(encoding="utf-8")
    return "np.polyfit(pred[m], gt[m], 1)" in text


def main() -> int:
    score = _score_check()
    if not score["matches_official_mean"]:
        raise SystemExit("score.py mean does not match the hand mean")
    aspect = _aspect_fixture()
    if aspect["abs_delta_deg"] < 10.0:
        raise SystemExit("isotropic spacing did not move the fixture angle")
    d_l = first_order_length_change_mm(50.0, 15.0, 1.0, 50.0 * math.sin(math.radians(15.0)), 0.0)
    if abs(abs(d_l) - 3.26) > 0.02:
        raise SystemExit(f"expected about 3.26 mm FL change, got {d_l}")
    explained = explained_length_residual_mm(1.0, 0.0, 15.0, 50.0)
    isotropic = _isotropic_calls()
    separate = _separate_scale_calls()
    measure_flags = _measure_uses()
    blocks = [
        {
            "block": "scale_anchor",
            "solved": False,
            "verdict": "open",
            "why": "Per-image scale rows are not in this clone. family_prior is still a usable label. sx, sy were not fit to expert outputs.",
        },
        {
            "block": "coordinate_pullback",
            "solved": False,
            "verdict": "partial_synthetic_only",
            "why": (
                "host_faithful.measure agreed on one synthetic INTER_AREA resize when sx and sy differed. "
                f"{len(isotropic)} MultExp call sites still pass the same px_per_mm for both axes. "
                f"{len(separate)} infer call sites pass sx, sy separately. "
                "Forcing sy = sx on a (1, 1) pixel step moves the physical angle by "
                f"{aspect['abs_delta_deg']:.2f} degrees. That call-site split is not closed."
            ),
            "isotropic_calls": isotropic,
            "separate_calls": separate,
            "aspect_fixture": aspect,
        },
        {
            "block": "apo_geometry",
            "solved": False,
            "verdict": "open",
            "why": (
                "MT uses centreline separation minus half thickness. "
                "PA uses slope_deg of the apo_curves centreline polynomial, not the rich inner-edge tangent. "
                "No GT-apo comparison was run; images are not in this clone."
            ),
            "measure": measure_flags,
        },
        {
            "block": "fasc_orientation",
            "solved": False,
            "verdict": "open",
            "why": "The production aggregator is a weighted median of SVD segments. Vesselness, Hough, and local projection/LOESS were not compared on held-out images.",
        },
        {
            "block": "measurement_definition",
            "solved": False,
            "verdict": "open",
            "why": (
                "The coupling identity is implemented as a diagnostic only. "
                f"At L=50 mm and 15 degrees, a +1 degree pennation error predicts {d_l:.4f} mm of FL change with thickness fixed. "
                "No per-image signed residuals exist here, so the identity has not explained the FL error. "
                "line_curve_intersection does not reject near-parallel crossings."
            ),
            "one_degree_fl_change_mm": d_l,
            "explained_matches_first_order": abs(explained - d_l) < 1e-12,
            "measure": measure_flags,
        },
        {
            "block": "calibration_transfer",
            "solved": False,
            "verdict": "open",
            "why": (
                "The frozen affine is ordinary least squares (np.polyfit), not a regularized absolute-deviation fit. "
                "The affine JSON marks FL as applied while infer_rl_refiner leaves FL raw. "
                "No new coefficients were fit."
            ),
            "fit_is_least_squares": _affine_is_least_squares(),
        },
        {
            "block": "deployment_parity",
            "solved": False,
            "verdict": "open",
            "why": "infer_rl_refiner still rewrites non-.tif ids, and the submitted CSV is absent. Eval and deploy are not one function.",
        },
    ]
    payload = {
        "production_readiness": "BLOCKED",
        "private_data_visible": False,
        "private_weights_visible": False,
        "any_of_seven_solved": False,
        "score_is_mean": score,
        "supplied_mae_not_remeasured": SUPPLIED_MAE,
        "three_d_branch": "not_opened",
        "blocks": blocks,
    }
    if any(row["solved"] for row in blocks):
        raise SystemExit("refusing to mark a block solved")
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({row["block"]: row["verdict"] for row in blocks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
