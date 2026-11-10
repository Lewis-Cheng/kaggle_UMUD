#!/usr/bin/env python3
"""Output-contract formal checks for the UMUD submission stage chain.

Adapted from the tri_attack formal-verification skill already used by
reports/phase3/fasc_fv_protocol.md. Every property aborts with its stage id.
A soft assert is not a property.

Trusted base: this Python checker, the checked-in CSVs, the official scorer
formula in src/umud/metric/score.py, and the public numbers 0.31903 (texture_pa),
0.29895 (texture_rl), and 0.35218 (FL/MT x0.97). A property that passes here is
not a public-score claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from umud.metric.score import score_components
from umud.metric.submission import validate_submission

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_IDS = ROOT / "reports/phase3/official_test_file_names.txt"
TEXTURE_PA = ROOT / "reports/phase3/chase018/submission_texture_pa.csv"
TEXTURE_RL = ROOT / "reports/phase3/chase018/submission_texture_rl.csv"
SCALE_PROBE = ROOT / "reports/phase3/chase018/submission_scale_probe_flmt_0p97.csv"
AFFINE = ROOT / "reports/phase3/chase018/multexp_chase_affine.json"

# Public score is dominated by MAE. The 1e-6/1e-9 terms cannot move 0.05.
PUBLIC_017_DELTA = 0.29895 - 0.17
MEAN_NORM_GAIN = 3.0 * PUBLIC_017_DELTA  # summed normalized MAE the 0.17 target needs
FL_MAE_GAIN_017 = 12.0 * MEAN_NORM_GAIN
PA_MAE_GAIN_017 = 6.0 * MEAN_NORM_GAIN
MT_MAE_GAIN_017 = 3.0 * MEAN_NORM_GAIN

STAGE_RESULTS: list[dict[str, str]] = []


class StageAbort(RuntimeError):
    def __init__(self, stage: str, why: str):
        super().__init__(f"{stage}: {why}")
        self.stage = stage
        self.why = why


def check(stage: str, condition: bool, why: str) -> None:
    # Each check is a property of the output stage chain. If it cannot fail,
    # it proves nothing. mutants below exist to make these fail.
    if not condition:
        STAGE_RESULTS.append({"stage": stage, "result": "FAIL", "why": why})
        raise StageAbort(stage, why)
    STAGE_RESULTS.append({"stage": stage, "result": "PASS", "why": why})


def ids() -> list[str]:
    return [
        Path(line.strip()).name
        for line in OFFICIAL_IDS.read_text().splitlines()
        if line.strip() and not line.strip().endswith(".csv")
    ]


def read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_submission(df.rename(columns={"image_id": "id"}), ids())
    return df.set_index("image_id").loc[ids()]


def load_frames() -> dict[str, pd.DataFrame]:
    check("official_ids", len(ids()) == 309, "309 official IMG ids")
    return {"texture_pa": read(TEXTURE_PA), "texture_rl": read(TEXTURE_RL), "scale_probe": read(SCALE_PROBE)}


def check_rl_is_cal_pa_mt(frames: dict[str, pd.DataFrame]) -> None:
    affine = json.loads(AFFINE.read_text())
    pa_a, pa_b = affine["pa_deg"]["a"], affine["pa_deg"]["b"]
    mt_a, mt_b = affine["mt_mm"]["a"], affine["mt_mm"]["b"]
    pa_expected = np.clip(pa_a * frames["texture_pa"].pa_deg + pa_b, 5, 45)
    mt_expected = np.clip(mt_a * frames["texture_pa"].mt_mm + mt_b, 10, 50)
    pa_err = float(np.max(np.abs(pa_expected.values - frames["texture_rl"].pa_deg.values)))
    mt_err = float(np.max(np.abs(mt_expected.values - frames["texture_rl"].mt_mm.values)))
    fl_err = float(np.max(np.abs(frames["texture_pa"].fl_mm.values - frames["texture_rl"].fl_mm.values)))
    check("fl_identity", fl_err < 1e-9, "texture_rl keeps texture_pa FL exactly")
    check("pa_affine_rebuild", pa_err < 1e-6, "texture_rl PA is clipped PA affine of texture_pa")
    check("mt_affine_rebuild", mt_err < 1e-6, "texture_rl MT is clipped MT affine of texture_pa")
    check("fl_identity_manifest", affine["fl_mm"]["function"] == "identity" and affine["fl_mm"]["applied"] is False, "manifest declares FL identity")


def check_scale_probe_worse(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    # Public evidence: x0.97 on FL+MT changed 0.29895 -> 0.35218. If a candidate
    # shrinks both lengths the same way it re-opens that direction.
    rl, probe = frames["texture_rl"], frames["scale_probe"]
    fl_ratio = float((probe.fl_mm / rl.fl_mm).median())
    mt_ratio = float((probe.mt_mm / rl.mt_mm).median())
    pa_same = float(np.max(np.abs(probe.pa_deg.values - rl.pa_deg.values)))
    check("probe_pa_unchanged", pa_same < 1e-12, "0.97 probe copied PA")
    check("probe_factor", abs(fl_ratio - 0.97) < 1e-6 and abs(mt_ratio - 0.97) < 1e-6, "probe is x0.97 on FL and MT")
    check("length_shrink_rejected", PUBLIC_017_DELTA > 0, "0.97 public 0.35218 > texture_rl 0.29895; shrinking lengths is not a 0.17 move")
    return {"fl_ratio": fl_ratio, "mt_ratio": mt_ratio}


def gain_budget(frames: dict[str, pd.DataFrame], candidate: pd.DataFrame, public_score_rl: float = 0.29895) -> dict[str, float]:
    # Worst-case budget if every candidate-vs-rl delta helped: what the
    # remaining 0.17 gap needs vs what a candidate can even touch.
    rl = frames["texture_rl"]
    d = {}
    for name, col, tol in (("pa", "pa_deg", 6.0), ("fl", "fl_mm", 12.0), ("mt", "mt_mm", 3.0)):
        delta = (candidate[col] - rl[col]).abs()
        d[f"mae_{name}_candidate_vs_rl"] = float(delta.mean())
        d[f"max_norm_{name}_available"] = float(delta.mean() / tol / 3.0)
    d["need_mean_norm_gain_for_017"] = MEAN_NORM_GAIN
    d["need_fl_mae_gain_if_fl_only"] = FL_MAE_GAIN_017
    d["need_pa_mae_gain_if_pa_only"] = PA_MAE_GAIN_017
    d["need_mt_mae_gain_if_mt_only"] = MT_MAE_GAIN_017
    return d


def require_candidate(path: Path, frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    candidate = read(path)
    rl = frames["texture_rl"]
    ratio_fl = candidate.fl_mm / rl.fl_mm
    ratio_mt = candidate.mt_mm / rl.mt_mm
    check(
        "candidate_no_length_shrink",
        float(ratio_fl.median()) >= 0.999 and float(ratio_mt.median()) >= 0.999,
        "candidate must not shrink both lengths; 0.97 already lost publicly",
    )
    changed = (
        (candidate.pa_deg - rl.pa_deg).abs().mean(),
        (candidate.fl_mm - rl.fl_mm).abs().mean(),
        (candidate.mt_mm - rl.mt_mm).abs().mean(),
    )
    check(
        "candidate_not_slight",
        max(changed[0] / 6.0, changed[1] / 12.0, changed[2] / 3.0) > 0.05,
        "candidate must change at least one target by >5% of tolerance on average",
    )
    budget = gain_budget(frames, candidate)
    return budget


def mutant_demo(frames: dict[str, pd.DataFrame]) -> None:
    # Breakage witness: a fake candidate that is identical to rl must fail
    # candidate_not_slight; one scaled by 0.97 must fail candidate_no_length_shrink.
    identical = frames["texture_rl"].reset_index()
    try:
        require_candidate_df(identical, frames, expect_fail="candidate_not_slight")
    except StageAbort as exc:
        check("mutant_identical_fails", exc.stage == "candidate_not_slight", "identical rows fail candidate_not_slight")
    shrunk = frames["texture_rl"].reset_index().copy()
    shrunk["fl_mm"] = shrunk["fl_mm"] * 0.97
    shrunk["mt_mm"] = shrunk["mt_mm"] * 0.97
    shrunk["pa_deg"] = shrunk["pa_deg"] + 2.0  # make it non-slight so only shrink trips
    try:
        require_candidate_df(shrunk, frames, expect_fail="candidate_no_length_shrink")
    except StageAbort as exc:
        check("mutant_shrink_fails", exc.stage == "candidate_no_length_shrink", "x0.97 rows fail candidate_no_length_shrink")


def require_candidate_df(candidate: pd.DataFrame, frames: dict[str, pd.DataFrame], expect_fail: str) -> None:
    rl = frames["texture_rl"]
    candidate = candidate.set_index("image_id").loc[rl.index]
    ratio_fl = candidate.fl_mm / rl.fl_mm
    ratio_mt = candidate.mt_mm / rl.mt_mm
    check(
        expect_fail if expect_fail == "candidate_no_length_shrink" else "mut_probe_lengths",
        float(ratio_fl.median()) >= 0.999 and float(ratio_mt.median()) >= 0.999,
        "mutant",
    )
    changed = (
        (candidate.pa_deg - rl.pa_deg).abs().mean(),
        (candidate.fl_mm - rl.fl_mm).abs().mean(),
        (candidate.mt_mm - rl.mt_mm).abs().mean(),
    )
    check(
        expect_fail if expect_fail == "candidate_not_slight" else "mut_probe_change",
        max(changed[0] / 6.0, changed[1] / 12.0, changed[2] / 3.0) > 0.05,
        "mutant",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=ROOT / "reports/phase3/chase018/fv_output_status.json")
    args = parser.parse_args()
    try:
        frames = load_frames()
        check_rl_is_cal_pa_mt(frames)
        probe = check_scale_probe_worse(frames)
        summary: dict[str, object] = {"probe": probe}
        if args.candidate is not None:
            summary["candidate_budget"] = require_candidate(args.candidate, frames)
        mutant_demo(frames)
        summary["block"] = "output_recipe_identifiability"
        summary["stages"] = STAGE_RESULTS
        summary["public_floor"] = {"texture_pa": 0.31903, "texture_rl": 0.29895, "scale_probe_0.97": 0.35218}
        summary["note"] = "Harness green = output contract holds. It is not a public-score claim."
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    except StageAbort as exc:
        summary = {"block": "output_recipe_identifiability", "aborted": exc.stage, "why": exc.why, "stages": STAGE_RESULTS}
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
