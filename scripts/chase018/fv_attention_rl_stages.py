#!/usr/bin/env python3
"""Promotion gates for the spatial-attention + RL pipeline (SPEC.md, section "Promotion gates").

Every stage records PASS, FAIL, or BLOCKED (an input is missing on this
machine). Only an all-PASS run clears one Kaggle submission, and even then
nothing here is a public-score claim. Breakage mutants run whenever the
texture_rl CSV is present, so each CSV gate is shown to be able to fail.

Trusted base: this checker, the Lean kernel for formal/umud_rl, pytest for
the mirrored contracts, the score formula in src/umud/metric/score.py, and the
public number 0.29895 for texture_rl.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.baselines.io import load_official_submission_ids  # noqa: E402

TEXTURE_RL_PUBLIC = 0.29895
TARGET_PUBLIC = 0.16
NEED_S1 = TEXTURE_RL_PUBLIC - TARGET_PUBLIC
LAB = ROOT / "reports/phase3/chase018/attention_rl/lab"
LEAN = ROOT / "formal/umud_rl"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

STAGES: list[dict] = []


def record(stage: str, result: str, why: str, **extra) -> str:
    STAGES.append({"stage": stage, "result": result, "why": why, **extra})
    return result


def _verifier():
    spec = importlib.util.spec_from_file_location("umud_score_distance_verifier", ROOT / "UMUD score-distance verifier.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def stage_contracts() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_attention_rl_contracts.py", "tests/test_attention_rl_lab_layout.py"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
    record("contracts", "PASS" if proc.returncode == 0 else "FAIL", tail)


def stage_lean() -> None:
    lake = shutil.which("lake") or str(Path.home() / ".elan/bin/lake")
    if not Path(lake).exists():
        record("lean_build", "BLOCKED", "Lean toolchain (lake) not installed on this machine")
        return
    src = (LEAN / "UmudRl/Basic.lean").read_text()
    if re.search(r"\b(sorry|admit|native_decide)\b", src):
        record("lean_build", "FAIL", "proof escape hatch in sources")
        return
    build = subprocess.run([lake, "build"], cwd=LEAN, capture_output=True, text=True)
    if build.returncode != 0:
        record("lean_build", "FAIL", build.stdout[-400:] + build.stderr[-400:])
        return
    audit = subprocess.run([lake, "env", "lean", "Audit.lean"], cwd=LEAN, capture_output=True, text=True)
    axioms = {}
    for ln in audit.stdout.splitlines():
        m = re.match(r"'(.+)' depends on axioms: \[(.*)\]", ln)
        if m:
            axioms[m.group(1)] = sorted(a.strip() for a in m.group(2).split(",") if a.strip())
    bad = {k: v for k, v in axioms.items() if not set(v) <= ALLOWED_AXIOMS}
    ok = audit.returncode == 0 and len(axioms) == 11 and not bad
    record("lean_build", "PASS" if ok else "FAIL", f"{len(axioms)} theorems audited; disallowed axioms: {bad or 'none'}", axioms=axioms)


def stage_synthetic(path: Path) -> None:
    if not path.exists():
        record("synthetic_learns", "BLOCKED", f"missing {path.relative_to(ROOT)}")
        return
    r = json.loads(path.read_text())
    g = r["gates"]
    ok = g["val_attention_beats_maps"]
    record(
        "synthetic_learns",
        "PASS" if ok else "FAIL",
        "attention readout beats host measure on maps for PA and FL (synthetic val)",
        rl_beats_init_synthetic=g["val_rl_beats_init"],
    )


def stage_val(lab: Path) -> dict | None:
    path = lab / "val_report.json"
    if not path.exists():
        record("val_attention_beats_maps", "BLOCKED", "no lab val_report.json; run the train stage on the lab GPU")
        record("val_rl_beats_init", "BLOCKED", "no lab val_report.json")
        return None
    r = json.loads(path.read_text())
    v = r["vs_mask_geometry"]
    record(
        "val_attention_beats_maps",
        "PASS" if r["gates"]["val_attention_beats_maps"] else "FAIL",
        "held-out competition images, against mask geometry",
        maps=v["maps_host_measure"], attention=v["attention_readout"],
    )
    record(
        "val_rl_beats_init",
        "PASS" if r["gates"]["val_rl_beats_init"] else "RL_DISABLED",
        "RL refinement must lower line angle and offset error, else it is switched off",
        angle=r["line_angle_err_deg"], offset=r["line_offset_err_px"],
    )
    return r


def stage_rl_consistency(lab: Path, val: dict | None) -> None:
    meta_path = lab / "test_meta.json"
    if val is None or not meta_path.exists():
        record("rl_use_consistent", "BLOCKED", "needs val_report.json and test_meta.json")
        return
    meta = json.loads(meta_path.read_text())
    ok = (not meta["rl_used"]) or bool(val["gates"]["val_rl_beats_init"])
    record("rl_use_consistent", "PASS" if ok else "FAIL", "test inference used RL only if the validation gate allowed it")
    record(
        "fl_identity",
        "PASS" if meta.get("fl_affine") is None else "FAIL",
        "no FL affine in the candidate",
    )


def stage_multexp(lab: Path) -> None:
    path = lab / "multexp_report.json"
    if not path.exists():
        record("multexp_not_worse", "BLOCKED", "no multexp_report.json; run eval-multexp on the lab")
        return
    r = json.loads(path.read_text())
    record(
        "multexp_not_worse",
        "PASS" if r["gate_multexp_not_worse"] else "FAIL",
        "stress test only; MultExp ranking did not transfer to public before",
        baseline=r["baseline_host_measure"]["loocv_pa_mt_affine"], candidate=r["candidate"]["loocv_pa_mt_affine"],
    )


def _read(path: Path, ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.set_index("image_id").loc[ids] if set(df.image_id) == set(ids) else df.set_index("image_id")


def csv_gates(candidate: pd.DataFrame, rl: pd.DataFrame, ids: list[str], public_frac_min: float, verifier) -> dict[str, tuple[str, str, dict]]:
    out: dict[str, tuple[str, str, dict]] = {}
    lit = list(candidate.index)
    ok_ids = lit == ids and len(set(lit)) == len(lit) == len(ids)
    out["official_ids"] = ("PASS" if ok_ids else "FAIL", f"{len(lit)} literal ids in official order", {})
    if not ok_ids:
        return out
    base = rl[["pa_deg", "fl_mm", "mt_mm"]].to_numpy().tolist()
    cand = candidate[["pa_deg", "fl_mm", "mt_mm"]].to_numpy().tolist()
    k = max(1, math.ceil(public_frac_min * len(ids)))
    cert_any = verifier.distance_certificate(base, cand)
    cert_k = verifier.distance_certificate(base, cand, public_count=k)
    screen_any = verifier.assess_target_bound(cert_any["absolute_score_change_bound"], TEXTURE_RL_PUBLIC, TARGET_PUBLIC, 0.000005)
    screen_k = verifier.assess_target_bound(cert_k["absolute_score_change_bound"], TEXTURE_RL_PUBLIC, TARGET_PUBLIC, 0.000005)
    reachable = not screen_k["target_ruled_out_by_numerical_distance_screen"]
    out["movement_budget"] = (
        "PASS" if reachable else "FAIL",
        f"top-{k} movement bound {cert_k['subset_S1_bound']:.5f} vs need {NEED_S1:.5f} (assumes public subset >= {public_frac_min:.0%} of test)",
        {"full_set_S1_distance": cert_any["full_set_S1_distance"], "top_k_bound": cert_k["subset_S1_bound"],
         "ruled_out_any_subset": screen_any["target_ruled_out_by_numerical_distance_screen"], "k": k},
    )
    fl_ratio = float((candidate.fl_mm / rl.fl_mm).median())
    mt_ratio = float((candidate.mt_mm / rl.mt_mm).median())
    shrink = fl_ratio < 0.999 and mt_ratio < 0.999
    out["no_joint_length_shrink"] = ("FAIL" if shrink else "PASS", f"median FL ratio {fl_ratio:.4f}, MT ratio {mt_ratio:.4f}", {})
    return out


def stage_csvs(lab: Path, public_frac_min: float) -> None:
    ids_path = ROOT / "reports/phase3/official_test_file_names.txt"
    rl_path = ROOT / "reports/phase3/chase018/submission_texture_rl.csv"
    if not rl_path.exists():
        for s in ("official_ids", "movement_budget", "no_joint_length_shrink", "csv_mutants"):
            record(s, "BLOCKED", "submission_texture_rl.csv is not on this machine")
        return
    ids = load_official_submission_ids(ids_path)
    rl = _read(rl_path, ids)
    verifier = _verifier()
    found = False
    for name in ("pa_only", "fl_only", "pa_fl"):
        path = lab / f"submission_attn_rl_{name}.csv"
        if not path.exists():
            continue
        found = True
        for stage, (res, why, extra) in csv_gates(_read(path, ids), rl, ids, public_frac_min, verifier).items():
            record(f"{stage}[{name}]", res, why, **extra)
    if not found:
        record("candidate_csvs", "BLOCKED", "no submission_attn_rl_*.csv; run infer-test on the lab")
    identical = rl.copy()
    shrunk = rl.copy()
    shrunk["fl_mm"] *= 0.97
    shrunk["mt_mm"] *= 0.97
    renamed = rl.rename(index={ids[0]: ids[0].replace(".", "_x.")})
    m1 = csv_gates(identical, rl, ids, public_frac_min, verifier)["movement_budget"][0] == "FAIL"
    m2 = csv_gates(shrunk, rl, ids, public_frac_min, verifier)["no_joint_length_shrink"][0] == "FAIL"
    m3 = csv_gates(renamed, rl, ids, public_frac_min, verifier)["official_ids"][0] == "FAIL"
    record("csv_mutants", "PASS" if m1 and m2 and m3 else "FAIL", "identical, x0.97 lengths, and a renamed id each fail their gate",
           identical_fails=m1, shrink_fails=m2, renamed_fails=m3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", type=Path, default=LAB)
    ap.add_argument("--synthetic", type=Path, default=ROOT / "reports/phase3/chase018/attention_rl/synthetic_run.json")
    ap.add_argument("--public-frac-min", type=float, default=0.3)
    ap.add_argument("--skip-contracts", action="store_true")
    ap.add_argument("--out-json", type=Path, default=ROOT / "reports/phase3/chase018/attention_rl/fv_status.json")
    args = ap.parse_args()
    if not args.skip_contracts:
        stage_contracts()
    stage_lean()
    stage_synthetic(args.synthetic)
    val = stage_val(args.lab)
    stage_rl_consistency(args.lab, val)
    stage_multexp(args.lab)
    stage_csvs(args.lab, args.public_frac_min)
    allowed = {"PASS", "RL_DISABLED"}
    cleared = all(s["result"] in allowed for s in STAGES) and any(s["stage"].startswith("movement_budget") for s in STAGES)
    summary = {
        "block": "attention_rl_promotion",
        "cleared_for_one_submission": cleared,
        "counts": {r: sum(s["result"] == r for s in STAGES) for r in ("PASS", "FAIL", "BLOCKED", "RL_DISABLED")},
        "stages": STAGES,
        "public": {"texture_rl": TEXTURE_RL_PUBLIC, "target": TARGET_PUBLIC, "need_S1_drop": NEED_S1},
        "note": "Green = contracts and gates hold. It is not a public-score claim.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("cleared_for_one_submission", "counts")}, indent=2))
    for s in STAGES:
        print(f"{s['result']:<12} {s['stage']:<32} {s['why']}")
    return 0 if cleared else 1


if __name__ == "__main__":
    raise SystemExit(main())
