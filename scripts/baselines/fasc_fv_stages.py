#!/usr/bin/env python3
"""Formal-verification stage harness for fasc thin-structure ceiling (v5 baseline).

Block id: fasc_thin_ceiling
Failing number: native macro Dice 0.2846 << 0.40 usable / reopen bar.

Anthropic-style: each stage aborts with a named stage id on break (raise / nonzero).
Soft assert is not a property.

Stages (trusted chain):
  align_nearest | edt_radius | mstiny_forward | deep_sup_aux
  | loss_mild_weights | native_macro_dice | vs_v5_floor
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.seg.ms_unet import MSTinyUNet  # noqa: E402

# --- Frozen v5 knobs (must match train_seg_fasc_v5) ---
V5_EDT_RADIUS = 2.5
V5_CLDICE_W = 0.5
V5_HARD_DICE_W = 0.5
V5_AUX_W = (0.3, 0.2)  # aux3, aux2
V5_FLOOR = 0.2846309831975224  # recorded best native macro Dice
V5_FLOOR_TOL = 1e-4
USABLE_BAR = 0.40
BLOCK_ID = "fasc_thin_ceiling"


class StageFail(Exception):
    """Aborting property failure with named stage id."""

    def __init__(self, stage: str, msg: str):
        self.stage = stage
        super().__init__(f"[{stage}] {msg}")


def align_mask_to_img(hard: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if hard.shape == (h, w):
        return hard.astype(np.uint8)
    return (
        np.asarray(Image.fromarray(hard.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    ).astype(np.uint8)


def edt_soft(hard: np.ndarray, radius: float = V5_EDT_RADIUS) -> np.ndarray:
    from scipy import ndimage

    if hard.max() == 0:
        return hard.astype(np.float32)
    dist = ndimage.distance_transform_edt(hard.astype(bool))
    return np.clip(dist / float(radius), 0.0, 1.0).astype(np.float32)


def stage_align_nearest() -> dict:
    """mask HW matches image after NEAREST resize."""
    img_hw = (97, 130)
    # mismatched mask
    hard = np.zeros((50, 80), dtype=np.uint8)
    hard[10:40, 20:40] = 1
    hard[25, 30:70] = 1
    aligned = align_mask_to_img(hard, img_hw)
    if aligned.shape != img_hw:
        raise StageFail("align_nearest", f"aligned shape {aligned.shape} != img {img_hw}")
    if aligned.dtype != np.uint8:
        raise StageFail("align_nearest", f"dtype {aligned.dtype} != uint8")
    # NEAREST must not invent mid-gray: values only 0/1
    uniq = set(np.unique(aligned).tolist())
    if not uniq.issubset({0, 1}):
        raise StageFail("align_nearest", f"NEAREST leaked non-binary values: {uniq}")
    return {"img_hw": list(img_hw), "aligned_hw": list(aligned.shape), "uniq": sorted(uniq)}


def stage_edt_radius(*, radius: float = V5_EDT_RADIUS) -> dict:
    """soft target uses r=2.5 (config frozen)."""
    if abs(float(radius) - V5_EDT_RADIUS) > 1e-9:
        raise StageFail(
            "edt_radius",
            f"config radius={radius} != frozen v5 {V5_EDT_RADIUS} (forbidden: edt>3.0 / drift)",
        )
    hard = np.zeros((64, 64), dtype=np.uint8)
    hard[20:45, 30:35] = 1  # thin vertical strip
    soft = edt_soft(hard, radius=radius)
    if soft.shape != hard.shape:
        raise StageFail("edt_radius", f"soft shape {soft.shape} != hard {hard.shape}")
    if soft.max() > 1.0 + 1e-6 or soft.min() < -1e-6:
        raise StageFail("edt_radius", f"soft out of [0,1]: [{soft.min()}, {soft.max()}]")
    # interior should be near 1 when thickness/2 ≈ radius
    # for strip width 5, max dist ≈ 2.5 → soft max ≈ 1.0 at center
    if soft[hard.astype(bool)].max() < 0.5:
        raise StageFail("edt_radius", "soft thickness collapsed (max on FG < 0.5)")
    return {"radius": radius, "soft_max": float(soft.max()), "soft_fg_mean": float(soft[hard.astype(bool)].mean())}


def stage_mstiny_forward(*, base: int = 48, size: int = 64, n_aux: int = 2) -> dict:
    """MSTinyUNet returns logits (+ aux if return_aux)."""
    model = MSTinyUNet(1, base, n_aux=n_aux)
    model.eval()
    x = torch.zeros(1, 1, size, size)
    with torch.no_grad():
        logits = model(x, return_aux=False)
        aux_out = model(x, return_aux=True)
    if not isinstance(logits, torch.Tensor):
        raise StageFail("mstiny_forward", f"logits type {type(logits)}")
    if logits.shape != (1, 1, size, size):
        raise StageFail("mstiny_forward", f"logits shape {tuple(logits.shape)} != (1,1,{size},{size})")
    if not isinstance(aux_out, tuple):
        raise StageFail("mstiny_forward", "return_aux did not return tuple")
    expected_len = 1 + n_aux
    if len(aux_out) != expected_len:
        raise StageFail(
            "mstiny_forward",
            f"return_aux len={len(aux_out)} expected {expected_len} (logits+{n_aux} aux)",
        )
    return {
        "logits_shape": list(logits.shape),
        "n_aux": n_aux,
        "aux_tuple_len": len(aux_out),
        "base": base,
    }


def stage_deep_sup_aux(*, n_aux: int = 2, size: int = 64, base: int = 48) -> dict:
    """aux3/aux2 (and aux1 if n_aux=3) present and same spatial size as logits."""
    model = MSTinyUNet(1, base, n_aux=n_aux)
    model.eval()
    x = torch.zeros(1, 1, size, size)
    with torch.no_grad():
        out = model(x, return_aux=True)
    logits = out[0]
    auxs = out[1:]
    if len(auxs) < 2:
        raise StageFail("deep_sup_aux", f"need aux3/aux2, got {len(auxs)} aux tensors")
    for i, a in enumerate(auxs):
        if a.shape != logits.shape:
            raise StageFail(
                "deep_sup_aux",
                f"aux[{i}] shape {tuple(a.shape)} != logits {tuple(logits.shape)}",
            )
    names = ["aux3", "aux2"] + (["aux1"] if n_aux >= 3 else [])
    if n_aux >= 3 and not hasattr(model, "aux1"):
        raise StageFail("deep_sup_aux", "n_aux=3 but model missing aux1 module")
    # v5 default must stay loadable without aux1 key
    if n_aux == 2 and hasattr(model, "aux1"):
        raise StageFail("deep_sup_aux", "n_aux=2 must not allocate aux1 (v5 ckpt compat)")
    return {"n_aux": n_aux, "aux_names": names, "spatial": list(logits.shape[-2:])}


def stage_loss_mild_weights(*, cldice_w: float = V5_CLDICE_W, hard_dice_w: float = V5_HARD_DICE_W) -> dict:
    """clDice weight == 0.5, hard Dice == 0.5 (not amplified)."""
    if abs(float(cldice_w) - V5_CLDICE_W) > 1e-9:
        raise StageFail(
            "loss_mild_weights",
            f"cldice_w={cldice_w} != frozen {V5_CLDICE_W} (forbidden: clDice>=1.0)",
        )
    if abs(float(hard_dice_w) - V5_HARD_DICE_W) > 1e-9:
        raise StageFail(
            "loss_mild_weights",
            f"hard_dice_w={hard_dice_w} != frozen {V5_HARD_DICE_W}",
        )
    if cldice_w >= 1.0:
        raise StageFail("loss_mild_weights", "clDice amplified (>=1.0) — forbidden under fasc_thin_ceiling")
    return {"cldice_w": cldice_w, "hard_dice_w": hard_dice_w, "aux_w_v5": list(V5_AUX_W)}


def stage_native_macro_dice(*, eval_json: Path | None, rank_json: Path | None) -> dict:
    """Eval metric path documented; if eval JSON present, assert recorded best == ~0.2846."""
    documented = {
        "gate_metric": "native-res per-image MACRO Dice on hard masks",
        "mode": "square_resize_native",
        "winner_variant": "fasc_v5_edt_cldice_ms",
        "v5_floor": V5_FLOOR,
        "usable_bar": USABLE_BAR,
        "block_id": BLOCK_ID,
    }
    recorded = None
    source = None
    if eval_json is not None and eval_json.is_file():
        data = json.loads(eval_json.read_text())
        # tolerate several shapes
        recorded = (
            data.get("best_native_macro_dice")
            or data.get("macro_dice")
            or (data.get("best_detail") or {}).get("macro_dice")
            or data.get("val_dice")
        )
        source = str(eval_json)
    elif rank_json is not None and rank_json.is_file():
        data = json.loads(rank_json.read_text())
        ranked = data.get("ranked") or []
        if ranked:
            recorded = ranked[0].get("best_native_macro_dice")
            source = str(rank_json) + "#ranked[0]"
    if recorded is None:
        # Property: path must be documented even without live JSON — do not soft-pass a fake score.
        documented["status"] = "eval_json_absent_path_documented"
        documented["note"] = (
            "No eval JSON on this host; stage passes documentation-only. "
            "Re-run on SSH with --eval-json reports/phase3/eval_seg_fasc_v5_best_native.json"
        )
        return documented
    recorded = float(recorded)
    if abs(recorded - V5_FLOOR) > V5_FLOOR_TOL:
        raise StageFail(
            "native_macro_dice",
            f"recorded best {recorded} != v5 floor {V5_FLOOR} within {V5_FLOOR_TOL}",
        )
    documented["recorded"] = recorded
    documented["source"] = source
    documented["status"] = "matched_v5_floor"
    if recorded >= USABLE_BAR:
        raise StageFail(
            "native_macro_dice",
            f"unexpected: recorded {recorded} >= usable bar {USABLE_BAR} — reopen protocol, do not silently clear block",
        )
    return documented


def stage_vs_v5_floor(*, candidate_score: float | None, require_candidate: bool) -> dict:
    """Any candidate ckpt must not score below v5 (property for mutants)."""
    if candidate_score is None:
        if require_candidate:
            raise StageFail("vs_v5_floor", "candidate_score required but missing")
        return {
            "status": "skipped_no_candidate",
            "v5_floor": V5_FLOOR,
            "rule": "mutant native macro Dice must be > V5_FLOOR or abort",
        }
    score = float(candidate_score)
    if score < V5_FLOOR:
        raise StageFail(
            "vs_v5_floor",
            f"candidate {score} < v5 floor {V5_FLOOR} — mutant rejected",
        )
    return {"status": "ok", "candidate": score, "v5_floor": V5_FLOOR, "delta": score - V5_FLOOR}


def load_v5_ckpt_smoke(ckpt: Path, *, device: str = "cpu") -> dict:
    """Optional: load v5 weights into n_aux=2 model (strict)."""
    if not ckpt.is_file():
        return {"status": "ckpt_absent", "path": str(ckpt)}
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = MSTinyUNet(1, int(blob.get("base_ch", 48)) if isinstance(blob, dict) else 48, n_aux=2)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # v5 must load with zero missing for core weights; aux1 must not be required
    bad_missing = [k for k in missing if not k.startswith("aux1")]
    if bad_missing:
        raise StageFail("mstiny_forward", f"v5 ckpt missing keys: {bad_missing[:8]}")
    if any(k.startswith("aux1") for k in unexpected):
        raise StageFail("mstiny_forward", f"unexpected aux1 in v5 ckpt: {unexpected}")
    model.eval()
    with torch.no_grad():
        y = model(torch.zeros(1, 1, 512, 512))
    return {
        "status": "loaded",
        "path": str(ckpt),
        "logits_shape": list(y.shape),
        "val_dice_meta": float(blob.get("val_dice", -1)) if isinstance(blob, dict) else None,
        "variant": blob.get("variant") if isinstance(blob, dict) else None,
    }


STAGES = [
    "align_nearest",
    "edt_radius",
    "mstiny_forward",
    "deep_sup_aux",
    "loss_mild_weights",
    "native_macro_dice",
    "vs_v5_floor",
]


def run_all(args: argparse.Namespace) -> dict:
    histogram: dict[str, str] = {}
    details: dict[str, object] = {}
    failed_stage = None
    err_msg = None

    runners = [
        ("align_nearest", lambda: stage_align_nearest()),
        ("edt_radius", lambda: stage_edt_radius(radius=args.edt_radius)),
        ("mstiny_forward", lambda: stage_mstiny_forward(n_aux=args.n_aux)),
        ("deep_sup_aux", lambda: stage_deep_sup_aux(n_aux=args.n_aux)),
        ("loss_mild_weights", lambda: stage_loss_mild_weights(cldice_w=args.cldice_w)),
        (
            "native_macro_dice",
            lambda: stage_native_macro_dice(eval_json=args.eval_json, rank_json=args.rank_json),
        ),
        (
            "vs_v5_floor",
            lambda: stage_vs_v5_floor(
                candidate_score=args.candidate_score,
                require_candidate=args.require_candidate,
            ),
        ),
    ]

    for name, fn in runners:
        try:
            details[name] = fn()
            histogram[name] = "PASS"
        except StageFail as e:
            histogram[name] = "FAIL"
            details[name] = {"error": str(e)}
            failed_stage = e.stage
            err_msg = str(e)
            break
        except Exception as e:
            histogram[name] = "FAIL"
            details[name] = {"error": repr(e), "traceback": traceback.format_exc()}
            failed_stage = name
            err_msg = f"[{name}] unexpected: {e}"
            break

    ckpt_info = None
    if args.ckpt is not None:
        try:
            ckpt_info = load_v5_ckpt_smoke(args.ckpt)
        except StageFail as e:
            if failed_stage is None:
                failed_stage = e.stage
                err_msg = str(e)
                histogram["mstiny_forward"] = "FAIL"
            ckpt_info = {"error": str(e)}

    # Also verify n_aux=3 path exists for v12 mutant (non-aborting probe unless --check-aux1)
    aux1_probe = None
    if args.check_aux1 and failed_stage is None:
        try:
            aux1_probe = {
                "forward": stage_mstiny_forward(n_aux=3),
                "deep_sup": stage_deep_sup_aux(n_aux=3),
            }
            histogram["aux1_probe"] = "PASS"
        except StageFail as e:
            histogram["aux1_probe"] = "FAIL"
            failed_stage = e.stage
            err_msg = str(e)
            aux1_probe = {"error": str(e)}

    report = {
        "block_id": BLOCK_ID,
        "failing_number": V5_FLOOR,
        "usable_bar": USABLE_BAR,
        "attention_blocked": True,
        "histogram": histogram,
        "details": details,
        "ckpt_smoke": ckpt_info,
        "aux1_probe": aux1_probe,
        "failed_stage": failed_stage,
        "ok": failed_stage is None,
        "v5_structure_frozen": {
            "align": "NEAREST mask→img",
            "edt_radius": V5_EDT_RADIUS,
            "cldice_w": V5_CLDICE_W,
            "hard_dice_w": V5_HARD_DICE_W,
            "aux_w": list(V5_AUX_W),
            "arch": "MSTinyUNet base48 @512",
        },
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edt-radius", type=float, default=V5_EDT_RADIUS)
    ap.add_argument("--cldice-w", type=float, default=V5_CLDICE_W)
    ap.add_argument("--n-aux", type=int, default=2, choices=[2, 3])
    ap.add_argument("--eval-json", type=Path, default=ROOT / "reports/phase3/eval_seg_fasc_v5_best_native.json")
    ap.add_argument("--rank-json", type=Path, default=ROOT / "reports/phase3/fasc_native_rank.json")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "reports/phase3/seg_ckpts/seg_fasc_v5_best.pt")
    ap.add_argument("--candidate-score", type=float, default=None, help="mutant native macro Dice")
    ap.add_argument("--require-candidate", action="store_true")
    ap.add_argument("--check-aux1", action="store_true", help="also probe n_aux=3 path for v12")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/phase3/fasc_fv_status.json")
    args = ap.parse_args()

    report = run_all(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"block_id={BLOCK_ID} failing_number={V5_FLOOR} usable_bar={USABLE_BAR}", flush=True)
    print("stage_histogram:", flush=True)
    for s in STAGES + (["aux1_probe"] if "aux1_probe" in report["histogram"] else []):
        if s in report["histogram"]:
            print(f"  {s}: {report['histogram'][s]}", flush=True)
    if report["ckpt_smoke"] is not None:
        print(f"ckpt_smoke: {report['ckpt_smoke'].get('status', report['ckpt_smoke'])}", flush=True)
    print(f"wrote {args.out}", flush=True)

    if not report["ok"]:
        print(f"ABORT stage={report['failed_stage']}", flush=True)
        return 1
    print("ALL_STAGES_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
