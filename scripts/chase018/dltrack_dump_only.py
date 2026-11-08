#!/usr/bin/env python3
"""Dump MultExp fasc maps with Zenodo DLTrack VGG16 h5 (run under .venv_tf)."""
from __future__ import annotations
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path("/home/chocheng9/kaggle_UMUD")
H5 = ROOT / "data/external/osf_xbawc/labtrain_DLTrack_models/DLTrack_models/model-fasc-VGG16-BCE-512.h5"
CACHE = ROOT / "reports/phase3/chase018/cache/maps_multexp_dltrack_fasc"
APO = ROOT / "reports/phase3/chase018/cache/maps_multexp"

print("tf", tf.__version__, "devices", tf.config.list_physical_devices(), flush=True)
CACHE.mkdir(parents=True, exist_ok=True)

from tensorflow.keras.layers import Conv2DTranspose as _Conv2DTranspose  # noqa: E402


class Conv2DTranspose(_Conv2DTranspose):
    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        cfg.pop("groups", None)
        return super().from_config(cfg)


model = tf.keras.models.load_model(
    str(H5), compile=False, custom_objects={"Conv2DTranspose": Conv2DTranspose}
)
print("input", model.input_shape, "output", model.output_shape, flush=True)
expert = pd.read_csv(ROOT / "data/external/umud_expert_arch_v0.1.0/expert_consensus.csv")
expert_dir = ROOT / "data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
th, tw, ch = model.input_shape[1], model.input_shape[2], model.input_shape[3]
n = 0
for _, er in expert.iterrows():
    name = er["image_file"]
    g = cv2.imread(str(expert_dir / name), cv2.IMREAD_GRAYSCALE)
    zapo = APO / f"{name}.npz"
    if g is None or not zapo.exists():
        print("skip", name, flush=True)
        continue
    apo = np.load(zapo)["apo"].astype(np.float32)
    x = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.stack([x, x, x], -1) if ch == 3 else x[..., None]
    pred = model.predict(x[None], verbose=0)[0]
    fasc = pred[..., 0] if pred.ndim == 3 else pred
    fasc = fasc.astype(np.float32)
    if float(fasc.max()) > 1.5:
        fasc = 1.0 / (1.0 + np.exp(-fasc))
    if fasc.shape != apo.shape:
        fasc = cv2.resize(fasc, (apo.shape[1], apo.shape[0]), interpolation=cv2.INTER_LINEAR)
    np.savez_compressed(
        CACHE / f"{name}.npz", apo=apo.astype(np.float16), fasc=fasc.astype(np.float16)
    )
    n += 1
    print("ok", n, name, "fasc", float(fasc.min()), float(fasc.max()), float(fasc.mean()), flush=True)
print("dumped", n, flush=True)
(ROOT / "reports/phase3/chase018/dltrack_dump_done.json").write_text(
    json.dumps({"n": n, "h5": str(H5), "tf": tf.__version__, "mode": "cpu"}, indent=2)
)