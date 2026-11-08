import sys
from pathlib import Path
ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "scripts/chase018"))
sys.path.insert(0, str(ROOT / "src"))
import inspect
from umud.seg.frundles_smp import load_ckpt
print(inspect.getsource(load_ckpt))
from adapt_multexp_fasc import score_components
print(inspect.getsource(score_components))
# how other scripts load frundles
import torch, segmentation_models_pytorch as smp
p = ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt"
m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
load_ckpt(m, p)
print("load_ckpt_default_OK", sum(x.numel() for x in m.parameters()))
ck = torch.load(p, map_location="cpu", weights_only=False)
print("ck keys", ck.keys())
print("state type", type(ck["state"]), "len", len(ck["state"]) if isinstance(ck["state"], dict) else None)
if isinstance(ck["state"], dict):
    print("state sample", list(ck["state"].keys())[:6])
