import inspect, sys
from pathlib import Path
ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT / "scripts/chase018"))
sys.path.insert(0, str(ROOT / "src"))
from adapt_multexp_fasc import loocv_cal_pa_mt, FROZEN
print("FROZEN", FROZEN)
src = inspect.getsource(loocv_cal_pa_mt)
print(src[:3000])
print("---RETURN HINTS---")
for line in src.splitlines():
    if "return" in line:
        print(line)
import torch, segmentation_models_pytorch as smp
from umud.seg.frundles_smp import load_ckpt
p = ROOT / "data/external/frundles_weights/fasc_resnet34_384x576_f0.pt"
m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
try:
    load_ckpt(m, p, map_location="cpu")
    print("FRUNDLES_OK", sum(x.numel() for x in m.parameters()))
except Exception as e:
    print("FRUNDLES_FAIL", type(e).__name__, e)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("keys", list(ck.keys())[:20] if isinstance(ck, dict) else type(ck))
    if isinstance(ck, dict):
        sd = ck.get("model", ck.get("state_dict", ck))
        if isinstance(sd, dict):
            print("sd_n", len(sd), "sample", list(sd.keys())[:8])
            try:
                m.load_state_dict(sd, strict=False)
                print("STRICT_FALSE_OK")
            except Exception as e2:
                print("STRICT_FALSE_FAIL", e2)
