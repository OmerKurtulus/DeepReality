# =============================================================================
#  PIN-B3 / PIN-D1 bottleneck probe
#  Run in the SAME session, after stopping the extraction cell.
#  Models stay loaded, so this takes about 30 seconds.
# =============================================================================
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/content/DeepReality"); os.chdir("/content/DeepReality")
from PIL import Image

img = Image.new("RGB", (256, 256))
try:
    import glob
    cand = glob.glob("/content/_work/*")
    if cand:
        img = Image.open(cand[0]).convert("RGB")
except Exception:
    pass

def bench(label, fn, n=10, sync=False):
    fn()                                   # warm
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    print(f"  {label:<42} min {min(ts)*1000:9.2f} ms   median {sorted(ts)[n//2]*1000:9.2f} ms")

print("=" * 74)
print("  A · frequency transform, CPU only")
print("=" * 74)
from layer2_detection_core.pin_b3_freq import image_to_frequency_map, _load_model, _get_device
bench("image_to_frequency_map", lambda: image_to_frequency_map(img))

import cv2
print(f"\n  cv2 threads : {cv2.getNumThreads()}")
print(f"  torch threads: {torch.get_num_threads()}")

print("\n" + "=" * 74)
print("  B · PIN-B3 model forward, GPU")
print("=" * 74)
model = _load_model()
dev = _get_device()
print(f"  device: {dev}   model on: {next(model.parameters()).device}")

fmap = image_to_frequency_map(img)
tensor = torch.from_numpy(fmap).unsqueeze(0).to(dev)

def fwd():
    with torch.no_grad():
        logits, _ = model(tensor)
        _ = torch.softmax(logits, dim=1)[0, 0].item()
bench("B3 forward (tensor already on GPU)", fwd, sync=True)

def full():
    fm = image_to_frequency_map(img)
    t = torch.from_numpy(fm).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits, _ = model(t)
        _ = torch.softmax(logits, dim=1)[0, 0].item()
bench("B3 transform + H2D copy + forward", full, sync=True)

print("\n" + "=" * 74)
print("  C · cudnn.benchmark effect")
print("=" * 74)
print(f"  cudnn.benchmark currently: {torch.backends.cudnn.benchmark}")
torch.backends.cudnn.benchmark = False
bench("B3 forward with benchmark OFF", fwd, sync=True)
torch.backends.cudnn.benchmark = True

print("\n" + "=" * 74)
print("  D · whole pins, single-threaded (no orchestrator)")
print("=" * 74)
tmp = "/content/_probe.jpg"
img.save(tmp, quality=92)
from layer2_detection_core.pin_b3_freq import PinB3Freq
from layer2_detection_core.pin_b1_clip import PinB1Clip
from layer2_detection_core.pin_b2_siglip2 import PinB2Siglip
from layer4_xai.pin_d1_gradcam import PinD1GradCam

b1, b2, b3 = PinB1Clip(), PinB2Siglip(), PinB3Freq()
r1, r2, r3 = b1.run(tmp), b2.run(tmp), b3.run(tmp)
for p in (b1, b2, b3):
    bench(f"{p.pin_id} alone", lambda p=p: p.run(tmp), n=5)

d1 = PinD1GradCam()
ctx = {"PIN-B1": r1, "PIN-B2": r2, "PIN-B3": r3, "_pins": {}}
bench("PIN-D1 alone", lambda: d1.run(tmp, context=ctx), n=5)

print("""
=====================================================================
  READING THE RESULT
  A ~1 ms and B ~10 ms but D showing seconds  -> orchestrator contention
  B slow                                      -> GPU or cudnn issue
  A slow                                      -> CPU/threading on scipy
=====================================================================
""")
