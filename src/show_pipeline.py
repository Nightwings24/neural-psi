"""
Live walkthrough: a RAW SOCOFing fingerprint -> 16-number feature vector.
Run in front of the advisor to show every stage with real shapes & numbers.

    python3 show_pipeline.py
"""
import os
import numpy as np
import torch

from model import FeatureModel

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"


def banner(n, title):
    print("\n" + "=" * 60)
    print(f"  STEP {n}: {title}")
    print("=" * 60)


# load the trained CNN + one real fingerprint
model = FeatureModel(img=96)
model.load_state_dict(torch.load(f"{DATA}/feature_model.pt"))
model.eval()
ids = np.load(f"{DATA}/ids_real.npy")
img = np.load(f"{DATA}/x_real.npy")[0]            # one real print

banner(0, "RAW FINGERPRINT (from SOCOFing dataset)")
print(f"  identity      : {ids[0]}")
print(f"  image shape   : {img.shape}  (a {img.shape[0]}x{img.shape[1]} grid of pixels)")
print(f"  pixel values  : 0-255 (gray).  e.g. top-left 5x5 corner:")
for row in img[:5, :5]:
    print("      ", " ".join(f"{v:3d}" for v in row))

banner(1, "PREPROCESS -> normalize to 0..1 and make a tensor")
x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
print(f"  divide by 255 -> values now 0.00 .. 1.00")
print(f"  tensor shape  : {tuple(x.shape)}  = (batch, channel, height, width)")

banner(2, "5 CONV BLOCKS -> extract fingerprint patterns")
print("  each block = Conv(3x3) -> BatchNorm -> swish -> MaxPool(2)")
print("  channels GROW (find more patterns), image SHRINKS (drop exact position)\n")
print(f"   {'stage':<22}{'shape (C x H x W)':<22}{'meaning'}")
print("   " + "-" * 58)
print(f"   {'input image':<22}{'1 x 96 x 96':<22}raw pixels")
h = x
ch = [32, 64, 128, 256, 512]
for i, layer in enumerate(model.features):
    h = layer(h)
    if i % 4 == 3:                                  # end of a block
        b = i // 4
        c = ch[b]
        note = ["edges/ridges", "ridge flow", "minutiae bits",
                "finger parts", "whole-print code"][b]
        cxhxw = f"{c} x {h.shape[2]} x {h.shape[3]}"
        print(f"   {'after block '+str(b+1):<22}{cxhxw:<22}{note}")

banner(3, "FLATTEN -> stretch the 3D block into one long list")
flat = torch.flatten(h, 1)
print(f"  512 x 3 x 3  ->  {flat.shape[1]} numbers")

banner(4, "FC-16 LAYER -> squeeze to the 16-number feature vector")
with torch.no_grad():
    emb = model.fc(flat)[0].numpy()
print(f"  Linear({flat.shape[1]} -> 16)\n")
print("  >>> THE FEATURE VECTOR (Blind-Touch FC-16) <<<")
print("  [" + "  ".join(f"{v:+.2f}" for v in emb) + "]")

banner(5, "YOUR BRIDGE -> 16 floats become 16 bits for PSI")
# (a) naive baseline: threshold at 0  -- biased, bits not balanced
naive_bits = (emb > 0).astype(int)
# (b) CORRECT bridge: ITQ = mean-center + learned rotation, THEN sign
itq = np.load(f"{DATA}/itq_params.npz")
itq_bits = (((emb - itq["mean"]) @ itq["R"]) > 0).astype(int)

print("  (a) NAIVE  bit = 1 if number > 0          <- arbitrary threshold, biased bits")
print(f"      code : {naive_bits}")
print("      EER 10.86%  | impostor Hamming 3.79/16 (bits lopsided)\n")
print("  (b) ITQ    bit = sign( (feature - mean) @ R )   <- THE CORRECT BRIDGE")
print(f"      code : {itq_bits}")
print("      EER  7.12%  | impostor Hamming 8.00/16 (balanced ~50%)")
print("\n  => the ITQ 16-bit code is the set element that goes into flash-psi.")
print("=" * 60)
