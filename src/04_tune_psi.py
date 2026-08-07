"""
Week 1 — Step 4: Tune the flash-psi fuzzy matcher (weight, t) for 16-bit vectors.

The default flash-psi params (weight=14, T=64, t=2) are for 128-bit vectors.
Our Blind-Touch bridge emits 16 bits, so we re-derive (weight, t) from the REAL
held-out SOCOFing Hamming distributions:

  - TAR (true accept rate)  = fraction of GENUINE pairs the matcher accepts.
  - FAR (false accept rate) = fraction of IMPOSTOR pairs the matcher accepts.

We sweep weight and threshold t (T fixed = 64) and report the operating point
that drives FAR ~ 0 while keeping TAR high. Uses the ITQ bridge (the better one).
Writes the chosen params to data/psi_params.npz for the real demo to load.
"""
import math
import os
import numpy as np
import torch

from model import FeatureModel
from flpsi_match import SubSampler

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
D, T = 16, 64
rng = np.random.default_rng(42)


def embed_all(model, x, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1)
            out.append(model(xb).numpy())
    return np.concatenate(out)


def main():
    x_real = np.load(f"{DATA}/x_real.npy")
    ids_real = np.load(f"{DATA}/ids_real.npy")
    x_probe = np.load(f"{DATA}/x_probe.npy")
    ids_probe = np.load(f"{DATA}/ids_probe.npy")
    ids_test = set(np.load(f"{DATA}/ids_test.npy").tolist())
    itq = np.load(f"{DATA}/itq_params.npz")
    mean, R = itq["mean"], itq["R"]

    model = FeatureModel(img=96)
    model.load_state_dict(torch.load(f"{DATA}/feature_model.pt"))

    real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
    common = sorted(set(real_by) & set(prb_by))
    gal = np.stack([real_by[i] for i in common])
    prb = np.stack([prb_by[i] for i in common])

    def bits(x):
        return ((embed_all(model, x) - mean) @ R > 0).astype(np.int8)

    bgal, bprb = bits(gal), bits(prb)
    n = len(common)

    # Per-pair "differs" vectors (1 where the two 16-bit codes disagree).
    # A pair sub-matches on mask m  <=>  no selected position differs  <=>  diff·m == 0.
    gen_diff = (bprb != bgal).astype(np.int32)                       # (n, D)
    imp_rows = []
    for i in range(n):
        j = rng.choice(np.delete(np.arange(n), i), size=50, replace=False)
        imp_rows.append((bprb[i] != bgal[j]).astype(np.int32))      # (50, D)
    imp_diff = np.concatenate(imp_rows)                              # (n*50, D)
    print(f"[tune] genuine pairs {len(gen_diff)} | impostor pairs {len(imp_diff)} | D={D} T={T}")

    best = None
    print(f"\n  {'weight':>6} {'t':>3} {'TAR%':>7} {'FAR%':>7}")
    for weight in range(1, D):
        if math.comb(D, weight) < T:           # not enough unique masks (e.g. w=1,15)
            continue
        masks = SubSampler(D, weight, T, seed=1).masks.astype(np.int32)   # (T, D)
        # sub-match count per pair = # masks whose selected positions all agree
        gen_sm = ((gen_diff @ masks.T) == 0).sum(1)                  # (n,)
        imp_sm = ((imp_diff @ masks.T) == 0).sum(1)                  # (n*50,)
        for t in range(1, T + 1):
            tar = np.mean(gen_sm >= t)
            far = np.mean(imp_sm >= t)
            # objective: maximize TAR while FAR ~ 0; tie-break on lower FAR
            score = (tar - 10.0 * far, -far)
            if best is None or score > best[0]:
                best = (score, weight, t, tar, far)
        # print a readable row at a sensible t for this weight
        t_show = max(1, T // 8)
        print(f"  {weight:>6} {t_show:>3} {np.mean(gen_sm>=t_show)*100:>6.1f} "
              f"{np.mean(imp_sm>=t_show)*100:>6.2f}")

    _, w, t, tar, far = best
    print("\n" + "=" * 52)
    print("  CHOSEN flash-psi operating point (16-bit, ITQ)")
    print("=" * 52)
    print(f"  weight = {w}   t = {t}   T = {T}")
    print(f"  TAR (genuine accepted)  = {tar*100:.2f}%")
    print(f"  FAR (impostor accepted) = {far*100:.2f}%")
    print("=" * 52)
    np.savez(f"{DATA}/psi_params.npz", d=D, weight=w, t=t, T=T)
    print(f"[tune] saved -> {DATA}/psi_params.npz")


if __name__ == "__main__":
    main()
