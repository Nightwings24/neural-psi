"""
Step 21: identity-level bootstrap 95% CI for Super-Bit vs ortho-thermometer EER (Altered-Easy).
Fast: EER via cumulative Hamming histograms; resample the 1200 held-out identities 1000x.
"""
import json, os
import numpy as np
from quantizer import NeuralPSIQuantizer, QuantizerConfig
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); Dc = 128

z = np.load(f"{DATA}/dim_ablation_emb_224.npz", allow_pickle=True)
gal, prb, tr = z["gal"].astype(np.float64), z["prb"].astype(np.float64), z["train"].astype(np.float64)
n = len(gal); mu = tr.mean(0)

def hamm(A, B): A = A.astype(np.int32); B = B.astype(np.int32); return A@(1-B).T+(1-A)@B.T
def superbit():
    q = NeuralPSIQuantizer.fit(tr, np.ones(16), QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    return q.transform(gal).astype(np.uint8), q.transform(prb).astype(np.uint8)
def orthothermo(K=16, Bp=8, seed=1):
    r = np.random.default_rng(seed); rows = []
    while len(rows) < K:
        q, _ = np.linalg.qr(r.standard_normal((16, 16))); rows.extend(q.T)
    A = np.array(rows[:K]); pj = (tr-mu)@A.T
    qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)
    f = lambda X: (((X-mu)@A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(gal), f(prb)

def eer_from_counts(gh, ih):
    # gh, ih: histograms over H=0..128. EER via cumulative (accept if H<=t).
    G = gh.sum(); I = ih.sum(); cg = np.cumsum(gh); ci = np.cumsum(ih)
    frr = 1 - cg/G; far = ci/I
    k = np.argmin(np.abs(frr-far)); return (frr[k]+far[k])/2

def boot(H, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    gen = np.diag(H).astype(int)
    pt = eer_from_counts(np.bincount(gen, minlength=Dc+1),
                         np.bincount(H[~np.eye(n, dtype=bool)].astype(int), minlength=Dc+1))
    outs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        g = gen[idx]
        rows = H[idx]                       # (n, n)
        im = rows.copy()
        # remove each resampled probe's own-gallery entry (genuine) from impostor pool
        im[np.arange(n), idx] = -1
        imv = im[im >= 0]
        outs.append(eer_from_counts(np.bincount(g, minlength=Dc+1), np.bincount(imv, minlength=Dc+1)))
    return pt, float(np.percentile(outs, 2.5)), float(np.percentile(outs, 97.5))

res = {}
for name, (cg, cp) in [("Super-Bit", superbit()), ("ortho-thermo", orthothermo())]:
    H = hamm(cp, cg); pt, lo, hi = boot(H)
    res[name] = {"eer": pt, "ci95": [lo, hi]}
    print(f"{name:14} EER {pt*100:.2f}%  95% CI [{lo*100:.2f}, {hi*100:.2f}]")
with open(f"{OUT}/bridge-eer-ci.json", "w") as f: json.dump(res, f, indent=2)
print(f"[ci] wrote {OUT}/bridge-eer-ci.json")
