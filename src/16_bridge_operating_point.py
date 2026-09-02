"""
Step 16: operating-point comparison, Super-Bit vs L2-thermometer bridge.

Exact closed-form FAR/FRR over the full held-out impostor set (1200x1199) via Hamming
histograms + the flash-psi accept model accept(H)=P[Binom(T, q(H,w)) >= t]. Finds each
bridge's OWN best (w,t) operating point (fair comparison, not Super-Bit's point applied to
both), the per-record FAR at matched genuine acceptance, the 1:N query-FAR, and the
collision floor. This is the "does the metric-aware bridge actually improve the operating
point?" measurement.

Codes built from cached embeddings (dim_ablation_emb_224.npz) -> no GPU needed.
"""
import argparse, json, os
from math import comb
import numpy as np
from scipy import stats
from quantizer import NeuralPSIQuantizer, QuantizerConfig

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results"))
D, T = 128, 64


def q_clean(H, w, d=D):
    H = int(H); return 0.0 if d - H < w else comb(d - H, w) / comb(d, w)

def binom_tail_ge(n, k, p):
    if k <= 0: return 1.0
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    return float(sum(comb(n, j) * p**j * (1-p)**(n-j) for j in range(k, n+1)))

# accept(H) cache per (w,t)
def accept_vec(w, t):
    return np.array([binom_tail_ge(T, t, q_clean(H, w)) for H in range(D+1)])


def hamm(A, B):
    A = A.astype(np.int32); B = B.astype(np.int32)
    return A @ (1-B).T + (1-A) @ B.T

def histograms(cg, cp):
    H = hamm(cp, cg); n = len(cg); eye = np.eye(n, dtype=bool)
    gen = H[eye].astype(int); imp = H[~eye].astype(int)
    gh = np.bincount(gen, minlength=D+1).astype(float); gh /= gh.sum()
    ih = np.bincount(imp, minlength=D+1).astype(float); ih /= ih.sum()
    return gh, ih, gen, imp

def superbit(tr, g, p):
    q = NeuralPSIQuantizer.fit(tr, np.ones(tr.shape[1]),
                               QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    return q.transform(g).astype(np.uint8), q.transform(p).astype(np.uint8)

def thermo(tr, g, p, K=16, Bp=8, seed=1):
    # ortho-thermometer: orthonormal projection blocks (Super-Bit variance reduction) + magnitude thresholds
    r = np.random.default_rng(seed); din = tr.shape[1]; rows = []
    while len(rows) < K:
        q, _ = np.linalg.qr(r.standard_normal((din, din))); rows.extend(q.T)
    A = np.array(rows[:K]); mu = tr.mean(0)
    pj = (tr-mu) @ A.T; qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)
    f = lambda X: (((X-mu) @ A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(g), f(p)

def poisson_ci(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.chi2.ppf(a/2, 2*k)/2
    return lo/n, stats.chi2.ppf(1-a/2, 2*(k+1))/2/n


def analyze(name, gh, ih, gen, imp):
    ws = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40]; ts = [1, 2, 3, 4]
    grid = []
    for w in ws:
        for t in ts:
            av = accept_vec(w, t)
            far = float((ih * av).sum()); tar = float((gh * av).sum())
            grid.append((w, t, far, tar))
    # best FAR subject to TAR>=target
    def best_at(tar_min):
        feas = [(far, w, t, tar) for (w, t, far, tar) in grid if tar >= tar_min]
        return min(feas) if feas else None
    coll = int((imp == 0).sum()); lo, hi = poisson_ci(coll, imp.size)
    res = {"name": name, "n_imp": int(imp.size), "eer_note": "see 14/07",
           "imp_mean": float(imp.mean()), "imp_sd": float(imp.std()),
           "gen_mean": float(gen.mean()), "collisions": coll, "p0": coll/imp.size, "p0_ci": [lo, hi]}
    for tgt in [0.99, 0.95, 0.90]:
        b = best_at(tgt)
        if b:
            far, w, t, tar = b
            fq = {N: 1-(1-far)**N for N in [100, 1000, 5000]}
            res[f"tar>={tgt}"] = {"w": w, "t": t, "far": far, "tar": tar, "far_query": fq}
    return res


def main():
    z = np.load(os.path.join(DATA, "dim_ablation_emb_224.npz"), allow_pickle=True)
    gal, prb, tr = z["gal"].astype(np.float64), z["prb"].astype(np.float64), z["train"].astype(np.float64)
    out = {}
    print(f"{'bridge':<12}{'op(w,t)':>9}{'FAR/rec':>11}{'TAR':>8}{'FARq@5k':>11}{'impSD':>7}{'H=0':>5}")
    for name, (cg, cp) in [("Super-Bit", superbit(tr, gal, prb)), ("L2-thermo", thermo(tr, gal, prb))]:
        gh, ih, gen, imp = histograms(cg, cp)
        r = analyze(name, gh, ih, gen, imp); out[name] = r
        for tgt in ["tar>=0.99", "tar>=0.95"]:
            if tgt in r:
                d = r[tgt]
                print(f"{name:<12}{'('+str(d['w'])+','+str(d['t'])+')':>9}{d['far']:>11.2e}{d['tar']*100:>7.1f}%"
                      f"{d['far_query'][5000]:>11.2e}{r['imp_sd']:>7.1f}{r['collisions']:>5}   [{tgt}]")
    with open(os.path.join(OUT, "bridge-operating-point.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[op] wrote {OUT}/bridge-operating-point.json")
    # headline delta
    for tgt in ["tar>=0.99", "tar>=0.95"]:
        if tgt in out["Super-Bit"] and tgt in out["L2-thermo"]:
            sb = out["Super-Bit"][tgt]["far"]; th = out["L2-thermo"][tgt]["far"]
            print(f"  {tgt}: per-record FAR  Super-Bit {sb:.2e}  vs  L2-thermo {th:.2e}  "
                  f"=> {sb/th:.1f}x lower" if th > 0 else f"  {tgt}: thermo FAR=0")


if __name__ == "__main__":
    main()
