"""
Step 22: Fragile/reliable-bit DIAGNOSTIC for the ortho-thermometer 128-bit code.

Purpose (plan round-2, step 1 — prerequisite for the quantization-aware head):
quantify *why* the 128-bit code (EER 0.79%) sits so far above the float ceiling (0.33%).
The hypothesis is that the code wastes most of its 128 bits: impostor Hamming sigma is
~14.6 vs a uniform code's 5.66, implying only a fraction of the bits are effectively
independent. This script MEASURES that on the held-out embeddings and tests whether a
cheap fragile-bit reweighting / pruning already recovers some accuracy (the Config-B
drop-in baseline the learned head must beat).

Metrics reported (all [MEASURED] on src/data/dim_ablation_emb_224.npz):
  - per-bit genuine flip-rate  f_g[b]  = P(bit differs across a genuine pair)   -> low = reliable
  - per-bit balance            p1[b]   = P(bit = 1)                             -> ~0.5 = balanced
  - per-bit impostor flip-rate f_i[b]  = P(bit differs across impostors)        -> ~0.5 ideal
  - per-bit discriminability   d[b]    = f_i[b] - f_g[b]                        -> large = useful
  - effective #bits (Daugman)  N_eff   = p(1-p)/sigma_frac^2 on impostor HD
  - effective DOF (participation ratio of the bit-correlation eigenspectrum)
  - EER of: full 128-bit Hamming; discriminability-weighted Hamming; top-k pruned Hamming

No GPU, no training, no crypto backend change (a reweighted/pruned Hamming stays expressible
through FLPSI's w-of-d subsampling knob). Run: python3 22_fragile_bits.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results"))
Dc = 128


def orthothermo(tr, gal, prb, K=16, Bp=8, seed=1):
    """Canonical ortho-thermometer (matches 21_bridge_ci.py): orthonormal QR projection
    blocks + per-projection magnitude quantile thresholds -> K*Bp = 128 bit code."""
    r = np.random.default_rng(seed); rows = []
    while len(rows) < K:
        q, _ = np.linalg.qr(r.standard_normal((tr.shape[1], tr.shape[1]))); rows.extend(q.T)
    A = np.array(rows[:K]); mu = tr.mean(0); pj = (tr - mu) @ A.T
    qs = np.quantile(pj, [i / (Bp + 1) for i in range(1, Bp + 1)], axis=0)
    f = lambda X: (((X - mu) @ A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K * Bp).astype(np.uint8)
    return f(gal), f(prb)


def wham(A, B, w):
    """Weighted Hamming distance matrix: sum_b w[b]*(A[i,b] XOR B[j,b])."""
    A = A.astype(np.float64); B = B.astype(np.float64); w = w.astype(np.float64)
    return (A * w) @ (1 - B).T + ((1 - A) * w) @ B.T


def eer(gen, imp):
    """Exact EER via searchsorted (accept if distance <= t)."""
    sg = np.sort(gen); si = np.sort(imp)
    ng, ni = len(sg), len(si)
    cand = np.unique(np.concatenate([sg, si]))
    frr = 1.0 - np.searchsorted(sg, cand, "right") / ng
    far = np.searchsorted(si, cand, "right") / ni
    k = int(np.argmin(np.abs(frr - far)))
    return float((frr[k] + far[k]) / 2.0)


def main():
    z = np.load(f"{DATA}/dim_ablation_emb_224.npz", allow_pickle=True)
    gal, prb, tr = z["gal"].astype(np.float64), z["prb"].astype(np.float64), z["train"].astype(np.float64)
    n = len(gal)
    cg, cp = orthothermo(tr, gal, prb)                 # (n,128) uint8 codes for gallery / probe
    off = ~np.eye(n, dtype=bool)

    # ---- baseline full-128 Hamming HD distributions ----
    full = wham(cp, cg, np.ones(Dc))
    genHD = np.diag(full); impHD = full[off]
    imp_mean, imp_sd = float(impHD.mean()), float(impHD.std())
    eer_full = eer(genHD, impHD)

    # ---- per-bit statistics ----
    f_g = (cg != cp).mean(0)                            # genuine flip-rate per bit
    pg1, pp1 = cg.mean(0), cp.mean(0)
    p1 = (pg1 + pp1) / 2.0                              # balance
    f_i = pg1 * (1 - pp1) + (1 - pg1) * pp1             # impostor flip-rate (independent identities)
    disc = f_i - f_g                                    # discriminability

    # ---- effective #bits: Daugman N = p(1-p)/sigma^2 on impostor NORMALISED HD ----
    p = imp_mean / Dc; sfrac = imp_sd / Dc
    n_eff_daugman = float(p * (1 - p) / sfrac ** 2)

    # ---- effective DOF: participation ratio of the bit correlation eigenspectrum ----
    allc = np.vstack([cg, cp]).astype(np.float64)
    allc = allc[:, allc.std(0) > 1e-9]                  # drop dead bits before correlation
    C = np.corrcoef(allc, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    n_eff_pr = float(ev.sum() ** 2 / (ev ** 2).sum())

    # ---- cheap reweighting / pruning tests (Config-B baseline) ----
    w_disc = np.clip(disc, 0, None)                     # discriminability-weighted Hamming
    wd = wham(cp, cg, w_disc); eer_wdisc = eer(np.diag(wd), wd[off])
    order = np.argsort(-disc)                           # rank bits by usefulness
    prune = {}
    for k in (Dc, 96, 64, 48, 32, 24, 16, 12, 10, 8):
        m = np.zeros(Dc); m[order[:k]] = 1.0
        M = wham(cp, cg, m); prune[k] = eer(np.diag(M), M[off])

    res = {
        "code": "ortho-thermometer-128", "n_ids": n,
        "eer_full128": eer_full, "imp_mean": imp_mean, "imp_sd": imp_sd,
        "gen_mean": float(genHD.mean()),
        "n_eff_daugman": n_eff_daugman, "n_eff_participation_ratio": n_eff_pr,
        "sigma_uniform_ref": float(np.sqrt(Dc) / 2),   # 5.657 for d=128
        "bit_genuine_flip_mean": float(f_g.mean()), "bit_genuine_flip_max": float(f_g.max()),
        "bit_balance_mean": float(p1.mean()),
        "bit_disc_mean": float(disc.mean()), "bit_disc_min": float(disc.min()), "bit_disc_max": float(disc.max()),
        "eer_disc_weighted": eer_wdisc,
        "eer_pruned_topk": {str(k): v for k, v in prune.items()},
    }
    print(f"[fragile] ortho-thermometer 128-bit, {n} held-out identities")
    print(f"  EER full-128            {eer_full*100:.2f}%   (float ceiling 0.33%)")
    print(f"  impostor HD  mean {imp_mean:.2f}  sd {imp_sd:.2f}  (uniform sd {np.sqrt(Dc)/2:.2f})")
    print(f"  effective #bits  Daugman {n_eff_daugman:.1f} | participation-ratio {n_eff_pr:.1f}  (of 128)")
    print(f"  per-bit genuine flip  mean {f_g.mean()*100:.1f}%  max {f_g.max()*100:.1f}%   balance mean {p1.mean()*100:.1f}%")
    print(f"  per-bit discriminability  mean {disc.mean()*100:.1f}%  min {disc.min()*100:.1f}%  max {disc.max()*100:.1f}%")
    print(f"  EER discriminability-weighted  {eer_wdisc*100:.2f}%")
    print(f"  EER pruned to top-k bits:")
    for k, v in prune.items():
        print(f"      k={k:>3}  {v*100:.2f}%")
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/fragile-bits.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[fragile] wrote {OUT}/fragile-bits.json")


if __name__ == "__main__":
    main()
