"""
Step 17: multi-finger fusion — Super-Bit vs L2-thermometer bridge.

Reuses the EXACT validated flash-psi accept model from 11_operating_point.py
(subsample_match_probs + accept_matrix + Poisson-binomial over real subject pairs).
Question: does the metric-aware bridge's thinner impostor tail (sd 14 vs 20) make a
config that is simultaneously SECURE (query-FAR@5000 low) and USABLE (low FRR)?

Codes from cached embeddings; labels from sb_codes_224.npz (order-matched). No GPU.
"""
import json, os
from math import comb, lgamma, log, log1p
import numpy as np
from quantizer import NeuralPSIQuantizer, QuantizerConfig

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results"))
D, T = 128, 64


def subsample_match_probs(w, t, d=D):
    denom = comb(d, w); out = np.empty(d+1)
    for H in range(d+1):
        s = sum(comb(H, i) * comb(d-H, w-i) for i in range(0, min(t, H, w)+1))
        out[H] = s/denom
    return out

def accept_matrix(p, T_sub=T):
    j = np.arange(T_sub+1)
    logC = np.array([lgamma(T_sub+1)-lgamma(k+1)-lgamma(T_sub-k+1) for k in j])
    out = np.zeros((len(p), T_sub+1))
    for h, ph in enumerate(p):
        if ph <= 0: pmf = np.zeros(T_sub+1); pmf[0] = 1.0
        elif ph >= 1: pmf = np.zeros(T_sub+1); pmf[T_sub] = 1.0
        else: pmf = np.exp(logC + j*log(ph) + (T_sub-j)*log1p(-ph))
        out[h] = np.concatenate([[1.0], np.cumsum(pmf[::-1])[::-1][1:]])
    return np.clip(out, 0, 1)

def poisson_binom_ge(P, k):
    K = P.shape[1]; dist = np.zeros((len(P), K+1)); dist[:, 0] = 1.0
    for m in range(K):
        p = P[:, m:m+1]; nxt = dist*(1-p); nxt[:, 1:] += dist[:, :-1]*p; dist = nxt
    return dist[:, k:].sum(1)

def hamm(A, B):
    A = A.astype(np.int32); B = B.astype(np.int32)
    return A @ (1-B).T + (1-A) @ B.T

def superbit(tr, g, p):
    q = NeuralPSIQuantizer.fit(tr, np.ones(tr.shape[1]),
                               QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    return q.transform(g).astype(np.uint8), q.transform(p).astype(np.uint8)

def thermo(tr, g, p, K=16, Bp=8, seed=1):
    # ortho-thermometer: orthonormal projection blocks + magnitude thresholds
    r = np.random.default_rng(seed); din = tr.shape[1]; rows = []
    while len(rows) < K:
        q, _ = np.linalg.qr(r.standard_normal((din, din))); rows.extend(q.T)
    A = np.array(rows[:K]); mu = tr.mean(0)
    pj = (tr-mu) @ A.T; qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)
    f = lambda X: (((X-mu) @ A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(g), f(p)


def fusion_frontier(name, cg, cp, fingers, N=5000, K=3):
    Hpg = hamm(cp, cg); gen_H = np.diag(Hpg).astype(int)
    subj = {}
    for idx, fp in enumerate(fingers):
        subj.setdefault(str(fp).split("_")[0], []).append(idx)
    usable = sorted(s for s, v in subj.items() if len(v) >= K)
    fidx = {s: subj[s][:K] for s in usable}
    keys = usable
    # Precompute Hamming-distance INDICES once (they don't change with w,t,theta).
    gen_idx = np.array([[gen_H[i] for i in fidx[s]] for s in keys])                 # (S, K)
    imp_idx = np.array([[Hpg[fidx[keys[a]][m], fidx[keys[b]][m]] for m in range(K)]
                        for a in range(len(keys)) for b in range(len(keys)) if a != b])  # (P, K)
    best = {k: None for k in (2, 3)}
    for w in [8, 10, 12, 14, 16, 20, 24, 28, 32, 40]:
        for t in [0, 1, 2]:
            A = accept_matrix(subsample_match_probs(w, t))
            for th in range(1, T+1):
                acc = A[:, th]
                pg = acc[gen_idx]; pi = acc[imp_idx]
                for k in (2, 3):
                    frr = float(1 - poisson_binom_ge(pg, k).mean())
                    far = float(poisson_binom_ge(pi, k).mean())
                    fq = 1-(1-far)**N
                    if fq <= 1e-2 and (best[k] is None or frr < best[k]["frr"]):
                        best[k] = dict(w=w, t=t, theta=th, k=k, K=K, frr=frr, far=far, far_query=fq)
    print(f"[{name}] {len(keys)} subjects >= {K} fingers")
    for k in (2, 3):
        b = best[k]
        if b: print(f"  {k}-of-{K}: w={b['w']} t={b['t']} theta={b['theta']}  per-rec FAR {b['far']:.2e}  "
                    f"query-FAR@{N} {b['far_query']:.2e}  FRR {b['frr']*100:.1f}%")
        else: print(f"  {k}-of-{K}: no config reaches query-FAR@{N}<=1e-2")
    return {"name": name, "n_subj": len(keys), "best": best}


def main():
    z = np.load(os.path.join(DATA, "dim_ablation_emb_224.npz"), allow_pickle=True)
    gal, prb, tr = z["gal"].astype(np.float64), z["prb"].astype(np.float64), z["train"].astype(np.float64)
    fingers = np.load(os.path.join(DATA, "sb_codes_224.npz"), allow_pickle=True)["fingers"]
    out = {}
    print("Goal: min-FRR config reaching query-FAR@5000 <= 1e-2 (secure AND usable)\n")
    for name, (cg, cp) in [("Super-Bit", superbit(tr, gal, prb)), ("L2-thermo", thermo(tr, gal, prb))]:
        out[name] = fusion_frontier(name, cg, cp, fingers)
    with open(os.path.join(OUT, "bridge-fusion.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[fusion] wrote {OUT}/bridge-fusion.json")
    for k in (2, 3):
        sb = out["Super-Bit"]["best"][k]; th = out["L2-thermo"]["best"][k]
        if sb and th:
            print(f"  {k}-of-3 secure point: FRR  Super-Bit {sb['frr']*100:.1f}%  vs  L2-thermo {th['frr']*100:.1f}%")


if __name__ == "__main__":
    main()
