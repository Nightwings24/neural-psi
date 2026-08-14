"""
Step 12: what the 128-bit code actually contains, and how it degrades.

Step 11 shows the impostor Hamming distribution has 3.6x the standard deviation the
FLPSI error analysis assumes, and that this alone breaks the published operating point.
This script establishes *why*, and puts confidence intervals on the headline numbers.

  (a) BIT DEPENDENCE. The FLPSI analysis needs sub-sample outcomes to behave as if the
      d bit positions were exchangeable and near-independent. Super-Bit's whole advantage
      comes from orthogonalising its projections -- that is, from making bits *dependent*.
      We measure the inter-bit correlation directly, and compare the empirical
      q-hat(H) = P[a random w-subset of positions is error-free | total distance H]
      against the hypergeometric C(d-H,w)/C(d,w) the analysis assumes.

  (b) EFFECTIVE DIMENSION. 128 bits from a 16-D embedding cannot carry 128 bits of
      information. We report the eigenspectrum of the bit correlation matrix and the
      participation ratio, which is the honest statement of how many independent
      binary decisions the code really encodes.

  (c) BIT-BUDGET-MATCHED BASELINE. Comparing 128-bit Super-Bit against 16-bit ITQ is a
      bit-budget mismatch. We fit ITQ at 128 bits and compare like for like.

  (d) DEGRADATION AND ERROR BARS. EER on Altered-Easy / Medium / Hard, each with a
      95% confidence interval bootstrapped over IDENTITIES rather than pairs, because
      pairs sharing a finger are not independent observations.
"""
import argparse
import json
import os

import numpy as np
import torch

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.abspath(os.path.join(HERE, "..", "docs", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "paper", "figures"))
D = 128


def embed_all(model, x, batch=256):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.asarray(x[i:i + batch]).astype(np.float32) / 255.0)
            out.append(model(xb.unsqueeze(1).to(dev)).cpu().numpy())
    return np.concatenate(out)


def fit_itq(emb, bits, iters=60, seed=0):
    """Iterative Quantization (Gong & Lazebnik). PCA to `bits` dims, then rotate to
    minimise quantisation loss. When bits > dim the PCA basis is padded with random
    orthogonal directions, which is the standard way to run ITQ above the data rank."""
    rng = np.random.default_rng(seed)
    mu = emb.mean(0)
    X = emb - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    dim = Vt.shape[0]
    W = Vt[:min(bits, dim)].T                       # (dim, k)
    if bits > dim:                                  # pad beyond the data rank
        W = np.concatenate([W, rng.standard_normal((dim, bits - dim))], axis=1)
    V = X @ W
    R = np.linalg.qr(rng.standard_normal((bits, bits)))[0]
    for _ in range(iters):
        B = np.sign(V @ R); B[B == 0] = 1
        Uu, _, Vv = np.linalg.svd(B.T @ V)
        R = (Uu @ Vv).T
    return mu, W, R


def itq_codes(emb, mu, W, R):
    return ((emb - mu) @ W @ R > 0).astype(np.uint8)


def eer(gen, imp):
    th = np.unique(np.concatenate([gen, imp]))
    best = (2.0, 0.5)
    for t in th:
        frr = float(np.mean(gen > t)); far = float(np.mean(imp <= t))
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2.0)
    return best[1]


def eer_ci(gen, score_mat, n=400, seed=0):
    """95% CI by bootstrapping over IDENTITIES, not pairs.

    Pairs sharing a finger are not independent, so resampling pairs understates the
    interval. We resample identities with replacement and rebuild both score sets from
    the same draw. A pair is impostor iff the two draws are different *identities* --
    checking positions instead would mislabel an identity drawn twice as an impostor.
    """
    rng = np.random.default_rng(seed)
    m = len(gen)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        sub = score_mat[np.ix_(idx, idx)]
        off = idx[:, None] != idx[None, :]
        vals.append(eer(gen[idx], sub[off]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(A, B, block=512):
    out = np.empty((len(A), len(B)), dtype=np.int16)
    for i in range(0, len(A), block):
        out[i:i + block] = _POPCNT[np.bitwise_xor(A[i:i + block, None, :],
                                                  B[None, :, :])].sum(axis=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--ckpt", default="feature_model_224.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureModel(img=args.img).to(dev)
    model.load_state_dict(torch.load(f"{DATA}/{args.ckpt}", map_location=dev))

    x_real = np.load(f"{DATA}/x_real_{args.img}.npy", mmap_mode="r")
    id_real = np.load(f"{DATA}/ids_real_{args.img}.npy", allow_pickle=True)
    test = set(np.load(f"{DATA}/ids_test_{args.img}.npy", allow_pickle=True).tolist())

    tr_idx = np.array([k for k, i in enumerate(id_real) if i not in test])
    emb_train = embed_all(model, x_real[tr_idx])
    quant = NeuralPSIQuantizer.fit(
        emb_train, np.ones(16),
        QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    print(f"[code] quantiser + ITQ fitted on {len(tr_idx)} train images (disjoint from test)")

    real_by = {i: k for k, i in enumerate(id_real) if i in test}
    out = {}

    # ---------------------------------------------------------------- (a) + (b)
    # Bit statistics are measured on the gallery codes of held-out identities.
    common = sorted(real_by)
    emb_gal = embed_all(model, x_real[[real_by[i] for i in common]])
    C = quant.transform(emb_gal).astype(np.float64)          # (n, 128) in {0,1}
    n = len(C)
    print(f"[code] {n} held-out gallery codes")

    bal = C.mean(0)
    Z = C - bal
    sd = Z.std(0)
    corr = (Z.T @ Z) / n / np.outer(sd, sd)
    offdiag = corr[~np.eye(D, dtype=bool)]
    ev = np.linalg.eigvalsh(corr)[::-1]
    part_ratio = float(ev.sum() ** 2 / (ev ** 2).sum())      # participation ratio
    ev_frac = np.cumsum(ev) / ev.sum()
    n90 = int(np.searchsorted(ev_frac, 0.90) + 1)

    out["bits"] = dict(
        balance_min=float(bal.min()), balance_max=float(bal.max()),
        mean_abs_corr=float(np.abs(offdiag).mean()),
        max_abs_corr=float(np.abs(offdiag).max()),
        frac_corr_gt_0p3=float((np.abs(offdiag) > 0.3).mean()),
        participation_ratio=part_ratio, n_eig_90pct=n90,
        top_eigs=[float(v) for v in ev[:8]])
    print(f"[code] bit balance in [{bal.min():.3f}, {bal.max():.3f}]  "
          f"(0.5 = perfectly balanced)")
    print(f"[code] inter-bit |correlation|: mean {np.abs(offdiag).mean():.3f}  "
          f"max {np.abs(offdiag).max():.3f}  "
          f"{(np.abs(offdiag) > 0.3).mean()*100:.1f}% of pairs exceed 0.3")
    print(f"[code] participation ratio {part_ratio:.1f} of {D} bits; "
          f"{n90} eigenvalues carry 90% of the variance")

    # empirical q-hat(H) vs the hypergeometric the FLPSI analysis assumes
    from math import comb
    W_TEST = 14
    packed = np.packbits(C.astype(np.uint8), axis=1)
    Hm = hamming_matrix(packed, packed)
    iu = np.triu_indices(n, 1)
    Hvals = Hm[iu]
    xor = (C[iu[0]] != C[iu[1]])                             # (pairs, 128) mismatch mask
    qs = []
    for H in range(1, 41):
        sel = np.flatnonzero(Hvals == H)
        if len(sel) < 40:
            continue
        pick = sel if len(sel) <= 400 else rng.choice(sel, 400, replace=False)
        # Monte-Carlo over sub-sample draws for these real pairs
        hits = 0; trials = 0
        for p in pick:
            mask = xor[p]
            for _ in range(200):
                pos = rng.choice(D, W_TEST, replace=False)
                hits += int(not mask[pos].any()); trials += 1
        theo = comb(D - H, W_TEST) / comb(D, W_TEST)
        qs.append(dict(H=int(H), n_pairs=int(len(sel)), q_emp=hits / trials, q_theory=theo,
                       ratio=(hits / trials) / theo if theo > 0 else float("nan")))
    out["q_hat"] = qs
    if qs:
        r = np.array([q["ratio"] for q in qs])
        print(f"[code] empirical q(H) / hypergeometric q(H): "
              f"median {np.median(r):.3f}  range [{r.min():.3f}, {r.max():.3f}]")

    # ---------------------------------------------------------------- (c) + (d)
    itq_fits = {b: fit_itq(emb_train, b, seed=args.seed) for b in (16, 128)}
    sb_gal = np.packbits(quant.transform(emb_gal).astype(np.uint8), axis=1)
    itq_gal = {b: np.packbits(itq_codes(emb_gal, *itq_fits[b]), axis=1)
               for b in itq_fits}

    levels = [("Altered-Easy", ""), ("Altered-Medium", "_med"), ("Altered-Hard", "_hard")]
    rows = []
    for name, sfx in levels:
        xp = f"{DATA}/x_probe_{args.img}{sfx}.npy"
        ip = f"{DATA}/ids_probe_{args.img}{sfx}.npy"
        if not os.path.exists(xp):
            print(f"[code] {name}: arrays absent ({os.path.basename(xp)}), skipped")
            continue
        x_prb = np.load(xp, mmap_mode="r")
        id_prb = np.load(ip, allow_pickle=True)
        prb_by = {i: k for k, i in enumerate(id_prb) if i in test}
        ids = sorted(set(real_by) & set(prb_by))
        gi = np.array([common.index(i) for i in ids])
        emb_p = embed_all(model, x_prb[[prb_by[i] for i in ids]])

        for label, gal_codes, prb_codes in [
                ("Super-Bit 128", sb_gal[gi],
                 np.packbits(quant.transform(emb_p).astype(np.uint8), axis=1)),
                ("ITQ 16", itq_gal[16][gi],
                 np.packbits(itq_codes(emb_p, *itq_fits[16]), axis=1)),
                ("ITQ 128", itq_gal[128][gi],
                 np.packbits(itq_codes(emb_p, *itq_fits[128]), axis=1))]:
            Hpg = hamming_matrix(prb_codes, gal_codes).astype(float)
            m = len(ids); e = np.eye(m, dtype=bool)
            g, im = Hpg[e], Hpg[~e]
            v = eer(g, im)
            lo, hi = eer_ci(g, Hpg)
            rows.append(dict(level=name, coder=label, n_ids=m,
                             eer=v, ci_lo=lo, ci_hi=hi,
                             gen_mean=float(g.mean()), imp_mean=float(im.mean()),
                             imp_sd=float(im.std()),
                             collisions=int((im == 0).sum()), n_imp=int(im.size)))
            print(f"[code] {name:15} {label:14} EER {v*100:6.2f}% "
                  f"[{lo*100:5.2f}, {hi*100:5.2f}]  "
                  f"H gen {g.mean():5.1f} imp {im.mean():5.1f} (sd {im.std():5.2f})  "
                  f"H=0 collisions {int((im == 0).sum())}/{im.size:,}")
    out["eer"] = rows

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "code-analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[code] wrote {RESULTS}/code-analysis.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIGS, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
        im0 = ax[0].imshow(corr, cmap="RdBu_r", vmin=-.5, vmax=.5)
        ax[0].set_title(f"Inter-bit correlation (mean $|\\rho|$={np.abs(offdiag).mean():.2f})")
        ax[0].set_xlabel("bit"); ax[0].set_ylabel("bit")
        fig.colorbar(im0, ax=ax[0], fraction=.046)
        if qs:
            hh = [q["H"] for q in qs]
            ax[1].semilogy(hh, [q["q_theory"] for q in qs], "k--", label="hypergeometric (assumed)")
            ax[1].semilogy(hh, [q["q_emp"] for q in qs], "o-", ms=3, label="measured on real codes")
            ax[1].set_xlabel("Hamming distance $H$")
            ax[1].set_ylabel(f"$q(H)$, $w={W_TEST}$")
            ax[1].set_title("Sub-sample match probability"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{FIGS}/bit-dependence.pdf")
        print(f"[code] wrote {FIGS}/bit-dependence.pdf")
    except ImportError:
        print("[code] matplotlib unavailable; skipped figure")


if __name__ == "__main__":
    main()
