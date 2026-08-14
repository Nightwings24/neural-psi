"""
Step 11: measure the FLPSI operating point against the REAL code distribution.

The published parameter set (d=128, T=64, w=14, t=2, theta=2) was chosen against an
idealised model in which impostor codes are uniform, i.e. H ~ Binomial(128, 1/2) with
sigma = 5.66. Under that model the per-record false-accept rate is 2.95e-5.

Our codes are not uniform. They are 128 Super-Bit projections of a 16-dimensional
embedding, so they carry roughly 16 real degrees of freedom, and the impostor Hamming
distribution inherits the embedding's angular distribution -- including a heavy left tail,
because some fingers genuinely look alike. This script measures that distribution and
recomputes the entire error analysis from it.

What it produces:
  1. genuine / impostor Hamming histograms (exact, all held-out pairs)
  2. the measured-vs-idealised gap: mean, sigma, and the induced FAR ratio
  3. per-record FAR and FRR over a (w, t, theta) grid, computed EXACTLY from the
     histograms -- no resampling noise, since accept() depends only on H
  4. query-level FAR at database size N, which is what an attacker actually faces
  5. a resolution flag: an FAR estimate is untrustworthy if it is dominated by
     histogram bins holding fewer than MIN_BIN observed pairs
  6. the k-of-K multi-finger frontier, computed over REAL subject pairs via a
     Poisson-binomial, so correlation between a subject's own fingers is preserved
     rather than assumed away

Everything downstream of the histograms is closed-form, so the numbers are reproducible
to the last digit rather than Monte-Carlo estimates.
"""
import argparse
import json
import os
from math import comb, lgamma, log, log1p

import numpy as np
import torch

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.abspath(os.path.join(HERE, "..", "docs", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "paper", "figures"))

D = 128                 # code length
T_SUB = 64              # number of sub-samples
MIN_BIN = 10            # a bin with fewer pairs cannot support a tail estimate


# --------------------------------------------------------------------------- #
# accept probability, vectorised over H
# --------------------------------------------------------------------------- #
def subsample_match_probs(w: int, t: int, d: int = D) -> np.ndarray:
    """p(H) for H = 0..d: P[a random w-subset of d positions covers <= t of H mismatches].

    Exact integer combinatorics (hypergeometric tail), then one float division.
    """
    denom = comb(d, w)
    out = np.empty(d + 1)
    for H in range(d + 1):
        s = 0
        for i in range(0, min(t, H, w) + 1):
            s += comb(H, i) * comb(d - H, w - i)
        out[H] = s / denom
    return out


def accept_matrix(p: np.ndarray, T: int = T_SUB) -> np.ndarray:
    """accept[H, theta] = P[Binomial(T, p(H)) >= theta] for theta = 0..T.

    Computed in log space: p(H) spans many orders of magnitude and the naive
    product form underflows well before the tail stops mattering.
    """
    j = np.arange(T + 1)
    logC = np.array([lgamma(T + 1) - lgamma(k + 1) - lgamma(T - k + 1) for k in j])
    out = np.zeros((len(p), T + 1))
    for h, ph in enumerate(p):
        if ph <= 0.0:
            pmf = np.zeros(T + 1); pmf[0] = 1.0
        elif ph >= 1.0:
            pmf = np.zeros(T + 1); pmf[T] = 1.0
        else:
            pmf = np.exp(logC + j * log(ph) + (T - j) * log1p(-ph))
        # survival function P[X >= theta], theta = 0..T
        out[h] = np.concatenate([[1.0], np.cumsum(pmf[::-1])[::-1][1:]])
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# embedding / coding
# --------------------------------------------------------------------------- #
def embed_all(model, x, batch=256):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0)
            out.append(model(xb.unsqueeze(1).to(dev)).cpu().numpy())
    return np.concatenate(out)


def pack(codes: np.ndarray) -> np.ndarray:
    """(n, 128) {0,1} -> (n, 16) uint8, so Hamming is a popcount over bytes."""
    return np.packbits(codes.astype(np.uint8), axis=1)


_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(A: np.ndarray, B: np.ndarray, block: int = 512) -> np.ndarray:
    """All-pairs Hamming between packed code sets, in blocks to bound memory."""
    out = np.empty((len(A), len(B)), dtype=np.uint8)
    for i in range(0, len(A), block):
        chunk = np.bitwise_xor(A[i:i + block, None, :], B[None, :, :])
        out[i:i + block] = _POPCNT[chunk].sum(axis=2)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--ckpt", default="feature_model_224.pt")
    ap.add_argument("--n-db", type=int, default=5000, help="database size N for query-level FAR")
    ap.add_argument("--far-target", type=float, default=1e-3,
                    help="query-level FAR the corrected operating point must meet")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureModel(img=args.img).to(dev)
    model.load_state_dict(torch.load(f"{DATA}/{args.ckpt}", map_location=dev))

    x_real = np.load(f"{DATA}/x_real_{args.img}.npy", mmap_mode="r")
    x_prb = np.load(f"{DATA}/x_probe_{args.img}.npy", mmap_mode="r")
    id_real = np.load(f"{DATA}/ids_real_{args.img}.npy", allow_pickle=True)
    id_prb = np.load(f"{DATA}/ids_probe_{args.img}.npy", allow_pickle=True)
    test = set(np.load(f"{DATA}/ids_test_{args.img}.npy", allow_pickle=True).tolist())

    real_by = {i: k for k, i in enumerate(id_real) if i in test}
    prb_by = {i: k for k, i in enumerate(id_prb) if i in test}
    common = sorted(set(real_by) & set(prb_by))
    print(f"[op] {len(common)} held-out identities with both an enrolment and a probe")

    # quantiser is fitted on TRAIN identities only (never on anything scored below)
    tr_idx = np.array([k for k, i in enumerate(id_real) if i not in test])
    emb_train = embed_all(model, np.asarray(x_real[tr_idx]))
    quant = NeuralPSIQuantizer.fit(
        emb_train, np.ones(16),
        QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    print(f"[op] quantiser fitted on {len(tr_idx)} train images, disjoint from test")

    gal = pack(quant.transform(embed_all(model, np.asarray(x_real[[real_by[i] for i in common]]))))
    prb = pack(quant.transform(embed_all(model, np.asarray(x_prb[[prb_by[i] for i in common]]))))

    n = len(common)
    Hpg = hamming_matrix(prb, gal)                     # probe x gallery
    Hgg = hamming_matrix(gal, gal)                     # gallery x gallery
    eye = np.eye(n, dtype=bool)
    gen_H = Hpg[eye]
    imp_H = np.concatenate([Hpg[~eye].ravel(), Hgg[~eye].ravel()])
    print(f"[op] {len(gen_H):,} genuine and {len(imp_H):,} impostor pairs")

    gen_hist = np.bincount(gen_H.astype(int), minlength=D + 1).astype(float)
    imp_hist = np.bincount(imp_H.astype(int), minlength=D + 1).astype(float)
    gen_p, imp_p = gen_hist / gen_hist.sum(), imp_hist / imp_hist.sum()

    # ---- the measured-vs-idealised gap ---------------------------------------
    unif = np.array([comb(D, h) for h in range(D + 1)], dtype=float)
    unif /= unif.sum()                                  # H ~ Binomial(128, 1/2)
    meas = dict(mean=float(imp_H.mean()), sd=float(imp_H.std()),
                p01=float(np.percentile(imp_H, 0.1)), min=int(imp_H.min()))
    ideal = dict(mean=D / 2, sd=float(np.sqrt(D) / 2))
    print(f"[op] impostor H: measured mean {meas['mean']:.1f} sd {meas['sd']:.2f} "
          f"min {meas['min']}  |  uniform model mean {ideal['mean']:.1f} sd {ideal['sd']:.2f}")
    print(f"[op] genuine  H: mean {gen_H.mean():.1f} sd {gen_H.std():.2f} "
          f"max {int(gen_H.max())}")

    # ---- (w, t, theta) frontier ----------------------------------------------
    WS = [8, 10, 12, 14, 16, 20, 24, 32, 40, 48, 56, 64]
    TS = list(range(0, 9))
    grid, cache = [], {}
    for w in WS:
        for t in TS:
            if t >= w:
                continue
            A = cache.setdefault((w, t), accept_matrix(subsample_match_probs(w, t)))
            far = imp_p @ A                              # (T+1,) per-record FAR
            frr = 1.0 - (gen_p @ A)                      # (T+1,)
            # resolution: the share of FAR contributed by bins with < MIN_BIN pairs
            for th in range(1, T_SUB + 1):
                contrib = imp_p * A[:, th]
                thin = contrib[imp_hist < MIN_BIN].sum()
                share = thin / contrib.sum() if contrib.sum() > 0 else 1.0
                grid.append(dict(w=w, t=t, theta=th, far=float(far[th]),
                                 frr=float(frr[th]), thin_share=float(share)))
    print(f"[op] evaluated {len(grid):,} (w,t,theta) combinations")

    def q_far(f, N):                                    # query-level FAR over N records
        return -np.expm1(N * np.log1p(-f)) if f < 1 else 1.0

    for g in grid:
        g["far_query"] = float(q_far(g["far"], args.n_db))

    # the published point, and the idealisation it was chosen under
    pub = next(g for g in grid if (g["w"], g["t"], g["theta"]) == (14, 0, 2))
    A_pub = cache[(14, 0)]
    far_ideal = float(unif @ A_pub[:, 2])
    print(f"\n[op] published point w=14 theta=2 (error-free sub-samples, t=0):")
    print(f"       per-record FAR  measured {pub['far']:.3e}   uniform-model {far_ideal:.3e}"
          f"   ratio {pub['far']/far_ideal:,.0f}x")
    print(f"       FRR             {pub['frr']:.3e}")
    print(f"       query FAR @ N={args.n_db:,}  {pub['far_query']:.4f}")

    # ---- corrected operating point -------------------------------------------
    def best_at(target):
        ok = [g for g in grid if g["far_query"] <= target and g["thin_share"] < 0.5]
        return min(ok, key=lambda g: g["frr"]) if ok else None

    # ---- where does the operating point break? -------------------------------
    # Verification (1:1, N=1) and identification (1:N) are different problems on the
    # same parameters: FAR_query = 1-(1-FAR)^N grows with the database. This sweep
    # locates the N at which the published point stops being a security boundary.
    Ns = [1, 10, 100, 1_000, 5_000, 10_000, 100_000]
    n_sweep = [dict(N=N, far_query=float(q_far(pub["far"], N))) for N in Ns]
    print("\n[op] published point, query-level FAR vs database size:")
    for r in n_sweep:
        print(f"       N = {r['N']:>7,}   FAR_query = {r['far_query']:.4f}")
    n_break = next((r["N"] for r in n_sweep if r["far_query"] > 0.5), None)
    print(f"       FRR is {pub['frr']:.2%} throughout; FAR_query exceeds 50% at "
          f"N = {n_break:,}" if n_break else "       FAR_query stays below 50%")

    print(f"\n[op] best single-finger point per query-FAR target at N={args.n_db:,}:")
    ladder = []
    for tg in sorted({1e-1, 1e-2, args.far_target, 1e-4, 1e-6}, reverse=True):
        b = best_at(tg)
        ladder.append(dict(target=tg, point=b))
        if b:
            print(f"       FAR_query <= {tg:.0e} : w={b['w']:2} t={b['t']} theta={b['theta']:2}"
                  f"  FRR={b['frr']:.4f}  per-record FAR={b['far']:.3e}")
        else:
            print(f"       FAR_query <= {tg:.0e} : unreachable "
                  f"(no (w,t,theta) with usable histogram support)")
    best = best_at(args.far_target)

    # ---- multi-finger: k-of-K over REAL subject pairs -------------------------
    # SOCOFing test ids are "<subject>_<hand>_<finger>" (e.g. 547_right_thumb); group by
    # subject so correlation between one person's own fingers is measured, not assumed.
    subj = {}
    for k, i in enumerate(common):
        subj.setdefault(str(i).split("_")[0], []).append(k)
    usable = {s: v for s, v in subj.items() if len(v) >= 3}
    print(f"\n[op] {len(usable)} held-out subjects contribute >= 3 fingers "
          f"(of {len(subj)} subjects present)")

    fusion = []
    if len(usable) >= 2:
        K = 3
        keys = sorted(usable)
        fidx = {s: usable[s][:K] for s in keys}
        pts = [(14, 0, 2)] + ([(best["w"], best["t"], best["theta"])] if best else [(24, 0, 2)])
        for (w, t, th) in pts:
            A = cache.setdefault((w, t), accept_matrix(subsample_match_probs(w, t)))
            acc = A[:, th]
            # genuine: each subject's K fingers, probe vs own gallery
            pg = np.array([[acc[gen_H[k]] for k in fidx[s]] for s in keys])
            # impostor: subject a's probes against subject b's gallery, finger-aligned
            pi = []
            for a in range(len(keys)):
                for b in range(len(keys)):
                    if a == b:
                        continue
                    pi.append([acc[Hpg[fidx[keys[a]][m], fidx[keys[b]][m]]] for m in range(K)])
            pi = np.array(pi)

            def poisson_binom_ge(P, k):
                """P[at least k of row P's independent Bernoulli trials succeed], per row.

                The trials are the K per-finger accept events; their probabilities differ
                (each finger has its own Hamming distance), so this is a Poisson-binomial,
                not a binomial. Rows come from real subject pairs, so any correlation
                between one person's fingers is already carried by the probabilities.
                """
                K = P.shape[1]
                dist = np.zeros((len(P), K + 1)); dist[:, 0] = 1.0
                for m in range(K):
                    p = P[:, m:m + 1]
                    nxt = dist * (1.0 - p)
                    nxt[:, 1:] += dist[:, :-1] * p
                    dist = nxt
                return dist[:, k:].sum(axis=1)

            for k in range(1, K + 1):
                fusion.append(dict(w=w, t=t, theta=th, K=K, k=k,
                                   frr=float(1 - poisson_binom_ge(pg, k).mean()),
                                   far=float(poisson_binom_ge(pi, k).mean())))
        for f in fusion:
            f["far_query"] = float(q_far(f["far"], args.n_db))
            print(f"       w={f['w']:2} t={f['t']} theta={f['theta']}  {f['k']}-of-{f['K']}  "
                  f"FAR/record {f['far']:.3e}  query FAR@{args.n_db:,} {f['far_query']:.4f}  "
                  f"FRR {f['frr']:.4f}")

    # ---- persist --------------------------------------------------------------
    os.makedirs(RESULTS, exist_ok=True)
    # Histograms go in the npz (small, and everything else derives from them); the
    # 6,848-row grid goes to a gzipped CSV rather than a JSON blob inside the archive,
    # which is what made this file 4 MB.
    np.savez_compressed(os.path.join(RESULTS, "operating-point-histograms.npz"),
                        gen_hist=gen_hist, imp_hist=imp_hist)
    import csv, gzip
    with gzip.open(os.path.join(RESULTS, "operating-point-grid.csv.gz"), "wt",
                   newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(grid[0]))
        wr.writeheader(); wr.writerows(grid)
    with open(os.path.join(RESULTS, "operating-point.json"), "w") as f:
        json.dump(dict(n_identities=n, n_genuine=int(gen_hist.sum()),
                       n_impostor=int(imp_hist.sum()),
                       genuine=dict(mean=float(gen_H.mean()), sd=float(gen_H.std()),
                                    max=int(gen_H.max())),
                       impostor=meas, uniform_model=ideal,
                       published=dict(**{k: pub[k] for k in
                                         ("w", "t", "theta", "far", "frr", "far_query")},
                                      far_uniform_model=far_ideal,
                                      ratio=pub["far"] / far_ideal),
                       corrected=best, ladder=ladder, fusion=fusion,
                       n_sweep=n_sweep, n_break=n_break,
                       n_db=args.n_db, far_target=args.far_target), f, indent=2)
    print(f"\n[op] wrote {RESULTS}/operating-point.json")

    # ---- figures --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIGS, exist_ok=True)

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].bar(np.arange(D + 1), gen_p, width=1.0, alpha=.75, label="genuine")
        ax[0].bar(np.arange(D + 1), imp_p, width=1.0, alpha=.75, label="impostor")
        ax[0].plot(np.arange(D + 1), unif, "k--", lw=1.2,
                   label=r"uniform model $\mathrm{Bin}(128,\frac{1}{2})$")
        ax[0].set_xlabel("Hamming distance $H$"); ax[0].set_ylabel("probability")
        ax[0].set_title("Measured code distributions"); ax[0].legend(fontsize=8)

        ax[1].semilogy(np.arange(D + 1), np.maximum(imp_p, 1e-9), label="impostor (measured)")
        ax[1].semilogy(np.arange(D + 1), np.maximum(unif, 1e-9), "k--",
                       label="uniform model")
        ax[1].semilogy(np.arange(D + 1), np.maximum(gen_p, 1e-9), label="genuine")
        ax[1].axvline(pub["w"], color="grey", lw=.8)
        ax[1].set_xlabel("Hamming distance $H$"); ax[1].set_ylabel("probability (log)")
        ax[1].set_title("Left tail is what sets the FAR"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{FIGS}/hamming-histograms.pdf")

        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for w in WS:
            pts = sorted([g for g in grid if g["w"] == w and g["thin_share"] < 0.5],
                         key=lambda g: g["far"])
            if pts:
                ax.loglog([max(g["far"], 1e-12) for g in pts],
                          [max(g["frr"], 1e-6) for g in pts], marker=".", ms=3, label=f"w={w}")
        ax.scatter([pub["far"]], [max(pub["frr"], 1e-6)], marker="*", s=160, color="crimson",
                   zorder=5, label="published (14,2,2)")
        ax.set_xlabel("per-record FAR"); ax.set_ylabel("FRR")
        ax.set_title("DET frontier over $(w,t,\\theta)$"); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(f"{FIGS}/det-frontier.pdf")

        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        Ncurve = np.unique(np.round(np.logspace(0, 5.3, 90)).astype(int))
        ax.semilogx(Ncurve, [q_far(pub["far"], N) for N in Ncurve],
                    lw=2, label="published $(w{=}14,\\theta{=}2)$, 1 finger")
        for f in fusion:
            if f["k"] == f["K"]:
                ax.semilogx(Ncurve, [q_far(f["far"], N) for N in Ncurve], "--",
                            label=f"$w{{=}}{f['w']}$, {f['k']}-of-{f['K']} fingers")
        ax.axhline(0.5, color="grey", lw=.8, ls=":")
        ax.axvline(args.n_db, color="grey", lw=.8, ls=":")
        ax.set_xlabel("database size $N$"); ax.set_ylabel("query-level FAR")
        ax.set_title("A per-record FAR is not a security boundary")
        ax.legend(fontsize=7); ax.set_ylim(-0.03, 1.03)
        fig.tight_layout(); fig.savefig(f"{FIGS}/far-vs-dbsize.pdf")
        print(f"[op] wrote 3 figures to {FIGS}")
    except ImportError:
        print("[op] matplotlib unavailable; skipped figures")


if __name__ == "__main__":
    main()
