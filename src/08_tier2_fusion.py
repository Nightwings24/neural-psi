"""
Tier-2: re-tune the flash-psi operating point for the 224-model Super-Bit codes, and
evaluate MULTI-FINGER FUSION (k-of-K quorum) to crush the false-accept rate.

Two levers, both no-retrain:
  (1) OPERATING-POINT RE-TUNE. The 224 model gives tighter genuine Hamming (~8/128) than the
      96px model, so the old flash-psi point (weight=2, t=54) is wrong. We sweep (weight, t) at
      T=64 and report TAR/FAR, picking a genuine-safe, near-zero-FAR point.
  (2) MULTI-FINGER FUSION. Enrol K fingers per identity as K independent flash-psi matches and
      require a q-of-K quorum. Independent fingers => FAR drops ~ geometrically.

flash-psi decision model (faithful to flash-psi/src/subsample.rs and ayan's flpsi_match.py):
  one mask = `weight` random positions; a sub-sample "matches" iff the two codes agree on ALL of
  them (0 mismatches). Over T iid masks the clean-match count ~ Binomial(T, q(H)) with
  q(H) = C(d-H, weight) / C(d, weight); the parties ACCEPT iff that count >= t. So the expected
  per-pair accept probability is exact in closed form (no Monte-Carlo needed for the sweep).

Uses plain Super-Bit (center + Super-Bit projection + median), NO whitening - the Tier-1 winner.
GPU-accelerated embedding if available.
"""
import os
from math import comb

import numpy as np
import torch

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig, hamming
from flpsi_match import SubSampler

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
D = 128
T = 64
IMG = 224
CKPT = "feature_model_224.pt"
rng = np.random.default_rng(0)


# ---------- flash-psi closed-form accept probability ----------
def q_clean(H, weight, d=D):
    """P[a single weight-subset of d positions avoids all H mismatched bits] = C(d-H,w)/C(d,w)."""
    H = int(H)
    if d - H < weight:
        return 0.0
    return comb(d - H, weight) / comb(d, weight)


def binom_tail_ge(n, k, p):
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return float(sum(comb(n, j) * p**j * (1 - p)**(n - j) for j in range(k, n + 1)))


def accept_prob_vec(H_arr, weight, t, T_sub=T):
    """Expected accept prob per pair given its Hamming distance (vectorised over H_arr)."""
    # cache by integer H (distances are small ints)
    uniq = {int(h): binom_tail_ge(T_sub, t, q_clean(int(h), weight)) for h in set(int(x) for x in H_arr)}
    return np.array([uniq[int(h)] for h in H_arr])


def far_tar(gen_H, imp_H, weight, t):
    """TAR = mean accept over genuine pairs; FAR = mean accept over impostor pairs (expected)."""
    tar = float(accept_prob_vec(gen_H, weight, t).mean())
    far = float(accept_prob_vec(imp_H, weight, t).mean())
    return tar, far


# ---------- embedding + Super-Bit encoding ----------
def embed_all(model, x, batch=256):
    model.eval(); dev = next(model.parameters()).device; out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureModel(img=IMG).to(dev)
    model.load_state_dict(torch.load(f"{DATA}/{CKPT}", map_location=dev))
    print(f"[tier2] model={CKPT} on {dev}")

    def L(base):
        p = f"{DATA}/{base}_{IMG}.npy"
        return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{base}.npy", allow_pickle=True)
    x_real, ids_real = L("x_real"), L("ids_real")
    x_probe, ids_probe = L("x_probe"), L("ids_probe")
    ids_test = set(L("ids_test").tolist())

    # plain Super-Bit fit on TRAIN embeddings only (no leakage), encode held-out test fingers
    train_real = np.stack([x for i, x in zip(ids_real, x_real) if i not in ids_test])
    quant = NeuralPSIQuantizer.fit(embed_all(model, train_real), np.ones(16),
                                   QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
    common = sorted(set(real_by) & set(prb_by))
    enr = {i: quant.transform(embed_all(model, real_by[i][None])[0]) for i in common}   # Real enroll code
    qry = {i: quant.transform(embed_all(model, prb_by[i][None])[0]) for i in common}     # Altered probe code
    print(f"[tier2] held-out test fingers: {len(common)}")

    # single-finger genuine / impostor Hamming distributions
    fingers = common
    idx_of = {f: k for k, f in enumerate(fingers)}
    gen_H = np.array([int(hamming(qry[i], enr[i])) for i in fingers])
    imp_H, imp_pairs = [], []
    for i in fingers:
        others = rng.choice([k for k in range(len(fingers)) if k != idx_of[i]],
                            size=min(200, len(fingers) - 1), replace=False)
        for j in others:
            imp_H.append(int(hamming(qry[i], enr[fingers[j]]))); imp_pairs.append((i, fingers[j]))
    imp_H = np.array(imp_H)
    print(f"[tier2] genuine Hamming: mean {gen_H.mean():.1f} (max {gen_H.max()}) | "
          f"impostor: mean {imp_H.mean():.1f} (min {imp_H.min()})  [of {D}]")
    # diagnose the impostor tail (overlap with genuine drives single-finger FAR)
    overlap = int((imp_H <= gen_H.max()).sum())
    print(f"[tier2] impostor pairs with H <= genuine-max({gen_H.max()}): {overlap}/{len(imp_H)} "
          f"({overlap/len(imp_H)*100:.2f}%)  <- the tail overlap that floors single-finger FAR")
    lo = np.argsort(imp_H)[:5]
    print(f"[tier2] closest impostor pairs (probe vs different-finger enroll):")
    for k in lo:
        print(f"          H={imp_H[k]:>3}  {imp_pairs[k][0]}  vs  {imp_pairs[k][1]}")

    # ---------- (1) operating-point re-tune: t MUST stay small (Shamir cost), so sweep WEIGHT ----------
    print("\n===== (1) flash-psi operating-point re-tune - t fixed small, tune weight (T=64) =====")
    print(f"  {'weight':>6} {'t':>3} {'TAR%':>8} {'FAR':>12}")
    best = None
    for t in (2, 3):
        for weight in range(8, 41, 2):
            tar, far = far_tar(gen_H, imp_H, weight, t)
            print(f"  {weight:>6} {t:>3} {tar*100:>7.2f} {far:>12.2e}")
            # pick the lowest-FAR point that still keeps TAR >= 0.97 (fusion will recover TAR)
            if tar >= 0.97:
                score = (-far, tar)
                if best is None or score > best[0]:
                    best = (score, weight, t, tar, far)
        print("  " + "-" * 32)
    _, w_opt, t_opt, tar_opt, far_opt = best
    print(f"  >> CHOSEN single-finger point: weight={w_opt}, t={t_opt}, T={T}  ->  "
          f"TAR={tar_opt*100:.2f}%  FAR={far_opt:.2e}")
    print(f"     (old 96px point was weight=2, t=54 - wrong for the tighter 224 histograms)")

    # ---------- (2) multi-finger fusion ----------
    print("\n===== (2) multi-finger fusion (q-of-K), single-finger op-point above =====")
    p_tar, p_far = tar_opt, far_opt
    print(f"  single-finger: TAR={p_tar*100:.3f}%  FAR={p_far:.2e}")
    print(f"  {'K':>2} {'rule':>8} {'fused TAR%':>11} {'fused FAR':>12}")
    def fuse(p, K, q):
        return sum(comb(K, i) * p**i * (1 - p)**(K - i) for i in range(q, K + 1))
    rules = [(1, 1), (2, 2), (3, 2), (3, 3)]   # (K, q): single, 2-of-2(AND), 2-of-3(majority), 3-of-3
    for K, q in rules:
        name = "single" if K == 1 else (f"{q}-of-{K}")
        print(f"  {K:>2} {name:>8} {fuse(p_tar,K,q)*100:>10.3f} {fuse(p_far,K,q):>12.2e}")

    # ---------- empirical multi-finger validation (subject-grouped) ----------
    subj = {}
    for i in fingers:
        subj.setdefault(i.split("_")[0], []).append(i)
    multi = {s: fs for s, fs in subj.items() if len(fs) >= 2}
    print(f"\n  empirical check: {len(multi)} test subjects have >=2 held-out fingers")
    sampler = SubSampler(d=D, weight=w_opt, subsample_count=T, seed=1)
    def finger_match(qi, ej):
        return sampler.sub_matches(qry[qi], enr[ej]) >= t_opt
    # 2-of-2 genuine: a subject's 2 fingers both match their own enroll
    gen_ok = gen_tot = imp_ok = imp_tot = 0
    subs = list(multi)
    for s in subs:
        fs = multi[s][:2]
        gen_tot += 1
        if sum(finger_match(f, f) for f in fs) >= 2:
            gen_ok += 1
        # impostor person: same 2 query fingers vs ANOTHER subject's 2 fingers (position-matched)
        s2 = subs[(subs.index(s) + 1) % len(subs)]
        fs2 = multi[s2][:2]
        imp_tot += 1
        if sum(finger_match(fs[k], fs2[k]) for k in range(2)) >= 2:
            imp_ok += 1
    print(f"  empirical 2-of-2: TAR={gen_ok}/{gen_tot}={gen_ok/max(gen_tot,1)*100:.1f}%  "
          f"FAR={imp_ok}/{imp_tot}={imp_ok/max(imp_tot,1):.2e}")
    print("\n===== TIER-2 COMPLETE =====")


if __name__ == "__main__":
    main()
