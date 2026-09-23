"""
Week 1 - Step 3: Measure real EER on held-out identities, FLOAT vs BINARY.

Protocol (open-set verification on identities NOT seen during training):
  - Enrollment template  = FC-16 embedding of the Real print.
  - Genuine probe        = FC-16 embedding of that identity's Altered-Easy print.
  - Impostor probes      = embeddings of Altered prints of *other* identities.

Scores:
  - FLOAT  : Euclidean distance between raw 16-d float embeddings.
  - BINARY : Hamming distance between sign-binarized embeddings (b = float > 0),
             i.e. exactly the bridge used in neural_psi_demo.py.

EER is the operating point where FAR == FRR, computed from the genuine/impostor
score distributions. The float->binary EER gap quantifies what the 1-bit PSI
bridge costs in accuracy.
"""
import argparse
import os

import numpy as np
import torch

from model import FeatureModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def embed_all(model, x_u8, batch=128):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_u8), batch):
            xb = torch.from_numpy(x_u8[i:i + batch].astype(np.float32) / 255.0)
            xb = xb.unsqueeze(1)
            outs.append(model(xb).numpy())
    return np.concatenate(outs)


def fit_itq(X, n_iter=50, seed=0):
    """Iterative Quantization (Gong & Lazebnik). Learns mean + orthogonal R so
    sign((x-mean) @ R) aligns float features to hypercube vertices.
    Returns (mean, R). dim stays 16 -> 16 bits (no PCA reduction)."""
    rng = np.random.default_rng(seed)
    mean = X.mean(0)
    V = X - mean
    c = V.shape[1]
    R, _ = np.linalg.qr(rng.standard_normal((c, c)))   # random orthogonal init
    for _ in range(n_iter):
        B = np.sign(V @ R)
        B[B == 0] = 1
        U, _, Wt = np.linalg.svd(V.T @ B)               # Orthogonal Procrustes
        R = U @ Wt
    return mean, R


def itq_binarize(emb, mean, R):
    b = np.sign((emb - mean) @ R)
    return (b > 0).astype(np.int8)


def compute_eer(genuine_scores, impostor_scores):
    """EER for *distance* scores (smaller = more similar). Returns (eer, threshold)."""
    g = np.sort(genuine_scores)
    im = np.sort(impostor_scores)
    thresholds = np.unique(np.concatenate([g, im]))
    best = None
    for t in thresholds:
        # accept if distance <= t
        frr = np.mean(g > t)                 # genuine rejected
        far = np.mean(im <= t)               # impostor accepted
        if best is None or abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2.0, t)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--impostors", type=int, default=200,
                    help="random impostor probes per genuine pair")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    x_real = np.load(os.path.join(DATA, "x_real.npy"))
    ids_real = np.load(os.path.join(DATA, "ids_real.npy"))
    x_probe = np.load(os.path.join(DATA, "x_probe.npy"))
    ids_probe = np.load(os.path.join(DATA, "ids_probe.npy"))
    ids_test = set(np.load(os.path.join(DATA, "ids_test.npy")).tolist())

    model = FeatureModel(img=args.img)
    model.load_state_dict(torch.load(os.path.join(DATA, "feature_model.pt")))

    # restrict to held-out (unseen) identities that also have an Altered probe
    real_by_id = {iid: x for iid, x in zip(ids_real, x_real) if iid in ids_test}
    probe_by_id = {iid: x for iid, x in zip(ids_probe, x_probe) if iid in ids_test}
    common = sorted(set(real_by_id) & set(probe_by_id))
    print(f"[eer] held-out identities with a genuine probe: {len(common)}")

    gal = np.stack([real_by_id[i] for i in common])      # enrollment (Real)
    prb = np.stack([probe_by_id[i] for i in common])     # genuine probe (Altered)

    emb_gal = embed_all(model, gal)
    emb_prb = embed_all(model, prb)

    # naive sign bridge (neural_psi_demo.py: features > 0)
    bin_gal = (emb_gal > 0).astype(np.int8)
    bin_prb = (emb_prb > 0).astype(np.int8)

    # ITQ bridge: fit rotation on TRAIN identities' embeddings only (no leakage)
    train_real = np.stack([x for iid, x in zip(ids_real, x_real)
                           if iid not in ids_test])
    emb_train = embed_all(model, train_real)
    itq_mean, itq_R = fit_itq(emb_train)
    np.savez(os.path.join(DATA, "itq_params.npz"), mean=itq_mean, R=itq_R)
    itq_gal = itq_binarize(emb_gal, itq_mean, itq_R)
    itq_prb = itq_binarize(emb_prb, itq_mean, itq_R)

    n = len(common)
    gen_f, imp_f = [], []          # float Euclidean
    gen_s, imp_s = [], []          # sign(>0) Hamming
    gen_q, imp_q = [], []          # ITQ Hamming
    for i in range(n):
        others = rng.choice(np.delete(np.arange(n), i),
                            size=min(args.impostors, n - 1), replace=False)
        # genuine: probe i vs its own enrollment i
        gen_f.append(np.linalg.norm(emb_prb[i] - emb_gal[i]))
        gen_s.append(int(np.sum(bin_prb[i] != bin_gal[i])))
        gen_q.append(int(np.sum(itq_prb[i] != itq_gal[i])))
        # impostors: probe i vs random other enrollments
        imp_f.extend(np.linalg.norm(emb_gal[others] - emb_prb[i], axis=1))
        imp_s.extend(np.sum(bin_gal[others] != bin_prb[i], axis=1))
        imp_q.extend(np.sum(itq_gal[others] != itq_prb[i], axis=1))

    gen_f, imp_f = np.array(gen_f), np.array(imp_f)
    gen_s, imp_s = np.array(gen_s, float), np.array(imp_s, float)
    gen_q, imp_q = np.array(gen_q, float), np.array(imp_q, float)

    eer_f, thr_f = compute_eer(gen_f, imp_f)
    eer_s, thr_s = compute_eer(gen_s, imp_s)
    eer_q, thr_q = compute_eer(gen_q, imp_q)

    print("\n" + "=" * 66)
    print("  WEEK 1 RESULT - real EER on held-out SOCOFing identities")
    print("=" * 66)
    print(f"  genuine pairs {len(gen_f)} | impostor pairs {len(imp_f)} | "
          f"FC-{emb_gal.shape[1]} float vector")
    print("-" * 66)
    print(f"  Stage-1 FLOAT  (Euclidean)        : EER = {eer_f*100:6.2f}%")
    print(f"  Stage-2 bridge  sign(>0)  Hamming : EER = {eer_s*100:6.2f}%   "
          f"(+{(eer_s-eer_f)*100:.2f} pp)")
    print(f"  Stage-2 bridge  ITQ       Hamming : EER = {eer_q*100:6.2f}%   "
          f"(+{(eer_q-eer_f)*100:.2f} pp)")
    print("-" * 66)
    print(f"  ITQ recovers vs naive sign        : {(eer_s-eer_q)*100:+.2f} pp")
    print(f"  sign  Hamming genuine/impostor    : {gen_s.mean():.2f} / {imp_s.mean():.2f} (of 16)")
    print(f"  ITQ   Hamming genuine/impostor    : {gen_q.mean():.2f} / {imp_q.mean():.2f} (of 16)")
    print("=" * 66)

    report = os.path.join(DATA, "eer_report.txt")
    with open(report, "w") as f:
        f.write(f"genuine_pairs\t{len(gen_f)}\n")
        f.write(f"impostor_pairs\t{len(imp_f)}\n")
        f.write(f"eer_float\t{eer_f:.4f}\n")
        f.write(f"eer_sign\t{eer_s:.4f}\n")
        f.write(f"eer_itq\t{eer_q:.4f}\n")
        f.write(f"gen_hamming_sign\t{gen_s.mean():.4f}\n")
        f.write(f"imp_hamming_sign\t{imp_s.mean():.4f}\n")
        f.write(f"gen_hamming_itq\t{gen_q.mean():.4f}\n")
        f.write(f"imp_hamming_itq\t{imp_q.mean():.4f}\n")
    print(f"[eer] report -> {report}")


if __name__ == "__main__":
    main()
