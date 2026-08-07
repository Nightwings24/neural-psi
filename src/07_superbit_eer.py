"""
Week 1 — Step 7: the MISSING measurement — full Super-Bit EER on held-out identities.

06_superbit_export.py only writes the m=50 demo codes; results.md quotes a Super-Bit
EER of 5.16% but no shipped script computes it. This closes that gap: same open-set
protocol as 03_eval_eer.py (held-out identities, Real enroll vs Altered probe, random
impostors), but adds the friend's Super-Bit 128-bit bridge alongside float / sign / ITQ,
so the whole EER ladder in results.md is reproduced in one run.

  FLOAT     : Euclidean distance on the raw 16-D embedding        (HE's accuracy ceiling)
  sign(>0)  : Hamming on 16-bit naive-sign code
  ITQ       : Hamming on 16-bit ITQ code (learned rotation)
  Super-Bit : Hamming on 128-bit center+SuperBit+median code (quantizer.py)

The Super-Bit quantizer is fit on TRAIN embeddings only (no test leakage), exactly as
06_superbit_export.py does. Run with --whiten + --w1 <file> to also test metric whitening.
"""
import argparse
import os

import numpy as np
import torch

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig, hamming, separation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def embed_all(model, x_u8, batch=128):
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(x_u8), batch):
            xb = torch.from_numpy(x_u8[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def fit_itq(X, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    mean = X.mean(0)
    V = X - mean
    R, _ = np.linalg.qr(rng.standard_normal((V.shape[1], V.shape[1])))
    for _ in range(n_iter):
        B = np.sign(V @ R); B[B == 0] = 1
        U, _, Wt = np.linalg.svd(V.T @ B)
        R = U @ Wt
    return mean, R


def compute_eer(genuine_scores, impostor_scores):
    """EER for *distance* scores (smaller = more similar)."""
    thresholds = np.unique(np.concatenate([genuine_scores, impostor_scores]))
    best = None
    for t in thresholds:
        frr = np.mean(genuine_scores > t)
        far = np.mean(impostor_scores <= t)
        if best is None or abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2.0, t)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--impostors", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--whiten", action="store_true", help="enable diag(sqrt(|w1|)) metric whitening")
    ap.add_argument("--w1", type=str, default="", help="path to head_w1 .npy (required with --whiten)")
    ap.add_argument("--ckpt", default="feature_model.pt", help="feature-extractor checkpoint in data/")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # img-aware loading: prefer x_real_<img>.npy / ids_test_<img>.npy (Tier-1 224 run),
    # else fall back to the 96px defaults — so this one script serves both models.
    def dload(base):
        p = os.path.join(DATA, f"{base}_{args.img}.npy")
        return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(os.path.join(DATA, f"{base}.npy"), allow_pickle=True)

    x_real, ids_real = dload("x_real"), dload("ids_real")
    x_probe, ids_probe = dload("x_probe"), dload("ids_probe")
    ids_test = set(dload("ids_test").tolist())

    model = FeatureModel(img=args.img)
    model.load_state_dict(torch.load(os.path.join(DATA, args.ckpt)))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eer] model={args.ckpt} img={args.img} | data x_real{('_'+str(args.img)) if os.path.exists(os.path.join(DATA,f'x_real_{args.img}.npy')) else ''}")

    real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
    common = sorted(set(real_by) & set(prb_by))
    print(f"[eer] held-out identities with a genuine probe: {len(common)}")

    gal = np.stack([real_by[i] for i in common])     # enrollment (Real)
    prb = np.stack([prb_by[i] for i in common])      # genuine probe (Altered)
    emb_gal, emb_prb = embed_all(model, gal), embed_all(model, prb)

    # --- TRAIN-only embeddings for the data-dependent bridges (ITQ, Super-Bit) ---
    train_real = np.stack([x for i, x in zip(ids_real, x_real) if i not in ids_test])
    emb_train = embed_all(model, train_real)

    # naive sign
    bin_gal, bin_prb = (emb_gal > 0).astype(np.int8), (emb_prb > 0).astype(np.int8)
    # ITQ (16-bit)
    itq_mean, itq_R = fit_itq(emb_train)
    itq_gal = ((emb_gal - itq_mean) @ itq_R > 0).astype(np.int8)
    itq_prb = ((emb_prb - itq_mean) @ itq_R > 0).astype(np.int8)
    # Super-Bit (128-bit) — fit on train embeddings, friend's config
    if args.whiten:
        if not args.w1 or not os.path.exists(args.w1):
            raise SystemExit("--whiten needs --w1 <head_w1.npy> (run train_gpu_224.py to make it)")
        w1 = np.load(args.w1).reshape(-1)
        cfg = QuantizerConfig(center=True, whiten=True, superbit=True, balance=True)
        print(f"[eer] Super-Bit WITH metric whitening (|w1| range "
              f"{np.abs(w1).min():.3g}..{np.abs(w1).max():.3g})")
    else:
        w1 = np.ones(16)
        cfg = QuantizerConfig(center=True, whiten=False, superbit=True, balance=True)
        print("[eer] Super-Bit WITHOUT whitening (head weights not available)")
    quant = NeuralPSIQuantizer.fit(emb_train, w1, cfg)
    sb_gal, sb_prb = quant.transform(emb_gal), quant.transform(emb_prb)

    n = len(common)
    gen_f, imp_f, gen_s, imp_s, gen_q, imp_q, gen_b, imp_b = ([] for _ in range(8))
    for i in range(n):
        others = rng.choice(np.delete(np.arange(n), i),
                            size=min(args.impostors, n - 1), replace=False)
        gen_f.append(np.linalg.norm(emb_prb[i] - emb_gal[i]))
        gen_s.append(int(np.sum(bin_prb[i] != bin_gal[i])))
        gen_q.append(int(np.sum(itq_prb[i] != itq_gal[i])))
        gen_b.append(int(np.sum(sb_prb[i] != sb_gal[i])))
        imp_f.extend(np.linalg.norm(emb_gal[others] - emb_prb[i], axis=1))
        imp_s.extend(np.sum(bin_gal[others] != bin_prb[i], axis=1))
        imp_q.extend(np.sum(itq_gal[others] != itq_prb[i], axis=1))
        imp_b.extend(np.sum(sb_gal[others] != sb_prb[i], axis=1))

    arr = lambda v: np.asarray(v, float)
    eer_f, _ = compute_eer(arr(gen_f), arr(imp_f))
    eer_s, _ = compute_eer(arr(gen_s), arr(imp_s))
    eer_q, _ = compute_eer(arr(gen_q), arr(imp_q))
    eer_b, _ = compute_eer(arr(gen_b), arr(imp_b))

    print("\n" + "=" * 72)
    print("  SUPER-BIT EER LADDER — real held-out SOCOFing identities")
    print("=" * 72)
    print(f"  genuine pairs {len(gen_f)} | impostor pairs {len(imp_f)}")
    print("-" * 72)
    print(f"  FLOAT      (Euclidean, 16-D)      : EER = {eer_f*100:6.2f}%   (ceiling)")
    print(f"  sign(>0)   (Hamming, 16-bit)      : EER = {eer_s*100:6.2f}%   (+{(eer_s-eer_f)*100:.2f} pp)")
    print(f"  ITQ        (Hamming, 16-bit)      : EER = {eer_q*100:6.2f}%   (+{(eer_q-eer_f)*100:.2f} pp)")
    print(f"  Super-Bit  (Hamming, 128-bit)     : EER = {eer_b*100:6.2f}%   (+{(eer_b-eer_f)*100:.2f} pp)")
    print("-" * 72)
    print(f"  Super-Bit vs ITQ                  : {(eer_q-eer_b)*100:+.2f} pp")
    print(f"  Super-Bit Hamming gen/imp         : {arr(gen_b).mean():.2f} / {arr(imp_b).mean():.2f}  (of 128)")
    print(f"  Super-Bit d-prime                 : {separation(arr(gen_b), arr(imp_b)):.3f}")
    print(f"  Super-Bit mean per-bit balance    : {sb_gal.mean()*100:.1f}%  (50% = ideal)")
    print("=" * 72)

    with open(os.path.join(DATA, "eer_report.txt"), "a") as f:
        tag = "itq+whiten" if args.whiten else "noWhiten"
        f.write(f"eer_superbit_{tag}\t{eer_b:.4f}\n")
        f.write(f"gen_hamming_superbit_{tag}\t{arr(gen_b).mean():.4f}\n")
        f.write(f"imp_hamming_superbit_{tag}\t{arr(imp_b).mean():.4f}\n")
    print(f"[eer] appended Super-Bit rows -> {os.path.join(DATA, 'eer_report.txt')}")


if __name__ == "__main__":
    main()
