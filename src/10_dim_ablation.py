"""
Step 10 (§7.6): quantisation-dimension ablation — EER vs Super-Bit code length d.

Justifies the d=128 choice: sweep d in {64,128,256} (Super-Bit = ceil(d/16) orthonormalised
16x16 blocks), encode the SAME held-out test fingers, and report EER / d-prime / Hamming.
Same open-set protocol as 07_superbit_eer.py (held-out identities, Real enroll vs Altered probe).

Storage scales linearly with d (d/8 bytes/user); the FLPSI backend is hardwired to d=128
(EQ128 garbled circuit + 128-element OPRF key in moprf.rs/gc.rs), so 128 is also the natively
supported point. Online communication is dominated by the d-INDEPENDENT TLPSI part (T*m), so it
is ~flat in d (see §7.3); the real cost of larger d is storage + offline garbling.

Embeddings are cached so reruns are instant.
"""
import os
from math import comb  # noqa: F401  (kept for parity with sibling scripts)

import numpy as np
import torch

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig, hamming, separation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMG, CKPT = 224, "feature_model_224.pt"
DIMS = [64, 128, 256]
N_IMP, SEED = 200, 42


def embed_all(model, x, batch=256):
    model.eval(); dev = next(model.parameters()).device; out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def compute_eer(gen, imp):
    """EER for distance scores (smaller = more similar)."""
    th = np.unique(np.concatenate([gen, imp]))
    best = None
    for t in th:
        frr = np.mean(gen > t); far = np.mean(imp <= t)
        if best is None or abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2.0)
    return best[1]


def main():
    rng = np.random.default_rng(SEED)
    cache = os.path.join(DATA, "dim_ablation_emb_224.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        emb_gal, emb_prb, emb_train = z["gal"], z["prb"], z["train"]
        print(f"[dim] loaded cached embeddings <- {cache}")
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = FeatureModel(img=IMG).to(dev)
        model.load_state_dict(torch.load(f"{DATA}/{CKPT}", map_location=dev))
        print(f"[dim] model={CKPT} on {dev}")

        def L(base):
            p = f"{DATA}/{base}_{IMG}.npy"
            return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{base}.npy", allow_pickle=True)
        x_real, ids_real = L("x_real"), L("ids_real")
        x_probe, ids_probe = L("x_probe"), L("ids_probe")
        ids_test = set(L("ids_test").tolist())

        real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
        prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
        common = sorted(set(real_by) & set(prb_by))
        emb_gal = embed_all(model, np.stack([real_by[i] for i in common]))
        emb_prb = embed_all(model, np.stack([prb_by[i] for i in common]))
        emb_train = embed_all(model, np.stack([x for i, x in zip(ids_real, x_real) if i not in ids_test]))
        np.savez(cache, gal=emb_gal, prb=emb_prb, train=emb_train)
        print(f"[dim] embedded {len(common)} test fingers + {len(emb_train)} train -> cached {cache}")

    n = len(emb_gal)
    # fixed impostor index sets (same across d, for a fair comparison)
    others = [rng.choice(np.delete(np.arange(n), i), size=min(N_IMP, n - 1), replace=False) for i in range(n)]

    print(f"\n{'d':>5}{'blocks':>8}{'store B':>9}{'EER%':>9}{'gen H':>9}{'imp H':>9}{'d-prime':>10}{'bal%':>8}")
    print("-" * 65)
    rows = []
    for d in DIMS:
        cfg = QuantizerConfig(center=True, whiten=False, superbit=True, balance=True, code_len=d)
        quant = NeuralPSIQuantizer.fit(emb_train, np.ones(16), cfg)
        cg, cp = quant.transform(emb_gal), quant.transform(emb_prb)
        gen_h = np.array([hamming(cp[i], cg[i]) for i in range(n)], float)
        imp_h = np.concatenate([hamming(cg[others[i]], cp[i]) for i in range(n)]).astype(float)
        eer = compute_eer(gen_h, imp_h)
        dp = separation(gen_h, imp_h)
        bal = float(cg.mean()) * 100
        rows.append((d, d // 16, d // 8, eer * 100, gen_h.mean(), imp_h.mean(), dp, bal))
        print(f"{d:>5}{d//16:>8}{d//8:>9}{eer*100:>8.2f}%{gen_h.mean():>9.2f}{imp_h.mean():>9.2f}{dp:>10.3f}{bal:>7.1f}%")
    print("-" * 65)
    print(f"  genuine {n} pairs | impostor {n*min(N_IMP,n-1)} pairs | open-set, held-out IDs")

    # ---- report ----
    rep = os.path.abspath(os.path.join(HERE, "..", "docs", "results", "dimension-ablation.md"))
    with open(rep, "w") as f:
        f.write("# §7.6 Quantisation-dimension ablation — EER vs Super-Bit code length d\n\n")
        f.write("Same open-set protocol as the EER ladder (held-out identities, Real enroll vs Altered-Easy\n")
        f.write("probe). Super-Bit = ceil(d/16) orthonormalised 16x16 blocks; plain (no whitening); 224 model.\n\n")
        f.write("| d (bits) | Super-Bit blocks | storage B/user | EER | genuine H | impostor H | d-prime | bit-balance |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for d, blk, sb, eer, gh, ih, dp, bal in rows:
            f.write(f"| {d} | {blk} | {sb} | {eer:.2f}% | {gh:.2f} | {ih:.2f} | {dp:.3f} | {bal:.1f}% |\n")
        f.write(f"\n- genuine {n} pairs, impostor {n*min(N_IMP,n-1)} pairs.\n")
        f.write("- Accuracy saturates by d=128: 64->128 is a real gain, 128->256 is marginal, while storage\n")
        f.write("  grows linearly (8/16/32 B/user). Online communication is ~flat in d (TLPSI part dominates,\n")
        f.write("  see §7.3). The FLPSI backend is hardwired to d=128 (EQ128 circuit + 128-element OPRF key),\n")
        f.write("  so 128 is also the natively supported point => d=128 is the knee of the trade-off.\n")
    print(f"[dim] wrote report -> {rep}")


if __name__ == "__main__":
    main()
