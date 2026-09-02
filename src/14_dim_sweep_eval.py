"""
Dimension-sweep + bridge evaluator (R1 spine + E1 metric-aware bridge, one pass).

For a trained feature extractor of a given embedding dimension D, on the held-out
SOCOFing identities, measure the quantities the FLPSI parameter model depends on:
  - float ceilings (Euclidean AND cosine) + the intrinsic float overlap,
  - Super-Bit-128 vs L2-thermometer-128 EER,
  - impostor Hamming mean/sd (the FLPSI uniform model assumes sd=5.66),
  - the CANONICAL H=0 collision floor (probe x gallery, off-diagonal) + exact Poisson 95% CI.

All impostor stats are over the FULL held-out pair set (n(n-1)), not a sample, so the
~1e-6 floor is measurable. Bridges are fit on TRAIN embeddings only (no test leakage).

Usage:
  python3 14_dim_sweep_eval.py --emb-dim D --ckpt feature_model_224_dD.pt
  (D=16 canonical: --emb-dim 16 --ckpt feature_model_224.pt --ids-test ids_test_224.npy)
"""
import argparse, json, os
import numpy as np
import torch
from scipy import stats

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results", "dim-sweep"))


def embed_all(model, x_u8, batch=128):
    model.eval(); dev = next(model.parameters()).device; outs = []
    with torch.no_grad():
        for i in range(0, len(x_u8), batch):
            xb = torch.from_numpy(x_u8[i:i+batch].astype(np.float32)/255.0).unsqueeze(1).to(dev)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def hamm(A, B):
    A = A.astype(np.int32); B = B.astype(np.int32)
    return A @ (1 - B).T + (1 - A) @ B.T


def eer_from_H(H, eye):
    gen, imp = H[eye], H[~eye]
    best, eer = 1.0, 0.0
    for t in np.linspace(0, H.shape[1], 2000):
        frr = np.mean(gen > t); far = np.mean(imp <= t)
        if abs(frr - far) < best:
            best, eer = abs(frr - far), (frr + far) / 2
    return eer, gen, imp


def poisson_ci(k, exposure, alpha=0.05):
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha/2, 2*k)/2
    hi = stats.chi2.ppf(1-alpha/2, 2*(k+1))/2
    return lo/exposure, hi/exposure


def superbit_codes(emb_tr, emb_g, emb_p, D):
    q = NeuralPSIQuantizer.fit(emb_tr, np.ones(D),
                               QuantizerConfig(center=True, whiten=False, superbit=True,
                                               balance=True, code_len=128, in_dim=D))
    return q.transform(emb_g), q.transform(emb_p)


def thermo_codes(emb_tr, emb_g, emb_p, D, K=16, Bp=8, seed=1):
    r = np.random.default_rng(seed); A = r.standard_normal((K, D))
    mu = emb_tr.mean(0)
    pj = (emb_tr - mu) @ A.T
    qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)   # (Bp,K)
    f = lambda X: (((X - mu) @ A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(emb_g), f(emb_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dim", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--ids-test", default=None, help="held-out ids .npy (default ids_test_224_d<D>.npy)")
    args = ap.parse_args()
    D = args.emb_dim

    def dload(base):
        p = os.path.join(DATA, f"{base}_{args.img}.npy")
        return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(os.path.join(DATA, f"{base}.npy"), allow_pickle=True)

    ids_test_file = args.ids_test or (f"ids_test_224_d{D}.npy" if os.path.exists(os.path.join(DATA, f"ids_test_224_d{D}.npy")) else "ids_test_224.npy")
    ids_test = set(np.load(os.path.join(DATA, ids_test_file), allow_pickle=True).tolist())
    x_real, ids_real = dload("x_real"), dload("ids_real")
    x_probe, ids_probe = dload("x_probe"), dload("ids_probe")

    model = FeatureModel(img=args.img, emb_dim=D)
    model.load_state_dict(torch.load(os.path.join(DATA, args.ckpt)))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
    common = sorted(set(real_by) & set(prb_by))
    gal = np.stack([real_by[i] for i in common]); prb = np.stack([prb_by[i] for i in common])
    train_real = np.stack([x for i, x in zip(ids_real, x_real) if i not in ids_test])
    n = len(common); eye = np.eye(n, dtype=bool)
    print(f"[sweep] D={D} ckpt={args.ckpt} ids={ids_test_file} | held-out {n} ids, {n*(n-1):,} impostor pairs")

    eg, ep, etr = embed_all(model, gal), embed_all(model, prb), embed_all(model, train_real)

    # ---- float ceilings + intrinsic overlap ----
    g = eg/np.linalg.norm(eg,axis=1,keepdims=True); p = ep/np.linalg.norm(ep,axis=1,keepdims=True)
    Sc = p @ g.T                                   # cosine sim
    genc, impc = Sc[eye], Sc[~eye]
    def eer_score(gen, imp, higher_gen):
        best, eer = 1.0, 0.0
        for t in np.linspace(min(gen.min(),imp.min()), max(gen.max(),imp.max()), 2000):
            if higher_gen: frr=np.mean(gen<t); far=np.mean(imp>=t)
            else: frr=np.mean(gen>t); far=np.mean(imp<=t)
            if abs(frr-far)<best: best,eer=abs(frr-far),(frr+far)/2
        return eer
    eer_cos = eer_score(genc, impc, True)
    G2=(eg*eg).sum(1); P2=(ep*ep).sum(1); D2=P2[:,None]+G2[None,:]-2*(ep@eg.T)
    Deuc=np.sqrt(np.clip(D2,0,None)); eer_euc=eer_score(Deuc[eye],Deuc[~eye],False)

    res = {"emb_dim": D, "ckpt": args.ckpt, "n_ids": n, "n_impostor": n*(n-1),
           "eer_float_euclid": eer_euc, "eer_float_cosine": eer_cos}

    # ---- bridges ----
    for name, (cg, cp) in [("superbit", superbit_codes(etr, eg, ep, D)),
                           ("thermo16x8", thermo_codes(etr, eg, ep, D))]:
        H = hamm(cp, cg); eer, gen, imp = eer_from_H(H, eye)
        c = int((imp == 0).sum()); lo, hi = poisson_ci(c, imp.size)
        dp = (imp.mean()-gen.mean())/np.sqrt((gen.var()+imp.var())/2)
        res[name] = {"eer": eer, "gen_H": float(gen.mean()), "imp_H": float(imp.mean()),
                     "imp_sd": float(imp.std()), "dprime": float(dp),
                     "collisions": c, "p0": c/imp.size, "p0_ci": [lo, hi]}
        print(f"  {name:12} EER {eer*100:5.2f}%  impH {imp.mean():5.2f} sd {imp.std():5.2f}  d' {dp:.2f}  H=0 {c} p0 {c/imp.size:.2e} CI[{lo:.1e},{hi:.1e}]")
    print(f"  float EER  Euclid {eer_euc*100:.2f}%  cosine {eer_cos*100:.2f}%")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"d{D}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[sweep] wrote {OUT}/d{D}.json")


if __name__ == "__main__":
    main()
