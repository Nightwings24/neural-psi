"""
Step 19: PolyU cross-sensor generalization of the SOCOFing-trained system (zero-shot).

Embeds PolyU contact + contactless prints with the SOCOFing-trained feature model and
measures EER for three protocols:
  (a) contact -> contact       (same-sensor baseline)
  (b) contactless -> contactless (same-sensor baseline)
  (c) contact  -> contactless   (CROSS-sensor: the real generalization test)
For each: float EER (Euclidean), Super-Bit-128 EER, L2-thermometer EER, impostor sd, H=0 floor.
Bridges are fit on SOCOFing TRAIN embeddings (zero-shot transfer). Runs on CPU by default so
it does not contend with a GPU training sweep.

This directly answers the red-team's #1 rejection risk: does anything survive a real sensor change?
"""
import argparse, json, os
from math import comb
import numpy as np
from scipy import stats
import torch
from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); D = 128


def embed_all(model, x, dev, batch=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i+batch].astype(np.float32)/255.0).unsqueeze(1).to(dev)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def first_and_rest(x, ids):
    """enroll = first sample per finger; probes = the rest."""
    byid = {}
    for k, i in enumerate(ids):
        byid.setdefault(i, []).append(k)
    fids = sorted(byid, key=lambda s: int(s) if s.isdigit() else s)
    enroll_idx = [byid[f][0] for f in fids]
    probe_idx = {f: byid[f][1:] for f in fids}
    return fids, enroll_idx, probe_idx


def eer_dist(gen, imp):
    s = np.concatenate([gen, imp]); best, e = 1.0, 0.0
    for t in np.linspace(s.min(), s.max(), 2000):
        frr = np.mean(gen > t); far = np.mean(imp <= t)
        if abs(frr-far) < best: best, e = abs(frr-far), (frr+far)/2
    return e


def hamm(A, B):
    A = A.astype(np.int32); B = B.astype(np.int32)
    return A @ (1-B).T + (1-A) @ B.T


def superbit(tr, g, p):
    q = NeuralPSIQuantizer.fit(tr, np.ones(16), QuantizerConfig(center=True, whiten=False, superbit=True, balance=True))
    return q.transform(g).astype(np.uint8), q.transform(p).astype(np.uint8)

def thermo(tr, g, p, K=16, Bp=8, seed=1):
    r = np.random.default_rng(seed); A = r.standard_normal((K, tr.shape[1])); mu = tr.mean(0)
    pj = (tr-mu)@A.T; qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)
    f = lambda X: (((X-mu)@A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(g), f(p)


def eval_protocol(name, emb_enr, id_enr, emb_prb, id_prb, tr_emb):
    """enroll set (one per finger) vs probe set (many per finger). Genuine=same fid; impostor=diff."""
    fen, ridx, _ = first_and_rest(emb_enr, id_enr) if False else (None, None, None)
    # enroll = one per finger
    byen = {}
    for k, i in enumerate(id_enr): byen.setdefault(i, k)   # last occurrence; fine
    fids = sorted(byen)
    gal = np.stack([emb_enr[byen[f]] for f in fids]); gal_id = fids
    # probes = all probe rows whose fid is enrolled
    keep = [k for k, i in enumerate(id_prb) if i in byen]
    prb = emb_prb[keep]; prb_id = [id_prb[k] for k in keep]
    # genuine/impostor float (Euclidean) via distance matrix
    def dmat(P, G):
        P2 = (P*P).sum(1); G2 = (G*G).sum(1)
        return np.sqrt(np.clip(P2[:, None]+G2[None, :]-2*P@G.T, 0, None))
    Df = dmat(prb, gal)
    gi = np.array([gal_id.index(i) for i in prb_id])
    genf = Df[np.arange(len(prb)), gi]
    mask = np.ones_like(Df, bool); mask[np.arange(len(prb)), gi] = False
    impf = Df[mask]
    res = {"protocol": name, "n_gal": len(gal), "n_probe": len(prb),
           "eer_float_euclid": eer_dist(genf, impf)}
    # bridges (fit on SOCOFing train)
    for bn, (cg, cp) in [("superbit", superbit(tr_emb, gal, prb)), ("thermo16x8", thermo(tr_emb, gal, prb))]:
        H = hamm(cp, cg)
        genH = H[np.arange(len(prb)), gi]; impH = H[mask]
        c = int((impH == 0).sum())
        lo = 0.0 if c == 0 else stats.chi2.ppf(0.025, 2*c)/2/impH.size
        hi = stats.chi2.ppf(0.975, 2*(c+1))/2/impH.size
        res[bn] = {"eer": eer_dist(genH.astype(float), impH.astype(float)),
                   "imp_mean": float(impH.mean()), "imp_sd": float(impH.std()),
                   "gen_mean": float(genH.mean()), "collisions": c, "p0": c/impH.size, "p0_ci": [lo, hi]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu (default, avoids GPU sweep) or cuda")
    ap.add_argument("--ckpt", default="feature_model_224.pt")
    args = ap.parse_args()
    dev = args.device
    model = FeatureModel(img=224).to(dev)
    model.load_state_dict(torch.load(os.path.join(DATA, args.ckpt), map_location=dev))
    print(f"[polyu] model={args.ckpt} device={dev}")

    cx = np.load(f"{DATA}/polyu_contact_224.npy"); cid = np.load(f"{DATA}/polyu_contact_ids.npy", allow_pickle=True)
    lx = np.load(f"{DATA}/polyu_cless_224.npy");  lid = np.load(f"{DATA}/polyu_cless_ids.npy", allow_pickle=True)
    tr_emb = np.load(f"{DATA}/dim_ablation_emb_224.npz", allow_pickle=True)["train"].astype(np.float64)
    print(f"[polyu] embedding {len(cx)} contact + {len(lx)} contactless (this is the slow step on CPU)...")
    ec = embed_all(model, cx, dev); el = embed_all(model, lx, dev)
    print("[polyu] embedded. scoring protocols...")

    protocols = [
        ("contact->contact", ec, cid, ec, cid),
        ("contactless->contactless", el, lid, el, lid),
        ("contact->contactless", ec, cid, el, lid),
    ]
    out = []
    print(f"\n{'protocol':<26}{'floatEER':>9}{'SB_EER':>8}{'SB_sd':>7}{'therm_EER':>10}{'th_sd':>7}")
    for name, ee, ie, ep, ip in protocols:
        r = eval_protocol(name, ee, ie, ep, ip, tr_emb); out.append(r)
        print(f"{name:<26}{r['eer_float_euclid']*100:>8.2f}%{r['superbit']['eer']*100:>7.2f}%"
              f"{r['superbit']['imp_sd']:>7.1f}{r['thermo16x8']['eer']*100:>9.2f}%{r['thermo16x8']['imp_sd']:>7.1f}")
    with open(os.path.join(OUT, "polyu-crosssensor.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[polyu] wrote {OUT}/polyu-crosssensor.json")


if __name__ == "__main__":
    main()
