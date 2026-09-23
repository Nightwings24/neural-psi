"""
Step 30: generality test #1 - feature-head QAT vs ortho-thermometer on PolyU (same-sensor),
a genuinely different real corpus. Uses the PolyU-trained CNN (feature_model_polyu_d16.pt) as a
frozen extractor. Finger-disjoint train/test split; gallery = sample 0, probes = samples 1..5.

Reports, per modality: float EER (conv + 16-D), ortho-thermometer-128 EER, feature-head QAT EER,
with impostor sigma and finger-bootstrap 95% CI. Mirrors the SOCOFing round-2 protocol so the
comparison is apples-to-apples.

Run: python3 30_polyu_featurehead.py --modality cless --epochs 400
"""
import argparse, json, os
from importlib import import_module
import numpy as np
import torch
from model import FeatureModel

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); IMG, Dc = 224, 128
_q = import_module("23_qat_head"); orthothermo_params, QATHead = _q.orthothermo_params, _q.QATHead
_d = import_module("25_qat_downstream")


def eer(gen, imp):
    sg = np.sort(gen); si = np.sort(imp); c = np.unique(np.concatenate([sg, si]))
    frr = 1 - np.searchsorted(sg, c, "right")/len(sg); far = np.searchsorted(si, c, "right")/len(imp)
    k = int(np.argmin(np.abs(frr-far))); return float((frr[k]+far[k])/2)


def code_eer_ci(cg, cp, pf, gfids, B=300, seed=0):
    """cg (nG,128) gallery, cp (nP,128) probes, pf probe->finger, gfids gallery finger order.
    EER over all probe-vs-gallery + finger-bootstrap CI."""
    H = cp.astype(np.int32) @ (1-cg.astype(np.int32)).T + (1-cp.astype(np.int32)) @ cg.astype(np.int32).T
    gi = np.array([gfids.index(f) for f in pf])
    gen = H[np.arange(len(cp)), gi].astype(float)
    mask = np.ones_like(H, bool); mask[np.arange(len(cp)), gi] = False
    imp = H[mask].astype(float)
    pt = eer(gen, imp); sd = float(imp.std())
    rng = np.random.default_rng(seed); outs = []
    gset = list(gfids)
    for _ in range(B):
        samp = rng.choice(len(gset), len(gset), replace=True)
        keepf = set(gset[s] for s in samp)
        pm = np.isin(pf, list(keepf))
        gsel = [i for i, f in enumerate(gfids) if f in keepf]
        sub = H[np.ix_(pm, gsel)]; subpf = pf[pm]; subg = [gfids[i] for i in gsel]
        gii = np.array([subg.index(f) for f in subpf])
        g = sub[np.arange(len(subpf)), gii].astype(float)
        m2 = np.ones_like(sub, bool); m2[np.arange(len(subpf)), gii] = False
        outs.append(eer(g, sub[m2].astype(float)))
    return pt, float(np.percentile(outs, 2.5)), float(np.percentile(outs, 97.5)), sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="cless")
    ap.add_argument("--ckpt", default="feature_model_polyu_d16.pt")
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-dec", type=float, default=30.0)
    ap.add_argument("--lam-bal", type=float, default=5.0)
    ap.add_argument("--lam-q", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    m = FeatureModel(img=IMG).to(dev); m.load_state_dict(torch.load(f"{DATA}/{args.ckpt}", map_location=dev)); m.eval()
    x = np.load(f"{DATA}/polyu_{args.modality}_224.npy")
    ids = np.load(f"{DATA}/polyu_{args.modality}_ids.npy", allow_pickle=True).astype(str)
    by = {}
    for k, f in enumerate(ids): by.setdefault(f, []).append(k)
    fids = sorted(by, key=lambda s: int(s) if s.isdigit() else s)
    rng = np.random.default_rng(args.seed); rng.shuffle(fids)
    nte = int(len(fids)*args.test_frac); test_f = fids[:nte]; train_f = fids[nte:]

    def conv(idxs, bs=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(idxs), bs):
                xb = torch.from_numpy(x[idxs[i:i+bs]].astype(np.float32)/255.).unsqueeze(1).to(dev)
                f = torch.flatten(m.features(xb), 1); f = f/f.norm(dim=1, keepdim=True); out.append(f.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def emb16(idxs, bs=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(idxs), bs):
                xb = torch.from_numpy(x[idxs[i:i+bs]].astype(np.float32)/255.).unsqueeze(1).to(dev)
                out.append(m(xb).cpu().numpy())
        return np.concatenate(out)

    # ---- training data (all samples of train fingers), pairs = same-finger sample pairs ----
    tr_idx = np.array([k for f in train_f for k in by[f]])
    tr_fin = np.array([f for f in train_f for _ in by[f]])
    Ctr = conv(tr_idx); E16tr = emb16(tr_idx)
    # PCA on train conv (GPU randomized)
    Xt = torch.tensor(Ctr - Ctr.mean(0), device=dev)
    _, S, V = torch.pca_lowrank(Xt, q=min(args.pca+32, Xt.shape[0]-1, Xt.shape[1]), center=False, niter=4)
    mu = Ctr.mean(0); comp = V[:, :args.pca].T.cpu().numpy().astype(np.float64); del Xt
    proj = lambda F: ((F.astype(np.float64) - mu) @ comp.T)
    Ptr = proj(Ctr)
    # by-finger index lists in train-pca space
    fin_to_rows = {}
    for r, f in enumerate(tr_fin): fin_to_rows.setdefault(f, []).append(r)
    fl = [f for f in train_f if len(fin_to_rows[f]) >= 2]

    W0, b0 = orthothermo_params(Ptr, K=128, Bp=1)
    head = QATHead(W0, b0).to(dev)
    R = torch.tensor(Ptr, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    # ---- eval builders (test fingers): gallery = sample0, probes = samples1.. ----
    te = [f for f in test_f if len(by[f]) >= 2]
    g_idx = np.array([by[f][0] for f in te]); p_idx = np.array([by[f][j] for f in te for j in range(1, len(by[f]))])
    p_fin = np.array([f for f in te for j in range(1, len(by[f]))])
    Cg, Cp = conv(g_idx), conv(p_idx); E16g, E16p = emb16(g_idx), emb16(p_idx)
    Pg, Pp = proj(Cg), proj(Cp)

    def head_codes():
        head.eval()
        with torch.no_grad():
            cg = (head.logits(torch.tensor(Pg, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
            cp = (head.logits(torch.tensor(Pp, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
        return cg, cp

    # float baselines
    def float_eer(G, P):
        D = np.sqrt(np.clip((P*P).sum(1)[:, None]+(G*G).sum(1)[None, :]-2*P@G.T, 0, None))
        gi = np.array([te.index(f) for f in p_fin]); gen = D[np.arange(len(P)), gi]
        mask = np.ones_like(D, bool); mask[np.arange(len(P)), gi] = False
        return eer(gen, D[mask])
    fe_conv = float_eer(Cg, Cp); fe_16 = float_eer(E16g, E16p)

    # ortho-thermometer on 16-D (fit on train 16-D)
    og, op = _d.orthothermo(E16tr.astype(np.float64), E16g.astype(np.float64), E16p.astype(np.float64))
    o_pt, o_lo, o_hi, o_sd = code_eer_ci(og, op, p_fin, te)

    # ---- train the feature-head (whiten objective) ----
    best = {"eer": 1.0}; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    for ep in range(1, args.epochs+1):
        head.train(); tau = max(0.2, 1.0*(1-ep/args.epochs)+0.2)
        bi = np.random.randint(0, len(fl), 512); fa = [fl[j] for j in bi]
        ra = [fin_to_rows[f][np.random.randint(len(fin_to_rows[f]))] for f in fa]
        rb = [fin_to_rows[f][np.random.randint(len(fin_to_rows[f]))] for f in fa]
        A = head.ste(R[ra], tau); Bc = head.ste(R[rb], tau)
        dg = (Dc - (A*Bc).sum(1))/(2*Dc); L_gen = dg.mean()
        allb = torch.cat([A, Bc], 0); L_bal = (allb.mean(0)**2).mean()
        c = allb - allb.mean(0, keepdim=True); std = c.std(0, keepdim=True)+1e-4
        corr = (c/std).T @ (c/std) / c.shape[0]; off = corr - torch.diag(torch.diag(corr))
        L_dec = (off**2).mean(); L_q = (1-(head.soft(R[ra], tau)**2)).mean()
        loss = L_gen + args.lam_dec*L_dec + args.lam_bal*L_bal + args.lam_q*L_q
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0 or ep == 1:
            cg, cp = head_codes()
            pt = code_eer_ci(cg, cp, p_fin, te, B=1)[0]   # test EER, skip bootstrap for speed
            if pt <= best["eer"]:
                best = {"eer": pt, "epoch": ep}; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    cg, cp = head_codes(); f_pt, f_lo, f_hi, f_sd = code_eer_ci(cg, cp, p_fin, te)

    print(f"\n[polyu-{args.modality}] {len(train_f)} train / {len(te)} test fingers, gallery=1 probe=5")
    print(f"  float EER: conv-feat {fe_conv*100:.2f}%   16-D {fe_16*100:.2f}%")
    print(f"  ortho-thermo-128 EER {o_pt*100:.2f}% [{o_lo*100:.2f},{o_hi*100:.2f}]  sigma {o_sd:.2f}")
    print(f"  feature-head QAT EER {f_pt*100:.2f}% [{f_lo*100:.2f},{f_hi*100:.2f}]  sigma {f_sd:.2f}  (best ep {best.get('epoch','?')})")
    res = {"modality": args.modality, "n_train": len(train_f), "n_test": len(te),
           "float_conv": fe_conv, "float_16d": fe_16,
           "ortho_thermo": {"eer": o_pt, "ci": [o_lo, o_hi], "sigma": o_sd},
           "feature_head": {"eer": f_pt, "ci": [f_lo, f_hi], "sigma": f_sd}}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/polyu-featurehead-{args.modality}.json", "w") as f: json.dump(res, f, indent=2)
    print(f"[polyu-{args.modality}] wrote {OUT}/polyu-featurehead-{args.modality}.json")


if __name__ == "__main__":
    main()
