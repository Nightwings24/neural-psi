"""
Step 28: QAT head on the FROZEN CNN's rich conv features (not the 16-D Blind-Touch bottleneck).

Measured motivation: the 25088-D unit-normalised conv features have float EER 0.164% -- HALF the
16-D embedding's 0.33%. The 16-D fc bottleneck discards discriminative signal a 128-bit code
could carry. Here we PCA-reduce the frozen conv features (public linear map fit on train) then
train a quantization-aware 128-bit head on them (whiten objective: genuine-close + bit balance +
bit decorrelation + binarisation), warm-started from an ortho-thermometer on the PCA features so
it never regresses below that baseline. Same frozen CNN, same 128-bit flash-psi backend.

Reports EER + identity-bootstrap CI, impostor sigma, H=0; exports codes + public (PCA,head) map.
Run: python3 28_qat_featurehead.py --pca 512 --epochs 500 --tag feat512
"""
import argparse, json, os
from importlib import import_module
import numpy as np
import torch, torch.nn as nn
from model import FeatureModel

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); IMG, Dc = 224, 128
_q = import_module("23_qat_head")
orthothermo_params, QATHead, evaluate = _q.orthothermo_params, _q.QATHead, _q.evaluate


def L(b):
    p = f"{DATA}/{b}_{IMG}.npy"
    return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{b}.npy", allow_pickle=True)


def extract_feats():
    """Unit-normalised 25088-D conv features for non-test (real,probe) pairs + test gal/prb."""
    cache = f"{DATA}/qat_convfeat_224.npz"
    if os.path.exists(cache):
        z = np.load(cache); return z["r"], z["p"], z["gg"], z["pp"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = FeatureModel(img=IMG).to(dev); m.load_state_dict(torch.load(f"{DATA}/feature_model_224.pt", map_location=dev)); m.eval()
    xr, ir = L("x_real"), L("ids_real"); xp, ip = L("x_probe"), L("ids_probe"); test = set(L("ids_test").tolist())
    ri = {i: k for k, i in enumerate(ir)}; pi = {i: k for k, i in enumerate(ip)}
    tr_ids = sorted((set(ir) & set(ip)) - test); te_ids = sorted((set(ir) & set(ip)) & test)

    def feats(idxs, stack):
        out = []
        with torch.no_grad():
            for i in range(0, len(idxs), 128):
                b = idxs[i:i+128]; xb = torch.from_numpy(stack[b].astype(np.float32)/255.).unsqueeze(1).to(dev)
                f = torch.flatten(m.features(xb), 1); f = f/f.norm(dim=1, keepdim=True)
                out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out)
    r = feats(np.array([ri[i] for i in tr_ids]), xr); p = feats(np.array([pi[i] for i in tr_ids]), xp)
    gg = feats(np.array([ri[i] for i in te_ids]), xr); pp = feats(np.array([pi[i] for i in te_ids]), xp)
    np.savez(cache, r=r, p=p, gg=gg, pp=pp)
    print(f"[feat] extracted conv features -> {cache}  train {len(r)} test {len(gg)}", flush=True)
    return r, p, gg, pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-gen", type=float, default=1.0, help="weight on genuine-closeness (lower genuine Hamming helps fusion)")
    ap.add_argument("--lam-dec", type=float, default=100.0)
    ap.add_argument("--lam-bal", type=float, default=10.0)
    ap.add_argument("--lam-q", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--proj-k", type=int, default=128, help="# projection directions in warm-start (128 for high-D feats)")
    ap.add_argument("--proj-bp", type=int, default=1, help="# thresholds per projection (K*Bp must = 128)")
    ap.add_argument("--whiten", action="store_true", help="PCA-whiten (decorrelate+scale) the components")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="feat")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    r, p, gg, pp = extract_feats()
    # carve a VALIDATION identity split from train (for epoch selection; test stays untouched)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(r))
    nval = int(0.15 * len(r)); vidx = perm[:nval]; tidx = perm[nval:]
    rv, pv = r[vidx], p[vidx]; r, p = r[tidx], p[tidx]
    # PCA fit on TRAIN real+probe only (public linear map) via fast randomized SVD on GPU
    X = np.vstack([r, p]).astype(np.float32); mu = X.mean(0)
    Xt = torch.tensor(X - mu, device=dev)
    U, S, V = torch.pca_lowrank(Xt, q=min(args.pca + 32, Xt.shape[0]-1, Xt.shape[1]), center=False, niter=4)
    comp = V[:, :args.pca].T.cpu().numpy().astype(np.float64)          # (pca, 25088)
    S = S.cpu().numpy()
    scale = (S[:args.pca] / np.sqrt(len(X)-1)) if args.whiten else np.ones(args.pca)
    del Xt
    proj = lambda F: ((F.astype(np.float64) - mu) @ comp.T) / (scale + 1e-8)
    rP, pP, gP, pPt = proj(r), proj(p), proj(gg), proj(pp)
    rvP, pvP = proj(rv), proj(pv)
    print(f"[feat] PCA {args.pca} (whiten={args.whiten}) | var captured {(S[:args.pca]**2).sum()/(S**2).sum():.3f}", flush=True)

    W0, b0 = orthothermo_params(rP, K=args.proj_k, Bp=args.proj_bp)   # warm-start init on PCA feats
    head = QATHead(W0, b0, hidden=args.hidden).to(dev)
    print("[feat] warm-start (ortho-thermo on PCA feats) [test]:", flush=True)
    base = evaluate(head, gP, pPt, "warm-start/test", dev)

    R = torch.tensor(rP, dtype=torch.float32, device=dev); P = torch.tensor(pP, dtype=torch.float32, device=dev); nT = len(rP)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    best_val = 1.0; best = dict(base); best["epoch"] = 0; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    for ep in range(1, args.epochs+1):
        head.train(); tau = max(0.2, 1.0*(1-ep/args.epochs)+0.2)
        idx = torch.randint(0, nT, (args.batch,), device=dev)
        ba = head.ste(R[idx], tau); bp = head.ste(P[idx], tau)
        dg = (Dc - (ba*bp).sum(1))/(2*Dc); L_gen = dg.mean()
        allb = torch.cat([ba, bp], 0); L_bal = (allb.mean(0)**2).mean()
        c = allb - allb.mean(0, keepdim=True); std = c.std(0, keepdim=True)+1e-4
        corr = (c/std).T @ (c/std) / c.shape[0]; off = corr - torch.diag(torch.diag(corr))
        L_dec = (off**2).mean(); L_q = (1-(head.soft(R[idx], tau)**2)).mean()
        loss = args.lam_gen*L_gen + args.lam_dec*L_dec + args.lam_bal*L_bal + args.lam_q*L_q
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0 or ep == 1:
            mv = evaluate(head, rvP, pvP, f"ep{ep}/val", dev)          # selection metric (val)
            if mv["eer"] < best_val:                                    # select by VAL, report TEST
                best_val = mv["eer"]; mt = evaluate(head, gP, pPt, f"ep{ep}/test", dev)
                best = dict(mt); best["epoch"] = ep; best["val_eer"] = mv["eer"]
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    print(f"[feat] BEST (val-selected) @ epoch {best['epoch']}: TEST EER {best['eer']*100:.3f}% "
          f"[{best['ci95'][0]*100:.3f},{best['ci95'][1]*100:.3f}]  sigma {best['imp_sd']:.2f}  H0 {best['h0']}  (val EER {best.get('val_eer',0)*100:.3f}%)", flush=True)
    # export codes for downstream
    with torch.no_grad():
        cg = (head.logits(torch.tensor(gP, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
        cp = (head.logits(torch.tensor(pPt, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
    np.savez(f"{DATA}/qat_codes_{args.tag}.npz", gal=cg, prb=cp)
    np.savez(f"{DATA}/qat_head_{args.tag}_map.npz", mu=mu, comp=comp, scale=scale,
             W=head.lin.weight.detach().cpu().numpy(), b=head.lin.bias.detach().cpu().numpy())
    with open(f"{OUT}/qat-featurehead-{args.tag}.json", "w") as f:
        json.dump({"warm_start": base, "best": best, "args": vars(args),
                   "conv_feat_float_eer": 0.00164}, f, indent=2)
    print(f"[feat] wrote {OUT}/qat-featurehead-{args.tag}.json, codes {DATA}/qat_codes_{args.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
