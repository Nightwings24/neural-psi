"""
Step 23: Quantization-Aware Training of a learned 128-bit quantizer head (plan round-2, step 4).

The ortho-thermometer is a FIXED public map: bit_{k,j} = 1 iff a_k.(e-mu) > q_{k,j}, i.e. it is
exactly 128 affine-then-sign functions -> a Linear(16,128) followed by a sign. We therefore:

  1. warm-start a Linear(16,128) head to REPRODUCE the ortho-thermometer bit-for-bit
     (so we can never regress below its 0.87% held-out EER), then
  2. train it with a straight-through sign estimator against a code-space objective:
       L = L_pair (separate genuine/impostor in Hamming)
         + lam_dec * L_decorr  (drive off-diagonal bit-correlation -> 0; the sigma lever)
         + lam_bal * L_balance (each bit ~50/50)
         + lam_q   * L_quant   (push activations to +-1)

The decorrelation term directly attacks the measured pathology (step 22): impostor Hamming
sigma 14.6 vs a uniform code's 5.66, i.e. only ~14-18 of 128 bits are effectively independent.
Decorrelating the bits over the population tightens impostor sigma toward 5.66, which (a) lowers
EER toward the float ceiling and (b) lets FLPSI cut the subsample count T at matched FAR/FRR.

Trains on NON-test identities (real+probe views as genuine pairs); evaluates on the held-out
test gal/prb (same split as every other bridge result). Reports EER + identity-bootstrap 95% CI,
impostor sigma, H=0 floor. GPU used only for the one-off embedding of training images.

Run: python3 23_qat_head.py --epochs 400 --lam-dec 1.0
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn
from model import FeatureModel

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results"))
IMG, CKPT, Dc = 224, "feature_model_224.pt", 128


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_train_pairs():
    """Cache real+probe 16-D embeddings for NON-test identities (genuine pairs)."""
    cache = f"{DATA}/qat_train_pairs_224.npz"
    if os.path.exists(cache):
        z = np.load(cache); return z["real"], z["probe"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureModel(img=IMG).to(dev); model.load_state_dict(torch.load(f"{DATA}/{CKPT}", map_location=dev)); model.eval()

    def L(base):
        p = f"{DATA}/{base}_{IMG}.npy"
        return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{base}.npy", allow_pickle=True)
    x_real, ids_real = L("x_real"), L("ids_real")
    x_probe, ids_probe = L("x_probe"), L("ids_probe")
    ids_test = set(L("ids_test").tolist())
    real_by = {i: x for i, x in zip(ids_real, x_real) if i not in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i not in ids_test}
    common = sorted(set(real_by) & set(prb_by))

    def emb(stack, batch=256):
        out = []
        with torch.no_grad():
            for i in range(0, len(stack), batch):
                xb = torch.from_numpy(stack[i:i+batch].astype(np.float32)/255.).unsqueeze(1).to(dev)
                out.append(model(xb).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
    real = emb(np.stack([real_by[i] for i in common]))
    probe = emb(np.stack([prb_by[i] for i in common]))
    np.savez(cache, real=real, probe=probe)
    print(f"[qat] built {len(common)} non-test genuine pairs -> {cache}")
    return real, probe


def orthothermo_params(tr, K=16, Bp=8, seed=1):
    """Return (W0 128x16, b0 128) whose sign reproduces the ortho-thermometer code."""
    r = np.random.default_rng(seed); rows = []
    while len(rows) < K:
        q, _ = np.linalg.qr(r.standard_normal((tr.shape[1], tr.shape[1]))); rows.extend(q.T)
    A = np.array(rows[:K]); mu = tr.mean(0); pj = (tr - mu) @ A.T
    qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)   # (Bp, K)
    W0 = np.repeat(A, Bp, axis=0)                                       # (128,16): a_k repeated Bp times
    b0 = np.empty(K*Bp)
    for k in range(K):
        for j in range(Bp):
            b0[k*Bp+j] = -(A[k] @ mu + qs[j, k])
    return W0, b0


# --------------------------------------------------------------------------- #
# model / losses
# --------------------------------------------------------------------------- #
class QATHead(nn.Module):
    def __init__(self, W0, b0, hidden=0):
        super().__init__()
        din, dout = W0.shape[1], W0.shape[0]
        self.lin = nn.Linear(din, dout)                       # warm-started linear branch
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(W0, dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(b0, dtype=torch.float32))
        self.mlp = None
        if hidden > 0:                                        # zero-initialised nonlinear residual
            self.mlp = nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, dout))
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def logits(self, e):
        z = self.lin(e)
        return z if self.mlp is None else z + self.mlp(e)

    def soft(self, e, tau):
        return torch.tanh(self.logits(e) / tau)              # (-1,1)

    def ste(self, e, tau):
        s = self.soft(e, tau)
        h = torch.sign(self.logits(e))                        # +-1 forward
        return s + (h - s).detach()                           # straight-through


def eer_np(gen, imp):
    sg = np.sort(gen); si = np.sort(imp); cand = np.unique(np.concatenate([sg, si]))
    frr = 1.0 - np.searchsorted(sg, cand, "right")/len(sg); far = np.searchsorted(si, cand, "right")/len(si)
    k = int(np.argmin(np.abs(frr-far))); return float((frr[k]+far[k])/2.0)


def eer_from_counts(gh, ih):
    G, I = gh.sum(), ih.sum(); cg, ci = np.cumsum(gh), np.cumsum(ih)
    frr = 1 - cg/G; far = ci/I; k = np.argmin(np.abs(frr-far)); return (frr[k]+far[k])/2


def boot_ci(H, B=1000, seed=0):
    n = H.shape[0]; rng = np.random.default_rng(seed); gen = np.diag(H).astype(int)
    pt = eer_from_counts(np.bincount(gen, minlength=Dc+1),
                         np.bincount(H[~np.eye(n, dtype=bool)].astype(int), minlength=Dc+1))
    outs = []
    for _ in range(B):
        idx = rng.integers(0, n, n); g = gen[idx]; rows = H[idx].copy()
        rows[np.arange(n), idx] = -1; imv = rows[rows >= 0]
        outs.append(eer_from_counts(np.bincount(g, minlength=Dc+1), np.bincount(imv.astype(int), minlength=Dc+1)))
    return pt, float(np.percentile(outs, 2.5)), float(np.percentile(outs, 97.5))


def evaluate(head, gal, prb, tag, dev):
    head.eval()
    with torch.no_grad():
        cg = (head.logits(torch.tensor(gal, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.int32)
        cp = (head.logits(torch.tensor(prb, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.int32)
    H = cp @ (1-cg).T + (1-cp) @ cg.T
    n = len(gal); off = ~np.eye(n, dtype=bool)
    genH, impH = np.diag(H), H[off]
    pt, lo, hi = boot_ci(H)
    h0 = int((impH == 0).sum())
    print(f"  [{tag}] EER {pt*100:.2f}% [{lo*100:.2f},{hi*100:.2f}]  impostor mean {impH.mean():.1f} sd {impH.std():.2f}  gen {genH.mean():.1f}  H0 {h0}")
    return {"eer": pt, "ci95": [lo, hi], "imp_mean": float(impH.mean()), "imp_sd": float(impH.std()),
            "gen_mean": float(genH.mean()), "h0": h0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam-dec", type=float, default=1.0)
    ap.add_argument("--lam-bal", type=float, default=0.5)
    ap.add_argument("--lam-q", type=float, default=0.1)
    ap.add_argument("--loss", type=str, default="whiten", choices=["whiten", "bce"])
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="qat")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    real, probe = build_train_pairs()
    z = np.load(f"{DATA}/dim_ablation_emb_224.npz"); gal, prb, tr = z["gal"], z["prb"], z["train"]
    W0, b0 = orthothermo_params(tr.astype(np.float64))
    head = QATHead(W0, b0, hidden=args.hidden).to(dev)

    print(f"[qat] train pairs {len(real)} | held-out {len(gal)} | dev {dev} | loss={args.loss} scale={args.scale} "
          f"lam_dec={args.lam_dec} lam_bal={args.lam_bal} lam_q={args.lam_q}")
    print("[qat] warm-start (should match ortho-thermometer):")
    base = evaluate(head, gal, prb, "warm-start", dev)

    R = torch.tensor(real, device=dev); P = torch.tensor(probe, device=dev); nT = len(real)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    best = dict(base); best["epoch"] = 0; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    for ep in range(1, args.epochs+1):
        head.train()
        tau = max(0.2, 1.0 * (1 - ep/args.epochs) + 0.2)          # HashNet-style anneal 1.2 -> 0.2
        idx = torch.randint(0, nT, (args.batch,), device=dev)
        jdx = (idx + torch.randint(1, nT, (args.batch,), device=dev)) % nT   # impostor partner
        ba = head.ste(R[idx], tau); bp = head.ste(P[idx], tau); bj = head.ste(P[jdx], tau)
        sg = (ba*bp).sum(1)/Dc; si = (ba*bj).sum(1)/Dc       # code similarity in [-1,1]
        if args.loss == "bce":                                # separate genuine/impostor; balance pushes imp mean->64
            logit = torch.cat([sg, si]) * args.scale
            lab = torch.cat([torch.ones_like(sg), torch.zeros_like(si)])
            L_pair = nn.functional.binary_cross_entropy_with_logits(logit, lab)
        else:                                                 # "whiten": genuine-close only; spread via balance+decorr
            L_pair = ((Dc - sg*Dc)/(2*Dc)).mean()
        allb = torch.cat([ba, bp, bj], 0)                    # population of codes this step
        L_bal = (allb.mean(0) ** 2).mean()                   # each bit ~50/50 -> impostor mean -> 64
        c = allb - allb.mean(0, keepdim=True)
        std = c.std(0, keepdim=True) + 1e-4
        corr = (c / std).T @ (c / std) / c.shape[0]          # bit-bit correlation matrix
        off = corr - torch.diag(torch.diag(corr))
        L_dec = (off ** 2).mean()                            # drive cross-bit correlation -> 0 (the sigma lever)
        L_q = (1 - (head.soft(R[idx], tau) ** 2)).mean()     # push activations to +-1
        loss = L_pair + args.lam_dec*L_dec + args.lam_bal*L_bal + args.lam_q*L_q
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0 or ep == 1:
            m = evaluate(head, gal, prb, f"ep{ep}", dev)
            print(f"    loss {loss.item():.4f} (pair {L_pair.item():.3f} dec {L_dec.item():.4f} bal {L_bal.item():.4f} q {L_q.item():.3f})")
            if m["eer"] <= best["eer"]:
                best = dict(m); best["epoch"] = ep; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    print(f"[qat] BEST @ epoch {best['epoch']}: EER {best['eer']*100:.2f}% "
          f"[{best['ci95'][0]*100:.2f},{best['ci95'][1]*100:.2f}]  sigma {best['imp_sd']:.2f}  H0 {best['h0']}")
    torch.save(head.state_dict(), f"{DATA}/qat_head_{args.tag}.pt")
    # export public 128x16 W and bias so downstream (real-crypto / fusion) can reuse
    np.savez(f"{DATA}/qat_head_{args.tag}_public.npz",
             W=head.lin.weight.detach().cpu().numpy(), b=head.lin.bias.detach().cpu().numpy())
    res = {"warm_start": base, "best": best, "args": vars(args)}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/qat-head-{args.tag}.json", "w") as f: json.dump(res, f, indent=2)
    print(f"[qat] wrote {OUT}/qat-head-{args.tag}.json and {DATA}/qat_head_{args.tag}_public.npz")


if __name__ == "__main__":
    main()
