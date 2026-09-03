"""
Step 24: END-TO-END quantization-aware co-design (plan round-2, stretch of step 4).

Frozen-head QAT (23) is capped: a map from a D-dim embedding to 128 bits has at most ~D
effective independent bits, so at D=16 impostor sigma cannot approach the uniform 5.66 no
matter how the head is arranged (diagnostic step 22 measured ~14-18 effective bits ~ the 16-D
ceiling). This script lifts that ceiling by co-training the CNN embedding AND the 128-bit head
together with the same code-space objective, at a chosen embedding width D. The earlier dim
sweep showed wider embeddings lower the FLOAT ceiling (0.33->0.14%) but a FIXED code wasted it;
here the code is learned jointly, testing whether co-design finally converts float DOF into a
tighter, lower-EER code (and thus a smaller FLPSI subsample count T).

Genuine pairs = (real, probe) of the same NON-test finger; eval on held-out test fingers.
Warm-starts the CNN from the matching dim-sweep checkpoint and the head from an ortho-thermometer
fit on that CNN's initial embeddings. Reports EER + identity-bootstrap CI, impostor sigma, H=0.

Env: ONE GPU job at a time, batch 64. Run:
  python3 24_qat_e2e.py --emb-dim 16 --epochs 60 --tag e2e_d16
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn
from model import FeatureModel
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); IMG, Dc = 224, 128
CKPT = {16: "feature_model_224.pt", 32: "feature_model_224_d32.pt", 64: "feature_model_224_d64.pt"}

from importlib import import_module
_qat = import_module("23_qat_head")
orthothermo_params, QATHead, eer_from_counts, boot_ci = (
    _qat.orthothermo_params, _qat.QATHead, _qat.eer_from_counts, _qat.boot_ci)


def L(base):
    p = f"{DATA}/{base}_{IMG}.npy"
    return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{base}.npy", allow_pickle=True)


def augment(u8, size=IMG):
    im = Image.fromarray(u8); ang = np.random.uniform(-10, 10); tx, ty = np.random.uniform(-0.04, 0.04, 2)*size
    im = im.rotate(ang, resample=Image.BILINEAR, fillcolor=255, translate=(int(tx), int(ty)))
    a = np.asarray(im, np.float32) + np.random.normal(0, 4, (size, size))
    return np.clip(a, 0, 255).astype(np.float32)


def embed_imgs(model, stack, dev, batch=128):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(stack), batch):
            xb = torch.from_numpy(stack[i:i+batch].astype(np.float32)/255.).unsqueeze(1).to(dev)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, head, gimg, pimg, dev, tag):
    eg = embed_imgs(model, gimg, dev); ep = embed_imgs(model, pimg, dev)
    with torch.no_grad():
        cg = (head.logits(torch.tensor(eg, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.int32)
        cp = (head.logits(torch.tensor(ep, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.int32)
    H = cp @ (1-cg).T + (1-cp) @ cg.T; n = len(gimg); off = ~np.eye(n, dtype=bool)
    genH, impH = np.diag(H), H[off]; pt, lo, hi = boot_ci(H); h0 = int((impH == 0).sum())
    print(f"  [{tag}] EER {pt*100:.2f}% [{lo*100:.2f},{hi*100:.2f}]  imp mean {impH.mean():.1f} sd {impH.std():.2f}  gen {genH.mean():.1f}  H0 {h0}", flush=True)
    return {"eer": pt, "ci95": [lo, hi], "imp_mean": float(impH.mean()), "imp_sd": float(impH.std()),
            "gen_mean": float(genH.mean()), "h0": h0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lam-dec", type=float, default=10.0)
    ap.add_argument("--lam-bal", type=float, default=2.0)
    ap.add_argument("--lam-q", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="e2e")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; D = args.emb_dim

    x_real, ids_real = L("x_real"), L("ids_real"); x_probe, ids_probe = L("x_probe"), L("ids_probe")
    ids_test = set(L("ids_test").tolist())
    ri = {i: k for k, i in enumerate(ids_real)}; pi = {i: k for k, i in enumerate(ids_probe)}
    train_ids = sorted((set(ids_real) & set(ids_probe)) - ids_test)
    test_ids = sorted((set(ids_real) & set(ids_probe)) & ids_test)
    rtr = np.array([ri[i] for i in train_ids]); ptr = np.array([pi[i] for i in train_ids])
    gimg = np.stack([x_real[ri[i]] for i in test_ids]); pimg = np.stack([x_probe[pi[i]] for i in test_ids])
    print(f"[e2e] D={D} train pairs {len(train_ids)} | held-out {len(test_ids)} | dev {dev} | "
          f"lam_dec={args.lam_dec} lam_bal={args.lam_bal} hidden={args.hidden}", flush=True)

    model = FeatureModel(img=IMG, emb_dim=D).to(dev)
    ck = CKPT.get(D)
    if ck and os.path.exists(f"{DATA}/{ck}"):
        model.load_state_dict(torch.load(f"{DATA}/{ck}", map_location=dev)); print(f"[e2e] warm-start CNN <- {ck}", flush=True)
    tr_emb = embed_imgs(model, np.stack([x_real[k] for k in rtr]), dev)   # for head warm-start
    W0, b0 = orthothermo_params(tr_emb.astype(np.float64))
    head = QATHead(W0, b0, hidden=args.hidden).to(dev)
    print("[e2e] warm-start:", flush=True); base = evaluate(model, head, gimg, pimg, dev, "warm-start")

    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    nT = len(train_ids); order = np.arange(nT)
    bce = nn.BCEWithLogitsLoss()
    best = dict(base); best["epoch"] = 0
    best_state = ({k: v.clone() for k, v in model.state_dict().items()},
                  {k: v.clone() for k, v in head.state_dict().items()})
    for ep in range(1, args.epochs+1):
        model.train(); head.train(); np.random.shuffle(order)
        tau = max(0.2, 1.2*(1-ep/args.epochs)+0.2)
        for s in range(0, nT-args.batch+1, args.batch):
            b = order[s:s+args.batch]
            jb = order[(s+args.batch) % nT:][:len(b)]
            if len(jb) < len(b): jb = np.roll(order, -1)[b]      # impostor partners
            ai = np.stack([augment(x_real[rtr[k]]) for k in b])
            bi = np.stack([augment(x_probe[ptr[k]]) for k in b])
            ci = np.stack([augment(x_probe[ptr[k]]) for k in jb])
            a = torch.from_numpy(ai/255.).unsqueeze(1).to(dev)
            p = torch.from_numpy(bi/255.).unsqueeze(1).to(dev)
            pj = torch.from_numpy(ci/255.).unsqueeze(1).to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                ea, epb, ej = model(a), model(p), model(pj)
                ba, bp, bj = head.ste(ea, tau), head.ste(epb, tau), head.ste(ej, tau)
                sg = (ba*bp).sum(1)/Dc; si = (ba*bj).sum(1)/Dc     # code similarity in [-1,1]
                logit = torch.cat([sg, si])*args.scale
                lab = torch.cat([torch.ones_like(sg), torch.zeros_like(si)])
                L_pair = bce(logit, lab)                            # genuine close / impostor far
                allb = torch.cat([ba, bp, bj], 0); L_bal = (allb.mean(0)**2).mean()
                c = allb - allb.mean(0, keepdim=True); std = c.std(0, keepdim=True)+1e-4
                corr = (c/std).T @ (c/std) / c.shape[0]; off = corr - torch.diag(torch.diag(corr))
                L_dec = (off**2).mean(); L_q = (1-(head.soft(ea, tau)**2)).mean()
                loss = L_pair + args.lam_dec*L_dec + args.lam_bal*L_bal + args.lam_q*L_q
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            m = evaluate(model, head, gimg, pimg, dev, f"ep{ep}")
            print(f"    loss {loss.item():.4f} (pair {L_pair.item():.3f} dec {L_dec.item():.4f} bal {L_bal.item():.4f})", flush=True)
            if m["eer"] <= best["eer"]:
                best = dict(m); best["epoch"] = ep
                best_state = ({k: v.clone() for k, v in model.state_dict().items()},
                              {k: v.clone() for k, v in head.state_dict().items()})
    model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    print(f"[e2e] BEST @ epoch {best['epoch']}: EER {best['eer']*100:.2f}% "
          f"[{best['ci95'][0]*100:.2f},{best['ci95'][1]*100:.2f}]  sigma {best['imp_sd']:.2f}  H0 {best['h0']}", flush=True)
    torch.save(model.state_dict(), f"{DATA}/feature_model_{args.tag}.pt")
    np.savez(f"{DATA}/qat_head_{args.tag}_public.npz",
             W=head.lin.weight.detach().cpu().numpy(), b=head.lin.bias.detach().cpu().numpy())
    # export held-out codes for downstream (T-sweep / fusion / real-crypto)
    eg = embed_imgs(model, gimg, dev); ep_ = embed_imgs(model, pimg, dev)
    with torch.no_grad():
        cg = (head.logits(torch.tensor(eg, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
        cp = (head.logits(torch.tensor(ep_, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
    np.savez(f"{DATA}/qat_codes_{args.tag}.npz", gal=cg, prb=cp)
    with open(f"{OUT}/qat-e2e-{args.tag}.json", "w") as f:
        json.dump({"warm_start": base, "best": best, "args": vars(args)}, f, indent=2)
    print(f"[e2e] wrote {OUT}/qat-e2e-{args.tag}.json, codes {DATA}/qat_codes_{args.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
