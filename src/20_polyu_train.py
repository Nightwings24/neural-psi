"""
Step 20: PolyU cross-sensor training + evaluation (E4 proper - not zero-shot).

Trains a Siamese CNN on PolyU with GENUINE pairs = (contact, contactless) of the SAME finger
and impostor pairs = different fingers, on a finger-disjoint train split. Then evaluates the
real cross-sensor task on held-out fingers: enroll = contact, probe = contactless.
Reports float EER, Super-Bit-128 EER, L2-thermometer EER, impostor sd, H=0 collision floor.

This tests whether (a) the system works at all under a real sensor change when properly
trained, and (b) the collision-floor / metric-mismatch phenomena reproduce on a second,
genuinely different corpus - the generalization the SOCOFing-only results cannot claim.

Run (after the GPU dim-sweep frees the card):
  python3 20_polyu_train.py --emb-dim 16 --epochs 120 --batch 64
"""
import argparse, json, os
from math import comb
import numpy as np
from scipy import stats
from PIL import Image
import torch, torch.nn as nn
from model import SiameseModel, FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results"))


def augment(img_u8, size):
    im = Image.fromarray(img_u8)
    ang = np.random.uniform(-12, 12); tx, ty = np.random.uniform(-0.05, 0.05, 2)*size
    im = im.rotate(ang, resample=Image.BILINEAR, fillcolor=255, translate=(int(tx), int(ty)))
    a = np.asarray(im, np.float32) + np.random.normal(0, 5, (size, size))
    return np.clip(a, 0, 255).astype(np.float32)


def embed(model, x, dev, batch=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i+batch].astype(np.float32)/255.).unsqueeze(1).to(dev)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def eer_dist(gen, imp):
    s = np.concatenate([gen, imp]); best, e = 1.0, 0.0
    for t in np.linspace(s.min(), s.max(), 2000):
        frr = np.mean(gen > t); far = np.mean(imp <= t)
        if abs(frr-far) < best: best, e = abs(frr-far), (frr+far)/2
    return e

def hamm(A, B):
    A = A.astype(np.int32); B = B.astype(np.int32); return A@(1-B).T+(1-A)@B.T

def superbit(tr, g, p, D):
    q = NeuralPSIQuantizer.fit(tr, np.ones(D), QuantizerConfig(center=True, whiten=False, superbit=True, balance=True, in_dim=D))
    return q.transform(g).astype(np.uint8), q.transform(p).astype(np.uint8)
def thermo(tr, g, p, K=16, Bp=8, seed=1):
    r = np.random.default_rng(seed); A = r.standard_normal((K, tr.shape[1])); mu = tr.mean(0)
    pj = (tr-mu)@A.T; qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)
    f = lambda X: (((X-mu)@A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(g), f(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = args.emb_dim

    cx = np.load(f"{DATA}/polyu_contact_{args.img}.npy"); cid = np.load(f"{DATA}/polyu_contact_ids.npy", allow_pickle=True).astype(str)
    lx = np.load(f"{DATA}/polyu_cless_{args.img}.npy");   lid = np.load(f"{DATA}/polyu_cless_ids.npy", allow_pickle=True).astype(str)
    fids = sorted(set(cid) & set(lid), key=lambda s: int(s) if s.isdigit() else s)
    rng = np.random.default_rng(args.seed); rng.shuffle(fids)
    n_test = int(len(fids)*args.test_frac); test_f = set(fids[:n_test]); train_f = set(fids[n_test:])
    print(f"[polyu-train] {len(train_f)} train / {len(test_f)} test fingers | emb_dim={D} dev={dev}")

    # Cross-modal pair dataset: genuine = (contact, contactless) same finger; impostor = different.
    class DS(torch.utils.data.Dataset):
        def __init__(self):
            self.byc = {}; self.byl = {}
            for k, f in enumerate(cid):
                if f in train_f: self.byc.setdefault(f, []).append(k)
            for k, f in enumerate(lid):
                if f in train_f: self.byl.setdefault(f, []).append(k)
            self.fl = [f for f in train_f if f in self.byc and f in self.byl]
        def __len__(self): return 6400
        def __getitem__(self, _):
            f = self.fl[np.random.randint(len(self.fl))]
            a = augment(cx[self.byc[f][np.random.randint(len(self.byc[f]))]], args.img)
            if np.random.rand() < 0.5:
                b = augment(lx[self.byl[f][np.random.randint(len(self.byl[f]))]], args.img); y = 1.0
            else:
                g = f
                while g == f: g = self.fl[np.random.randint(len(self.fl))]
                b = augment(lx[self.byl[g][np.random.randint(len(self.byl[g]))]], args.img); y = 0.0
            return (torch.from_numpy(a/255.).unsqueeze(0), torch.from_numpy(b/255.).unsqueeze(0), torch.tensor(y))
    dl = torch.utils.data.DataLoader(DS(), batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)

    model = SiameseModel(img=args.img, emb_dim=D).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda")); loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(1, args.epochs+1):
        model.train(); run = c = tot = 0
        for a, b, y in dl:
            a, b, y = a.to(dev, non_blocking=True), b.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                logit = model(a, b); loss = loss_fn(logit, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()*len(y); c += ((logit > 0).float() == y).sum().item(); tot += len(y)
        sched.step()
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"  epoch {ep}/{args.epochs} loss={run/tot:.4f} pair_acc={c/tot:.3f}", flush=True)
    torch.save(model.feature_model.state_dict(), f"{DATA}/feature_model_polyu_d{D}.pt")

    # ---- cross-sensor eval on held-out fingers: enroll=contact(1/finger), probe=contactless(rest) ----
    fm = model.feature_model
    te = sorted(test_f)
    byc = {}; byl = {}
    for k, f in enumerate(cid):
        if f in test_f: byc.setdefault(f, []).append(k)
    for k, f in enumerate(lid):
        if f in test_f: byl.setdefault(f, []).append(k)
    te = [f for f in te if f in byc and f in byl]
    gal_x = np.stack([cx[byc[f][0]] for f in te])                     # enroll: 1 contact per finger
    prb_x = np.concatenate([lx[[byl[f][j] for j in range(len(byl[f]))]] for f in te])
    prb_f = np.concatenate([[f]*len(byl[f]) for f in te])
    tr_c_idx = [k for k, f in enumerate(cid) if f in train_f]
    emb_tr = embed(fm, cx[tr_c_idx], dev)                            # bridge fit on train contact
    eg = embed(fm, gal_x, dev); ep_ = embed(fm, prb_x, dev)
    gi = np.array([te.index(f) for f in prb_f])
    def dmat(P, G):
        P2 = (P*P).sum(1); G2 = (G*G).sum(1); return np.sqrt(np.clip(P2[:, None]+G2[None, :]-2*P@G.T, 0, None))
    Df = dmat(ep_, eg); genf = Df[np.arange(len(ep_)), gi]
    mask = np.ones_like(Df, bool); mask[np.arange(len(ep_)), gi] = False; impf = Df[mask]
    res = {"emb_dim": D, "n_train_fingers": len(train_f), "n_test_fingers": len(te),
           "n_gal": len(eg), "n_probe": len(ep_), "eer_float_euclid": eer_dist(genf, impf)}
    for bn, (cg, cp) in [("superbit", superbit(emb_tr, eg, ep_, D)), ("thermo16x8", thermo(emb_tr, eg, ep_))]:
        H = hamm(cp, cg); genH = H[np.arange(len(ep_)), gi]; impH = H[mask]
        cc = int((impH == 0).sum()); lo = 0.0 if cc == 0 else stats.chi2.ppf(0.025, 2*cc)/2/impH.size
        hi = stats.chi2.ppf(0.975, 2*(cc+1))/2/impH.size
        res[bn] = {"eer": eer_dist(genH.astype(float), impH.astype(float)), "imp_mean": float(impH.mean()),
                   "imp_sd": float(impH.std()), "gen_mean": float(genH.mean()), "collisions": cc, "p0": cc/impH.size, "p0_ci": [lo, hi]}
    print(f"\n[polyu-train] CROSS-SENSOR (trained) enroll=contact probe=contactless, {len(te)} held-out fingers")
    print(f"  float EER {res['eer_float_euclid']*100:.2f}%")
    print(f"  Super-Bit EER {res['superbit']['eer']*100:.2f}%  sd {res['superbit']['imp_sd']:.2f}  H=0 {res['superbit']['collisions']}")
    print(f"  L2-thermo EER {res['thermo16x8']['eer']*100:.2f}%  sd {res['thermo16x8']['imp_sd']:.2f}  H=0 {res['thermo16x8']['collisions']}")
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/polyu-trained-d{D}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[polyu-train] wrote {OUT}/polyu-trained-d{D}.json")


if __name__ == "__main__":
    main()
