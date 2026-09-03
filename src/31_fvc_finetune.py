"""
Step 31: generality test #2 — FVC2002. The SOCOFing CNN does not transfer to FVC (float EER
22-38%), so we FINE-TUNE it on pooled FVC2002 train fingers (warm-started from feature_model_224),
then run feature-head QAT vs ortho-thermometer per DB on held-out test fingers.

Protocol: per DB, finger-disjoint split; enroll = impression 1, probe = impressions 2..8.
Fine-tune the shared FeatureModel on pooled train pairs (genuine = same (db,finger); impostor =
different, same DB). Reports per DB: float EER (conv/16-D), ortho-thermo EER, feature-head EER,
sigma, finger-bootstrap CI. Tests whether the round-2 quantizer win reproduces on a 2nd real corpus.

Run: python3 31_fvc_finetune.py --epochs 40
"""
import argparse, json, os, glob, re
from importlib import import_module
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from model import SiameseModel

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); IMG, Dc = 224, 128
FVC = "/mnt/SharedData/fvc/fvc2002/FVC2002/Dbs"; DBS = ["Db1_a", "Db2_a", "Db3_a", "Db4_a"]
_p = import_module("30_polyu_featurehead"); code_eer_ci, eer = _p.code_eer_ci, _p.eer
_q = import_module("23_qat_head"); orthothermo_params, QATHead = _q.orthothermo_params, _q.QATHead
_d = import_module("25_qat_downstream")


def img224(p):
    return np.asarray(Image.open(p).convert("L").resize((IMG, IMG), Image.BILINEAR), np.float32)


def load_db(db):
    fing = {}
    for f in sorted(glob.glob(f"{FVC}/{db}/*.tif")):
        mm = re.match(r"(\d+)_(\d+)\.tif", os.path.basename(f))
        if mm: fing.setdefault(int(mm.group(1)), []).append((int(mm.group(2)), f))
    return {k: [img224(p) for _, p in sorted(v)] for k, v in fing.items()}


def augment(a):
    im = Image.fromarray(a.astype(np.uint8)); ang = np.random.uniform(-12, 12); tx, ty = np.random.uniform(-0.05, 0.05, 2)*IMG
    im = im.rotate(ang, resample=Image.BILINEAR, fillcolor=255, translate=(int(tx), int(ty)))
    return np.clip(np.asarray(im, np.float32)+np.random.normal(0, 5, (IMG, IMG)), 0, 255).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--head-epochs", type=int, default=400)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dbdata = {db: load_db(db) for db in DBS}
    rng = np.random.default_rng(args.seed)
    split = {}
    for db in DBS:
        fids = sorted(dbdata[db]); rng.shuffle(fids); nte = int(len(fids)*args.test_frac)
        split[db] = {"test": set(fids[:nte]), "train": set(fids[nte:])}
    print(f"[fvc] loaded {DBS}; per-DB ~{len(next(iter(dbdata.values())))} fingers", flush=True)

    # ---- fine-tune shared FeatureModel on pooled train pairs ----
    model = SiameseModel(img=IMG).to(dev)
    model.feature_model.load_state_dict(torch.load(f"{DATA}/feature_model_224.pt", map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda")); bce = nn.BCEWithLogitsLoss()
    pool = [(db, f) for db in DBS for f in split[db]["train"]]
    steps = max(1, (len(pool)*8)//args.batch)
    for ep in range(1, args.epochs+1):
        model.train(); run = 0.0
        for _ in range(steps):
            A = np.empty((args.batch, IMG, IMG), np.float32); B = np.empty_like(A); y = np.empty(args.batch, np.float32)
            for i in range(args.batch):
                db, f = pool[np.random.randint(len(pool))]; imgs = dbdata[db][f]
                A[i] = augment(imgs[np.random.randint(len(imgs))])
                if np.random.rand() < 0.5:
                    B[i] = augment(imgs[np.random.randint(len(imgs))]); y[i] = 1.0
                else:
                    g = f
                    tf = list(split[db]["train"])
                    while g == f: g = tf[np.random.randint(len(tf))]
                    B[i] = augment(dbdata[db][g][np.random.randint(len(dbdata[db][g]))]); y[i] = 0.0
            a = torch.from_numpy(A/255.).unsqueeze(1).to(dev); b = torch.from_numpy(B/255.).unsqueeze(1).to(dev)
            yt = torch.from_numpy(y).to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                loss = bce(model(a, b), yt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run += loss.item()
        if ep % 5 == 0 or ep == 1:
            print(f"  [fvc-ft] epoch {ep}/{args.epochs} loss {run/steps:.4f}", flush=True)
    fm = model.feature_model; fm.eval()
    torch.save(fm.state_dict(), f"{DATA}/feature_model_fvc2002.pt")

    def conv(arrs, bs=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(arrs), bs):
                xb = torch.from_numpy(np.stack(arrs[i:i+bs])/255.).unsqueeze(1).to(dev)
                f = torch.flatten(fm.features(xb), 1); f = f/f.norm(dim=1, keepdim=True); out.append(f.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def emb16(arrs, bs=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(arrs), bs):
                xb = torch.from_numpy(np.stack(arrs[i:i+bs])/255.).unsqueeze(1).to(dev)
                out.append(fm(xb).cpu().numpy())
        return np.concatenate(out)

    results = {}
    for db in DBS:
        trainf = sorted(split[db]["train"]); testf = sorted(split[db]["test"])
        tr_arrs = [im for f in trainf for im in dbdata[db][f]]
        Ctr = conv(tr_arrs); E16tr = emb16(tr_arrs)
        Xt = torch.tensor(Ctr-Ctr.mean(0), device=dev)
        _, S, V = torch.pca_lowrank(Xt, q=min(args.pca+32, Xt.shape[0]-1, Xt.shape[1]), center=False, niter=4)
        mu = Ctr.mean(0); comp = V[:, :args.pca].T.cpu().numpy().astype(np.float64); del Xt
        proj = lambda F: (F.astype(np.float64)-mu) @ comp.T
        Ptr = proj(Ctr)
        # train-finger row groups
        rows = {}; r = 0
        for f in trainf:
            for _ in dbdata[db][f]: rows.setdefault(f, []).append(r); r += 1
        fl = [f for f in trainf if len(rows[f]) >= 2]
        # eval sets
        g_arr = [dbdata[db][f][0] for f in testf]
        p_arr = [dbdata[db][f][j] for f in testf for j in range(1, len(dbdata[db][f]))]
        p_fin = np.array([f for f in testf for j in range(1, len(dbdata[db][f]))])
        Cg, Cp = conv(g_arr), conv(p_arr); E16g, E16p = emb16(g_arr), emb16(p_arr)
        Pg, Pp = proj(Cg), proj(Cp)

        def float_eer(G, P):
            D = np.sqrt(np.clip((P*P).sum(1)[:, None]+(G*G).sum(1)[None, :]-2*P@G.T, 0, None))
            gi = np.array([testf.index(f) for f in p_fin]); gen = D[np.arange(len(P)), gi]
            mask = np.ones_like(D, bool); mask[np.arange(len(P)), gi] = False
            return eer(gen, D[mask])
        fe_conv, fe_16 = float_eer(Cg, Cp), float_eer(E16g, E16p)
        og, op = _d.orthothermo(E16tr.astype(np.float64), E16g.astype(np.float64), E16p.astype(np.float64))
        o_pt, o_lo, o_hi, o_sd = code_eer_ci(og, op, p_fin, testf)

        # feature-head
        W0, b0 = orthothermo_params(Ptr, K=128, Bp=1); head = QATHead(W0, b0).to(dev)
        R = torch.tensor(Ptr, dtype=torch.float32, device=dev); opt2 = torch.optim.Adam(head.parameters(), lr=3e-4)
        best = {"eer": 1.0}; bstate = {k: v.clone() for k, v in head.state_dict().items()}
        for ep in range(1, args.head_epochs+1):
            head.train(); tau = max(0.2, 1.0*(1-ep/args.head_epochs)+0.2)
            fa = [fl[np.random.randint(len(fl))] for _ in range(512)]
            ra = [rows[f][np.random.randint(len(rows[f]))] for f in fa]; rb = [rows[f][np.random.randint(len(rows[f]))] for f in fa]
            Aa = head.ste(R[ra], tau); Bb = head.ste(R[rb], tau)
            dg = (Dc-(Aa*Bb).sum(1))/(2*Dc); L_gen = dg.mean()
            allb = torch.cat([Aa, Bb], 0); L_bal = (allb.mean(0)**2).mean()
            c = allb-allb.mean(0, keepdim=True); std = c.std(0, keepdim=True)+1e-4
            corr = (c/std).T@(c/std)/c.shape[0]; off = corr-torch.diag(torch.diag(corr))
            L_dec = (off**2).mean(); L_q = (1-(head.soft(R[ra], tau)**2)).mean()
            loss = L_gen+30*L_dec+5*L_bal+0.1*L_q
            opt2.zero_grad(); loss.backward(); opt2.step()
            if ep % 25 == 0:
                head.eval()
                with torch.no_grad():
                    cg = (head.logits(torch.tensor(Pg, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
                    cp = (head.logits(torch.tensor(Pp, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
                pt = code_eer_ci(cg, cp, p_fin, testf, B=1)[0]
                if pt <= best["eer"]: best = {"eer": pt, "epoch": ep}; bstate = {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(bstate); head.eval()
        with torch.no_grad():
            cg = (head.logits(torch.tensor(Pg, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
            cp = (head.logits(torch.tensor(Pp, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
        f_pt, f_lo, f_hi, f_sd = code_eer_ci(cg, cp, p_fin, testf)
        print(f"[{db}] {len(trainf)}tr/{len(testf)}te | float conv {fe_conv*100:.2f}% 16-D {fe_16*100:.2f}% | "
              f"ortho {o_pt*100:.2f}% [{o_lo*100:.2f},{o_hi*100:.2f}] sd {o_sd:.1f} | "
              f"feat {f_pt*100:.2f}% [{f_lo*100:.2f},{f_hi*100:.2f}] sd {f_sd:.2f}", flush=True)
        results[db] = {"n_train": len(trainf), "n_test": len(testf), "float_conv": fe_conv, "float_16d": fe_16,
                       "ortho_thermo": {"eer": o_pt, "ci": [o_lo, o_hi], "sigma": o_sd},
                       "feature_head": {"eer": f_pt, "ci": [f_lo, f_hi], "sigma": f_sd}}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/fvc2002-featurehead.json", "w") as f: json.dump(results, f, indent=2)
    print(f"[fvc] wrote {OUT}/fvc2002-featurehead.json", flush=True)


if __name__ == "__main__":
    main()
