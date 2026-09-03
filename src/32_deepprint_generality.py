"""
Step 32: generality with a STRONG pretrained backbone (DeepPrint, Rohwedder reimpl, 512-D).
Tests whether, given a strong general extractor, (a) DeepPrint transfers to PolyU/FVC with good
float EER (the precondition our own small CNN failed), and if so (b) the QAT feature-head beats
the ortho-thermometer at quantizing the 512-D embedding to a 128-bit code (i.e. the quantization
gap re-appears and our method closes it).

Uses the pretrained DeepPrint_Tex 512-D model. No training of the backbone. Per corpus:
gallery = first sample/impression, probes = rest; finger-disjoint train/test for the code stage.

Run: python3 32_deepprint_generality.py --weights <best_model.pt> --corpus fvc_Db1_a --float-only
     python3 32_deepprint_generality.py --weights <best_model.pt> --corpus polyu_cless
"""
import argparse, glob, json, os, re, sys
from importlib import import_module
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results"))
FLX = "/tmp/claude-1000/-home-nightwings-neural-psi/74d72758-4a8c-40b0-807e-eac648ecb2b7/scratchpad/flx_repo"
FVC = "/mnt/SharedData/fvc/fvc2002/FVC2002/Dbs"; Dc = 128
sys.path.insert(0, FLX)
_p = import_module("30_polyu_featurehead"); code_eer_ci, eer = _p.code_eer_ci, _p.eer
_q = import_module("23_qat_head"); orthothermo_params, QATHead = _q.orthothermo_params, _q.QATHead
_d = import_module("25_qat_downstream")


def _stub_torchvision():
    # DeepPrint_Tex never uses the localization net (only DeepPrint_Loc* does); stub torchvision
    # so we avoid installing a version mismatched to this custom torch build.
    import types
    if "torchvision" in sys.modules: return
    tv = types.ModuleType("torchvision"); tvt = types.ModuleType("torchvision.transforms")
    class _R:
        def __init__(self, *a, **k): pass
        def __call__(self, x): return x
    tvt.Resize = _R; tv.transforms = tvt
    sys.modules["torchvision"] = tv; sys.modules["torchvision.transforms"] = tvt


def load_deepprint(weights, dev):
    _stub_torchvision()
    from flx.models.deep_print_arch import DeepPrint_TexMinu
    ck = torch.load(weights, map_location="cpu")
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    nf = sd["texture_logits.0.weight"].shape[0]
    tex = sd["texture_branch._6_linear.weight"].shape[0]
    minu = sd["minutia_embedding._4_linear.weight"].shape[0]
    model = DeepPrint_TexMinu(nf, tex, minu)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[dp] TexMinu tex={tex} minu={minu} num_train_fp={nf} | missing {len(missing)} unexpected {len(unexpected)}", flush=True)
    return model.to(dev).eval(), tex+minu


def embed(model, imgs_u8, dev, bs=32):
    """-> (N, tex+minu) float: per-branch L2-normalised texture||minutia embedding (DeepPrint fusion)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs_u8), bs):
            batch = []
            for a in imgs_u8[i:i+bs]:
                im = Image.fromarray(a).convert("L")                 # DeepPrint preprocessing:
                w, h = im.size; s = max(w, h)                        # pad to square with white
                cv = Image.new("L", (s, s), 255)
                cv.paste(im, ((s-w)//2, (s-h)//2))
                cv = cv.resize((299, 299), Image.BILINEAR)           # then resize to 299
                batch.append(np.asarray(cv, np.float32)/255.)
            x = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(dev)
            o = model(x)
            t = torch.nn.functional.normalize(o.texture_embeddings, dim=1)
            mn = torch.nn.functional.normalize(o.minutia_embeddings, dim=1)
            out.append(torch.cat([t, mn], 1).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def load_corpus(corpus):
    """returns dict finger_id -> list of uint8 images (>=2 samples)."""
    if corpus.startswith("polyu_"):
        tag = corpus.split("_", 1)[1]
        x = np.load(f"{DATA}/polyu_{tag}_224.npy"); ids = np.load(f"{DATA}/polyu_{tag}_ids.npy", allow_pickle=True).astype(str)
        by = {}
        for k, f in enumerate(ids): by.setdefault(f, []).append(x[k])
        return {f: v for f, v in by.items() if len(v) >= 2}
    if corpus.startswith("fvc_"):
        db = corpus.split("_", 1)[1]
        by = {}
        for fp in sorted(glob.glob(f"{FVC}/{db}/*.tif")):
            mm = re.match(r"(\d+)_(\d+)\.tif", os.path.basename(fp))
            if mm: by.setdefault(mm.group(1), []).append((int(mm.group(2)), fp))
        return {f: [np.asarray(Image.open(p).convert("L")) for _, p in sorted(v)] for f, v in by.items() if len(v) >= 2}
    raise ValueError(corpus)


def float_eer(G, P, gfids, pf):
    D = np.sqrt(np.clip((P*P).sum(1)[:, None]+(G*G).sum(1)[None, :]-2*P@G.T, 0, None))
    gi = np.array([gfids.index(f) for f in pf]); gen = D[np.arange(len(P)), gi]
    mask = np.ones_like(D, bool); mask[np.arange(len(P)), gi] = False
    return eer(gen, D[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--float-only", action="store_true")
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, D = load_deepprint(args.weights, dev)

    by = load_corpus(args.corpus); fids = sorted(by)
    rng = np.random.default_rng(args.seed); rng.shuffle(fids)
    nte = int(len(fids)*args.test_frac); test_f = fids[:nte]; train_f = fids[nte:]

    # embed test gallery(sample0)+probes(rest)
    g_imgs = [by[f][0] for f in test_f]; p_imgs = [by[f][j] for f in test_f for j in range(1, len(by[f]))]
    p_fin = np.array([f for f in test_f for j in range(1, len(by[f]))])
    Eg, Ep = embed(model, g_imgs, dev), embed(model, p_imgs, dev)
    fe = float_eer(Eg, Ep, test_f, p_fin)
    print(f"[{args.corpus}] {len(train_f)}tr/{len(test_f)}te | DeepPrint-512 float EER {fe*100:.2f}%", flush=True)
    res = {"corpus": args.corpus, "n_train": len(train_f), "n_test": len(test_f), "float_eer_deepprint512": fe}

    if not args.float_only:
        # ortho-thermo(512->128) vs QAT head(512->128), head trained on train fingers
        tr_imgs = [im for f in train_f for im in by[f]]; tr_fin = [f for f in train_f for _ in by[f]]
        Etr = embed(model, tr_imgs, dev)
        og, op = _d.orthothermo(Etr, Eg, Ep)  # ortho-thermo on 512-D
        o_pt, o_lo, o_hi, o_sd = code_eer_ci(og, op, p_fin, test_f)
        # PCA -> QAT head
        Xt = torch.tensor(Etr - Etr.mean(0), device=dev, dtype=torch.float32)
        _, S, V = torch.pca_lowrank(Xt, q=min(args.pca+16, Xt.shape[0]-1, Xt.shape[1]), center=False, niter=4)
        mu = Etr.mean(0); comp = V[:, :args.pca].T.cpu().numpy().astype(np.float64); del Xt
        pj = lambda F: (F - mu) @ comp.T
        Ptr, Pg, Pp = pj(Etr), pj(Eg), pj(Ep)
        rows = {}
        for r, f in enumerate(tr_fin): rows.setdefault(f, []).append(r)
        fl = [f for f in train_f if len(rows[f]) >= 2]
        W0, b0 = orthothermo_params(Ptr, K=128, Bp=1); head = QATHead(W0, b0).to(dev)
        R = torch.tensor(Ptr, dtype=torch.float32, device=dev); opt = torch.optim.Adam(head.parameters(), lr=3e-4)
        best = {"eer": 1.0}; bstate = {k: v.clone() for k, v in head.state_dict().items()}
        for ep in range(1, args.epochs+1):
            head.train(); tau = max(0.2, 1.0*(1-ep/args.epochs)+0.2)
            fa = [fl[np.random.randint(len(fl))] for _ in range(512)]
            ra = [rows[f][np.random.randint(len(rows[f]))] for f in fa]; rb = [rows[f][np.random.randint(len(rows[f]))] for f in fa]
            A = head.ste(R[ra], tau); Bb = head.ste(R[rb], tau)
            dg = (Dc-(A*Bb).sum(1))/(2*Dc); Lg = dg.mean()
            allb = torch.cat([A, Bb], 0); Lb = (allb.mean(0)**2).mean()
            c = allb-allb.mean(0, keepdim=True); std = c.std(0, keepdim=True)+1e-4
            corr = (c/std).T@(c/std)/c.shape[0]; off = corr-torch.diag(torch.diag(corr))
            Ld = (off**2).mean(); Lq = (1-(head.soft(R[ra], tau)**2)).mean()
            loss = Lg+30*Ld+5*Lb+0.1*Lq; opt.zero_grad(); loss.backward(); opt.step()
            if ep % 25 == 0:
                head.eval()
                with torch.no_grad():
                    cg = (head.logits(torch.tensor(Pg, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
                    cp = (head.logits(torch.tensor(Pp, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
                pt = code_eer_ci(cg, cp, p_fin, test_f, B=1)[0]
                if pt <= best["eer"]: best = {"eer": pt, "epoch": ep}; bstate = {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(bstate); head.eval()
        with torch.no_grad():
            cg = (head.logits(torch.tensor(Pg, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
            cp = (head.logits(torch.tensor(Pp, dtype=torch.float32, device=dev)) > 0).cpu().numpy().astype(np.uint8)
        f_pt, f_lo, f_hi, f_sd = code_eer_ci(cg, cp, p_fin, test_f)
        print(f"  ortho-thermo(512->128) EER {o_pt*100:.2f}% [{o_lo*100:.2f},{o_hi*100:.2f}] sd {o_sd:.1f}", flush=True)
        print(f"  feature-head(512->128)  EER {f_pt*100:.2f}% [{f_lo*100:.2f},{f_hi*100:.2f}] sd {f_sd:.2f}", flush=True)
        res["ortho_thermo"] = {"eer": o_pt, "ci": [o_lo, o_hi], "sigma": o_sd}
        res["feature_head"] = {"eer": f_pt, "ci": [f_lo, f_hi], "sigma": f_sd}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/deepprint-{args.corpus}.json", "w") as f: json.dump(res, f, indent=2)
    print(f"[dp] wrote {OUT}/deepprint-{args.corpus}.json", flush=True)


if __name__ == "__main__":
    main()
