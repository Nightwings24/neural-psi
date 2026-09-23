"""
Step 34: all-pairs float EER on PolyU test fingers, matching the Blind-Touch protocol
(genuine = every C(6,2) within-finger pair; impostor = one pair per distinct finger pair).
Compares directly to Blind-Touch's PolyU 2.5% EER. Uses the same finger split as
30_polyu_featurehead.py (seed / test-frac), so the CNN never saw the test fingers.

Run: python3 34_polyu_allpairs.py --modality cless --ckpt feature_model_polyu_ss_cless.pt
"""
import argparse, os
from itertools import combinations
import numpy as np
import torch
from model import FeatureModel

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); IMG = 224


def eer(gen, imp):
    sg = np.sort(gen); si = np.sort(imp); c = np.unique(np.concatenate([sg, si]))
    frr = 1 - np.searchsorted(sg, c, "right")/len(sg); far = np.searchsorted(si, c, "right")/len(imp)
    k = int(np.argmin(np.abs(frr-far))); return float((frr[k]+far[k])/2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="cless")
    ap.add_argument("--ckpt", default="feature_model_polyu_ss_cless.pt")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--test-frac", type=float, default=0.35)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = FeatureModel(img=IMG).to(dev); m.load_state_dict(torch.load(f"{DATA}/{args.ckpt}", map_location=dev)); m.eval()
    x = np.load(f"{DATA}/polyu_{args.modality}_224.npy")
    ids = np.load(f"{DATA}/polyu_{args.modality}_ids.npy", allow_pickle=True).astype(str)
    by = {}
    for k, f in enumerate(ids): by.setdefault(f, []).append(k)
    fids = sorted(by, key=lambda s: int(s) if s.isdigit() else s)
    rng = np.random.default_rng(args.seed); rng.shuffle(fids)
    nte = int(len(fids)*args.test_frac); test_f = [f for f in fids[:nte] if len(by[f]) >= 2]

    idx = np.array([k for f in test_f for k in by[f]]); fin = np.array([f for f in test_f for _ in by[f]])
    def emb(kind):
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                xb = torch.from_numpy(x[idx[i:i+64]].astype(np.float32)/255.).unsqueeze(1).to(dev)
                if kind == "conv":
                    f = torch.flatten(m.features(xb), 1); f = f/f.norm(dim=1, keepdim=True)
                else:
                    f = m(xb)
                out.append(f.cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    pos = {f: np.where(fin == f)[0] for f in test_f}
    for kind in ("16d", "conv"):
        E = emb(kind)
        gen = [np.linalg.norm(E[a]-E[b]) for f in test_f for a, b in combinations(pos[f], 2)]
        # one impostor pair per distinct finger pair: first sample of each
        firsts = {f: pos[f][0] for f in test_f}
        imp = [np.linalg.norm(E[firsts[a]]-E[firsts[b]]) for a, b in combinations(test_f, 2)]
        print(f"[allpairs-{kind}] {len(test_f)} test fingers | genuine {len(gen)} imp {len(imp)} | "
              f"float EER {eer(np.array(gen), np.array(imp))*100:.2f}%", flush=True)
    print("[allpairs] Blind-Touch reference (contactless-2D PolyU, 150 ep): 2.5% EER", flush=True)


if __name__ == "__main__":
    main()
