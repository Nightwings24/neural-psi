"""
Step 29: accuracy ladder (Altered Easy/Medium/Hard) for the feature-head QAT code vs the
ortho-thermometer baseline, on held-out TEST identities. Confirms the round-2 win holds across
difficulty, not just Altered-Easy.

Gallery = enrolled real print (test ids). Probe = altered print at each difficulty.
Feature-head code: frozen conv features -> saved public PCA map -> saved 128-bit head (map from
28_qat_featurehead.py, tag feat_final). Baseline: ortho-thermometer on the 16-D embedding
(fit on train), same as round 1.

Run: python3 29_qat_ladder.py --map data/qat_head_feat_final_map.npz
"""
import argparse, json, os
from importlib import import_module
import numpy as np
import torch
from model import FeatureModel

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); IMG, Dc = 224, 128
_q = import_module("23_qat_head"); boot_ci = _q.boot_ci
_d = import_module("25_qat_downstream")


def L(b):
    p = f"{DATA}/{b}_{IMG}.npy"
    return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{b}.npy", allow_pickle=True)


def eer_ci_sigma(cg, cp):
    H = cp.astype(np.int32) @ (1-cg.astype(np.int32)).T + (1-cp.astype(np.int32)) @ cg.astype(np.int32).T
    n = len(cg); off = ~np.eye(n, dtype=bool); imp = H[off]
    pt, lo, hi = boot_ci(H)
    return pt, lo, hi, float(imp.std()), int((imp == 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="data/qat_head_feat_final_map.npz")
    ap.add_argument("--tag", default="feat_final")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = FeatureModel(img=IMG).to(dev); m.load_state_dict(torch.load(f"{DATA}/feature_model_224.pt", map_location=dev)); m.eval()
    M = np.load(args.map if os.path.isabs(args.map) else os.path.join(HERE, args.map))
    mu, comp, scale, W, b = M["mu"], M["comp"], M["scale"], M["W"], M["b"]

    def conv_feat(stack):
        out = []
        with torch.no_grad():
            for i in range(0, len(stack), 128):
                xb = torch.from_numpy(stack[i:i+128].astype(np.float32)/255.).unsqueeze(1).to(dev)
                f = torch.flatten(m.features(xb), 1); f = f/f.norm(dim=1, keepdim=True); out.append(f.cpu().numpy())
        return np.concatenate(out)

    def emb16(stack):
        out = []
        with torch.no_grad():
            for i in range(0, len(stack), 128):
                xb = torch.from_numpy(stack[i:i+128].astype(np.float32)/255.).unsqueeze(1).to(dev)
                out.append(m(xb).cpu().numpy())
        return np.concatenate(out)

    def feat_code(F):
        z = ((F.astype(np.float64) - mu) @ comp.T) / (scale + 1e-8)
        return ((z @ W.T + b) > 0).astype(np.uint8)

    ids_test = set(L("ids_test").tolist())
    x_real, ids_real = L("x_real"), L("ids_real")
    ri = {i: k for k, i in enumerate(ids_real)}
    # 16-D ortho-thermo fit on train (dim_ablation)
    tr16 = np.load(f"{DATA}/dim_ablation_emb_224.npz")["train"].astype(np.float64)

    def load_probe(diff):
        if diff == "easy":
            return L("x_probe"), L("ids_probe")
        return (np.load(f"{DATA}/x_probe_{IMG}_{diff}.npy", allow_pickle=True),
                np.load(f"{DATA}/ids_probe_{IMG}_{diff}.npy", allow_pickle=True))

    rows = {}
    for diff in ["easy", "med", "hard"]:
        xp, ipr = load_probe(diff); pi = {i: k for k, i in enumerate(ipr)}
        common = sorted((set(ids_real) & set(ipr)) & ids_test)
        gr = np.stack([x_real[ri[i]] for i in common]); pr = np.stack([xp[pi[i]] for i in common])
        # feature-head codes
        fcg, fcp = feat_code(conv_feat(gr)), feat_code(conv_feat(pr))
        fe = eer_ci_sigma(fcg, fcp)
        # ortho-thermo 16-D baseline (fit on train16)
        e16g, e16p = emb16(gr), emb16(pr)
        og, op = _d.orthothermo(tr16, e16g.astype(np.float64), e16p.astype(np.float64))
        oe = eer_ci_sigma(og, op)
        rows[diff] = {"n": len(common),
                      "ortho": {"eer": oe[0], "ci": [oe[1], oe[2]], "sigma": oe[3], "h0": oe[4]},
                      "feat": {"eer": fe[0], "ci": [fe[1], fe[2]], "sigma": fe[3], "h0": fe[4]}}
        print(f"[{diff:6}] n={len(common)}  ortho EER {oe[0]*100:.2f}% [{oe[1]*100:.2f},{oe[2]*100:.2f}] sd {oe[3]:.1f}  "
              f"| feat EER {fe[0]*100:.2f}% [{fe[1]*100:.2f},{fe[2]*100:.2f}] sd {fe[3]:.2f} h0 {fe[4]}", flush=True)
    with open(f"{OUT}/qat-ladder-{args.tag}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[ladder] wrote {OUT}/qat-ladder-{args.tag}.json")


if __name__ == "__main__":
    main()
