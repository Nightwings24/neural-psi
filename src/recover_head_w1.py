"""
Recover the FC head weights w1 that the Super-Bit metric-whitening needs.

results.md disables whitening because "CNN head weights not saved". But the quantizer's
diag(sqrt(|w1|)) whitening only needs the 16 per-dimension weights of the Blind-Touch
matching head, whose score is sigmoid( sum_k w1_k * (e_q,k - e_db,k)^2 + b ). We can
recover those 16 numbers WITHOUT the original head: fit a logistic regression on the
SQUARED differences of train-identity embedding pairs (genuine=1 / impostor=0). The
learned coefficients ARE w1 (negative: bigger squared diff -> lower match prob).

Writes data/head_w1.npy (16,) for `07_superbit_eer.py --whiten --w1 data/head_w1.npy`.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from model import FeatureModel

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
rng = np.random.default_rng(0)


def embed_all(model, x, batch=128):
    model.eval(); out = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1).to(dev)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--ckpt", default="feature_model.pt")
    ap.add_argument("--out", default="", help="output .npy (default head_w1[ _<img>].npy)")
    args = ap.parse_args()

    def dload(base):
        p = f"{DATA}/{base}_{args.img}.npy"
        return np.load(p, allow_pickle=True) if os.path.exists(p) else np.load(f"{DATA}/{base}.npy", allow_pickle=True)

    x_real, ids_real = dload("x_real"), dload("ids_real")
    x_probe, ids_probe = dload("x_probe"), dload("ids_probe")
    ids_test = set(dload("ids_test").tolist())

    model = FeatureModel(img=args.img)
    model.load_state_dict(torch.load(f"{DATA}/{args.ckpt}"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # TRAIN identities only (no leakage into the held-out EER set)
    real_by = {i: x for i, x in zip(ids_real, x_real) if i not in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i not in ids_test}
    common = sorted(set(real_by) & set(prb_by))
    er = embed_all(model, np.stack([real_by[i] for i in common]))
    ep = embed_all(model, np.stack([prb_by[i] for i in common]))
    n = len(common)

    # genuine pairs (real_i, probe_i); impostor pairs (real_i, probe_j != i)
    j = (np.arange(n) + rng.integers(1, n, size=n)) % n
    gen = (er - ep) ** 2                      # (n,16) squared diffs, genuine
    imp = (er - ep[j]) ** 2                   # (n,16) squared diffs, impostor
    X = np.vstack([gen, imp]).astype(np.float32)
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)

    # logistic regression: sigmoid(X @ w1 + b) ~ match.  w1 are the head weights.
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    lin = nn.Linear(16, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    lossf = nn.BCEWithLogitsLoss()
    for ep_i in range(300):
        opt.zero_grad()
        loss = lossf(lin(Xt).squeeze(1), yt)
        loss.backward(); opt.step()
    w1 = lin.weight.detach().numpy().reshape(-1)
    acc = float(((lin(Xt).squeeze(1) > 0).float().numpy() == y).mean())

    out = args.out or (f"{DATA}/head_w1.npy" if args.img == 96 else f"{DATA}/head_w1_{args.img}.npy")
    out = out if os.path.isabs(out) or "/" in out else f"{DATA}/{out}"
    np.save(out, w1)
    print(f"[recover] head logistic-regression train acc = {acc:.3f}")
    print(f"[recover] w1 (head weights) = {np.round(w1, 3)}")
    print(f"[recover] sign: {int((w1 < 0).sum())}/16 negative (expect mostly negative)")
    print(f"[recover] |w1| range {np.abs(w1).min():.3g}..{np.abs(w1).max():.3g}")
    print(f"[recover] saved -> {out}")


if __name__ == "__main__":
    main()
