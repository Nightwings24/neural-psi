"""
Step 33: train a SAME-SENSOR PolyU extractor (per-dataset tuning, as Blind-Touch did), so the
generality test uses a proper PolyU-tuned CNN instead of the cross-modal one.

Genuine pairs = two distinct samples of the SAME finger, SAME modality. Finger-disjoint split
REPLICATED EXACTLY from 30_polyu_featurehead.py (same seed / test-frac / finger sort) so the CNN
is trained only on that script's TRAIN fingers and never sees its held-out test fingers.

Warm-starts the FeatureModel from feature_model_224.pt. Saves feature_model_polyu_ss_<mod>.pt.
Then run:  python3 30_polyu_featurehead.py --modality <mod> --ckpt feature_model_polyu_ss_<mod>.pt

Run: python3 33_polyu_samesensor_train.py --modality cless --epochs 60
"""
import argparse, os
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from model import SiameseModel

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); IMG = 224


def augment(u8):
    im = Image.fromarray(u8); ang = np.random.uniform(-12, 12); tx, ty = np.random.uniform(-0.05, 0.05, 2)*IMG
    im = im.rotate(ang, resample=Image.BILINEAR, fillcolor=255, translate=(int(tx), int(ty)))
    return np.clip(np.asarray(im, np.float32)+np.random.normal(0, 5, (IMG, IMG)), 0, 255).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="cless")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)          # MUST match 30_polyu_featurehead default
    ap.add_argument("--test-frac", type=float, default=0.35)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x = np.load(f"{DATA}/polyu_{args.modality}_224.npy")
    ids = np.load(f"{DATA}/polyu_{args.modality}_ids.npy", allow_pickle=True).astype(str)
    by = {}
    for k, f in enumerate(ids): by.setdefault(f, []).append(k)
    # split IDENTICAL to 30_polyu_featurehead.py
    fids = sorted(by, key=lambda s: int(s) if s.isdigit() else s)
    rng = np.random.default_rng(args.seed); rng.shuffle(fids)
    nte = int(len(fids)*args.test_frac); train_f = fids[nte:]
    train_f = [f for f in train_f if len(by[f]) >= 2]
    print(f"[polyu-ss] {args.modality}: {len(train_f)} train fingers (held out {nte}) | dev {dev}", flush=True)

    model = SiameseModel(img=IMG).to(dev)
    model.feature_model.load_state_dict(torch.load(f"{DATA}/feature_model_224.pt", map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda")); bce = nn.BCEWithLogitsLoss()
    steps = max(1, (len(train_f)*6)//args.batch)
    for ep in range(1, args.epochs+1):
        model.train(); run = 0.0
        for _ in range(steps):
            A = np.empty((args.batch, IMG, IMG), np.float32); B = np.empty_like(A); y = np.empty(args.batch, np.float32)
            for i in range(args.batch):
                f = train_f[np.random.randint(len(train_f))]; s = by[f]
                a1 = s[np.random.randint(len(s))]; A[i] = augment(x[a1])
                if np.random.rand() < 0.5:                       # genuine: distinct sample same finger
                    a2 = a1
                    while a2 == a1: a2 = s[np.random.randint(len(s))]
                    B[i] = augment(x[a2]); y[i] = 1.0
                else:                                            # impostor: other finger
                    g = f
                    while g == f: g = train_f[np.random.randint(len(train_f))]
                    B[i] = augment(x[by[g][np.random.randint(len(by[g]))]]); y[i] = 0.0
            a = torch.from_numpy(A/255.).unsqueeze(1).to(dev); b = torch.from_numpy(B/255.).unsqueeze(1).to(dev)
            yt = torch.from_numpy(y).to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                loss = bce(model(a, b), yt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  [polyu-ss] epoch {ep}/{args.epochs} loss {run/steps:.4f}", flush=True)
    out = f"{DATA}/feature_model_polyu_ss_{args.modality}.pt"
    torch.save(model.feature_model.state_dict(), out)
    print(f"[polyu-ss] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
