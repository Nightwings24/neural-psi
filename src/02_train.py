"""
Week 1 - Step 2: Train the Blind-Touch Siamese CNN on SOCOFing (CPU).

- Splits the 6000 identities into train / test (held-out for EER).
- Builds genuine pairs from augmented views of the same Real print, and
  impostor pairs from different prints (mirrors the notebook's DataGenerator).
- Trains the L1-distance verification head with BCE.
- Saves:
    data/feature_model.pt   trained feature extractor weights (FC-16)
    data/ids_test.npy       identities held out for EER measurement

Defaults are tuned to finish on a 16-core CPU in a few minutes while producing
a *real* embedding. Scale --epochs / --img up on a GPU box for full fidelity.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import SiameseModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def augment(img_u8: np.ndarray) -> np.ndarray:
    """Light second-capture simulation: rotate/translate + gaussian noise."""
    im = Image.fromarray(img_u8)
    angle = np.random.uniform(-15, 15)
    tx, ty = np.random.uniform(-6, 6, size=2)
    im = im.rotate(angle, resample=Image.BILINEAR, fillcolor=255,
                   translate=(int(tx), int(ty)))
    a = np.asarray(im, dtype=np.float32)
    a += np.random.normal(0, 6, a.shape)
    return np.clip(a, 0, 255).astype(np.float32)


class PairDataset(torch.utils.data.Dataset):
    """Each item: (anchor, partner, label). 50% genuine (aug of same), 50% impostor."""

    def __init__(self, x_u8: np.ndarray):
        self.x = x_u8
        self.n = len(x_u8)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = augment(self.x[i])
        if np.random.rand() < 0.5:
            b, y = augment(self.x[i]), 1.0           # genuine
        else:
            j = np.random.randint(self.n - 1)
            j = j + 1 if j >= i else j               # any other print
            b, y = augment(self.x[j]), 0.0           # impostor
        a = torch.from_numpy(a / 255.0).unsqueeze(0)
        b = torch.from_numpy(b / 255.0).unsqueeze(0)
        return a, b, torch.tensor(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_real = np.load(os.path.join(DATA, "x_real.npy"))
    ids_real = np.load(os.path.join(DATA, "ids_real.npy"))
    idx = np.arange(len(x_real))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_test = int(round(len(idx) * args.test_size))
    te_idx, tr_idx = idx[:n_test], idx[n_test:]
    np.save(os.path.join(DATA, "ids_test.npy"), ids_real[te_idx])
    print(f"[train] identities: {len(tr_idx)} train / {len(te_idx)} test (held out for EER)")

    ds = PairDataset(x_real[tr_idx])
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=True)

    device = "cpu"
    model = SiameseModel(img=args.img).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M | device={device} "
          f"| threads={torch.get_num_threads()}")

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, correct, run = 0, 0, 0.0
        for a, b, y in dl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            logit = model(a, b)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            run += loss.item() * len(y)
            correct += ((logit > 0).float() == y).sum().item()
            tot += len(y)
        print(f"[train] epoch {ep}/{args.epochs}  loss={run/tot:.4f}  "
              f"pair_acc={correct/tot:.3f}")

    torch.save(model.feature_model.state_dict(),
               os.path.join(DATA, "feature_model.pt"))
    print(f"[train] saved feature_model -> {DATA}/feature_model.pt")


if __name__ == "__main__":
    main()
