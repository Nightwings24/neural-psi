"""
GPU-ready training at the ORIGINAL Blind-Touch resolution (224x224), full schedule.

Drop-in upgrade of 02_train.py for when a GPU is available. Differences:
  * 224x224 input  -> flatten dim 512*7*7 = 25088 (matches the original Keras model,
    and the dim the friend's quantizer/Super-Bit expects).
  * auto CUDA + mixed precision (AMP) + pinned memory + more workers.
  * 150-epoch default (paper schedule); cosine LR.
  * SAVES THE HEAD too -> head_w1_224.npy, so the Super-Bit quantizer can use
    metric whitening (diag(sqrt(|w1|))), which was disabled on the CPU model.

Prereq:  python3 01_extract.py --img 224     (makes 224x224 x_real/x_probe)
Run:     python3 train_gpu_224.py             (uses GPU automatically if present)

Outputs (kept SEPARATE from the 96x96 artifacts so nothing is overwritten):
  data/feature_model_224.pt   trained feature extractor (img=224)
  data/head_w1_224.npy        final-layer weights w1 (16,) for Super-Bit whitening
  data/ids_test_224.npy       held-out identities

Downstream: re-run eval/export with img=224 and these filenames, e.g.
  FeatureModel(img=224); load feature_model_224.pt
  06_superbit_export.py: QuantizerConfig(whiten=True), w1 = np.load(head_w1_224.npy)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import SiameseModel

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"


def augment(img_u8, size):
    """Second-capture simulation: rotate/translate + gaussian noise (resolution-agnostic)."""
    im = Image.fromarray(img_u8)
    angle = np.random.uniform(-15, 15)
    tx, ty = np.random.uniform(-0.06, 0.06, size=2) * size
    im = im.rotate(angle, resample=Image.BILINEAR, fillcolor=255,
                   translate=(int(tx), int(ty)))
    a = np.asarray(im, dtype=np.float32) + np.random.normal(0, 6, (size, size))
    return np.clip(a, 0, 255).astype(np.float32)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, x_u8, size):
        self.x, self.n, self.size = x_u8, len(x_u8), size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = augment(self.x[i], self.size)
        if np.random.rand() < 0.5:
            b, y = augment(self.x[i], self.size), 1.0
        else:
            j = np.random.randint(self.n - 1)
            j = j + 1 if j >= i else j
            b, y = augment(self.x[j], self.size), 0.0
        a = torch.from_numpy(a / 255.0).unsqueeze(0)
        b = torch.from_numpy(b / 255.0).unsqueeze(0)
        return a, b, torch.tensor(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--emb-dim", type=int, default=16,
                    help="embedding dimension (FC output); outputs are tagged _d<emb-dim>")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"[train] device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " - NO GPU; this will be slow at 224²"))

    xr = os.path.join(DATA, f"x_real_{args.img}.npy")
    if not os.path.exists(xr):
        xr = os.path.join(DATA, "x_real.npy")
    x_real = np.load(xr)
    if x_real.shape[1] != args.img:
        raise SystemExit(f"{os.path.basename(xr)} is {x_real.shape[1]}px but --img={args.img}. "
                         f"Run: python3 01b_extract_dir.py --img {args.img}")
    ids_real = np.load(os.path.join(DATA, f"ids_real_{args.img}.npy")
                       if os.path.exists(os.path.join(DATA, f"ids_real_{args.img}.npy"))
                       else os.path.join(DATA, "ids_real.npy"))

    idx = np.arange(len(x_real))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_test = int(round(len(idx) * args.test_size))
    te_idx, tr_idx = idx[:n_test], idx[n_test:]
    tag = f"_d{args.emb_dim}"
    np.save(os.path.join(DATA, f"ids_test_224{tag}.npy"), ids_real[te_idx])
    print(f"[train] {len(tr_idx)} train / {len(te_idx)} test identities | img={args.img} | emb_dim={args.emb_dim}")

    ds = PairDataset(x_real[tr_idx], args.img)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=True,
                                     pin_memory=use_amp, persistent_workers=args.workers > 0)

    model = SiameseModel(img=args.img, emb_dim=args.emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"[train] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"flatten dim {model.feature_model.flat_dim} | amp={use_amp}")

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, correct, run = 0, 0, 0.0
        for a, b, y in dl:
            a, b, y = a.to(device, non_blocking=True), b.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logit = model(a, b)
                loss = loss_fn(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(y)
            correct += ((logit > 0).float() == y).sum().item()
            tot += len(y)
        sched.step()
        print(f"[train] epoch {ep}/{args.epochs}  loss={run/tot:.4f}  "
              f"pair_acc={correct/tot:.3f}  lr={sched.get_last_lr()[0]:.2e}")

    torch.save(model.feature_model.state_dict(), os.path.join(DATA, f"feature_model_224{tag}.pt"))
    w1 = model.head.weight.detach().cpu().numpy().reshape(-1)   # (emb_dim,) for whitening
    np.save(os.path.join(DATA, f"head_w1_224{tag}.npy"), w1)
    print(f"[train] saved feature_model_224{tag}.pt + head_w1_224{tag}.npy -> {DATA}")


if __name__ == "__main__":
    main()
