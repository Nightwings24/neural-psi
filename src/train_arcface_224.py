"""
Tier-1 trainer: Blind-Touch feature extractor + ANGULAR-MARGIN head (ArcFace / CosFace).

Why: the NeuralPSI bridge binarises with Super-Bit (sign of random projections), so Hamming
distance tracks the ANGLE between embeddings. Training the 16-D embedding with an angular-margin
classification loss makes that angle identity-discriminative, which is the lever the improvement
search flagged as 'both lifts the float ceiling and closes the binary gap'. This is what
train_gpu_224.py (plain pairwise BCE) lacks.

Pipeline: image -> FeatureModel -> e in R^16 -> L2-normalise -> ArcFace/CosFace logits over
finger-identities -> cross-entropy. After training, the SAME FeatureModel produces the 16-D
embedding the quantizer consumes; the head metric weights w1 for whitening are recovered
post-hoc with recover_head_w1.py (works for any embedding).

GPU: auto-uses CUDA + AMP if available; falls back to CPU (slow at 224 - use --smoke to verify).
Data: loads x_real_<img>.npy / x_probe_<img>.npy if present, else x_real.npy (96px). Each finger
is a class; its Real + Altered-Easy prints (+ augmentation) are the samples.

Examples:
  python3 01b_extract_dir.py --img 224                 # make 224px arrays first
  python3 train_arcface_224.py --img 224 --epochs 150  # the real GPU run
  python3 train_arcface_224.py --img 96 --epochs 2 --max-ids 300 --smoke   # CPU smoke test
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model import FeatureModel

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"


def load_arr(img, base):
    p = os.path.join(DATA, f"{base}_{img}.npy")
    if not os.path.exists(p):
        p = os.path.join(DATA, f"{base}.npy")        # fall back to 96px default
    return np.load(p, allow_pickle=True)


def augment(img_u8, size):
    """Second-capture sim: small rotate/translate + gaussian noise (resolution-agnostic)."""
    im = Image.fromarray(img_u8)
    angle = np.random.uniform(-15, 15)
    tx, ty = (np.random.uniform(-0.06, 0.06, size=2) * size).astype(int)
    im = im.rotate(angle, resample=Image.BILINEAR, fillcolor=255, translate=(int(tx), int(ty)))
    a = np.asarray(im, dtype=np.float32) + np.random.normal(0, 6, (size, size))
    return np.clip(a, 0, 255).astype(np.float32)


class FingerDataset(torch.utils.data.Dataset):
    """One sample per (image, finger-label); Real + Altered prints both belong to the finger."""
    def __init__(self, imgs, labels, size, train=True):
        self.imgs, self.labels, self.size, self.train = imgs, labels, size, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = augment(self.imgs[i], self.size) if self.train else self.imgs[i].astype(np.float32)
        x = torch.from_numpy(x / 255.0).unsqueeze(0)
        return x, int(self.labels[i])


class ArcMarginHead(nn.Module):
    """ArcFace (additive angular margin) / CosFace (additive cosine margin) classifier."""
    def __init__(self, in_dim, n_classes, s=16.0, m=0.30, mode="arcface"):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m, self.mode = s, m, mode

    def cosine(self, e):
        """Plain cosine similarity to each class weight (no margin) - for the accuracy metric."""
        return F.linear(F.normalize(e), F.normalize(self.W))

    def forward(self, e, labels):
        cos = self.cosine(e).clamp(-1 + 1e-7, 1 - 1e-7)
        onehot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        if self.mode == "cosface":
            logits = self.s * (cos - self.m * onehot)
        else:  # arcface: cos(theta + m) on the true class
            theta = torch.acos(cos)
            logits = self.s * torch.cos(theta + self.m * onehot)
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--mode", choices=["arcface", "cosface"], default="arcface")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-ids", type=int, default=0, help="cap #fingers (0=all); for smoke tests")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the code path")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2); args.workers = 0
        args.max_ids = args.max_ids or 300

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"[train] device={device}{'' if use_amp else ' (NO GPU - slow at 224; use --smoke)'} | "
          f"mode={args.mode} margin={args.margin} scale={args.scale} img={args.img}")

    x_real = load_arr(args.img, "x_real"); ids_real = load_arr(args.img, "ids_real")
    x_probe = load_arr(args.img, "x_probe"); ids_probe = load_arr(args.img, "ids_probe")
    assert x_real.shape[1] == args.img, f"x_real is {x_real.shape[1]}px but --img={args.img}; run 01b_extract_dir.py --img {args.img}"

    # finger identities (each a class); split by identity so test fingers are unseen
    fingers = sorted(set(ids_real.tolist()))
    rng = np.random.default_rng(args.seed); rng.shuffle(fingers)
    if args.max_ids:
        fingers = fingers[:args.max_ids]
    n_test = int(round(len(fingers) * args.test_frac))
    test_ids = set(fingers[:n_test]); train_ids = [f for f in fingers if f not in test_ids]
    cls = {f: k for k, f in enumerate(train_ids)}
    np.save(os.path.join(DATA, f"ids_test_{args.img}.npy"), np.array(sorted(test_ids)))
    print(f"[train] fingers: {len(train_ids)} train classes / {len(test_ids)} held-out test")

    # build the per-image training set (Real + Altered of each TRAIN finger)
    imgs, labels = [], []
    for arr, ids in [(x_real, ids_real), (x_probe, ids_probe)]:
        for x, iid in zip(arr, ids):
            if iid in cls:
                imgs.append(x); labels.append(cls[iid])
    imgs = np.stack(imgs); labels = np.array(labels)
    print(f"[train] training images: {len(imgs)} over {len(train_ids)} classes")

    ds = FingerDataset(imgs, labels, args.img, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=True,
                                     pin_memory=use_amp, persistent_workers=args.workers > 0)

    feat = FeatureModel(img=args.img).to(device)
    head = ArcMarginHead(16, len(train_ids), s=args.scale, m=args.margin, mode=args.mode).to(device)
    opt = torch.optim.Adam(list(feat.parameters()) + list(head.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss()
    print(f"[train] params {sum(p.numel() for p in feat.parameters())/1e6:.2f}M | flatten {feat.flat_dim}")

    for ep in range(1, args.epochs + 1):
        feat.train(); head.train(); tot = correct = 0; run = 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                e = feat(x)
                logits = head(e, y)
                loss = ce(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            # accuracy from PLAIN cosine (no margin) - argmax of the margin-penalised logits
            # under-reports because ArcFace lowers the true-class logit by design.
            run += loss.item() * len(y); correct += (head.cosine(e.float()).argmax(1) == y).sum().item(); tot += len(y)
        sched.step()
        print(f"[train] epoch {ep}/{args.epochs}  loss={run/tot:.4f}  cls_acc={correct/tot:.3f}  "
              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

    out = os.path.join(DATA, f"feature_model_{args.img}.pt" if args.img != 96 else "feature_model_arcface.pt")
    torch.save(feat.state_dict(), out)
    print(f"[train] saved feature extractor -> {out}")
    print(f"[train] next: recover_head_w1.py (for whitening), then 07_superbit_eer.py / 06_superbit_export.py "
          f"pointed at this checkpoint to measure the Super-Bit EER.")


if __name__ == "__main__":
    main()
