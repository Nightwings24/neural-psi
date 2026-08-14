"""
Step 13: render the CNN stage-by-stage figures used by the pipeline walkthrough.

The walkthrough document explains what each convolutional block does to a fingerprint.
It referenced seven figures that had never been generated, so it could not be built.
This script produces them from the trained model and one real held-out fingerprint:

    figures/00_fingerprint.png    the raw input
    figures/01_block1.png ...     feature maps after each of the 5 conv blocks
    figures/05_block5.png
    figures/06_feature_vector.png the resulting 16-D embedding

Each block figure shows a sample of that block's channels on a shared grid, so the
progression from ridge-level texture to abstract, spatially-coarse features is visible.
"""
import argparse
import os

import numpy as np
import torch

from model import FeatureModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.abspath(os.path.join(HERE, "..", "paper", "figures"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--ckpt", default="feature_model_224.pt")
    ap.add_argument("--identity", default=None,
                    help="which held-out identity to render (default: the first)")
    ap.add_argument("--channels", type=int, default=8, help="channels shown per block")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureModel(img=args.img).to(dev).eval()
    model.load_state_dict(torch.load(f"{DATA}/{args.ckpt}", map_location=dev))

    x = np.load(f"{DATA}/x_real_{args.img}.npy", mmap_mode="r")
    ids = np.load(f"{DATA}/ids_real_{args.img}.npy", allow_pickle=True)
    test = set(np.load(f"{DATA}/ids_test_{args.img}.npy", allow_pickle=True).tolist())
    pick = (np.flatnonzero(ids == args.identity)[0] if args.identity
            else next(k for k, i in enumerate(ids) if i in test))
    img = np.asarray(x[pick])
    print(f"[fig] rendering identity {ids[pick]} ({args.img}x{args.img})")

    # ---- 00: the raw input -------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(img, cmap="gray"); ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(f"{FIGS}/00_fingerprint.png", dpi=200)
    plt.close(fig)

    # ---- 01..05: feature maps after each conv block ------------------------
    # model.features is 5 x [Conv, BatchNorm, SiLU, MaxPool]; capture after each MaxPool.
    t = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(dev)
    acts, h = [], t
    with torch.no_grad():
        for k, layer in enumerate(model.features):
            h = layer(h)
            if isinstance(layer, torch.nn.MaxPool2d):
                acts.append(h[0].cpu().numpy())

    for b, a in enumerate(acts, start=1):
        c = min(args.channels, a.shape[0])
        # show the channels with the most activation energy: the near-dead ones are
        # uninformative and would just look like empty tiles.
        order = np.argsort(a.reshape(a.shape[0], -1).std(axis=1))[::-1][:c]
        cols = 4
        rows = int(np.ceil(c / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        for j, ax in enumerate(np.atleast_1d(axes).ravel()):
            ax.axis("off")
            if j < c:
                ax.imshow(a[order[j]], cmap="viridis")
        fig.suptitle(f"Block {b}: {a.shape[0]} channels @ {a.shape[1]}$\\times${a.shape[2]}",
                     fontsize=9)
        fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.94))
        fig.savefig(f"{FIGS}/{b:02d}_block{b}.png", dpi=170)
        plt.close(fig)
        print(f"[fig]   block {b}: {a.shape[0]} ch @ {a.shape[1]}x{a.shape[2]}")

    # ---- 06: the 16-D embedding -------------------------------------------
    with torch.no_grad():
        e = model(t)[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.2, 2.0))
    ax.bar(np.arange(len(e)), e, color=["#c0392b" if v < 0 else "#2874a6" for v in e])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(e))); ax.set_xlabel("embedding dimension")
    ax.set_ylabel("value"); ax.set_title("The 16-D output embedding", fontsize=10)
    fig.tight_layout(pad=0.3); fig.savefig(f"{FIGS}/06_feature_vector.png", dpi=200)
    plt.close(fig)
    print(f"[fig] wrote 7 figures to {FIGS}")


if __name__ == "__main__":
    main()
