"""
Week 1 — Step 1b: extract SOCOFing from an UNZIPPED directory (not the Kaggle zip).

01_extract.py reads resources/archive.zip; this box instead has the dataset already
unpacked at  <repo>/dataset/SOCOFing/{Real, Altered/Altered-Easy}.  This script mirrors
01_extract.py's identity parsing and Real<->Altered-Easy pairing exactly, but reads BMPs
from the directory and writes IMG-suffixed arrays so the 96px artifacts are NOT clobbered.

Outputs (for --img 224):
  data/x_real_224.npy    (N,224,224) uint8   data/ids_real_224.npy
  data/x_probe_224.npy   (M,224,224) uint8   data/ids_probe_224.npy

Run:  python3 01b_extract_dir.py --img 224
"""
import argparse
import os
import re

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
# repo-root dataset:  src/ -> ../dataset/SOCOFing (see README: download separately)
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, "..", "dataset", "SOCOFing"))

ID_RE = re.compile(r"(\d+)__[MF]_(Left|Right)_(\w+?)_finger", re.IGNORECASE)


def identity(basename: str):
    m = ID_RE.search(basename)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}".lower()


def load_gray(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("L").resize((size, size))
    return np.asarray(im, dtype=np.uint8)


def list_bmps(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.upper().endswith(".BMP"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--altered", default="Altered-Easy")
    ap.add_argument("--probe-per-id", type=int, default=1)
    ap.add_argument("--suffix", default=None, help="filename suffix; default _<img>")
    args = ap.parse_args()
    suffix = args.suffix if args.suffix is not None else f"_{args.img}"
    os.makedirs(OUT, exist_ok=True)

    real_dir = os.path.join(args.root, "Real")
    alt_dir = os.path.join(args.root, "Altered", args.altered)
    assert os.path.isdir(real_dir), f"missing {real_dir}"
    assert os.path.isdir(alt_dir), f"missing {alt_dir}"

    # ---- Real prints ----
    real = list_bmps(real_dir)
    print(f"[extract] Real BMPs: {len(real)}  (resizing to {args.img}x{args.img})")
    x_real, ids_real = [], []
    for i, n in enumerate(real):
        iid = identity(os.path.basename(n))
        if iid is None:
            continue
        x_real.append(load_gray(n, args.img)); ids_real.append(iid)
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(real)}")
    x_real = np.stack(x_real); ids_real = np.array(ids_real)
    real_id_set = set(ids_real.tolist())
    assert len(real_id_set) == len(ids_real), "Real identities must be unique"

    # ---- Altered-Easy probes: one per identity ----
    alt = list_bmps(alt_dir)
    print(f"[extract] {args.altered} BMPs: {len(alt)}")
    seen, x_probe, ids_probe = {}, [], []
    for n in alt:
        iid = identity(os.path.basename(n))
        if iid is None or iid not in real_id_set or seen.get(iid, 0) >= args.probe_per_id:
            continue
        seen[iid] = seen.get(iid, 0) + 1
        x_probe.append(load_gray(n, args.img)); ids_probe.append(iid)
    x_probe = np.stack(x_probe); ids_probe = np.array(ids_probe)

    np.save(os.path.join(OUT, f"x_real{suffix}.npy"), x_real)
    np.save(os.path.join(OUT, f"ids_real{suffix}.npy"), ids_real)
    np.save(os.path.join(OUT, f"x_probe{suffix}.npy"), x_probe)
    np.save(os.path.join(OUT, f"ids_probe{suffix}.npy"), ids_probe)
    print(f"[extract] saved x_real{suffix} {x_real.shape}, x_probe{suffix} {x_probe.shape}")
    print(f"[extract] identities with a probe: {len(ids_probe)}/{len(ids_real)} -> {OUT}")


if __name__ == "__main__":
    main()
