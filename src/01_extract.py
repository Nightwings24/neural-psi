"""
Week 1 - Step 1: Extract SOCOFing (SOKOTO) dataset from the Kaggle zip.

Reads resources/archive.zip directly (no full unzip), keeps only the canonical
top-level `SOCOFing/` root (the zip ships a duplicate `socofing/SOCOFing/`),
converts each fingerprint to IMG x IMG grayscale uint8, and saves:

  data/x_real.npy      (N, IMG, IMG) uint8   -- the 6000 Real prints
  data/ids_real.npy    (N,) <U..    string   -- identity = subject_hand_finger
  data/x_probe.npy     (M, IMG, IMG) uint8   -- one Altered-Easy probe per Real
  data/ids_probe.npy   (M,) string           -- matching identity for each probe

Identity string is parsed from filenames like:
  100__M_Left_index_finger.BMP            -> 100_Left_index
  100__M_Left_index_finger_CR.BMP (Alt.)  -> 100_Left_index
"""
import argparse
import io
import os
import re
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "..", "resources", "archive.zip")
OUT = os.path.join(HERE, "data")

# 100__M_Left_index_finger[...].BMP  -> capture subject, hand, finger
ID_RE = re.compile(r"(\d+)__[MF]_(Left|Right)_(\w+?)_finger", re.IGNORECASE)


def identity(basename: str) -> str | None:
    m = ID_RE.search(basename)
    if not m:
        return None
    subject, hand, finger = m.group(1), m.group(2), m.group(3)
    return f"{subject}_{hand}_{finger}".lower()


def load_gray(zf: zipfile.ZipFile, name: str, size: int) -> np.ndarray:
    im = Image.open(io.BytesIO(zf.read(name))).convert("L").resize((size, size))
    return np.asarray(im, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=96, help="square image size")
    ap.add_argument("--root", default="SOCOFing", help="canonical zip root to use")
    ap.add_argument("--altered", default="Altered-Easy",
                    help="Altered subset to draw genuine probes from")
    ap.add_argument("--probe-per-id", type=int, default=1,
                    help="how many altered probes to keep per identity")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)
    names = zf.namelist()

    # ---- Real prints (canonical root only) ----
    real = sorted(n for n in names
                  if n.startswith(args.root + "/Real/") and n.upper().endswith(".BMP"))
    print(f"[extract] Real images under {args.root}/Real/: {len(real)}")

    x_real, ids_real = [], []
    for i, n in enumerate(real):
        iid = identity(os.path.basename(n))
        if iid is None:
            continue
        x_real.append(load_gray(zf, n, args.img))
        ids_real.append(iid)
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(real)}")
    x_real = np.stack(x_real)
    ids_real = np.array(ids_real)
    real_id_set = set(ids_real.tolist())
    assert len(real_id_set) == len(ids_real), "identities should be unique per Real print"

    # ---- Altered-Easy probes: one per identity (genuine probe for EER) ----
    alt = sorted(n for n in names
                 if n.startswith(f"{args.root}/Altered/{args.altered}/")
                 and n.upper().endswith(".BMP"))
    print(f"[extract] {args.altered} images: {len(alt)}")

    seen: dict[str, int] = {}
    x_probe, ids_probe = [], []
    for n in alt:
        iid = identity(os.path.basename(n))
        if iid is None or iid not in real_id_set:
            continue
        if seen.get(iid, 0) >= args.probe_per_id:
            continue
        seen[iid] = seen.get(iid, 0) + 1
        x_probe.append(load_gray(zf, n, args.img))
        ids_probe.append(iid)
    x_probe = np.stack(x_probe)
    ids_probe = np.array(ids_probe)

    np.save(os.path.join(OUT, "x_real.npy"), x_real)
    np.save(os.path.join(OUT, "ids_real.npy"), ids_real)
    np.save(os.path.join(OUT, "x_probe.npy"), x_probe)
    np.save(os.path.join(OUT, "ids_probe.npy"), ids_probe)

    print(f"[extract] saved x_real   {x_real.shape} {x_real.dtype}")
    print(f"[extract] saved x_probe  {x_probe.shape} (one {args.altered} per identity)")
    print(f"[extract] identities covered by a probe: {len(ids_probe)}/{len(ids_real)}")
    print(f"[extract] -> {OUT}")


if __name__ == "__main__":
    main()
