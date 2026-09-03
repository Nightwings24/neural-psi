"""
Step 18: extract PolyU Contactless-2D-to-Contact-2D cross-sensor arrays.

Source: /mnt/SharedData/Cross_Fingerprint_Images_Database (Lin & Kumar, IEEE TIP 2018).
  contact-based_fingerprints/first_session/<fid>_<s>.jpg           (URU reader, 328x356)
  processed_contactless_2d_fingerprint_images/first_session/p<fid>/p<s>.bmp  (downsampled grayscale)
336 fingers, 6 samples each (first session).

Writes (grayscale, 224x224 uint8, matching the SOCOFing pipeline):
  data/polyu_contact_224.npy   + data/polyu_contact_ids.npy
  data/polyu_cless_224.npy     + data/polyu_cless_ids.npy
ids are the finger id (str) so the same finger's contact and contactless rows share an id.
"""
import argparse, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "data")
CF = "/mnt/SharedData/Cross_Fingerprint_Images_Database"


def load_gray(path, size):
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--session", default="first_session")
    args = ap.parse_args()
    S = args.img

    contact_dir = f"{CF}/contact-based_fingerprints/{args.session}"
    cless_dir = f"{CF}/processed_contactless_2d_fingerprint_images/{args.session}"

    # contact: <fid>_<s>.jpg
    cx, cid = [], []
    for fn in sorted(os.listdir(contact_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".bmp")):
            continue
        fid = fn.split("_")[0]
        try:
            cx.append(load_gray(os.path.join(contact_dir, fn), S)); cid.append(fid)
        except Exception as e:
            print(f"  skip contact {fn}: {e}")
    cx = np.stack(cx); cid = np.array(cid)
    np.save(f"{OUT}/polyu_contact_{S}.npy", cx); np.save(f"{OUT}/polyu_contact_ids.npy", cid)
    print(f"[polyu] contact: {cx.shape} from {len(set(cid))} fingers -> polyu_contact_{S}.npy")

    # contactless: p<fid>/p<s>.bmp
    lx, lid = [], []
    for folder in sorted(os.listdir(cless_dir)):
        d = os.path.join(cless_dir, folder)
        if not os.path.isdir(d):
            continue
        fid = folder.lstrip("pP")
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith((".bmp", ".jpg", ".jpeg")):
                continue
            try:
                lx.append(load_gray(os.path.join(d, fn), S)); lid.append(fid)
            except Exception as e:
                print(f"  skip cless {folder}/{fn}: {e}")
    lx = np.stack(lx); lid = np.array(lid)
    np.save(f"{OUT}/polyu_cless_{S}.npy", lx); np.save(f"{OUT}/polyu_cless_ids.npy", lid)
    print(f"[polyu] contactless: {lx.shape} from {len(set(lid))} fingers -> polyu_cless_{S}.npy")
    print(f"[polyu] shared finger ids: {len(set(cid) & set(lid))}")


if __name__ == "__main__":
    main()
