"""
Step 35: extract PolyU contactless-2D over BOTH sessions as 496 distinct subjects
(336 first-session + 160 second-session), matching the Blind-Touch PolyU protocol.
Second-session fingers get an "_s2" id suffix so each (finger, session) is a separate subject.

Source: /mnt/SharedData/Cross_Fingerprint_Images_Database
  processed_contactless_2d_fingerprint_images/{first,second}_session/p<fid>/p<s>.bmp
Writes: data/polyu_cless2_224.npy + data/polyu_cless2_ids.npy  (2976 images, 496 ids, 6 each).

Run: python3 35_extract_polyu_2sess.py --img 224
"""
import argparse, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "data")
CF = "/mnt/SharedData/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images"


def load_gray(path, size):
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--img", type=int, default=224); args = ap.parse_args()
    S = args.img; X, ID = [], []
    for sess, suffix in [("first_session", ""), ("second_session", "_s2")]:
        base = f"{CF}/{sess}"
        for pdir in sorted(os.listdir(base)):
            d = os.path.join(base, pdir)
            if not os.path.isdir(d): continue
            sid = pdir + suffix
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(".bmp"): continue
                try:
                    X.append(load_gray(os.path.join(d, fn), S)); ID.append(sid)
                except Exception as e:
                    print(f"  skip {sess}/{pdir}/{fn}: {e}")
    X = np.stack(X); ID = np.array(ID)
    np.save(f"{OUT}/polyu_cless2_{S}.npy", X); np.save(f"{OUT}/polyu_cless2_ids.npy", ID)
    print(f"[polyu2] contactless: {X.shape} | {len(set(ID))} subjects (want 496) | "
          f"first {sum(1 for i in set(ID) if not i.endswith('_s2'))} second {sum(1 for i in set(ID) if i.endswith('_s2'))}")


if __name__ == "__main__":
    main()
