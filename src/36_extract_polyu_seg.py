"""
Step 36: PolyU contactless extraction WITH fingerprint preprocessing (CLAHE contrast enhancement
+ variance-based ROI segmentation + aspect-preserving resize), both sessions, 496 subjects.
Tests whether the ~4x float-EER gap to Blind-Touch (10.87% vs 2.5%) is closed by the standard
PolyU preprocessing we previously skipped (raw square-resize).

Writes: data/polyu_cless2seg_224.npy + data/polyu_cless2seg_ids.npy (496 ids, "_s2" for session-2).
Run: python3 36_extract_polyu_seg.py --img 224
"""
import argparse, os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "data")
CF = "/mnt/SharedData/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images"


def preprocess(gray, size):
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)   # contrast enhancement
    g = cl.astype(np.float32); k = 15                                       # local std -> foreground
    mean = cv2.boxFilter(g, -1, (k, k)); sq = cv2.boxFilter(g*g, -1, (k, k))
    std = np.sqrt(np.clip(sq - mean*mean, 0, None))
    m = (std > np.percentile(std, 55)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(m > 0)
    if len(xs) > 50:                                                        # crop ROI bbox + margin
        pad = 8; y0 = max(0, ys.min()-pad); x0 = max(0, xs.min()-pad)
        y1 = min(cl.shape[0], ys.max()+pad); x1 = min(cl.shape[1], xs.max()+pad)
        cl = cl[y0:y1, x0:x1]
    h, w = cl.shape; s = max(h, w)                                          # pad square (aspect-safe)
    canvas = np.full((s, s), int(cl[:3].mean()), np.uint8)
    canvas[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = cl
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--img", type=int, default=224); args = ap.parse_args()
    S = args.img; X, ID = [], []
    for sess, suf in [("first_session", ""), ("second_session", "_s2")]:
        base = f"{CF}/{sess}"
        for pdir in sorted(os.listdir(base)):
            d = os.path.join(base, pdir)
            if not os.path.isdir(d): continue
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(".bmp"): continue
                g = cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE)
                if g is None: continue
                X.append(preprocess(g, S)); ID.append(pdir + suf)
    X = np.stack(X).astype(np.uint8); ID = np.array(ID)
    np.save(f"{OUT}/polyu_cless2seg_{S}.npy", X); np.save(f"{OUT}/polyu_cless2seg_ids.npy", ID)
    print(f"[polyu-seg] {X.shape} | {len(set(ID))} subjects | first {sum(1 for i in set(ID) if not i.endswith('_s2'))} second {sum(1 for i in set(ID) if i.endswith('_s2'))}")


if __name__ == "__main__":
    main()
