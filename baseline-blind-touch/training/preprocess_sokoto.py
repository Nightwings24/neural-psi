#!/usr/bin/env python3
"""
Preprocess the SOCOFing 'Real' fingerprint images into the .npy array the
training notebook expects.

This step is NOT in the upstream repo or the implementation guide, but the
training notebook does `np.load(SOKOTO_DATA_PATH) / 255.` on a preprocessed
array - Kaggle only gives you raw .BMP files, so something has to build it.

Run this INSIDE the client container (it has opencv + numpy), after the
SOCOFing dataset is mounted at /workspace/dataset:

    docker exec -it blindtouch-client python3 /workspace/training/preprocess_sokoto.py

Input : /workspace/dataset/SOCOFing/Real/*.BMP   (6000 images, 96x103, grayscale)
Output: /workspace/shared_data/data/sokoto_real_224.npy  (uint8, shape (6000,224,224,1))
        The training notebook divides by 255. itself.
"""
import os
import sys
import glob
import numpy as np
import cv2

IN_DIR  = '/workspace/dataset/SOCOFing/Real'
OUT     = '/workspace/shared_data/data/sokoto_real_224.npy'
SIZE    = 224

def main():
    if not os.path.isdir(IN_DIR):
        sys.exit(f"ERROR: {IN_DIR} not found. Is the SOCOFing dataset unzipped under ./dataset "
                 f"and mounted? Expected ./dataset/SOCOFing/Real/ on the host.")

    files = sorted(glob.glob(os.path.join(IN_DIR, '*.BMP')) +
                   glob.glob(os.path.join(IN_DIR, '*.bmp')))
    if not files:
        sys.exit(f"ERROR: no .BMP files in {IN_DIR}")
    print(f"Found {len(files)} images")

    imgs = np.empty((len(files), SIZE, SIZE, 1), dtype=np.uint8)
    for i, f in enumerate(files):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            sys.exit(f"ERROR: failed to read {f}")
        g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        imgs[i, :, :, 0] = g
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, imgs)
    print(f"Saved {imgs.shape} {imgs.dtype} -> {OUT}")
    print("NOTE: the training notebook assigns one identity per image (y_real[i]=i), "
          "so it expects exactly 6000 images for the default config.")

if __name__ == '__main__':
    main()
