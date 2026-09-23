"""
Week 1 - Step 5: Export REAL fingerprint codes for the real flash-psi protocol.

Builds input files for flash-psi/src/bin/fingerprint.rs from real held-out
SOCOFing identities, for BOTH bridges (ITQ and naive sign), and for BOTH a
genuine and an impostor query:

  - Bob's DB  = m enrolled identities (their Real print -> 16-bit code).
  - genuine   = an enrolled identity's Altered probe (should match its row).
  - impostor  = a probe whose identity is NOT enrolled (should match nothing).

flash-psi's masked OPRF requires exactly 128-bit codes, but Blind-Touch emits
16 bits. We widen 16 -> 128 by repeating each bit 8x. This preserves Hamming
geometry (distances scale x8), so the paper's params (weight=14, T=64) apply.

Writes {itq,sign}_{genuine,impostor}.txt under data/psi/ in the format:
    line0: "<m> 128"
    line1: query bits (128 chars)
    line2..: m db rows (128 chars each)
Also prints the ground-truth genuine row index.
"""
import os
import numpy as np
import torch

from model import FeatureModel

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
OUT = DATA + "/psi"
REPEAT = 8                       # 16 bits * 8 = 128
M = 50                           # DB size for the demo
rng = np.random.default_rng(2024)


def embed_all(model, x, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1)
            out.append(model(xb).numpy())
    return np.concatenate(out)


def widen(bits16):
    """16-bit code -> 128-bit by repeating each bit REPEAT times."""
    return np.repeat(bits16, REPEAT)


def to_line(bits):
    return "".join("1" if b else "0" for b in bits)


def write_case(path, query128, db128):
    with open(path, "w") as f:
        f.write(f"{len(db128)} 128\n")
        f.write(to_line(query128) + "\n")
        for row in db128:
            f.write(to_line(row) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    x_real = np.load(f"{DATA}/x_real.npy")
    ids_real = np.load(f"{DATA}/ids_real.npy")
    x_probe = np.load(f"{DATA}/x_probe.npy")
    ids_probe = np.load(f"{DATA}/ids_probe.npy")
    ids_test = set(np.load(f"{DATA}/ids_test.npy").tolist())
    itq = np.load(f"{DATA}/itq_params.npz")
    mean, R = itq["mean"], itq["R"]

    model = FeatureModel(img=96)
    model.load_state_dict(torch.load(f"{DATA}/feature_model.pt"))

    real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
    common = sorted(set(real_by) & set(prb_by))
    rng.shuffle(common)

    db_ids = common[:M]                  # enrolled
    genuine_id = db_ids[7]               # an enrolled identity (its probe = genuine)
    impostor_id = common[M + 3]          # NOT enrolled -> impostor probe

    def codes(img):
        emb = embed_all(model, img[None])[0]
        sign_bits = (emb > 0).astype(np.int8)
        itq_bits = (((emb - mean) @ R) > 0).astype(np.int8)
        return sign_bits, itq_bits

    # DB rows (enrollment = Real print) for both bridges
    db_sign, db_itq = [], []
    for iid in db_ids:
        s, q = codes(real_by[iid])
        db_sign.append(widen(s))
        db_itq.append(widen(q))

    # queries
    g_sign, g_itq = codes(prb_by[genuine_id])        # genuine = enrolled person's probe
    i_sign, i_itq = codes(prb_by[impostor_id])       # impostor = outsider's probe

    g_idx = db_ids.index(genuine_id)
    write_case(f"{OUT}/itq_genuine.txt", widen(g_itq), db_itq)
    write_case(f"{OUT}/itq_impostor.txt", widen(i_itq), db_itq)
    write_case(f"{OUT}/sign_genuine.txt", widen(g_sign), db_sign)
    write_case(f"{OUT}/sign_impostor.txt", widen(i_sign), db_sign)

    # report ground truth + raw 16-bit Hamming to the correct row (sanity)
    hd_itq = int(np.sum(g_itq != db_itq[g_idx][::REPEAT]))
    hd_sign = int(np.sum(g_sign != db_sign[g_idx][::REPEAT]))
    print(f"[export] DB size m={M}, codes widened 16->128 (x{REPEAT})")
    print(f"[export] genuine identity  : {genuine_id}  -> DB row {g_idx}")
    print(f"[export] impostor identity : {impostor_id}  (not enrolled)")
    print(f"[export] genuine 16-bit Hamming to its row:  ITQ={hd_itq}  sign={hd_sign}")
    print(f"[export] wrote 4 cases -> {OUT}/")
    # save ground truth for the runner
    with open(f"{OUT}/ground_truth.txt", "w") as f:
        f.write(f"genuine_row {g_idx}\n")


if __name__ == "__main__":
    main()
