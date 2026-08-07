"""
Week 1 — Step 6: COMBINE friend's Super-Bit quantizer with the real flash-psi.

Friend's quantizer.py turns a 16-D CNN embedding into a real 128-bit Hamming
code (center + Super-Bit LSH + median balance) -- a proper 16->128 projection,
NOT the crude 8x bit-repeat used in 05_export_psi_input.py.

Here we apply it to OUR trained Blind-Touch CNN on OUR held-out SOCOFing prints
and write genuine/impostor input files for the real flash-psi `fingerprint`
binary, so real Super-Bit codes go through the real cryptographic protocol.

Outputs data/psi_sb/{genuine,impostor}.txt  (128-bit codes, m=50 DB).
"""
import os
import numpy as np
import torch

from model import FeatureModel
from quantizer import NeuralPSIQuantizer, QuantizerConfig, hamming

DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
OUT = DATA + "/psi_sb"
M = 50
rng = np.random.default_rng(2024)


def embed_all(model, x, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch].astype(np.float32) / 255.0).unsqueeze(1)
            out.append(model(xb).numpy())
    return np.concatenate(out)


def to_line(bits):
    return "".join("1" if b else "0" for b in bits)


def write_case(path, query, db):
    with open(path, "w") as f:
        f.write(f"{len(db)} 128\n")
        f.write(to_line(query) + "\n")
        for row in db:
            f.write(to_line(row) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    x_real = np.load(f"{DATA}/x_real.npy"); ids_real = np.load(f"{DATA}/ids_real.npy")
    x_probe = np.load(f"{DATA}/x_probe.npy"); ids_probe = np.load(f"{DATA}/ids_probe.npy")
    ids_test = set(np.load(f"{DATA}/ids_test.npy").tolist())

    model = FeatureModel(img=96)
    model.load_state_dict(torch.load(f"{DATA}/feature_model.pt"))

    # ---- fit the Super-Bit quantizer on TRAIN embeddings only (no leakage) ----
    train_real = np.stack([x for i, x in zip(ids_real, x_real) if i not in ids_test])
    calib = embed_all(model, train_real)            # (Ntrain, 16) float embeddings
    cfg = QuantizerConfig(center=True, whiten=False, superbit=True, balance=True)
    w1 = np.ones(16)                                 # whiten disabled -> w1 unused
    quant = NeuralPSIQuantizer.fit(calib, w1, cfg)
    quant.save(f"{OUT}/quantizer_artifacts")
    print(f"[superbit] fitted on {len(calib)} train embeddings -> 128-bit codes")

    # ---- held-out test identities ----
    real_by = {i: x for i, x in zip(ids_real, x_real) if i in ids_test}
    prb_by = {i: x for i, x in zip(ids_probe, x_probe) if i in ids_test}
    common = sorted(set(real_by) & set(prb_by)); rng.shuffle(common)
    db_ids = common[:M]
    genuine_id = db_ids[7]
    impostor_id = common[M + 3]

    def code(img):
        return quant.transform(embed_all(model, img[None])[0])

    db = np.stack([code(real_by[i]) for i in db_ids])      # enrollment codes
    g_code = code(prb_by[genuine_id])                       # genuine probe
    i_code = code(prb_by[impostor_id])                      # impostor probe
    g_idx = db_ids.index(genuine_id)

    write_case(f"{OUT}/genuine.txt", g_code, db)
    write_case(f"{OUT}/impostor.txt", i_code, db)
    with open(f"{OUT}/ground_truth.txt", "w") as f:
        f.write(f"genuine_row {g_idx}\n")

    # diagnostics: real 128-bit Hamming distributions
    gh = int(hamming(g_code, db[g_idx]))
    imp_to_db = hamming(np.broadcast_to(i_code, db.shape), db)
    gen_to_others = hamming(np.broadcast_to(g_code, db.shape), db)
    print(f"[superbit] genuine identity {genuine_id} -> DB row {g_idx}")
    print(f"[superbit] genuine 128-bit Hamming to its own row : {gh}")
    print(f"[superbit] genuine Hamming to OTHER rows  min/mean : "
          f"{np.delete(gen_to_others, g_idx).min()}/{np.delete(gen_to_others, g_idx).mean():.1f}")
    print(f"[superbit] impostor Hamming to all rows   min/mean : "
          f"{imp_to_db.min()}/{imp_to_db.mean():.1f}")
    print(f"[superbit] wrote genuine.txt + impostor.txt -> {OUT}/")


if __name__ == "__main__":
    main()
