"""
Step 26: multi-finger fusion (secure+usable 1:N point) for a learned code vs ortho-thermometer.
Reuses the validated Poisson-binomial accept model from 17_bridge_fusion.py; only swaps in the
QAT code (order-matched to the cached embeddings / sb_codes fingers).

Run: python3 26_qat_fusion.py --codes data/qat_codes_lin.npz --tag lin
"""
import argparse, json, os
from importlib import import_module
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results"))
_f = import_module("17_bridge_fusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=str, required=True)
    ap.add_argument("--tag", type=str, default="qat")
    args = ap.parse_args()
    z = np.load(f"{DATA}/dim_ablation_emb_224.npz", allow_pickle=True)
    gal, prb, tr = z["gal"].astype(np.float64), z["prb"].astype(np.float64), z["train"].astype(np.float64)
    fingers = np.load(f"{DATA}/sb_codes_224.npz", allow_pickle=True)["fingers"]
    c = np.load(args.codes if os.path.isabs(args.codes) else os.path.join(HERE, args.codes))
    print("Goal: min-FRR config reaching query-FAR@5000 <= 1e-2 (secure AND usable)\n")
    out = {}
    out["ortho-thermo"] = _f.fusion_frontier("ortho-thermo", *_f.thermo(tr, gal, prb), fingers)
    out[args.tag] = _f.fusion_frontier(args.tag, c["gal"].astype(np.uint8), c["prb"].astype(np.uint8), fingers)
    with open(f"{OUT}/qat-fusion-{args.tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[fusion] wrote {OUT}/qat-fusion-{args.tag}.json")
    for k in (2, 3):
        a = out["ortho-thermo"]["best"][k]; b = out[args.tag]["best"][k]
        if a and b:
            print(f"  {k}-of-3 secure point FRR: ortho-thermo {a['frr']*100:.1f}%  vs  {args.tag} {b['frr']*100:.1f}%")


if __name__ == "__main__":
    main()
