"""
Step 27: real-crypto validation of a learned QAT code through the ACTUAL flash-psi binary.
Reuses 15_realcrypto_thermo's fingerprint-binary driver + accept model, but passes the subsample
count T EXPLICITLY (15's helpers bind T as a default arg at import time, so it must be passed
through, not set on the module). Confirms the code's real accept/reject decisions match the
closed-form q-model at the chosen operating point.

Run: python3 27_qat_realcrypto.py --codes data/qat_codes_feat_final.npz --tag feat_final -w 6 -t 3 -s 18
"""
import argparse, os
from importlib import import_module
import numpy as np
_rc = import_module("15_realcrypto_thermo")
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True); ap.add_argument("--tag", default="qat")
    ap.add_argument("--batch", type=int, default=100); ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("-w", "--weight", type=int, default=6); ap.add_argument("-t", type=int, default=3)
    ap.add_argument("-s", "--subs", type=int, default=18); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    T = args.subs
    c = np.load(args.codes if os.path.isabs(args.codes) else os.path.join(HERE, args.codes))
    enr, qry = c["gal"].astype(np.uint8), c["prb"].astype(np.uint8)
    B = min(args.batch, len(enr)); nq = min(args.queries, B)
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(len(enr), size=B, replace=False)); db = enr[sel]
    print(f"[rc-qat] {args.tag}: B={B} nq={nq} w={args.weight} t={args.t} T={T}", flush=True)
    gen_real, gen_pred, imp_real, imp_pred, gen_H, imp_H = [], [], [], [], [], []
    for qi in range(nq):
        query = qry[sel[qi]]
        matched, dur = _rc.run_fingerprint(query, db, args.weight, args.t, ss=T)   # explicit T
        for j in range(B):
            H = int(_rc.hamming(query, db[j])); acc = int(j in matched)
            pr = _rc.accept_prob(H, args.weight, args.t, T_sub=T)                  # explicit T
            if j == qi: gen_real.append(acc); gen_pred.append(pr); gen_H.append(H)
            else: imp_real.append(acc); imp_pred.append(pr); imp_H.append(H)
        if (qi+1) % 20 == 0 or qi == nq-1:
            print(f"  {qi+1}/{nq} last {dur:.0f}ms | real TAR {np.mean(gen_real)*100:.1f}% FAR {np.mean(imp_real):.2e}", flush=True)
    gr, ir = np.array(gen_real), np.array(imp_real); gp, ip = np.array(gen_pred), np.array(imp_pred)
    print(f"\n  real TAR {gr.mean()*100:.1f}%  real FAR {ir.mean():.2e}  pred FAR {ip.mean():.2e}  "
          f"genH {np.mean(gen_H):.1f}  impH {np.mean(imp_H):.1f}", flush=True)
    print(f"  confident-REJECT {(ir[ip<0.01]==0).sum()}/{(ip<0.01).sum()}  "
          f"confident-ACCEPT {(gr[gp>0.99]==1).sum()}/{(gp>0.99).sum()}", flush=True)


if __name__ == "__main__":
    main()
