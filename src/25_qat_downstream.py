"""
Step 25: downstream operating-point + COMMUNICATION comparison for a learned code vs the
ortho-thermometer baseline (plan round-2, steps 5+7).

The cross-layer thesis: a code with tighter impostor sigma separates the FLPSI accept
distributions, so it can hit the SAME per-record (TAR, FAR) with FEWER subsamples T -> less
communication. This script measures that end-to-end:

  1. load a code npz (gal/prb uint8, from 24_qat_e2e or the ortho-thermometer baseline),
  2. for each code find the minimum T (searching w,t) that reaches a target operating point
     under the validated accept model q(H)=C(d-H,w)/C(d,w), accept=P[Binom(T,q(H))>=t],
  3. call the REAL flash-psi `simulation --track-io --csv` at each chosen (w,t,T) to read the
     actual communication (KB) at m=5000, and report the comms delta.

Run: python3 25_qat_downstream.py --codes data/qat_codes_e2e_d64.npz --tag e2e_d64
"""
import argparse, json, os, subprocess
from math import comb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "results")); Dc = 128
SIM = os.environ.get("FLASH_SIM_BIN", os.path.abspath(
    os.path.join(HERE, "..", "crypto", "flash-psi", "target", "release", "simulation")))


def orthothermo(tr, gal, prb, K=16, Bp=8, seed=1):
    r = np.random.default_rng(seed); rows = []
    while len(rows) < K:
        q, _ = np.linalg.qr(r.standard_normal((tr.shape[1], tr.shape[1]))); rows.extend(q.T)
    A = np.array(rows[:K]); mu = tr.mean(0); pj = (tr - mu) @ A.T
    qs = np.quantile(pj, [i/(Bp+1) for i in range(1, Bp+1)], axis=0)
    f = lambda X: (((X - mu) @ A.T)[:, :, None] > qs.T[None, :, :]).reshape(len(X), K*Bp).astype(np.uint8)
    return f(gal), f(prb)


def q_clean(H, w):
    H = int(H); return 0.0 if Dc - H < w else comb(Dc - H, w) / comb(Dc, w)


def binom_tail_ge(n, k, p):
    if k <= 0: return 1.0
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    return float(sum(comb(n, j) * p**j * (1-p)**(n-j) for j in range(k, n+1)))


def dists(cg, cp):
    H = cp.astype(np.int32) @ (1-cg.astype(np.int32)).T + (1-cp.astype(np.int32)) @ cg.astype(np.int32).T
    n = len(cg); off = ~np.eye(n, dtype=bool)
    return np.diag(H).astype(int), H[off].astype(int)


def min_T(genH, impH, tar_min, far_max, w=14, t=2, Ts=range(4, 129, 2)):
    """Smallest subsample count T (at FIXED w,t) with TAR>=tar_min and per-record FAR<=far_max.
    w is held fixed across codes so the comms delta is attributable to T alone, and because the
    simulation binary's runtime explodes with w (w=14 is the paper's operating point)."""
    gc = np.bincount(genH, minlength=Dc+1); ic = np.bincount(impH, minlength=Dc+1)
    Hs = np.arange(Dc+1); qv = np.array([q_clean(H, w) for H in Hs])
    for T in Ts:
        acc = np.array([binom_tail_ge(T, t, q) for q in qv])
        tar = float((gc*acc).sum()/gc.sum()); far = float((ic*acc).sum()/ic.sum())
        if tar >= tar_min and far <= far_max:
            return {"T": T, "w": w, "t": t, "tar": tar, "far": far}
    return None


def comms_kb(w, T, t, m=5000):
    out = subprocess.run([SIM, "-m", str(m), "-w", str(w), "-s", str(T), "-t", str(t),
                          "--track-io", "--csv"], capture_output=True, text=True, timeout=600)
    for line in out.stdout.splitlines():
        if line.startswith("flpsi"):
            for tok in line.split(","):
                tok = tok.strip()
                if tok.endswith("KB"):
                    return float(tok[:-2].strip())
    raise RuntimeError(f"no comms parsed: {out.stdout[:200]} {out.stderr[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=str, default=None, help="npz with gal/prb uint8; omit for baseline only")
    ap.add_argument("--tag", type=str, default="qat")
    ap.add_argument("--tar", type=float, default=0.95)
    ap.add_argument("--far", type=float, default=1e-2)
    ap.add_argument("--w", type=int, default=14)
    ap.add_argument("--t", type=int, default=2)
    args = ap.parse_args()

    z = np.load(f"{DATA}/dim_ablation_emb_224.npz"); gal, prb, tr = z["gal"], z["prb"], z["train"].astype(np.float64)
    codes = {}
    bg, bp = orthothermo(tr, gal.astype(np.float64), prb.astype(np.float64)); codes["ortho-thermo"] = (bg, bp)
    if args.codes:
        c = np.load(args.codes if os.path.isabs(args.codes) else os.path.join(HERE, args.codes))
        codes[args.tag] = (c["gal"].astype(np.uint8), c["prb"].astype(np.uint8))

    rows = []
    print(f"[down] target TAR>={args.tar*100:.0f}% per-record FAR<={args.far:.0e}, fixed w={args.w} t={args.t}, m=5000\n")
    for name, (cg, cp) in codes.items():
        genH, impH = dists(cg, cp)
        cfg = min_T(genH, impH, args.tar, args.far, w=args.w, t=args.t)
        if cfg is None:
            print(f"  {name:16} no (w,t,T) meets target"); rows.append({"code": name, "feasible": False}); continue
        kb = comms_kb(cfg["w"], cfg["T"], cfg["t"])
        print(f"  {name:16} imp mean {impH.mean():5.1f} sd {impH.std():5.2f} | "
              f"min T={cfg['T']:>3} (w={cfg['w']},t={cfg['t']}) TAR {cfg['tar']*100:.1f}% FAR {cfg['far']:.1e} | comms {kb:.0f} KB")
        rows.append({"code": name, "imp_mean": float(impH.mean()), "imp_sd": float(impH.std()),
                     **cfg, "comms_kb": kb})
    if len(rows) == 2 and all(r.get("comms_kb") for r in rows):
        base = next(r for r in rows if r["code"] == "ortho-thermo"); new = next(r for r in rows if r["code"] != "ortho-thermo")
        print(f"\n  comms: {base['comms_kb']:.0f} KB (T={base['T']}) -> {new['comms_kb']:.0f} KB (T={new['T']})  "
              f"= {100*(1-new['comms_kb']/base['comms_kb']):.0f}% reduction")
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/qat-downstream-{args.tag}.json", "w") as f: json.dump(rows, f, indent=2)
    print(f"\n[down] wrote {OUT}/qat-downstream-{args.tag}.json")


if __name__ == "__main__":
    main()
